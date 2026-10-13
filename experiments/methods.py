"""Construct the score and calibrator combinations used in the paper."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from experiments.campaign import experiment_device
from socp.conformal.bandwidth import ess_bandwidth_record, select_ess_bandwidth
from socp.conformal.guan_tuning import (
    tune_guan_lcp_classification_scores,
    tune_guan_lcp_regression_scores,
)
from socp.conformal.localizer import Localizer
from socp.conformal.localizers import (
    GaussianKernelLocalizer,
    IdentityLocalizer,
    KMeansLocalizer,
    LaplaceKernelLocalizer,
    SOMLocalizer,
)
from socp.conformal.mechanism import Mechanism
from socp.conformal.mechanisms import (
    RLCP,
    ClasswiseCalibration,
    ClusteredCalibration,
    GuanLCP,
    MondrianCalibration,
    PooledCalibration,
    WeightedCalibration,
)
from socp.conformal.mechanisms.clustered import (
    authors_cluster_plan,
    authors_random_split,
)
from socp.conformal.routing import Regime3RoutingConfig
from socp.conformal.rules import ConformalRule
from socp.conformal.score import (
    Score,
    Task,
    bind_score_observation_ids,
    evaluate_score,
    fit_score,
    invert_score,
)
from socp.conformal.scores import (
    AbsoluteResidualScore,
    APSScore,
    CHRScore,
    CQRScore,
    LabelRankingScore,
    LACScore,
    NormalizingFlowScore,
    RAPSScore,
    RectifiedScore,
    SAPSScore,
)
from socp.conformal.scores.selection import select_raps_parameters, select_saps_weight
from socp.models.flow import FLOW_GAMMA, FLOW_RESIDUAL_FLOOR
from socp.randomness import (
    derive_seed,
    observation_tie_breakers,
    observation_uniforms,
)

Array = NDArray[np.float64]
ScoreBuilder = Callable[[], Score]


class _ProbabilityModel(Protocol):
    def predict_proba(self, X: Array) -> Array: ...


@runtime_checkable
class _ClassificationMatrixScore(Protocol):
    def score_matrix(self, X: Array) -> Array: ...


@dataclass(frozen=True, slots=True)
class ScoreChoice:
    """A reusable score constructor plus selection-only metadata."""

    score_id: str
    build: ScoreBuilder
    selection: dict[str, object]


@dataclass(frozen=True, slots=True)
class MethodChoice:
    """One score, mechanism, and localizer composition, ready to fit."""

    score_id: str
    calibrator_id: str
    rule: ConformalRule
    selection: dict[str, object]
    auxiliary_indices: NDArray[np.intp] | None = None
    proper_indices: NDArray[np.intp] | None = None


@dataclass(frozen=True, slots=True)
class SharedGeometry:
    """Pre-calibration geometry reused across score-matched comparisons."""

    rlcp: GaussianKernelLocalizer | None
    mondrian: KMeansLocalizer | None
    som: SOMLocalizer | None
    metadata: dict[str, object]


def select_mondrian_groups(
    X_train: Array,
    X_selection: Array,
    *,
    candidates: tuple[int, ...],
    seed: int,
) -> tuple[int, tuple[tuple[int, float], ...]]:
    """Select the Mondrian group count by silhouette score on the selection inputs.

    Standardization and every candidate k-means model are fitted on training inputs.
    Exact silhouette ties keep the smaller count.
    """
    train = np.asarray(X_train, dtype=np.float64)
    selection = np.asarray(X_selection, dtype=np.float64)
    if (
        train.ndim != 2
        or selection.ndim != 2
        or min(train.shape) < 1
        or min(selection.shape) < 1
        or train.shape[1] != selection.shape[1]
    ):
        raise ValueError("train and selection must be non-empty compatible matrices")
    if not np.isfinite(train).all() or not np.isfinite(selection).all():
        raise ValueError("train and selection features must be finite")
    if not candidates or any(groups < 2 for groups in candidates):
        raise ValueError("Mondrian group candidates must be integers of at least 2")

    ordered = tuple(sorted(set(candidates)))
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    selection_scaled = scaler.transform(selection)
    silhouettes: list[tuple[int, float]] = []
    for groups in ordered:
        if groups > len(train):
            continue
        model = KMeans(n_clusters=groups, n_init=10, random_state=seed)
        assignments = model.fit(train_scaled).predict(selection_scaled)
        assigned_groups = np.unique(assignments).size
        if assigned_groups < 2 or assigned_groups >= len(selection):
            continue
        try:
            value = float(silhouette_score(selection_scaled, assignments))
        except ValueError:
            continue
        if np.isfinite(value):
            silhouettes.append((groups, value))

    if silhouettes:
        selected, _ = max(silhouettes, key=lambda item: (item[1], -item[0]))
        return selected, tuple(silhouettes)
    if 5 in ordered and len(train) >= 5:
        return 5, ()
    raise ValueError("no Mondrian group candidate is feasible for this split")


def _observation_draws(
    dataset: str,
    seed: int,
    score_id: str,
) -> Callable[[NDArray[np.int64], str, int | None], Array]:
    def draw(
        row_ids: NDArray[np.int64],
        purpose: str,
        width: int | None,
    ) -> Array:
        if width is not None:
            return observation_tie_breakers(
                dataset,
                seed,
                score_id,
                row_ids,
                n_classes=width,
            )
        return observation_uniforms(
            dataset,
            seed,
            score_id,
            row_ids,
            purpose=purpose,
        )

    return draw


def _classification_selection(
    score_id: str,
    probability_fn: Callable[[Array], Array],
    X_selection: Array,
    y_selection: Array,
    *,
    seed: int,
    alpha: float,
) -> tuple[dict[str, object], dict[str, object]]:
    if score_id not in {"raps", "saps"}:
        return {}, {}
    probabilities = np.asarray(probability_fn(X_selection), dtype=np.float64)
    parameters: dict[str, object]
    if score_id == "raps":
        selected = select_raps_parameters(
            probabilities,
            y_selection,
            alpha=alpha,
            seed=seed,
        )
        parameters = {
            "penalty": selected.selected_value,
            "k_reg": cast(int, selected.k_reg),
        }
    else:
        selected = select_saps_weight(
            probabilities,
            y_selection,
            alpha=alpha,
            seed=seed,
        )
        parameters = {"weight": selected.selected_value}
    metadata: dict[str, object] = {
        selected.parameter_name: selected.selected_value,
        "k_reg": selected.k_reg,
        "selection_average_sizes": selected.average_set_sizes,
        "selection_cutoffs": selected.cutoffs,
        "selection_grid": selected.grid,
        "selection_source_revision": selected.source_revision,
    }
    return parameters, metadata


def score_choice(
    score_id: str,
    models: dict[str, Any],
    X_selection: Array,
    y_selection: Array,
    selection_row_ids: NDArray[np.int64],
    *,
    dataset: str,
    seed: int,
    alpha: float,
) -> ScoreChoice:
    """Return the constructor and metadata of one declared score."""
    if score_id == "abs_residual":
        return ScoreChoice(
            score_id,
            lambda: AbsoluteResidualScore(models["point"].predict),
            {},
        )
    if score_id == "cqr":

        def quantiles(X: Array) -> Array:
            return np.column_stack(
                (models["lower"].predict(X), models["upper"].predict(X))
            )

        return ScoreChoice(score_id, lambda: CQRScore(quantiles), {})
    if score_id == "chr":
        draws = _observation_draws(dataset, seed, score_id)
        histogram = models["histogram"]
        return ScoreChoice(
            score_id,
            lambda: CHRScore(
                histogram.bin_probabilities,
                histogram.bin_edges,
                randomized=True,
                observation_draws=draws,
            ),
            {},
        )
    if score_id == "nf_cp":
        response_scale = float(models["flow_response_scale"])
        return ScoreChoice(
            score_id,
            lambda: NormalizingFlowScore(
                models["point"].predict,
                models["flow_scale"],
                gamma=FLOW_GAMMA * response_scale,
                residual_floor=FLOW_RESIDUAL_FLOOR * response_scale,
            ),
            {},
        )
    if score_id == "rcp":
        return ScoreChoice(
            score_id,
            lambda: RectifiedScore(
                AbsoluteResidualScore(models["point"].predict),
                models["rcp_scale"],
            ),
            {},
        )

    predictor = cast(_ProbabilityModel, models["classifier"])
    probability_fn = predictor.predict_proba
    parameters, metadata = _classification_selection(
        score_id,
        probability_fn,
        X_selection,
        y_selection,
        seed=seed,
        alpha=alpha,
    )
    draws = _observation_draws(dataset, seed, score_id)
    if score_id == "lac":

        def builder() -> Score:
            return LACScore(probability_fn)
    elif score_id == "aps":

        def builder() -> Score:
            return APSScore(probability_fn, observation_draws=draws)
    elif score_id == "raps":

        def builder() -> Score:
            return RAPSScore(
                probability_fn,
                penalty=cast(float, parameters["penalty"]),
                k_reg=cast(int, parameters["k_reg"]),
                observation_draws=draws,
            )
    elif score_id == "saps":

        def builder() -> Score:
            return SAPSScore(
                probability_fn,
                weight=cast(float, parameters["weight"]),
                observation_draws=draws,
            )
    elif score_id == "rank":

        def builder() -> Score:
            return LabelRankingScore(probability_fn, observation_draws=draws)
    else:
        raise ValueError(f"unknown score {score_id!r}")
    return ScoreChoice(score_id, builder, metadata)


def fit_som_localizer(
    X_train: Array,
    *,
    seed: int,
    dataset_config: dict[str, Any],
    campaign_config: dict[str, Any],
    grid: tuple[int, int] | None = None,
    radius: int | None = None,
    routing: Regime3RoutingConfig | None = None,
) -> SOMLocalizer:
    """Fit one declared SOM, optionally with training-derived Regime 3 routing."""
    from torchsom import SOM

    settings = dataset_config["som"]
    rows, columns = grid or tuple(int(value) for value in settings["grid"])
    defaults = campaign_config["som"]
    # TorchSOM draws seeded random prototypes here; fit() starts from them.
    som = SOM(
        rows,
        columns,
        X_train.shape[1],
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        sigma=float(settings["sigma"]),
        learning_rate=float(settings["learning_rate"]),
        neighborhood_order=int(settings["radius"] if radius is None else radius),
        topology=str(defaults["topology"]),
        lr_decay_function=str(defaults["lr_decay_function"]),
        sigma_decay_function=str(defaults["sigma_decay_function"]),
        neighborhood_function="gaussian",
        distance_function="euclidean",
        pbc=False,
        search_backend=str(defaults["search_backend"]),
        device=experiment_device(),
        random_seed=seed,
    )
    localizer = (
        SOMLocalizer(som, n_extra_cells=0)
        if routing is None
        else SOMLocalizer(som, routing=routing)
    )
    localizer.fit(X_train)
    return localizer


def shared_geometry(
    X_train: Array,
    X_selection: Array,
    *,
    seed: int,
    n_calibration: int,
    dataset_config: dict[str, Any],
    campaign_config: dict[str, Any],
    required_calibrators: set[str],
) -> SharedGeometry:
    """Fit the feature-only geometry of RLCP, Mondrian, and SOCP before calibration."""
    required = required_calibrators

    rlcp: GaussianKernelLocalizer | None = None
    mondrian: KMeansLocalizer | None = None
    som_localizer: SOMLocalizer | None = None
    metadata: dict[str, object] = {}

    if "rlcp" in required:
        target = float(campaign_config["selection"]["localized_bandwidth_target_ess"])
        standardized = StandardScaler().fit_transform(X_train)
        bandwidth = select_ess_bandwidth(
            standardized,
            n_calibration=n_calibration,
            target=target,
            random_seed=derive_seed(seed, "bandwidth"),
        )
        rlcp = GaussianKernelLocalizer(bandwidth.bandwidth)
        rlcp.fit(X_train)
        metadata["rlcp"] = ess_bandwidth_record(bandwidth)

    if "mondrian_kmeans" in required:
        candidate_groups = tuple(
            int(value)
            for value in campaign_config["selection"]["mondrian_group_candidates"]
        )
        kmeans_seed = derive_seed(seed, "kmeans")
        groups, silhouettes = select_mondrian_groups(
            X_train,
            X_selection,
            candidates=candidate_groups,
            seed=kmeans_seed,
        )
        mondrian = KMeansLocalizer(
            n_clusters=groups,
            random_state=kmeans_seed,
        )
        mondrian.fit(X_train)
        metadata["mondrian_kmeans"] = {
            "selected_groups": groups,
            "candidate_groups": candidate_groups,
            "validation_silhouettes": {
                str(candidate): value for candidate, value in silhouettes
            },
            "fit_split": "train",
            "selection_split": "selection",
            "criterion": "silhouette",
            "tie_break": "smaller_k",
        }

    if "socp" in required:
        som_localizer = fit_som_localizer(
            X_train,
            seed=seed,
            dataset_config=dataset_config,
            campaign_config=campaign_config,
        )
    return SharedGeometry(
        rlcp=rlcp,
        mondrian=mondrian,
        som=som_localizer,
        metadata=metadata,
    )


def _score_matrix(score: Score, X: Array) -> Array:
    if not isinstance(score, _ClassificationMatrixScore):
        raise TypeError(f"score {score.spec.name!r} lacks classification score_matrix")
    return np.asarray(score.score_matrix(X), dtype=np.float64)


def _guan_bandwidth(
    choice: ScoreChoice,
    X_train: Array,
    y_train: Array,
    X_selection: Array,
    y_selection: Array,
    selection_row_ids: NDArray[np.int64],
    *,
    task: Task,
    alpha: float,
    seed: int,
    n_calibration: int,
) -> tuple[float, dict[str, object]]:
    X_tune = X_selection
    y_tune = y_selection
    tune_ids = selection_row_ids
    score = choice.build()
    fit_score(score, X_train, y_train)
    if score.spec.randomized:
        bind_score_observation_ids(score, tune_ids)
    true_scores = evaluate_score(score, X_tune, y_tune)

    if task == "classification":
        if score.spec.randomized:
            bind_score_observation_ids(score, tune_ids)
        matrix = _score_matrix(score, X_tune)
        result = tune_guan_lcp_classification_scores(
            X_train,
            X_tune,
            matrix,
            y_tune,
            n_calibration=n_calibration,
            alpha=alpha,
            seed=seed,
        )
    else:
        if isinstance(score, CHRScore):
            bind_score_observation_ids(score, tune_ids)
            prediction_measure = score.measure_lookup(X_tune)
        else:

            def prediction_measure(
                queries: NDArray[np.int64],
                thresholds: Array,
            ) -> Array:
                measures = []
                for query, threshold in zip(queries, thresholds, strict=True):
                    if score.spec.randomized:
                        bind_score_observation_ids(score, tune_ids[[int(query)]])
                    prediction = invert_score(
                        score,
                        X_tune[int(query)],
                        float(threshold),
                    )
                    measures.append(prediction.measure())
                return np.asarray(measures, dtype=np.float64)

        result = tune_guan_lcp_regression_scores(
            X_train,
            X_tune,
            true_scores,
            prediction_measure,
            n_calibration=n_calibration,
            alpha=alpha,
            seed=seed,
        )
    bandwidth = math.inf if result.bandwidth is None else result.bandwidth
    metadata = result.provenance()
    metadata["lcp_tuning_rows"] = len(X_tune)
    metadata["lcp_uniform_fallback"] = result.bandwidth is None
    return bandwidth, metadata


def method_choice(
    choice: ScoreChoice,
    calibrator_id: str,
    geometry: SharedGeometry | None,
    X_train: Array,
    y_train: Array,
    X_selection: Array,
    y_selection: Array,
    selection_row_ids: NDArray[np.int64],
    y_calibration: Array,
    *,
    task: Task,
    alpha: float,
    seed: int,
    n_classes: int | None,
    campaign_config: dict[str, Any],
) -> MethodChoice:
    """Build one final score-matched conformal method."""
    selection = dict(choice.selection)
    auxiliary: NDArray[np.intp] | None = None
    proper: NDArray[np.intp] | None = None
    localizer: Localizer
    mechanism: Mechanism
    if calibrator_id == "pooled":
        localizer = IdentityLocalizer()
        mechanism = PooledCalibration(alpha=alpha, task=task)
    elif calibrator_id == "classwise":
        if n_classes is None:
            raise ValueError("classwise calibration requires n_classes")
        localizer = IdentityLocalizer()
        mechanism = ClasswiseCalibration(alpha=alpha, n_classes=n_classes)
    elif calibrator_id == "clustered":
        if n_classes is None:
            raise ValueError("clustered calibration requires n_classes")
        counts = np.bincount(
            np.asarray(y_calibration, dtype=np.int64),
            minlength=n_classes,
        )
        plan = authors_cluster_plan(counts, alpha=alpha)
        fallback = plan.n_clusters < 1 or plan.auxiliary_fraction == 0.0
        if fallback:
            order = np.random.default_rng(seed).permutation(len(y_calibration))
            middle = max(1, len(order) // 2)
            auxiliary = np.asarray(order[:middle], dtype=np.intp)
            proper = np.asarray(order[middle:], dtype=np.intp)
        else:
            auxiliary, proper = authors_random_split(
                len(y_calibration),
                fraction=plan.auxiliary_fraction,
                seed=seed,
            )
        if auxiliary.size == 0 or proper.size == 0:
            raise RuntimeError(
                "clustered CP's released split produced an empty block; "
                "this composition is not estimable for the current sample"
            )
        localizer = IdentityLocalizer()
        mechanism = ClusteredCalibration(
            alpha=alpha,
            n_classes=n_classes,
            n_clusters=plan.n_clusters,
            random_state=seed,
        )
        selection["cluster_plan"] = {
            "auxiliary_fraction": plan.auxiliary_fraction,
            "n_clusters": plan.n_clusters,
            "auxiliary_rows": int(auxiliary.size),
            "proper_rows": int(proper.size),
            "pooled_small_sample_fallback": fallback,
        }
    elif calibrator_id == "guan_lcp":
        bandwidth, tuning = _guan_bandwidth(
            choice,
            X_train,
            y_train,
            X_selection,
            y_selection,
            selection_row_ids,
            task=task,
            alpha=alpha,
            seed=seed,
            n_calibration=len(y_calibration),
        )
        localizer = LaplaceKernelLocalizer(bandwidth)
        mechanism = GuanLCP(localizer, alpha=alpha, task=task)
        selection["guan_lcp"] = tuning
    elif calibrator_id == "rlcp":
        if geometry is None or geometry.rlcp is None:
            raise RuntimeError("RLCP requires shared training geometry")
        localizer = geometry.rlcp
        mechanism = RLCP(
            localizer,
            alpha=alpha,
            task=task,
            rng=np.random.default_rng(derive_seed(seed, "rlcp")),
        )
        selection["rlcp"] = geometry.metadata["rlcp"]
    elif calibrator_id == "mondrian_kmeans":
        if geometry is None or geometry.mondrian is None:
            raise RuntimeError("Mondrian requires shared training geometry")
        localizer = geometry.mondrian
        mechanism = MondrianCalibration(localizer, alpha=alpha, task=task)
        selection["mondrian_kmeans"] = geometry.metadata["mondrian_kmeans"]
    elif calibrator_id == "socp":
        if geometry is None or geometry.som is None:
            raise RuntimeError("SOCP requires shared training geometry")
        localizer = geometry.som
        mechanism = WeightedCalibration(localizer, alpha=alpha, task=task)
    else:
        raise ValueError(f"unknown calibrator {calibrator_id!r}")

    return MethodChoice(
        score_id=choice.score_id,
        calibrator_id=calibrator_id,
        rule=ConformalRule(choice.build(), mechanism, localizer),
        selection=selection,
        auxiliary_indices=auxiliary,
        proper_indices=proper,
    )


__all__ = (
    "MethodChoice",
    "ScoreChoice",
    "SharedGeometry",
    "fit_som_localizer",
    "method_choice",
    "score_choice",
    "shared_geometry",
)

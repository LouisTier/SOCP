"""Calibrate every paper method and write its test predictions and metrics."""

from __future__ import annotations

import csv
import json
import math
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from experiments.analysis_contract import paper_score
from experiments.campaign import (
    DEFAULT_CONFIG,
    Campaign,
    load_campaign,
    regime3_calibration_sizes,
)
from experiments.methods import (
    ScoreChoice,
    SharedGeometry,
    fit_som_localizer,
    method_choice,
    score_choice,
    shared_geometry,
)
from experiments.prepare import run_directory
from socp.conformal.localizer import Localizer
from socp.conformal.localizers import IdentityLocalizer
from socp.conformal.mechanism import Mechanism
from socp.conformal.mechanisms import (
    PooledCalibration,
    WeightedCalibration,
)
from socp.conformal.routing import Regime3RoutingConfig
from socp.conformal.rules import ConformalRule
from socp.conformal.score import Cutoff, Task
from socp.conformal.sets import PredictionSet
from socp.evaluation.conditional import (
    excess_risk_of_target,
    size_stratified_coverage,
    worst_slab_coverage,
)
from socp.evaluation.metrics import (
    ScalarMetric,
    coverage_gaps,
    prediction_metrics,
)
from socp.evaluation.partitions import fit_frozen_kmeans_partition
from socp.randomness import derive_seed

TORCHSOM_REVISION = "020ec6890aafa10def406fea1ddcdb0e266a601e"


def _cutoff_payload(cutoff: Cutoff) -> str:
    """Record scalar cutoffs; label-indexed ones (Classwise, Clustered) are not kept."""
    if isinstance(cutoff, np.ndarray):
        return ""
    return json.dumps(float(cutoff), allow_nan=True)


def load_prepared(
    source: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Load one prepared split: metadata, fitted models, and arrays."""
    with (source / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (source / "models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    with np.load(source / "data.npz") as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return metadata, models, arrays


def load_predictions(
    path: Path,
    row_ids: NDArray[np.int64],
    *,
    cutoffs: bool = False,
) -> dict[str, np.ndarray]:
    """Read one method's predictions.csv in the order of ``row_ids``.

    Set ``cutoffs`` for calibrators with scalar cutoffs to also read them.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    position = {int(row["row_id"]): index for index, row in enumerate(rows)}
    if len(position) != len(rows) or position.keys() != set(row_ids.tolist()):
        raise ValueError(f"{path} does not match the prepared test rows")
    ordered = [rows[position[int(row_id)]] for row_id in row_ids]
    predictions = {
        "covered": np.asarray([int(row["covered"]) for row in ordered], dtype=np.bool_),
        "size": np.asarray([float(row["size"]) for row in ordered], dtype=np.float64),
    }
    if cutoffs:
        predictions["cutoff"] = np.asarray(
            [float(row["cutoff"]) for row in ordered], dtype=np.float64
        )
    return predictions


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _outcomes(
    predictions: tuple[PredictionSet, ...],
    cutoffs: tuple[Cutoff, ...],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_]]:
    """Return the coverage, size, and infinite-cutoff flag of every prediction."""
    covered = np.asarray(
        [
            prediction.contains(float(target))
            for prediction, target in zip(predictions, targets, strict=True)
        ],
        dtype=np.bool_,
    )
    sizes = np.asarray(
        [prediction.measure() for prediction in predictions],
        dtype=np.float64,
    )
    infinite_cutoff = np.asarray(
        [
            bool(np.any(np.isinf(cutoff)))
            if isinstance(cutoff, np.ndarray)
            else math.isinf(float(cutoff))
            for cutoff in cutoffs
        ],
        dtype=np.bool_,
    )
    return covered, sizes, infinite_cutoff


def _write_run(
    destination: Path,
    *,
    metadata: dict[str, Any],
    row_ids: NDArray[np.int64],
    targets: NDArray[np.float64],
    features: NDArray[np.float64],
    predictions: tuple[PredictionSet, ...],
    cutoffs: tuple[Cutoff, ...],
    external_assignments: dict[int, NDArray[np.int64]],
    alpha: float,
    conditional_metrics: bool,
) -> None:
    """Write one method's outputs. The first external partition is the primary one."""
    destination.mkdir(parents=True, exist_ok=True)
    covered, sizes, infinite_cutoff = _outcomes(predictions, cutoffs, targets)
    empty = np.asarray(
        [prediction.is_empty() for prediction in predictions],
        dtype=np.bool_,
    )
    metrics = prediction_metrics(
        covered,
        sizes,
        empty=empty,
        infinite_cutoff=infinite_cutoff,
    )
    primary, *others = external_assignments
    external_gaps = {
        groups: coverage_gaps(covered, assignments, alpha=alpha).wcovgap
        for groups, assignments in external_assignments.items()
    }

    wsc: ScalarMetric | None
    ssc: ScalarMetric | None
    ert: ScalarMetric | None
    if conditional_metrics:
        evaluation = metadata["evaluation"]
        wsc = worst_slab_coverage(
            features,
            covered,
            n_directions=int(evaluation["wsc"]["directions"]),
            n_splits=int(evaluation["wsc"]["splits"]),
            delta=float(evaluation["wsc"]["delta"]),
            seed=derive_seed(int(metadata["seed"]), "wsc"),
        )
        ssc = size_stratified_coverage(
            sizes,
            covered,
            alpha=alpha,
            seed=derive_seed(int(metadata["seed"]), "ssc"),
            n_groups=int(evaluation["ssc"]["groups"]),
        )
        ert = excess_risk_of_target(
            features,
            covered,
            alpha=alpha,
            seed=derive_seed(int(metadata["seed"]), "ert"),
            n_splits=int(evaluation["ert"]["folds"]),
        )
    else:
        wsc = None
        ssc = None
        ert = None

    with (destination / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_id", "covered", "size", "cutoff"),
        )
        writer.writeheader()
        for row_id, is_covered, size, cutoff in zip(
            row_ids, covered, sizes, cutoffs, strict=True
        ):
            writer.writerow(
                {
                    "row_id": int(row_id),
                    "covered": int(is_covered),
                    "size": float(size),
                    "cutoff": _cutoff_payload(cutoff),
                }
            )

    metric_row = {
        "dataset": metadata["dataset"],
        "seed": metadata["seed"],
        "task": metadata["task"],
        "score": metadata["score"],
        "calibrator": metadata["calibrator"],
        "support": metrics.support,
        "coverage": _optional_float(metrics.coverage),
        "size_mean": _optional_float(metrics.size_mean),
        "empty_rate": _optional_float(metrics.empty_rate),
        "infinite_cutoff_rate": _optional_float(metrics.infinite_cutoff_rate),
        "external_wcovgap": _optional_float(external_gaps[primary]),
        **{
            f"external_wcovgap_k{groups}": _optional_float(external_gaps[groups])
            for groups in others
        },
        "wsc": _optional_float(wsc.value) if wsc else None,
        "ssc": _optional_float(ssc.value) if ssc else None,
        "ert": _optional_float(ert.value) if ert else None,
    }
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(metric_row))
        writer.writeheader()
        writer.writerow(metric_row)

    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _method_grid(campaign: Campaign, task: Task) -> tuple[tuple[str, str], ...]:
    if campaign.profile == "smoke":
        return ((paper_score(task), "pooled"), (paper_score(task), "socp"))
    methods = campaign.config["methods"][task]
    return tuple(
        (str(score), str(calibrator))
        for score in methods["scores"]
        for calibrator in methods["calibrators"]
    )


def _analysis_metrics(
    predictions: tuple[PredictionSet, ...],
    cutoffs: tuple[Cutoff, ...],
    targets: NDArray[np.float64],
    external_assignments: NDArray[np.int64],
    *,
    alpha: float,
) -> dict[str, float | int]:
    covered, sizes, infinite = _outcomes(predictions, cutoffs, targets)
    gaps = coverage_gaps(covered, external_assignments, alpha=alpha)
    return {
        "support": len(covered),
        "coverage": float(np.mean(covered)),
        "size_mean": float(np.mean(sizes)),
        "external_wcovgap": float(gaps.wcovgap)
        if gaps.wcovgap is not None
        else math.nan,
        "infinite_cutoff_rate": float(np.mean(infinite)),
    }


def _run_analysis_rule(
    choice: ScoreChoice,
    localizer: Localizer,
    mechanism: Mechanism,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_calibration: NDArray[np.float64],
    y_calibration: NDArray[np.float64],
    calibration_ids: NDArray[np.int64],
    X_test: NDArray[np.float64],
    test_ids: NDArray[np.int64],
) -> tuple[tuple[PredictionSet, ...], tuple[Cutoff, ...]]:
    rule = ConformalRule(choice.build(), mechanism, localizer)
    rule.fit(X_train, y_train)
    rule.calibrate(X_calibration, y_calibration, row_ids=calibration_ids)
    return rule.predict_with_cutoffs(X_test, row_ids=test_ids)


def _write_analysis(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_sensitivity(
    campaign: Campaign,
    dataset: str,
    seed: int,
    choice: ScoreChoice,
    dataset_config: dict[str, Any],
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_calibration: NDArray[np.float64],
    y_calibration: NDArray[np.float64],
    calibration_ids: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    test_ids: NDArray[np.int64],
    external_assignments: NDArray[np.int64],
    output_dir: Path,
) -> None:
    settings = campaign.config["sensitivity"]
    grids = settings["datasets"].get(dataset)
    if grids is None:
        return
    rows: list[dict[str, object]] = []
    for raw_grid in grids:
        grid = (int(raw_grid[0]), int(raw_grid[1]))
        for raw_radius in settings["radii"]:
            radius = int(raw_radius)
            localizer = fit_som_localizer(
                X_train,
                seed=seed,
                dataset_config=dataset_config,
                campaign_config=campaign.config,
                grid=grid,
                radius=radius,
            )
            predictions, cutoffs = _run_analysis_rule(
                choice,
                localizer,
                WeightedCalibration(localizer, alpha=campaign.alpha, task="regression"),
                X_train,
                y_train,
                X_calibration,
                y_calibration,
                calibration_ids,
                X_test,
                test_ids,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "score": choice.score_id,
                    "grid_rows": grid[0],
                    "grid_columns": grid[1],
                    "radius": radius,
                    **_analysis_metrics(
                        predictions,
                        cutoffs,
                        y_test,
                        external_assignments,
                        alpha=campaign.alpha,
                    ),
                }
            )
    _write_analysis(output_dir / "sensitivity.csv", rows)


def _run_regime3(
    campaign: Campaign,
    dataset: str,
    seed: int,
    choice: ScoreChoice,
    dataset_config: dict[str, Any],
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_calibration: NDArray[np.float64],
    y_calibration: NDArray[np.float64],
    calibration_ids: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    test_ids: NDArray[np.int64],
    external_assignments: NDArray[np.int64],
    output_dir: Path,
) -> None:
    settings = campaign.config["regime3"]
    sample_sizes = regime3_calibration_sizes(
        settings,
        dataset=dataset,
        full_rows=len(X_calibration),
    )
    if not sample_sizes:
        return
    order = np.random.default_rng(derive_seed(seed, "regime3_calibration")).permutation(
        len(X_calibration)
    )
    rows: list[dict[str, object]] = []
    for n_calibration in sample_sizes:
        selected = np.asarray(order[:n_calibration], dtype=np.intp)
        X_cal = X_calibration[selected]
        y_cal = y_calibration[selected]
        ids = calibration_ids[selected]
        for mode in ("pooled", "regime2", "regime3"):
            localizer: Localizer
            mechanism: Mechanism
            if mode == "pooled":
                localizer = IdentityLocalizer()
                mechanism = PooledCalibration(alpha=campaign.alpha, task="regression")
            else:
                localizer = fit_som_localizer(
                    X_train,
                    seed=seed,
                    dataset_config=dataset_config,
                    campaign_config=campaign.config,
                    routing=(
                        Regime3RoutingConfig(
                            calibration_rows=n_calibration,
                            projected_buffer_floor=float(
                                settings["projected_buffer_floor"]
                            ),
                        )
                        if mode == "regime3"
                        else None
                    ),
                )
                mechanism = WeightedCalibration(
                    localizer, alpha=campaign.alpha, task="regression"
                )
            predictions, cutoffs = _run_analysis_rule(
                choice,
                localizer,
                mechanism,
                X_train,
                y_train,
                X_cal,
                y_cal,
                ids,
                X_test,
                test_ids,
            )
            routing = getattr(localizer, "routing_plan", None)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "score": choice.score_id,
                    "calibration_rows": n_calibration,
                    "mode": mode,
                    "extra_cells": (
                        getattr(localizer, "n_extra_cells", None)
                        if mode != "pooled"
                        else None
                    ),
                    "target_reached": (
                        int(routing.target_reached) if routing is not None else None
                    ),
                    **_analysis_metrics(
                        predictions,
                        cutoffs,
                        y_test,
                        external_assignments,
                        alpha=campaign.alpha,
                    ),
                }
            )
    _write_analysis(output_dir / "regime3.csv", rows)


def _run_one(
    campaign: Campaign,
    dataset: str,
    seed: int,
    prepared_dir: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    source = run_directory(prepared_dir, dataset, seed)
    prepared_metadata, models, arrays = load_prepared(source)
    task = cast(Task, str(prepared_metadata["task"]))
    X_train = np.asarray(arrays["X_train"], dtype=np.float64)
    y_train = np.asarray(arrays["y_train"], dtype=np.float64)
    X_selection = np.asarray(arrays["X_selection"], dtype=np.float64)
    y_selection = np.asarray(arrays["y_selection"], dtype=np.float64)
    X_calibration = np.asarray(arrays["X_calibration"], dtype=np.float64)
    y_calibration = np.asarray(arrays["y_calibration"], dtype=np.float64)
    X_test = np.asarray(arrays["X_test"], dtype=np.float64)
    y_test = np.asarray(arrays["y_test"], dtype=np.float64)
    selection_ids = np.asarray(arrays["row_id_selection"], dtype=np.int64)
    calibration_ids = np.asarray(arrays["row_id_calibration"], dtype=np.int64)
    test_ids = np.asarray(arrays["row_id_test"], dtype=np.int64)
    dataset_config = campaign.config["datasets"][dataset]
    grid = _method_grid(campaign, task)

    geometry_calibrators = {
        calibrator
        for _, calibrator in grid
        if calibrator in {"rlcp", "mondrian_kmeans", "socp"}
    }
    geometry: SharedGeometry | None = None
    if geometry_calibrators:
        geometry = shared_geometry(
            X_train,
            X_selection,
            seed=seed,
            n_calibration=len(X_calibration),
            dataset_config=dataset_config,
            campaign_config=campaign.config,
            required_calibrators=geometry_calibrators,
        )
    evaluation = campaign.config["evaluation"]
    group_counts = (
        int(evaluation["external_partition"]["groups"]),
        *(int(value) for value in evaluation["external_partition_sensitivity"]),
    )
    external_assignments = {
        groups: fit_frozen_kmeans_partition(
            X_train,
            n_groups=groups,
            seed=derive_seed(seed, f"external_partition_k{groups}"),
        ).assign(X_test)
        for groups in dict.fromkeys(group_counts)
    }
    n_classes = int(dataset_config["classes"]) if task == "classification" else None
    outputs = []
    choices: dict[str, Any] = {}
    for score_id, calibrator in grid:
        if score_id not in choices:
            choices[score_id] = score_choice(
                score_id,
                models,
                X_selection,
                y_selection,
                selection_ids,
                dataset=dataset,
                seed=seed,
                alpha=campaign.alpha,
            )
        method = method_choice(
            choices[score_id],
            calibrator,
            geometry,
            X_train,
            y_train,
            X_selection,
            y_selection,
            selection_ids,
            y_calibration,
            task=task,
            alpha=campaign.alpha,
            seed=seed,
            n_classes=n_classes,
            campaign_config=campaign.config,
        )
        method.rule.fit(X_train, y_train)
        method.rule.calibrate(
            X_calibration,
            y_calibration,
            row_ids=calibration_ids,
            auxiliary_indices=method.auxiliary_indices,
            proper_indices=method.proper_indices,
        )
        predictions, cutoffs = method.rule.predict_with_cutoffs(
            X_test,
            row_ids=test_ids,
        )
        metadata = {
            **prepared_metadata,
            "alpha": campaign.alpha,
            "profile": campaign.profile,
            "score": method.score_id,
            "calibrator": method.calibrator_id,
            "selection": method.selection,
            "evaluation": campaign.config["evaluation"],
            "software": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "scikit_learn": version("scikit-learn"),
                "torchsom_revision": TORCHSOM_REVISION,
            },
        }
        destination = (
            run_directory(output_dir, dataset, seed)
            / f"{method.score_id}__{method.calibrator_id}"
        )
        _write_run(
            destination,
            metadata=metadata,
            row_ids=test_ids,
            targets=y_test,
            features=X_test,
            predictions=predictions,
            cutoffs=cutoffs,
            external_assignments=external_assignments,
            alpha=campaign.alpha,
            conditional_metrics=campaign.profile == "full",
        )
        outputs.append(destination)
    if campaign.profile == "full" and task == "regression":
        analysis_root = run_directory(output_dir, dataset, seed) / "analyses"
        base_choice = cast(ScoreChoice, choices["abs_residual"])
        _run_sensitivity(
            campaign,
            dataset,
            seed,
            base_choice,
            dataset_config,
            X_train,
            y_train,
            X_calibration,
            y_calibration,
            calibration_ids,
            X_test,
            y_test,
            test_ids,
            external_assignments[group_counts[0]],
            analysis_root,
        )
        _run_regime3(
            campaign,
            dataset,
            seed,
            base_choice,
            dataset_config,
            X_train,
            y_train,
            X_calibration,
            y_calibration,
            calibration_ids,
            X_test,
            y_test,
            test_ids,
            external_assignments[group_counts[0]],
            analysis_root,
        )
    return tuple(outputs)


def run_campaign(
    campaign: Campaign,
    prepared_dir: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Run every method for every prepared dataset and seed."""
    return tuple(
        destination
        for dataset in campaign.datasets
        for seed in campaign.seeds
        for destination in _run_one(
            campaign,
            dataset,
            seed,
            prepared_dir,
            output_dir,
        )
    )


def run(
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    profile: str = "smoke",
    datasets: str | None = None,
    seeds: str | None = None,
    prepared_dir: Path | str = Path("data/prepared"),
    output_dir: Path | str = Path("results/runs"),
) -> tuple[Path, ...]:
    campaign = load_campaign(
        config_path,
        profile=profile,
        datasets=datasets,
        seeds=seeds,
    )
    return run_campaign(campaign, Path(prepared_dir), Path(output_dir))


__all__ = ("load_predictions", "load_prepared", "run", "run_campaign")

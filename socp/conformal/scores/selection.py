"""Selection of the RAPS and SAPS score parameters on the selection split."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from socp.conformal.scores._classification_core import (
    _raps_score_matrix,
    _saps_score_matrix,
    _validated_probabilities,
)

RAPS_PENALTY_GRID: Final = (0.001, 0.01, 0.1, 0.2, 0.5)
SAPS_WEIGHT_GRID: Final = (
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
)
SelectionScore = Literal["raps", "saps"]


@dataclass(frozen=True, slots=True)
class ClassificationScoreSelection:
    """Immutable winning score parameter and every evaluated criterion."""

    score_id: SelectionScore
    parameter_name: str
    selected_value: float
    k_reg: int | None
    grid: tuple[float, ...]
    average_set_sizes: tuple[float, ...]
    cutoffs: tuple[float, ...]
    source_revision: str
    tie_policy: str


def _selection_inputs(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.floating] | NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Validate and align the immutable selection inputs."""
    checked_probabilities = _validated_probabilities(probabilities)
    n_classes = checked_probabilities.shape[1]
    values = np.asarray(labels, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
        or np.any(values < 0.0)
        or np.any(values >= n_classes)
    ):
        raise ValueError(f"selection labels must be integers in [0, {n_classes})")
    checked_labels = values.astype(np.int64)
    checked_labels.setflags(write=False)
    return checked_probabilities, checked_labels


def _fixed_set_rank_quantile(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    alpha: float,
) -> int:
    """Return the higher ``1 - alpha`` quantile of the one-based true-label ranks."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    true_ranks = np.argmax(order == labels[:, None], axis=1) + 1
    return int(np.quantile(true_ranks, 1.0 - alpha, method="higher"))


def _candidate_stream(
    seed: int,
    *,
    owner: int,
    candidate: int,
    phase: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(owner, candidate, phase))
    )


def _stable_tie_breakers(shape: tuple[int, int]) -> NDArray[np.float64]:
    return np.asarray(
        np.broadcast_to(np.arange(shape[1], dtype=np.float64), shape),
        dtype=np.float64,
    )


def _raps_candidate_matrix(
    probabilities: NDArray[np.float64],
    *,
    penalty: float,
    k_reg: int,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    def matrix(
        uniforms: NDArray[np.float64],
        ties: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _raps_score_matrix(
            probabilities,
            uniforms,
            ties,
            penalty=penalty,
            k_reg=k_reg,
        )

    return matrix


def _saps_candidate_matrix(
    probabilities: NDArray[np.float64],
    *,
    weight: float,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    def matrix(
        uniforms: NDArray[np.float64],
        ties: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _saps_score_matrix(
            probabilities,
            uniforms,
            ties,
            weight=weight,
        )

    return matrix


def _candidate_criterion(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    alpha: float,
    seed: int,
    owner: int,
    candidate: int,
    complement_uniforms: bool,
    matrix: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[float, float]:
    """Return the mean set size and the threshold on the given rows.

    The threshold and the set sizes use independent uniform draws.
    """
    ties = _stable_tie_breakers(probabilities.shape)
    calibration_uniforms = _candidate_stream(
        seed,
        owner=owner,
        candidate=candidate,
        phase=0,
    ).uniform(size=len(labels))
    if complement_uniforms:
        calibration_uniforms = 1.0 - calibration_uniforms
    calibration = matrix(calibration_uniforms, ties)
    true_scores = calibration[np.arange(len(labels)), labels]
    cutoff = float(np.quantile(true_scores, 1.0 - alpha, method="higher"))
    prediction_uniforms = _candidate_stream(
        seed,
        owner=owner,
        candidate=candidate,
        phase=1,
    ).uniform(size=len(labels))
    if complement_uniforms:
        prediction_uniforms = 1.0 - prediction_uniforms
    prediction = matrix(prediction_uniforms, ties)
    size = float(np.mean(np.sum(prediction <= cutoff + 1e-12, axis=1)))
    return size, cutoff


def _first_strict_minimum(values: tuple[float, ...]) -> int:
    """Return the index of the first smallest value."""
    winner = 0
    best = math.inf
    for index, value in enumerate(values):
        if value < best:
            winner = index
            best = value
    return winner


def select_raps_parameters(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.floating] | NDArray[np.integer],
    *,
    alpha: float,
    seed: int,
) -> ClassificationScoreSelection:
    """Select the RAPS ``k_reg`` and penalty.

    ``k_reg`` is the rank quantile, and the penalty has the smallest mean set size.
    """
    checked_probabilities, checked_labels = _selection_inputs(probabilities, labels)
    k_reg = _fixed_set_rank_quantile(
        checked_probabilities,
        checked_labels,
        alpha=alpha,
    )
    outcomes = tuple(
        _candidate_criterion(
            checked_probabilities,
            checked_labels,
            alpha=alpha,
            seed=seed,
            owner=1,
            candidate=index,
            complement_uniforms=True,
            matrix=_raps_candidate_matrix(
                checked_probabilities,
                penalty=penalty,
                k_reg=k_reg,
            ),
        )
        for index, penalty in enumerate(RAPS_PENALTY_GRID)
    )
    sizes = tuple(size for size, _ in outcomes)
    cutoffs = tuple(cutoff for _, cutoff in outcomes)
    winner = _first_strict_minimum(sizes)
    return ClassificationScoreSelection(
        score_id="raps",
        parameter_name="penalty",
        selected_value=RAPS_PENALTY_GRID[winner],
        k_reg=k_reg,
        grid=RAPS_PENALTY_GRID,
        average_set_sizes=sizes,
        cutoffs=cutoffs,
        source_revision=(
            "aangelopoulos/conformal_classification@"
            "881eba9d6b1f9243d552989b92eb955a0f17cc5c"
        ),
        tie_policy="stable_label_ties_and_candidate_specific_threshold_set_streams",
    )


def select_saps_weight(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.floating] | NDArray[np.integer],
    *,
    alpha: float,
    seed: int,
) -> ClassificationScoreSelection:
    """Select the SAPS ranking weight with the smallest mean set size."""
    checked_probabilities, checked_labels = _selection_inputs(probabilities, labels)
    outcomes = tuple(
        _candidate_criterion(
            checked_probabilities,
            checked_labels,
            alpha=alpha,
            seed=seed,
            owner=2,
            candidate=index,
            complement_uniforms=False,
            matrix=_saps_candidate_matrix(
                checked_probabilities,
                weight=weight,
            ),
        )
        for index, weight in enumerate(SAPS_WEIGHT_GRID)
    )
    sizes = tuple(size for size, _ in outcomes)
    cutoffs = tuple(cutoff for _, cutoff in outcomes)
    winner = _first_strict_minimum(sizes)
    return ClassificationScoreSelection(
        score_id="saps",
        parameter_name="weight",
        selected_value=SAPS_WEIGHT_GRID[winner],
        k_reg=None,
        grid=SAPS_WEIGHT_GRID,
        average_set_sizes=sizes,
        cutoffs=cutoffs,
        source_revision=(
            "ml-stat-Sustech/conformal_prediction_via_label_ranking@"
            "bc8b19752ce9d5519e52824d157cdfdcffee0b65"
        ),
        tie_policy="stable_label_ties_and_candidate_specific_threshold_set_streams",
    )


__all__ = (
    "ClassificationScoreSelection",
    "RAPS_PENALTY_GRID",
    "SAPS_WEIGHT_GRID",
    "select_raps_parameters",
    "select_saps_weight",
)

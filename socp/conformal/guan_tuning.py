"""Bandwidth selection for LCP on the selection split.

A resampling objective on prediction-set size compares 20 candidate bandwidths. The
size is the interval width for regression and the set size, computed with the
evaluated score, for classification.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from socp.conformal.localized import LocalizedCache, localized_threshold
from socp.ranks import validate_alpha

GUAN_B = 2
GUAN_LAMBDA = 1.0
GUAN_GRID_SIZE = 20
GuanTask = Literal["regression", "classification"]


@dataclass(frozen=True, slots=True)
class GuanBandwidthSelection:
    """One score-aware bandwidth decision and its candidate diagnostics."""

    bandwidth: float | None
    status: Literal["selected", "infeasible"]
    task: GuanTask
    candidates: tuple[float, ...]
    degenerate_rates: tuple[float, ...]
    mean_measures: tuple[float, ...]
    variability: tuple[float, ...]
    objectives: tuple[float, ...]
    selected_index: int | None
    n_selection: int
    n_calibration: int
    B: int
    delta: float
    lambda_variability: float
    grid_floor_adjusted: bool

    def provenance(self) -> dict[str, object]:
        """Return the metadata stored with each experiment."""
        selected_rate = (
            None
            if self.selected_index is None
            else self.degenerate_rates[self.selected_index]
        )
        selected_objective = (
            None
            if self.selected_index is None
            else self.objectives[self.selected_index]
        )
        return {
            "procedure": "LeyingGuan/LCP@64e51668:autoTune_distance",
            "adaptation": (
                "official_regression"
                if self.task == "regression"
                else "score_matched_classification_set_size"
            ),
            "status": self.status,
            "bandwidth": self.bandwidth,
            "selection_rows": self.n_selection,
            "calibration_rows": self.n_calibration,
            "B": self.B,
            "delta": self.delta,
            "lambda_variability": self.lambda_variability,
            "grid": self.candidates,
            "grid_floor_adjusted": self.grid_floor_adjusted,
            "selected_degenerate_rate": selected_rate,
            "selected_objective": selected_objective,
        }


Measure = Callable[[NDArray[np.int64], NDArray[np.float64]], NDArray[np.float64]]


def _rms_distances(X: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.sum(X * X, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.asarray(np.sqrt(squared / X.shape[1]), dtype=np.float64)


def _grid_from_distances(
    distances: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    lower = float(np.quantile(distances, 0.01))
    upper = 2.0 * float(np.max(distances))
    adjusted = lower <= 0.0
    if adjusted:
        positive = distances[distances > 0.0]
        lower = float(np.min(positive)) if positive.size else 1.0
    if upper <= lower:
        upper = 2.0 * lower
        adjusted = True
    grid = np.geomspace(lower, upper, GUAN_GRID_SIZE)
    grid[0], grid[-1] = lower, upper
    return np.asarray(grid, dtype=np.float64), adjusted


def _cache_from_counts(
    scores: NDArray[np.float64],
    weights: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> tuple[LocalizedCache, NDArray[np.intp]]:
    order = np.asarray(np.argsort(scores, kind="stable"), dtype=np.intp)
    sorted_scores = scores[order]
    sorted_counts = counts[order]
    expanded_order = np.asarray(np.repeat(order, sorted_counts), dtype=np.intp)
    if not expanded_order.size:
        raise ValueError("bootstrap sample is empty")

    cumulative = np.cumsum(weights[:, order] * sorted_counts[None, :], axis=1)
    left = np.searchsorted(sorted_scores, scores, side="left")
    q_ell = np.zeros(len(scores), dtype=np.float64)
    has_lower = left > 0
    rows = np.flatnonzero(has_lower)
    q_ell[rows] = cumulative[rows, left[rows] - 1]
    cumulative_counts = np.cumsum(sorted_counts)
    ell = np.zeros(len(scores), dtype=np.intp)
    ell[has_lower] = cumulative_counts[left[has_lower] - 1]
    return (
        LocalizedCache(
            scores[expanded_order],
            ell[expanded_order],
            q_ell[expanded_order],
            cumulative[:, -1][expanded_order],
        ),
        expanded_order,
    )


def _bootstrap_thresholds(
    scores: NDArray[np.float64],
    weights: NDArray[np.float64],
    counts: NDArray[np.int64],
    alpha: float,
) -> NDArray[np.float64]:
    cache, expanded_order = _cache_from_counts(scores, weights, counts)
    return np.asarray(
        [
            localized_threshold(cache, weights[row, expanded_order], 1.0, alpha)
            for row in range(len(scores))
        ],
        dtype=np.float64,
    )


def _leave_one_out_thresholds(
    scores: NDArray[np.float64],
    weights: NDArray[np.float64],
    subset: NDArray[np.int64],
    alpha: float,
) -> NDArray[np.float64]:
    subset_scores = scores[subset]
    order = np.argsort(subset_scores, kind="stable")
    sorted_scores = np.asarray(subset_scores[order], dtype=np.float64)
    ordered_rows = subset[order]
    kernel = weights[np.ix_(ordered_rows, ordered_rows)]
    n_rows = len(sorted_scores)
    full_total = np.sum(kernel, axis=1)
    lower = np.searchsorted(sorted_scores, sorted_scores, side="left")
    cumulative = np.cumsum(kernel, axis=1)
    full_q_ell = np.zeros(n_rows, dtype=np.float64)
    has_lower = lower > 0
    rows = np.flatnonzero(has_lower)
    full_q_ell[rows] = cumulative[rows, lower[rows] - 1]

    sorted_thresholds = np.empty(n_rows, dtype=np.float64)
    positions = np.arange(n_rows)
    for query in range(n_rows):
        keep = positions != query
        remaining_scores = sorted_scores[keep]
        remaining_ell = np.searchsorted(
            remaining_scores,
            remaining_scores,
            side="left",
        ).astype(np.intp)
        q_total = full_total[keep] - kernel[keep, query]
        query_is_lower = sorted_scores[query] < remaining_scores
        q_ell = full_q_ell[keep] - kernel[keep, query] * query_is_lower
        cache = LocalizedCache(remaining_scores, remaining_ell, q_ell, q_total)
        sorted_thresholds[query] = localized_threshold(
            cache,
            kernel[query, keep],
            1.0,
            alpha,
        )
    thresholds = np.empty(n_rows, dtype=np.float64)
    thresholds[order] = sorted_thresholds
    return thresholds


def _validate_geometry(
    train_geometry: NDArray[np.floating],
    selection_geometry: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    train = np.asarray(train_geometry, dtype=np.float64)
    selection = np.asarray(selection_geometry, dtype=np.float64)
    if train.ndim != 2 or selection.ndim != 2 or min(train.shape) < 1:
        raise ValueError("train and selection geometry must be non-empty matrices")
    if selection.shape[0] < 2 or train.shape[1] != selection.shape[1]:
        raise ValueError("selection needs two rows and the same feature dimension")
    if not np.isfinite(train).all() or not np.isfinite(selection).all():
        raise ValueError("geometry must be finite")
    return train, selection


def _select_bandwidth(
    train_geometry: NDArray[np.floating],
    selection_geometry: NDArray[np.floating],
    scores: NDArray[np.floating],
    measure: Measure,
    *,
    full_measure: float,
    task: GuanTask,
    n_calibration: int,
    alpha: float,
    seed: int,
    B: int = GUAN_B,
    lambda_variability: float = GUAN_LAMBDA,
) -> GuanBandwidthSelection:
    validate_alpha(alpha)
    train, selection = _validate_geometry(train_geometry, selection_geometry)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(selection),) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector aligned with selection")
    if n_calibration < 2 or B < 1 or lambda_variability < 0.0:
        raise ValueError("invalid Guan tuning constants")

    scaled = StandardScaler().fit(train).transform(selection)
    distances = _rms_distances(np.asarray(scaled, dtype=np.float64))
    grid, adjusted = _grid_from_distances(distances)
    delta = alpha / 2.0
    n_selection = len(selection)
    n1 = min(n_selection, n_calibration + 1)
    B1 = 1 if n1 >= (2.0 / 3.0) * n_selection else B
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(41,)))
    mean_subsets = tuple(
        np.sort(rng.choice(n_selection, size=n1, replace=False)) for _ in range(B1)
    )
    bootstrap_counts = tuple(
        rng.multinomial(n_calibration, np.full(n_selection, 1.0 / n_selection))
        for _ in range(B)
    )

    rates: list[float] = []
    means: list[float] = []
    variations: list[float] = []
    objectives: list[float] = []
    all_rows = np.arange(n_selection, dtype=np.int64)
    for bandwidth in grid:
        weights = np.exp(-distances / float(bandwidth))
        replicate_rates: list[float] = []
        replicate_means: list[float] = []
        for subset in mean_subsets:
            thresholds = _leave_one_out_thresholds(values, weights, subset, alpha)
            outcomes = np.asarray(measure(subset, thresholds), dtype=np.float64)
            if outcomes.shape != thresholds.shape or np.isnan(outcomes).any():
                raise ValueError("prediction-set measures must align and not be NaN")
            usable = outcomes < full_measure
            replicate_rates.append(1.0 - float(np.mean(usable)))
            replicate_means.append(
                float(np.mean(outcomes[usable])) if usable.any() else math.inf
            )

        resampled = np.empty((n_selection, B), dtype=np.float64)
        for column, counts in enumerate(bootstrap_counts):
            thresholds = _bootstrap_thresholds(values, weights, counts, alpha)
            resampled[:, column] = measure(all_rows, thresholds)
        usable = resampled < full_measure
        deviations: list[NDArray[np.float64]] = []
        for row, mask in zip(resampled, usable, strict=True):
            if mask.any():
                deviations.append((row[mask] - np.mean(row[mask])) ** 2)
        variability = (
            float(np.sqrt(np.mean(np.concatenate(deviations))))
            if deviations
            else math.inf
        )
        rate = float(np.mean(replicate_rates))
        mean = float(np.mean(replicate_means))
        rates.append(rate)
        means.append(mean)
        variations.append(variability)
        objectives.append(mean + lambda_variability * variability)

    feasible = [
        index
        for index, (rate, objective) in enumerate(zip(rates, objectives, strict=True))
        if rate <= delta and math.isfinite(objective)
    ]
    selected = min(feasible, key=objectives.__getitem__) if feasible else None
    return GuanBandwidthSelection(
        bandwidth=None if selected is None else float(grid[selected]),
        status="infeasible" if selected is None else "selected",
        task=task,
        candidates=tuple(float(value) for value in grid),
        degenerate_rates=tuple(rates),
        mean_measures=tuple(means),
        variability=tuple(variations),
        objectives=tuple(objectives),
        selected_index=selected,
        n_selection=n_selection,
        n_calibration=n_calibration,
        B=B,
        delta=delta,
        lambda_variability=lambda_variability,
        grid_floor_adjusted=adjusted,
    )


def tune_guan_lcp_classification_scores(
    train_geometry: NDArray[np.floating],
    selection_geometry: NDArray[np.floating],
    score_matrix: NDArray[np.floating],
    labels: NDArray[np.generic],
    *,
    n_calibration: int,
    alpha: float,
    seed: int,
) -> GuanBandwidthSelection:
    """Select the LCP bandwidth by the set size of the evaluated score."""
    matrix = np.asarray(score_matrix, dtype=np.float64)
    numeric_labels = np.asarray(labels, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(selection_geometry):
        raise ValueError("score_matrix must align with selection geometry")
    if (
        numeric_labels.shape != (len(matrix),)
        or not np.equal(numeric_labels, np.floor(numeric_labels)).all()
    ):
        raise ValueError("labels must be aligned integers")
    integer_labels = numeric_labels.astype(np.int64)
    if np.any(integer_labels < 0) or np.any(integer_labels >= matrix.shape[1]):
        raise ValueError("labels must index score_matrix columns")

    def cardinality(
        rows: NDArray[np.int64],
        thresholds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.asarray(
            np.sum(matrix[rows] <= thresholds[:, None], axis=1),
            dtype=np.float64,
        )

    return _select_bandwidth(
        train_geometry,
        selection_geometry,
        matrix[np.arange(len(matrix)), integer_labels],
        cardinality,
        full_measure=float(matrix.shape[1]),
        task="classification",
        n_calibration=n_calibration,
        alpha=alpha,
        seed=seed,
    )


def tune_guan_lcp_regression_scores(
    train_geometry: NDArray[np.floating],
    selection_geometry: NDArray[np.floating],
    scores: NDArray[np.floating],
    prediction_measure: Measure,
    *,
    n_calibration: int,
    alpha: float,
    seed: int,
) -> GuanBandwidthSelection:
    """Select the LCP bandwidth by the interval width of the evaluated score."""
    return _select_bandwidth(
        train_geometry,
        selection_geometry,
        scores,
        prediction_measure,
        full_measure=math.inf,
        task="regression",
        n_calibration=n_calibration,
        alpha=alpha,
        seed=seed,
    )


__all__ = (
    "GuanBandwidthSelection",
    "tune_guan_lcp_classification_scores",
    "tune_guan_lcp_regression_scores",
)

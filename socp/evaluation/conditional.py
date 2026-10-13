"""Held-out WSC, size-stratified coverage, and ERT diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from covmetrics import ERT, SSC  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from socp.evaluation.metrics import ScalarMetric


def _worst_slab(
    X: NDArray[np.float64],
    covered: NDArray[np.bool_],
    directions: NDArray[np.float64],
    delta: float,
) -> tuple[NDArray[np.float64], float, float] | None:
    window = int(np.ceil(delta * len(X)))
    if window < 1 or len(X) < window:
        return None
    best_coverage = np.inf
    best = None
    for direction in directions:
        projection = X @ direction
        order = np.argsort(projection, kind="stable")
        sorted_coverage = covered[order].astype(np.float64)
        prefix = np.concatenate(([0.0], np.cumsum(sorted_coverage)))
        means = (prefix[window:] - prefix[:-window]) / window
        start = int(np.argmin(means))
        if means[start] < best_coverage:
            best_coverage = float(means[start])
            sorted_projection = projection[order]
            best = (
                direction,
                float(sorted_projection[start]),
                float(sorted_projection[start + window - 1]),
            )
    return best


def held_out_worst_slab_coverage(
    X_search: NDArray[np.floating],
    covered_search: NDArray[np.bool_],
    X_evaluation: NDArray[np.floating],
    covered_evaluation: NDArray[np.bool_],
    *,
    search_ids: NDArray[np.integer],
    evaluation_ids: NDArray[np.integer],
    directions: NDArray[np.floating],
    delta: float,
) -> ScalarMetric:
    """Select the worst slab on one split and evaluate it on a disjoint split."""
    search = np.asarray(X_search, dtype=np.float64)
    evaluation = np.asarray(X_evaluation, dtype=np.float64)
    search_coverage = np.asarray(covered_search, dtype=np.bool_)
    evaluation_coverage = np.asarray(covered_evaluation, dtype=np.bool_)
    search_keys = np.asarray(search_ids)
    evaluation_keys = np.asarray(evaluation_ids)
    direction_values = np.asarray(directions, dtype=np.float64)
    if (
        search.ndim != 2
        or evaluation.ndim != 2
        or search.shape[1] != evaluation.shape[1]
    ):
        raise ValueError("search and evaluation features must have matching columns")
    if search_coverage.shape != (len(search),) or search_keys.shape != (len(search),):
        raise ValueError("search features, coverage, and IDs must align")
    if evaluation_coverage.shape != (len(evaluation),) or evaluation_keys.shape != (
        len(evaluation),
    ):
        raise ValueError("evaluation features, coverage, and IDs must align")
    if np.intersect1d(search_keys, evaluation_keys).size:
        raise ValueError("WSC search and evaluation views must be disjoint")
    if not 0.0 < delta <= 1.0:
        raise ValueError("delta must lie in (0, 1]")
    if direction_values.ndim != 2 or direction_values.shape[1] != search.shape[1]:
        raise ValueError("directions must have shape (n_directions, n_features)")

    slab = _worst_slab(search, search_coverage, direction_values, delta)
    if slab is None:
        return ScalarMetric(None, 0)
    direction, lower, upper = slab
    projection = evaluation @ direction
    mask = (projection >= lower) & (projection <= upper)
    support = int(np.sum(mask))
    return ScalarMetric(
        float(np.mean(evaluation_coverage[mask])) if support else None,
        support,
    )


def worst_slab_coverage(
    X: NDArray[np.floating],
    covered: NDArray[np.bool_],
    *,
    delta: float = 0.1,
    n_directions: int = 500,
    n_splits: int = 10,
    seed: int = 0,
) -> ScalarMetric:
    """Average held-out WSC over seeded random search/evaluation splits."""
    features = np.asarray(X, dtype=np.float64)
    coverage = np.asarray(covered, dtype=np.bool_)
    if features.ndim != 2 or coverage.shape != (len(features),) or len(features) < 2:
        raise ValueError("WSC requires at least two aligned observations")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_splits):
        order = rng.permutation(len(features))
        middle = len(features) // 2
        search, evaluation = order[:middle], order[middle:]
        directions = rng.normal(size=(n_directions, features.shape[1]))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(norms, np.finfo(np.float64).eps)
        result = held_out_worst_slab_coverage(
            features[search],
            coverage[search],
            features[evaluation],
            coverage[evaluation],
            search_ids=search,
            evaluation_ids=evaluation,
            directions=directions,
            delta=delta,
        )
        if result.value is None:
            return ScalarMetric(None, len(features))
        values.append(result.value)
    return ScalarMetric(float(np.mean(values)), len(features))


def size_stratified_coverage(
    sizes: NDArray[np.floating],
    covered: NDArray[np.bool_],
    *,
    alpha: float,
    seed: int,
    n_groups: int = 10,
) -> ScalarMetric:
    """Compute SSC while keeping infinite sizes in the widest stratum."""
    values = np.asarray(sizes, dtype=np.float64)
    coverage = np.asarray(covered, dtype=np.bool_)
    if values.ndim != 1 or coverage.shape != values.shape:
        raise ValueError("sizes and covered must be aligned vectors")
    if np.any(np.isnan(values)):
        raise ValueError("sizes must not contain NaN")
    finite = np.isfinite(values)
    if not finite.any():
        return ScalarMetric(None, len(values))
    sentinel = float(np.max(values[finite])) * 1.05 + 1.0
    safe_sizes = np.where(finite, values, sentinel)
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        result = SSC().evaluate(
            safe_sizes,
            coverage.astype(np.float64),
            alpha=alpha,
            number_max_groups=n_groups,
        )
    return ScalarMetric(float(result), len(values))


def excess_risk_of_target(
    X: NDArray[np.floating],
    covered: NDArray[np.bool_],
    *,
    alpha: float,
    seed: int,
    n_splits: int = 5,
) -> ScalarMetric:
    """Compute ERT with covmetrics.

    When every set covers, it returns ``alpha``, and when none does, ``1 - alpha``.
    """
    features = np.asarray(X, dtype=np.float64)
    coverage = np.asarray(covered, dtype=np.bool_)
    if features.ndim != 2 or coverage.shape != (len(features),):
        raise ValueError("X and covered must be aligned")
    if len(coverage) == 0:
        return ScalarMetric(None, 0)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if coverage.all():
        return ScalarMetric(float(alpha), len(coverage))
    if not coverage.any():
        return ScalarMetric(float(1.0 - alpha), len(coverage))
    estimator = ERT(model_cls=HistGradientBoostingClassifier, random_state=seed)
    value = estimator.evaluate(
        features,
        coverage.astype(np.float64),
        alpha=alpha,
        n_splits=n_splits,
        random_state=seed,
    )
    return ScalarMetric(float(value), len(coverage))


__all__ = (
    "excess_risk_of_target",
    "held_out_worst_slab_coverage",
    "size_stratified_coverage",
    "worst_slab_coverage",
)

"""Coverage and efficiency metrics used in the experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

GroupKey = TypeVar("GroupKey")


@dataclass(frozen=True, slots=True)
class ScalarMetric:
    """A scalar estimate and the number of observations supporting it."""

    value: float | None
    support: int


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Marginal metrics for one method on one evaluation split."""

    support: int
    coverage: float | None
    size_mean: float | None
    empty_rate: float | None
    infinite_cutoff_rate: float | None


@dataclass(frozen=True, slots=True)
class CoverageGaps:
    """Support-weighted coverage gap (WCovGap) over an evaluation partition."""

    wcovgap: float | None
    n_groups: int
    minimum_group_support: int


def _aligned_vectors(
    covered: NDArray[np.bool_],
    sizes: NDArray[np.floating],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    coverage = np.asarray(covered, dtype=np.bool_)
    efficiency = np.asarray(sizes, dtype=np.float64)
    if coverage.ndim != 1 or efficiency.ndim != 1 or coverage.shape != efficiency.shape:
        raise ValueError("covered and sizes must be aligned vectors")
    if np.any(np.isnan(efficiency) | (efficiency < 0.0)):
        raise ValueError("sizes must be non-negative and must not contain NaN")
    return coverage, efficiency


def _optional_rate(
    values: NDArray[np.bool_] | None,
    *,
    support: int,
    name: str,
) -> float | None:
    if values is None:
        return None
    flags = np.asarray(values)
    if flags.dtype != np.bool_ or flags.shape != (support,):
        raise ValueError(f"{name} must be a boolean vector aligned with predictions")
    return float(np.mean(flags)) if support else None


def prediction_metrics(
    covered: NDArray[np.bool_],
    sizes: NDArray[np.floating],
    *,
    empty: NDArray[np.bool_] | None = None,
    infinite_cutoff: NDArray[np.bool_] | None = None,
) -> PredictionMetrics:
    """Summarize raw predictions without replacing missing or infinite values."""
    coverage, efficiency = _aligned_vectors(covered, sizes)
    support = len(coverage)
    return PredictionMetrics(
        support=support,
        coverage=float(np.mean(coverage)) if support else None,
        size_mean=float(np.mean(efficiency)) if support else None,
        empty_rate=_optional_rate(empty, support=support, name="empty"),
        infinite_cutoff_rate=_optional_rate(
            infinite_cutoff,
            support=support,
            name="infinite_cutoff",
        ),
    )


def weighted_coverage_gap(
    group_coverages: Mapping[GroupKey, float],
    group_counts: Mapping[GroupKey, int],
    alpha: float,
) -> float:
    """Support-weighted absolute group-coverage deviation, in percentage points."""
    if group_coverages.keys() != group_counts.keys():
        raise ValueError("group coverages and counts must use the same groups")
    total = sum(group_counts.values())
    if total == 0:
        return float("nan")
    target = 1.0 - alpha
    return 100.0 * sum(
        group_counts[group] / total * abs(coverage - target)
        for group, coverage in group_coverages.items()
    )


def coverage_gaps(
    covered: NDArray[np.bool_],
    assignments: NDArray[np.integer],
    *,
    alpha: float,
) -> CoverageGaps:
    """Compute WCovGap from one fixed evaluation partition."""
    values = np.asarray(covered, dtype=np.bool_)
    groups = np.asarray(assignments)
    if (
        values.ndim != 1
        or groups.shape != values.shape
        or not np.issubdtype(
            groups.dtype,
            np.integer,
        )
    ):
        raise ValueError("covered and assignments must be aligned vectors")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if values.size == 0:
        return CoverageGaps(None, 0, 0)

    unique = np.unique(groups)
    coverages = {
        int(group): float(np.mean(values[groups == group])) for group in unique
    }
    counts = {int(group): int(np.sum(groups == group)) for group in unique}
    return CoverageGaps(
        wcovgap=weighted_coverage_gap(coverages, counts, alpha),
        n_groups=len(unique),
        minimum_group_support=min(counts.values()),
    )


__all__ = (
    "CoverageGaps",
    "PredictionMetrics",
    "ScalarMetric",
    "coverage_gaps",
    "prediction_metrics",
    "weighted_coverage_gap",
)

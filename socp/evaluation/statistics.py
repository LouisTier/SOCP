"""Exact paired Wilcoxon tests with Holm's family-wise correction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A candidate-versus-baseline comparison across paired seeds."""

    n_pairs: int
    mean_delta: float
    wilcoxon_p: float


def holm(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Apply Holm's step-down family-wise error correction.

    Args:
        p_values: Raw p-values in hypothesis order.

    Returns:
        Adjusted p-values in the same order.
    """
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return tuple(float(value) for value in adjusted)


def paired_comparison(
    candidate: NDArray[np.floating],
    baseline: NDArray[np.floating],
) -> PairedComparison:
    """Return the mean paired difference and its exact two-sided Wilcoxon p-value.

    Args:
        candidate: Candidate values by paired seed.
        baseline: Baseline values by paired seed.

    Returns:
        Number of pairs, mean difference, and raw p-value.
    """
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if candidate_values.ndim != 1 or candidate_values.shape != baseline_values.shape:
        raise ValueError("candidate and baseline must be aligned vectors")
    if not (
        np.all(np.isfinite(candidate_values)) and np.all(np.isfinite(baseline_values))
    ):
        raise ValueError("every paired value must be finite")
    deltas = candidate_values - baseline_values
    return PairedComparison(
        n_pairs=int(deltas.size),
        mean_delta=float(np.mean(deltas)),
        wilcoxon_p=exact_wilcoxon_sign_flip(deltas),
    )


def exact_wilcoxon_sign_flip(deltas: NDArray[np.floating]) -> float:
    """Return the exact two-sided signed-rank randomization p-value.

    Zero deltas are discarded, and absolute nonzero deltas receive average ranks,
    including tied magnitudes. The two-sided statistic is ``min(W+, W-)``. Its
    randomization p-value is the fraction of all sign assignments whose statistic is no
    greater than the observed statistic. More than ten nonzero deltas raise
    ``ValueError``.

    Args:
        deltas: Paired candidate-minus-baseline deltas.

    Returns:
        Exact sign-flip p-value.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not bool(np.all(np.isfinite(values))):
        raise ValueError("deltas must be a finite vector")
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return 1.0
    if nonzero.size > 10:
        raise ValueError("exact Wilcoxon enumeration supports at most 10 nonzero pairs")
    ranks = np.asarray(
        stats.rankdata(np.abs(nonzero), method="average"),
        dtype=np.float64,
    )
    observed = min(
        float(np.sum(ranks[nonzero > 0.0])),
        float(np.sum(ranks[nonzero < 0.0])),
    )
    extreme = 0
    total = 1 << int(nonzero.size)
    for signs in product((-1.0, 1.0), repeat=int(nonzero.size)):
        signed_ranks = ranks * np.asarray(signs, dtype=np.float64)
        statistic = min(
            float(np.sum(ranks[signed_ranks > 0.0])),
            float(np.sum(ranks[signed_ranks < 0.0])),
        )
        if statistic <= observed:
            extreme += 1
    return extreme / total


__all__ = (
    "PairedComparison",
    "exact_wilcoxon_sign_flip",
    "holm",
    "paired_comparison",
)

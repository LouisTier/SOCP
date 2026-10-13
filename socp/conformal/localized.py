"""Corrected-level sweep of localized conformal prediction (LCP)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from socp.ranks import validate_alpha


def _finite_vector(
    values: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a detached finite float vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class LocalizedCache:
    """Calibration quantities that the LCP sweep reuses for every query.

    Attributes:
        sorted_scores: Calibration scores in stable ascending order.
        ell: For each score, the number of calibration scores strictly below it.
        q_ell: Localized calibration mass strictly below each point's score.
        q_total: Total localized calibration mass around each calibration point.
    """

    sorted_scores: NDArray[np.float64]
    ell: NDArray[np.intp]
    q_ell: NDArray[np.float64]
    q_total: NDArray[np.float64]

    def __post_init__(self) -> None:
        scores = _finite_vector(self.sorted_scores, name="sorted_scores")
        if np.any(scores[1:] < scores[:-1]):
            raise ValueError("sorted_scores must be non-decreasing")
        n_rows = scores.shape[0]

        ell = np.asarray(self.ell, dtype=np.intp)
        if ell.shape != (n_rows,):
            raise ValueError(f"ell must have shape ({n_rows},), got {ell.shape}")
        if np.any(ell < 0) or np.any(ell > n_rows):
            raise ValueError("ell entries must lie between zero and the cache size")
        immutable_ell = ell.copy()
        immutable_ell.setflags(write=False)

        q_ell = _finite_vector(self.q_ell, name="q_ell")
        q_total = _finite_vector(self.q_total, name="q_total")
        if q_ell.shape != (n_rows,) or q_total.shape != (n_rows,):
            raise ValueError("q_ell and q_total must align with sorted_scores")
        if np.any(q_ell < 0.0) or np.any(q_total < 0.0):
            raise ValueError("localized masses must be non-negative")
        if np.any(q_ell > q_total):
            raise ValueError("q_ell cannot exceed q_total")

        object.__setattr__(self, "sorted_scores", scores)
        object.__setattr__(self, "ell", immutable_ell)
        object.__setattr__(self, "q_ell", q_ell)
        object.__setattr__(self, "q_total", q_total)


def localized_precompute(
    scores: NDArray[np.float64],
    row_weights: Callable[[int], NDArray[np.float64]],
) -> LocalizedCache:
    """Precompute the calibration quantities of the corrected LCP sweep.

    ``row_weights(i)`` receives a position in stable score order and must return that
    calibration point's weights against all calibration points, in the same order.

    Args:
        scores: Finite calibration scores in arbitrary order.
        row_weights: Supplier of one score-ordered localizer row at a time.

    Returns:
        An immutable cache reusable for every query.

    Raises:
        ValueError: If scores or a supplied row violates the numerical contract.
    """
    checked_scores = _finite_vector(scores, name="scores")
    n_rows = checked_scores.shape[0]
    order = np.argsort(checked_scores, kind="stable")
    sorted_scores = checked_scores[order]
    ell = np.searchsorted(sorted_scores, sorted_scores, side="left").astype(
        np.intp,
        copy=False,
    )
    q_ell = np.zeros(n_rows, dtype=np.float64)
    q_total = np.empty(n_rows, dtype=np.float64)

    for row_index in range(n_rows):
        row = np.asarray(row_weights(row_index), dtype=np.float64)
        if row.shape != (n_rows,):
            raise ValueError(
                f"row {row_index} must have shape ({n_rows},), got {row.shape}"
            )
        if not np.all(np.isfinite(row)):
            raise ValueError("localizer weights must be finite")
        if np.any(row < 0.0):
            raise ValueError("localizer weights must be non-negative")
        cumulative = np.cumsum(row, dtype=np.float64)
        q_total[row_index] = cumulative[-1]
        if ell[row_index] > 0:
            q_ell[row_index] = cumulative[ell[row_index] - 1]

    return LocalizedCache(sorted_scores, ell, q_ell, q_total)


def _safe_ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Divide where the denominator is positive and return zero elsewhere."""
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def localized_threshold(
    cache: LocalizedCache,
    test_weights: NDArray[np.float64],
    self_weight: float,
    alpha: float,
) -> float:
    """Return the corrected LCP cutoff for one query.

    The sweep uses strict comparisons and keeps the last crossing. Without a finite
    crossing, the cutoff is ``+inf``.

    Args:
        cache: Calibration-only quantities from :func:`localized_precompute`.
        test_weights: Query-to-calibration weights in cache score order.
        self_weight: Query-to-itself localizer mass.
        alpha: Target miscoverage.

    Returns:
        A calibration score or positive infinity.

    Raises:
        ValueError: If weights, self mass, or alpha are invalid.
    """
    validate_alpha(alpha)
    if not math.isfinite(self_weight) or self_weight <= 0.0:
        raise ValueError(f"self_weight must be finite and positive, got {self_weight}")

    n_rows = cache.sorted_scores.shape[0]
    weights = np.asarray(test_weights, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError(
            f"test_weights must have shape ({n_rows},), got {weights.shape}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("test_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("test_weights must be non-negative")

    cumulative = np.cumsum(weights, dtype=np.float64)
    test_total = float(cumulative[-1])
    test_denominator = test_total + self_weight
    mass_below = np.zeros(n_rows, dtype=np.float64)
    positive_ell = cache.ell > 0
    mass_below[positive_ell] = cumulative[cache.ell[positive_ell] - 1]

    calibration_denominator = cache.q_total + weights
    theta_plus = _safe_ratio(cache.q_ell + weights, calibration_denominator)
    theta = _safe_ratio(cache.q_ell, calibration_denominator)
    theta_tilde = mass_below / test_denominator
    theta_test = test_total / test_denominator

    all_plus = np.append(theta_plus, 1.0)
    all_theta = np.append(theta, theta_test)
    all_tilde = np.append(theta_tilde, theta_test)
    all_ell = np.append(cache.ell, n_rows).astype(np.float64, copy=False)

    in_a1 = all_plus < all_tilde
    in_a2 = all_theta >= all_tilde
    in_a3 = ~in_a1 & ~in_a2
    sorted_a1 = np.sort(all_plus[in_a1])
    sorted_a2 = np.sort(all_theta[in_a2])
    sorted_a3 = np.sort(all_ell[in_a3])

    count_a1 = count_a2 = count_a3 = 0
    last_crossing = -1
    for score_index in range(n_rows + 1):
        level = all_tilde[score_index]
        ell_value = all_ell[score_index]
        while count_a1 < sorted_a1.size and sorted_a1[count_a1] < level:
            count_a1 += 1
        while count_a2 < sorted_a2.size and sorted_a2[count_a2] < level:
            count_a2 += 1
        while count_a3 < sorted_a3.size and sorted_a3[count_a3] < ell_value:
            count_a3 += 1
        corrected_level = (count_a1 + count_a2 + count_a3) / (n_rows + 1.0)
        if corrected_level < 1.0 - alpha:
            last_crossing = score_index

    if last_crossing < 0 or last_crossing >= n_rows:
        return math.inf
    return float(cache.sorted_scores[last_crossing])

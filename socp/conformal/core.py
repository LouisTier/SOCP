"""Finite-sample conformal quantiles."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from socp.ranks import conformal_rank as conformal_rank
from socp.ranks import (
    minimum_finite_calibration_count as minimum_finite_calibration_count,
)
from socp.ranks import validate_alpha as _validate_alpha

_TIE_TOLERANCE: float = 1e-12


def _finite_vector(
    values: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return ``values`` as a finite one-dimensional float array.

    Args:
        values: Candidate numerical vector.
        name: Field name used in error messages.

    Returns:
        A NumPy float64 view or copy.

    Raises:
        ValueError: If the array is empty, non-vectorial, or non-finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def split_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Return the exact unweighted split-conformal threshold.

    Args:
        scores: Finite calibration nonconformity scores, in any order.
        alpha: Target miscoverage.

    Returns:
        The ``conformal_rank(n, alpha)``-th smallest score, or exactly ``+inf``
        when that rank exceeds the number of calibration scores.

    Raises:
        ValueError: If ``scores`` or ``alpha`` violates the numerical contract.
    """
    checked_scores = _finite_vector(scores, name="scores")
    rank = conformal_rank(int(checked_scores.shape[0]), alpha)
    if rank > checked_scores.shape[0]:
        return math.inf
    return float(np.partition(checked_scores, rank - 1)[rank - 1])


def weighted_quantile(
    scores: NDArray[np.float64],
    weights: NDArray[np.float64],
    self_weight: float,
    alpha: float,
) -> float:
    """Return a weighted conformal threshold with query mass at ``+inf``.

    Calibration masses are proportional to ``weights``, and the query adds
    ``self_weight`` at ``+inf``.

    Args:
        scores: Finite calibration nonconformity scores.
        weights: Finite non-negative calibration weights.
        self_weight: Finite positive query weight.
        alpha: Target miscoverage.

    Returns:
        The smallest finite score whose cumulative mass reaches ``1 - alpha``, or
        exactly ``+inf`` when the query mass prevents a finite threshold.

    Raises:
        ValueError: If shapes, finiteness, signs, ``self_weight``, or ``alpha`` are
            invalid.
    """
    _validate_alpha(alpha)
    checked_scores = _finite_vector(scores, name="scores")
    checked_weights = _finite_vector(weights, name="weights")
    if checked_scores.shape[0] != checked_weights.shape[0]:
        raise ValueError(
            "scores and weights must have the same length, got "
            f"{checked_scores.shape[0]} and {checked_weights.shape[0]}"
        )
    if np.any(checked_weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not math.isfinite(self_weight) or self_weight <= 0.0:
        raise ValueError(f"self_weight must be finite and positive, got {self_weight}")

    order = np.argsort(checked_scores, kind="stable")
    sorted_scores = checked_scores[order]
    sorted_weights = checked_weights[order]
    calibration_mass = float(sorted_weights.sum())
    total_mass = calibration_mass + self_weight
    if not math.isfinite(total_mass):
        raise ValueError("weights and self_weight must have a finite total mass")

    target = (1.0 - alpha) * total_mass * (1.0 - _TIE_TOLERANCE)
    cumulative_mass = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative_mass, target, side="left"))
    if index >= sorted_scores.shape[0]:
        return math.inf
    return float(sorted_scores[index])

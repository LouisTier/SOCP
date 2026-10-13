"""Probability ordering and score matrices for multiclass CP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_PROBABILITY_ATOL = 1e-6


def _validated_probabilities(
    probabilities: NDArray[np.float64],
    *,
    n_rows: int | None = None,
    n_classes: int | None = None,
) -> NDArray[np.float64]:
    """Return a finite row-stochastic probability matrix."""
    checked = np.asarray(probabilities, dtype=np.float64)
    if checked.ndim != 2 or checked.shape[0] == 0 or checked.shape[1] < 2:
        raise ValueError(
            "probabilities must have shape (n, C) with n > 0 and C >= 2, "
            f"got {checked.shape}"
        )
    if n_rows is not None and checked.shape[0] != n_rows:
        raise ValueError(
            f"probability predictor returned {checked.shape[0]} rows, expected {n_rows}"
        )
    if n_classes is not None and checked.shape[1] != n_classes:
        raise ValueError(
            "probability predictor changed its class count from "
            f"{n_classes} to {checked.shape[1]}"
        )
    if not np.all(np.isfinite(checked)) or np.any(checked < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(
        checked.sum(axis=1),
        1.0,
        atol=_PROBABILITY_ATOL,
        rtol=_PROBABILITY_ATOL,
    ):
        raise ValueError("probability rows must sum to one")
    return checked


def _validated_uniforms(
    uniforms: NDArray[np.float64],
    *,
    n_rows: int,
) -> NDArray[np.float64]:
    """Validate one boundary randomizer per observation."""
    checked = np.asarray(uniforms, dtype=np.float64)
    if checked.shape != (n_rows,):
        raise ValueError(f"uniforms must have shape ({n_rows},), got {checked.shape}")
    if not np.all(np.isfinite(checked)) or np.any((checked < 0.0) | (checked > 1.0)):
        raise ValueError("uniforms must lie in the closed unit interval")
    return checked


def _validated_tie_breakers(
    tie_breakers: NDArray[np.float64],
    *,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Validate independent keys used only to order equal probabilities."""
    checked = np.asarray(tie_breakers, dtype=np.float64)
    if checked.shape != shape:
        raise ValueError(f"tie_breakers must have shape {shape}, got {checked.shape}")
    if not np.all(np.isfinite(checked)):
        raise ValueError("tie_breakers must contain only finite values")
    return checked


def _descending_order(
    probabilities: NDArray[np.float64],
    tie_breakers: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Order probabilities descending and exact ties by supplied random keys."""
    labels = np.arange(probabilities.shape[1], dtype=np.int64)
    rows = [
        np.lexsort((labels, tie_breakers[row], -probabilities[row]))
        for row in range(probabilities.shape[0])
    ]
    return np.asarray(rows, dtype=np.int64)


def _restore_label_order(
    sorted_values: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Move rank-ordered values back to their original label columns."""
    restored = np.empty_like(sorted_values)
    np.put_along_axis(restored, order, sorted_values, axis=1)
    return restored


def _ordered_score_matrix(
    probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    tie_breakers: NDArray[np.float64],
    *,
    rank_penalty: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return cumulative-before plus one coupled boundary fraction per row."""
    order = _descending_order(probabilities, tie_breakers)
    ordered = np.take_along_axis(probabilities, order, axis=1)
    cumulative_before = np.cumsum(ordered, axis=1) - ordered
    sorted_scores = cumulative_before + uniforms[:, None] * ordered
    if rank_penalty is not None:
        sorted_scores = sorted_scores + rank_penalty[None, :]
    return _restore_label_order(sorted_scores, order)


def _lac_score_matrix(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``1 - p_y`` for every candidate label."""
    return np.asarray(1.0 - probabilities, dtype=np.float64)


def _label_ranking_score_matrix(
    probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the rank-only score ``rank - 1 + U``, with one uniform per row.

    Stable sorting breaks exact probability ties by ascending class index, so every row
    contains ranks ``1, ..., C``.
    """
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty_like(order)
    positions = np.broadcast_to(
        np.arange(1, probabilities.shape[1] + 1, dtype=np.int64),
        probabilities.shape,
    )
    np.put_along_axis(ranks, order, positions, axis=1)
    return np.asarray(ranks - 1 + uniforms[:, None], dtype=np.float64)


def _aps_score_matrix(
    probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    tie_breakers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return APS scores, with one uniform shared by each row."""
    return _ordered_score_matrix(probabilities, uniforms, tie_breakers)


def _raps_score_matrix(
    probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    tie_breakers: NDArray[np.float64],
    *,
    penalty: float,
    k_reg: int,
) -> NDArray[np.float64]:
    """Return APS plus ``penalty * max(rank - k_reg, 0)``."""
    ranks = np.arange(1, probabilities.shape[1] + 1, dtype=np.float64)
    rank_penalty = penalty * np.maximum(ranks - k_reg, 0.0)
    return _ordered_score_matrix(
        probabilities,
        uniforms,
        tie_breakers,
        rank_penalty=rank_penalty,
    )


def _saps_score_matrix(
    probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    tie_breakers: NDArray[np.float64],
    *,
    weight: float,
) -> NDArray[np.float64]:
    """Return SAPS after replacing every non-maximum mass by ``weight``."""
    order = _descending_order(probabilities, tie_breakers)
    ordered_probabilities = np.take_along_axis(probabilities, order, axis=1)
    effective_masses = np.full_like(ordered_probabilities, weight)
    effective_masses[:, 0] = ordered_probabilities[:, 0]
    cumulative_before = np.cumsum(effective_masses, axis=1) - effective_masses
    sorted_scores = cumulative_before + uniforms[:, None] * effective_masses
    return _restore_label_order(sorted_scores, order)

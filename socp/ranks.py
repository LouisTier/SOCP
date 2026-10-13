"""Finite-sample conformal-rank arithmetic."""

from __future__ import annotations

import math

_RANK_TOLERANCE: float = 1e-9


def validate_alpha(alpha: float) -> None:
    """Validate a miscoverage level.

    Args:
        alpha: Target miscoverage.

    Raises:
        ValueError: If ``alpha`` is not finite and strictly between zero and one.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha must be finite and lie strictly in (0, 1), got {alpha}"
        )


def conformal_rank(n: int, alpha: float) -> int:
    """Return the one-based split-conformal order-statistic index.

    Args:
        n: Number of calibration scores.
        alpha: Target miscoverage.

    Returns:
        ``ceil((n + 1) * (1 - alpha))``, floored at one. A value greater
        than ``n`` means that the finite-sample threshold is ``+inf``.

    Raises:
        ValueError: If ``n`` is negative or ``alpha`` is invalid.
    """
    validate_alpha(alpha)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return max(1, math.ceil((n + 1) * (1.0 - alpha) - _RANK_TOLERANCE))


def minimum_finite_calibration_count(alpha: float) -> int:
    """Return the smallest calibration count yielding a finite threshold.

    Args:
        alpha: Target miscoverage.

    Returns:
        The smallest ``n`` for which ``conformal_rank(n, alpha) <= n``.
    """
    conformal_rank(0, alpha)
    lower = 0
    upper = 1
    while conformal_rank(upper, alpha) > upper:
        lower = upper
        upper *= 2

    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if conformal_rank(middle, alpha) <= middle:
            upper = middle
        else:
            lower = middle
    return upper

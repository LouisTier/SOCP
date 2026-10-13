"""CHR interval construction from a fixed conditional histogram."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

_MASS_TOLERANCE = 1e-12
_SOURCE_DENOMINATOR_GUARD = 1e-5


def _validated_probabilities(
    values: NDArray[np.float64],
    *,
    n_rows: int,
    n_bins: int,
) -> NDArray[np.float64]:
    """Return finite, non-negative, normalized conditional histograms."""
    probabilities = np.asarray(values, dtype=np.float64)
    expected = (n_rows, n_bins)
    if probabilities.shape != expected:
        raise ValueError(
            f"histogram predictor returned shape {probabilities.shape}, "
            f"expected {expected}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("histogram probabilities must be finite")
    if np.any(probabilities < 0.0):
        raise ValueError("histogram probabilities must be non-negative")
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("each histogram row must have positive total mass")
    if not np.allclose(totals, 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("each histogram row must sum to one")
    normalized: NDArray[np.float64] = np.asarray(
        probabilities / totals[:, None],
        dtype=np.float64,
    )
    return normalized


def _validated_mass_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
    masses = np.asarray(values, dtype=np.float64)
    if masses.ndim != 1 or masses.size < 2:
        raise ValueError(
            "CHR requires a one-dimensional histogram with at least two bins"
        )
    checked = _validated_probabilities(
        masses[None, :],
        n_rows=1,
        n_bins=masses.size,
    )
    return np.asarray(checked[0], dtype=np.float64)


def _shortest_window(
    cumulative: NDArray[np.float64],
    tau: float,
    lo: int,
    hi: int,
    *,
    must_lo: int | None = None,
    must_hi: int | None = None,
) -> tuple[int, int]:
    """Return the shortest bin window whose mass reaches ``tau``.

    Among windows of equal width, the one with the smallest mass wins, then the
    leftmost.
    """
    if (must_lo is None) != (must_hi is None):
        raise ValueError("must_lo and must_hi must be supplied together")
    if not 0 <= lo <= hi < cumulative.size - 1:
        raise ValueError(f"invalid admissible window [{lo}, {hi}]")
    if must_lo is not None:
        assert must_hi is not None
        if not lo <= must_lo <= must_hi <= hi:
            raise ValueError(
                "the required interval must lie inside the admissible window"
            )
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must lie strictly between zero and one, got {tau}")

    first_end = must_hi if must_hi is not None else lo
    last_start = must_lo if must_lo is not None else hi
    masses = np.diff(cumulative)
    start = lo
    current_mass = float(cumulative[first_end] - cumulative[start])
    best: tuple[int, int] | None = None
    best_length = hi - lo + 2
    best_mass = math.inf

    for end in range(first_end, hi + 1):
        current_mass += float(masses[end])
        while (
            current_mass + _MASS_TOLERANCE >= tau
            and start <= end
            and start <= last_start
        ):
            length = end - start + 1
            mass_better = current_mass < best_mass - _MASS_TOLERANCE
            mass_tied = math.isclose(
                current_mass,
                best_mass,
                rel_tol=0.0,
                abs_tol=_MASS_TOLERANCE,
            )
            leftmost = best is None or start < best[0]
            if length < best_length or (
                length == best_length and (mass_better or (mass_tied and leftmost))
            ):
                best = (start, end)
                best_length = length
                best_mass = current_mass
            current_mass -= float(masses[start])
            start += 1

    if best is None:
        raise RuntimeError(f"no CHR window in [{lo}, {hi}] reaches target mass {tau}")
    return best


def _randomize_window(
    cumulative: NDArray[np.float64],
    masses: NDArray[np.float64],
    window: tuple[int, int],
    tau: float,
    epsilon: float,
) -> tuple[int, int]:
    """Randomly drop the lighter endpoint bin of a window.

    The bin is dropped when ``epsilon <= (mass - tau) / endpoint_mass``.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
    lo, hi = window
    if hi - lo <= 1:
        return window
    total = float(cumulative[hi + 1] - cumulative[lo])
    drop_left = float(masses[lo]) < float(masses[hi])
    endpoint_mass = float(masses[lo] if drop_left else masses[hi])
    probability = (total - tau) / (endpoint_mass + _SOURCE_DENOMINATOR_GUARD)
    if epsilon > probability:
        return window
    return (lo + 1, hi) if drop_left else (lo, hi - 1)


def _nested_sequence(
    probabilities: NDArray[np.float64],
    *,
    n_levels: int,
    anchor_level: int,
    epsilon: float | None,
) -> NDArray[np.int64]:
    """Build the nested intervals on levels ``0, ..., n_levels``.

    Below the anchor, each interval is intersected with its randomized successor, so the
    family stays nested.
    """
    masses = _validated_mass_vector(probabilities)
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least two, got {n_levels}")
    if not 1 <= anchor_level < n_levels:
        raise ValueError(
            f"anchor_level must lie in 1..{n_levels - 1}, got {anchor_level}"
        )
    if epsilon is not None and not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")

    n_bins = masses.size
    cumulative = np.concatenate(([0.0], np.cumsum(masses)))
    deterministic = np.empty((n_levels + 1, 2), dtype=np.int64)
    sequence = np.empty((n_levels + 1, 2), dtype=np.int64)
    deterministic[0] = sequence[0] = (n_bins, -1)
    deterministic[n_levels] = sequence[n_levels] = (0, n_bins - 1)

    anchor_tau = anchor_level / n_levels
    deterministic[anchor_level] = _shortest_window(
        cumulative,
        anchor_tau,
        0,
        n_bins - 1,
    )
    sequence[anchor_level] = deterministic[anchor_level]
    if epsilon is not None:
        sequence[anchor_level] = _randomize_window(
            cumulative,
            masses,
            (
                int(deterministic[anchor_level, 0]),
                int(deterministic[anchor_level, 1]),
            ),
            anchor_tau,
            epsilon,
        )

    for level in range(anchor_level + 1, n_levels):
        previous_lo, previous_hi = (
            int(sequence[level - 1, 0]),
            int(sequence[level - 1, 1]),
        )
        required = (
            {}
            if previous_lo > previous_hi
            else {"must_lo": previous_lo, "must_hi": previous_hi}
        )
        deterministic[level] = _shortest_window(
            cumulative,
            level / n_levels,
            0,
            n_bins - 1,
            **required,
        )
        candidate = (
            int(deterministic[level, 0]),
            int(deterministic[level, 1]),
        )
        if epsilon is not None:
            candidate = _randomize_window(
                cumulative,
                masses,
                candidate,
                level / n_levels,
                epsilon,
            )
        candidate_nested = previous_lo > previous_hi or (
            candidate[0] <= previous_lo and previous_hi <= candidate[1]
        )
        sequence[level] = candidate if candidate_nested else deterministic[level]

    for level in range(anchor_level - 1, 0, -1):
        successor_lo, successor_hi = (
            int(deterministic[level + 1, 0]),
            int(deterministic[level + 1, 1]),
        )
        deterministic[level] = _shortest_window(
            cumulative,
            level / n_levels,
            successor_lo,
            successor_hi,
        )
        candidate = (
            int(deterministic[level, 0]),
            int(deterministic[level, 1]),
        )
        if epsilon is not None:
            candidate = _randomize_window(
                cumulative,
                masses,
                candidate,
                level / n_levels,
                epsilon,
            )
        randomized_successor = sequence[level + 1]
        sequence[level] = (
            max(candidate[0], int(randomized_successor[0])),
            min(candidate[1], int(randomized_successor[1])),
        )

    for level, (current, following) in enumerate(
        zip(sequence[:-1], sequence[1:], strict=True)
    ):
        lo, hi = int(current[0]), int(current[1])
        next_lo, next_hi = int(following[0]), int(following[1])
        if lo <= hi and not (next_lo <= lo and hi <= next_hi):
            raise RuntimeError(
                f"CHR nesting failed between levels {level} and {level + 1}"
            )
    return sequence


def _score_response(
    sequence: NDArray[np.int64],
    edges: NDArray[np.float64],
    response: float,
) -> float:
    """Return the first level whose closed interval contains ``response``."""
    top_level = sequence.shape[0] - 1
    if response < float(edges[0]) or response > float(edges[-1]):
        return float(top_level)
    for level, (lo_value, hi_value) in enumerate(sequence):
        lo, hi = int(lo_value), int(hi_value)
        if lo <= hi and float(edges[lo]) <= response <= float(edges[hi + 1]):
            return float(level)
    return float(top_level)


def _scores_from_probabilities(
    probabilities: NDArray[np.float64],
    edges: NDArray[np.float64],
    responses: NDArray[np.float64],
    *,
    n_levels: int,
    anchor_level: int,
    uniforms: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Evaluate one CHR score per aligned histogram/response/uniform row."""
    if responses.shape != (probabilities.shape[0],):
        raise ValueError(
            f"responses have shape {responses.shape}, expected {(probabilities.shape[0],)}"
        )
    if uniforms is not None and uniforms.shape != (probabilities.shape[0],):
        raise ValueError(
            f"uniforms have shape {uniforms.shape}, expected {(probabilities.shape[0],)}"
        )
    scores = np.empty(probabilities.shape[0], dtype=np.float64)
    for row, (masses, response) in enumerate(
        zip(probabilities, responses, strict=True)
    ):
        epsilon = None if uniforms is None else float(uniforms[row])
        sequence = _nested_sequence(
            masses,
            n_levels=n_levels,
            anchor_level=anchor_level,
            epsilon=epsilon,
        )
        scores[row] = _score_response(sequence, edges, float(response))
    return scores

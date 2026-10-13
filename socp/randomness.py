"""Independent seeds and observation-keyed uniforms for paired experiments."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


def _seed(base_seed: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    return base_seed


def derive_seed(base_seed: int, owner: str) -> int:
    """Derive one stable 32-bit seed for an independent scientific component."""
    if not owner.strip():
        raise ValueError("seed owner must not be empty")
    payload = f"socp-owned-seed-v1\0{_seed(base_seed)}\0{owner}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def _uniform(
    dataset: str,
    base_seed: int,
    score: str,
    row_id: int,
    purpose: str,
) -> float:
    payload = "\0".join(
        (
            "socp-observation-draw-v1",
            dataset,
            str(base_seed),
            score,
            str(row_id),
            purpose,
        )
    ).encode()
    bits = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return float(bits >> 11) / float(1 << 53)


def observation_uniforms(
    dataset: str,
    base_seed: int,
    score: str,
    row_ids: NDArray[np.integer],
    *,
    purpose: str,
) -> NDArray[np.float64]:
    """Return uniforms fixed by dataset, seed, score, row identity, and purpose."""
    _seed(base_seed)
    if not dataset.strip() or not score.strip() or not purpose.strip():
        raise ValueError("observation draw identities must not be empty")
    identifiers = np.asarray(row_ids)
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise ValueError("row_ids must be a one-dimensional integer array")
    return np.asarray(
        [_uniform(dataset, base_seed, score, int(row), purpose) for row in identifiers],
        dtype=np.float64,
    )


def observation_tie_breakers(
    dataset: str,
    base_seed: int,
    score: str,
    row_ids: NDArray[np.integer],
    *,
    n_classes: int,
) -> NDArray[np.float64]:
    """Return an independent observation-keyed tie key for every candidate label."""
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError("n_classes must be an integer of at least two")
    return np.column_stack(
        [
            observation_uniforms(
                dataset,
                base_seed,
                score,
                row_ids,
                purpose=f"tie_breaker:label={label}",
            )
            for label in range(n_classes)
        ]
    )


__all__ = ("derive_seed", "observation_tie_breakers", "observation_uniforms")

"""Shared numerical boundary checks for conformal components."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def immutable_finite_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Validate and copy a finite numerical array into immutable float64 storage.

    Args:
        values: Numerical input.
        name: Field name for diagnostics.
        ndim: Required number of dimensions.
        allow_empty: Whether a zero-length dimension is permitted.

    Returns:
        A read-only float64 array detached from caller-owned storage.

    Raises:
        ValueError: If dimensionality, size, or finiteness is invalid.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    immutable = array.copy()
    immutable.setflags(write=False)
    return immutable


def feature_matrix(values: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    """Return a non-empty immutable finite feature matrix."""
    matrix = immutable_finite_array(values, name=name, ndim=2)
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    return matrix


def feature_row(values: ArrayLike, *, name: str = "x") -> NDArray[np.float64]:
    """Return a non-empty immutable finite feature row."""
    return immutable_finite_array(values, name=name, ndim=1)


def immutable_row_ids(
    values: ArrayLike,
    *,
    expected_rows: int,
    name: str = "row_ids",
) -> NDArray[np.int64]:
    """Return unique immutable integer identities aligned with a row block."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != expected_rows:
        raise ValueError(
            f"{name} must align with {expected_rows} rows, got {raw.shape}"
        )
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite integers") from error
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain finite integers")
    identities = np.asarray(numeric, dtype=np.int64)
    if np.unique(identities).size != identities.size:
        raise ValueError(f"{name} must contain unique identities")
    immutable = identities.copy()
    immutable.setflags(write=False)
    return immutable

"""Localizer protocol.

A localizer fits feature geometry and returns non-negative weights, without seeing
scores, outcomes, or ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from socp.conformal._validation import (
    feature_matrix,
    feature_row,
    immutable_finite_array,
)

KernelType = Literal["indicator", "smooth"]


@dataclass(frozen=True, slots=True)
class LocalizerSpec:
    """Immutable metadata describing localization geometry."""

    kernel: KernelType
    embedding_id: str

    def __post_init__(self) -> None:
        if self.kernel not in ("indicator", "smooth"):
            raise ValueError(
                f"kernel must be 'indicator' or 'smooth', got {self.kernel!r}"
            )
        if not self.embedding_id.strip():
            raise ValueError("embedding_id must not be empty")


@runtime_checkable
class Localizer(Protocol):
    """Fit feature geometry and weight reference points for a query."""

    spec: LocalizerSpec

    @property
    def is_fitted(self) -> bool:
        """Whether geometry has been frozen."""
        ...

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit geometry on features only."""
        ...

    def weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one non-negative weight per reference row."""
        ...


def _reference_matrix(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate references while preserving safe identity-based cache keys."""
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.dtype(np.float64)
        and values.ndim == 2
        and values.size > 0
        and values.shape[1] > 0
        and values.flags.owndata
        and not values.flags.writeable
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError("X_ref must contain only finite values")
        return values
    return feature_matrix(values, name="X_ref")


def fit_localizer(localizer: Localizer, X: NDArray[np.float64]) -> None:
    """Validate feature-only fit data and verify the lifecycle transition."""
    checked_X = feature_matrix(X)
    localizer.fit(checked_X)
    if not localizer.is_fitted:
        raise RuntimeError(
            f"localizer {localizer.spec.embedding_id!r} did not enter fitted state"
        )


def localizer_weights(
    localizer: Localizer,
    x: NDArray[np.float64],
    X_ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return validated weights from frozen feature geometry."""
    if not localizer.is_fitted:
        raise RuntimeError(
            f"localizer {localizer.spec.embedding_id!r} must be fit before use"
        )
    checked_x = feature_row(x)
    checked_reference = _reference_matrix(X_ref)
    if checked_x.shape[0] != checked_reference.shape[1]:
        raise ValueError(
            "x and X_ref must have the same feature dimension, got "
            f"{checked_x.shape[0]} and {checked_reference.shape[1]}"
        )

    raw_weights = np.asarray(
        localizer.weights(checked_x, checked_reference),
        dtype=np.float64,
    )
    expected_shape = (checked_reference.shape[0],)
    if raw_weights.shape != expected_shape:
        raise ValueError(
            f"localizer returned shape {raw_weights.shape}, expected {expected_shape}"
        )
    checked_weights = immutable_finite_array(
        raw_weights,
        name="weights",
        ndim=1,
    )
    if np.any(checked_weights < 0.0):
        raise ValueError("localizer weights must be non-negative")
    return checked_weights

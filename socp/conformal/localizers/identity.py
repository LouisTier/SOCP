"""Uniform geometry representing the absence of localization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from socp.conformal.localizer import LocalizerSpec


class IdentityLocalizer:
    """Assign every row to one cell and every reference a unit weight."""

    spec = LocalizerSpec(kernel="indicator", embedding_id="identity")

    def __init__(self) -> None:
        self._is_fitted = False
        self._n_features: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the feature dimension has been frozen."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64]) -> None:
        """Freeze the feature dimension; uniform geometry has no parameters."""
        if self._is_fitted:
            raise RuntimeError("localizer is already fitted")
        self._n_features = int(X.shape[1])
        self._is_fitted = True

    def _check_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate a batch against the frozen feature dimension."""
        if not self._is_fitted or self._n_features is None:
            raise RuntimeError("localizer must be fitted before use")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._n_features:
            raise ValueError(
                f"X must have shape (n, {self._n_features}), got {values.shape}"
            )
        return values

    def weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one for every validated reference row."""
        query = np.asarray(x, dtype=np.float64)
        references = self._check_matrix(X_ref)
        if query.shape != (references.shape[1],):
            raise ValueError("x and X_ref must have the same feature dimension")
        return np.ones(references.shape[0], dtype=np.float64)

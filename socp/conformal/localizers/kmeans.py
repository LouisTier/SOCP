"""Frozen standardized k-means partition geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from socp.conformal.localizer import LocalizerSpec


def _same_cell_weights(
    query_cell: int,
    reference_cells: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Return partition-label-permutation-invariant indicator weights."""
    cells = np.asarray(reference_cells, dtype=np.intp)
    if cells.ndim != 1:
        raise ValueError("reference_cells must be a vector")
    return np.asarray(cells == query_cell, dtype=np.float64)


class KMeansLocalizer:
    """Indicator weights from k-means fit on standardized training features."""

    spec = LocalizerSpec(kernel="indicator", embedding_id="identity")

    def __init__(self, *, n_clusters: int, random_state: int) -> None:
        if (
            isinstance(n_clusters, bool)
            or int(n_clusters) != n_clusters
            or n_clusters < 1
        ):
            raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}")
        self._n_clusters = int(n_clusters)
        self._model = KMeans(
            n_clusters=self._n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self._scaler: StandardScaler | None = None
        self._is_fitted = False
        self._n_features: int | None = None
        self._reference_cache: tuple[NDArray[np.float64], NDArray[np.intp]] | None = (
            None
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the standardizer and cell prototypes are frozen."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit standardized k-means exactly once on pre-calibration geometry."""
        if self._is_fitted:
            raise RuntimeError("localizer is already fitted")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(f"X must be a non-empty matrix, got {values.shape}")
        if values.shape[0] < self._n_clusters:
            raise ValueError("X must contain at least n_clusters rows")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        self._n_features = int(values.shape[1])
        self._scaler = StandardScaler().fit(values)
        self._model.fit(self._scaler.transform(values))
        self._reference_cache = None
        self._is_fitted = True

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate and standardize rows in the frozen training geometry."""
        if not self._is_fitted or self._scaler is None or self._n_features is None:
            raise RuntimeError("localizer must be fitted before assignment")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._n_features:
            raise ValueError(
                f"X must have shape (n, {self._n_features}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return np.asarray(self._scaler.transform(values), dtype=np.float64)

    def assign(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
        """Assign rows to frozen k-means cells."""
        return np.asarray(self._model.predict(self._transform(X)), dtype=np.intp)

    def _reference_cells(
        self,
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.intp]:
        """Cache assignments for a repeatedly reused multi-row reference object."""
        if X_ref.shape[0] <= 1:
            return self.assign(X_ref)
        if self._reference_cache is None or self._reference_cache[0] is not X_ref:
            self._reference_cache = (X_ref, self.assign(X_ref))
        return self._reference_cache[1]

    def weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one exactly for references sharing the query's frozen cell."""
        query = np.asarray(x, dtype=np.float64)
        references = np.asarray(X_ref, dtype=np.float64)
        if query.ndim != 1 or references.ndim != 2:
            raise ValueError("x must be one row and X_ref must be a matrix")
        query_cell = int(self.assign(query[None, :])[0])
        return _same_cell_weights(query_cell, self._reference_cells(references))

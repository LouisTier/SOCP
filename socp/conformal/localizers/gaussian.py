"""Frozen standardized Gaussian and Laplace kernel geometry."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from socp.conformal.localizer import LocalizerSpec


class _StandardizedKernelLocalizer:
    """Shared one-shot coordinate handling for fixed global kernels."""

    def __init__(
        self,
        bandwidth: float,
        *,
        standardize: bool = True,
        embedding_id: str = "identity",
    ) -> None:
        if math.isnan(bandwidth) or bandwidth <= 0.0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        if not isinstance(standardize, bool):
            raise TypeError("standardize must be boolean")
        self.spec = LocalizerSpec(kernel="smooth", embedding_id=embedding_id)
        self._bandwidth = float(bandwidth)
        self._standardize = standardize
        self._scaler: StandardScaler | None = None
        self._n_features: int | None = None
        self._is_fitted = False
        self._reference_cache: (
            tuple[NDArray[np.float64], NDArray[np.float64]] | None
        ) = None

    @property
    def is_fitted(self) -> bool:
        """Whether kernel coordinates have been frozen."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit the optional standardizer once on declared pre-calibration rows."""
        if self._is_fitted:
            raise RuntimeError("localizer is already fitted")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(f"X must be a non-empty matrix, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        self._n_features = int(values.shape[1])
        if self._standardize:
            self._scaler = StandardScaler().fit(values)
        self._reference_cache = None
        self._is_fitted = True

    def kernel_coordinates(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map rows to the exact frozen coordinates used by the radial profile."""
        if not self._is_fitted or self._n_features is None:
            raise RuntimeError("localizer must be fitted before kernel_coordinates")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._n_features:
            raise ValueError(
                f"X must have shape (n, {self._n_features}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        if self._scaler is None:
            return values.copy()
        return np.asarray(self._scaler.transform(values), dtype=np.float64)

    def _reference_coordinates(
        self,
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transform and cache a repeated multi-row reference object."""
        if X_ref.shape[0] <= 1:
            return self.kernel_coordinates(X_ref)
        if self._reference_cache is None or self._reference_cache[0] is not X_ref:
            self._reference_cache = (X_ref, self.kernel_coordinates(X_ref))
        return self._reference_cache[1]

    def weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the frozen radial profile around one query."""
        query = np.asarray(x, dtype=np.float64)
        references = np.asarray(X_ref, dtype=np.float64)
        if query.ndim != 1 or references.ndim != 2:
            raise ValueError("x must be one row and X_ref must be a matrix")
        if references.shape[1] != query.shape[0]:
            raise ValueError("x and X_ref must have the same feature dimension")
        return np.asarray(np.exp(self.log_weights(query, references)), dtype=np.float64)

    def log_weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate radial log masses without exponential underflow."""
        query = np.asarray(x, dtype=np.float64)
        references = np.asarray(X_ref, dtype=np.float64)
        if query.ndim != 1 or references.ndim != 2:
            raise ValueError("x must be one row and X_ref must be a matrix")
        if references.shape[1] != query.shape[0]:
            raise ValueError("x and X_ref must have the same feature dimension")
        query_coordinates = self.kernel_coordinates(query[None, :])[0]
        reference_coordinates = self._reference_coordinates(references)
        if math.isinf(self._bandwidth):
            return np.zeros(references.shape[0], dtype=np.float64)
        squared = np.sum(
            (reference_coordinates - query_coordinates) ** 2,
            axis=1,
        )
        return self._log_profile(np.asarray(squared, dtype=np.float64))

    def _log_profile(self, squared: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a subclass log radial profile from squared distances."""
        raise NotImplementedError


class GaussianKernelLocalizer(_StandardizedKernelLocalizer):
    """Gaussian geometry ``exp(-||z-z'||^2 / (2 h^2))``."""

    def _log_profile(self, squared: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(-squared / (2.0 * self._bandwidth**2), dtype=np.float64)

    def sample_anchor(
        self,
        x: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw one RLCP anchor isotropically in frozen kernel coordinates."""
        query = np.asarray(x, dtype=np.float64)
        if query.ndim != 1:
            raise ValueError(f"x must be one feature row, got {query.shape}")
        query_coordinates = self.kernel_coordinates(query[None, :])
        if math.isinf(self._bandwidth):
            return query.copy()
        anchor_coordinates = query_coordinates + self._bandwidth * rng.standard_normal(
            query_coordinates.shape
        )
        if self._scaler is None:
            return np.asarray(anchor_coordinates[0], dtype=np.float64)
        return np.asarray(
            self._scaler.inverse_transform(anchor_coordinates)[0],
            dtype=np.float64,
        )


class LaplaceKernelLocalizer(_StandardizedKernelLocalizer):
    """Laplace kernel ``exp(-RMS-distance / h)``, used by LCP."""

    def _log_profile(self, squared: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._n_features is None:
            raise RuntimeError("localizer must be fitted before weights")
        root_mean_squared = np.sqrt(squared / self._n_features)
        return np.asarray(-root_mean_squared / self._bandwidth, dtype=np.float64)

"""Frozen evaluation partitions for conditional-coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class FrozenKMeansPartition:
    """A standardized k-means partition fitted once on training features."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    centers: NDArray[np.float64]

    def assign(self, X: NDArray[np.floating]) -> NDArray[np.int64]:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.centers.shape[1]:
            raise ValueError("X has the wrong feature dimension for this partition")
        standardized = (values - self.mean) / self.scale
        distances = np.sum(
            (standardized[:, None, :] - self.centers[None, :, :]) ** 2,
            axis=2,
        )
        return np.asarray(np.argmin(distances, axis=1), dtype=np.int64)


def fit_frozen_kmeans_partition(
    X_train: NDArray[np.floating],
    *,
    n_groups: int,
    seed: int,
) -> FrozenKMeansPartition:
    """Fit the external partition on training features only."""
    values = np.asarray(X_train, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("X_train must be a non-empty feature matrix")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    scaler = StandardScaler().fit(values)
    model = KMeans(
        n_clusters=min(n_groups, len(values)),
        n_init=10,
        random_state=seed,
    ).fit(scaler.transform(values))
    return FrozenKMeansPartition(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        centers=np.asarray(model.cluster_centers_, dtype=np.float64),
    )


__all__ = ("FrozenKMeansPartition", "fit_frozen_kmeans_partition")

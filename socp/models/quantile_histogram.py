"""Training-only quantile-forest conditional histograms for CHR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

CHR_GRID_POINTS = 1_000
CHR_INTERVALS = CHR_GRID_POINTS - 1
CHR_QUANTILE_LEVELS = np.arange(0.01, 1.0, 0.01, dtype=np.float64)
_PROBABILITY_FLOOR = 1e-6
_JITTER_SCALE = 1e-5


def _weighted_percentiles(
    values: NDArray[np.float32],
    weights: NDArray[np.float32],
    levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return weighted percentiles, each value placed at the midpoint of its weight."""
    order = np.argsort(values)
    ordered_values = np.asarray(values[order], dtype=np.float64)
    ordered_weights = np.asarray(weights[order], dtype=np.float64)
    positive = ordered_weights > 0.0
    ordered_values = ordered_values[positive]
    ordered_weights = ordered_weights[positive]
    if ordered_values.size == 0:
        raise RuntimeError("a quantile-forest query received no positive leaf weight")
    cumulative = np.cumsum(ordered_weights)
    locations = 100.0 * (cumulative - ordered_weights / 2.0) / cumulative[-1]
    return np.asarray(
        np.interp(
            100.0 * levels,
            locations,
            ordered_values,
            left=float(ordered_values[0]),
            right=float(ordered_values[-1]),
        ),
        dtype=np.float64,
    )


def _histograms_from_quantiles(
    predicted: NDArray[np.float64],
    edges: NDArray[np.float64],
    levels: NDArray[np.float64],
    *,
    probability_floor: float,
    jitter_scale: float,
) -> NDArray[np.float64]:
    """Interpolate monotone conditional quantiles onto a valid response grid."""
    expected = (predicted.shape[0], levels.size)
    if predicted.ndim != 2 or predicted.shape != expected:
        raise ValueError(f"quantiles have shape {predicted.shape}, expected {expected}")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted quantiles must be finite")
    if np.any(np.diff(predicted, axis=1) < -1e-10):
        raise ValueError("predicted quantiles must be non-decreasing")

    percentiles = np.concatenate(([0.0], levels, [1.0]))
    jitter = (
        np.arange(1, percentiles.size + 1, dtype=np.float64)
        * jitter_scale
        / (percentiles.size + 1)
    )
    probabilities = np.empty((predicted.shape[0], edges.size - 1), dtype=np.float64)
    for row, quantiles in enumerate(predicted):
        clipped = np.clip(quantiles, edges[0], edges[-1])
        support = np.concatenate(([edges[0]], clipped, [edges[-1]])) + jitter
        cdf = np.interp(edges, support, percentiles, left=0.0, right=1.0)
        masses = np.maximum(np.diff(cdf), 0.0) + probability_floor
        probabilities[row] = masses / masses.sum()
    return probabilities


@dataclass(frozen=True, slots=True)
class _LeafDistribution:
    responses: NDArray[np.float32]
    weights: NDArray[np.float32]


class QuantileForestHistogram:
    """Conditional histogram for CHR from quantile-forest predictions.

    The defaults use 100 trees and ``min_samples_leaf=50``. The 1,000 grid points span
    the training responses, which fixes the 999 bins before calibration. Small fixed
    increasing offsets keep the quantile support strictly increasing.
    """

    def __init__(
        self,
        *,
        n_bins: int = CHR_INTERVALS,
        n_estimators: int = 100,
        min_samples_leaf: int = 50,
        random_state: int = 0,
        probability_floor: float = _PROBABILITY_FLOOR,
    ) -> None:
        if isinstance(n_bins, bool) or n_bins < 2:
            raise ValueError(f"n_bins must be at least two, got {n_bins}")
        if isinstance(n_estimators, bool) or n_estimators < 1:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if isinstance(min_samples_leaf, bool) or min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be positive, got {min_samples_leaf}"
            )
        if probability_floor < 0.0 or not np.isfinite(probability_floor):
            raise ValueError(
                f"probability_floor must be finite and non-negative, got {probability_floor}"
            )
        self._n_bins = int(n_bins)
        self._probability_floor = float(probability_floor)
        self._forest = RandomForestRegressor(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            random_state=random_state,
            n_jobs=1,
        )
        self._edges: NDArray[np.float64] | None = None
        self._leaf_distributions: list[dict[int, _LeafDistribution]] | None = None
        self._n_features: int | None = None

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Immutable training-response grid edges."""
        if self._edges is None:
            raise RuntimeError("QuantileForestHistogram.fit must be called first")
        return self._edges

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> QuantileForestHistogram:
        """Fit the forest and cache bootstrap-aware leaf distributions."""
        features = np.asarray(X, dtype=np.float64)
        responses = np.asarray(y, dtype=np.float64)
        if (
            features.ndim != 2
            or responses.ndim != 1
            or features.shape[0] != responses.shape[0]
        ):
            raise ValueError(
                "X and y must have aligned shapes (n, d) and (n,), got "
                f"{features.shape} and {responses.shape}"
            )
        if features.size == 0 or features.shape[1] == 0:
            raise ValueError("X and y must not be empty")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(responses)):
            raise ValueError("X and y must contain only finite values")
        y_min, y_max = float(np.min(responses)), float(np.max(responses))
        if not y_min < y_max:
            raise ValueError("responses must contain at least two distinct values")

        features32 = np.asarray(features, dtype=np.float32)
        responses32 = np.asarray(responses, dtype=np.float32)
        edges = np.linspace(y_min, y_max, self._n_bins + 1, dtype=np.float64)
        edges.setflags(write=False)
        self._edges = edges
        self._n_features = features.shape[1]
        self._forest.fit(features32, responses32)
        leaves = np.asarray(self._forest.apply(features32), dtype=np.int64)
        if leaves.ndim == 1:
            leaves = leaves[:, None]

        cached: list[dict[int, _LeafDistribution]] = []
        n_rows = responses.shape[0]
        for tree_index, estimator in enumerate(self._forest.estimators_):
            bootstrap = np.random.RandomState(estimator.random_state).randint(
                0,
                n_rows,
                n_rows,
            )
            counts = np.bincount(bootstrap, minlength=n_rows).astype(np.float64)
            by_leaf: dict[int, _LeafDistribution] = {}
            for leaf in np.unique(leaves[:, tree_index]):
                members = (leaves[:, tree_index] == leaf) & (counts > 0.0)
                leaf_counts = counts[members]
                by_leaf[int(leaf)] = _LeafDistribution(
                    responses=np.asarray(responses32[members], dtype=np.float32),
                    weights=np.asarray(
                        leaf_counts / leaf_counts.sum(),
                        dtype=np.float32,
                    ),
                )
            cached.append(by_leaf)
        self._leaf_distributions = cached
        return self

    def conditional_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict conditional quantiles at levels 0.01 through 0.99."""
        if self._leaf_distributions is None or self._n_features is None:
            raise RuntimeError("QuantileForestHistogram.fit must be called first")
        features = np.asarray(X, dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[1] != self._n_features
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(
                f"X must be a finite matrix with {self._n_features} columns, "
                f"got {features.shape}"
            )
        leaves = np.asarray(
            self._forest.apply(np.asarray(features, dtype=np.float32)),
            dtype=np.int64,
        )
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        result = np.empty(
            (features.shape[0], CHR_QUANTILE_LEVELS.size),
            dtype=np.float64,
        )
        for row in range(features.shape[0]):
            pieces = [
                by_leaf[int(leaves[row, tree])]
                for tree, by_leaf in enumerate(self._leaf_distributions)
            ]
            values = np.concatenate([piece.responses for piece in pieces])
            weights = np.concatenate([piece.weights for piece in pieces])
            result[row] = _weighted_percentiles(
                values,
                weights,
                CHR_QUANTILE_LEVELS,
            )
        return result

    def bin_probabilities(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return positive, unit-sum conditional masses on the frozen support."""
        return _histograms_from_quantiles(
            self.conditional_quantiles(X),
            self.bin_edges,
            CHR_QUANTILE_LEVELS,
            probability_floor=self._probability_floor,
            jitter_scale=_JITTER_SCALE,
        )

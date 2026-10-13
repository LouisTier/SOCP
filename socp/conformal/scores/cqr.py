"""Conformalized quantile-regression score and exact inverse set."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import Cutoff, ScoreSpec
from socp.conformal.scores._regression import _scalar_cutoff
from socp.conformal.sets import Interval, IntervalUnion, PredictionSet

_SPEC = ScoreSpec(name="cqr", task="regression")


class CQRScore:
    """CQR score for lower and upper quantile predictors frozen before calibration.

    The two quantile predictions of each row are sorted before use.

    Args:
        quantile_fn: Fitted function returning two quantile predictions per row.
    """

    def __init__(
        self,
        quantile_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        self.spec = _SPEC
        self._quantile_fn = quantile_fn
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the fixed score has entered its post-training lifecycle."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Freeze the already-fitted quantile predictor."""
        self._is_fitted = True

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return ``max(q_lo(X) - y, y - q_hi(X))``."""
        self._require_fitted()
        lower, upper = self._bounds(X)
        return np.asarray(np.maximum(lower - y, y - upper), dtype=np.float64)

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return ``[q_lo-cutoff, q_hi+cutoff]`` or its exact degenerate set."""
        self._require_fitted()
        threshold = _scalar_cutoff(cutoff)
        if threshold == math.inf:
            return Interval.unbounded()
        if threshold == -math.inf:
            return IntervalUnion(())
        lower, upper = self._bounds(x[None, :])
        set_lower = float(lower[0] - threshold)
        set_upper = float(upper[0] + threshold)
        if set_lower > set_upper:
            return IntervalUnion(())
        return Interval(set_lower, set_upper)

    def _bounds(
        self,
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        quantiles = np.asarray(self._quantile_fn(X), dtype=np.float64)
        expected = (X.shape[0], 2)
        if quantiles.shape != expected:
            raise ValueError(
                f"quantile predictor returned shape {quantiles.shape}, "
                f"expected {expected}"
            )
        if not np.all(np.isfinite(quantiles)):
            raise ValueError("quantile predictor must return only finite values")
        return (
            np.minimum(quantiles[:, 0], quantiles[:, 1]),
            np.maximum(quantiles[:, 0], quantiles[:, 1]),
        )

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("cqr must be fit before use")

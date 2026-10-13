"""Absolute-residual nonconformity and its exact inverse set."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import Cutoff, ScoreSpec
from socp.conformal.scores._regression import (
    _finite_prediction_vector,
    _scalar_cutoff,
)
from socp.conformal.sets import Interval, IntervalUnion, PredictionSet

_SPEC = ScoreSpec(name="abs_residual", task="regression")


class AbsoluteResidualScore:
    """Absolute residual for a predictor frozen before calibration.

    Args:
        predict_fn: Fitted point-prediction function. This score never refits it.
    """

    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        self.spec = _SPEC
        self._predict_fn = predict_fn
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the fixed score has entered its post-training lifecycle."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Freeze the already-fitted predictor without reading training outcomes."""
        self._is_fitted = True

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return ``abs(y - predict_fn(X))``."""
        self._require_fitted()
        predictions = _finite_prediction_vector(
            self._predict_fn(X),
            n_rows=X.shape[0],
            owner="predictor",
        )
        return np.asarray(np.abs(y - predictions), dtype=np.float64)

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return the exact sublevel set at ``cutoff``."""
        self._require_fitted()
        threshold = _scalar_cutoff(cutoff)
        if threshold == math.inf:
            return Interval.unbounded()
        if threshold < 0.0:
            return IntervalUnion(())
        centre = float(
            _finite_prediction_vector(
                self._predict_fn(x[None, :]),
                n_rows=1,
                owner="predictor",
            )[0]
        )
        return Interval(centre - threshold, centre + threshold)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("abs_residual must be fit before use")

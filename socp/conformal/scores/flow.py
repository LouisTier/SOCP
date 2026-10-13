"""NFCP score and its exact inverse set.

The score is ``(|y - f(x)| + residual_floor) / (gamma + |g(x)|)``, with the point
predictor ``f`` and the scale ``|g(x)|`` fitted before calibration.
"""

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

_SPEC = ScoreSpec(name="nf_cp", task="regression")


def _flow_ratio_scores(
    residuals: NDArray[np.float64],
    scales: NDArray[np.float64],
    *,
    gamma: float,
    residual_floor: float,
) -> NDArray[np.float64]:
    """Return the NFCP ratio scores."""
    return np.asarray(
        (residuals + residual_floor) / (gamma + scales),
        dtype=np.float64,
    )


def _flow_residual_cutoffs(
    ratio_cutoffs: NDArray[np.float64],
    scales: NDArray[np.float64],
    *,
    gamma: float,
    residual_floor: float,
) -> NDArray[np.float64]:
    """Transport positive-ratio cutoffs back to absolute-residual units."""
    transported_with_floor = np.asarray(
        ratio_cutoffs * (gamma + scales),
        dtype=np.float64,
    )
    closed_upper_endpoint = np.nextafter(
        np.nextafter(transported_with_floor, math.inf),
        math.inf,
    )
    return np.asarray(
        closed_upper_endpoint - residual_floor,
        dtype=np.float64,
    )


class NormalizingFlowScore:
    """NFCP score with a scale fitted before calibration.

    Args:
        predict_fn: Fitted point predictor ``f(x)``.
        scale_fn: Fitted non-negative network output ``|g(x)|``. The caller must
            express it in the same response units as the residual.
        gamma: Strictly positive denominator regularizer, in response units.
        residual_floor: Strictly positive constant added to the residual, in response
            units.

    Raises:
        ValueError: If either numerical constant is not finite and positive.
    """

    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        scale_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        gamma: float = 1e-3,
        residual_floor: float = 1e-4,
    ) -> None:
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(f"gamma must be finite and positive, got {gamma}")
        if not math.isfinite(residual_floor) or residual_floor <= 0.0:
            raise ValueError(
                f"residual_floor must be finite and positive, got {residual_floor}"
            )
        self.spec = _SPEC
        self._predict_fn = predict_fn
        self._scale_fn = scale_fn
        self._gamma = gamma
        self._residual_floor = residual_floor
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the fixed score has entered its post-training lifecycle."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Freeze the already-fitted center and scale functions."""
        self._is_fitted = True

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return ``(abs(y-f(X)) + floor) / (gamma + abs(g(X)))``."""
        self._require_fitted()
        predictions = _finite_prediction_vector(
            self._predict_fn(X),
            n_rows=X.shape[0],
            owner="flow point predictor",
        )
        scales = self._scales(X)
        residuals = np.asarray(np.abs(y - predictions), dtype=np.float64)
        return _flow_ratio_scores(
            residuals,
            scales,
            gamma=self._gamma,
            residual_floor=self._residual_floor,
        )

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return the exact response sublevel set at a ratio cutoff."""
        self._require_fitted()
        threshold = _scalar_cutoff(cutoff)
        if threshold == math.inf:
            return Interval.unbounded()
        if threshold == -math.inf:
            return IntervalUnion(())

        row = x[None, :]
        centre = float(
            _finite_prediction_vector(
                self._predict_fn(row),
                n_rows=1,
                owner="flow point predictor",
            )[0]
        )
        scale = self._scales(row)
        residual_cutoff = float(
            _flow_residual_cutoffs(
                np.array([threshold], dtype=np.float64),
                scale,
                gamma=self._gamma,
                residual_floor=self._residual_floor,
            )[0]
        )
        if residual_cutoff < 0.0:
            return IntervalUnion(())
        if residual_cutoff == math.inf:
            return Interval.unbounded()
        return Interval(centre - residual_cutoff, centre + residual_cutoff)

    def _scales(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        scales = _finite_prediction_vector(
            self._scale_fn(X),
            n_rows=X.shape[0],
            owner="flow scale",
        )
        if np.any(scales < 0.0):
            raise ValueError("flow scale must return non-negative values")
        denominator = scales + self._gamma
        if not np.all(np.isfinite(denominator)):
            raise ValueError("flow denominator must remain finite")
        return scales

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("nf_cp must be fit before use")

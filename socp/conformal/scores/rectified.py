"""Rectified conformal prediction (RCP) score.

A conditional quantile ``tau(x)`` of the base score ``V`` rectifies it as
``V(x, y) / tau(x)`` or ``V(x, y) - tau(x)``. Inversion maps the cutoff back to
base-score units and inverts the base score.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import (
    Cutoff,
    Score,
    ScoreSpec,
    evaluate_score,
    fit_score,
    invert_score,
)
from socp.conformal.scores._regression import (
    _finite_prediction_vector,
    _scalar_cutoff,
)
from socp.conformal.sets import PredictionSet

Adjustment = Literal["linear", "difference"]
_SPEC = ScoreSpec(name="rcp", task="regression")


def _rectify_scores(
    base_scores: NDArray[np.float64],
    tau: NDArray[np.float64],
    *,
    adjustment: Adjustment,
) -> NDArray[np.float64]:
    """Return ``V / tau`` for ``linear`` or ``V - tau`` for ``difference``."""
    if adjustment == "linear":
        return np.asarray(base_scores / tau, dtype=np.float64)
    return np.asarray(base_scores - tau, dtype=np.float64)


def _transport_cutoffs(
    cutoffs: NDArray[np.float64],
    tau: NDArray[np.float64],
    *,
    adjustment: Adjustment,
) -> NDArray[np.float64]:
    """Map rectified cutoffs back to base-score units."""
    if adjustment == "linear":
        return np.asarray(cutoffs * tau, dtype=np.float64)
    return np.asarray(cutoffs + tau, dtype=np.float64)


class RectifiedScore:
    """RCP score from a base score and its conditional-quantile estimate.

    The experiments use the default ``linear`` adjustment, ``V / tau``.

    Args:
        base: Base nonconformity score ``V``.
        tau_fn: Conditional-quantile estimate fitted before calibration. It returns
            one finite value per row and must be strictly positive for ``linear``.
        adjustment: ``"linear"`` for ``V/tau`` or ``"difference"`` for
            ``V-tau``.

    Raises:
        ValueError: If ``adjustment`` is unsupported.
    """

    def __init__(
        self,
        base: Score,
        tau_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        adjustment: Adjustment = "linear",
    ) -> None:
        if adjustment not in ("linear", "difference"):
            raise ValueError(
                f"adjustment must be 'linear' or 'difference', got {adjustment!r}"
            )
        if base.spec.task != "regression":
            raise ValueError("RCP currently requires a regression base score")
        self.spec = _SPEC
        self._base = base
        self._tau_fn = tau_fn
        self._adjustment: Adjustment = adjustment
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the base and fixed rectifier are ready for calibration."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit only an unfitted base; ``tau_fn`` is already fitted."""
        if not self._base.is_fitted:
            fit_score(self._base, X, y)
        self._is_fitted = True

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the multiplicatively or additively rectified base score."""
        self._require_fitted()
        base_scores = evaluate_score(self._base, X, y)
        tau = self._tau(X)
        return _rectify_scores(
            base_scores,
            tau,
            adjustment=self._adjustment,
        )

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Transport ``cutoff`` to base-score units and delegate inversion."""
        self._require_fitted()
        threshold = _scalar_cutoff(cutoff)
        if threshold == math.inf:
            return invert_score(self._base, x, math.inf)
        tau = self._tau(x[None, :])
        transported = float(
            _transport_cutoffs(
                np.array([threshold], dtype=np.float64),
                tau,
                adjustment=self._adjustment,
            )[0]
        )
        rounding_guard = (
            8.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(threshold),
                abs(float(tau[0])),
                abs(transported),
            )
        )
        transported = float(transported + rounding_guard)
        return invert_score(self._base, x, transported)

    def _tau(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        tau = _finite_prediction_vector(
            self._tau_fn(X),
            n_rows=X.shape[0],
            owner="RCP conditional quantile",
        )
        if self._adjustment == "linear" and np.any(tau <= 0.0):
            raise ValueError(
                "RCP conditional quantiles must be strictly positive for "
                "the linear adjustment"
            )
        return tau

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("rcp must be fit before use")

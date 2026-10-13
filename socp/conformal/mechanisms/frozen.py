"""Pooled and fixed-partition conformal calibration mechanisms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from socp.conformal.core import split_quantile
from socp.conformal.localizer import Localizer
from socp.conformal.mechanism import CalibrationBatch, MechanismSpec
from socp.conformal.score import Task
from socp.ranks import validate_alpha


class PooledCalibration:
    """Ordinary split-conformal calibration over the full calibration split."""

    def __init__(self, *, alpha: float, task: Task) -> None:
        validate_alpha(alpha)
        self.spec = MechanismSpec(name="pooled", task=task)
        self._alpha = float(alpha)
        self._cutoff: float | None = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Whether the immutable pooled cutoff has been computed."""
        return self._is_calibrated

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Compute the shared exact split-conformal quantile once."""
        if self._is_calibrated:
            raise RuntimeError("mechanism is already calibrated")
        if batch.task != self.spec.task:
            raise ValueError("calibration task does not match mechanism task")
        self._cutoff = split_quantile(batch.scores, self._alpha)
        self._is_calibrated = True

    def threshold(self, x: NDArray[np.float64]) -> float:
        """Return the query-independent pooled cutoff."""
        del x
        if not self._is_calibrated or self._cutoff is None:
            raise RuntimeError("mechanism must be calibrated before threshold")
        return self._cutoff


class MondrianCalibration:
    """Split conformal within a frozen hard partition of feature space."""

    def __init__(self, localizer: Localizer, *, alpha: float, task: Task) -> None:
        validate_alpha(alpha)
        if localizer.spec.kernel != "indicator":
            raise ValueError("Mondrian calibration requires an indicator localizer")
        if not callable(getattr(localizer, "assign", None)):
            raise TypeError("Mondrian calibration requires localizer.assign")
        self.spec = MechanismSpec(name="mondrian", task=task)
        self._localizer = localizer
        self._alpha = float(alpha)
        self._cutoffs: dict[int, float] = {}
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Whether all observed partition cells have frozen cutoffs."""
        return self._is_calibrated

    @property
    def localizer(self) -> Localizer:
        """Return the exact frozen partition geometry consumed by this mechanism."""
        return self._localizer

    def _assign(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return validated integer cell identifiers."""
        assign = cast(
            Callable[[NDArray[np.float64]], NDArray[np.int64]],
            getattr(self._localizer, "assign"),
        )
        raw = np.asarray(assign(X))
        if raw.shape != (X.shape[0],):
            raise ValueError(
                f"localizer.assign returned shape {raw.shape}, expected {(X.shape[0],)}"
            )
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("partition labels must be finite integers") from error
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("partition labels must be finite integers")
        return np.asarray(numeric, dtype=np.int64)

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Freeze one exact split quantile per observed cell."""
        if self._is_calibrated:
            raise RuntimeError("mechanism is already calibrated")
        if batch.task != self.spec.task:
            raise ValueError("calibration task does not match mechanism task")
        if not self._localizer.is_fitted:
            raise RuntimeError("localizer must be fitted before calibration")
        cells = self._assign(batch.features)
        self._cutoffs = {
            int(cell): split_quantile(batch.scores[cells == cell], self._alpha)
            for cell in np.unique(cells)
        }
        self._is_calibrated = True

    def threshold(self, x: NDArray[np.float64]) -> float:
        """Return the query cell's cutoff, conservatively infinite if unseen."""
        if not self._is_calibrated:
            raise RuntimeError("mechanism must be calibrated before threshold")
        cell = int(self._assign(np.asarray(x, dtype=np.float64)[None, :])[0])
        return self._cutoffs.get(cell, math.inf)

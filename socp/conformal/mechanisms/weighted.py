"""Direct weighted, corrected LCP, and RLCP calibration mechanisms."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from socp.conformal.core import weighted_quantile
from socp.conformal.localized import (
    LocalizedCache,
    localized_precompute,
    localized_threshold,
)
from socp.conformal.localizer import Localizer, localizer_weights
from socp.conformal.mechanism import CalibrationBatch, MechanismSpec
from socp.conformal.score import Task
from socp.ranks import validate_alpha


class _WeightedMechanism:
    """Shared lifecycle for mechanisms that retain calibration rows and scores."""

    def __init__(
        self,
        localizer: Localizer,
        *,
        alpha: float,
        task: Task,
        name: str,
    ) -> None:
        validate_alpha(alpha)
        self.spec = MechanismSpec(name=name, task=task)
        self._localizer = localizer
        self._alpha = float(alpha)
        self._calibration_features: NDArray[np.float64] | None = None
        self._calibration_scores: NDArray[np.float64] | None = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Whether immutable calibration data have been stored."""
        return self._is_calibrated

    @property
    def localizer(self) -> Localizer:
        """Return the exact frozen geometry consumed by this mechanism."""
        return self._localizer

    def _check_calibration(self, batch: CalibrationBatch) -> None:
        """Enforce the one-shot, frozen-geometry calibration boundary."""
        if self._is_calibrated:
            raise RuntimeError("mechanism is already calibrated")
        if batch.task != self.spec.task:
            raise ValueError("calibration task does not match mechanism task")
        if not self._localizer.is_fitted:
            raise RuntimeError("localizer must be fitted before calibration")

    def _store_calibration(self, batch: CalibrationBatch) -> None:
        """Retain detached immutable calibration data."""
        features = np.asarray(batch.features, dtype=np.float64).copy()
        scores = np.asarray(batch.scores, dtype=np.float64).copy()
        features.setflags(write=False)
        scores.setflags(write=False)
        self._calibration_features = features
        self._calibration_scores = scores
        self._is_calibrated = True

    def _calibration(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return frozen calibration data after checking lifecycle state."""
        if (
            not self._is_calibrated
            or self._calibration_features is None
            or self._calibration_scores is None
        ):
            raise RuntimeError("mechanism must be calibrated before threshold")
        return self._calibration_features, self._calibration_scores

    def _self_weight(self, x: NDArray[np.float64]) -> float:
        """Return the query's localizer mass on itself."""
        return float(localizer_weights(self._localizer, x, x[None, :])[0])


class WeightedCalibration(_WeightedMechanism):
    """Weighted split conformal centred on the query, without the LCP correction."""

    def __init__(self, localizer: Localizer, *, alpha: float, task: Task) -> None:
        super().__init__(
            localizer,
            alpha=alpha,
            task=task,
            name="direct_weighted",
        )

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Freeze calibration features and scores without fitting geometry."""
        self._check_calibration(batch)
        self._store_calibration(batch)

    def threshold(self, x: NDArray[np.float64]) -> float:
        """Return the shared weighted quantile with query mass at infinity."""
        features, scores = self._calibration()
        weights = localizer_weights(self._localizer, x, features)
        return weighted_quantile(scores, weights, self._self_weight(x), self._alpha)


class GuanLCP(_WeightedMechanism):
    """Localized conformal prediction (LCP) with the corrected level sweep."""

    def __init__(self, localizer: Localizer, *, alpha: float, task: Task) -> None:
        if localizer.spec.kernel != "smooth":
            raise ValueError("Guan LCP requires a smooth localizer")
        super().__init__(localizer, alpha=alpha, task=task, name="guan_lcp")
        self._cache: LocalizedCache | None = None
        self._sorted_features: NDArray[np.float64] | None = None

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Precompute calibration-point localized CDF quantities once."""
        self._check_calibration(batch)
        order = np.argsort(batch.scores, kind="stable")
        sorted_features = np.asarray(batch.features[order], dtype=np.float64).copy()
        sorted_features.setflags(write=False)
        self._sorted_features = sorted_features
        self._cache = localized_precompute(
            batch.scores,
            lambda row: localizer_weights(
                self._localizer,
                sorted_features[row],
                sorted_features,
            ),
        )
        self._store_calibration(batch)

    def threshold(self, x: NDArray[np.float64]) -> float:
        """Return the last corrected-level crossing for one query."""
        self._calibration()
        if self._cache is None or self._sorted_features is None:
            raise RuntimeError("Guan cache is unavailable before calibration")
        weights = localizer_weights(self._localizer, x, self._sorted_features)
        return localized_threshold(
            self._cache,
            weights,
            self._self_weight(x),
            self._alpha,
        )


class RLCP(_WeightedMechanism):
    """Randomly localized conformal prediction with one seeded anchor per query."""

    def __init__(
        self,
        localizer: Localizer,
        *,
        alpha: float,
        task: Task,
        rng: np.random.Generator,
    ) -> None:
        if localizer.spec.kernel != "smooth":
            raise ValueError("RLCP requires a smooth localizer")
        if not callable(getattr(localizer, "sample_anchor", None)):
            raise TypeError("RLCP requires localizer.sample_anchor")
        if not callable(getattr(localizer, "log_weights", None)):
            raise TypeError("RLCP requires underflow-safe localizer.log_weights")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        super().__init__(localizer, alpha=alpha, task=task, name="rlcp")
        self._rng = rng
        self._last_anchor: NDArray[np.float64] | None = None

    @property
    def last_anchor(self) -> NDArray[np.float64]:
        """Return the immutable most recently sampled anchor."""
        if self._last_anchor is None:
            raise RuntimeError("no RLCP anchor has been sampled")
        return self._last_anchor

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Freeze calibration features and scores without consuming randomness."""
        self._check_calibration(batch)
        self._store_calibration(batch)

    def threshold(self, x: NDArray[np.float64]) -> float:
        """Draw one anchor and return the quantile weighted around that anchor."""
        features, scores = self._calibration()
        sample_anchor = cast(
            Callable[
                [NDArray[np.float64], np.random.Generator],
                NDArray[np.float64],
            ],
            getattr(self._localizer, "sample_anchor"),
        )
        anchor = np.asarray(sample_anchor(x, self._rng), dtype=np.float64)
        if anchor.shape != x.shape:
            raise ValueError(
                f"localizer anchor has shape {anchor.shape}, expected {x.shape}"
            )
        if not np.all(np.isfinite(anchor)):
            raise ValueError("localizer anchor must contain only finite values")
        immutable_anchor = anchor.copy()
        immutable_anchor.setflags(write=False)
        self._last_anchor = immutable_anchor

        log_weights = cast(
            Callable[
                [NDArray[np.float64], NDArray[np.float64]],
                NDArray[np.float64],
            ],
            getattr(self._localizer, "log_weights"),
        )
        calibration_logs = np.asarray(
            log_weights(immutable_anchor, features),
            dtype=np.float64,
        )
        self_logs = np.asarray(
            log_weights(
                immutable_anchor,
                np.asarray(x, dtype=np.float64)[None, :],
            ),
            dtype=np.float64,
        )
        if calibration_logs.shape != (features.shape[0],) or self_logs.shape != (1,):
            raise ValueError("localizer.log_weights returned an invalid shape")
        if np.any(np.isnan(calibration_logs)) or np.any(np.isnan(self_logs)):
            raise ValueError("localizer.log_weights must not return NaN")
        shift = max(float(np.max(calibration_logs)), float(self_logs[0]))
        if not np.isfinite(shift):
            raise ValueError("RLCP log masses must contain at least one finite value")
        weights = np.exp(calibration_logs - shift)
        self_weight = float(np.exp(self_logs[0] - shift))
        return weighted_quantile(scores, weights, self_weight, self._alpha)

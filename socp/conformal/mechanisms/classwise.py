"""Label-conditional conformal calibration with one cutoff per class."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from socp.conformal._validation import immutable_finite_array
from socp.conformal.core import split_quantile
from socp.conformal.mechanism import CalibrationBatch, MechanismSpec
from socp.ranks import validate_alpha


def _class_count(n_classes: int) -> int:
    """Validate and return a positive candidate-label count."""
    if isinstance(n_classes, bool) or int(n_classes) != n_classes or n_classes <= 0:
        raise ValueError(f"n_classes must be a positive integer, got {n_classes}")
    return int(n_classes)


def _label_vector(
    values: NDArray[np.float64],
    *,
    n_classes: int,
    name: str,
    allow_empty: bool = False,
) -> NDArray[np.intp]:
    """Return validated immutable zero-based class labels."""
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or (numeric.size == 0 and not allow_empty):
        qualifier = "vector" if allow_empty else "non-empty vector"
        raise ValueError(f"{name} must be a {qualifier}, got {numeric.shape}")
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain finite integer labels")
    if np.any(numeric < 0.0) or np.any(numeric >= n_classes):
        raise ValueError(f"{name} must lie in [0, {n_classes})")
    labels = np.asarray(numeric, dtype=np.intp).copy()
    labels.setflags(write=False)
    return labels


def _routing_labels(batch: CalibrationBatch, *, n_classes: int) -> NDArray[np.intp]:
    """Extract required label routing from a calibration batch."""
    if batch.routing is None:
        raise ValueError("class-conditional calibration requires routing labels")
    return _label_vector(
        batch.routing,
        n_classes=n_classes,
        name="calibration labels",
    )


class ClasswiseCalibration:
    """Compute a separate split-conformal threshold for each label.

    A label with no calibration scores, or too few for a finite threshold, gets
    ``+inf``.
    """

    def __init__(self, *, alpha: float, n_classes: int) -> None:
        validate_alpha(alpha)
        self.spec = MechanismSpec(
            name="classwise",
            task="classification",
            requires_routing=True,
            routing_source="outcome",
        )
        self._alpha = float(alpha)
        self._n_classes = _class_count(n_classes)
        self._cutoffs: NDArray[np.float64] | None = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Whether label-indexed thresholds have been frozen."""
        return self._is_calibrated

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Compute one split-conformal quantile per calibration label."""
        if self._is_calibrated:
            raise RuntimeError("mechanism is already calibrated")
        if batch.task != "classification":
            raise ValueError("classwise calibration requires classification scores")
        labels = _routing_labels(batch, n_classes=self._n_classes)
        cutoffs = np.full(self._n_classes, math.inf, dtype=np.float64)
        counts = np.bincount(labels, minlength=self._n_classes).astype(np.intp)
        for label in range(self._n_classes):
            if counts[label] > 0:
                cutoffs[label] = split_quantile(
                    batch.scores[labels == label],
                    self._alpha,
                )
        cutoffs.setflags(write=False)
        self._cutoffs = cutoffs
        self._is_calibrated = True

    def threshold(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the query-independent vector of candidate-label cutoffs."""
        del x
        if not self._is_calibrated or self._cutoffs is None:
            raise RuntimeError("mechanism must be calibrated before threshold")
        return self._cutoffs


def _finite_scores(
    values: NDArray[np.float64],
    *,
    name: str,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Return an immutable non-empty finite score vector for taxonomy fitting."""
    return immutable_finite_array(
        values,
        name=name,
        ndim=1,
        allow_empty=allow_empty,
    )

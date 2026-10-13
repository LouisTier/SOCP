"""Private validation shared by regression-score implementations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import Cutoff


def _scalar_cutoff(cutoff: Cutoff) -> float:
    """Return a non-NaN scalar cutoff for a regression score."""
    if isinstance(cutoff, np.ndarray):
        raise TypeError("regression scores do not accept label-indexed cutoffs")
    value = float(cutoff)
    if math.isnan(value):
        raise ValueError("cutoff must not be NaN")
    return value


def _finite_prediction_vector(
    values: NDArray[np.float64],
    *,
    n_rows: int,
    owner: str,
) -> NDArray[np.float64]:
    """Validate a fitted predictor's one-dimensional output."""
    predictions = np.asarray(values, dtype=np.float64)
    expected = (n_rows,)
    if predictions.shape != expected:
        raise ValueError(
            f"{owner} returned shape {predictions.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{owner} must return only finite values")
    return predictions

"""Nonconformity score protocol.

A score is fitted before calibration, evaluated with ``__call__``, and inverted into
prediction sets with ``invert``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from socp.conformal._validation import (
    feature_matrix,
    feature_row,
    immutable_finite_array,
)
from socp.conformal.sets import PredictionSet

Task = Literal["regression", "classification"]
Cutoff = float | NDArray[np.float64]


def validate_task(task: str) -> None:
    """Validate a supported prediction task."""
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """Immutable metadata required to compose a score safely."""

    name: str
    task: Task
    randomized: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("score name must not be empty")
        validate_task(self.task)


@runtime_checkable
class Score(Protocol):
    """A fitted mapping from feature-label pairs to nonconformity values."""

    spec: ScoreSpec

    @property
    def is_fitted(self) -> bool:
        """Whether training has completed and the score is frozen."""
        ...

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit score-owned state on declared pre-calibration data."""
        ...

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one nonconformity value per feature-label pair."""
        ...

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return ``{y : score(x, y) <= cutoff}`` for one feature row."""
        ...


@runtime_checkable
class ObservationIdentifiedScore(Protocol):
    """Randomized score accepting immutable row identities for its next call."""

    def bind_observation_ids(self, row_ids: NDArray[np.integer]) -> None:
        """Bind row identities consumed by the next score or inversion call."""
        ...


def bind_score_observation_ids(
    score: Score,
    row_ids: NDArray[np.integer],
) -> None:
    """Bind row identities to a randomized score before its next call.

    Deterministic scores are left unchanged. A randomized score without
    observation-keyed draws raises ``TypeError``.

    Args:
        score: Fitted or unfitted score.
        row_ids: Identities aligned with the next score call.
    """
    if not score.spec.randomized:
        return
    if not isinstance(score, ObservationIdentifiedScore):
        raise TypeError(
            f"randomized score {score.spec.name!r} lacks observation-keyed draws"
        )
    score.bind_observation_ids(row_ids)


def _supervised_data(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    task: Task,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and detach aligned supervised data."""
    checked_X = feature_matrix(X)
    checked_y = immutable_finite_array(y, name="y", ndim=1)
    if checked_y.shape[0] != checked_X.shape[0]:
        raise ValueError(
            "X and y must contain the same number of rows, got "
            f"{checked_X.shape[0]} and {checked_y.shape[0]}"
        )
    if task == "classification":
        integer_labels = np.equal(checked_y, np.floor(checked_y))
        if np.any(checked_y < 0.0) or not np.all(integer_labels):
            raise ValueError("classification labels must be non-negative integers")
    return checked_X, checked_y


def _require_fitted(score: Score) -> None:
    if not score.is_fitted:
        raise RuntimeError(f"score {score.spec.name!r} must be fit before use")


def fit_score(
    score: Score,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> None:
    """Validate fit data, fit ``score``, and verify the lifecycle transition."""
    checked_X, checked_y = _supervised_data(X, y, task=score.spec.task)
    score.fit(checked_X, checked_y)
    if not score.is_fitted:
        raise RuntimeError(
            f"score {score.spec.name!r} did not enter fitted state after fit()"
        )


def evaluate_score(
    score: Score,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a fitted score with strict alignment and finiteness checks."""
    _require_fitted(score)
    checked_X, checked_y = _supervised_data(X, y, task=score.spec.task)
    raw_scores = np.asarray(score(checked_X, checked_y), dtype=np.float64)
    expected_shape = (checked_X.shape[0],)
    if raw_scores.shape != expected_shape:
        raise ValueError(
            f"score returned shape {raw_scores.shape}, expected {expected_shape}"
        )
    return immutable_finite_array(raw_scores, name="scores", ndim=1)


def validate_cutoff(cutoff: Cutoff, *, task: Task) -> Cutoff:
    """Validate a scalar cutoff or a classification-only label-indexed vector."""
    if isinstance(cutoff, np.ndarray):
        if task != "classification":
            raise TypeError("label-indexed cutoffs are valid only for classification")
        array = np.asarray(cutoff, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("a label-indexed cutoff must be a non-empty vector")
        if np.any(np.isnan(array)):
            raise ValueError("cutoffs must not contain NaN")
        immutable = array.copy()
        immutable.setflags(write=False)
        return immutable

    scalar = float(cutoff)
    if math.isnan(scalar):
        raise ValueError("cutoff must not be NaN")
    return scalar


def invert_score(
    score: Score,
    x: NDArray[np.float64],
    cutoff: Cutoff,
) -> PredictionSet:
    """Invert a fitted score after validating query and cutoff semantics."""
    _require_fitted(score)
    checked_x = feature_row(x)
    checked_cutoff = validate_cutoff(cutoff, task=score.spec.task)
    result = score.invert(checked_x, checked_cutoff)
    if not isinstance(result, PredictionSet):
        raise TypeError("score inversion must return a PredictionSet")
    return result

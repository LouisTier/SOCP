"""Calibration mechanism protocol.

A mechanism turns frozen calibration scores, with optional routing information, into
score cutoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from socp.conformal._validation import (
    feature_matrix,
    feature_row,
    immutable_finite_array,
    immutable_row_ids,
)
from socp.conformal.score import Cutoff, Task, validate_cutoff, validate_task

RoutingSource = Literal["none", "outcome"]


@dataclass(frozen=True, slots=True)
class MechanismSpec:
    """Immutable metadata controlling mechanism compatibility."""

    name: str
    task: Task
    requires_routing: bool = False
    routing_source: RoutingSource = "none"
    requires_auxiliary_split: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("mechanism name must not be empty")
        validate_task(self.task)
        if self.routing_source not in ("none", "outcome"):
            raise ValueError(f"unknown routing source {self.routing_source!r}")
        if self.routing_source == "outcome" and not self.requires_routing:
            raise ValueError("outcome routing requires requires_routing=True")
        if self.requires_auxiliary_split and self.routing_source != "outcome":
            raise ValueError("auxiliary split calibration requires outcome routing")


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    """Immutable aligned inputs available to a calibration mechanism."""

    task: Task
    features: NDArray[np.float64]
    scores: NDArray[np.float64]
    routing: NDArray[np.float64] | None = None
    row_ids: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        validate_task(self.task)
        checked_features = feature_matrix(self.features, name="features")
        checked_scores = immutable_finite_array(self.scores, name="scores", ndim=1)
        if checked_scores.shape[0] != checked_features.shape[0]:
            raise ValueError(
                "features and scores must have the same number of rows, got "
                f"{checked_features.shape[0]} and {checked_scores.shape[0]}"
            )

        checked_routing: NDArray[np.float64] | None = None
        if self.routing is not None:
            checked_routing = immutable_finite_array(
                self.routing,
                name="routing",
                ndim=1,
            )
            if checked_routing.shape[0] != checked_features.shape[0]:
                raise ValueError(
                    "routing must align with calibration rows, got "
                    f"{checked_routing.shape[0]} and {checked_features.shape[0]}"
                )

        object.__setattr__(self, "features", checked_features)
        object.__setattr__(self, "scores", checked_scores)
        object.__setattr__(self, "routing", checked_routing)
        checked_row_ids: NDArray[np.int64] | None = None
        if self.row_ids is not None:
            checked_row_ids = immutable_row_ids(
                self.row_ids,
                expected_rows=checked_features.shape[0],
            )
        object.__setattr__(self, "row_ids", checked_row_ids)


@runtime_checkable
class Mechanism(Protocol):
    """Turn frozen calibration scores into query-dependent score cutoffs."""

    spec: MechanismSpec

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has completed."""
        ...

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Consume immutable frozen scores and declared routing information."""
        ...

    def threshold(self, x: NDArray[np.float64]) -> Cutoff:
        """Return the score cutoff for one query row."""
        ...


def calibrate_mechanism(
    mechanism: Mechanism,
    batch: CalibrationBatch,
) -> None:
    """Validate compatibility, calibrate, and verify the lifecycle transition."""
    if mechanism.spec.task != batch.task:
        raise ValueError(
            f"mechanism task {mechanism.spec.task!r} does not match "
            f"calibration task {batch.task!r}"
        )
    if mechanism.spec.requires_routing and batch.routing is None:
        raise ValueError(f"mechanism {mechanism.spec.name!r} requires routing")

    mechanism.calibrate(batch)
    if not mechanism.is_calibrated:
        raise RuntimeError(
            f"mechanism {mechanism.spec.name!r} did not enter calibrated state"
        )


def mechanism_threshold(
    mechanism: Mechanism,
    x: NDArray[np.float64],
) -> Cutoff:
    """Return a validated cutoff from a calibrated mechanism."""
    if not mechanism.is_calibrated:
        raise RuntimeError(
            f"mechanism {mechanism.spec.name!r} must be calibrated before use"
        )
    checked_x = feature_row(x)
    return validate_cutoff(mechanism.threshold(checked_x), task=mechanism.spec.task)

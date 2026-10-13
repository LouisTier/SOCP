"""Effective-sample-size bandwidth rule for RLCP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

OFFICIAL_H_MIN: Final = 0.02
OFFICIAL_H_MAX: Final = 6.0
OFFICIAL_H_STEP: Final = 0.02
ESS_ESTIMATION_ROWS: Final = 2_000
"""Maximum number of training rows used to estimate the effective sample size."""
ESS_EQUATION_VERSION: Final = (
    "rohanhore/RLCP@89dd18be:n_over_frobenius_squared_minus_one:"
    "train_geometry_cap_2000:calibration_size_projection:"
    "fresh_anchor_draws_per_candidate:pcg64"
)
SolverStatus = Literal[
    "official_grid",
    "uniform_limit_selected",
    "uniform_limit_unattainable",
]


@dataclass(frozen=True, slots=True)
class ESSBandwidthSelection:
    """One training-only RLCP bandwidth decision."""

    bandwidth: float
    target: float
    achieved_ess: float
    target_reached: bool
    solver_status: SolverStatus
    n_training: int
    n_estimation: int
    n_calibration: int


def _matrix(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _squared_distances(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    left_norm = np.sum(left * left, axis=1)
    right_norm = np.sum(right * right, axis=1)
    return np.asarray(
        np.maximum(
            left_norm[:, None] + right_norm[None, :] - 2.0 * left @ right.T,
            0.0,
        ),
        dtype=np.float64,
    )


def rlcp_effective_sample_size(
    coordinates: NDArray[np.floating],
    bandwidth: float,
    standard_normals: NDArray[np.floating],
    *,
    n_calibration: int | None = None,
) -> float:
    """Estimate the effective sample size, projected to ``n_calibration`` if given."""
    points = _matrix(coordinates, "coordinates")
    normals = _matrix(standard_normals, "standard_normals")
    if normals.shape != points.shape:
        raise ValueError("standard_normals must match coordinates")
    if math.isnan(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    calibration_rows = len(points) if n_calibration is None else n_calibration
    if calibration_rows < 1:
        raise ValueError("n_calibration must be positive")
    if math.isinf(bandwidth):
        return float(calibration_rows - 1)

    anchors = points + bandwidth * normals
    log_weights = -_squared_distances(anchors, points) / (2.0 * bandwidth**2)
    relative = np.exp(log_weights - np.max(log_weights, axis=1, keepdims=True))
    row_totals = np.sum(relative, axis=1)
    concentration = np.sum(relative * relative, axis=1) / (row_totals * row_totals)
    mean_concentration = float(np.mean(concentration))
    achieved = calibration_rows / (len(points) * mean_concentration) - 1.0
    return float(np.clip(achieved, 0.0, calibration_rows - 1.0))


def _selection(
    bandwidth: float,
    target: float,
    achieved: float,
    reached: bool,
    status: SolverStatus,
    n_training: int,
    n_estimation: int,
    n_calibration: int,
) -> ESSBandwidthSelection:
    return ESSBandwidthSelection(
        bandwidth,
        target,
        achieved,
        reached,
        status,
        n_training,
        n_estimation,
        n_calibration,
    )


def select_ess_bandwidth(
    coordinates: NDArray[np.floating],
    *,
    n_calibration: int,
    target: float,
    random_seed: int,
) -> ESSBandwidthSelection:
    """Select the first grid bandwidth whose effective sample size reaches ``target``.

    The grid is ``0.02, 0.04, ..., 6``. At most ``ESS_ESTIMATION_ROWS`` training rows
    estimate the effective sample size, projected to ``n_calibration`` rows. When no
    bandwidth reaches the target, the bandwidth is infinite and the weights are uniform.
    """
    training_points = _matrix(coordinates, "coordinates")
    n_training = len(training_points)
    n_estimation = min(n_training, ESS_ESTIMATION_ROWS)
    if n_calibration < 1:
        raise ValueError("n_calibration must be positive")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target must be finite and positive")
    maximum = float(n_calibration - 1)
    if target > maximum:
        return _selection(
            math.inf,
            target,
            maximum,
            False,
            "uniform_limit_unattainable",
            n_training,
            n_estimation,
            n_calibration,
        )

    if n_training > ESS_ESTIMATION_ROWS:
        subset_rng = np.random.default_rng(
            np.random.SeedSequence(random_seed, spawn_key=(0,))
        )
        subset = subset_rng.choice(
            n_training,
            size=ESS_ESTIMATION_ROWS,
            replace=False,
        )
        points = training_points[np.sort(subset)]
    else:
        points = training_points

    rng = np.random.Generator(np.random.PCG64(random_seed))
    grid = np.arange(
        OFFICIAL_H_MIN,
        OFFICIAL_H_MAX + OFFICIAL_H_STEP / 2.0,
        OFFICIAL_H_STEP,
    )
    for bandwidth in grid:
        achieved = rlcp_effective_sample_size(
            points,
            float(bandwidth),
            rng.standard_normal(points.shape),
            n_calibration=n_calibration,
        )
        if achieved >= target:
            return _selection(
                float(bandwidth),
                target,
                achieved,
                True,
                "official_grid",
                n_training,
                n_estimation,
                n_calibration,
            )
    return _selection(
        math.inf,
        target,
        maximum,
        True,
        "uniform_limit_selected",
        n_training,
        n_estimation,
        n_calibration,
    )


def ess_bandwidth_record(selection: ESSBandwidthSelection) -> dict[str, object]:
    """Return the metadata stored with each RLCP experiment."""
    return {
        "bandwidth": selection.bandwidth,
        "ess_equation": "rlcp",
        "ess_equation_version": ESS_EQUATION_VERSION,
        "ess_target": selection.target,
        "ess_achieved": selection.achieved_ess,
        "ess_target_reached": selection.target_reached,
        "ess_solver_status": selection.solver_status,
        "ess_n_training": selection.n_training,
        "ess_n_estimation": selection.n_estimation,
        "ess_n_calibration": selection.n_calibration,
    }


__all__ = (
    "ESS_ESTIMATION_ROWS",
    "ESSBandwidthSelection",
    "ess_bandwidth_record",
    "rlcp_effective_sample_size",
    "select_ess_bandwidth",
)

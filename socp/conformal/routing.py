"""Regime 3 global cell budget from training BMU counts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _positive_integer(value: int, *, name: str) -> int:
    """Return a validated positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value}")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return integer


def _positive_finite(value: float, *, name: str) -> float:
    """Return a validated positive finite scalar."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return scalar


@dataclass(frozen=True, slots=True)
class Regime3RoutingConfig:
    """Regime 3 inputs fixed before calibration.

    Args:
        calibration_rows: Planned calibration size ``n``.
        projected_buffer_floor: Target ``b`` for the smallest projected buffer size.
    """

    calibration_rows: int
    projected_buffer_floor: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_rows",
            _positive_integer(self.calibration_rows, name="calibration_rows"),
        )
        object.__setattr__(
            self,
            "projected_buffer_floor",
            _positive_finite(
                self.projected_buffer_floor,
                name="projected_buffer_floor",
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievalOrder:
    """One fixed all-cell order with its topological prefix length."""

    cells: tuple[int, ...]
    base_size: int

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("retrieval order must contain at least one cell")
        if any(
            isinstance(cell, bool)
            or not isinstance(cell, (int, np.integer))
            or int(cell) < 0
            for cell in self.cells
        ):
            raise ValueError("retrieval cell identifiers must be non-negative integers")
        if (
            isinstance(self.base_size, bool)
            or not isinstance(self.base_size, (int, np.integer))
            or not 1 <= int(self.base_size) <= len(self.cells)
        ):
            raise ValueError("base_size must index a non-empty retrieval prefix")
        object.__setattr__(self, "cells", tuple(int(cell) for cell in self.cells))
        object.__setattr__(self, "base_size", int(self.base_size))


@dataclass(frozen=True, slots=True)
class Regime3Budget:
    """Frozen global budget and whether it reaches the projected buffer floor."""

    extra_cells: int
    target_reached: bool


def _training_counts(
    values: NDArray[np.integer],
    *,
    n_cells: int,
) -> NDArray[np.int64]:
    """Return immutable, integral training hits with one value per cell."""
    raw = np.asarray(values)
    if raw.shape != (n_cells,):
        raise ValueError(
            "training_hit_counts must contain one value per cell, got "
            f"{raw.shape} for {n_cells} cells"
        )
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training_hit_counts must contain non-negative integers"
        ) from error
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise ValueError("training_hit_counts must contain non-negative integers")
    counts = np.asarray(numeric, dtype=np.int64)
    if int(np.sum(counts)) < 1:
        raise ValueError("training_hit_counts must contain at least one training row")
    counts.setflags(write=False)
    return counts


def _validate_orders(
    orders: tuple[RetrievalOrder, ...],
) -> None:
    """Require one complete permutation of all map cells per query cell."""
    n_cells = len(orders)
    if n_cells < 1:
        raise ValueError("orders must contain one retrieval order per map cell")
    expected = list(range(n_cells))
    for order in orders:
        if len(order.cells) != n_cells or sorted(order.cells) != expected:
            raise ValueError(
                "every retrieval order must contain every map cell exactly once"
            )


def _worst_projected_buffer(
    counts: NDArray[np.int64],
    orders: tuple[RetrievalOrder, ...],
    *,
    n_cal: int,
    extra_cells: int,
) -> float:
    """Project the smallest query-cell buffer for one fixed cell budget."""
    training_rows = int(np.sum(counts))
    projected = []
    for order in orders:
        stop = min(len(order.cells), order.base_size + extra_cells)
        cells = np.asarray(order.cells[:stop], dtype=np.intp)
        projected.append(n_cal * float(np.sum(counts[cells])) / training_rows)
    return float(min(projected))


def select_training_routed_budget(
    training_hit_counts: NDArray[np.integer],
    orders: tuple[RetrievalOrder, ...],
    *,
    n_cal: int,
    projected_buffer_floor: float,
) -> Regime3Budget:
    """Select the smallest global budget ``L`` whose projected buffers all reach ``b``.

    Empty cells stay in every order and use a budget position. When no budget reaches
    ``b``, every cell is retrieved and ``target_reached`` is ``False``.

    Args:
        training_hit_counts: Training BMU counts in flattened cell order.
        orders: Fixed order of all cells for every query BMU.
        n_cal: Planned calibration size ``n``.
        projected_buffer_floor: Target ``b`` for the smallest projected buffer size.

    Returns:
        The selected global budget and whether it reaches ``b``.
    """
    _validate_orders(orders)
    calibration_rows = _positive_integer(n_cal, name="n_cal")
    floor = _positive_finite(
        projected_buffer_floor,
        name="projected_buffer_floor",
    )
    counts = _training_counts(training_hit_counts, n_cells=len(orders))
    curve = tuple(
        _worst_projected_buffer(
            counts,
            orders,
            n_cal=calibration_rows,
            extra_cells=extra_cells,
        )
        for extra_cells in range(len(orders) + 1)
    )
    selected = next(
        (index for index, projected in enumerate(curve) if projected >= floor),
        len(orders),
    )
    return Regime3Budget(
        extra_cells=selected,
        target_reached=curve[selected] >= floor,
    )

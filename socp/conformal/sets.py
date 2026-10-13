"""Prediction-set representations for regression and classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PredictionSet(Protocol):
    """A set of admissible labels or response values."""

    def contains(self, y: float) -> bool:
        """Return whether ``y`` belongs to the set."""
        ...

    def measure(self) -> float:
        """Return Lebesgue measure for regression or cardinality for classes."""
        ...

    def is_empty(self) -> bool:
        """Return whether the represented set contains no admissible value."""
        ...


@dataclass(frozen=True, slots=True)
class Interval:
    """A non-empty closed interval ``[lo, hi]`` over the extended real line."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if math.isnan(self.lo) or math.isnan(self.hi) or self.lo > self.hi:
            raise ValueError(
                f"lo must not exceed hi and neither may be NaN, got {self}"
            )
        if self.lo == self.hi and not math.isfinite(self.lo):
            raise ValueError("an interval cannot be a singleton at infinity")

    @classmethod
    def unbounded(cls) -> Interval:
        """Return the whole real line, used for an infinite threshold."""
        return cls(-math.inf, math.inf)

    def contains(self, y: float) -> bool:
        """Return whether ``y`` lies in this closed interval."""
        return bool(self.lo <= y <= self.hi)

    def measure(self) -> float:
        """Return interval length, including ``+inf`` for an unbounded interval."""
        return self.hi - self.lo

    def is_empty(self) -> bool:
        """Return ``False`` because every valid closed interval contains an endpoint."""
        return False


@dataclass(frozen=True, slots=True)
class IntervalUnion:
    """A finite union of normalized disjoint closed intervals.

    ``pieces`` must already be ascending and non-touching.
    """

    pieces: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        previous_hi = -math.inf
        for index, (lo, hi) in enumerate(self.pieces):
            invalid_endpoint = math.isnan(lo) or math.isnan(hi) or lo > hi
            singleton_at_infinity = lo == hi and not math.isfinite(lo)
            overlaps_previous = index > 0 and lo <= previous_hi
            if invalid_endpoint or singleton_at_infinity or overlaps_previous:
                raise ValueError(
                    f"pieces must be disjoint and sorted, got {self.pieces}"
                )
            previous_hi = hi

    def contains(self, y: float) -> bool:
        """Return whether ``y`` lies in at least one closed piece."""
        return any(lo <= y <= hi for lo, hi in self.pieces)

    def measure(self) -> float:
        """Return the sum of piece lengths, or zero for the empty union."""
        return float(sum(hi - lo for lo, hi in self.pieces))

    def is_empty(self) -> bool:
        """Return whether the union has no pieces."""
        return not self.pieces


@dataclass(frozen=True, slots=True, eq=False)
class LabelSet:
    """An immutable subset of a finite label space, represented by a boolean mask."""

    mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.mask, np.ndarray):
            raise ValueError("mask must be a one-dimensional boolean vector")
        if self.mask.ndim != 1 or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a one-dimensional boolean vector")

        immutable_mask = self.mask.copy()
        immutable_mask.setflags(write=False)
        object.__setattr__(self, "mask", immutable_mask)

    def contains(self, y: float) -> bool:
        """Return whether an integer-valued class index is admitted.

        Raises:
            ValueError: If ``y`` is not a finite integer-valued number.
            IndexError: If the integer index is outside the label space.
        """
        value = float(y)
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"label must be a finite integer-valued index, got {y}")
        index = int(value)
        if not 0 <= index < self.mask.shape[0]:
            raise IndexError(f"label {index} outside 0..{self.mask.shape[0] - 1}")
        return bool(self.mask[index])

    def measure(self) -> float:
        """Return the number of admitted classes."""
        return float(self.mask.sum())

    def is_empty(self) -> bool:
        """Return whether no class is admitted."""
        return not bool(self.mask.any())

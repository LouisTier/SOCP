"""Conformal Histogram Regression score with observation-owned randomization."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import Cutoff, ScoreSpec
from socp.conformal.scores._chr_core import (
    _nested_sequence,
    _scores_from_probabilities,
    _validated_probabilities,
)
from socp.conformal.scores._regression import _scalar_cutoff
from socp.conformal.sets import Interval, IntervalUnion, PredictionSet

_INTEGER_GUARD = 1e-9
_ObservationDrawFn = Callable[
    [NDArray[np.int64], str, int | None],
    NDArray[np.float64],
]


class CHRScore:
    """Score the randomized nested CHR intervals.

    The conditional histogram and its response grid are fitted on training data before
    this score is built, and ``fit`` only freezes them.

    Args:
        probability_fn: Frozen predictor returning one unit-sum histogram per row.
        bin_edges: Strictly increasing training-only response-grid edges.
        n_levels: Resolution ``T`` of the nested coverage grid.
        tau_anchor: Coverage level from which the nested intervals are built.
        randomized: Whether to randomize the interval endpoints.
        rng: Generator owning one independent uniform per scored observation.
    """

    def __init__(
        self,
        probability_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        bin_edges: NDArray[np.float64],
        *,
        n_levels: int = 1_000,
        tau_anchor: float = 0.9,
        randomized: bool = False,
        rng: np.random.Generator | None = None,
        observation_draws: _ObservationDrawFn | None = None,
    ) -> None:
        edges = np.asarray(bin_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError(
                "bin_edges must contain at least three one-dimensional edges"
            )
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin_edges must be finite and strictly increasing")
        if isinstance(n_levels, bool) or n_levels < 2:
            raise ValueError(
                f"n_levels must be an integer at least two, got {n_levels}"
            )
        if not 0.0 < tau_anchor < 1.0:
            raise ValueError(
                f"tau_anchor must lie strictly in (0, 1), got {tau_anchor}"
            )
        if randomized and rng is None and observation_draws is None:
            raise ValueError("randomized=True requires an rng or observation_draws")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if observation_draws is not None and not callable(observation_draws):
            raise TypeError("observation_draws must be callable")
        if rng is not None and observation_draws is not None:
            raise ValueError("rng and observation_draws are mutually exclusive")

        immutable_edges = edges.copy()
        immutable_edges.setflags(write=False)
        self.spec = ScoreSpec(name="chr", task="regression", randomized=randomized)
        self._probability_fn = probability_fn
        self._edges = immutable_edges
        self._n_levels = int(n_levels)
        rounded_anchor = math.floor(tau_anchor * self._n_levels + 0.5)
        self._anchor_level = int(np.clip(rounded_anchor, 1, self._n_levels - 1))
        self._randomized = randomized
        self._rng = rng
        self._observation_draws = observation_draws
        self._bound_row_ids: NDArray[np.int64] | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the injected histogram has passed the fitted boundary."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Validate and freeze the already-trained histogram predictor."""
        self._probabilities(X[:1])
        self._is_fitted = True

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the smallest CHR level whose randomized interval contains ``y``."""
        self._require_fitted()
        probabilities = self._probabilities(X)
        uniforms = self._draw_uniforms(X.shape[0])
        return _scores_from_probabilities(
            probabilities,
            self._edges,
            np.asarray(y, dtype=np.float64),
            n_levels=self._n_levels,
            anchor_level=self._anchor_level,
            uniforms=uniforms,
        )

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return the randomized CHR interval at ``floor(cutoff)``."""
        self._require_fitted()
        threshold = _scalar_cutoff(cutoff)
        if threshold < 0.0:
            return IntervalUnion(())
        if threshold == math.inf or threshold >= self._n_levels:
            return Interval.unbounded()
        nearest_integer = round(threshold)
        if math.isclose(
            threshold,
            nearest_integer,
            rel_tol=0.0,
            abs_tol=_INTEGER_GUARD,
        ):
            threshold = float(nearest_integer)
        if threshold >= self._n_levels:
            return Interval.unbounded()
        level = math.floor(threshold)
        probabilities = self._probabilities(x[None, :])[0]
        epsilon = None
        if self._randomized:
            uniforms = self._draw_uniforms(1)
            if uniforms is None:
                raise RuntimeError("randomized CHR failed to produce a uniform")
            epsilon = float(uniforms[0])
        sequence = _nested_sequence(
            probabilities,
            n_levels=self._n_levels,
            anchor_level=self._anchor_level,
            epsilon=epsilon,
        )
        lo, hi = int(sequence[level, 0]), int(sequence[level, 1])
        if lo > hi:
            return IntervalUnion(())
        return Interval(float(self._edges[lo]), float(self._edges[hi + 1]))

    def measure_lookup(
        self,
        X: NDArray[np.float64],
    ) -> Callable[[NDArray[np.int64], NDArray[np.float64]], NDArray[np.float64]]:
        """Precompute CHR widths for repeated cutoff evaluation on fixed rows."""
        self._require_fitted()
        probabilities = self._probabilities(X)
        uniforms = self._draw_uniforms(len(X))
        widths = np.zeros((len(X), self._n_levels + 1), dtype=np.float64)
        widths[:, self._n_levels] = math.inf
        for row, masses in enumerate(probabilities):
            epsilon = None if uniforms is None else float(uniforms[row])
            sequence = _nested_sequence(
                masses,
                n_levels=self._n_levels,
                anchor_level=self._anchor_level,
                epsilon=epsilon,
            )
            lo = sequence[: self._n_levels, 0]
            hi = sequence[: self._n_levels, 1]
            nonempty = lo <= hi
            widths[row, : self._n_levels][nonempty] = (
                self._edges[hi[nonempty] + 1] - self._edges[lo[nonempty]]
            )

        def lookup(
            rows: NDArray[np.int64],
            cutoffs: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            indices = np.asarray(rows, dtype=np.int64)
            thresholds = np.asarray(cutoffs, dtype=np.float64)
            if indices.ndim != 1 or thresholds.shape != indices.shape:
                raise ValueError("rows and cutoffs must be aligned vectors")
            if np.any(indices < 0) or np.any(indices >= len(X)):
                raise ValueError("rows must index the precomputed CHR observations")
            levels = np.empty(len(thresholds), dtype=np.int64)
            for position, threshold in enumerate(thresholds):
                if threshold < 0.0:
                    levels[position] = 0
                elif math.isinf(float(threshold)) or threshold >= self._n_levels:
                    levels[position] = self._n_levels
                else:
                    nearest = round(float(threshold))
                    adjusted = (
                        float(nearest)
                        if math.isclose(
                            float(threshold),
                            nearest,
                            rel_tol=0.0,
                            abs_tol=_INTEGER_GUARD,
                        )
                        else float(threshold)
                    )
                    levels[position] = math.floor(adjusted)
            return np.asarray(widths[indices, levels], dtype=np.float64)

        return lookup

    def _probabilities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = np.asarray(self._probability_fn(X), dtype=np.float64)
        return _validated_probabilities(
            raw,
            n_rows=X.shape[0],
            n_bins=self._edges.size - 1,
        )

    def _draw_uniforms(self, n_rows: int) -> NDArray[np.float64] | None:
        if not self._randomized:
            return None
        if self._observation_draws is not None:
            identifiers = self._bound_row_ids
            if identifiers is None or identifiers.shape != (n_rows,):
                raise RuntimeError(
                    "observation-keyed CHR call requires aligned bound row IDs"
                )
            self._bound_row_ids = None
            return self._observation_draws(identifiers, "boundary", None)
        assert self._rng is not None
        return np.asarray(self._rng.uniform(size=n_rows), dtype=np.float64)

    def bind_observation_ids(self, row_ids: NDArray[np.integer]) -> None:
        """Bind immutable row identities for the next randomized score call."""
        identifiers = np.asarray(row_ids)
        if identifiers.ndim != 1 or not np.issubdtype(
            identifiers.dtype,
            np.integer,
        ):
            raise ValueError("score row IDs must be a one-dimensional integer array")
        bound = identifiers.astype(np.int64, copy=True)
        bound.setflags(write=False)
        self._bound_row_ids = bound

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("chr must be fit before use")

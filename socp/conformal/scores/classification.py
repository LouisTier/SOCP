"""LAC, APS, RAPS, SAPS, and rank-only classification scores.

Every ordered score has the form ``cumulative_before_label + U * boundary_mass``, with
one uniform ``U`` per observation shared by all candidate labels. APS, RAPS, and SAPS
order exact probability ties by random keys, from ``observation_draws`` or ``rng``,
or by ascending label index when neither is given. The rank-only score always uses
the label index.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from socp.conformal.score import Cutoff, ScoreSpec
from socp.conformal.scores._classification_core import (
    _aps_score_matrix,
    _label_ranking_score_matrix,
    _lac_score_matrix,
    _raps_score_matrix,
    _saps_score_matrix,
    _validated_probabilities,
    _validated_tie_breakers,
    _validated_uniforms,
)
from socp.conformal.sets import LabelSet, PredictionSet

_ProbabilityFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
_ObservationDrawFn = Callable[
    [NDArray[np.int64], str, int | None],
    NDArray[np.float64],
]


class _ClassificationScore:
    """Common fitted lifecycle and candidate-set inversion."""

    def __init__(self, probability_fn: _ProbabilityFn, *, name: str) -> None:
        self.spec = ScoreSpec(name=name, task="classification")
        self._probability_fn = probability_fn
        self._is_fitted = False
        self._n_classes: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the frozen probability predictor has been validated."""
        return self._is_fitted

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Validate and freeze a classifier fitted before calibration."""
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty two-dimensional feature matrix")
        probabilities = _validated_probabilities(
            self._probability_fn(X[:1]),
            n_rows=1,
        )
        self._validate_n_classes(probabilities.shape[1])
        self._n_classes = probabilities.shape[1]
        self._is_fitted = True

    def _validate_n_classes(self, n_classes: int) -> None:
        """Validate class-count-dependent score parameters."""

    def _probabilities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self._require_fitted()
        assert self._n_classes is not None
        return _validated_probabilities(
            self._probability_fn(X),
            n_rows=X.shape[0],
            n_classes=self._n_classes,
        )

    def score_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one coupled score for every observation-candidate pair."""
        return self.score_matrix_from_probabilities(self._probabilities(X))

    def score_matrix_from_probabilities(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None = None,
        tie_breakers: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Score an already-computed probability matrix.

        Random inputs left unset are drawn from the score's generator.
        """
        self._require_fitted()
        assert self._n_classes is not None
        checked = _validated_probabilities(
            probabilities,
            n_classes=self._n_classes,
        )
        return self._matrix(
            checked,
            uniforms=uniforms,
            tie_breakers=tie_breakers,
        )

    def __call__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the candidate-matrix entry for every observed label."""
        matrix = self.score_matrix(X)
        labels = self._labels(y, n_rows=matrix.shape[0], n_classes=matrix.shape[1])
        return np.asarray(matrix[np.arange(matrix.shape[0]), labels], dtype=np.float64)

    def invert(self, x: NDArray[np.float64], cutoff: Cutoff) -> PredictionSet:
        """Return all labels whose coupled score is at most ``cutoff``."""
        row = self.score_matrix(x[None, :])[0]
        return LabelSet(self._membership(row, cutoff))

    def invert_batch(
        self,
        X: NDArray[np.float64],
        cutoffs: Sequence[Cutoff],
    ) -> tuple[LabelSet, ...]:
        """Invert one scalar or label-indexed cutoff per observation."""
        if len(cutoffs) != X.shape[0]:
            raise ValueError(
                f"cutoffs must contain {X.shape[0]} entries, got {len(cutoffs)}"
            )
        matrix = self.score_matrix(X)
        return tuple(
            LabelSet(self._membership(row, cutoff))
            for row, cutoff in zip(matrix, cutoffs, strict=True)
        )

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def _labels(
        y: NDArray[np.float64],
        *,
        n_rows: int,
        n_classes: int,
    ) -> NDArray[np.int64]:
        labels = np.asarray(y, dtype=np.float64)
        if (
            labels.shape != (n_rows,)
            or not np.all(np.isfinite(labels))
            or not np.all(labels == np.floor(labels))
        ):
            raise ValueError(
                "y must contain integer class labels, one finite label per row"
            )
        integer_labels = labels.astype(np.int64)
        if np.any((integer_labels < 0) | (integer_labels >= n_classes)):
            raise ValueError(f"class label outside the valid range 0..{n_classes - 1}")
        return integer_labels

    @staticmethod
    def _membership(row: NDArray[np.float64], cutoff: Cutoff) -> NDArray[np.bool_]:
        if isinstance(cutoff, np.ndarray):
            threshold = np.asarray(cutoff, dtype=np.float64)
            if threshold.shape != row.shape:
                raise ValueError(
                    "label-indexed cutoff must have one entry per class, got "
                    f"{threshold.shape} for {row.shape[0]} classes"
                )
            if np.any(np.isnan(threshold)):
                raise ValueError("cutoff must not contain NaN")
            return np.asarray(row <= threshold, dtype=np.bool_)
        scalar_threshold = float(cutoff)
        if math.isnan(scalar_threshold):
            raise ValueError("cutoff must not be NaN")
        return np.asarray(row <= scalar_threshold, dtype=np.bool_)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.spec.name} must be fit before use")


class _OrderedClassificationScore(_ClassificationScore):
    """Own boundary and probability-tie randomization for ordered scores."""

    def __init__(
        self,
        probability_fn: _ProbabilityFn,
        *,
        name: str,
        randomized: bool,
        rng: np.random.Generator | None,
        observation_draws: _ObservationDrawFn | None,
    ) -> None:
        if not isinstance(randomized, bool):
            raise TypeError("randomized must be a boolean")
        if randomized and rng is None and observation_draws is None:
            raise ValueError("randomized=True requires an rng or observation_draws")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if observation_draws is not None and not callable(observation_draws):
            raise TypeError("observation_draws must be callable")
        if rng is not None and observation_draws is not None:
            raise ValueError("rng and observation_draws are mutually exclusive")
        super().__init__(probability_fn, name=name)
        self.spec = ScoreSpec(name=name, task="classification", randomized=randomized)
        self._randomized = randomized
        self._rng = rng
        self._observation_draws = observation_draws
        self._bound_row_ids: NDArray[np.int64] | None = None

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

    def _bound_ids(self, n_rows: int) -> NDArray[np.int64]:
        identifiers = self._bound_row_ids
        if identifiers is None or identifiers.shape != (n_rows,):
            raise RuntimeError(
                "observation-keyed score call requires aligned bound row IDs"
            )
        self._bound_row_ids = None
        return identifiers

    def _random_inputs(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_rows, n_classes = probabilities.shape
        keyed_ids = None
        if self._observation_draws is not None and (
            tie_breakers is None or (uniforms is None and self._randomized)
        ):
            keyed_ids = self._bound_ids(n_rows)
        if tie_breakers is None:
            if self._observation_draws is not None:
                assert keyed_ids is not None
                keys_array = self._observation_draws(
                    keyed_ids,
                    "tie_breaker",
                    n_classes,
                )
            elif self._rng is None:
                keys_array = np.asarray(
                    np.broadcast_to(
                        np.arange(n_classes, dtype=np.float64),
                        probabilities.shape,
                    ),
                    dtype=np.float64,
                )
            else:
                keys_array = np.asarray(
                    self._rng.uniform(size=probabilities.shape),
                    dtype=np.float64,
                )
        else:
            keys_array = _validated_tie_breakers(
                tie_breakers,
                shape=probabilities.shape,
            )

        if uniforms is None:
            if self._randomized:
                if self._observation_draws is not None:
                    assert keyed_ids is not None
                    boundary_array = self._observation_draws(
                        keyed_ids,
                        "boundary",
                        None,
                    )
                else:
                    assert self._rng is not None
                    boundary_array = np.asarray(
                        self._rng.uniform(size=n_rows),
                        dtype=np.float64,
                    )
            else:
                boundary_array = np.ones(n_rows, dtype=np.float64)
        else:
            boundary_array = _validated_uniforms(uniforms, n_rows=n_rows)
        return boundary_array, keys_array


class LACScore(_ClassificationScore):
    """Least-ambiguous-classification score ``1 - probability(label)``."""

    def __init__(self, probability_fn: _ProbabilityFn) -> None:
        super().__init__(probability_fn, name="lac")

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if uniforms is not None or tie_breakers is not None:
            raise ValueError("LAC is deterministic and accepts no random inputs")
        return _lac_score_matrix(probabilities)


class LabelRankingScore(_OrderedClassificationScore):
    """Randomized rank-only score ``rank(y) - 1 + U``.

    Ranks are one-based, by decreasing probability with ties in label order, and one
    uniform is shared by every candidate label of an observation.
    """

    def __init__(
        self,
        probability_fn: _ProbabilityFn,
        *,
        observation_draws: _ObservationDrawFn,
    ) -> None:
        super().__init__(
            probability_fn,
            name="rank",
            randomized=True,
            rng=None,
            observation_draws=observation_draws,
        )

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        n_rows = probabilities.shape[0]
        if uniforms is not None:
            boundary = _validated_uniforms(uniforms, n_rows=n_rows)
        else:
            assert self._observation_draws is not None
            boundary = self._observation_draws(
                self._bound_ids(n_rows), "boundary", None
            )
        return _label_ranking_score_matrix(probabilities, boundary)


class APSScore(_OrderedClassificationScore):
    """Randomized adaptive-prediction-set (APS) score."""

    def __init__(
        self,
        probability_fn: _ProbabilityFn,
        *,
        randomized: bool = True,
        rng: np.random.Generator | None = None,
        observation_draws: _ObservationDrawFn | None = None,
    ) -> None:
        super().__init__(
            probability_fn,
            name="aps",
            randomized=randomized,
            rng=rng,
            observation_draws=observation_draws,
        )

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        boundary, keys = self._random_inputs(
            probabilities,
            uniforms=uniforms,
            tie_breakers=tie_breakers,
        )
        return _aps_score_matrix(probabilities, boundary, keys)


class RAPSScore(_OrderedClassificationScore):
    """APS plus the RAPS rank penalty after ``k_reg`` labels."""

    def __init__(
        self,
        probability_fn: _ProbabilityFn,
        *,
        penalty: float,
        k_reg: int,
        randomized: bool = True,
        rng: np.random.Generator | None = None,
        observation_draws: _ObservationDrawFn | None = None,
    ) -> None:
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(f"penalty must be finite and non-negative, got {penalty}")
        if isinstance(k_reg, bool) or not isinstance(k_reg, int):
            raise TypeError("k_reg must be an integer")
        if k_reg < 0:
            raise ValueError(f"k_reg must be non-negative, got {k_reg}")
        super().__init__(
            probability_fn,
            name="raps",
            randomized=randomized,
            rng=rng,
            observation_draws=observation_draws,
        )
        self._penalty = float(penalty)
        self._k_reg = k_reg

    def _validate_n_classes(self, n_classes: int) -> None:
        if self._k_reg > n_classes:
            raise ValueError(
                f"k_reg must not exceed the class count {n_classes}, got {self._k_reg}"
            )

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        boundary, keys = self._random_inputs(
            probabilities,
            uniforms=uniforms,
            tie_breakers=tie_breakers,
        )
        return _raps_score_matrix(
            probabilities,
            boundary,
            keys,
            penalty=self._penalty,
            k_reg=self._k_reg,
        )


class SAPSScore(_OrderedClassificationScore):
    """SAPS score using only maximum probability and candidate rank."""

    def __init__(
        self,
        probability_fn: _ProbabilityFn,
        *,
        weight: float,
        randomized: bool = True,
        rng: np.random.Generator | None = None,
        observation_draws: _ObservationDrawFn | None = None,
    ) -> None:
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"weight must be finite and positive, got {weight}")
        super().__init__(
            probability_fn,
            name="saps",
            randomized=randomized,
            rng=rng,
            observation_draws=observation_draws,
        )
        self._weight = float(weight)

    def _matrix(
        self,
        probabilities: NDArray[np.float64],
        *,
        uniforms: NDArray[np.float64] | None,
        tie_breakers: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        boundary, keys = self._random_inputs(
            probabilities,
            uniforms=uniforms,
            tie_breakers=tie_breakers,
        )
        return _saps_score_matrix(
            probabilities,
            boundary,
            keys,
            weight=self._weight,
        )

"""Compose one score, calibration mechanism, and feature localizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from socp.conformal._validation import feature_matrix, immutable_row_ids
from socp.conformal.localizer import Localizer, fit_localizer
from socp.conformal.mechanism import CalibrationBatch, Mechanism, calibrate_mechanism
from socp.conformal.score import (
    Cutoff,
    Score,
    bind_score_observation_ids,
    evaluate_score,
    fit_score,
    invert_score,
)
from socp.conformal.sets import PredictionSet


@runtime_checkable
class _LocalizerBoundMechanism(Protocol):
    @property
    def localizer(self) -> Localizer: ...


@runtime_checkable
class _ClusteredMechanism(Protocol):
    def fit_taxonomy(
        self,
        auxiliary_scores: NDArray[np.float64],
        auxiliary_labels: NDArray[np.float64],
        *,
        row_ids: NDArray[np.integer],
    ) -> None: ...


@runtime_checkable
class _BatchScore(Protocol):
    def invert_batch(
        self,
        X: NDArray[np.float64],
        cutoffs: Sequence[Cutoff],
    ) -> tuple[PredictionSet, ...]: ...


class ConformalRule:
    """A fitted score calibrated by one mechanism on one feature geometry."""

    def __init__(
        self,
        score: Score,
        mechanism: Mechanism,
        localizer: Localizer,
    ) -> None:
        if not isinstance(score, Score):
            raise TypeError("score must satisfy the Score protocol")
        if not isinstance(mechanism, Mechanism):
            raise TypeError("mechanism must satisfy the Mechanism protocol")
        if not isinstance(localizer, Localizer):
            raise TypeError("localizer must satisfy the Localizer protocol")
        if score.spec.task != mechanism.spec.task:
            raise ValueError(
                f"score task {score.spec.task!r} does not match "
                f"mechanism task {mechanism.spec.task!r}"
            )
        if (
            isinstance(mechanism, _LocalizerBoundMechanism)
            and mechanism.localizer is not localizer
        ):
            raise ValueError("mechanism and rule must use the same localizer instance")

        self.score = score
        self.mechanism = mechanism
        self.localizer = localizer
        self._is_fitted = False
        self._is_calibrated = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> ConformalRule:
        """Fit score-owned state and feature geometry before calibration."""
        if self._is_fitted:
            raise RuntimeError("rule is already fitted")
        fit_score(self.score, X, y)
        if not self.localizer.is_fitted:
            fit_localizer(self.localizer, X)
        self._is_fitted = True
        return self

    def calibrate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        row_ids: NDArray[np.integer] | None = None,
        auxiliary_indices: NDArray[np.integer] | None = None,
        proper_indices: NDArray[np.integer] | None = None,
    ) -> ConformalRule:
        """Fit the calibration cutoff on the calibration sample.

        Clustered calibration also takes a disjoint split of these rows. The
        ``auxiliary_indices`` rows cluster the labels, and the remaining rows,
        ``proper_indices``, set the cutoffs.
        """
        if not self._is_fitted:
            raise RuntimeError("rule must be fitted before calibration")
        if self._is_calibrated:
            raise RuntimeError("rule is already calibrated")

        features = feature_matrix(X)
        ids = self._row_ids(row_ids, features.shape[0])
        if self.score.spec.randomized:
            if ids is None:
                raise ValueError("randomized scores require calibration row_ids")
            bind_score_observation_ids(self.score, ids)
        scores = evaluate_score(self.score, features, y)

        clustered = self.mechanism.spec.requires_auxiliary_split
        if clustered:
            self._calibrate_clustered(
                features,
                np.asarray(y, dtype=np.float64),
                scores,
                ids,
                auxiliary_indices,
                proper_indices,
            )
        else:
            if auxiliary_indices is not None or proper_indices is not None:
                raise ValueError(
                    "auxiliary/proper indices are only valid for clustered CP"
                )
            routing = y if self.mechanism.spec.routing_source == "outcome" else None
            calibrate_mechanism(
                self.mechanism,
                CalibrationBatch(
                    task=self.score.spec.task,
                    features=features,
                    scores=scores,
                    routing=routing,
                    row_ids=ids,
                ),
            )

        self._is_calibrated = True
        return self

    def _calibrate_clustered(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        scores: NDArray[np.float64],
        row_ids: NDArray[np.int64] | None,
        auxiliary_indices: NDArray[np.integer] | None,
        proper_indices: NDArray[np.integer] | None,
    ) -> None:
        if not isinstance(self.mechanism, _ClusteredMechanism):
            raise TypeError("clustered mechanism lacks taxonomy fitting")
        if auxiliary_indices is None or proper_indices is None:
            raise ValueError("clustered CP requires auxiliary and proper indices")
        auxiliary, proper = self._partition(
            auxiliary_indices,
            proper_indices,
            n_rows=X.shape[0],
        )
        ids = row_ids if row_ids is not None else np.arange(X.shape[0], dtype=np.int64)
        mechanism = cast(_ClusteredMechanism, self.mechanism)
        mechanism.fit_taxonomy(
            scores[auxiliary],
            y[auxiliary],
            row_ids=ids[auxiliary],
        )
        calibrate_mechanism(
            self.mechanism,
            CalibrationBatch(
                task=self.score.spec.task,
                features=X[proper],
                scores=scores[proper],
                routing=y[proper],
                row_ids=ids[proper],
            ),
        )

    def predict(
        self,
        X: NDArray[np.float64],
        *,
        row_ids: NDArray[np.integer] | None = None,
    ) -> tuple[PredictionSet, ...]:
        """Return one prediction set per query row."""
        predictions, _ = self.predict_with_cutoffs(X, row_ids=row_ids)
        return predictions

    def predict_with_cutoffs(
        self,
        X: NDArray[np.float64],
        *,
        row_ids: NDArray[np.integer] | None = None,
    ) -> tuple[tuple[PredictionSet, ...], tuple[Cutoff, ...]]:
        """Return prediction sets together with the cutoffs that generated them."""
        if not self._is_calibrated:
            raise RuntimeError("rule must be calibrated before prediction")
        queries = feature_matrix(X)
        ids = self._row_ids(row_ids, queries.shape[0])
        if self.score.spec.randomized and ids is None:
            raise ValueError("randomized scores require prediction row_ids")
        cutoffs = tuple(self.mechanism.threshold(query) for query in queries)
        if isinstance(self.score, _BatchScore):
            if self.score.spec.randomized:
                assert ids is not None
                bind_score_observation_ids(self.score, ids)
            predictions = self.score.invert_batch(queries, cutoffs)
        else:
            predictions = tuple(
                self._invert_one(
                    query,
                    cutoff,
                    row_id=None if ids is None else int(ids[index]),
                )
                for index, (query, cutoff) in enumerate(
                    zip(queries, cutoffs, strict=True)
                )
            )
        return predictions, cutoffs

    def _invert_one(
        self,
        query: NDArray[np.float64],
        cutoff: Cutoff,
        *,
        row_id: int | None,
    ) -> PredictionSet:
        if self.score.spec.randomized:
            assert row_id is not None
            bind_score_observation_ids(
                self.score,
                np.asarray([row_id], dtype=np.int64),
            )
        return invert_score(self.score, query, cutoff)

    @staticmethod
    def _row_ids(
        row_ids: NDArray[np.integer] | None,
        n_rows: int,
    ) -> NDArray[np.int64] | None:
        if row_ids is None:
            return None
        return immutable_row_ids(row_ids, expected_rows=n_rows)

    @staticmethod
    def _partition(
        auxiliary_indices: NDArray[np.integer],
        proper_indices: NDArray[np.integer],
        *,
        n_rows: int,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        blocks: list[NDArray[np.intp]] = []
        for name, values in (
            ("auxiliary_indices", auxiliary_indices),
            ("proper_indices", proper_indices),
        ):
            raw = np.asarray(values)
            if (
                raw.ndim != 1
                or raw.size == 0
                or not np.issubdtype(raw.dtype, np.integer)
            ):
                raise ValueError(f"{name} must be a non-empty integer vector")
            indices = np.asarray(raw, dtype=np.intp)
            if np.any(indices < 0) or np.any(indices >= n_rows):
                raise ValueError(f"{name} contains an out-of-range row")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"{name} must not repeat rows")
            blocks.append(indices)
        auxiliary, proper = blocks
        if np.intersect1d(auxiliary, proper, assume_unique=True).size:
            raise ValueError("auxiliary and proper indices must be disjoint")
        if not np.array_equal(
            np.sort(np.concatenate((auxiliary, proper))),
            np.arange(n_rows),
        ):
            raise ValueError("auxiliary and proper indices must partition every row")
        return auxiliary, proper


__all__ = ("ConformalRule",)

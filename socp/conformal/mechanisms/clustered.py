"""Clustered class-conditional conformal calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from socp.conformal._validation import immutable_row_ids
from socp.conformal.core import split_quantile
from socp.conformal.mechanism import CalibrationBatch, MechanismSpec
from socp.conformal.mechanisms.classwise import (
    _class_count,
    _finite_scores,
    _label_vector,
    _routing_labels,
)
from socp.ranks import minimum_finite_calibration_count, validate_alpha

EMBEDDING_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
"""Score quantile levels that embed each label for clustering."""

_PLAN_OFFSET: float = 75.0


@dataclass(frozen=True, slots=True)
class AuthorsClusterPlan:
    """Automatic choice of the auxiliary split and the number of label clusters."""

    auxiliary_per_rarest_class: int
    auxiliary_fraction: float
    n_clusters: int


def rarity_floor(alpha: float) -> int:
    """Return the rare-label threshold: the smallest count with a finite threshold."""
    return minimum_finite_calibration_count(alpha)


def authors_cluster_plan(
    class_counts: NDArray[np.intp],
    *,
    alpha: float,
) -> AuthorsClusterPlan:
    """Choose the auxiliary fraction and the number of clusters from the label counts.

    Args:
        class_counts: Total calibration count for every declared label, including
            zeros for absent labels.
        alpha: Target miscoverage.

    Returns:
        The per-rarest-class auxiliary budget, fraction, and cluster count.
    """
    validate_alpha(alpha)
    raw_counts = np.asarray(class_counts)
    if raw_counts.ndim != 1 or raw_counts.size == 0:
        raise ValueError("class_counts must be a non-empty vector")
    try:
        numeric = np.asarray(raw_counts, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("class_counts must contain non-negative integers") from error
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or not np.all(numeric == np.floor(numeric))
    ):
        raise ValueError("class_counts must contain non-negative integers")
    counts = np.asarray(numeric, dtype=np.intp)
    target_count = max(int(np.min(counts)), rarity_floor(alpha))
    eligible = int(np.count_nonzero(counts >= target_count))
    auxiliary_count = (
        int(target_count * eligible / (_PLAN_OFFSET + eligible)) if eligible else 0
    )
    fraction = auxiliary_count / target_count if target_count else 0.0
    return AuthorsClusterPlan(
        auxiliary_per_rarest_class=auxiliary_count,
        auxiliary_fraction=float(fraction),
        n_clusters=int(math.floor(auxiliary_count / 2.0)),
    )


def authors_random_split(
    n_rows: int,
    *,
    fraction: float,
    seed: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split rows into auxiliary and remaining rows by independent Bernoulli draws."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and lie in [0, 1]")
    random_state = np.random.RandomState(seed)
    auxiliary_mask = random_state.uniform(size=n_rows) < fraction
    return (
        np.flatnonzero(auxiliary_mask).astype(np.intp),
        np.flatnonzero(~auxiliary_mask).astype(np.intp),
    )


class ClusteredCalibration:
    """Cluster labels on auxiliary calibration rows, then calibrate on the other rows.

    Call ``fit_taxonomy`` with the auxiliary rows, then ``calibrate`` with the
    remaining calibration rows. Labels too rare to embed join the null cluster,
    which uses the pooled cutoff of the remaining rows.
    """

    def __init__(
        self,
        *,
        alpha: float,
        n_classes: int,
        n_clusters: int,
        random_state: int = 0,
    ) -> None:
        validate_alpha(alpha)
        if (
            isinstance(n_clusters, bool)
            or int(n_clusters) != n_clusters
            or n_clusters < 0
        ):
            raise ValueError(
                f"n_clusters must be a non-negative integer, got {n_clusters}"
            )
        self.spec = MechanismSpec(
            name="clustered",
            task="classification",
            requires_routing=True,
            routing_source="outcome",
            requires_auxiliary_split=True,
        )
        self._alpha = float(alpha)
        self._n_classes = _class_count(n_classes)
        self._n_clusters = int(n_clusters)
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("random_state must be an integer")
        self._random_state = random_state
        self._groups: NDArray[np.intp] | None = None
        self._cutoffs: NDArray[np.float64] | None = None
        self._auxiliary_row_ids: NDArray[np.int64] | None = None
        self._taxonomy_is_fitted = False
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Whether the cluster quantiles are frozen."""
        return self._is_calibrated

    @property
    def label_groups(self) -> NDArray[np.intp]:
        """Return immutable label-to-cluster assignments; ``-1`` is the null cluster."""
        if self._groups is None:
            raise RuntimeError("taxonomy must be fitted before label_groups")
        return self._groups

    def fit_taxonomy(
        self,
        auxiliary_scores: NDArray[np.float64],
        auxiliary_labels: NDArray[np.float64],
        *,
        row_ids: NDArray[np.integer] | None,
    ) -> None:
        """Embed each eligible label by its score quantiles and fit weighted k-means."""
        if self._taxonomy_is_fitted:
            raise RuntimeError("taxonomy is already fitted")
        scores = _finite_scores(
            auxiliary_scores,
            name="auxiliary_scores",
            allow_empty=True,
        )
        labels = _label_vector(
            auxiliary_labels,
            n_classes=self._n_classes,
            name="auxiliary labels",
            allow_empty=True,
        )
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("auxiliary scores and labels must have the same length")
        if row_ids is None:
            raise ValueError("row_ids are required for clustered auxiliary fitting")
        auxiliary_row_ids = immutable_row_ids(
            row_ids,
            expected_rows=scores.shape[0],
            name="auxiliary row_ids",
        )

        floor = rarity_floor(self._alpha)
        embedded_labels: list[int] = []
        embeddings: list[NDArray[np.float64]] = []
        counts: list[int] = []
        _, first_indices = np.unique(labels, return_index=True)
        labels_in_source_order = labels[np.sort(first_indices)]
        levels = np.asarray(EMBEDDING_LEVELS, dtype=np.float64)
        for raw_label in labels_in_source_order:
            label = int(raw_label)
            member_scores = scores[labels == label]
            if member_scores.shape[0] < floor:
                continue
            embedded_labels.append(label)
            embeddings.append(
                np.asarray(np.quantile(member_scores, levels), dtype=np.float64)
            )
            counts.append(int(member_scores.shape[0]))

        groups = np.full(self._n_classes, -1, dtype=np.intp)
        if len(embedded_labels) > self._n_clusters and self._n_clusters > 1:
            fitted = KMeans(
                n_clusters=self._n_clusters,
                n_init=10,
                random_state=self._random_state,
            ).fit(
                np.asarray(embeddings, dtype=np.float64),
                sample_weight=np.sqrt(np.asarray(counts, dtype=np.float64)),
            )
            assignments = np.asarray(fitted.labels_, dtype=np.intp)
            if assignments.shape != (len(embedded_labels),):
                raise RuntimeError("k-means returned one label per embedded class")
            for label, cluster in zip(embedded_labels, assignments, strict=True):
                groups[label] = int(cluster)

        groups.setflags(write=False)
        self._groups = groups
        self._auxiliary_row_ids = auxiliary_row_ids
        self._taxonomy_is_fitted = True

    def calibrate(self, batch: CalibrationBatch) -> None:
        """Compute the cluster quantiles from the remaining calibration rows."""
        if self._is_calibrated:
            raise RuntimeError("mechanism is already calibrated")
        if (
            not self._taxonomy_is_fitted
            or self._groups is None
            or self._auxiliary_row_ids is None
        ):
            raise RuntimeError("taxonomy must be fitted before calibration")
        if batch.task != "classification":
            raise ValueError("clustered calibration requires classification scores")
        if batch.row_ids is None:
            raise ValueError("row_ids are required for clustered proper calibration")
        overlap = np.intersect1d(
            self._auxiliary_row_ids,
            batch.row_ids,
            assume_unique=True,
        )
        if overlap.size:
            raise ValueError(
                "clustered auxiliary and proper row_ids must be disjoint, got "
                f"{overlap.size} overlapping rows"
            )
        labels = _routing_labels(batch, n_classes=self._n_classes)

        pooled = split_quantile(batch.scores, self._alpha)
        cutoffs = np.full(self._n_classes, pooled, dtype=np.float64)
        learned = self._groups >= 0
        cutoffs[learned] = math.inf
        realized_clusters = (
            int(np.max(self._groups[learned])) + 1 if np.any(learned) else 0
        )
        routed_groups = self._groups[labels]
        for cluster in range(realized_clusters):
            member_scores = batch.scores[routed_groups == cluster]
            if member_scores.size:
                cutoffs[self._groups == cluster] = split_quantile(
                    member_scores,
                    self._alpha,
                )

        cutoffs.setflags(write=False)
        self._cutoffs = cutoffs
        self._is_calibrated = True

    def threshold(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the query-independent cutoff associated with each candidate label."""
        del x
        if not self._is_calibrated or self._cutoffs is None:
            raise RuntimeError("mechanism must be calibrated before threshold")
        return self._cutoffs

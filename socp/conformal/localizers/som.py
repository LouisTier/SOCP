"""Geometry-only wrapper around the pinned TorchSOM fixed-cell retrieval API."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from numpy.typing import NDArray
from torchsom import SOM

from socp.conformal.localizer import LocalizerSpec
from socp.conformal.routing import (
    Regime3Budget,
    Regime3RoutingConfig,
    RetrievalOrder,
    select_training_routed_budget,
)


class SOMLocalizer:
    """Fit a supplied SOM and freeze a fixed retrieval set for every map cell.

    The caller either sets ``n_extra_cells`` or passes a Regime 3 routing
    configuration. In the latter case, the global budget comes from training BMU
    counts and the planned calibration size, and is frozen before calibration.
    """

    def __init__(
        self,
        som: SOM,
        *,
        n_extra_cells: int | None = None,
        routing: Regime3RoutingConfig | None = None,
        embedding_id: str = "identity",
    ) -> None:
        if (n_extra_cells is None) == (routing is None):
            raise ValueError("provide exactly one of n_extra_cells or routing")
        if n_extra_cells is not None and (
            isinstance(n_extra_cells, bool)
            or not isinstance(n_extra_cells, (int, np.integer))
            or n_extra_cells < 0
        ):
            raise ValueError(
                f"n_extra_cells must be a non-negative integer, got {n_extra_cells}"
            )
        if routing is not None and not isinstance(routing, Regime3RoutingConfig):
            raise TypeError("routing must be a Regime3RoutingConfig")
        self.spec = LocalizerSpec(kernel="indicator", embedding_id=embedding_id)
        self._som = som
        self._explicit_n_extra_cells = (
            int(n_extra_cells) if n_extra_cells is not None else None
        )
        self._routing_config = routing
        self._active_n_extra_cells: int | None = None
        self._routing_plan: Regime3Budget | None = None
        self._is_fitted = False
        self._n_features: int | None = None
        self._retrieval_mask: NDArray[np.bool_] | None = None
        self._reference_cache: tuple[NDArray[np.float64], NDArray[np.intp]] | None = (
            None
        )

    @property
    def is_fitted(self) -> bool:
        """Whether SOM prototypes and all retrieval sets are frozen."""
        return self._is_fitted

    @property
    def n_cells(self) -> int:
        """Return the declared number of map cells."""
        return int(self._som.x * self._som.y)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the fitted SOM grid shape as ``(rows, columns)``."""
        return int(self._som.x), int(self._som.y)

    def distance_map(self) -> NDArray[np.float64]:
        """Return a detached copy of the pinned TorchSOM distance map."""
        if not self._is_fitted:
            raise RuntimeError("localizer must be fitted before distance_map")
        values = self._som.build_map("distance")
        return np.asarray(values.detach().cpu().numpy(), dtype=np.float64).copy()

    def hit_map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return BMU hit counts for one feature split."""
        cells = self.assign(X)
        return (
            np.bincount(cells, minlength=self.n_cells)
            .reshape(self.grid_shape)
            .astype(
                np.float64,
                copy=False,
            )
        )

    def retrieval_mask(self) -> NDArray[np.bool_]:
        """Return a copy of the frozen query-cell retrieval membership mask."""
        if self._retrieval_mask is None:
            raise RuntimeError("localizer must be fitted before retrieval_mask")
        return np.array(self._retrieval_mask, dtype=np.bool_, copy=True)

    @property
    def n_extra_cells(self) -> int:
        """Return the fixed active global cell budget."""
        if self._active_n_extra_cells is None:
            raise RuntimeError("localizer must be fitted before n_extra_cells")
        return self._active_n_extra_cells

    @property
    def routing_plan(self) -> Regime3Budget | None:
        """Return the Regime 3 budget, or ``None`` when ``n_extra_cells`` was set."""
        if not self._is_fitted:
            raise RuntimeError("localizer must be fitted before routing_plan")
        return self._routing_plan

    @property
    def prototype_checksum(self) -> str:
        """Return a stable checksum of the fitted prototype tensor."""
        if not self._is_fitted:
            raise RuntimeError("localizer must be fitted before prototype_checksum")
        weights = np.ascontiguousarray(
            self._som.weights.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        digest = hashlib.sha256()
        digest.update(str(weights.dtype).encode())
        digest.update(np.asarray(weights.shape, dtype=np.int64).tobytes())
        digest.update(weights.tobytes())
        return digest.hexdigest()

    def fit(self, X: NDArray[np.float64]) -> None:
        """Train the SOM once and materialize fixed retrieval geometry."""
        if self._is_fitted:
            raise RuntimeError("localizer is already fitted")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(f"X must be a non-empty matrix, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        self._n_features = int(values.shape[1])
        self._som.fit(torch.tensor(values, dtype=torch.float32), verbose=False)

        orders = self._build_retrieval_orders()
        training_cells = self._identify_bmus(values)
        training_counts = np.bincount(
            training_cells,
            minlength=self.n_cells,
        ).astype(np.int64)
        if self._routing_config is None:
            assert self._explicit_n_extra_cells is not None
            active_n_extra_cells = self._explicit_n_extra_cells
        else:
            self._routing_plan = select_training_routed_budget(
                training_counts,
                orders,
                n_cal=self._routing_config.calibration_rows,
                projected_buffer_floor=self._routing_config.projected_buffer_floor,
            )
            active_n_extra_cells = self._routing_plan.extra_cells

        mask = np.zeros((self.n_cells, self.n_cells), dtype=np.bool_)
        for query_cell in range(self.n_cells):
            cells = self._som.get_retrieval_cells(
                divmod(query_cell, self._som.y),
                n_extra_cells=active_n_extra_cells,
            )
            flattened = tuple(int(row * self._som.y + column) for row, column in cells)
            if not flattened or len(set(flattened)) != len(flattened):
                raise RuntimeError(
                    "TorchSOM retrieval cells must be non-empty and unique"
                )
            if min(flattened) < 0 or max(flattened) >= self.n_cells:
                raise RuntimeError("TorchSOM returned an out-of-range retrieval cell")
            mask[query_cell, np.asarray(flattened, dtype=np.intp)] = True

        mask.setflags(write=False)
        self._retrieval_mask = mask
        self._active_n_extra_cells = active_n_extra_cells
        self._reference_cache = None
        self._is_fitted = True

    def _build_retrieval_orders(self) -> tuple[RetrievalOrder, ...]:
        """Read and validate pinned TorchSOM's occupancy-independent cell orders."""
        orders = []
        expected = list(range(self.n_cells))
        for query_cell in range(self.n_cells):
            position = divmod(query_cell, self._som.y)
            base = self._som.get_retrieval_cells(position, n_extra_cells=0)
            full = self._som.get_retrieval_cells(
                position,
                n_extra_cells=self.n_cells,
            )
            base_flat = tuple(row * self._som.y + column for row, column in base)
            full_flat = tuple(row * self._som.y + column for row, column in full)
            if not base_flat or full_flat[: len(base_flat)] != base_flat:
                raise RuntimeError(
                    "TorchSOM retrieval must start with the complete base neighborhood"
                )
            if len(full_flat) != self.n_cells or sorted(full_flat) != expected:
                raise RuntimeError(
                    "TorchSOM retrieval must contain every map cell exactly once"
                )
            orders.append(RetrievalOrder(full_flat, len(base_flat)))
        return tuple(orders)

    def _identify_bmus(self, values: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return flattened BMUs for a validated feature matrix."""
        bmus = self._som.identify_bmus(torch.tensor(values, dtype=torch.float32))
        coordinates = np.asarray(bmus.cpu().numpy(), dtype=np.intp)
        if coordinates.shape != (values.shape[0], 2):
            raise RuntimeError(
                f"TorchSOM returned BMU shape {coordinates.shape}, expected {(values.shape[0], 2)}"
            )
        flattened = coordinates[:, 0] * self._som.y + coordinates[:, 1]
        if np.any(flattened < 0) or np.any(flattened >= self.n_cells):
            raise RuntimeError("TorchSOM returned an out-of-range BMU")
        return np.asarray(flattened, dtype=np.intp)

    def _feature_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate a feature matrix against the trained SOM dimension."""
        if not self._is_fitted or self._n_features is None:
            raise RuntimeError("localizer must be fitted before use")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._n_features:
            raise ValueError(
                f"X must have shape (n, {self._n_features}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return values

    def assign(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return flattened best-matching-unit indices."""
        values = self._feature_matrix(X)
        return self._identify_bmus(values)

    def _reference_cells(
        self,
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.intp]:
        """Cache BMUs for a repeatedly reused multi-row reference object."""
        if X_ref.shape[0] <= 1:
            return self.assign(X_ref)
        if self._reference_cache is None or self._reference_cache[0] is not X_ref:
            self._reference_cache = (X_ref, self.assign(X_ref))
        return self._reference_cache[1]

    def weights(
        self,
        x: NDArray[np.float64],
        X_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return fixed retrieval membership for every reference row."""
        if self._retrieval_mask is None:
            raise RuntimeError("localizer must be fitted before weights")
        query = np.asarray(x, dtype=np.float64)
        references = np.asarray(X_ref, dtype=np.float64)
        if query.ndim != 1 or references.ndim != 2:
            raise ValueError("x must be one row and X_ref must be a matrix")
        query_cell = int(self.assign(query[None, :])[0])
        reference_cells = self._reference_cells(references)
        return np.asarray(
            self._retrieval_mask[query_cell, reference_cells],
            dtype=np.float64,
        )

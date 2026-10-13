"""Conditional score-quantile network for RCP.

Three linear layers of width 100, with GELU and dropout 0.1, minimize the pinball
loss at ``1-alpha`` with AdamW and plateau scheduling. The first 80% of the rows
train the network, after standardization fitted on them, and the last 20% decide
early stopping. The caller passes out-of-fold scores on training data.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


class ScoreQuantileEstimator:
    """Estimate the conditional ``(1-alpha)`` quantile of base scores.

    Args:
        alpha: Target miscoverage.
        random_state: Model-owned seed for initialization, dropout, and shuffling.
        floor: Positive prediction floor required by multiplicative RCP.
        hidden_size: Hidden width.
        num_layers: Number of linear layers.
        batch_size: Minibatch size.
        max_epochs: Maximum number of epochs.
        patience: Epochs without improvement before early stopping.

    Raises:
        ValueError: If a hyperparameter is invalid.
    """

    def __init__(
        self,
        *,
        alpha: float,
        random_state: int = 0,
        floor: float = 1e-12,
        hidden_size: int = 100,
        num_layers: int = 3,
        batch_size: int = 512,
        max_epochs: int = 5000,
        patience: int = 15,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")
        if hidden_size < 1 or num_layers < 2:
            raise ValueError("hidden_size must be positive and num_layers at least 2")
        if batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        self._alpha = alpha
        self._random_state = random_state
        self._floor = floor
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._network: nn.Sequential | None = None
        self._x_mean: NDArray[np.float64] | None = None
        self._x_scale: NDArray[np.float64] | None = None
        self._score_mean: float | None = None
        self._score_scale: float | None = None

    @property
    def network_(self) -> nn.Sequential:
        """Return the fitted network, or raise before ``fit``."""
        if self._network is None:
            raise RuntimeError("fit must be called before accessing network_")
        return self._network

    def _new_network(self, input_dim: int) -> nn.Sequential:
        modules: list[nn.Module] = []
        width = input_dim
        for _ in range(self._num_layers - 1):
            modules.extend(
                [
                    nn.Linear(width, self._hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            width = self._hidden_size
        modules.append(nn.Linear(width, 1))
        return nn.Sequential(*modules)

    @staticmethod
    def _pinball(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        level: float,
    ) -> torch.Tensor:
        difference = target - prediction
        return torch.mean(
            difference * (level - (difference < 0.0).to(dtype=difference.dtype))
        )

    def _standardize_features(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._x_mean is None or self._x_scale is None:
            raise RuntimeError("ScoreQuantileEstimator.fit must be called first")
        return np.asarray((X - self._x_mean) / self._x_scale, dtype=np.float64)

    def fit(
        self,
        X: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> ScoreQuantileEstimator:
        """Fit on pre-calibration feature/score pairs."""
        features = np.asarray(X, dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
        invalid_shape = (
            features.ndim != 2
            or values.ndim != 1
            or features.shape[0] != values.shape[0]
            or features.shape[0] < 2
            or features.shape[1] == 0
        )
        if invalid_shape:
            raise ValueError(
                "X and scores must have shapes (n, d) and (n,) with n >= 2, "
                f"got {features.shape} and {values.shape}"
            )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(values)):
            raise ValueError("X and scores must contain only finite values")

        split = min(max(int(features.shape[0] * 0.8), 1), features.shape[0] - 1)
        fit_X, validation_X = features[:split], features[split:]
        fit_scores, validation_scores = values[:split], values[split:]
        self._x_mean = fit_X.mean(axis=0)
        x_std = (
            fit_X.std(axis=0, ddof=1)
            if fit_X.shape[0] > 1
            else np.zeros(fit_X.shape[1], dtype=np.float64)
        )
        self._x_scale = np.where(
            np.isfinite(x_std) & (x_std > 1e-7),
            x_std,
            1.0,
        )
        self._score_mean = float(fit_scores.mean())
        score_std = float(fit_scores.std(ddof=1)) if fit_scores.size > 1 else 0.0
        self._score_scale = (
            score_std if np.isfinite(score_std) and score_std > 1e-7 else 1.0
        )

        train_X = torch.as_tensor(
            self._standardize_features(fit_X),
            dtype=torch.float32,
        )
        val_X = torch.as_tensor(
            self._standardize_features(validation_X),
            dtype=torch.float32,
        )
        train_y = torch.as_tensor(
            (fit_scores - self._score_mean) / self._score_scale,
            dtype=torch.float32,
        ).reshape(-1, 1)
        val_y = torch.as_tensor(
            (validation_scores - self._score_mean) / self._score_scale,
            dtype=torch.float32,
        ).reshape(-1, 1)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._random_state)
            network = self._new_network(features.shape[1])
            optimizer = torch.optim.AdamW(
                network.parameters(),
                lr=1e-3,
                weight_decay=1e-5,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
            )
            level = 1.0 - self._alpha
            best_loss = math.inf
            best_state: dict[str, torch.Tensor] | None = None
            stale_epochs = 0
            for epoch in range(self._max_epochs):
                network.train()
                order = torch.randperm(train_X.shape[0])
                size = min(self._batch_size, train_X.shape[0])
                for start in range(0, train_X.shape[0], size):
                    batch = order[start : start + size]
                    optimizer.zero_grad()
                    loss = self._pinball(
                        network(train_X[batch]),
                        train_y[batch],
                        level=level,
                    )
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()

                network.eval()
                with torch.no_grad():
                    val_loss = float(
                        self._pinball(
                            network(val_X),
                            val_y,
                            level=level,
                        ).item()
                    )
                if not np.isfinite(val_loss):
                    raise FloatingPointError(
                        "RCP quantile training produced a non-finite validation loss"
                    )
                if (epoch + 1) % 2 == 0:
                    scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(network.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= self._patience:
                    break

        if best_state is None:
            raise RuntimeError("RCP quantile training produced no checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        self._network = network
        return self

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return finite strictly positive conditional quantile estimates."""
        if self._score_mean is None or self._score_scale is None:
            raise RuntimeError("ScoreQuantileEstimator.fit must be called first")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("X must be a finite two-dimensional feature matrix")
        inputs = torch.as_tensor(
            self._standardize_features(features),
            dtype=torch.float32,
        )
        with torch.no_grad():
            standardized = self.network_(inputs).squeeze(-1).numpy()
        raw = np.asarray(
            standardized * self._score_scale + self._score_mean,
            dtype=np.float64,
        )
        return np.maximum(raw, self._floor)

"""Conditional scale network ``|g(x)|`` for NFCP.

Up to ten PCA components of the features feed five hidden ReLU layers of width 100.
Adam, at learning rate ``1e-4`` on minibatches of one fifth of the training and
selection rows, minimizes the Gaussian negative log-likelihood of the NFCP
transformation. Training is deterministic given ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import nn

FLOW_GAMMA = 1e-3
FLOW_RESIDUAL_FLOOR = 1e-4
FLOW_PCA_COMPONENTS = 10
FLOW_DEFAULT_UPDATES = 3001


def flow_response_scale(y: NDArray[np.float64]) -> float:
    """Return the response scale ``|1e-4 + max(y)|`` of the training targets."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("y must be a non-empty finite vector")
    scale = float(abs(FLOW_RESIDUAL_FLOOR + np.max(values)))
    return scale if scale > 1e-12 else 1.0


def gaussian_flow_objective(
    residuals: torch.Tensor,
    raw_scale: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Return the Gaussian negative log-likelihood term of the NFCP transformation.

    The change-of-variables Jacobian is ``1/A`` and has no trainable
    parameter, leaving the mean squared transformed score.

    Args:
        residuals: Strictly positive residuals, floor included.
        raw_scale: Non-negative network values ``|g(x)|``.
        gamma: Positive denominator regularizer.

    Returns:
        Scalar mean of ``log(A / (gamma + |g(x)|)) ** 2``.
    """
    transformed = torch.log(residuals / (gamma + raw_scale))
    return torch.mean(transformed**2)


@dataclass(frozen=True, slots=True)
class FlowUpdateSelection:
    """Number of updates selected on the selection split, with its diagnostics."""

    selected_updates: int
    batch_size: int
    best_validation_loss: float
    validation_losses: tuple[float, ...]
    max_training_loss: float
    source_restart_threshold_crossed: bool


class FlowScaleEstimator:
    """Learn the non-negative conditional residual scale ``|g(x)|``.

    Args:
        hidden_size: Units per hidden layer.
        hidden_layers: Number of hidden ReLU layers.
        updates: Number of Adam updates among which the selection split chooses.
        learning_rate: Adam step size.
        gamma: Positive denominator regularizer.
        residual_floor: Positive value added to absolute residuals.
        random_state: Model-owned seed for initialization and minibatches.

    Raises:
        ValueError: If a hyperparameter is outside its admissible domain.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 100,
        hidden_layers: int = 5,
        updates: int = FLOW_DEFAULT_UPDATES,
        learning_rate: float = 1e-4,
        gamma: float = FLOW_GAMMA,
        residual_floor: float = FLOW_RESIDUAL_FLOOR,
        random_state: int = 0,
    ) -> None:
        if hidden_size < 1 or hidden_layers < 1:
            raise ValueError("hidden_size and hidden_layers must be positive")
        if updates < 0:
            raise ValueError(f"updates must be non-negative, got {updates}")
        if learning_rate <= 0.0 or gamma <= 0.0 or residual_floor <= 0.0:
            raise ValueError(
                "learning_rate, gamma, and residual_floor must be positive"
            )
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._updates = updates
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._residual_floor = residual_floor
        self._random_state = random_state
        self._network: nn.Sequential | None = None
        self._feature_scale: NDArray[np.float64] | None = None
        self._pca: PCA | None = None
        self.selection_: FlowUpdateSelection | None = None

    @property
    def network_(self) -> nn.Sequential:
        """Return the fitted network, or raise before fitting."""
        if self._network is None:
            raise RuntimeError("fit must be called before accessing network_")
        return self._network

    @staticmethod
    def _validate_arrays(
        X: NDArray[np.float64],
        residuals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        features = np.asarray(X, dtype=np.float64)
        errors = np.asarray(residuals, dtype=np.float64)
        invalid_shape = (
            features.ndim != 2
            or errors.ndim != 1
            or features.shape[0] != errors.shape[0]
            or features.shape[0] == 0
            or features.shape[1] == 0
        )
        if invalid_shape:
            raise ValueError(
                "X and residuals must have shapes (n, d) and (n,) with n,d > 0, "
                f"got {features.shape} and {errors.shape}"
            )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(errors)):
            raise ValueError("X and residuals must contain only finite values")
        if np.any(errors < 0.0):
            raise ValueError("residuals must be non-negative")
        return features, errors

    def _fit_preprocessor(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        denominators = np.abs(self._residual_floor + np.max(X, axis=0))
        self._feature_scale = np.where(denominators > 1e-12, denominators, 1.0)
        scaled = X / self._feature_scale
        components = min(FLOW_PCA_COMPONENTS, scaled.shape[0], scaled.shape[1])
        self._pca = PCA(n_components=components, svd_solver="full")
        return np.asarray(self._pca.fit_transform(scaled), dtype=np.float64)

    def _transform_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._feature_scale is None or self._pca is None:
            raise RuntimeError("flow feature preprocessor has not been fitted")
        return np.asarray(
            self._pca.transform(X / self._feature_scale),
            dtype=np.float64,
        )

    def _new_network(self, input_dim: int) -> nn.Sequential:
        modules: list[nn.Module] = []
        width = input_dim
        for _ in range(self._hidden_layers):
            modules.extend([nn.Linear(width, self._hidden_size), nn.ReLU()])
            width = self._hidden_size
        modules.append(nn.Linear(width, 1))
        return nn.Sequential(*modules)

    def _train_network(
        self,
        train_X: torch.Tensor,
        train_residuals: torch.Tensor,
        *,
        updates: int,
        seed: int,
        batch_size: int,
        validation_X: torch.Tensor | None = None,
        validation_residuals: torch.Tensor | None = None,
    ) -> tuple[nn.Sequential, tuple[float, ...], float, bool]:
        validation_losses: list[float] = []
        max_training_loss = 0.0
        source_restart_threshold_crossed = False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = self._new_network(train_X.shape[1])
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=self._learning_rate,
            )
            for _ in range(updates):
                batch = torch.randperm(train_X.shape[0])[:batch_size]
                optimizer.zero_grad()
                loss = gaussian_flow_objective(
                    train_residuals[batch],
                    network(train_X[batch]).abs(),
                    gamma=self._gamma,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("flow training produced a non-finite loss")
                training_value = float(loss.item())
                max_training_loss = max(max_training_loss, training_value)
                source_restart_threshold_crossed |= training_value > 10.0
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

                if validation_X is not None and validation_residuals is not None:
                    with torch.no_grad():
                        validation_loss = gaussian_flow_objective(
                            validation_residuals,
                            network(validation_X).abs(),
                            gamma=self._gamma,
                        )
                    value = float(validation_loss.item())
                    if not np.isfinite(value):
                        raise FloatingPointError(
                            "flow validation produced a non-finite loss"
                        )
                    source_restart_threshold_crossed |= value > 10.0
                    validation_losses.append(value)
        network.eval()
        return (
            network,
            tuple(validation_losses),
            max_training_loss,
            source_restart_threshold_crossed,
        )

    def fit_with_validation(
        self,
        train_X: NDArray[np.float64],
        train_residuals: NDArray[np.float64],
        validation_X: NDArray[np.float64],
        validation_residuals: NDArray[np.float64],
    ) -> FlowScaleEstimator:
        """Select the number of updates on the selection split, then refit on training.

        ``train_residuals`` must be out-of-fold residuals on the training split.
        The selection rows, ``validation_X`` and ``validation_residuals``, only score
        each update. The feature preprocessing is fitted on training features.
        """
        if self._updates < 1:
            raise ValueError("validation selection requires at least one update")
        features, errors = self._validate_arrays(train_X, train_residuals)
        validation_features, validation_errors = self._validate_arrays(
            validation_X,
            validation_residuals,
        )
        if validation_features.shape[1] != features.shape[1]:
            raise ValueError("training and validation feature dimensions must match")
        transformed_train = torch.as_tensor(
            self._fit_preprocessor(features),
            dtype=torch.float32,
        )
        transformed_validation = torch.as_tensor(
            self._transform_features(validation_features),
            dtype=torch.float32,
        )
        training_tensor = torch.as_tensor(
            errors + self._residual_floor,
            dtype=torch.float32,
        ).reshape(-1, 1)
        validation_tensor = torch.as_tensor(
            validation_errors + self._residual_floor,
            dtype=torch.float32,
        ).reshape(-1, 1)
        batch_size = max(1, (len(features) + len(validation_features)) // 5)
        _, losses, max_loss, crossed = self._train_network(
            transformed_train,
            training_tensor,
            updates=self._updates,
            seed=self._random_state,
            batch_size=batch_size,
            validation_X=transformed_validation,
            validation_residuals=validation_tensor,
        )
        selected_index = int(np.argmin(np.asarray(losses, dtype=np.float64)))
        selected_updates = selected_index + 1
        network, _, _, _ = self._train_network(
            transformed_train,
            training_tensor,
            updates=selected_updates,
            seed=self._random_state,
            batch_size=batch_size,
        )
        self._network = network
        self.selection_ = FlowUpdateSelection(
            selected_updates=selected_updates,
            batch_size=batch_size,
            best_validation_loss=losses[selected_index],
            validation_losses=losses,
            max_training_loss=max_loss,
            source_restart_threshold_crossed=crossed,
        )
        return self

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the fitted non-negative ``|g(x)|`` scale."""
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("X must be a finite two-dimensional feature matrix")
        transformed = torch.as_tensor(
            self._transform_features(features),
            dtype=torch.float32,
        )
        with torch.no_grad():
            raw = self.network_(transformed).abs().squeeze(-1).numpy()
        return np.asarray(raw, dtype=np.float64)


class ScaledFlowScale:
    """Convert a normalized NFCP scale back to response units."""

    def __init__(self, estimator: FlowScaleEstimator, response_scale: float) -> None:
        if not np.isfinite(response_scale) or response_scale <= 0.0:
            raise ValueError("response_scale must be finite and positive")
        self._estimator = estimator
        self._response_scale = response_scale

    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the fitted scale in the original response units."""
        return np.asarray(
            self._response_scale * self._estimator(X),
            dtype=np.float64,
        )

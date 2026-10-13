"""Image classifier and probability cache."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


class ProbabilityCache:
    """Map prepared embeddings to their corresponding base-model probabilities."""

    def __init__(
        self,
        features: NDArray[np.floating],
        probabilities: NDArray[np.floating],
    ) -> None:
        X = np.ascontiguousarray(features, dtype=np.float64)
        values = np.asarray(probabilities, dtype=np.float32)
        if X.ndim != 2 or values.ndim != 2 or len(X) != len(values):
            raise ValueError("features and probabilities must be aligned matrices")
        totals = values.sum(axis=1, keepdims=True)
        if (
            not np.all(np.isfinite(X))
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not np.allclose(totals, 1.0, atol=5e-4)
        ):
            raise ValueError(
                "probability rows must be finite, non-negative, and sum to one"
            )
        values = values / totals
        self._n_features = X.shape[1]
        kept: list[NDArray[np.float32]] = []
        self._lookup: dict[bytes, int] = {}
        for row, probability in zip(X, values, strict=True):
            key = row.tobytes()
            previous = self._lookup.get(key)
            if previous is not None and not np.array_equal(kept[previous], probability):
                raise ValueError("duplicate features have different probabilities")
            if previous is None:
                self._lookup[key] = len(kept)
                kept.append(probability.copy())
        self._probabilities = np.asarray(kept, dtype=np.float32)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        features = np.ascontiguousarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self._n_features:
            raise ValueError(f"X must have shape (n, {self._n_features})")
        try:
            indices = [self._lookup[row.tobytes()] for row in features]
        except KeyError as error:
            raise ValueError("row is outside the prepared probability cache") from error
        return np.asarray(self._probabilities[indices], dtype=np.float64)


@contextmanager
def _isolated_torch_rng(seed: int) -> Iterator[None]:
    """Seed model work without advancing the caller's Torch RNG streams."""
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    mps_state = (
        torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
                yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class ResNetModel:
    """Fine-tuned ResNet-18 returning softmax probabilities and embeddings."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        epochs: int = 5,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        input_channels: int = 3,
        image_height: int = 32,
        image_width: int = 32,
        pretrained: bool = True,
        small_input_stem: bool = False,
        device: str,
        random_seed: int = 42,
    ) -> None:
        if (
            min(
                num_classes,
                epochs,
                batch_size,
                input_channels,
                image_height,
                image_width,
            )
            < 1
        ):
            raise ValueError(
                "class count, dimensions, epochs, and batch size must be positive"
            )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.random_seed = random_seed
        self.device = torch.device(device)

        with _isolated_torch_rng(random_seed):
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            network = models.resnet18(weights=weights)
            if input_channels != 3 or small_input_stem:
                convolution = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=3 if small_input_stem else 7,
                    stride=1 if small_input_stem else 2,
                    padding=1 if small_input_stem else 3,
                    bias=False,
                )
                if pretrained:
                    self._adapt_stem(convolution, network.conv1.weight)
                network.conv1 = convolution
            if small_input_stem:
                network.maxpool = nn.Identity()
            network.fc = nn.Linear(512, num_classes)
        self._model = network.to(self.device)
        self._channel_mean: NDArray[np.float32] | None = None
        self._channel_scale: NDArray[np.float32] | None = None
        self._is_fitted = False

    @staticmethod
    def _adapt_stem(target: nn.Conv2d, source: torch.Tensor) -> None:
        weights = source.detach()
        if target.in_channels == 1:
            weights = weights.mean(dim=1, keepdim=True)
        elif target.in_channels != 3:
            weights = weights.mean(dim=1, keepdim=True).repeat(
                1, target.in_channels, 1, 1
            )
            weights *= 3.0 / target.in_channels
        height, width = target.kernel_size
        start_h = (weights.shape[-2] - height) // 2
        start_w = (weights.shape[-1] - width) // 2
        weights = weights[:, :, start_h : start_h + height, start_w : start_w + width]
        if weights.shape != target.weight.shape:
            raise ValueError("pretrained stem cannot be adapted to the requested input")
        with torch.no_grad():
            target.weight.copy_(weights)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self.image_height, self.image_width, self.input_channels

    def _validate_images(self, X: NDArray[np.floating]) -> NDArray[np.float32]:
        images = np.asarray(X, dtype=np.float32)
        if images.ndim != 4 or images.shape[1:] != self.image_shape:
            raise ValueError(f"images must have shape (n, {self.image_shape})")
        if not np.all(np.isfinite(images)) or np.any((images < 0.0) | (images > 1.0)):
            raise ValueError("images must contain finite values in [0, 1]")
        return images

    def _tensor(self, X: NDArray[np.floating]) -> torch.Tensor:
        images = self._validate_images(X)
        if self._channel_mean is None or self._channel_scale is None:
            raise RuntimeError("fit must be called before prediction")
        normalized = (images - self._channel_mean) / self._channel_scale
        return torch.from_numpy(
            np.ascontiguousarray(np.transpose(normalized, (0, 3, 1, 2)))
        )

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> None:
        images = self._validate_images(X)
        labels = np.asarray(y, dtype=np.int64)
        if labels.shape != (len(images),):
            raise ValueError("images and labels must have the same number of rows")
        self._channel_mean = images.mean(axis=(0, 1, 2), dtype=np.float64).astype(
            np.float32
        )
        scale = images.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
        self._channel_scale = np.maximum(scale, np.float32(1e-6))
        dataset = TensorDataset(self._tensor(images), torch.from_numpy(labels))
        generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        with _isolated_torch_rng(self.random_seed):
            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            self._model.train()
            for _ in range(self.epochs):
                for batch_X, batch_y in loader:
                    logits = self._model(batch_X.to(self.device))
                    loss = nn.functional.cross_entropy(logits, batch_y.to(self.device))
                    optimizer.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
        self._is_fitted = True

    def _embeddings(self, images: torch.Tensor) -> torch.Tensor:
        model = self._model
        values = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        values = model.layer4(model.layer3(model.layer2(model.layer1(values))))
        return torch.flatten(model.avgpool(values), 1)

    def predict_with_embeddings(
        self,
        X: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before prediction")
        tensor = self._tensor(X)
        probabilities, embeddings = [], []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                features = self._embeddings(
                    tensor[start : start + self.batch_size].to(self.device)
                )
                embeddings.append(features.cpu().numpy().astype(np.float32))
                probabilities.append(
                    torch.softmax(self._model.fc(features), dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        return np.concatenate(probabilities), np.concatenate(embeddings)


__all__ = ("ProbabilityCache", "ResNetModel")

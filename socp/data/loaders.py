"""Load the nine public datasets from the folders that ``make download`` fills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing, fetch_covtype
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

Features = NDArray[np.float64]
Targets = NDArray[np.float64]
Labels = NDArray[np.int64]

DATA_DIR = Path("data")
CSV_FILES = {
    "concrete": "Concrete_Data.csv",
    "bike": "bike_train.csv",
    "bio": "CASP.csv",
    "mpg": "data_mpg.csv",
}


def _read_csv(dataset: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / CSV_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `make download`")
    return pd.read_csv(path)


def _csv(
    dataset: str,
    *,
    target_column: int,
    clean_numeric: bool = False,
) -> tuple[Features, Targets]:
    frame = _read_csv(dataset)
    if clean_numeric:
        frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0)
    values = frame.to_numpy(dtype=np.float64)
    if target_column == 0:
        return values[:, 1:], values[:, 0]
    return values[:, :-1], values[:, -1]


def load_california_housing() -> tuple[Features, Targets]:
    data = fetch_california_housing(data_home=str(DATA_DIR / "sklearn"))
    return (
        np.asarray(data.data, dtype=np.float64),
        np.asarray(data.target, dtype=np.float64),
    )


def load_concrete() -> tuple[Features, Targets]:
    return _csv("concrete", target_column=-1)


def load_bio() -> tuple[Features, Targets]:
    return _csv("bio", target_column=0)


def load_mpg() -> tuple[Features, Targets]:
    return _csv("mpg", target_column=0, clean_numeric=True)


def load_bike() -> tuple[Features, Targets]:
    frame = _read_csv("bike")
    frame = pd.concat(
        [
            frame.drop(columns=["season", "weather"]),
            pd.get_dummies(frame["season"], prefix="season", dtype=float),
            pd.get_dummies(frame["weather"], prefix="weather", dtype=float),
        ],
        axis=1,
    )
    timestamps = pd.DatetimeIndex(frame["datetime"])
    frame["hour"] = timestamps.hour
    frame["day"] = timestamps.dayofweek
    frame["month"] = timestamps.month
    frame["year"] = timestamps.year - 2011
    frame = frame.drop(columns=["datetime", "casual", "registered"])
    return (
        frame.drop(columns=["count"]).to_numpy(dtype=np.float64),
        frame["count"].to_numpy(dtype=np.float64),
    )


def load_tabular_regression_dataset(dataset: str) -> tuple[Features, Targets]:
    loaders = {
        "california_housing": load_california_housing,
        "concrete": load_concrete,
        "bike": load_bike,
        "bio": load_bio,
        "mpg": load_mpg,
    }
    if dataset not in loaders:
        raise ValueError(f"unknown tabular regression dataset: {dataset!r}")
    return loaders[dataset]()


def load_covertype() -> tuple[Features, Labels]:
    data = fetch_covtype(data_home=str(DATA_DIR / "sklearn"), download_if_missing=True)
    return (
        np.asarray(data.data, dtype=np.float64),
        np.asarray(data.target, dtype=np.int64) - 1,
    )


def load_tabular_classification_dataset(dataset: str) -> tuple[Features, Labels]:
    if dataset == "covertype":
        return load_covertype()
    raise ValueError(f"unknown tabular classification dataset: {dataset!r}")


def _torchvision_arrays(dataset: Any) -> tuple[NDArray[np.float32], Labels]:
    images = np.asarray(dataset.data, dtype=np.float32)
    if images.ndim == 3:
        images = images[..., None]
    return images / 255.0, np.asarray(dataset.targets, dtype=np.int64)


def _torchvision_splits(
    dataset_class: Any,
) -> tuple[NDArray[np.float32], Labels, NDArray[np.float32], Labels]:
    train_X, train_y = _torchvision_arrays(
        dataset_class(root=str(DATA_DIR), train=True, download=True)
    )
    test_X, test_y = _torchvision_arrays(
        dataset_class(root=str(DATA_DIR), train=False, download=True)
    )
    return train_X, train_y, test_X, test_y


def load_cifar10() -> tuple[NDArray[np.float32], Labels, NDArray[np.float32], Labels]:
    return _torchvision_splits(CIFAR10)


def load_cifar100() -> tuple[NDArray[np.float32], Labels, NDArray[np.float32], Labels]:
    return _torchvision_splits(CIFAR100)


def load_mnist() -> tuple[NDArray[np.float32], Labels, NDArray[np.float32], Labels]:
    return _torchvision_splits(MNIST)


def load_image_classification_dataset(
    dataset: str,
) -> tuple[NDArray[np.float32], Labels, NDArray[np.float32], Labels]:
    loaders = {"cifar10": load_cifar10, "cifar100": load_cifar100, "mnist": load_mnist}
    if dataset not in loaders:
        raise ValueError(f"unknown image dataset: {dataset!r}")
    return loaders[dataset]()


__all__ = (
    "load_bike",
    "load_bio",
    "load_california_housing",
    "load_cifar10",
    "load_cifar100",
    "load_concrete",
    "load_covertype",
    "load_image_classification_dataset",
    "load_mnist",
    "load_mpg",
    "load_tabular_classification_dataset",
    "load_tabular_regression_dataset",
)

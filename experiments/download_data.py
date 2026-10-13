"""Download the public datasets used by the full experiment profile."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DATASETS = (
    "bike",
    "bio",
    "california_housing",
    "concrete",
    "mpg",
    "cifar10",
    "mnist",
    "covertype",
    "cifar100",
)

_CQR = "https://raw.githubusercontent.com/yromano/cqr/master/datasets"
_CSV = {
    "bike": (f"{_CQR}/bike_train.csv", RAW_DIR / "bike_train.csv"),
    "bio": (f"{_CQR}/CASP.csv", RAW_DIR / "CASP.csv"),
    "concrete": (f"{_CQR}/Concrete_Data.csv", RAW_DIR / "Concrete_Data.csv"),
}
_MPG_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _download_csv(dataset: str, force: bool) -> None:
    url, path = _CSV[dataset]
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} -> {path}")
    urllib.request.urlretrieve(url, path)


def _download_mpg(force: bool) -> None:
    path = RAW_DIR / "data_mpg.csv"
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    )
    data = pd.read_csv(
        _MPG_URL,
        sep=r"\s+",
        header=None,
        names=columns,
        na_values="?",
    )
    data.drop(columns="car_name").dropna().to_csv(path, index=False)


def _download_torchvision(dataset: str, force: bool) -> None:
    classes = {"cifar10": CIFAR10, "cifar100": CIFAR100, "mnist": MNIST}
    folders = {
        "cifar10": "cifar-10-batches-py",
        "cifar100": "cifar-100-python",
        "mnist": "MNIST",
    }
    if force:
        _remove(DATA_DIR / folders[dataset])
    dataset_class = classes[dataset]
    dataset_class(root=str(DATA_DIR), train=True, download=True)
    dataset_class(root=str(DATA_DIR), train=False, download=True)


def download(dataset: str, *, force: bool = False) -> None:
    """Download one public dataset to the locations used by the loaders."""
    if dataset in _CSV:
        _download_csv(dataset, force)
    elif dataset == "mpg":
        _download_mpg(force)
    elif dataset in {"cifar10", "cifar100", "mnist"}:
        _download_torchvision(dataset, force)
    elif dataset == "covertype":
        fetch_covtype(data_home=str(DATA_DIR / "sklearn"), download_if_missing=True)
    elif dataset == "california_housing":
        fetch_california_housing(data_home=str(DATA_DIR / "sklearn"))
    else:
        raise ValueError(f"unknown public dataset: {dataset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "all"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = PUBLIC_DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in targets:
        print(f"Preparing {dataset}")
        download(dataset, force=args.force)


if __name__ == "__main__":
    main()

"""Prepare deterministic data splits and fitted predictors."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler

from experiments.campaign import (
    DEFAULT_CONFIG,
    Campaign,
    experiment_device,
    load_campaign,
)
from socp.data.loaders import (
    load_image_classification_dataset,
    load_tabular_classification_dataset,
    load_tabular_regression_dataset,
)
from socp.models.classification import ProbabilityCache, ResNetModel
from socp.models.flow import (
    FlowScaleEstimator,
    ScaledFlowScale,
    flow_response_scale,
)
from socp.models.quantile_histogram import QuantileForestHistogram
from socp.models.score_quantile import ScoreQuantileEstimator


class _Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> _Regressor: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def run_directory(root: Path, dataset: str, seed: int) -> Path:
    return root / dataset / f"seed_{seed}"


def _integer_split(config: dict[str, Any]) -> tuple[int, int, int, int]:
    split = config["split"]
    counts = (
        int(split["train"]),
        int(split["selection"]),
        int(split["calibration"]),
        int(split["test"]),
    )
    if any(count < 1 for count in counts):
        raise ValueError("synthetic split counts must be positive")
    return counts[0], counts[1], counts[2], counts[3]


def _fractional_split(
    n_rows: int,
    split: dict[str, Any],
) -> tuple[int, int, int, int]:
    fractions = tuple(
        float(split[name])
        for name in (
            "train_fraction",
            "selection_fraction",
            "calibration_fraction",
            "test_fraction",
        )
    )
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(
            "train, selection, calibration, and test fractions must sum to one"
        )
    n_train = int(n_rows * fractions[0])
    n_selection = int(n_rows * fractions[1])
    n_calibration = int(n_rows * fractions[2])
    n_test = n_rows - n_train - n_selection - n_calibration
    if min(n_train, n_selection, n_calibration, n_test) < 1:
        raise ValueError("every split must contain at least one observation")
    return n_train, n_selection, n_calibration, n_test


def _indices(
    n_rows: int,
    counts: tuple[int, int, int, int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_rows)
    boundaries = np.cumsum((0, *counts))
    blocks = tuple(
        np.asarray(order[boundaries[index] : boundaries[index + 1]], dtype=np.int64)
        for index in range(4)
    )
    return blocks[0], blocks[1], blocks[2], blocks[3]


def cross_fitted_absolute_residuals(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_factory: Callable[[], _Regressor],
) -> np.ndarray:
    """Return train-only absolute residuals from ten held-out contiguous folds."""
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    indices = np.arange(len(features))
    requested = min(10, len(indices))
    chunk_size = (len(indices) + requested - 1) // requested
    partitions = tuple(
        indices[start : start + chunk_size]
        for start in range(0, len(indices), chunk_size)
    )
    residuals = np.empty(len(indices), dtype=np.float64)
    for fold_index, held_out in enumerate(partitions):
        fit_indices = np.concatenate(
            [fold for index, fold in enumerate(partitions) if index != fold_index]
        )
        model = model_factory().fit(features[fit_indices], targets[fit_indices])
        residuals[held_out] = np.abs(
            targets[held_out] - np.asarray(model.predict(features[held_out]))
        )
    return residuals


def _synthetic(
    dataset: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    counts = _integer_split(config)
    if config["task"] == "regression":
        X, y = make_regression(
            n_samples=sum(counts),
            n_features=8,
            n_informative=6,
            noise=20.0,
            random_state=seed,
        )
    elif config["task"] == "classification":
        X, y = make_classification(
            n_samples=sum(counts),
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_classes=int(config["classes"]),
            n_clusters_per_class=1,
            class_sep=1.2,
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported synthetic task for {dataset}")
    return np.asarray(X, dtype=np.float64), np.asarray(y), counts


def _tabular(
    dataset: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    task = config["task"]
    loaded: tuple[np.ndarray, np.ndarray]
    if task == "regression":
        loaded = load_tabular_regression_dataset(dataset)
    elif task == "classification":
        loaded = load_tabular_classification_dataset(dataset)
    else:
        raise ValueError(f"unsupported task for {dataset}: {task!r}")
    X, y = loaded
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y)
    sample_size = int(config.get("sample_size", len(features)))
    if sample_size < len(features):
        selected = np.random.default_rng(seed).permutation(len(features))[:sample_size]
        features = features[selected]
        targets = targets[selected]
    return features, targets, _fractional_split(len(features), config["split"])


def _regression_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_selection: np.ndarray,
    y_selection: np.ndarray,
    *,
    alpha: float,
    seed: int,
    profile: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    common = settings["gradient_boosting"]
    n_estimators = int(common["n_estimators"])
    max_depth = int(common["max_depth"])
    learning_rate = float(common["learning_rate"])
    if profile == "smoke":
        n_estimators = min(n_estimators, 60)
        max_depth = min(max_depth, 3)

    def regressor(
        *, loss: str = "squared_error", quantile: float | None = None
    ) -> GradientBoostingRegressor:
        arguments: dict[str, Any] = {
            "loss": loss,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": seed,
        }
        if quantile is not None:
            arguments["alpha"] = quantile
        return GradientBoostingRegressor(**arguments)

    point = regressor().fit(X_train, y_train)
    models: dict[str, Any] = {"point": point}
    if profile == "smoke":
        return models

    models["lower"] = regressor(loss="quantile", quantile=alpha / 2).fit(
        X_train,
        y_train,
    )
    models["upper"] = regressor(loss="quantile", quantile=1 - alpha / 2).fit(
        X_train,
        y_train,
    )
    chr_settings = settings["chr"]
    models["histogram"] = QuantileForestHistogram(
        n_bins=int(chr_settings["n_bins"]),
        n_estimators=int(chr_settings["n_estimators"]),
        min_samples_leaf=int(chr_settings["min_samples_leaf"]),
        random_state=seed,
    ).fit(X_train, y_train)

    training_residuals = cross_fitted_absolute_residuals(
        X_train,
        y_train,
        model_factory=regressor,
    )
    validation_residuals = np.abs(y_selection - point.predict(X_selection))
    response_scale = flow_response_scale(y_train)
    flow_settings = settings["flow"]
    flow_estimator = FlowScaleEstimator(
        hidden_size=int(flow_settings["hidden_size"]),
        hidden_layers=int(flow_settings["hidden_layers"]),
        updates=int(flow_settings["updates"]),
        random_state=seed,
    ).fit_with_validation(
        X_train,
        training_residuals / response_scale,
        X_selection,
        validation_residuals / response_scale,
    )
    models["flow_scale"] = ScaledFlowScale(flow_estimator, response_scale)
    models["flow_response_scale"] = response_scale
    rcp_settings = settings["rcp"]
    models["rcp_scale"] = ScoreQuantileEstimator(
        alpha=alpha,
        random_state=seed,
        hidden_size=int(rcp_settings["hidden_size"]),
        max_epochs=int(rcp_settings["max_epochs"]),
        patience=int(rcp_settings["patience"]),
    ).fit(X_train, training_residuals)
    return models


def _classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    profile: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    model_settings = settings["histogram_gradient_boosting"]
    iterations = int(model_settings["max_iter"])
    if profile == "smoke":
        iterations = min(iterations, 60)
    model = HistGradientBoostingClassifier(
        max_iter=iterations,
        max_leaf_nodes=int(model_settings["max_leaf_nodes"]),
        learning_rate=float(model_settings["learning_rate"]),
        early_stopping=model_settings["early_stopping"],
        random_state=seed,
    ).fit(X_train, np.asarray(y_train, dtype=np.int64))
    return {"classifier": model}


def _reduce_embeddings(
    embeddings: dict[str, np.ndarray],
    *,
    dimension: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit PCA and standardization on training embeddings only."""
    train = np.asarray(embeddings["train"], dtype=np.float64)
    components = min(dimension, train.shape[0], train.shape[1])
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=seed,
    ).fit(train)
    projected = {
        name: np.asarray(pca.transform(values), dtype=np.float64)
        for name, values in embeddings.items()
    }
    scaler = StandardScaler().fit(projected["train"])
    return {
        name: np.asarray(scaler.transform(values), dtype=np.float64)
        for name, values in projected.items()
    }


def _probability_cache(
    features: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> ProbabilityCache:
    names = ("train", "selection", "calibration", "test")
    return ProbabilityCache(
        np.concatenate([features[name] for name in names]),
        np.concatenate([probabilities[name] for name in names]),
    )


def _effective_model_settings(
    settings: dict[str, Any],
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    """Return the shared model settings with the dataset's predictor overrides."""
    resolved = dict(settings)
    model_id = str(dataset_config["model"])
    common = settings.get(model_id, {})
    override = dataset_config.get("predictor", {})
    if not isinstance(common, dict) or not isinstance(override, dict):
        raise ValueError("predictor settings must be mappings")
    resolved[model_id] = {**common, **override}
    return resolved


def _imagenet_indices(
    split: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four disjoint seeded views of the 50,000 ImageNet rows."""
    counts: tuple[int, int, int, int] = (
        int(split["train"]),
        int(split["selection"]),
        int(split["calibration"]),
        int(split["test"]),
    )
    if any(count < 1 for count in counts) or sum(counts) != 50_000:
        raise ValueError(
            "ImageNet split must contain four positive blocks totaling 50,000"
        )
    return _indices(50_000, counts, seed=seed)


def _image_prepared(
    dataset: str,
    config: dict[str, Any],
    seed: int,
    settings: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    train_images, train_labels, test_images, test_labels = (
        load_image_classification_dataset(dataset)
    )
    split = config["split"]
    n_train = int(len(train_images) * float(split["train_fraction"]))
    n_selection = int(len(train_images) * float(split["selection_fraction"]))
    order = np.random.default_rng(seed).permutation(len(train_images))
    train_ids = order[:n_train]
    selection_ids = order[n_train : n_train + n_selection]
    calibration_ids = order[n_train + n_selection :]
    test_ids = len(train_images) + np.arange(len(test_images), dtype=np.int64)

    vision = settings["vision"]
    channels = 1 if dataset == "mnist" else 3
    height = 28 if dataset == "mnist" else 32
    model = ResNetModel(
        num_classes=int(config["classes"]),
        epochs=int(vision["epochs"]),
        batch_size=int(vision["batch_size"]),
        learning_rate=float(vision["learning_rate"]),
        input_channels=channels,
        image_height=height,
        image_width=height,
        pretrained=True,
        small_input_stem=dataset == "mnist",
        random_seed=seed,
        device=experiment_device(),
    )
    model.fit(train_images[train_ids], train_labels[train_ids])

    image_splits = {
        "train": train_images[train_ids],
        "selection": train_images[selection_ids],
        "calibration": train_images[calibration_ids],
        "test": test_images,
    }
    probabilities: dict[str, np.ndarray] = {}
    embeddings: dict[str, np.ndarray] = {}
    for name, images in image_splits.items():
        probability, embedding = model.predict_with_embeddings(images)
        probabilities[name] = probability
        embeddings[name] = embedding
    features = _reduce_embeddings(
        embeddings,
        dimension=int(vision["embedding_dim"]),
        seed=seed,
    )
    labels = {
        "train": np.asarray(train_labels[train_ids], dtype=np.int64),
        "selection": np.asarray(train_labels[selection_ids], dtype=np.int64),
        "calibration": np.asarray(train_labels[calibration_ids], dtype=np.int64),
        "test": np.asarray(test_labels, dtype=np.int64),
    }
    row_ids = {
        "train": np.asarray(train_ids, dtype=np.int64),
        "selection": np.asarray(selection_ids, dtype=np.int64),
        "calibration": np.asarray(calibration_ids, dtype=np.int64),
        "test": test_ids,
    }
    arrays = {
        key: value
        for name in ("train", "selection", "calibration", "test")
        for key, value in (
            (f"X_{name}", features[name]),
            (f"y_{name}", labels[name]),
            (f"row_id_{name}", row_ids[name]),
        )
    }
    accuracy = {
        name: float(np.mean(np.argmax(probabilities[name], axis=1) == labels[name]))
        for name in probabilities
    }
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "task": "classification",
        "source": config["source"],
        "model": config["model"],
        "representation": f"train_PCA{features['train'].shape[1]}_standardized",
        "split_sizes": {name: len(labels[name]) for name in labels},
        "model_accuracy": accuracy,
        "predictor_hyperparameters": dict(vision),
        "preprocessing": (
            "train-channel normalization; train-fitted PCA and standardization"
        ),
    }
    models = {"classifier": _probability_cache(features, probabilities)}
    return arrays, models, metadata


def _imagenet_prepared(
    config: dict[str, Any],
    seed: int,
    settings: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    cache_path = Path(
        config.get(
            "cache",
            "data/raw/imagenet_val_cache.npz",
        )
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"ImageNet cache not found at {cache_path}; "
            "run experiments/scripts/imagenet_cache.py first"
        )
    with np.load(cache_path) as cache:
        probabilities_all = np.asarray(cache["probs"], dtype=np.float32)
        embeddings_all = np.asarray(cache["embeddings"], dtype=np.float32)
        labels_all = np.asarray(cache["labels"], dtype=np.int64)
    if (
        probabilities_all.shape != (50_000, 1_000)
        or embeddings_all.shape[0] != 50_000
        or labels_all.shape != (50_000,)
    ):
        raise ValueError("ImageNet cache must contain 50,000 aligned validation rows")
    top1 = float(np.mean(np.argmax(probabilities_all, axis=1) == labels_all))
    if not 0.78 <= top1 <= 0.84:
        raise ValueError(f"ImageNet cache top-1 accuracy {top1:.4f} is implausible")

    split = config["split"]
    train_ids, selection_ids, calibration_ids, test_ids = _imagenet_indices(
        split,
        seed=seed,
    )
    ids = {
        "train": train_ids,
        "selection": selection_ids,
        "calibration": calibration_ids,
        "test": test_ids,
    }
    embeddings = {name: embeddings_all[index] for name, index in ids.items()}
    vision = settings["vision"]
    features = _reduce_embeddings(
        embeddings,
        dimension=int(vision["embedding_dim"]),
        seed=seed,
    )
    probabilities = {name: probabilities_all[index] for name, index in ids.items()}
    labels = {name: labels_all[index] for name, index in ids.items()}
    arrays = {
        key: value
        for name in ("train", "selection", "calibration", "test")
        for key, value in (
            (f"X_{name}", features[name]),
            (f"y_{name}", labels[name]),
            (f"row_id_{name}", np.asarray(ids[name], dtype=np.int64)),
        )
    }
    metadata = {
        "dataset": "imagenet1k",
        "seed": seed,
        "task": "classification",
        "source": config["source"],
        "model": config["model"],
        "representation": f"train_PCA{features['train'].shape[1]}_standardized",
        "split_sizes": {name: len(labels[name]) for name in labels},
        "model_accuracy": {"test": top1},
        "predictor_hyperparameters": {
            "weights": "ResNet50_Weights.IMAGENET1K_V2",
            "embedding_dim": int(vision["embedding_dim"]),
        },
        "preprocessing": (
            "ResNet50_Weights.IMAGENET1K_V2.transforms; "
            "train-view PCA and standardization"
        ),
        "cache": str(cache_path),
    }
    models = {"classifier": _probability_cache(features, probabilities)}
    return arrays, models, metadata


def _save_prepared(
    destination: Path,
    arrays: dict[str, np.ndarray],
    models: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "data.npz", **arrays)  # type: ignore[arg-type]
    with (destination / "models.pkl").open("wb") as handle:
        pickle.dump(models, handle)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def _prepare_one(
    campaign: Campaign,
    dataset: str,
    seed: int,
    output_dir: Path,
) -> Path:
    config = campaign.config["datasets"][dataset]
    source = str(config["source"])
    destination = run_directory(output_dir, dataset, seed)
    model_settings = _effective_model_settings(
        campaign.config["models"],
        config,
    )
    if dataset in {"cifar10", "cifar100", "mnist"}:
        prepared_arrays, prepared_models, prepared_metadata = _image_prepared(
            dataset,
            config,
            seed,
            model_settings,
        )
        return _save_prepared(
            destination,
            prepared_arrays,
            prepared_models,
            prepared_metadata,
        )
    if dataset == "imagenet1k":
        prepared_arrays, prepared_models, prepared_metadata = _imagenet_prepared(
            config, seed, model_settings
        )
        return _save_prepared(
            destination,
            prepared_arrays,
            prepared_models,
            prepared_metadata,
        )

    if source.startswith("sklearn.make_"):
        X, y, counts = _synthetic(dataset, config, seed)
    elif config["task"] in ("regression", "classification") and dataset in {
        "bike",
        "bio",
        "california_housing",
        "concrete",
        "mpg",
        "covertype",
    }:
        X, y, counts = _tabular(dataset, config, seed)
    else:
        raise NotImplementedError(f"preparation for {dataset!r} is not supported")

    train, selection, calibration, test = _indices(len(X), counts, seed=seed)
    scaler = StandardScaler().fit(X[train])
    features = np.asarray(scaler.transform(X), dtype=np.float64)
    task = str(config["task"])
    if task == "regression":
        models = _regression_models(
            features[train],
            np.asarray(y[train], dtype=np.float64),
            features[selection],
            np.asarray(y[selection], dtype=np.float64),
            alpha=campaign.alpha,
            seed=seed,
            profile=campaign.profile,
            settings=model_settings,
        )
    else:
        models = _classification_models(
            features[train],
            y[train],
            seed=seed,
            profile=campaign.profile,
            settings=model_settings,
        )

    arrays: dict[str, np.ndarray] = {}
    for name, indices in (
        ("train", train),
        ("selection", selection),
        ("calibration", calibration),
        ("test", test),
    ):
        arrays[f"X_{name}"] = features[indices]
        arrays[f"y_{name}"] = y[indices]
        arrays[f"row_id_{name}"] = indices
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "task": task,
        "source": source,
        "model": config["model"],
        "representation": config["representation"],
        "split_sizes": dict(zip(("train", "selection", "calibration", "test"), counts)),
        "predictor_hyperparameters": model_settings.get(str(config["model"]), {}),
    }
    return _save_prepared(destination, arrays, models, metadata)


def prepare_campaign(campaign: Campaign, output_dir: Path) -> tuple[Path, ...]:
    """Prepare every dataset/seed pair in a resolved campaign."""
    return tuple(
        _prepare_one(campaign, dataset, seed, output_dir)
        for dataset in campaign.datasets
        for seed in campaign.seeds
    )


def prepare(
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    profile: str = "smoke",
    datasets: str | None = None,
    seeds: str | None = None,
    output_dir: Path | str = Path("data/prepared"),
) -> tuple[Path, ...]:
    """Prepare deterministic splits and fitted models for one campaign."""
    campaign = load_campaign(
        config_path,
        profile=profile,
        datasets=datasets,
        seeds=seeds,
    )
    return prepare_campaign(campaign, Path(output_dir))


__all__ = (
    "cross_fitted_absolute_residuals",
    "prepare",
    "prepare_campaign",
    "run_directory",
)

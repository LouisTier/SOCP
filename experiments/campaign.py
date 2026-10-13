"""Select datasets and seeds from the experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def experiment_device() -> str:
    """Return the torch device used to fit SOMs and image classifiers.

    The default is CPU. Set ``SOCP_DEVICE`` (for example ``cuda`` or ``mps``) to use
    another device.
    """
    return os.environ.get("SOCP_DEVICE", "cpu")


@dataclass(frozen=True, slots=True)
class Campaign:
    """Resolved subset of the experiment grid."""

    profile: str
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    alpha: float
    config: dict[str, Any]


def _comma_separated(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("an override must contain at least one value")
    return items


def regime3_calibration_sizes(
    settings: dict[str, Any],
    *,
    dataset: str,
    full_rows: int,
) -> tuple[int, ...]:
    """Return a Regime 3 dataset's reduced calibration sizes, then the full one."""
    if dataset not in settings["primary_datasets"]:
        return ()
    n_full = int(full_rows)
    if n_full < 1:
        raise ValueError("full Regime 3 calibration size must be positive")
    sample_sizes = tuple(int(value) for value in settings["sensitivity_rows"][dataset])
    if sample_sizes != tuple(sorted(set(sample_sizes))):
        raise ValueError("Regime 3 sensitivity rows must be unique and increasing")
    if any(size < 1 or size >= n_full for size in sample_sizes):
        raise ValueError(
            "Regime 3 sensitivity rows must lie below the full calibration size"
        )
    return (*sample_sizes, n_full)


def select_campaign(
    config: dict[str, Any],
    *,
    profile: str,
    datasets: str | None = None,
    seeds: str | None = None,
) -> Campaign:
    """Resolve one smoke or full campaign."""
    profiles = config.get("profiles")
    dataset_configs = config.get("datasets")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    if not isinstance(dataset_configs, dict):
        raise ValueError("configuration must define datasets")

    profile_config = profiles[profile]
    if not isinstance(profile_config, dict):
        raise ValueError(f"profile {profile!r} must be a mapping")

    dataset_override = _comma_separated(datasets)
    selected_datasets = dataset_override or tuple(profile_config.get("datasets", ()))
    unknown = tuple(name for name in selected_datasets if name not in dataset_configs)
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")

    seed_override = _comma_separated(seeds)
    raw_seeds = seed_override or tuple(
        profile_config.get("seeds", config.get("seeds", ()))
    )
    try:
        selected_seeds = tuple(int(seed) for seed in raw_seeds)
    except (TypeError, ValueError) as error:
        raise ValueError("seeds must be integers") from error
    if not selected_datasets or not selected_seeds:
        raise ValueError("a campaign requires at least one dataset and seed")

    alpha = float(config["alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return Campaign(
        profile=profile,
        datasets=tuple(selected_datasets),
        seeds=selected_seeds,
        alpha=alpha,
        config=config,
    )


def load_campaign(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    profile: str,
    datasets: str | None = None,
    seeds: str | None = None,
) -> Campaign:
    """Load the YAML configuration and resolve one campaign."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"experiment configuration must be a mapping: {path}")
    return select_campaign(
        config,
        profile=profile,
        datasets=datasets,
        seeds=seeds,
    )


__all__ = (
    "Campaign",
    "DEFAULT_CONFIG",
    "load_campaign",
    "regime3_calibration_sizes",
    "select_campaign",
)

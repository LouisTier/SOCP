"""Definitions of the datasets, scores, and calibrators shown in the paper."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from types import MappingProxyType

import pandas as pd

REGRESSION_DATASETS = (
    "bike",
    "bio",
    "california_housing",
    "concrete",
    "mpg",
)
CLASSIFICATION_DATASETS = (
    "cifar10",
    "mnist",
    "covertype",
    "cifar100",
    "imagenet1k",
)
TASK_DATASETS = MappingProxyType(
    {
        "regression": REGRESSION_DATASETS,
        "classification": CLASSIFICATION_DATASETS,
    }
)
DATASET_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "bike": "Bike",
        "bio": "Bio",
        "california_housing": "California Housing",
        "concrete": "Concrete",
        "mpg": "Auto MPG",
        "cifar10": "CIFAR-10",
        "mnist": "MNIST",
        "covertype": "Covertype",
        "cifar100": "CIFAR-100",
        "imagenet1k": "ImageNet-1k",
    }
)
PAPER_BASE_SCORES = MappingProxyType(
    {
        "regression": "abs_residual",
        "classification": "lac",
    }
)
PAPER_CALIBRATORS = ("pooled", "socp")
VALIDATION_COLUMNS = frozenset({"task", "dataset", "score", "calibrator", "seed"})


def _require_task(task: str) -> None:
    if task not in TASK_DATASETS:
        known = ", ".join(sorted(TASK_DATASETS))
        raise ValueError(f"unknown task {task!r}; expected one of {known}")


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(VALIDATION_COLUMNS.difference(frame.columns))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"analysis frame is missing required columns: {names}")


def task_datasets(task: str) -> tuple[str, ...]:
    """Return the datasets of one task, in paper order."""
    _require_task(task)
    return TASK_DATASETS[task]


def paper_score(task: str) -> str:
    """Return the base score of the SCP vs SO-SCP comparison."""
    _require_task(task)
    return PAPER_BASE_SCORES[task]


def validate_paper_methods(
    frame: pd.DataFrame,
    *,
    task: str,
    seeds: Collection[int],
) -> None:
    """Check that every dataset and seed has exactly one SCP and one SO-SCP row."""
    _require_columns(frame)
    _require_task(task)
    expected_seeds = tuple(sorted({int(seed) for seed in seeds}))
    if not expected_seeds:
        raise ValueError("paper validation requires at least one seed")

    task_frame = frame.loc[frame["task"].eq(task)]
    base_score = paper_score(task)
    for dataset in task_datasets(task):
        for seed in expected_seeds:
            for calibrator in PAPER_CALIBRATORS:
                count = int(
                    task_frame.loc[
                        task_frame["dataset"].eq(dataset)
                        & task_frame["score"].eq(base_score)
                        & task_frame["calibrator"].eq(calibrator)
                        & task_frame["seed"].astype(int).eq(seed)
                    ].shape[0]
                )
                if count != 1:
                    raise ValueError(
                        "missing paper comparison cell or duplicate cell: "
                        f"task={task}, dataset={dataset}, seed={seed}, "
                        f"score={base_score}, calibrator={calibrator}, count={count}"
                    )

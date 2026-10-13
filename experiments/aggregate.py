"""Combine completed method runs into CSV summaries."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.analysis_contract import paper_score
from experiments.campaign import DEFAULT_CONFIG, Campaign, load_campaign
from experiments.prepare import run_directory
from socp.evaluation.statistics import PairedComparison, holm, paired_comparison

PRIMARY_FAMILY = "primary:socp_vs_pooled:external_wcovgap"

METRICS = (
    "coverage",
    "size_mean",
    "infinite_cutoff_rate",
    "external_wcovgap",
    "external_wcovgap_k10",
    "external_wcovgap_k50",
    "wsc",
    "ssc",
    "ert",
)


def _summary_column(metric: str) -> str:
    """Return ``{metric}_mean``, except that ``size_mean`` keeps its name."""
    return "size_mean" if metric == "size_mean" else f"{metric}_mean"


def _compositions(campaign: Campaign, task: str) -> tuple[tuple[str, str], ...]:
    if campaign.profile == "smoke":
        return ((paper_score(task), "pooled"), (paper_score(task), "socp"))
    methods = campaign.config["methods"][task]
    return tuple(
        (str(score), str(calibrator))
        for score in methods["scores"]
        for calibrator in methods["calibrators"]
    )


def _metric_rows(campaign: Campaign, runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing: list[Path] = []
    for dataset in campaign.datasets:
        task = str(campaign.config["datasets"][dataset]["task"])
        for seed in campaign.seeds:
            root = run_directory(runs_dir, dataset, seed)
            for score, calibrator in _compositions(campaign, task):
                path = root / f"{score}__{calibrator}" / "metrics.csv"
                if not path.exists():
                    missing.append(path)
                    continue
                with path.open(newline="") as handle:
                    rows.extend(csv.DictReader(handle))
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        remainder = "" if len(missing) <= 5 else f"\n... and {len(missing) - 5} more"
        raise FileNotFoundError(f"missing completed method runs:\n{preview}{remainder}")
    return rows


def _value(
    row: dict[str, str],
    metric: str,
    *,
    keep_infinity: bool = False,
) -> float | None:
    raw = row.get(metric, "")
    if raw in ("", None):
        return None
    value = float(raw)
    if math.isnan(value) or (math.isinf(value) and not keep_infinity):
        return None
    return value


def _mean(rows: list[dict[str, str]], metric: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := _value(row, metric, keep_infinity=metric == "size_mean"))
        is not None
    ]
    return float(np.mean(values)) if values else None


def _write_seed_metrics(rows: list[dict[str, str]], output_dir: Path) -> Path:
    output = output_dir / "seed_metrics.csv"
    fieldnames = tuple(rows[0]) if rows else ()
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_summary(rows: list[dict[str, str]], output_dir: Path) -> Path:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["task"], row["score"], row["calibrator"])].append(
            row
        )
    output = output_dir / "summary.csv"
    fieldnames = (
        "dataset",
        "task",
        "score",
        "calibrator",
        "n_seeds",
        *(_summary_column(metric) for metric in METRICS),
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, task, score, calibrator), group in sorted(grouped.items()):
            result: dict[str, object] = {
                "dataset": dataset,
                "task": task,
                "score": score,
                "calibrator": calibrator,
                "n_seeds": len(group),
            }
            for metric in METRICS:
                result[_summary_column(metric)] = _mean(group, metric)
            writer.writerow(result)
    return output


def _comparison_values(
    rows: list[dict[str, str]],
    *,
    metric: str,
) -> dict[tuple[str, str, str], dict[int, float]]:
    values: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = _value(row, metric)
        if value is not None:
            key = (row["dataset"], row["score"], row["calibrator"])
            values[key][int(row["seed"])] = value
    return values


def _regime3_primary_values(
    campaign: Campaign,
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[int, float]]:
    """Return full-calibration pooled/Regime 3 WCovGap pairs."""
    configured = campaign.config["regime3"]["primary_datasets"]
    primary_datasets = {
        str(dataset) for dataset in configured if dataset in campaign.datasets
    }
    observed_sizes: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset in primary_datasets:
            observed_sizes[dataset].add(int(row["calibration_rows"]))
    full_rows = {
        dataset: max(observed_sizes[dataset])
        for dataset in primary_datasets
        if observed_sizes[dataset]
    }
    missing = primary_datasets - full_rows.keys()
    if missing:
        raise ValueError(
            "no Regime 3 rows found for primary datasets: " + ", ".join(sorted(missing))
        )
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        dataset = row.get("dataset", "")
        mode = row.get("mode", "")
        if (
            dataset not in full_rows
            or mode not in {"pooled", "regime3"}
            or int(row["calibration_rows"]) != full_rows[dataset]
        ):
            continue
        value = _value(row, "external_wcovgap")
        if value is not None:
            calibrator = "socp" if mode == "regime3" else "pooled"
            values[(dataset, calibrator)][int(row["seed"])] = value
    expected = set(campaign.seeds)
    for dataset in full_rows:
        for calibrator in ("pooled", "socp"):
            observed = set(values[(dataset, calibrator)])
            if observed != expected:
                raise ValueError(
                    "full-calibration Regime 3 primary rows are incomplete for "
                    f"{dataset}/{calibrator}: expected {len(expected)}, "
                    f"found {len(observed)}"
                )
    return dict(values)


def _write_paired_tests(
    campaign: Campaign,
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    regime3_rows: list[dict[str, str]] | None = None,
) -> Path:
    """Test SO-SCP against pooled SCP on external WCovGap, with Holm across datasets.

    Concrete and Auto MPG use their full-calibration Regime 3 rows.
    """
    output = output_dir / "paired_tests.csv"
    fieldnames = (
        "family",
        "dataset",
        "score",
        "metric",
        "candidate",
        "baseline",
        "n_pairs",
        "mean_delta",
        "wilcoxon_p",
        "holm_p",
    )
    tests: list[tuple[str, str, PairedComparison]] = []
    if len(campaign.seeds) == 10:
        regime3_primary = (
            _regime3_primary_values(campaign, regime3_rows)
            if regime3_rows is not None
            else {}
        )
        values = _comparison_values(rows, metric="external_wcovgap")
        for dataset in campaign.datasets:
            task = str(campaign.config["datasets"][dataset]["task"])
            score = paper_score(task)
            if (score, "socp") not in _compositions(campaign, task):
                continue
            baseline_values = values.get((dataset, score, "pooled"), {})
            candidate_values = values.get((dataset, score, "socp"), {})
            if (dataset, "socp") in regime3_primary:
                baseline_values = regime3_primary[(dataset, "pooled")]
                candidate_values = regime3_primary[(dataset, "socp")]
            seeds = tuple(
                seed
                for seed in campaign.seeds
                if seed in baseline_values and seed in candidate_values
            )
            if len(seeds) != 10:
                continue
            comparison = paired_comparison(
                np.asarray([candidate_values[seed] for seed in seeds]),
                np.asarray([baseline_values[seed] for seed in seeds]),
            )
            tests.append((dataset, score, comparison))
    holm_values = holm(tuple(comparison.wilcoxon_p for _, _, comparison in tests))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, score, comparison), holm_p in zip(
            tests, holm_values, strict=True
        ):
            writer.writerow(
                {
                    "family": PRIMARY_FAMILY,
                    "dataset": dataset,
                    "score": score,
                    "metric": "external_wcovgap",
                    "candidate": "socp",
                    "baseline": "pooled",
                    "n_pairs": comparison.n_pairs,
                    "mean_delta": comparison.mean_delta,
                    "wilcoxon_p": comparison.wilcoxon_p,
                    "holm_p": holm_p,
                }
            )
    return output


def _analysis_rows(
    campaign: Campaign,
    runs_dir: Path,
    filename: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in campaign.datasets:
        for seed in campaign.seeds:
            path = run_directory(runs_dir, dataset, seed) / "analyses" / filename
            if path.exists():
                with path.open(newline="") as handle:
                    rows.extend(csv.DictReader(handle))
    return rows


def _natural_order(key: tuple[str, ...]) -> tuple[tuple[int, int | str], ...]:
    """Sort numeric fields as numbers, so a 4x4 grid comes before 10x10."""
    return tuple((0, int(value)) if value.isdigit() else (1, value) for value in key)


def _write_analysis_summary(
    rows: list[dict[str, str]],
    output: Path,
    *,
    group_fields: tuple[str, ...],
) -> Path | None:
    if not rows:
        return None
    metrics = tuple(
        metric
        for metric in (
            "coverage",
            "size_mean",
            "external_wcovgap",
            "infinite_cutoff_rate",
            "extra_cells",
            "target_reached",
        )
        if metric in rows[0]
    )
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    fields = (
        *group_fields,
        "n_seeds",
        *(f"{metric}_mean" for metric in metrics),
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped, key=_natural_order):
            result: dict[str, object] = dict(zip(group_fields, key, strict=True))
            result["n_seeds"] = len(grouped[key])
            for metric in metrics:
                result[f"{metric}_mean"] = _mean(grouped[key], metric)
            writer.writerow(result)
    return output


def aggregate_campaign(campaign: Campaign, runs_dir: Path, output_dir: Path) -> Path:
    """Write the seed-level, summary, paired-test, sensitivity, and Regime 3 tables."""
    rows = _metric_rows(campaign, runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_metrics(rows, output_dir)
    summary = _write_summary(rows, output_dir)
    regime3_rows = _analysis_rows(campaign, runs_dir, "regime3.csv")
    _write_paired_tests(
        campaign,
        rows,
        output_dir,
        regime3_rows=regime3_rows,
    )
    _write_analysis_summary(
        _analysis_rows(campaign, runs_dir, "sensitivity.csv"),
        output_dir / "sensitivity.csv",
        group_fields=("dataset", "score", "grid_rows", "grid_columns", "radius"),
    )
    _write_analysis_summary(
        regime3_rows,
        output_dir / "regime3.csv",
        group_fields=("dataset", "score", "calibration_rows", "mode"),
    )
    return summary


def aggregate(
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    profile: str = "smoke",
    datasets: str | None = None,
    seeds: str | None = None,
    runs_dir: Path | str = Path("results/runs"),
    output_dir: Path | str = Path("results/aggregate"),
) -> Path:
    campaign = load_campaign(
        config_path,
        profile=profile,
        datasets=datasets,
        seeds=seeds,
    )
    return aggregate_campaign(campaign, Path(runs_dir), Path(output_dir))


__all__ = ("aggregate", "aggregate_campaign")

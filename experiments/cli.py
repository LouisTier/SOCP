"""Command-line interface for the paper experiment workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiments.campaign import DEFAULT_CONFIG


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment configuration file.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="Run the quick smoke profile or the complete paper profile.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset override.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed override.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the three-command experiment parser."""
    parser = argparse.ArgumentParser(
        prog="socp-experiments",
        description="Prepare data, run conformal methods, and aggregate results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Prepare datasets and fitted models.")
    _add_common_arguments(prepare)
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared"),
        help="Directory for prepared datasets and fitted models.",
    )

    run = commands.add_parser("run", help="Run conformal calibration and evaluation.")
    _add_common_arguments(run)
    run.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/prepared"),
        help="Directory produced by the prepare command.",
    )
    run.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs"),
        help="Directory for raw predictions, metrics, and metadata.",
    )

    aggregate = commands.add_parser(
        "aggregate",
        help="Aggregate completed runs into seed summaries and paired tests.",
    )
    _add_common_arguments(aggregate)
    aggregate.add_argument(
        "--runs",
        type=Path,
        default=Path("results/runs"),
        help="Directory produced by the run command.",
    )
    aggregate.add_argument(
        "--output",
        type=Path,
        default=Path("results/aggregate"),
        help="Directory for the aggregate tables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one workflow command."""
    args = build_parser().parse_args(argv)

    from experiments.aggregate import aggregate
    from experiments.prepare import prepare
    from experiments.run import run

    if args.command == "prepare":
        prepare(
            config_path=args.config,
            profile=args.profile,
            datasets=args.datasets,
            seeds=args.seeds,
            output_dir=args.output,
        )
    elif args.command == "run":
        run(
            config_path=args.config,
            profile=args.profile,
            datasets=args.datasets,
            seeds=args.seeds,
            prepared_dir=args.prepared,
            output_dir=args.output,
        )
    else:
        aggregate(
            config_path=args.config,
            profile=args.profile,
            datasets=args.datasets,
            seeds=args.seeds,
            runs_dir=args.runs,
            output_dir=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

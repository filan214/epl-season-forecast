"""Command-line entrypoints for the pipeline.

Phase 1 exposes `validate`: load historical seasons and print the expanding-window
baseline RPS alongside the de-vigged market RPS. Later phases add fit/simulate
commands (IMPLEMENTATION_GUIDE.md §5, §7).
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eplforecast", description="EPL season forecast pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    v = sub.add_parser(
        "validate",
        help="Expanding-window baseline-vs-market RPS on historical seasons",
    )
    v.add_argument(
        "--seasons", nargs="+", required=True, help="e.g. 2015-16 2016-17 2017-18"
    )
    v.add_argument("--min-train-seasons", dest="min_train_seasons", type=int, default=3)
    v.add_argument("--window", type=int, default=5, help="rolling-form window (matches)")
    v.add_argument("--refresh", action="store_true", help="ignore parquet cache")
    return parser


def _run_validate(args: argparse.Namespace) -> None:
    from .evaluate.validation import print_validation_report, run_baseline_validation
    from .ingest.football_data_csv import load_historical

    matches = load_historical(args.seasons, refresh=args.refresh)
    results = run_baseline_validation(
        matches, min_train_seasons=args.min_train_seasons, window=args.window
    )
    print_validation_report(results)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        _run_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

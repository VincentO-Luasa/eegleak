"""Command-line interface: ``eegleak check splits.csv``."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .checks import run_metadata_checks


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)  # numpy scalars, then anything else


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eegleak", description="Check EEG train/val/test splits for leakage.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="run the metadata checks on a CSV with one row per window")
    check.add_argument("csv", help="path to the splits CSV")
    for name, default in [
        ("subject", "subject_id"),
        ("recording", "recording_id"),
        ("split", "split"),
        ("start", "start_s"),
        ("end", "end_s"),
        ("label", "label"),
    ]:
        check.add_argument(f"--{name}-col", default=default, help=f"column name (default: {default})")
    check.add_argument("--gap-s", type=float, default=0.0, help="warn if windows from different splits are closer")
    check.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    try:
        report = run_metadata_checks(
            df,
            subject_col=args.subject_col,
            recording_col=args.recording_col,
            split_col=args.split_col,
            start_col=args.start_col,
            end_col=args.end_col,
            label_col=args.label_col,
            gap_s=args.gap_s,
        )
    except ValueError as e:
        parser.error(str(e))
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    else:
        print(report.to_markdown(), end="")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Leakage checks and negative controls for EEG train/val/test splits.

Every check returns a :class:`~eegleak.report.Finding`, never ``None``. A check that cannot
run (missing column, single split, ...) returns a ``warning`` explaining why.
"""

from __future__ import annotations

import pandas as pd

from .report import Finding

PREVIEW = 5  # number of offending items quoted in a message


def _missing(df: pd.DataFrame, check: str, *cols: str) -> Finding | None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return Finding(
            check,
            "warning",
            f"Cannot run: missing column(s) {missing}.",
            {"missing": missing, "available": list(map(str, df.columns))},
        )
    return None


def _preview(items: list[str]) -> str:
    more = f", ... (+{len(items) - PREVIEW} more)" if len(items) > PREVIEW else ""
    return ", ".join(items[:PREVIEW]) + more


def _id_overlap(df: pd.DataFrame, id_col: str, split_col: str, check: str, noun: str) -> Finding:
    if f := _missing(df, check, id_col, split_col):
        return f
    nan = df[id_col].isna() | df[split_col].isna()
    # Compare IDs as strings so that 1 and "1" are treated as the same identifier.
    d = pd.DataFrame({"id": df[id_col][~nan].astype(str), "split": df[split_col][~nan].astype(str)})
    splits = sorted(d["split"].unique())
    details = {"n_checked": int(d["id"].nunique()), "splits": splits, "n_rows_missing": int(nan.sum())}
    if len(splits) < 2:
        return Finding(check, "warning", f"Only {len(splits)} split present {splits}; nothing to compare.", details)
    per_id = d.groupby("id")["split"].agg(lambda s: sorted(set(s)))
    bad = per_id[per_id.map(len) > 1]
    if len(bad):
        quoted = [f"{i} ({', '.join(s)})" for i, s in bad.items()]
        return Finding(
            check,
            "error",
            f"{len(bad)} {noun}(s) appear in more than one split: {_preview(quoted)}.",
            {**details, "overlapping": bad.to_dict()},
        )
    if nan.any():
        return Finding(
            check,
            "warning",
            f"No {noun} appears in more than one split, but {nan.sum()} row(s) with a missing "
            f"{id_col} or {split_col} could not be checked.",
            details,
        )
    return Finding(
        check,
        "info",
        f"No {noun} appears in more than one split ({details['n_checked']} {noun}s checked).",
        details,
    )


def subject_overlap(df: pd.DataFrame, subject_col: str = "subject_id", split_col: str = "split") -> Finding:
    """Error if any subject appears in more than one split."""
    return _id_overlap(df, subject_col, split_col, "subject_overlap", "subject")


def recording_overlap(df: pd.DataFrame, recording_col: str = "recording_id", split_col: str = "split") -> Finding:
    """Error if any recording appears in more than one split."""
    return _id_overlap(df, recording_col, split_col, "recording_overlap", "recording")

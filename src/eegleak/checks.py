"""Leakage checks and negative controls for EEG train/val/test splits.

Every check returns a :class:`~eegleak.report.Finding`, never ``None``. A check that cannot
run (missing column, single split, ...) returns a ``warning`` explaining why.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from . import registry
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


def window_temporal_overlap(
    df: pd.DataFrame,
    recording_col: str = "recording_id",
    start_col: str = "start_s",
    end_col: str = "end_s",
    split_col: str = "split",
    gap_s: float = 0.0,
) -> Finding:
    """Error if windows from different splits overlap in time within a recording.

    Warning if they are closer than ``gap_s`` seconds (adjacent windows are autocorrelated).
    For each window, the sweep finds the latest end among earlier-starting windows of every
    other split in the same recording: O(n log n) for the sort plus O(n * n_splits).
    """
    check = "window_temporal_overlap"
    if f := _missing(df, check, recording_col, start_col, end_col, split_col):
        return f
    cols = [recording_col, start_col, end_col, split_col]
    nan = df[cols].isna().any(axis=1)
    d = df.loc[~nan, cols].set_axis(["rec", "start", "end", "split"], axis=1).astype({"rec": str, "split": str})
    shared = d.groupby("rec")["split"].nunique().loc[lambda n: n > 1].index
    d = d[d["rec"].isin(shared)].sort_values(["rec", "start"], kind="stable")
    gap = pd.Series(np.inf, index=d.index)
    for split in d["split"].unique():
        other = d["split"] != split
        latest_end = d["end"].where(~other, -np.inf).groupby(d["rec"]).cummax()
        gap = gap.where(~other, np.minimum(gap, d["start"] - latest_end))
    details = {"n_recordings_in_several_splits": len(shared), "n_rows_missing": int(nan.sum()), "gap_s": gap_s}
    for severity, hit, what in [
        ("error", gap < 0, "overlap in time"),
        ("warning", gap < gap_s, f"are < {gap_s} s apart"),
    ]:
        if hit.any():
            recs = d.loc[hit, "rec"].value_counts().sort_index()
            examples = d[hit].assign(gap_s=gap[hit]).head(10).to_dict("records")
            return Finding(
                check,
                severity,
                f"{hit.sum()} window(s) in {len(recs)} recording(s) {what} with a window from another split: "
                f"{_preview(recs.index.tolist())}.",
                {**details, "windows_per_recording": recs.to_dict(), "examples": examples},
            )
    if nan.any():
        return Finding(
            check,
            "warning",
            f"No cross-split window overlap found, but {nan.sum()} row(s) had missing values.",
            details,
        )
    return Finding(
        check,
        "info",
        f"No windows from different splits overlap or lie within {gap_s} s "
        f"({len(shared)} recording(s) appear in more than one split).",
        details,
    )


def duplicate_windows(arrays: dict[str, np.ndarray], decimals: int = 6) -> Finding:
    """Error if identical windows (after rounding to ``decimals``) occur in more than one split.

    ``arrays`` maps a split name to an array of shape ``(n_windows, ...)``. Duplicates within a
    single split are not reported. Values within ``10**-decimals`` of each other can still round
    differently when they straddle a rounding boundary.
    """
    check = "duplicate_windows"
    if len(arrays) < 2:
        return Finding(check, "warning", f"Only {len(arrays)} split given {list(arrays)}; nothing to compare.")
    seen: dict[bytes, dict[str, list[int]]] = {}
    for split, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        rounded = np.round(arr.reshape(len(arr), -1), decimals) + 0.0  # + 0.0 maps -0.0 to 0.0
        for i, row in enumerate(rounded):
            key = hashlib.blake2b(str(arr.shape[1:]).encode() + row.tobytes(), digest_size=16).digest()
            seen.setdefault(key, {}).setdefault(split, []).append(i)
    groups = [g for g in seen.values() if len(g) > 1]
    details = {"n_windows": {s: len(a) for s, a in arrays.items()}, "decimals": decimals}
    if groups:
        pairs = pd.Series(["/".join(sorted(g)) for g in groups]).value_counts().to_dict()
        return Finding(
            check,
            "error",
            f"{len(groups)} distinct window(s) occur in more than one split ({pairs}).",
            {**details, "n_duplicate_groups": len(groups), "split_pairs": pairs, "examples": groups[:10]},
        )
    return Finding(
        check,
        "info",
        f"No window occurs in more than one split ({sum(details['n_windows'].values())} checked).",
        details,
    )


def label_shift(
    df: pd.DataFrame, label_col: str = "label", split_col: str = "split", tv_threshold: float = 0.1
) -> Finding:
    """Warning if the class distribution of any two splits differs by total variation > ``tv_threshold``."""
    check = "label_shift"
    if f := _missing(df, check, label_col, split_col):
        return f
    nan = df[label_col].isna() | df[split_col].isna()
    dist = pd.crosstab(df.loc[~nan, split_col].astype(str), df.loc[~nan, label_col].astype(str), normalize="index")
    tv = {
        f"{a}/{b}": round(0.5 * float((dist.loc[a] - dist.loc[b]).abs().sum()), 4)
        for a, b in itertools.combinations(dist.index, 2)
    }
    details = {"distributions": dist.round(4).to_dict("index"), "tv_distance": tv, "n_rows_missing": int(nan.sum())}
    if not tv:
        return Finding(check, "warning", f"Only {len(dist)} split present; nothing to compare.", details)
    worst = max(tv, key=tv.get)
    if tv[worst] > tv_threshold:
        return Finding(
            check,
            "warning",
            f"Class distributions differ between splits: total variation {tv[worst]} for {worst} "
            f"(threshold {tv_threshold}).",
            details,
        )
    return Finding(
        check,
        "info",
        f"Class distributions are similar across splits (max total variation {tv[worst]} for {worst}, "
        f"threshold {tv_threshold}).",
        details,
    )


def pretraining_overlap(eval_datasets: str | list[str], model: str) -> Finding:
    """Warning when an evaluation dataset is, or may overlap with, the model's pretraining corpus.

    "May overlap" means both datasets are drawn from the same parent corpus (e.g. TUAB and TUSZ
    are both subsets of the TUH EEG corpus). Only models in :mod:`eegleak.registry` are checked.
    """
    check = "pretraining_overlap"
    names = [eval_datasets] if isinstance(eval_datasets, str) else list(eval_datasets)
    key = registry.find_model(model)
    if key is None:
        return Finding(
            check,
            "info",
            f"No registry entry for {model!r}; pretraining overlap not checked.",
            {"known_models": sorted(registry.MODELS)},
        )
    entry = registry.MODELS[key]
    hits = {}
    for name in names:
        ds = registry.canonical_dataset(name)
        for seen in entry["pretraining"]:
            if registry.normalize(ds) == registry.normalize(seen):
                hits[name] = f"{name} is in the pretraining corpus"
            elif shared := registry.lineage(ds) & registry.lineage(seen):
                hits.setdefault(name, f"{name} may overlap {seen} (both part of {', '.join(sorted(shared))})")
    details = {"model": key, "pretraining": entry["pretraining"], "source": entry["source"], "overlaps": hits}
    if hits:
        return Finding(
            check, "warning", f"Possible pretraining overlap for {key}: {'; '.join(hits.values())}.", details
        )
    return Finding(
        check, "info", f"None of {names} is in the registered pretraining corpus of {key} (see source).", details
    )

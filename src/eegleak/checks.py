"""Leakage checks and negative controls for EEG train/val/test splits.

Every check returns a :class:`~eegleak.report.Finding`, never ``None``. A check that cannot
run (missing column, single split, ...) returns a ``warning`` explaining why.
"""

from __future__ import annotations

import hashlib
import itertools
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from . import registry
from .report import Finding, Report

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
        f"No windows from different splits overlap{f' or lie within {gap_s} s' if gap_s else ''} "
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


def _cv_setup(X, y, groups, n_splits: int, check: str):
    """Flatten ``X`` and choose a fold count usable by both stratified and group K-fold.

    Returns ``(X, y, groups, k, n_classes)`` or a warning ``Finding`` when the data cannot be split.
    """
    X = np.asarray(X)
    X, y, groups = X.reshape(len(X), -1), np.asarray(y), np.asarray(groups)
    if not len(X) == len(y) == len(groups):
        raise ValueError(f"X, y and groups must have the same length, got {len(X)}, {len(y)}, {len(groups)}")
    if pd.isna(groups).any() or pd.isna(y).any():
        return Finding(check, "warning", "Cannot run: y or groups contain missing values.")
    _, class_counts = np.unique(y, return_counts=True)
    k = min(n_splits, len(np.unique(groups)), class_counts.min())
    if len(class_counts) < 2 or k < 2:
        return Finding(
            check,
            "warning",
            f"Cannot run: need at least 2 classes, 2 groups and 2 samples per class "
            f"(got {len(class_counts)} classes, {len(np.unique(groups))} groups, smallest class {class_counts.min()}).",
        )
    return X, y, groups, int(k), len(class_counts)


def _cv_score(estimator, X, y, cv, groups=None) -> float:
    """Balanced accuracy of pooled out-of-fold predictions (defined even if a fold lacks a class)."""
    return float(balanced_accuracy_score(y, cross_val_predict(estimator, X, y, groups=groups, cv=cv)))


def group_vs_random_gap(X, y, groups, estimator, n_splits: int = 5, seed: int = 0, threshold: float = 0.05) -> Finding:
    """Compare random stratified K-fold with group K-fold for the same estimator.

    ``X`` has shape ``(n_windows, ...)`` and is flattened; ``y`` and ``groups`` have shape
    ``(n_windows,)``. A large positive gap shows how much a split that ignores ``groups`` (e.g.
    subjects) would inflate the score. Warning if the gap exceeds ``threshold``.
    """
    check = "group_vs_random_gap"
    setup = _cv_setup(X, y, groups, n_splits, check)
    if isinstance(setup, Finding):
        return setup
    X, y, groups, k, _ = setup
    random = _cv_score(estimator, X, y, StratifiedKFold(k, shuffle=True, random_state=seed))
    group = _cv_score(estimator, X, y, GroupKFold(k), groups)
    gap = random - group
    details = {"random_score": random, "group_score": group, "gap": gap, "n_splits": k, "threshold": threshold}
    scores = f"random K-fold {random:.3f} vs group K-fold {group:.3f} (balanced accuracy, {k} folds)"
    if gap > threshold:
        return Finding(check, "warning", f"A random split inflates the score by {gap:.3f}: {scores}.", details)
    return Finding(check, "info", f"Random and group splits agree within {threshold}: {scores}.", details)


def label_permutation_test(
    X, y, groups, estimator, n_permutations: int = 20, seed: int = 0, alpha: float = 0.05
) -> Finding:
    """Negative control: permute labels within each group and evaluate with group K-fold.

    With a sound pipeline the permuted-label balanced accuracy is at chance (``1 / n_classes``).
    Error if its mean is above chance by a one-sided z-test at level ``alpha`` (suggests a bug or
    leakage, e.g. one subject spread over several groups). Also reports the null distribution and
    the empirical p-value of the real score, ``(1 + #(null >= real)) / (1 + n_permutations)``; its
    smallest possible value is ``1 / (1 + n_permutations)``, so choose ``n_permutations`` accordingly.
    """
    check = "label_permutation_test"
    if n_permutations < 2:
        raise ValueError("n_permutations must be at least 2")
    setup = _cv_setup(X, y, groups, 5, check)
    if isinstance(setup, Finding):
        return setup
    X, y, groups, k, n_classes = setup
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if all(len(np.unique(y[idx])) == 1 for idx in members):
        return Finding(
            check, "warning", "Cannot run: labels are constant within every group, so permuting them changes nothing."
        )
    rng = np.random.default_rng(seed)
    cv = GroupKFold(k)
    real = _cv_score(estimator, X, y, cv, groups)
    null = []
    for _ in range(n_permutations):
        y_perm = y.copy()
        for idx in members:
            y_perm[idx] = y[rng.permutation(idx)]
        null.append(_cv_score(estimator, X, y_perm, cv, groups))
    null = np.array(null)
    chance = 1 / n_classes
    sem = null.std(ddof=1) / np.sqrt(n_permutations)
    z = (null.mean() - chance) / sem if sem > 0 else np.inf * np.sign(null.mean() - chance)
    p_value = (1 + int((null >= real).sum())) / (1 + n_permutations)
    details = {
        "real_score": real,
        "null_scores": null.tolist(),
        "null_mean": float(null.mean()),
        "chance": chance,
        "z": float(z),
        "p_value": p_value,
        "n_splits": k,
    }
    summary = (
        f"permuted-label score {null.mean():.3f} ± {null.std(ddof=1):.3f} vs chance {chance:.3f}; "
        f"real score {real:.3f}, p = {p_value:.3f} ({n_permutations} permutations)"
    )
    if z > NormalDist().inv_cdf(1 - alpha):
        return Finding(check, "error", f"Above chance with permuted labels, suggesting leakage: {summary}.", details)
    return Finding(check, "info", f"Permuted labels give chance performance: {summary}.", details)


def run_metadata_checks(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    recording_col: str = "recording_id",
    split_col: str = "split",
    start_col: str = "start_s",
    end_col: str = "end_s",
    label_col: str = "label",
    gap_s: float = 0.0,
    tv_threshold: float = 0.1,
) -> Report:
    """Run the subject, recording, temporal (if time columns exist) and label (if labels exist) checks.

    Raises ``ValueError`` if a required column (subject, recording, split) is missing.
    """
    if missing := [c for c in (subject_col, recording_col, split_col) if c not in df.columns]:
        raise ValueError(f"missing required column(s) {missing}; available: {list(df.columns)}")
    report = Report()
    report.add(subject_overlap(df, subject_col, split_col))
    report.add(recording_overlap(df, recording_col, split_col))
    if {start_col, end_col} <= set(df.columns):
        report.add(window_temporal_overlap(df, recording_col, start_col, end_col, split_col, gap_s))
    else:
        report.add(Finding("window_temporal_overlap", "info", f"Skipped: no {start_col!r}/{end_col!r} columns."))
    if label_col in df.columns:
        report.add(label_shift(df, label_col, split_col, tv_threshold))
    else:
        report.add(Finding("label_shift", "info", f"Skipped: no {label_col!r} column."))
    return report

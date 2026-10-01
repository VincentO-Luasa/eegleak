"""Generate examples/clean_splits.csv and examples/leaky_splits.csv (deterministic).

Run from the repository root: ``python examples/make_examples.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COUNTS = [4, 1, 9, 3, 3]  # per 20 windows: a typical night
OUT = Path(__file__).parent


def windows(rng, subject, recording, split, n=40, length=30.0, stride=30.0, counts=STAGE_COUNTS):
    """Windows with exact stage proportions (in shuffled order) so the clean example has no label shift."""
    start = np.arange(n) * stride
    return pd.DataFrame(
        {
            "subject_id": subject,
            "recording_id": recording,
            "split": split,
            "start_s": start,
            "end_s": start + length,
            "label": rng.permutation(np.resize(np.repeat(STAGES, counts), n)),
        }
    )


def clean(rng):
    splits = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    return pd.concat(
        [
            windows(rng, f"S{i:02d}", f"S{i:02d}-N{night}", split)
            for i, split in enumerate(splits, 1)
            for night in (1, 2)
        ]
    )


def leaky(rng):
    df = clean(rng)
    # 1. Subject S03's second night was put in the test set.
    df.loc[df["recording_id"] == "S03-N2", "split"] = "test"
    # 2. Recording S05-N1 was cut at 600 s into train/test, with 50%-overlapping windows.
    df = df[df["recording_id"] != "S05-N1"]
    cut = windows(rng, "S05", "S05-N1", "train", n=79, stride=15.0)
    cut.loc[cut["start_s"] >= 600, "split"] = "test"
    # 3. The test subjects S11 and S12 are mostly awake.
    awake = [
        windows(rng, s, f"{s}-N{night}", "test", counts=[14, 2, 2, 1, 1]) for s in ("S11", "S12") for night in (1, 2)
    ]
    return pd.concat([df[~df["subject_id"].isin(["S11", "S12"])], cut, *awake])


for name, make in [("clean", clean), ("leaky", leaky)]:
    df = make(np.random.default_rng(0)).reset_index(drop=True)
    df.insert(0, "window_id", range(len(df)))
    df.to_csv(OUT / f"{name}_splits.csv", index=False)

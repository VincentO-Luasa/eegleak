import numpy as np
from conftest import make_df

from eegleak import duplicate_windows, label_shift, window_temporal_overlap


def recording_split_in_two(test_start):
    """One recording, train windows on [0, 60), test windows from ``test_start``."""
    train = [("S1", "R1", "train", s, s + 30.0, 0) for s in (0.0, 30.0)]
    test = [("S1", "R1", "test", s, s + 30.0, 0) for s in (test_start, test_start + 30.0)]
    return make_df(test + train)  # unsorted on purpose


def test_temporal_clean(clean_df):
    f = window_temporal_overlap(clean_df, gap_s=10)
    assert f.severity == "info" and f.details["n_recordings_in_several_splits"] == 0


def test_temporal_disjoint_windows_in_shared_recording_pass():
    f = window_temporal_overlap(recording_split_in_two(60.0))
    assert f.severity == "info" and f.details["n_recordings_in_several_splits"] == 1


def test_windows_overlapping_by_one_second():
    f = window_temporal_overlap(recording_split_in_two(59.0))
    assert f.severity == "error"
    assert f.details["windows_per_recording"] == {"R1": 1}
    assert f.details["examples"][0]["gap_s"] == -1.0


def test_near_miss_gap_warns():
    df = recording_split_in_two(62.0)
    assert window_temporal_overlap(df, gap_s=0).severity == "info"
    f = window_temporal_overlap(df, gap_s=5)
    assert f.severity == "warning" and f.details["examples"][0]["gap_s"] == 2.0


def test_overlap_detected_when_earlier_window_spans_later_ones():
    df = make_df([("S", "R", "train", 0, 100, 0), ("S", "R", "test", 10, 20, 0), ("S", "R", "test", 120, 130, 0)])
    f = window_temporal_overlap(df)
    assert f.severity == "error" and f.details["examples"][0]["start"] == 10


def test_temporal_missing_columns_and_nans(clean_df):
    assert window_temporal_overlap(clean_df.drop(columns="end_s")).severity == "warning"
    df = clean_df.copy()
    df.loc[0, "start_s"] = np.nan
    f = window_temporal_overlap(df)
    assert f.severity == "warning" and f.details["n_rows_missing"] == 1


def test_duplicate_windows():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(10, 4, 8))
    clean = {"train": train, "test": rng.normal(size=(5, 4, 8))}
    assert duplicate_windows(clean).severity == "info"
    leaky = {"train": train, "test": np.concatenate([clean["test"], train[[3]]])}
    f = duplicate_windows(leaky)
    assert f.severity == "error" and f.details["examples"] == [{"train": [3], "test": [5]}]


def test_duplicate_windows_rounding_tolerance():
    base = np.array([[0.25, -0.5, 1.125]])
    assert duplicate_windows({"train": base, "test": base + 1e-9}, decimals=6).severity == "error"
    assert duplicate_windows({"train": base, "test": base + 1e-3}, decimals=6).severity == "info"
    # -1e-9 rounds to -0.0, which must hash like 0.0
    assert duplicate_windows({"train": np.zeros((1, 3)), "test": np.full((1, 3), -1e-9)}).severity == "error"


def test_duplicate_windows_single_split():
    assert duplicate_windows({"train": np.zeros((2, 3))}).severity == "warning"


def test_label_shift(clean_df):
    f = label_shift(clean_df)
    assert f.severity == "info" and f.details["tv_distance"]["test/train"] == 0.0
    df = clean_df.copy()
    df.loc[df["split"] == "test", "label"] = 1  # test is all class 1, train is 50/50
    f = label_shift(df)
    assert f.severity == "warning" and f.details["tv_distance"]["test/train"] == 0.5
    assert label_shift(df, tv_threshold=0.6).severity == "info"


def test_label_shift_cannot_run(clean_df):
    assert label_shift(clean_df.drop(columns="label")).severity == "warning"
    assert label_shift(clean_df.assign(split="train")).severity == "warning"

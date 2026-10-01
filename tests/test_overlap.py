import numpy as np
from conftest import make_df

from eegleak import recording_overlap, subject_overlap


def test_clean_split_passes(clean_df):
    for f in (subject_overlap(clean_df), recording_overlap(clean_df)):
        assert f.severity == "info" and "4 " in f.message


def test_subject_in_train_and_test(clean_df):
    df = clean_df.copy()
    df.loc[df.index[-1], "subject_id"] = "S1"
    f = subject_overlap(df)
    assert f.severity == "error"
    assert f.details["overlapping"] == {"S1": ["test", "train"]}


def test_subject_with_two_recordings_across_splits(clean_df):
    df = make_df([("S1", "A", "train", 0, 30, 0), ("S1", "B", "test", 0, 30, 0), ("S2", "A", "test", 30, 60, 1)])
    assert subject_overlap(df).severity == "error"
    rec = recording_overlap(df)
    assert rec.severity == "error" and rec.details["overlapping"] == {"A": ["test", "train"]}


def test_string_and_integer_ids_are_the_same_subject():
    df = make_df([(1, "a", "train", 0, 1, 0), ("1", "b", "test", 0, 1, 0)])
    assert subject_overlap(df).severity == "error"


def test_missing_column_single_split_and_nans(clean_df):
    f = subject_overlap(clean_df.drop(columns="subject_id"))
    assert f.severity == "warning" and f.details["missing"] == ["subject_id"]
    f = subject_overlap(clean_df.assign(split="train"))
    assert f.severity == "warning" and "Only 1 split" in f.message
    df = clean_df.copy()
    df.loc[0, "subject_id"] = np.nan
    f = subject_overlap(df)
    assert f.severity == "warning" and f.details["n_rows_missing"] == 1


def test_custom_column_names(clean_df):
    df = clean_df.rename(columns={"subject_id": "patient", "split": "fold"})
    assert subject_overlap(df, subject_col="patient", split_col="fold").severity == "info"

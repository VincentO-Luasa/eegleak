import pandas as pd
import pytest


def make_df(rows):
    """Build a window table from (subject, recording, split, start_s, end_s, label) tuples."""
    return pd.DataFrame(rows, columns=["subject_id", "recording_id", "split", "start_s", "end_s", "label"])


@pytest.fixture
def clean_df():
    rows = []
    for subj, split in [("S1", "train"), ("S2", "train"), ("S3", "val"), ("S4", "test")]:
        for k in range(4):
            rows.append((subj, f"{subj}-R1", split, 30.0 * k, 30.0 * (k + 1), k % 2))
    return make_df(rows)

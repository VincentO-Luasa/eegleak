"""eegleak: catch data leakage in EEG train/val/test splits."""

from .checks import (
    duplicate_windows,
    group_vs_random_gap,
    label_permutation_test,
    label_shift,
    pretraining_overlap,
    recording_overlap,
    subject_overlap,
    window_temporal_overlap,
)
from .report import Finding, Report

__version__ = "0.1.0"

__all__ = [
    "Finding",
    "Report",
    "duplicate_windows",
    "group_vs_random_gap",
    "label_permutation_test",
    "label_shift",
    "pretraining_overlap",
    "recording_overlap",
    "subject_overlap",
    "window_temporal_overlap",
    "__version__",
]

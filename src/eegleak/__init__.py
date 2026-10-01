"""eegleak: catch data leakage in EEG train/val/test splits."""

from .checks import recording_overlap, subject_overlap
from .report import Finding, Report

__version__ = "0.1.0"

__all__ = ["Finding", "Report", "recording_overlap", "subject_overlap", "__version__"]

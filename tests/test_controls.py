import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from eegleak import group_vs_random_gap, label_permutation_test

KNN = KNeighborsClassifier(n_neighbors=1)


def subjects(n_subjects=10, n_windows=20, n_features=8, seed=0):
    """Per-subject fingerprint vectors repeated over windows, plus window noise."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_subjects), n_windows)
    fingerprint = rng.normal(size=(n_subjects, n_features))[groups]
    return rng, groups, fingerprint + 0.1 * rng.normal(size=(len(groups), n_features))


def test_gap_large_with_subject_identifying_feature():
    rng, groups, X = subjects()
    y = groups % 2  # subject-level label, no real signal: only identity predicts it
    f = group_vs_random_gap(X, y, groups, KNN)
    assert f.severity == "warning"
    assert f.details["random_score"] > 0.95 and f.details["gap"] > 0.3


def test_gap_small_on_subject_independent_data():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 20)
    y = rng.integers(0, 2, size=len(groups))
    X = y[:, None] * 1.0 + rng.normal(size=(len(y), 4))  # signal independent of the subject
    f = group_vs_random_gap(X, y, groups, LogisticRegression())
    assert f.severity == "info" and abs(f.details["gap"]) < 0.05


def test_permutation_noise_with_clean_split_passes():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(10), 20)
    X, y = rng.normal(size=(len(groups), 2, 4)), rng.integers(0, 2, size=len(groups))
    f = label_permutation_test(X, y, groups, LogisticRegression(), n_permutations=20)
    assert f.severity == "info"
    assert len(f.details["null_scores"]) == 20 and 1 / 21 <= f.details["p_value"] <= 1


def test_permutation_detects_leaky_pipeline():
    """Groups are recordings, but each subject has two recordings and an identifiable EEG fingerprint.

    Labels have subject-specific base rates, so the model leaks labels across "groups" by recognising
    the subject; permuting within recordings keeps those rates and the score stays above chance.
    """
    rng, subject, X = subjects(n_subjects=8, n_windows=40)
    recording = subject * 2 + (np.arange(len(subject)) % 40 >= 20)
    majority = subject % 2
    y = np.where(rng.random(len(subject)) < 0.9, majority, 1 - majority)
    f = label_permutation_test(X, y, recording, KNN, n_permutations=10)
    assert f.severity == "error" and f.details["null_mean"] > 0.7
    # the same data grouped by subject is clean
    assert label_permutation_test(X, y, subject, KNN, n_permutations=10).severity == "info"


def test_seeded_controls_are_deterministic():
    rng, groups, X = subjects(seed=3)
    y = rng.integers(0, 2, size=len(groups))
    est = LogisticRegression()
    assert label_permutation_test(X, y, groups, est, seed=7) == label_permutation_test(X, y, groups, est, seed=7)
    assert group_vs_random_gap(X, y, groups, est, seed=7) == group_vs_random_gap(X, y, groups, est, seed=7)
    a = label_permutation_test(X, y, groups, est, n_permutations=5, seed=1)
    b = label_permutation_test(X, y, groups, est, n_permutations=5, seed=2)
    assert a.details["null_scores"] != b.details["null_scores"]


def test_controls_cannot_run():
    X, y = np.zeros((6, 2)), np.array([0, 1, 0, 1, 0, 1])
    assert group_vs_random_gap(X, y, np.zeros(6), KNN).severity == "warning"  # one group
    assert group_vs_random_gap(X, np.zeros(6), np.arange(6), KNN).severity == "warning"  # one class
    assert group_vs_random_gap(X, y, np.array([0, 1, 2, np.nan, 4, 5]), KNN).severity == "warning"
    # one sample per group: within-group permutation is a no-op
    f = label_permutation_test(X, y, np.arange(6), KNN)
    assert f.severity == "warning" and "constant within every group" in f.message
    with pytest.raises(ValueError):
        group_vs_random_gap(X, y[:5], np.arange(6), KNN)
    with pytest.raises(ValueError):
        label_permutation_test(X, y, np.arange(6), KNN, n_permutations=1)


def test_class_absent_from_a_fold():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(6), 10)
    y = np.where(groups == 0, 2, rng.integers(0, 2, size=len(groups)))  # class 2 only in group 0
    f = group_vs_random_gap(rng.normal(size=(len(y), 3)), y, groups, LogisticRegression(), n_splits=3)
    assert f.severity in {"info", "warning"} and 0 <= f.details["group_score"] <= 1

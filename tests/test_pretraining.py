from eegleak import pretraining_overlap, registry


def test_registry_hit():
    f = pretraining_overlap(["TUAB", "SleepEDF"], "BIOT")
    assert f.severity == "warning" and list(f.details["overlaps"]) == ["TUAB"]
    assert f.details["model"] == "BIOT-six-datasets" and f.details["source"]


def test_alias_hit_is_case_and_punctuation_insensitive():
    f = pretraining_overlap("chb mit", "biot")
    assert f.severity == "warning" and "is in the pretraining corpus" in f.message
    assert pretraining_overlap("PREST", "EEG-PREST-16-channels").severity == "warning"


def test_subset_of_pretraining_corpus():
    f = pretraining_overlap("TUH Abnormal", "CBraMod")
    assert f.severity == "warning" and "is a subset of TUEG" in f.message


def test_sibling_subsets_may_overlap():
    f = pretraining_overlap("TUAB", "LaBraM")
    assert f.severity == "warning" and "both subsets of TUEG" in f.message


def test_no_overlap():
    f = pretraining_overlap(["SEED-V", "Sleep-EDF"], "LaBraM")
    assert f.severity == "info" and f.details["overlaps"] == {}


def test_unknown_model():
    f = pretraining_overlap("TUAB", "MysteryNet")
    assert f.severity == "info" and "No registry entry for 'MysteryNet'" in f.message


def test_every_registry_entry_is_sourced():
    assert all(m["source"] for m in registry.MODELS.values())
    assert all(d["source"] for d in registry.DATASETS.values() if "part_of" in d)

"""Sourced registry of EEG foundation-model pretraining corpora.

Only facts with a citable source are recorded. Dataset names are matched case-insensitively,
ignoring punctuation and spaces, and through the aliases below.
"""

from __future__ import annotations

import re

TUH_SOURCE = (
    "Obeid & Picone (2016), 'The Temple University Hospital EEG Data Corpus', Front. Neurosci. 10:196; "
    "subset corpora listed at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
)

# canonical name -> aliases and, for subsets of a larger corpus, the parent corpus and its source.
DATASETS: dict[str, dict] = {
    "TUEG": {"aliases": ["TUH EEG", "TUH EEG Corpus", "Temple University Hospital EEG Corpus", "TUH"]},
    **{
        name: {"aliases": aliases, "part_of": "TUEG", "source": TUH_SOURCE}
        for name, aliases in {
            "TUAB": ["TUH Abnormal", "TUH EEG Abnormal Corpus"],
            "TUEV": ["TUH Events", "TUH EEG Events Corpus"],
            "TUSZ": ["TUH Seizure", "TUH EEG Seizure Corpus"],
            "TUEP": ["TUH Epilepsy", "TUH EEG Epilepsy Corpus"],
            "TUAR": ["TUH Artifact", "TUH EEG Artifact Corpus"],
            "TUSL": ["TUH Slowing", "TUH EEG Slowing Corpus"],
        }.items()
    },
    "MGH": {"aliases": ["PREST"]},  # BIOT paper calls the MGH resting EEG data "PREST"
    "SHHS": {"aliases": ["Sleep Heart Health Study"]},
    "CHB-MIT": {"aliases": ["CHB-MIT Scalp EEG Database"]},
    "IIIC Seizure": {"aliases": ["IIIC"]},
    "EEGMMIDB": {"aliases": ["EEG Motor Movement/Imagery Dataset", "PhysioMI", "PhysioNet MI"]},
    "HGD": {"aliases": ["High Gamma Dataset"]},
    "BCI Competition IV-1": {"aliases": ["BCIC IV-1", "BCIC-IV-1"]},
    "Grasp and Lift": {"aliases": ["Grasp and Lift EEG Challenge", "GAL"]},
    "Inria BCI Challenge": {"aliases": []},
    "Siena Scalp EEG Database": {"aliases": ["Siena"]},
    "Target Versus Non-Target": {"aliases": ["Brain Invaders"]},
}

MODELS: dict[str, dict] = {
    "BIOT-six-datasets": {
        "aliases": ["BIOT", "EEG-six-datasets-18-channels"],
        "pretraining": ["MGH", "SHHS", "CHB-MIT", "IIIC Seizure", "TUAB", "TUEV"],
        "source": (
            "Yang, Westover & Sun (2023), BIOT, NeurIPS, arXiv:2305.10351, Sec. 3.6; checkpoint "
            "EEG-six-datasets-18-channels.ckpt at https://github.com/ycq091044/BIOT; arXiv:2607.24519 (v3), Table 1"
        ),
        "notes": "Only the training splits of CHB-MIT, IIIC Seizure, TUAB and TUEV were used (supervised).",
    },
    "BIOT-SHHS+PREST": {
        "aliases": ["EEG-SHHS+PREST-18-channels"],
        "pretraining": ["MGH", "SHHS"],
        "source": "BIOT repository README, https://github.com/ycq091044/BIOT",
    },
    "BIOT-PREST": {
        "aliases": ["EEG-PREST-16-channels"],
        "pretraining": ["MGH"],
        "source": "BIOT repository README, https://github.com/ycq091044/BIOT",
    },
    "CBraMod": {
        "aliases": [],
        "pretraining": ["TUEG"],
        "source": "Wang et al. (2025), CBraMod, ICLR, arXiv:2412.07236, Sec. 3.1",
        "notes": "Over 9,000 hours after cleaning (of 27,062 raw hours), 19 channels of the 10-20 system.",
    },
    "LaBraM": {
        "aliases": ["Large Brain Model"],
        "pretraining": [
            "BCI Competition IV-1",
            "Emobrain",
            "Grasp and Lift",
            "Inria BCI Challenge",
            "EEGMMIDB",
            "Raw EEG Data (Trujillo 2020)",
            "Resting State EEG Data (Trujillo 2017)",
            "SEED",
            "SEED-IV",
            "SEED-GER",
            "SEED-FRA",
            "Siena Scalp EEG Database",
            "SPIS Resting State",
            "Target Versus Non-Target",
            "TUAR",
            "TUEP",
            "TUSZ",
            "TUSL",
            "LaBraM self-collected",
        ],
        "source": "Jiang, Zhao & Lu (2024), LaBraM, ICLR, arXiv:2405.18765, Appendix D",
        "notes": "2,534.78 hours in total; TUAB, TUEV, SEED-V and MoBI were excluded as downstream datasets.",
    },
    "EEGPT": {
        "aliases": [],
        "pretraining": ["EEGMMIDB", "HGD", "TSU SSVEP", "SEED", "M3CV"],
        "source": "Wang et al. (2024), EEGPT, NeurIPS, Table 1 and Appendix C.1",
    },
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", str(name).lower())


def _index(table: dict[str, dict]) -> dict[str, str]:
    return {normalize(alias): key for key, entry in table.items() for alias in [key, *entry["aliases"]]}


_DATASET_INDEX = _index(DATASETS)
_MODEL_INDEX = _index(MODELS)


def find_model(name: str) -> str | None:
    """Canonical model name for ``name`` or one of its aliases, or ``None``."""
    return _MODEL_INDEX.get(normalize(name))


def canonical_dataset(name: str) -> str:
    """Canonical dataset name for ``name``; unknown names are returned unchanged."""
    return _DATASET_INDEX.get(normalize(name), name)


def lineage(dataset: str) -> set[str]:
    """``dataset`` (a canonical name) and the corpus it is a subset of, if any."""
    parent = DATASETS.get(dataset, {}).get("part_of")
    return {dataset, parent} - {None}

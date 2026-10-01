# Agent Brief: `eegleak`

Read this whole file before writing code. If anything here conflicts with what you observe in libraries or data, trust what you observe, verify it, and record the discrepancy in `docs/NOTES.md`.

---

## 1. What this project is and why it exists

`eegleak` is a small open-source Python library and command-line tool that checks whether an EEG machine-learning train/validation/test split is valid, and runs "negative controls" that reveal when a reported score is too good to be true.

**Why it matters (explain this in the README in plain language):** EEG recordings are long time series from a few people. If windows from the same person end up in both the training and the test set, a model can learn to recognise the person instead of the thing being measured (a sleep stage, a seizure, a mental state). It then scores very well on the test set but fails on a new person. Recent benchmark papers on EEG foundation models also point to a second problem: evaluation data can overlap with the data a model was pretrained on. This tool makes those problems easy to detect before a number is published.

**Who it is for:** EEG/BCI researchers and ML engineers; hiring managers at neurotech companies who want to see rigorous, reproducible engineering from the owner (Vincent, a deep-learning research engineer).

**Success is credibility, not feature count.** A small, correct, well-tested, well-documented tool is the goal. Do not add features beyond this brief without asking.

A separate future project (a sleep-staging benchmark) will import this library, so keep the public API stable and well documented.

## 2. Ground rules

1. **Never invent results or outputs.** Every example output in the README must be pasted from a real run of the code in the repo.
2. **Test everything you claim.** Each check needs a passing case and at least one failing case in the test suite.
3. **No silent passes.** If a check cannot run (missing column, unknown model), return an explicit `info` or `warning` finding explaining why, never `None`.
4. **Determinism.** Every randomised function takes a `seed` and gives identical results for identical inputs.
5. **Minimal dependencies:** numpy, pandas, scikit-learn only (plus pytest, ruff for development). Python >= 3.10.
6. **Small commits, feature branches, PRs**, clear messages. The commit history is part of the portfolio.
7. **Ask before expanding scope.** Write blocking questions in `docs/QUESTIONS.md` and stop.
8. **Research tool, not a medical device.** Include a disclaimer in the README.
9. At the end of each session update `docs/STATUS.md`: what was done, commands run, test results, open questions, next steps.

## 3. Data model

Input is a pandas DataFrame (or a CSV read into one) with **one row per window**. Column names are configurable via arguments.

| default column | required | meaning |
|---|---|---|
| `subject_id` | yes | identifier of the person |
| `recording_id` | yes | identifier of one recording/night/session |
| `split` | yes | `train`, `val` or `test` |
| `start_s`, `end_s` | optional | window start/end in seconds within its recording |
| `label` | optional | class label |

Some checks take arrays aligned with the DataFrame rows (`X`, `y`, `groups`); document the expected shapes (`X`: `(n_windows, ...)` flattened internally when needed).

## 4. Core types

Already designed; implement exactly this (see Appendix A for code):

- `Finding(check, severity, message, details)` with `severity` in `{"error", "warning", "info"}`.
- `Report(findings)` with `add`, `extend`, `errors`, `warnings`, `ok`, `to_dict`, `to_markdown`. `ok` is true when there are no errors (warnings do not fail).

Each check returns a `Finding` (never `None`). When a check passes, return an `info` finding stating that it passed and what was checked (e.g. "No subject appears in more than one split (42 subjects checked)").

## 5. Checks (`src/eegleak/checks.py`)

1. **`subject_overlap(df, subject_col, split_col)`** -> error if any subject appears in more than one split. Details: offending ids and the splits they appear in.
2. **`recording_overlap(df, recording_col, split_col)`** -> error if a recording appears in more than one split.
3. **`window_temporal_overlap(df, recording_col, start_col, end_col, split_col, gap_s=0.0)`** -> error if windows from different splits in the same recording overlap in time; warning if closer than `gap_s` (adjacent-window autocorrelation). Use an interval sweep, O(n log n). Only recordings present in more than one split need examining.
4. **`duplicate_windows(arrays, decimals=6)`** where `arrays` maps split name to an array of windows -> error if identical windows (after rounding to `decimals`) occur in more than one split. Hash the bytes of each rounded window.
5. **`label_shift(df, label_col, split_col, tv_threshold=0.1)`** -> warning if class distributions differ between splits by total variation distance above the threshold; otherwise info with the distributions.
6. **`pretraining_overlap(eval_datasets, model)`** -> warning when an evaluation dataset is, or may be a subset of, the model's pretraining corpus. Ship a registry (`src/eegleak/registry.py`) containing **only facts you can source**, each entry with a `source` field. Facts known from the literature (verify against the papers before encoding):
   - BIOT (the checkpoint evaluated in arXiv 2607.24519): pretrained on MGH, SHHS, CHB-MIT, IIIC Seizure, TUAB, TUEV.
   - CBraMod: pretrained on the TUEG corpus (about 9,000 hours, 19 channels, 10-20 system).
   - LaBraM: about 2,500 hours from about 20 datasets; list the datasets only after reading the paper.
   Unknown models must return an `info` finding ("no registry entry for X"). Dataset-name matching should be case-insensitive and support aliases.
7. **`group_vs_random_gap(X, y, groups, estimator, n_splits=5, seed=0, threshold=0.05)`** -> evaluate the same scikit-learn estimator with (a) random stratified K-fold and (b) group K-fold; report both balanced-accuracy scores and the gap; warning if gap > threshold. This quantifies how much a leaky split would have inflated the score.
8. **`label_permutation_test(X, y, groups, estimator, n_permutations=20, seed=0)`** -> negative control. Permute labels **within each group**, evaluate with group K-fold, and compare with chance (1 / n_classes for balanced accuracy). Error if permuted-label performance is significantly above chance (suggests a bug or leakage). Report the null distribution and an empirical p-value for the real score (`(1 + #null >= real) / (1 + n_permutations)`). Document that `n_permutations` must be large enough for the desired p-value resolution.
9. **`run_metadata_checks(df, ...)`** -> run checks 1, 2, 3 (if time columns exist), 5 (if labels exist) and return a `Report`.

Edge cases to handle explicitly: missing columns (clear error message), a single split present, groups with one sample, classes absent from a fold, NaNs in IDs, string vs integer IDs.

## 6. CLI

`eegleak check splits.csv [--subject-col ...] [--recording-col ...] [--split-col ...] [--start-col ...] [--end-col ...] [--label-col ...] [--gap-s 0.0] [--format markdown|json]`

Exit code 1 if the report has errors, 0 otherwise. Markdown report by default. Entry point `eegleak = eegleak.cli:main`.

## 7. Tests (`tests/`, pytest)

Use small synthetic data. Required cases:
- clean split passes every metadata check,
- subject appearing in train and test (error),
- subject with two recordings split across train and test (error on both subject and recording),
- windows overlapping by one second across splits (error), and a near-miss gap case (warning),
- duplicate windows, including float noise below the rounding tolerance (still duplicates) and above (not duplicates),
- label shift above and below threshold,
- registry hit, alias hit, unknown model,
- `group_vs_random_gap` on data with a subject-identifying feature (large gap) and on subject-independent data (small gap),
- `label_permutation_test` on pure noise with a clean split (passes) and on a pipeline with deliberately leaked labels (fails),
- CLI exit codes and both output formats,
- determinism of seeded functions.

Target at least 90% line coverage on `checks.py`; report coverage in CI.

## 8. Docs and packaging

- `README.md`: one-paragraph problem statement for non-experts, install, 10-line quick start (Python and CLI), table of checks with severities, a real example report generated from `examples/leaky_splits.csv`, limitations, license, citation block, disclaimer. A draft is provided in `README_DRAFT.md`; fill in real outputs and fix anything that does not match the code.
- Limitations to state honestly: it cannot detect leakage through shared preprocessing statistics, label leakage inside features, near-duplicate recordings of the same person under different IDs, or pretraining overlap not present in the registry.
- `examples/leaky_splits.csv` and `examples/clean_splits.csv` (small, generated by a script committed in `examples/`).
- `docs/NOTES.md` (decisions, discrepancies), `docs/STATUS.md`, `docs/QUESTIONS.md`.
- GitHub Actions: ruff, pytest with coverage, Python 3.10 / 3.11 / 3.12.
- MIT license. Tag `v0.1.0` when done. Do not publish to PyPI without asking the owner.

## 9. Definition of done

- All tests pass in CI; coverage target met.
- `eegleak check examples/leaky_splits.csv` exits 1 and reports the planted problems; `examples/clean_splits.csv` exits 0.
- README quick start executed from a fresh virtual environment and every shown output is real.
- Registry entries each carry a source reference.
- `docs/STATUS.md` summarises the final state and known limitations.

## 10. Schedule (about one week part-time)

| Day | Goal |
|---|---|
| 1 | Repo setup, Finding/Report, CI skeleton, checks 1-2 with tests |
| 2 | Checks 3-5 with tests |
| 3 | Registry and check 6; read the cited papers to verify entries |
| 4 | Checks 7-8 with tests, including the leaky-pipeline test |
| 5 | `run_metadata_checks`, CLI, examples |
| 6 | README, docs, coverage, fresh-venv verification |
| 7 | Review pass, fix issues, tag `v0.1.0` |

## 11. References (verify before citing)

- "Stress-Testing EEG Foundation Models for Clinical Decoding: Dataset Identity and Targeted Negative Controls", arXiv 2607.24519 (negative-control protocol, pretraining details for BIOT/CBraMod/LaBraM).
- EEG-FM-Bench (arXiv 2508.17742), AdaBrain-Bench (arXiv 2507.09882), OmniEEG-Bench (arXiv 2606.00815).
- Original papers for LaBraM, CBraMod, EEGPT, BIOT.

---

## Appendix A: reference implementations already drafted

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eegleak"
version = "0.1.0"
description = "Catch data leakage and run negative controls on EEG train/val/test splits before you trust a benchmark number."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = ["numpy>=1.24", "pandas>=2.0", "scikit-learn>=1.3"]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov", "ruff"]

[project.scripts]
eegleak = "eegleak.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/eegleak/report.py`:

```python
"""Findings and reports."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SEVERITIES = ("error", "warning", "info")


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding | None) -> None:
        if finding is not None:
            self.findings.append(finding)

    def extend(self, findings: list[Finding | None]) -> None:
        for f in findings:
            self.add(f)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "findings": [asdict(f) for f in self.findings]}

    def to_markdown(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        head = f"**Result: {status}** ({len(self.errors)} errors, {len(self.warnings)} warnings)\n"
        if not self.findings:
            return head + "\nNo findings.\n"
        icon = {"error": "ERROR", "warning": "WARN", "info": "INFO"}
        lines = ["| Severity | Check | Message |", "|---|---|---|"]
        for f in sorted(self.findings, key=lambda x: SEVERITIES.index(x.severity)):
            msg = f.message.replace("|", "\\|")
            lines.append(f"| {icon[f.severity]} | {f.check} | {msg} |")
        return head + "\n" + "\n".join(lines) + "\n"
```

`src/eegleak/__init__.py` should export `Finding`, `Report`, the nine functions in section 5, and `__version__`.

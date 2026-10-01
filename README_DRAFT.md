# eegleak

Catch data leakage in EEG train/validation/test splits, and run negative controls, before you trust a benchmark number.

> **Draft.** Every command and every output below must be verified against the actual code before publishing. Replace each `<PASTE REAL OUTPUT>` with output from a real run.

## Why this exists

EEG datasets contain long recordings from a small number of people. If windows from the same person end up in both the training set and the test set, a model can learn to recognise the person instead of the thing you care about (a sleep stage, a seizure, a mental state). The test score then looks excellent, but the model fails on a new person. Evaluation data can also overlap with the data a pretrained model was trained on, which inflates results in a similar way.

`eegleak` makes these problems easy to detect:

- it checks split metadata for subject, recording and time-window overlap,
- it flags duplicate windows and large shifts in class balance between splits,
- it warns when an evaluation dataset overlaps a known pretraining corpus,
- it measures how much a leaky split would have inflated your score,
- it runs a label-permutation control that should drop to chance if your pipeline is sound.

## Install

```bash
pip install -e .
```

Requires Python 3.10+ (numpy, pandas, scikit-learn).

## Quick start

### Command line

Your splits file has one row per window:

```csv
window_id,subject_id,recording_id,split,start_s,end_s,label
```

```bash
eegleak check examples/leaky_splits.csv
```

```
<PASTE REAL OUTPUT>
```

The command exits with status 1 if any error is found, so it can run in CI.

### Python

```python
import pandas as pd
import eegleak

df = pd.read_csv("examples/leaky_splits.csv")
report = eegleak.run_metadata_checks(df)
print(report.to_markdown())
assert report.ok
```

### Negative controls on a model

```python
from sklearn.ensemble import RandomForestClassifier
import eegleak

est = RandomForestClassifier(n_estimators=100, random_state=0)
print(eegleak.group_vs_random_gap(X, y, groups, est))
print(eegleak.label_permutation_test(X, y, groups, est, n_permutations=20))
```

## Checks

| Check | Severity | What it catches |
|---|---|---|
| `subject_overlap` | error | A person appears in more than one split |
| `recording_overlap` | error | A recording appears in more than one split |
| `window_temporal_overlap` | error / warning | Windows from different splits overlap in time, or sit within `gap_s` of each other |
| `duplicate_windows` | error | Identical windows in more than one split |
| `label_shift` | warning | Class balance differs strongly between splits |
| `pretraining_overlap` | warning | Evaluation data overlaps a model's known pretraining corpus (from a sourced registry) |
| `group_vs_random_gap` | warning | Random-split score is much higher than subject-wise score |
| `label_permutation_test` | error | Score stays above chance with shuffled labels |

## Limitations

`eegleak` cannot detect:

- leakage through preprocessing statistics computed on all data before splitting,
- label information hidden in the features,
- the same person recorded under different IDs,
- pretraining overlap for models or datasets that are not in the registry (the registry only contains entries that have a cited source).

A passing report means the checks above found nothing, not that your evaluation is correct.

## Citation

If you use `eegleak`, please cite this repository. See the references in `docs/NOTES.md` for the papers motivating these checks.

## Disclaimer

Research software. It is not a medical device and must not be used for clinical decisions.

## License

MIT

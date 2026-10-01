# Notes: decisions and discrepancies

## Design decisions

- **IDs are compared as strings**, so `1` and `"1"` count as the same subject. Rows with a missing ID or split are skipped and reported as a `warning` (they could hide a leak).
- **A check that cannot run returns a `warning`.** This covers a missing column, fewer than two splits, or a degenerate permutation. Inside `run_metadata_checks`, an optional check whose columns are absent is reported as an `info` "skipped" finding.
- **Touching windows are not an overlap.** In `window_temporal_overlap`, a window ending at `t` and one starting at `t` do not overlap. They are flagged only when `gap_s > 0`.
- **`duplicate_windows` hashing:**
  - windows are cast to float64, rounded, and `-0.0` is mapped to `0.0`;
  - the hash covers each window's shape and its bytes (BLAKE2b, 128 bits);
  - two values within the tolerance can still round differently when they straddle a rounding boundary.
- **Pooled scoring in `group_vs_random_gap` and `label_permutation_test`.** Balanced accuracy is computed once on pooled out-of-fold predictions (`cross_val_predict`), not averaged per fold. This keeps the score defined when a class is absent from a test fold.
- **Fold count.** `n_splits` is reduced to the number of groups and to the smallest class count when either is smaller, so both CV schemes use the same number of folds.
- **"Significantly above chance"** in `label_permutation_test` is a one-sided z-test that the mean permuted-label score exceeds `1 / n_classes`. The threshold is `alpha` (default 0.05), a keyword argument added beyond the brief.
- **Formatting.** `report.py` matches Appendix A except that its imports are sorted to satisfy ruff (`asdict, dataclass, field`).

## Registry sources and discrepancies with the brief

Each claim below was checked against the primary source (October 2026).

- **arXiv 2607.24519** has changed title across versions:
  - v1 (27 Jul 2026): "Stress-Testing EEG Foundation Models for Clinical Decoding: Dataset Identity and Targeted Negative Controls" (the title in the brief);
  - v3 (13 Aug 2026): "A Negative-Control Protocol for Clinical EEG Foundation-Model Benchmarks: Dataset Identity and External-Cohort Stress Testing".

  The paper evaluates LaBraM, EEGMamba, CBraMod, REVE and BIOT (v1 also had BENDR). It does not evaluate EEGPT. Its Table 1 states that "the evaluated BIOT checkpoint was pretrained on MGH, SHHS, CHB-MIT, IIIC Seizure, TUAB, and TUEV".
- **BIOT** (arXiv:2305.10351, NeurIPS 2023) releases three checkpoints, so the registry has three entries:
  - `EEG-PREST-16-channels`
  - `EEG-SHHS+PREST-18-channels`
  - `EEG-six-datasets-18-channels`

  The plain alias `BIOT` maps to the six-dataset checkpoint, which is the one described in 2607.24519. Its supervised datasets (CHB-MIT, IIIC, TUAB, TUEV) contributed only their **training** splits. Their test splits are still the same cohorts, so the registry warns. The paper names the MGH data "PREST"; it is registered as an alias of `MGH`.
- **CBraMod** (arXiv:2412.07236, ICLR 2025, Sec. 3.1) is pretrained on TUEG. Raw TUEG is 27,062 h; "longer than 9000 hours" remained after cleaning, on 19 channels forming a subset of the 10-20 system. This confirms the brief.
- **LaBraM** (arXiv:2405.18765, ICLR 2024, Appendix D) uses 2,534.78 h. The appendix lists 16 entries; the SEED series counts as four datasets, giving 19, which matches "about 20". TUAB, TUEV, SEED-V and MoBI are explicitly excluded. LaBraM does use TUAR, TUEP, TUSZ and TUSL, which are subsets of the TUH EEG corpus, as TUAB and TUEV are. The registry therefore reports a *possible* overlap for TUAB/TUEV: patients may be shared. This is an inference from corpus membership, not a claim made in the LaBraM paper.
- **EEGPT** (NeurIPS 2024, Table 1 / Appendix C.1) pretrains on PhysioMI, HGD, TSU SSVEP, SEED and M3CV. Its introduction lists only three of these; the registry follows Table 1. PhysioMI is the PhysioNet EEG Motor Movement/Imagery dataset (109 subjects), the same one LaBraM lists, so both map to `EEGMMIDB`.
- **TUH subsets.** TUAB, TUEV, TUSZ, TUEP, TUAR and TUSL are registered as subsets of TUEG (Obeid & Picone 2016, and the TUH corpus pages).

## References

- Zare (2026). arXiv:2607.24519 (see the title note above).
- Xiong et al. EEG-FM-Bench, arXiv:2508.17742.
- Wu et al. AdaBrain-Bench, arXiv:2507.09882.
- Lu et al. OmniEEG-Bench, arXiv:2606.00815.
- Yang, Westover & Sun (2023). BIOT. NeurIPS. arXiv:2305.10351.
- Wang et al. (2025). CBraMod. ICLR. arXiv:2412.07236.
- Jiang, Zhao & Lu (2024). LaBraM. ICLR. arXiv:2405.18765.
- Wang et al. (2024). EEGPT. NeurIPS.

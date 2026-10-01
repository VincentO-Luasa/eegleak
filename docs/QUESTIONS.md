# Open questions for the owner

None of these block v0.1.0.

1. **Registry scope.** The brief asked for BIOT, CBraMod and LaBraM. I also added EEGPT (sourced from its NeurIPS 2024 paper) and the two other released BIOT checkpoints. I can drop them if you prefer the strict list. Other models evaluated in arXiv:2607.24519 (EEGMamba, REVE) are not registered; I can add them after reading their papers.
2. **Sibling subsets.** LaBraM excludes TUAB and TUEV but pretrains on other TUH subsets (TUSZ, TUEP, TUAR, TUSL), and the registry warns that they "may share patients or recordings". This is an inference from corpus membership; tell me if you would rather report it as `info`.
3. **PyPI.** The package is not published, as instructed. Should it be?

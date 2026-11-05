# Dataset provenance and attribution

## Wisconsin Diagnostic Breast Cancer

Creators credited by UCI: William Wolberg, Olvi Mangasarian, Nick Street and W. Street. Dataset citation: Wolberg et al. (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

- Source and license statement: [UCI dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- Data license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Distribution used here: the dataset bundled with scikit-learn 1.6.1, loaded with `load_breast_cancer(as_frame=True)`.
- Modifications: export to CSV, add a row-index sample identifier, recode the target to 1=malignant/0=benign, and generate fixed train/validation/test subsets. The original clinical IDs are not used.
- The data provider does not endorse this project. This project does not license or relicense scikit-learn or other third-party software.

The transformed CSV snapshots remain attributed to the original dataset creators. Keep the attribution, source and license links with redistributed copies. Source documentation copied from the installed package is retained in `data/DATASET.md`.

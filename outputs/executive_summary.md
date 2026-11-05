# Executive summary

Author: Wawan Saputra

Educational Wisconsin Diagnostic Breast Cancer classification study. Positive class: malignant.

Split train/validation/test: 341/114/114. Selected by training-only five-fold ROC-AUC: Logistic Regression.

Validation-selected F1 threshold: 0.68. Test ROC-AUC: 0.9954; precision: 1.0000; recall: 0.9286; F1: 0.9630.

Test false negatives: 3; false positives: 0; Brier score: 0.0214.

## Interpretation and recommendations

Compare error costs before choosing a threshold. F1 optimization does not encode clinical costs. Importance describes model reliance, not causation. Correlated features share importance and can make partial dependence unrealistic.

## Limitations and Future Work

Small historical single-source sample, no external validation, uncertainty intervals, patient grouping identifiers or subgroup fairness audit. Image-derived classification does not establish population risk or screening utility. This educational model is not validated for medical decisions. Next: external validation, repeated/nested evaluation, uncertainty estimates and domain-informed cost-sensitive thresholds.

## CV-ready bullets

- Compared four classification pipelines using stratified five-fold cross-validation and a held-out test set of 114 samples.

- Achieved test ROC-AUC 0.995 and F1 0.963 with Logistic Regression; assessed calibration and interpreted model reliance with permutation importance.

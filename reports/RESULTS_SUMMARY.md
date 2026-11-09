# Results summary (internal cross-validation only)

Data: 115 patients, 27 events (23.5 %). Design: 3 × repeated stratified 4-fold outer CV, 3-fold inner CV, 30 Optuna trials,
selection / tuning / threshold fitted on training rows only. **No external validation.** All numbers come from
`outputs/main/tables/metrics_summary.csv` (mean of per-repeat pooled out-of-fold metrics; CI = patient-level bootstrap,
test-sample variability only, hence optimistic). Run: `outputs/main`, 180 fold-tasks, 34.6 min, 0 failed Optuna trials.

## 1. Pre-specified primary analysis: L2 logistic regression + class weighting

| Configuration | ROC-AUC (95 % CI) | PR-AUC | Sens. | Spec. | Precision | F1 | Bal. acc. | Brier |
|---|---|---|---|---|---|---|---|---|
| CBs | 0.357 (0.283–0.436) | 0.184 | 0.333 | 0.455 | 0.156 | 0.212 | 0.394 | 0.269 |
| DVH | 0.953 (0.907–0.985) | 0.865 | 0.889 | 0.883 | 0.713 | 0.787 | 0.886 | 0.119 |
| CT RFs | 0.413 (0.328–0.499) | 0.201 | 0.370 | 0.538 | 0.190 | 0.240 | 0.454 | 0.267 |
| DFs | 0.897 (0.830–0.947) | 0.736 | 0.802 | 0.856 | 0.631 | 0.706 | 0.829 | 0.143 |
| CBs+DVH | 0.924 (0.867–0.970) | 0.820 | 0.827 | 0.920 | 0.761 | 0.792 | 0.874 | 0.112 |
| CBs+CT RFs | 0.390 (0.314–0.461) | 0.197 | 0.420 | 0.432 | 0.183 | 0.252 | 0.426 | 0.280 |
| CBs+DFs | 0.897 (0.830–0.947) | 0.736 | 0.802 | 0.852 | 0.625 | 0.702 | 0.827 | 0.143 |
| DVH+CT RFs | 0.942 (0.893–0.978) | 0.782 | 0.864 | 0.864 | 0.672 | 0.752 | 0.864 | 0.095 |
| DVH+DFs | 0.946 (0.902–0.979) | 0.843 | 0.852 | 0.871 | 0.678 | 0.753 | 0.862 | 0.093 |
| CT RFs+DFs | 0.897 (0.830–0.948) | 0.736 | 0.802 | 0.852 | 0.625 | 0.702 | 0.827 | 0.143 |
| CBs+DVH+CT RFs | 0.947 (0.901–0.980) | 0.795 | 0.864 | 0.886 | 0.705 | 0.775 | 0.875 | 0.089 |
| CBs+DVH+DFs | 0.946 (0.902–0.979) | 0.843 | 0.852 | 0.871 | 0.678 | 0.753 | 0.862 | 0.093 |
| CBs+CT RFs+DFs | 0.897 (0.830–0.948) | 0.736 | 0.790 | 0.841 | 0.606 | 0.684 | 0.816 | 0.143 |
| DVH+CT RFs+DFs | 0.939 (0.890–0.975) | 0.836 | 0.827 | 0.875 | 0.681 | 0.743 | 0.851 | 0.106 |
| CBs+DVH+CT RFs+DFs | 0.939 (0.890–0.975) | 0.836 | 0.827 | 0.875 | 0.681 | 0.743 | 0.851 | 0.106 |

Threshold = inner-CV Youden; fixed-0.5 metrics are in the same table (`*@0.5` columns). Calibration slope/intercept are in the CSV.

### What the numbers do and do not say
* **Dose-related information carries the signal.** Configurations that contain DVH (AUC 0.92–0.95) or dosiomic features (≈ 0.90) discriminate; configurations containing only CBs and/or CT radiomics are at or below chance (0.31–0.42).
* **AUC < 0.5 is not a "worse than chance" finding.** With no learnable signal, cross-validated AUC is typically slightly below 0.5 (selected noise features anti-correlate across folds); calibration slopes are negative for the same reason. It is consistent with the label-permutation null (below) centred at 0.49–0.51.
* **Adding CBs, CT RFs or DFs to DVH did not improve discrimination:** paired ΔAUC vs. DVH ranges −0.022 … −0.003 for the DVH-containing combinations, all with bootstrap CIs that include 0 (`tables/vs_reference_DVH_primary.csv`); DFs alone are 0.047 lower (CI −0.089 … −0.011). These are descriptive intervals, not multiplicity-adjusted tests.
* **Configurations that differ only by CBs/CT RFs give identical results** (e.g. DFs, CBs+DFs, CT RFs+DFs): stability selection retained the same dosiomic/DVH features in every case (see `figures/feature_selection_stability_heatmap`). DVH and dosiomic features are highly collinear (dosiomic first-order mean ≈ Dmean).
* The best single configuration (DVH, AUC 0.953) was identified on the same data → optimistic.
* Calibration slope > 1 for logistic models on DVH (≈ 2.5): probabilities are under-dispersed because of strong regularisation / class weighting; they are not absolute risks.

## 2. Pipeline sanity check: label-permutation null (`tables/permutation_test_summary.csv`, 50 permutations, primary model, 1 repeat)

| Config | Observed AUC (repeat 0) | Null mean ± SD | Null 95th pct | Permutation p |
|---|---|---|---|---|
| CBs | 0.377 | 0.494 ± 0.059 | 0.595 | 0.98 |
| CT RFs | 0.388 | 0.485 ± 0.084 | 0.626 | 0.90 |
| DVH | 0.949 | 0.506 ± 0.077 | 0.644 | 0.020 |
| CBs+DVH+CT RFs+DFs | 0.931 | 0.500 ± 0.062 | 0.599 | 0.020 |

The null is centred on 0.5, i.e. the nested pipeline does not manufacture discrimination from noise (supporting the absence of leakage). With 50 permutations the smallest attainable p is 1/51 = 0.020.

## 3. Imbalance handling and classifier choice (sensitivity analyses)

* **None vs class weights vs SMOTE:** mean AUC across the 15 configurations differs by ≤ 0.007 within each classifier (e.g. L2-LR 0.821 / 0.819 / 0.823; elastic-net 0.840 / 0.837 / 0.835; linear SVM 0.847 / 0.847 / 0.844; RF 0.817 / 0.816 / 0.810). No strategy is supported as superior; class weighting was pre-specified for the primary analysis. Strategy changes sensitivity/specificity balance and probability scale more than discrimination (Brier: DVH L2-LR 0.104 none vs 0.119 class-weight vs 0.127 SMOTE).
* **Classifier family:** linear models (elastic-net LR 0.835–0.840, linear SVM 0.844–0.847 mean AUC) ≥ L2-LR (0.82) ≈ shallow RF (0.81–0.82). Extended set (`outputs/extended`, no resampling): RBF-SVM 0.842, ExtraTrees 0.825, AdaBoost 0.820, LightGBM 0.819, GB 0.818, kNN 0.816, naïve Bayes 0.814, XGBoost 0.807. Flexible learners gave no advantage over regularised linear models.
* **Relaxed stability selection** (`outputs/sens_relaxed_stability`: threshold 0.5, ≥ 3 features): AUCs equal or slightly lower than the main run (e.g. DVH 0.953 vs 0.953; CBs+DVH+CT RFs+DFs 0.931 vs 0.939); conclusions unchanged.

## 4. Selection behaviour worth reporting
* In **all 180 outer fits the 0.75 stability threshold was never reached**; the adaptive fallback lowered it (median final threshold 0.27) down to the minimum of 2 features. With ≈ 21 training events and strongly collinear predictors the manuscript's 75 % criterion is effectively unattainable, so "stable predictors" are the two most frequently selected features. Report this transparently or revise the criterion before the final analysis (a data-independent choice, not tuned on these results).
* Most frequently selected: DVH_V30 (83 % of outer fits in DVH-containing configs), DVH_Dmax (DVH-only), DF glcm ClusterShade / first-order Skewness / Mean (dosiomics), CB SMOKE / thyroid volume for CBs only (non-predictive).

## 5. Not done / caveats
* 2000 stability iterations (manuscript) were not run; 500 used. The permutation test used 50 permutations and four configurations only.
* No external or temporal validation; no clinical-utility claim is supported by the descriptive decision curves.
* The 120-vs-115 and 20-vs-27 discrepancies remain unresolved (see `DATA_DISCREPANCY_120_vs_115.md`).

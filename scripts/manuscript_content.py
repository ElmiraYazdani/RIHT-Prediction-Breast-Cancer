"""Text of the manuscript (kept separate from the .docx builder).

Inline markup:  **bold**  *italic*  {{highlighted placeholder for the authors}}  [@key1,@key2] citations (auto-numbered).
Block types: ("h1"|"h2"|"p"|"bullet"|"fig"|"table"|"note", payload)
"""

TITLE = ("Do dosiomic features outperform dose–volume parameters, CT radiomics and clinical variables for predicting "
         "one-year radiation-induced hypothyroidism in breast cancer radiotherapy? A leakage-safe nested "
         "cross-validation study")
RUNNING = "Dosiomics vs DVH for radiation-induced hypothyroidism"
AUTHORS = "{{Author names and affiliations}}"
CORRESP = "{{Corresponding author: name, address, e-mail}}"

ABSTRACT = [
    ("Background", "Radiation-induced hypothyroidism (RIHT) is a frequent late effect of breast radiotherapy that includes the "
     "supraclavicular region. Dosiomics, which describes the spatial pattern of the three-dimensional dose distribution, has been "
     "proposed to predict toxicity better than dose–volume histogram (DVH) parameters, but evidence in breast cancer is lacking."),
    ("Objective", "To test whether dosiomic features (DFs) predict one-year RIHT better than DVH parameters, CT radiomic features "
     "(CT RFs) and clinical biomarkers (CBs), alone or in combination."),
    ("Methods", "In 115 breast-cancer patients (27 with hypothyroidism at one year, defined from serum TSH), 5 CBs, 12 DVH parameters, "
     "107 CT RFs and 107 DFs were obtained for the thyroid. All 15 non-empty feature-group combinations were evaluated with "
     "repeated (3×) nested stratified cross-validation (4 outer, 3 inner folds). Imputation, scaling, feature selection (volume "
     "filter, redundancy filter, mRMR, stability selection), class-imbalance handling, Optuna hyper-parameter tuning and the decision "
     "threshold were fitted on training data only. The primary model, pre-specified, was L2-regularised logistic regression with "
     "class weighting; 19 further classifier/imbalance combinations were sensitivity analyses. Outcomes were ROC-AUC, PR-AUC, "
     "sensitivity, specificity, precision, F1, balanced accuracy, Brier score and calibration, with patient-level bootstrap "
     "intervals and a label-permutation check."),
    ("Results", "DVH parameters reached an out-of-fold AUC of 0.953 (95% CI 0.907–0.985), DFs 0.897 (0.830–0.947), DVH+DFs 0.946 "
     "(0.902–0.979) and all four groups 0.939 (0.890–0.975). CBs (0.357) and CT RFs (0.413) carried no discriminative information; "
     "the permutation null was centred at AUC 0.49. The paired AUC difference was −0.047 (−0.089 to −0.011) for DFs versus DVH and "
     "−0.003 (−0.019 to 0.013) for DVH+DFs versus DVH. Across 20 classifier–imbalance combinations, DFs never exceeded DVH, and adding "
     "DFs to DVH improved AUC in 5 of 20 (maximum +0.011). The selected DFs largely re-encoded dose level (Spearman 0.82–0.91 with "
     "mean dose and V30)."),
    ("Conclusions", "In this internally validated cohort, dosiomic features did not improve one-year RIHT prediction over conventional "
     "DVH parameters, and clinical and CT-radiomic features added nothing. A thyroid DVH summary (V30, Dmax, Dmean) was sufficient. "
     "These hypothesis-generating findings require external validation."),
]
KEYWORDS = ("Radiation-induced hypothyroidism; breast cancer; dosiomics; dose–volume histogram; CT radiomics; machine learning; "
            "nested cross-validation; radiotherapy")

B = []  # body blocks
a = B.append

# =============================================================== 1 INTRODUCTION
a(("h1", "1. Introduction"))
a(("p", "Adjuvant radiotherapy, particularly when it includes the regional lymph nodes, improves locoregional control and survival in "
        "breast cancer, but the thyroid gland lies close to the supraclavicular field and receives unintended dose. Hypothyroidism "
        "is therefore a recognised late effect: a meta-analysis of 20 studies found a pooled relative risk of 1.48 (95% CI 1.17–1.87) "
        "in breast-cancer survivors compared with women without breast cancer, rising to 1.69 (1.16–2.46) when the supraclavicular "
        "region was irradiated [@solmunde2023]. In a population-based Swedish cohort of 21,268 women, regional nodal irradiation was "
        "associated with hypothyroidism (HR 1.68, 95% CI 1.42–1.99) whereas breast or chest-wall irradiation alone was not "
        "(HR 1.01) [@digkas2024]. In prospective and retrospective radiotherapy series, 21–26% of patients developed hypothyroidism "
        "[@zhao2023; @kanyilmaz2017], with the peak incidence 6–12 months after treatment [@zhao2023]. Because symptoms are "
        "non-specific and treatment is lifelong thyroid-hormone replacement, early identification of patients at risk is clinically "
        "relevant."))
a(("p", "Thyroid dose is the best-established determinant. A mean thyroid dose above about 21 Gy was associated with hypothyroidism "
        "in two breast-cancer cohorts [@zhao2023; @kanyilmaz2017], and dose–response analyses in head-and-neck cancer describe a "
        "dose–response relationship (an estimated 50% risk at 45 Gy) [@vogelius2011] and a systematic review identified mean dose and dose-volume parameters as the main dosimetric predictors [@chow2022]. Most of this evidence rests on a handful of "
        "dose-volume histogram (DVH) summary points. A DVH discards the spatial arrangement of dose within the organ: very "
        "different three-dimensional dose distributions can yield the same DVH [@yang2023; @ebert2021]."))
a(("p", "Radiomics converts images into high-dimensional minable data [@lambin2012; @gillies2016], and dosiomics applies the same "
        "feature families (first-order, shape and texture features) to the planned dose map, so that the spatial pattern of dose can be "
        "used for outcome prediction [@liang2019]. Dosiomic models have outperformed dosimetric or NTCP models for radiation "
        "pneumonitis (multivariable AUC 0.782 versus 0.676 for dosimetric factors in 70 lung-cancer patients) [@liang2019] and have added "
        "information for xerostomia [@gabrys2018] and for rectal, genitourinary and vaginal toxicity [@lucia2021]; dose-map features "
        "have also been used for survival modelling in head-and-neck cancer (concordance index 0.72–0.73) [@mansouri2024]. For hypothyroidism, the "
        "evidence comes from nasopharyngeal carcinoma (NPC): a dosiomics model reached an AUC of 0.70 against 0.61 for dose-volume "
        "factors in 145 patients [@ren2021]; a multiview radiomics–dosiomics–DVH–clinical model reached a C-index of 0.725 in a "
        "prospective test cohort [@yang2023]; and in 219 tomotherapy patients, a combined model (AUC 0.842) exceeded single-source "
        "models (clinical 0.798, DVH 0.673, radiomics 0.714, dosiomics 0.698) [@quan2024]. These cohorts are larger and "
        "irradiate the thyroid in a different way (high-dose, bilateral) from breast radiotherapy, in which the thyroid is "
        "incidentally exposed to low-to-intermediate dose. Whether dosiomics adds to DVH parameters for breast-cancer RIHT is unknown."))
a(("p", "Evaluating high-dimensional feature groups on few outcome events is also methodologically hazardous. Feature selection or "
        "hyper-parameter tuning performed outside the validation loop produces optimistic error estimates, even on pure noise "
        "[@ambroise2002; @varma2006], and the bias is large in small samples [@vabalas2019]. Reported radiomics and dosiomics "
        "performance is therefore only interpretable when every data-dependent step is nested inside cross-validation and uncertainty "
        "is reported [@collins2024]."))
a(("p", "The **primary aim** of this study was to determine whether dosiomic features lead to better prediction of one-year "
        "radiation-induced hypothyroidism than conventional DVH parameters, CT radiomic features and clinical biomarkers. "
        "Secondary aims were to quantify the contribution of each of the four feature groups and of all 15 of their combinations, to "
        "compare class-imbalance strategies and classifiers, and to characterise the stability of the selected features, using a "
        "validation design in which no outer-fold information can leak into feature selection, tuning or thresholding."))

# =============================================================== 2 METHODS
a(("h1", "2. Materials and Methods"))
a(("h2", "2.1. Study workflow"))
a(("p", "The study followed a four-stage workflow (Figure 1). **Stage A (data)**: a prospective breast-radiotherapy cohort provided "
        "planning CT images, three-dimensional dose distributions, clinical variables and one-year thyroid function; the thyroid was "
        "contoured and four feature groups were created (CBs, DVH, CT RFs and DFs). **Stage B (design)**: the four source files were "
        "merged on patient identifiers, validated, and combined into the 15 non-empty feature-group configurations. **Stage C "
        "(repeated nested cross-validation)**: for every configuration, an outer 4-fold stratified cross-validation (3 repeats) "
        "separated a held-out test fold from a training set; on the training set only, features were pre-processed and selected, "
        "classifiers and class-imbalance strategies were tuned by an inner 3-fold cross-validation with Optuna, a decision threshold "
        "was derived from inner out-of-fold predictions, and the final model was refitted and applied once to the held-out fold. "
        "**Stage D (evaluation)**: pooled out-of-fold predictions were summarised with discrimination, threshold-based, calibration and "
        "decision-curve metrics, with bootstrap uncertainty, paired comparisons between configurations, a label-permutation check and "
        "pre-planned sensitivity analyses. The subsections below describe each stage in the order of Figure 1; reporting follows "
        "TRIPOD+AI as far as applicable [@collins2024]."))
a(("fig", ("figure1_workflow", "Figure 1. Study workflow. (A) Cohort and feature groups; (B) data validation and the 15 feature-group "
           "configurations; (C) repeated nested cross-validation, in which every data-dependent step is fitted on the outer training "
           "set only and the held-out fold is used once for prediction; (D) evaluation and secondary analyses. CBs, clinical "
           "biomarkers; DVH, dose–volume histogram parameters; CT RFs, CT radiomic features; DFs, dosiomic features.", 5.4)))

a(("h2", "2.2. Stage A – Cohort and outcome"))
a(("p", "This prospective single-centre study included patients with breast cancer who received radiotherapy at {{institution name}} "
        "between 2024 and 2026 and who had planning CT images, dose distributions, and thyroid function tests before radiotherapy and "
        "at one-year follow-up. {{Inclusion and exclusion criteria, including prior thyroid disease or thyroid medication, "
        "elevated baseline TSH, and loss to follow-up.}} The final analysis set comprised 115 patients. {{Authors: an earlier draft "
        "stated 120 patients with 20 events; all four data files contain 115 patients with 27 events. Please reconcile, add a "
        "patient-flow diagram and give one outcome definition (see reports/DATA_DISCREPANCY_120_vs_115.md).}} "
        "The study was approved by the institutional review board of Iran University of Medical Sciences (IR.IUMS.REC.1405.211); "
        "{{informed-consent statement}}."))
a(("p", "Thyroid function was assessed by serum thyroid-stimulating hormone (TSH) measured before radiotherapy and one year after "
        "completion. The outcome was hypothyroidism at one-year follow-up ({{TSH cut-off, free-T4 criterion and handling of "
        "subclinical cases}}), coded 1 (hypothyroidism; n = 27, 23.5%) or 0 (no hypothyroidism; n = 88). The follow-up TSH value "
        "defines the outcome and was never used as a predictor; the data-loading code rejects any column whose name suggests an "
        "identifier or a post-outcome variable."))

a(("h2", "2.3. Stage A – Segmentation, dose and image processing"))
a(("p", "The thyroid gland was delineated manually on the planning CT by a radiation oncologist with 14 years of experience, slice "
        "by slice with review of adjacent slices to preserve three-dimensional continuity, and the contour defined the region of "
        "interest (ROI) for both image and dose analysis. {{Radiotherapy technique, prescription and fractionation, nodal fields "
        "(supraclavicular irradiation yes/no), CT scanner, tube voltage, slice thickness and use of contrast.}}"))

a(("h2", "2.4. Stage A – Feature groups"))
a(("p", "**Clinical biomarkers (CBs; n = 5):** age, diabetes mellitus, hypertension, smoking and CT-derived thyroid volume (cm³). "
        "Baseline TSH was not available as a predictor in the analysed data. **DVH parameters (n = 12):** mean dose (Dmean), "
        "maximum dose (Dmax) and V5 to V50 in 5-Gy steps, where Vx is the percentage of thyroid volume receiving at least x Gy. "
        "**CT radiomic features (CT RFs; n = 107):** shape, first-order and texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM) "
        "extracted from the thyroid ROI with 3D Slicer 4.11 [@fedorov2012] following the Image Biomarker Standardization Initiative "
        "[@zwanenburg2020], after resampling to 1 × 1 × 1 mm³ with a fixed gray-level bin width of 25. **Dosiomic features (DFs; "
        "n = 107):** the identical feature definitions were computed on the three-dimensional dose distribution inside the same "
        "ROI after resampling the dose grid to 1 × 1 × 1 mm³, with dose expressed in Gy [@liang2019]. Feature names were prefixed by "
        "group (CB_, DVH_, RF_, DF_) because the radiomic and dosiomic files share identical feature names."))

a(("h2", "2.5. Stage B – Data validation and feature-group configurations"))
a(("p", "The four source files (CB, DVH, radiomics, dosiomics) were merged on patient identifier, never on row position. The loader "
        "verified that the identifier sets and outcome labels were identical in all files, that identifiers were unique, that the "
        "outcome was binary, and that no feature was missing, non-finite or constant. Patient names present in the source files were "
        "used only for a consistency check and were not stored. One row per patient was available, so patient-level grouping of the "
        "cross-validation was unnecessary (the code switches to grouped splits automatically if identifiers ever repeat). The rows "
        "of the source files were sorted by outcome, which would invalidate unshuffled splits; all splits were shuffled and this "
        "was verified by an automated test."))
a(("p", "All 15 non-empty combinations of the four groups were analysed: four single groups (CBs, DVH, CT RFs, DFs), six pairs, four "
        "triples and the combination of all four. Each configuration was analysed independently with the full pipeline."))

a(("h2", "2.6. Stage C – Repeated nested cross-validation"))
a(("p", "Performance was estimated with nested stratified cross-validation [@varma2006; @vabalas2019]. The outer loop used shuffled "
        "stratified 4-fold splits (about 86 training and 29 test patients; 20–21 and 6–7 events, respectively), repeated three "
        "times with different random splits to reduce split-to-split variance; the same splits were used for every configuration, "
        "classifier and strategy, so all comparisons are paired. Within each outer training set, an inner stratified 3-fold "
        "cross-validation was used exclusively for hyper-parameter optimisation and threshold selection. No quantity computed from an "
        "outer test fold (features, labels or predictions) was used to select features, tune a model, set a threshold or scale data."))

a(("h2", "2.7. Stage C – Pre-processing and feature selection (training data only)"))
a(("p", "Within each training set, non-finite values were set to missing and imputed with the training median, features with standard "
        "deviation ≤ 10⁻¹⁰ were removed, and features were standardised with training-set means and standard deviations. Feature "
        "selection then proceeded in four stages (pure functions of the training rows, re-run inside every inner split): "
        "(1) *volume-dependence filter* – RF and DF features with |Spearman ρ| > 0.50 against thyroid volume were removed, because "
        "they may encode gland size rather than tissue or dose properties; (2) *redundancy filter* – within each feature group, "
        "features were ranked by univariate ROC-AUC and a feature was dropped if |Pearson r| > 0.60 with a better-ranked retained "
        "feature; (3) *mRMR* [@peng2005] – within each group, a greedy forward search maximised mutual information with the outcome minus "
        "the mean absolute correlation with already selected features, retaining up to 5 (CBs), 4 (DVH), 6 (CT RFs) and 6 (DFs) "
        "features; (4) *stability selection* [@meinshausen2010] – 500 repetitions of elastic-net logistic regression [@zou2005] on "
        "stratified 60% subsamples with randomly drawn penalty strength (C ∈ {0.001, 0.01, 0.05, 0.1, 0.5}) and mixing ratio "
        "(0.60–0.95), counting features with |coefficient| > 10⁻³; features selected in ≥ 75% of repetitions were retained. If "
        "fewer than two features passed, the threshold was lowered in steps of 0.01 to a floor of 0.55 and then the two most frequently "
        "selected features were taken; at most five features were kept (roughly one predictor per four training events), a parsimony cap that reflects the "
        "limited number of events relative to the number of candidate predictors [@riley2020]."))

a(("h2", "2.8. Stage C – Classifiers, class imbalance, hyper-parameter optimisation and threshold"))
a(("p", "The pre-specified primary model was L2-regularised logistic regression. Because the number of events is small, the main "
        "model set was restricted to parsimonious learners: L2 logistic regression, elastic-net logistic regression, linear support "
        "vector machine (Platt-scaled probabilities) and a shallow random forest. Eight further classifiers (RBF-SVM, extra-trees, "
        "gradient boosting, AdaBoost, k-nearest neighbours, Gaussian naïve Bayes, XGBoost and LightGBM) were evaluated without "
        "resampling as a sensitivity analysis. Three class-imbalance strategies were compared: none, cost-sensitive learning "
        "(balanced class weights) and SMOTE [@chawla2002] applied to training rows only inside the model pipeline (imputer, scaler, "
        "SMOTE, classifier [@lemaitre2017]); strategies not applicable to a learner (class weights for kNN, naïve Bayes, AdaBoost "
        "and gradient boosting) were skipped. The primary analysis used class weighting; this choice and the primary model were fixed "
        "in the analysis configuration before the results were generated, but the analysis was not formally registered."))
a(("p", "Hyper-parameters (for example the inverse regularisation strength C in 10⁻³–10, the elastic-net mixing ratio 0.1–0.9, or "
        "forest depth 2–4 and leaf size 2–8) were optimised with Optuna's tree-structured Parzen estimator [@akiba2019] with 30 trials "
        "per outer training set, maximising the mean ROC-AUC across the three inner folds; feature selection was re-run inside each "
        "inner split, so that tuning was never based on features chosen with inner validation data. Trials that failed were counted "
        "and the run aborted if all failed (none failed in the main analysis). The operating threshold for threshold-based metrics "
        "was the Youden-optimal cut-off [@youden1950] of the inner out-of-fold probabilities of the best trial; metrics at a fixed "
        "threshold of 0.5 are reported in addition. The final model was refitted on the whole outer training set with the selected "
        "features and best hyper-parameters and applied once to the held-out fold."))

a(("h2", "2.9. Leakage prevention and verification"))
a(("p", "Leakage safeguards were implemented in code and tested automatically: (i) imputation, scaling, SMOTE and classifier fitting "
        "occur in a pipeline fitted on training rows only; (ii) feature selection, hyper-parameter search and threshold selection use "
        "training rows only; (iii) identifiers, patient names, row order and outcome-related columns are excluded by rule; and "
        "(iv) the shuffled splits are reproducible and identical across analyses. Unit tests confirmed that scrambling the test-fold "
        "labels, or distorting the test-fold features, leaves the selected features, tuned hyper-parameters, thresholds and training "
        "AUC unchanged and does not alter the prediction of any other test patient; that scaler statistics equal the training "
        "statistics; and that SMOTE sees only training rows. A label-permutation analysis of the entire nested procedure (50 "
        "permutations; CBs, CT RFs, DVH and the four-group configuration; primary model) provided an empirical null distribution of "
        "the AUC [@ambroise2002]."))

a(("h2", "2.10. Stage D – Performance evaluation"))
a(("p", "For each repeat, out-of-fold predictions of all patients were pooled and metrics were computed; reported values are means "
        "over the three repeats (with the standard deviation across repeats as a measure of split sensitivity). Discrimination was "
        "quantified with ROC-AUC (primary metric) and PR-AUC (average precision); threshold-based performance with sensitivity, "
        "specificity, precision, F1 and balanced accuracy; and probabilistic accuracy with the Brier score and the calibration "
        "intercept and slope obtained by logistic recalibration of the out-of-fold probabilities [@vancalster2019; @steyerberg2010], "
        "complemented by reliability curves in quintiles (few bins because of 27 events). Decision-curve analysis [@vickers2006] "
        "estimated net benefit for threshold probabilities of 0.02–0.8. Uncertainty was expressed with 95% percentile intervals from "
        "1,000 patient-level bootstrap resamples (300 for PR-AUC) in which every repeat is evaluated on the same resample. These "
        "intervals reflect test-sample variability only and ignore variability of the training procedure; they are therefore "
        "optimistic. Curves in the figures use the patient-wise mean out-of-fold probability across repeats."))

a(("h2", "2.11. Stage D – Comparison of feature groups and sensitivity analyses"))
a(("p", "The comparisons that address the primary aim were DFs versus DVH, DVH+DFs versus DVH and, for completeness, all other "
        "configurations versus DVH, using the primary model. The difference in AUC was estimated with a paired patient-level bootstrap "
        "(2,000 resamples) on the repeat-averaged out-of-fold probabilities and reported with its 95% interval. DeLong's test "
        "[@delong1988] is not valid for pooled cross-validated predictions, because models trained on overlapping data are not independent, "
        "so no hypothesis tests or multiplicity-adjusted significance claims are made; interval estimates are descriptive. "
        "Sensitivity analyses repeated the comparison (i) across all 12 main classifier–imbalance combinations and the 8 further "
        "classifiers, (ii) with relaxed stability selection (threshold 0.5, at least three features; L2 logistic regression and linear "
        "SVM with class weighting), and (iii) via the permutation check described above. Patient characteristics (Table 1) were "
        "compared with Mann–Whitney and Fisher exact tests for description only, without correction for multiplicity."))

a(("h2", "2.12. Software and reproducibility"))
a(("p", "All analyses were performed in Python 3.11 with scikit-learn 1.9 [@pedregosa2011], imbalanced-learn 0.14, Optuna 5.0, "
        "XGBoost 3.2, LightGBM, pandas, SciPy and Matplotlib. A single random seed (111) and deterministic per-task seeds were used; "
        "re-running the pipeline reproduces the predictions exactly. The code, configuration, automated test suite, "
        "out-of-fold predictions and all tables are in the project repository ({{repository URL; no licence has been assigned}}). "
        "Patient-level data contain identifying information and cannot be shared publicly."))

# =============================================================== 3 RESULTS
a(("h1", "3. Results"))
a(("h2", "3.1. Cohort and descriptive analysis"))
a(("p", "The analysis included 115 patients, 27 (23.5%) of whom had hypothyroidism at one year. Table 1 summarises patient "
        "characteristics. Age, thyroid volume and comorbidities did not differ between groups, whereas thyroid dose did: "
        "median Dmean was 26.2 Gy (IQR 24.7–28.9) in patients with hypothyroidism and 19.3 Gy (15.4–22.0) in those without, and "
        "median V30 was 48.9% (46.6–54.1) versus 33.7% (24.7–40.0) (unadjusted p < 0.001). The median dose of the affected group lies "
        "above the 21-Gy value reported previously [@zhao2023; @kanyilmaz2017], that of the unaffected group below it. The "
        "apparent (unvalidated) single-feature AUC reflected the same pattern: the best DVH parameter (V30) reached 0.968, the best DF "
        "(first-order skewness) 0.932, the best CT RF 0.637 and the best CB (thyroid volume) 0.557 (Supplementary Table S2)."))
a(("table", ("table1", "Table 1. Patient characteristics by one-year outcome (descriptive; unadjusted tests, no multiplicity correction).")))

a(("h2", "3.2. Redundancy of dosiomic and DVH information"))
a(("p", "Dosiomic features were strongly related to DVH parameters. The first-order mean of the dose distribution had a Spearman "
        "correlation of 0.82 with Dmean, the first-order skewness correlated at −0.91 with V30, and 27 of the 107 dosiomic features "
        "had |ρ| ≥ 0.80 with at least one DVH parameter. By contrast, CT radiomic features were closely tied to gland size "
        "(maximum |ρ| with thyroid volume 0.93), which the volume filter removed (on average 22 of 107 CT RFs, 21%, and 34 of 107 DFs, 32%, per training set)."))

a(("h2", "3.3. Discrimination of the feature groups"))
a(("p", "Table 2 and Figure 2 present the pre-specified primary analysis (L2 logistic regression with class weighting). DVH "
        "parameters achieved an out-of-fold ROC-AUC of 0.953 (95% CI 0.907–0.985), PR-AUC 0.865, sensitivity 0.89 and specificity "
        "0.88 at the inner-CV threshold (balanced accuracy 0.89, F1 0.79). Dosiomic features reached an AUC of 0.897 (0.830–0.947; "
        "PR-AUC 0.736; sensitivity 0.80, specificity 0.86). The combination DVH+DFs achieved 0.946 (0.902–0.979; PR-AUC 0.843), and the "
        "combination of all four groups 0.939 (0.890–0.975). CBs (AUC 0.357) and CT RFs (0.413), alone or combined with each other "
        "(0.390), were below 0.5 and thus carried no usable information; an AUC below 0.5 is the expected behaviour of cross-validated "
        "AUC when features are uninformative and is consistent with the permutation null below. Configurations containing DVH "
        "parameters had AUCs of 0.92–0.95, those containing DFs but no DVH about 0.90, regardless of whether CBs or CT RFs were "
        "added (Figure 3a). The standard deviation of the AUC across the three outer repeats was ≤ 0.046 for every configuration "
        "(DVH 0.007, DFs 0.004, DVH+DFs 0.009)."))
a(("table", ("table2", "Table 2. Out-of-fold performance of the 15 feature-group configurations (primary model: L2 logistic regression "
             "with class weighting; mean of three outer repeats; threshold from inner cross-validation; 95% patient-bootstrap CI for AUC).")))
a(("fig", ("figure2_roc_pr", "Figure 2. Out-of-fold ROC (a) and precision–recall (b) curves for the four single feature groups and for "
           "DVH + DFs (primary model). The dashed line in (b) marks the prevalence (0.235). Internal cross-validation only.", 6.3)))

a(("h2", "3.4. Do dosiomic features improve on DVH parameters?"))
a(("p", "In the primary analysis the paired AUC difference between DFs and DVH was −0.047 (95% CI −0.089 to −0.011): the dosiomic "
        "model discriminated slightly worse than the DVH model. Adding DFs to DVH changed the AUC by −0.003 (−0.019 to 0.013), and "
        "adding CT RFs, CBs or both changed it by −0.011 (−0.031 to 0.005), −0.022 (−0.056 to 0.002) and −0.008 (−0.027 to 0.008), "
        "respectively; no combination improved on DVH alone (Figure 3b). Across all 20 evaluated classifier–imbalance combinations "
        "(Figure 4, Supplementary Table S1), DFs alone exceeded DVH in none (difference −0.081 to −0.020, median −0.056) and "
        "DVH+DFs exceeded DVH in 5 (difference −0.031 to +0.011, median −0.006); the best DF-only AUC (0.914, naïve Bayes) remained below "
        "the best DVH-only AUC (0.961, class-weighted elastic-net logistic regression), a comparison that is itself optimistic "
        "because the maximum is taken over many models. The few positive differences (naïve Bayes, LightGBM, XGBoost, RBF-SVM and "
        "unweighted elastic-net regression; ≤ +0.011) were within the resolution of this dataset."))
a(("fig", ("figure3_configs_forest", "Figure 3. (a) Out-of-fold ROC-AUC with 95% patient-bootstrap CI for the 15 configurations (primary "
           "model). (b) Paired AUC difference versus DVH alone (bootstrap 95% CI). Intervals reflect test-sample variability only.", 6.6)))
a(("fig", ("figure4_models_dvh_vs_df", "Figure 4. Out-of-fold ROC-AUC of DVH, DFs and DVH + DFs for each of the 20 classifier–imbalance "
           "combinations, ordered by the DVH result. DFs alone never exceeded DVH.", 5.4)))

a(("h2", "3.5. Selected features"))
a(("p", "Stability selection never reached the 75% threshold: in all 180 outer fits the threshold was lowered by the fallback "
        "(median final threshold 0.27) and the two most frequently selected features were retained. Selection was most stable "
        "for DVH features: DVH_Dmax was chosen in all 12 outer fits and V30 in 10 of 12 (Figure 5, Supplementary Table S3). In the DF-only "
        "configuration the most frequent features were GLCM cluster shade (10/12), first-order skewness (6/12) and first-order "
        "mean (5/12); in every configuration that contained DVH, V30 was the dominant feature (10/12) and DFs were selected "
        "less often (cluster shade and first-order mean 5/12 each in DVH+DFs). The CB-only model selected smoking (10/12) and "
        "thyroid volume (9/12), which merely reflects the enforced minimum of two features, since neither carries signal "
        "(AUC 0.357); CT RFs were selected inconsistently (most frequent feature 3/12). With relaxed stability selection (≥ 3 features), "
        "AUCs were similar or slightly lower (DVH 0.953, DFs 0.893, DVH+DFs 0.934, all four groups 0.931)."))
a(("fig", ("figure6_stability", "Figure 5. Feature-selection stability: proportion of outer training fits (4 folds × 3 repeats) in which each "
           "feature was selected, for the most frequently selected features of each configuration (primary pipeline).", 6.4)))

a(("h2", "3.6. Calibration and decision-curve analysis"))
a(("p", "The Brier score was 0.119 for DVH, 0.143 for DFs and 0.093 for DVH+DFs (0.269 and 0.267 for CBs and CT RFs, i.e. "
        "worse than the 0.180 of a model that always predicts the prevalence). Probabilities from class-weighted models over-estimated risk in the middle "
        "of the range (Figure 6a); the calibration slope was 2.54 for DVH, 1.65 for DFs and 1.16 for DVH+DFs (ideal 1). Decision "
        "curves showed that all three models had higher net benefit than treating all or no patients over most of the "
        "0.1–0.8 threshold range (Figure 6b); at a threshold of 0.2, net benefit was 0.113 (DVH), 0.100 (DFs) and 0.170 (DVH+DFs) versus "
        "0.043 for treat-all. The higher net benefit of DVH+DFs despite equal AUC reflects its better-calibrated probabilities rather "
        "than better ranking; because class-weighted probabilities were not recalibrated, the decision curves are descriptive only."))
a(("fig", ("figure5_calibration_dca", "Figure 6. (a) Calibration (observed event rate against mean predicted probability in quintiles) and (b) decision curves of "
           "the DVH, DFs and DVH + DFs models (primary model, out-of-fold predictions; probabilities not recalibrated).", 6.3)))

a(("h2", "3.7. Class imbalance, classifiers and permutation check"))
a(("p", "Imbalance handling changed discrimination marginally: mean AUC across the 15 configurations differed by ≤ 0.007 between no "
        "resampling, class weighting and SMOTE within every classifier (L2 logistic regression 0.821, 0.819 and 0.823; DVH-only AUC "
        "0.950, 0.953 and 0.950). Strategies mainly shifted sensitivity–specificity balance and probability scale (DVH Brier 0.104 without "
        "resampling, 0.119 with class weights and 0.127 with SMOTE). Regularised linear models performed at least as well as "
        "flexible learners (mean AUC over configurations: elastic-net logistic regression 0.835–0.840, linear SVM 0.844–0.847, "
        "random forest 0.810–0.817; RBF-SVM 0.842, extra-trees 0.825, XGBoost 0.807). The nested procedure produced no spurious "
        "discrimination: with permuted labels the AUC was 0.506 ± 0.077 for DVH and 0.500 ± 0.062 for the four-group configuration "
        "(observed 0.949 and 0.931; smallest attainable permutation p = 0.020 with 50 permutations), and 0.494 ± 0.059 and 0.485 ± 0.084 for "
        "CBs and CT RFs (observed 0.377 and 0.388, p = 0.98 and 0.90). The optimised regularisation strength for the DVH "
        "model was moderate (median C 0.25, IQR 0.08–1.62)."))

# =============================================================== 4 DISCUSSION
a(("h1", "4. Discussion"))
a(("h2", "4.1. Principal findings"))
a(("p", "In this cohort of 115 breast-cancer patients, dosiomic features did not predict one-year radiation-induced "
        "hypothyroidism better than conventional DVH parameters. Under strict nested cross-validation, DVH parameters alone achieved "
        "an AUC of about 0.95, dosiomic features alone about 0.90, and their combination was no better than DVH (ΔAUC −0.003; "
        "95% CI −0.019 to 0.013). The conclusion was stable across 20 classifier–imbalance combinations, in which dosiomics alone never "
        "exceeded DVH and its addition to DVH improved AUC by at most 0.011. Clinical biomarkers and CT radiomic features "
        "contained no usable information, and combining them with DVH or dosiomic features did not help. The primary hypothesis – "
        "that dosiomics improves on DVH – is therefore not supported by these data."))
a(("h2", "4.2. Why dosiomics added nothing here"))
a(("p", "Three explanations are consistent with the data. First, **the outcome is dominated by dose level**. The unaffected and "
        "affected groups differed by about 7 Gy in mean dose and 15 percentage points in V30, with the two medians on either side of "
        "the 21-Gy threshold described in breast radiotherapy [@zhao2023; @kanyilmaz2017]. When a single scalar summary of dose already "
        "separates the groups, spatial information has little left to explain. Second, **dosiomic features largely re-encode "
        "the DVH**: the first-order mean and skewness of the dose distribution correlated at 0.82 and −0.91 with Dmean and V30, "
        "and roughly a quarter of all dosiomic features correlated at ≥ 0.80 with a DVH parameter. The most frequently selected "
        "texture feature (GLCM cluster shade, a measure of asymmetry of co-occurring dose levels) may carry some independent "
        "information on the dose gradient across the gland, but it did not improve ranking when added to V30. Third, **statistical "
        "power is limited**: with 27 events only a few predictors can be estimated reliably [@riley2020], and differences of "
        "the order of 0.01 AUC cannot be resolved; the data nonetheless exclude a large benefit of dosiomics (the upper bound of the "
        "paired interval for DVH+DFs versus DVH was +0.013 AUC)."))
a(("p", "These results differ from reports in nasopharyngeal carcinoma, where dosiomic or combined models outperformed "
        "dose-volume models [@ren2021; @yang2023; @quan2024]. Several differences may explain the discrepancy. Field "
        "geometry and dose levels differ between NPC and breast radiotherapy, and in the NPC studies the dose-volume comparators "
        "performed modestly (AUC 0.61–0.67 [@ren2021; @quan2024]), leaving room for improvement, whereas the 12-parameter DVH used "
        "here already reaches 0.95. Moreover, gains in those studies were obtained with a single train/test split of 44 test patients "
        "[@quan2024] or modest effect sizes (AUC 0.70 versus 0.61 [@ren2021]), and benefits of dosiomics over dose-volume factors in "
        "other organs (lung [@liang2019], salivary glands [@gabrys2018], rectum, bladder and vagina [@lucia2021]) are likewise reported from "
        "cohorts of 70–153 patients. Dosiomic information may add more where a toxicity depends on "
        "the spatial pattern of dose within a large, heterogeneously irradiated organ [@ebert2021] than for a small organ with a "
        "single dominant dose–response relation."))
a(("h2", "4.3. Clinical biomarkers and CT radiomics"))
a(("p", "None of the five clinical variables was associated with the outcome (Table 1), univariate AUCs were at most 0.557, and "
        "the models were at or below chance (AUC 0.357; maximum 0.559 over 20 classifier–imbalance combinations), in agreement with "
        "the permutation analysis. The scope of this statement needs to be stated precisely: CBs here comprise age, diabetes, "
        "hypertension, smoking and thyroid volume. Elevated baseline TSH was an independent risk factor in a large prospective "
        "breast-cancer cohort [@zhao2023], and age and thyroid volume influenced risk in head-and-neck series [@lee2023; "
        "@mohammadnamdar2019]; baseline TSH and the nodal irradiation field were not available as predictors here, and the "
        "result should not be extended to them. CT radiomic features were uninformative (AUC 0.413; maximum 0.526 over all models), "
        "although the apparent single-feature AUC reached 0.637 – the typical profile of noise features picked out of a 107-feature "
        "set. Planning-CT texture of the thyroid may be unstable (non-contrast CT, resampling, small volume, strong dependence on gland "
        "size, with correlations up to 0.93 with volume in our data) [@zwanenburg2020]. Radiomics added value in NPC (AUC 0.714 [@quan2024]; "
        "C-index 0.671 [@yang2023]), but those cohorts were larger (145 to 1,223 patients) and the effects modest; our results show that in 115 patients "
        "no such signal could be detected."))
a(("h2", "4.4. Practical implications"))
a(("p", "A DVH summary of the thyroid (V30, Dmax, Dmean) was sufficient for prediction in this cohort and is available in every "
        "treatment-planning system; the results therefore support continued use of thyroid dose constraints (for example keeping Dmean "
        "below about 21 Gy [@zhao2023; @kanyilmaz2017]) rather than the introduction of dosiomic models for this indication. With "
        "specificity of 0.88 and sensitivity of 0.89 for the DVH model, the practical question is whether the observed performance "
        "can be reproduced in another centre, which this study cannot answer."))
a(("h2", "4.5. Methodological considerations"))
a(("p", "The results are valid in an internal sense because every data-dependent step was nested, tuning never used inner "
        "validation data for feature selection, the threshold came from training data, and a permutation test of the whole "
        "procedure returned the expected null. These safeguards matter: selection outside the validation loop inflates "
        "performance even in pure-noise data [@ambroise2002; @varma2006; @vabalas2019]. Three further points deserve attention. First, "
        "the 75% stability-selection criterion was never met; with about 21 training events and strongly collinear predictors the "
        "procedure effectively returned the two most frequently selected features, and the results were similar with a relaxed "
        "criterion, but the stability criterion should be reconsidered in future work. Second, class weighting and SMOTE do not "
        "improve discrimination but distort probabilities; calibration slopes of 0.91–2.54 among the DVH-containing configurations show that probabilities should not be interpreted as absolute risks and should be "
        "recalibrated before use in decision curves [@vancalster2019]. Third, the model with the highest AUC (DVH, class-weighted "
        "elastic-net) was identified on the same data and is optimistically biased; we therefore base conclusions on the "
        "pre-specified primary model and on the full grid of models, not on the best-performing one."))
a(("h2", "4.6. Limitations"))
a(("bullet", "Single-centre cohort with 27 events; all validation is internal. No external or temporal validation was performed, and "
             "the study is not powered to detect AUC differences of about 0.01."))
a(("bullet", "Confidence intervals reflect test-sample variability only and ignore variability of the training procedure; they are optimistic. "
             "The pairwise comparisons are descriptive and unadjusted for multiplicity."))
a(("bullet", "Baseline TSH, the nodal irradiation field, dose fractionation and thyroid-specific factors (for example thyroid antibodies, "
             "iodine status) were not available as predictors; the conclusion about clinical biomarkers is limited to the five variables studied."))
a(("bullet", "The outcome is a laboratory-defined event at a single time point (one year); later events and time-to-event information "
             "were not modelled, and misclassification of transient TSH elevations cannot be excluded. {{Outcome definition to be confirmed.}}"))
a(("bullet", "Features were extracted from one contour by one observer; the reproducibility of radiomic and dosiomic features "
             "(inter-observer, dose-grid resolution) was not assessed [@zwanenburg2020; @yang2023]."))
a(("bullet", "Stability selection used 500 instead of 2,000 repetitions and a minimum-feature fallback; a different selection strategy "
             "could change the selected dosiomic features, although a relaxed criterion gave similar results."))
a(("bullet", "Probabilities from class-weighted and SMOTE models were not recalibrated, which limits the interpretation of calibration and decision curves."))
a(("h2", "4.7. Future work"))
a(("p", "Priorities are external validation in an independent breast-radiotherapy cohort, inclusion of baseline TSH and nodal-field "
        "information, larger multicentre datasets that allow time-to-event modelling, assessment of the reproducibility of dosiomic features, "
        "and voxel-based or spatial dose models that go beyond global texture features [@ebert2021], together with recalibration and "
        "decision-analytic evaluation before any clinical use."))

# =============================================================== 5 CONCLUSIONS
a(("h1", "5. Conclusions"))
a(("p", "In 115 breast-cancer patients evaluated with repeated nested cross-validation, dosiomic features did not improve prediction of "
        "one-year radiation-induced hypothyroidism over conventional DVH parameters (DVH AUC 0.953, DFs 0.897, DVH+DFs 0.946; paired "
        "AUC difference for DVH+DFs versus DVH −0.003, 95% CI −0.019 to 0.013). Clinical biomarkers and CT radiomic features carried no "
        "predictive information. Thyroid V30, Dmax and mean dose were sufficient to rank patients by risk. These hypothesis-generating "
        "findings come from a small single-centre cohort and need external validation before any conclusion about clinical use."))

DECLARATIONS = [
    ("Ethics approval", "The study was approved by the institutional review board of Iran University of Medical Sciences "
     "(IR.IUMS.REC.1405.211). {{Consent to participate.}}"),
    ("Data and code availability", "Patient-level data contain identifying information and are not publicly available. Analysis code, "
     "configuration, tests, aggregate results and figures are available at {{repository URL}}."),
    ("Funding", "{{Funding statement}}"),
    ("Competing interests", "{{Competing-interest statement}}"),
    ("Author contributions", "{{Author contributions (CRediT)}}"),
    ("Acknowledgements", "{{Acknowledgements}}"),
]

ABBREV = ("AUC, area under the curve; CBs, clinical biomarkers; CT, computed tomography; DCA, decision-curve analysis; DFs, dosiomic "
          "features; Dmax/Dmean, maximum/mean thyroid dose; DVH, dose–volume histogram; IBSI, Image Biomarker Standardization Initiative; "
          "IQR, interquartile range; mRMR, minimum redundancy maximum relevance; NPC, nasopharyngeal carcinoma; NTCP, normal tissue "
          "complication probability; PR, precision–recall; RF(s), radiomic feature(s); RIHT, radiation-induced hypothyroidism; "
          "SMOTE, synthetic minority over-sampling technique; TSH, thyroid-stimulating hormone; Vx, percentage of thyroid volume "
          "receiving ≥ x Gy.")

REFS = {
    "solmunde2023": "Solmunde E, Falstie-Jensen AM, Lorenzen EL, Ewertz M, Reinertsen KV, Dekkers OM, et al. Breast cancer, breast cancer-directed radiation therapy and risk of hypothyroidism: a systematic review and meta-analysis. Breast. 2023;68:216–24.",
    "digkas2024": "Digkas E, Smith DR, Wennstig AK, Matikas A, Tegnelius E, Valachis A. Incidence and risk factors of hypothyroidism after treatment for early breast cancer: a population-based cohort study. Breast Cancer Res Treat. 2024;204:79–87.",
    "zhao2023": "Zhao XR, Fang H, Jing H, Tang Y, Song YW, Liu YP, et al. Radiation-induced hypothyroidism in patients with breast cancer after hypofractionated radiation therapy: a prospective cohort study. Int J Radiat Oncol Biol Phys. 2023;115(1):83–92.",
    "kanyilmaz2017": "Kanyilmaz G, Aktan M, Koc M, Demir H, Demir LS. Radiation-induced hypothyroidism in patients with breast cancer: a retrospective analysis of 243 cases. Med Dosim. 2017;42(3):190–6.",
    "vogelius2011": "Vogelius IR, Bentzen SM, Maraldo MV, Petersen PM, Specht L. Risk factors for radiation-induced hypothyroidism: a literature-based meta-analysis. Cancer. 2011;117(23):5250–60.",
    "chow2022": "Chow JC, Cheung KM, Cheung GT, Tam AH, Lui JC, Lee FK, et al. Dose-volume predictors of post-radiation primary hypothyroidism in head and neck cancer: a systematic review. Clin Transl Radiat Oncol. 2022;33:83–92.",
    "yang2023": "Yang SS, Peng QH, Wu AQ, Zhang BY, Liu ZQ, Chen EN, et al. Radiomics and dosiomics for predicting radiation-induced hypothyroidism and guiding intensity-modulated radiotherapy. iScience. 2023;26(12):108394.",
    "ebert2021": "Ebert MA, Gulliford S, Acosta O, et al. Spatial descriptions of radiotherapy dose: normal tissue complication models and statistical associations. Phys Med Biol. 2021;66(12):12TR01.",
    "lambin2012": "Lambin P, Rios-Velazquez E, Leijenaar R, Carvalho S, van Stiphout RGPM, Granton P, et al. Radiomics: extracting more information from medical images using advanced feature analysis. Eur J Cancer. 2012;48(4):441–6.",
    "gillies2016": "Gillies RJ, Kinahan PE, Hricak H. Radiomics: images are more than pictures, they are data. Radiology. 2016;278(2):563–77.",
    "liang2019": "Liang B, Yan H, Tian Y, Chen X, Yan L, Zhang T, et al. Dosiomics: extracting 3D spatial features from dose distribution to predict incidence of radiation pneumonitis. Front Oncol. 2019;9:269.",
    "gabrys2018": "Gabryś HS, Buettner F, Sterzing F, Hauswald H, Bangert M. Design and selection of machine learning methods using radiomics and dosiomics for normal tissue complication probability modeling of xerostomia. Front Oncol. 2018;8:35.",
    "lucia2021": "Lucia F, Bourbonne V, Visvikis D, Miranda O, Gujral DM, Gouders D, et al. Radiomics analysis of 3D dose distributions to predict toxicity of radiotherapy for cervical cancer. J Pers Med. 2021;11(5):398.",
    "ren2021": "Ren W, Liang B, Sun C, Wu R, Men K, Xu Y, et al. Dosiomics-based prediction of radiation-induced hypothyroidism in nasopharyngeal carcinoma patients. Phys Med. 2021;89:219–25.",
    "quan2024": "Quan KR, Lin WR, Hong JB, Lin YH, Chen KQ, Chen JH, et al. A machine learning approach for predicting radiation-induced hypothyroidism in patients with nasopharyngeal carcinoma undergoing tomotherapy. Sci Rep. 2024;14:8436.",
    "lee2023": "Lee TF, Lee SH, Tseng CD, Lin CH, Chiu CM, Lin GZ, et al. Using machine learning algorithm to analyse the hypothyroidism complications caused by radiotherapy in patients with head and neck cancer. Sci Rep. 2023;13:19185.",
    "mansouri2024": "Mansouri Z, Salimi Y, Amini M, Hajianfar G, Oveisi M, Shiri I, et al. Development and validation of survival prognostic models for head and neck cancer patients using machine learning and dosiomics and CT radiomics features: a multicentric study. Radiat Oncol. 2024;19:12.",
    "mohammadnamdar2019": "Mohammad-Namdar A, Sadeghi-Bazargani H, Kazemi-Motlagh AH, Mesbahi A, Mohammadzadeh M. Patient safety considerations using dose-response models in external radiation therapy of head-and-neck and breast cancers: clinical-dosimetric predictors of radiation-induced hypothyroidism. J Inj Violence Res. 2019;11(4 Suppl 2): Paper No. 183.",
    "ambroise2002": "Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. Proc Natl Acad Sci U S A. 2002;99(10):6562–6.",
    "varma2006": "Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91.",
    "vabalas2019": "Vabalas A, Gowen E, Poliakoff E, Casson AJ. Machine learning algorithm validation with a limited sample size. PLoS One. 2019;14(11):e0224365.",
    "collins2024": "Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.",
    "fedorov2012": "Fedorov A, Beichel R, Kalpathy-Cramer J, Finet J, Fillion-Robin JC, Pujol S, et al. 3D Slicer as an image computing platform for the Quantitative Imaging Network. Magn Reson Imaging. 2012;30(9):1323–41.",
    "zwanenburg2020": "Zwanenburg A, Vallières M, Abdalah MA, Aerts HJWL, Andrearczyk V, Apte A, et al. The Image Biomarker Standardization Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology. 2020;295(2):328–38.",
    "peng2005": "Peng H, Long F, Ding C. Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE Trans Pattern Anal Mach Intell. 2005;27(8):1226–38.",
    "meinshausen2010": "Meinshausen N, Bühlmann P. Stability selection. J R Stat Soc Series B Stat Methodol. 2010;72(4):417–73.",
    "zou2005": "Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc Series B Stat Methodol. 2005;67(2):301–20.",
    "riley2020": "Riley RD, Ensor J, Snell KIE, Harrell FE Jr, Martin GP, Reitsma JB, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441.",
    "chawla2002": "Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. J Artif Intell Res. 2002;16:321–57.",
    "lemaitre2017": "Lemaître G, Nogueira F, Aridas CK. Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning. J Mach Learn Res. 2017;18(17):1–5.",
    "akiba2019": "Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. New York: ACM; 2019. p. 2623–31.",
    "youden1950": "Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32–5.",
    "vancalster2019": "Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230.",
    "steyerberg2010": "Steyerberg EW, Vickers AJ, Cook NR, Gerds T, Gonen M, Obuchowski N, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. Epidemiology. 2010;21(1):128–38.",
    "vickers2006": "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565–74.",
    "delong1988": "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837–45.",
    "pedregosa2011": "Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–30.",
}

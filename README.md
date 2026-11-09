# 🧬 RIHT Prediction in Breast Cancer Radiotherapy

### A Machine Learning Framework for Comparing Clinical, Dosimetric, Radiomic, and Dosiomic Predictors

<p align="center">

**Radiation Oncology · Medical Physics · Machine Learning · Predictive Analytics**

</p>

---

## 🔬 Project Overview

Can advanced dose-distribution features improve the prediction of radiation-induced hypothyroidism beyond conventional dose–volume metrics?

This project investigates the predictive value of **clinical biomarkers, dose–volume histogram parameters, CT radiomics, and dosiomics** for predicting one-year radiation-induced hypothyroidism (RIHT) following breast cancer radiotherapy.

A reproducible machine-learning pipeline was developed to systematically evaluate individual feature groups and their combinations using leakage-controlled nested cross-validation.

Rather than assuming that more complex feature representations necessarily improve prediction, this study examines whether spatial and higher-order dose descriptors provide additional predictive information beyond conventional dosimetric parameters.

### 🎯 Research Objectives

- Evaluate the predictive contribution of four complementary feature groups.
- Compare all **15 non-empty feature-group combinations**.
- Investigate the incremental value of dosiomics over conventional DVH parameters.
- Assess the robustness of predictive performance across classifiers and class-imbalance strategies.
- Implement a leakage-controlled machine-learning workflow.
- Provide transparent computational materials to facilitate reproducibility.

---

## 📊 Study Snapshot

| Characteristic | Description |
|---|---|
| Clinical application | Radiation-induced hypothyroidism prediction |
| Patient cohort | 115 patients |
| RIHT events | 27 (23.5%) |
| Follow-up | 1 year |
| Study design | Prospective, single-center |
| Feature groups | 4 |
| Feature configurations | 15 |
| Validation framework | Repeated nested cross-validation |
| Hyperparameter optimization | Optuna |
| Programming language | Python |

> ⚠️ **Validation status:** All reported performance estimates are based on internal cross-validation. No external validation has been performed. Results are exploratory and should not be interpreted as evidence of clinical readiness.

---

## 🧩 Feature Engineering and Data Domains

Four complementary feature domains are investigated.

| Feature group | Description | Role |
|---|---|---|
| 🩺 **CBs** | Clinical biomarkers and patient characteristics | Clinical information |
| ☢️ **DVH** | Dose–volume histogram parameters | Conventional dosimetry |
| 🖥️ **CT RFs** | CT-derived radiomic features | Anatomical imaging |
| 🧬 **DFs** | Spatial and statistical dose-distribution features | Advanced dosimetry |

### 🔀 Feature Configuration Strategy

All possible non-empty combinations of the four feature groups are evaluated.

This produces 15 configurations, allowing systematic comparison of individual feature domains and their combinations.

The analysis is designed to determine whether advanced imaging- and dose-derived features provide meaningful incremental predictive information beyond conventional dosimetric parameters.

---

## ⚙️ Machine Learning Architecture

The computational framework integrates preprocessing, feature selection, hyperparameter optimization, model training, and evaluation within a nested cross-validation pipeline.

### 🔄 Analytical Workflow

```text
          Clinical and Imaging Data
                     │
                     ▼
              Data Validation
                     │
                     ▼
          Feature Group Integration
                     │
                     ▼
          15 Feature Configurations
                     │
                     ▼
          Repeated Nested Cross-Validation
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Inner CV               Outer CV
          │                     │
          ▼                     │
   Feature Selection            │
          │                     │
          ▼                     │
   Hyperparameter Tuning        │
          │                     │
          ▼                     │
   Threshold Optimization       │
          │                     │
          └──────────┬──────────┘
                     ▼
              Model Evaluation
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        ROC-AUC   Calibration   DCA
          │          │          │
          └──────────┼──────────┘
                     ▼
             Statistical Analysis
                     │
                     ▼
          Reproducible Results
```

### 🧠 Feature Selection

Feature selection is performed exclusively within training data to minimize information leakage.

The implemented pipeline includes:

1. Median imputation.
2. Zero-variance filtering.
3. Thyroid-volume correlation filtering for CT radiomic and dosiomic features.
4. Correlation-based redundancy filtering.
5. Minimum Redundancy Maximum Relevance (mRMR).
6. Elastic-net-based stability selection.

Feature selection is repeated within the inner cross-validation training folds during model optimization.

### 🤖 Machine Learning Algorithms

**Primary classifier set**

- Logistic Regression (L2)
- Logistic Regression (Elastic Net)
- Linear Support Vector Machine
- Shallow Random Forest

**Extended classifier set**

- RBF Support Vector Machine
- Extra Trees
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Naive Bayes
- XGBoost
- LightGBM

### ⚖️ Class-Imbalance Strategies

Three strategies are investigated:

- No resampling
- Class weighting
- Synthetic Minority Over-sampling Technique (SMOTE)

The pre-specified primary analysis uses L2 Logistic Regression with class weighting.

SMOTE is applied only within the training pipeline.

### 🎛️ Hyperparameter Optimization

Hyperparameter optimization is performed using **Optuna** with a Tree-structured Parzen Estimator (TPE) sampler.

The optimization objective is the mean ROC-AUC across inner cross-validation folds.

The outer test folds are excluded from feature selection, hyperparameter tuning, and threshold optimization.

---

## 🔬 Validation and Statistical Framework

### Nested Cross-Validation

| Component | Configuration |
|---|---|
| Outer CV | 4-fold stratified |
| Outer repetitions | 3 |
| Inner CV | 3-fold stratified |
| Optimization | Optuna |
| Optimization trials | 30 |
| Primary model | L2 Logistic Regression |
| Primary imbalance strategy | Class weighting |

Identical outer splits are used across feature configurations and models to facilitate paired comparisons.

### 📈 Model Performance Metrics

The following metrics are calculated:

- ROC-AUC
- PR-AUC (Average Precision)
- Sensitivity
- Specificity
- Precision
- F1-score
- Accuracy
- Balanced Accuracy
- Brier Score
- Calibration slope and intercept

### 📉 Additional Analyses

The framework also includes:

- ROC and precision–recall analysis.
- Confusion matrices.
- Calibration and reliability curves.
- Decision Curve Analysis (DCA).
- Patient-level bootstrap confidence intervals.
- Paired bootstrap comparisons of ΔAUC.
- Feature-selection stability analysis.
- Training versus test performance assessment.
- Label-permutation testing.
- Sensitivity analyses across classifiers and imbalance strategies.

---

## 🗂️ Repository Structure

```text
RIHT-Prediction-Breast-Cancer/
│
├── 📁 config/
│   └── config.yaml
│
├── 📁 data/
│   └── Local patient-level data (not publicly distributed)
│
├── 📁 thyroid_rihp/
│   ├── data.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── nested_cv.py
│   ├── evaluation.py
│   ├── stats.py
│   ├── reporting.py
│   └── visualization.py
│
├── 📁 scripts/
│   ├── validate_data.py
│   ├── run_nested_cv.py
│   ├── make_figures.py
│   ├── run_permutation_test.py
│   ├── fit_final_model.py
│   └── build_improved_notebook.py
│
├── 📁 notebooks/
│   └── thyroid_improved.ipynb
│
├── 📁 tests/
│
├── 📁 reports/
│
├── 📁 outputs/
│
├── 📁 figures/
│
├── 📁 supplementary/
│
├── 📄 requirements.txt
├── 📄 pytest.ini
├── 📄 .gitignore
└── 📄 README.md
```

---

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ElmiraYazdani/RIHT-Prediction-Breast-Cancer.git

cd RIHT-Prediction-Breast-Cancer
```

### 2. Create a virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### 🧪 1. Validate the input data

```bash
python scripts/validate_data.py
```

### 🧫 2. Run automated tests

```bash
python -m pytest -q
```

### ⚡ 3. Run a preliminary smoke test

```bash
python scripts/run_nested_cv.py --run-name smoke --smoke
```

### 🚀 4. Execute the primary analysis

```bash
python scripts/run_nested_cv.py --run-name main
```

This executes the main nested cross-validation workflow across the 15 feature configurations, primary classifier set, and class-imbalance strategies.

### 📊 5. Generate analytical figures and tables

```bash
python scripts/make_figures.py --run-name main
```

### 🔍 6. Perform label-permutation testing

```bash
python scripts/run_permutation_test.py --n-perm 50
```

### 🤖 7. Run extended classifier analyses

```bash
python scripts/run_nested_cv.py --run-name extended --models extended --strategies none
```

### 🧮 8. Fit a descriptive final model

```bash
python scripts/fit_final_model.py --config DVH
```

> Note: The all-data fitted model is descriptive and must not be used to estimate unbiased predictive performance.

---

### 🔁 Reproducibility

The computational workflow incorporates:

- Centralized configuration through `config/config.yaml`.
- Fixed random seeds.
- Deterministic task-level seed assignment.
- Version-controlled dependencies.
- Automated testing.
- Training-only preprocessing and feature selection.
- Cached computational tasks for recovery after interruptions.

The objective is to facilitate reproducibility of the analytical workflow using compatible software versions and appropriately authorized input data.

---

## ⚠️ Limitations

Several limitations should be considered:

- The cohort is relatively small and contains 27 RIHT events.
- The analysis is based on a single-center population.
- External and temporal validation have not been performed.
- DVH and dosiomic features may contain overlapping information.
- Class imbalance can affect classification thresholds and probability estimates.
- The best-performing configuration is identified within the same cohort used for evaluation, introducing potential model-selection optimism.

Accordingly, the findings should be interpreted as exploratory and require independent validation before clinical translation.

---

## 🔐 Data Availability and Ethics

Patient-level clinical data and original Excel datasets are not included in this public repository because of confidentiality and privacy restrictions.

The repository provides the computational source code, configuration files, and permitted derived analytical outputs.

Patient-level out-of-fold predictions and potentially identifiable information are not publicly distributed.

**Ethical approval:** Institutional Review Board of Iran University of Medical Sciences (IR.IUMS.REC.1405.211).

---

## 📚 Citation

If you use this repository or adapt its computational workflow, please cite the associated research article:

**Yazdani E, et al.**

*Dose–Volume Parameters Outperform Dosiomics for Predicting One-Year Radiation-Induced Hypothyroidism in Breast Cancer Radiotherapy: A Machine Learning Study Using Leakage-Free Nested Cross-Validation.*

Manuscript submitted for publication. Final bibliographic information will be added following publication.

---

## 👩‍🔬 Author

**Elmira Yazdani, PhD**

Department of Medical Physics  
Faculty of Medicine  
Iran University of Medical Sciences (IUMS)

📧 yazdani.el@iums.ac.ir

---

## 📜 License

No open-source license has currently been assigned. All rights are reserved unless otherwise specified. Please contact the author for permission to reuse or redistribute the source code.

---

<p align="center">

**🧬 Integrating Clinical Data, Imaging Biomarkers, and Dosimetry for Reproducible Predictive Modeling in Radiation Oncology.**

</p>

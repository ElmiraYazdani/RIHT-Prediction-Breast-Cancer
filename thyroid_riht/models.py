"""Classifier registry, Optuna search spaces and leakage-safe model pipelines.

Every pipeline is  median-imputer -> StandardScaler -> [SMOTE] -> classifier  (imblearn Pipeline), so
imputation, scaling and over-sampling are re-fitted inside every training fold and SMOTE is never applied
to validation/test rows.

Imbalance strategies
--------------------
none          plain fit
class_weight  cost-sensitive learning (class_weight='balanced' / scale_pos_weight); returns ``None`` for
              estimators that cannot use class weights (kNN, GaussianNB, AdaBoost, GradientBoosting), in
              which case the run is skipped and reported as "not applicable".
smote         SMOTE on the training portion only (k_neighbors adapts to the minority count)
"""
from __future__ import annotations

import warnings

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:  # optional
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:  # optional
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


class SafeSMOTE(SMOTE):
    """SMOTE whose k_neighbors is capped at (minority_count - 1) so tiny folds do not crash."""

    def fit_resample(self, X, y):
        minority = int(np.min(np.bincount(np.asarray(y).astype(int))))
        if minority < 2:                       # cannot interpolate: return data unchanged
            return X, y
        self.k_neighbors = max(1, min(5, minority - 1))
        return super().fit_resample(X, y)


# --------------------------------------------------------------------------------------
# model specifications: (suggest(trial) -> params, build(params, seed, class_weighted, y) -> estimator)
# --------------------------------------------------------------------------------------
def _lr_l2_suggest(t):
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True)}


def _lr_l2_build(p, seed, cw, y):
    return LogisticRegression(l1_ratio=0.0, C=p["C"], solver="lbfgs", max_iter=5000,
                              class_weight="balanced" if cw else None, random_state=seed)


def _lr_en_suggest(t):
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True), "l1_ratio": t.suggest_float("l1_ratio", 0.1, 0.9)}


def _lr_en_build(p, seed, cw, y):
    return LogisticRegression(l1_ratio=p["l1_ratio"], C=p["C"], solver="saga", max_iter=5000, tol=1e-4,
                              class_weight="balanced" if cw else None, random_state=seed)


def _svml_suggest(t):
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True)}


def _svml_build(p, seed, cw, y):
    return SVC(kernel="linear", C=p["C"], probability=True, class_weight="balanced" if cw else None,
               random_state=seed)


def _svmr_suggest(t):
    return {"C": t.suggest_float("C", 1e-2, 10.0, log=True), "gamma": t.suggest_float("gamma", 1e-3, 1.0, log=True)}


def _svmr_build(p, seed, cw, y):
    return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], probability=True,
               class_weight="balanced" if cw else None, random_state=seed)


def _rf_suggest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": t.suggest_int("max_depth", 2, 4),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 8),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2"])}


def _rf_build(p, seed, cw, y):
    return RandomForestClassifier(**p, class_weight="balanced" if cw else None, random_state=seed, n_jobs=1)


def _et_build(p, seed, cw, y):
    return ExtraTreesClassifier(**p, class_weight="balanced" if cw else None, random_state=seed, n_jobs=1)


def _gb_suggest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 25, 100, step=25),
            "max_depth": t.suggest_int("max_depth", 1, 3),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": t.suggest_float("subsample", 0.7, 1.0)}


def _gb_build(p, seed, cw, y):
    return GradientBoostingClassifier(**p, random_state=seed)


def _ada_suggest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 25, 200, step=25),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 1.0, log=True)}


def _ada_build(p, seed, cw, y):
    return AdaBoostClassifier(**p, random_state=seed)


def _knn_suggest(t):
    return {"n_neighbors": t.suggest_int("n_neighbors", 3, 15, step=2),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"])}


def _knn_build(p, seed, cw, y):
    return KNeighborsClassifier(**p, n_jobs=1)


def _nb_suggest(t):
    return {"var_smoothing": t.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)}


def _nb_build(p, seed, cw, y):
    return GaussianNB(**p)


def _xgb_suggest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": t.suggest_int("max_depth", 1, 3),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True)}


def _xgb_build(p, seed, cw, y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if cw else 1.0
    return XGBClassifier(**p, scale_pos_weight=spw, eval_metric="logloss", verbosity=0, n_jobs=1,
                         random_state=seed)


def _lgb_suggest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": t.suggest_int("max_depth", 1, 3),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 3, 8),
            "min_child_samples": t.suggest_int("min_child_samples", 3, 15),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True)}


def _lgb_build(p, seed, cw, y):
    return LGBMClassifier(**p, class_weight="balanced" if cw else None, verbose=-1, n_jobs=1, random_state=seed)


# name -> (suggest, build, supports_class_weight, available)
MODEL_REGISTRY = {
    "logreg_l2":         (_lr_l2_suggest, _lr_l2_build, True, True),
    "logreg_enet":       (_lr_en_suggest, _lr_en_build, True, True),
    "svm_linear":        (_svml_suggest, _svml_build, True, True),
    "svm_rbf":           (_svmr_suggest, _svmr_build, True, True),
    "random_forest":     (_rf_suggest, _rf_build, True, True),
    "extra_trees":       (_rf_suggest, _et_build, True, True),
    "gradient_boosting": (_gb_suggest, _gb_build, False, True),
    "adaboost":          (_ada_suggest, _ada_build, False, True),
    "knn":               (_knn_suggest, _knn_build, False, True),
    "naive_bayes":       (_nb_suggest, _nb_build, False, True),
    "xgboost":           (_xgb_suggest, _xgb_build, True, XGBClassifier is not None),
    "lightgbm":          (_lgb_suggest, _lgb_build, True, LGBMClassifier is not None),
}

STRATEGIES = ("none", "class_weight", "smote")


def is_applicable(model_name: str, strategy: str) -> bool:
    spec = MODEL_REGISTRY[model_name]
    if not spec[3]:
        return False
    if strategy == "class_weight" and not spec[2]:
        return False
    return strategy in STRATEGIES


def suggest_params(model_name: str, trial) -> dict:
    return MODEL_REGISTRY[model_name][0](trial)


def build_pipeline(model_name: str, params: dict, strategy: str, seed: int, y_train=None) -> Pipeline:
    """Return an unfitted leakage-safe pipeline."""
    if not is_applicable(model_name, strategy):
        raise ValueError(f"Strategy '{strategy}' not applicable to model '{model_name}'")
    _, build, _, _ = MODEL_REGISTRY[model_name]
    y_train = np.asarray(y_train) if y_train is not None else np.array([0, 1])
    clf = build(params, seed, strategy == "class_weight", y_train)
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if strategy == "smote":
        steps.append(("smote", SafeSMOTE(random_state=seed)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def predict_proba_pos(pipe: Pipeline, X) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.predict_proba(X)[:, 1]

"""End-to-end smoke tests (synthetic data, and the real data when it is available)."""
import pandas as pd
import pytest

from conftest import make_synthetic
from thyroid_rihp.config import PROJECT_ROOT, load_config
from thyroid_rihp.data import load_dataset
from thyroid_rihp.nested_cv import run_nested_cv
from thyroid_rihp.reporting import metrics_summary, save_run_outputs


def test_end_to_end_synthetic(cfg, synth, tmp_path):
    combos = [("DVH",), ("CB", "DVH", "RF", "DF")]
    res = run_nested_cv(cfg, synth, ["logreg_l2", "random_forest"], ["none", "class_weight", "smote"], tmp_path,
                        combos=combos, n_jobs=2, use_cache=False, verbose=False)
    preds, folds = res["preds"], res["folds"]
    # every patient predicted exactly once per (config, model, strategy, repeat): proper out-of-fold coverage
    cnt = preds.groupby(["config", "model", "strategy", "repeat"]).patient.agg(["count", "nunique"])
    assert (cnt["count"] == 64).all() and (cnt["nunique"] == 64).all()
    assert preds.groupby("patient").y_true.nunique().max() == 1
    assert len(folds) == 2 * 2 * 3 * 4                                   # configs x models x strategies x folds
    assert folds["n_features"].between(1, cfg["feature_selection"]["stability"]["max_features"]).all()
    assert folds["n_failed_trials"].max() == 0
    # features of the DVH-only config must come from DVH only, never from other groups
    f = folds[folds.config == "DVH"].features.str.split("|").explode()
    assert f.str.startswith("DVH_").all()
    out = save_run_outputs(res, tmp_path, cfg)
    s = out["metrics_summary"]
    assert {"roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "brier",
            "roc_auc_ci_low", "roc_auc_ci_high", "calibration_slope"} <= set(s.columns)
    assert (s.roc_auc_ci_low <= s.roc_auc).all() and (s.roc_auc <= s.roc_auc_ci_high).all()
    assert s.loc[s.config == "DVH", "roc_auc"].max() > 0.75               # synthetic signal is recovered
    for f_ in ("metrics_summary.csv", "per_fold_metrics.csv", "selected_features_per_fold.csv",
               "optimized_hyperparameters.csv", "feature_selection_stability.csv"):
        assert (tmp_path / "tables" / f_).exists()
    assert (tmp_path / "predictions" / "oof_predictions.csv").exists()


def test_results_are_reproducible(cfg, synth, tmp_path):
    kw = dict(combos=[("DVH",)], n_jobs=1, use_cache=False, verbose=False)
    a = run_nested_cv(cfg, synth, ["logreg_l2"], ["none"], tmp_path, **kw)["preds"]
    b = run_nested_cv(cfg, synth, ["logreg_l2"], ["none"], tmp_path, **kw)["preds"]
    pd.testing.assert_frame_equal(a, b)


def test_chance_level_when_no_signal(cfg, tmp_path):
    """With no signal anywhere the nested-CV AUC must hover around 0.5 - a guard against hidden leakage."""
    ds = make_synthetic(80, 20, seed=5, signal=False)
    res = run_nested_cv(cfg, ds, ["logreg_l2"], ["class_weight"], tmp_path, combos=[("DVH", "RF", "DF")],
                        n_jobs=1, use_cache=False, verbose=False)
    s = metrics_summary(res["preds"], res["folds"], 50, 0, 0.95)
    assert 0.3 < s.roc_auc.iloc[0] < 0.7


REAL = PROJECT_ROOT / "data" / "CB_Thyroid.xlsx"


@pytest.mark.skipif(not REAL.exists(), reason="confidential data not present")
def test_real_data_smoke(tmp_path):
    cfg = load_config(overrides={"cv": {"outer_repeats": 1}, "optuna": {"n_trials": 3},
                                 "feature_selection": {"stability": {"n_iterations": 25}}})
    ds = load_dataset(cfg)
    res = run_nested_cv(cfg, ds, ["logreg_l2"], ["class_weight"], tmp_path,
                        combos=[("CB",), ("DVH",), ("RF", "DF")], n_jobs=3, use_cache=False, verbose=False)
    cnt = res["preds"].groupby(["config", "model", "strategy", "repeat"]).patient.nunique()
    assert (cnt == 115).all()
    s = metrics_summary(res["preds"], res["folds"], 50, 0, 0.95).set_index("config")
    assert s.loc["DVH", "roc_auc"] > 0.8 and s.loc["CBs", "roc_auc"] < 0.75

"""Turn raw nested-CV tables into the summary tables saved under outputs/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import compute_metrics, summarize_runs_with_ci
from .stats import mean_prob_by_patient

KEYS = ["config", "model", "strategy"]


def metrics_summary(preds: pd.DataFrame, folds: pd.DataFrame, B: int, seed: int, level: float) -> pd.DataFrame:
    """One row per (config, model, strategy): mean over outer repeats of pooled-OOF metrics with bootstrap CI."""
    rows = []
    for key, d in preds.groupby(KEYS, sort=False):
        res = summarize_runs_with_ci(d, B, seed, level)
        row = dict(zip(KEYS, key))
        row["n_patients"] = d["patient"].nunique()
        row["n_events"] = int(d.drop_duplicates("patient")["y_true"].sum())
        row["n_repeats"] = d["repeat"].nunique()
        for k, (m, lo, hi, sd) in res.items():
            row[k] = m
            row[f"{k}_ci_low"], row[f"{k}_ci_high"], row[f"{k}_sd_repeats"] = lo, hi, sd
        # calibration of the repeat-averaged probability
        mp = mean_prob_by_patient(d)
        cm = compute_metrics(mp["y_true"], mp["y_prob"], 0.5, with_calibration=True)
        row["calibration_intercept"], row["calibration_slope"] = cm["calibration_intercept"], cm["calibration_slope"]
        f = folds[(folds.config == key[0]) & (folds.model == key[1]) & (folds.strategy == key[2])]
        row["mean_n_features"] = f["n_features"].mean()
        row["fold_auc_mean"], row["fold_auc_sd"] = f["test_auc"].mean(), f["test_auc"].std(ddof=1)
        row["train_auc_mean"] = f["train_auc"].mean()
        row["inner_cv_score_mean"] = f["inner_cv_score"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def feature_stability(selection: pd.DataFrame) -> pd.DataFrame:
    """Selection proportion of every feature across all outer folds x repeats, per configuration."""
    n_tasks = selection.groupby("config")[["repeat", "fold"]].apply(lambda d: len(d.drop_duplicates()))
    g = selection.groupby(["config", "feature"]).agg(times_selected=("selected", "sum"),
                                                     mean_stability_freq=("stability_freq", "mean")).reset_index()
    g["n_outer_fits"] = g["config"].map(n_tasks)
    g["selection_proportion"] = g["times_selected"] / g["n_outer_fits"]
    return g.sort_values(["config", "selection_proportion"], ascending=[True, False])


def hyperparameter_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR of each tuned hyper-parameter over outer fits."""
    rows = []
    for key, d in folds.groupby(KEYS):
        pdf = pd.DataFrame([json.loads(s) for s in d["best_params"]])
        for c in pdf.columns:
            col = pdf[c]
            if pd.api.types.is_numeric_dtype(col):
                rows.append({**dict(zip(KEYS, key)), "param": c, "median": col.median(),
                             "q25": col.quantile(.25), "q75": col.quantile(.75), "min": col.min(), "max": col.max()})
            else:
                vc = col.value_counts()
                rows.append({**dict(zip(KEYS, key)), "param": c, "median": vc.index[0], "q25": np.nan,
                             "q75": np.nan, "min": np.nan, "max": np.nan})
    return pd.DataFrame(rows)


def save_run_outputs(res: dict, out_dir: Path, cfg: dict) -> dict:
    """Write all tables. Patient-level files go to predictions/ (git-ignored)."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "hpo").mkdir(parents=True, exist_ok=True)
    ev = cfg["evaluation"]
    summ = metrics_summary(res["preds"], res["folds"], ev["bootstrap_n"], ev["bootstrap_seed"], ev["ci_level"])
    summ.to_csv(out_dir / "tables" / "metrics_summary.csv", index=False)
    res["folds"].drop(columns=["features", "best_params"]).to_csv(out_dir / "tables" / "per_fold_metrics.csv", index=False)
    res["folds"][KEYS + ["repeat", "fold", "n_features", "features"]].to_csv(
        out_dir / "tables" / "selected_features_per_fold.csv", index=False)
    res["folds"][KEYS + ["repeat", "fold", "best_params", "inner_cv_score", "threshold", "n_failed_trials"]].to_csv(
        out_dir / "tables" / "optimized_hyperparameters.csv", index=False)
    hyperparameter_summary(res["folds"]).to_csv(out_dir / "tables" / "hyperparameter_summary.csv", index=False)
    feature_stability(res["selection"]).to_csv(out_dir / "tables" / "feature_selection_stability.csv", index=False)
    res["stages"].to_csv(out_dir / "tables" / "feature_selection_stage_counts.csv", index=False)
    res["preds"].to_csv(out_dir / "predictions" / "oof_predictions.csv", index=False)
    res["trials"].to_csv(out_dir / "hpo" / "optuna_trials.csv", index=False)
    return {"metrics_summary": summ}

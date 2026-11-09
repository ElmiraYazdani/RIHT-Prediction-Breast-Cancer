"""Create all tables-derived figures and statistical comparison tables from a finished run.

Usage:  python scripts/make_figures.py --run-name main
Outputs: outputs/<run>/figures/*.png|pdf and outputs/<run>/tables/pairwise_*.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from thyroid_rihp import visualization as V
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import load_dataset
from thyroid_rihp.stats import pairwise_comparison, vs_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--skip-tsne", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    run = resolve(cfg, "output_dir") / a.run_name
    fig_dir, tab_dir = run / "figures", run / "tables"
    preds = pd.read_csv(run / "predictions" / "oof_predictions.csv")
    folds = pd.read_csv(tab_dir / "per_fold_metrics.csv")
    summ = pd.read_csv(tab_dir / "metrics_summary.csv")
    stab = pd.read_csv(tab_dir / "feature_selection_stability.csv")
    folds_full = pd.read_csv(tab_dir / "optimized_hyperparameters.csv")
    trials = pd.read_csv(run / "hpo" / "optuna_trials.csv")
    pm, ps = cfg["models"]["primary_model"], cfg["models"]["primary_strategy"]
    th = np.linspace(*cfg["evaluation"]["dca_thresholds"][:2], int(cfg["evaluation"]["dca_thresholds"][2]))
    models = list(dict.fromkeys(summ.model))
    strategies = list(dict.fromkeys(summ.strategy))
    made = []

    prim = V.select(preds, pm, ps)
    tag = f"{pm}_{ps}"
    made += V.plot_roc(prim, f"ROC (out-of-fold): {pm} / {ps}", fig_dir / f"roc_all_configs_{tag}")
    made += V.plot_pr(prim, f"Precision-recall (out-of-fold): {pm} / {ps}", fig_dir / f"pr_all_configs_{tag}")
    made += V.plot_confusion_grid(prim, f"Confusion matrices, inner-CV threshold (mean over repeats): {pm} / {ps}",
                                  fig_dir / f"confusion_all_configs_{tag}")
    made += V.plot_confusion_grid(prim, f"Confusion matrices at fixed 0.5 threshold: {pm} / {ps}",
                                  fig_dir / f"confusion_all_configs_{tag}_thr0.5", col="y_pred_05")
    made += V.plot_calibration(prim, f"Calibration (out-of-fold): {pm} / {ps}", fig_dir / f"calibration_all_configs_{tag}",
                               n_bins=cfg["evaluation"]["calibration_bins"])
    made += V.plot_dca(prim, f"Decision curves (out-of-fold): {pm} / {ps}", fig_dir / f"decision_curves_{tag}", th)
    made += V.plot_fold_auc(folds, pm, ps, fig_dir / f"fold_auc_train_test_{tag}")
    made += V.plot_config_auc(summ, pm, ps, fig_dir / f"config_auc_ci_{tag}")
    made += V.plot_model_heatmap(summ, "roc_auc", fig_dir / "heatmap_auc_config_model_strategy")
    made += V.plot_model_heatmap(summ, "pr_auc", fig_dir / "heatmap_prauc_config_model_strategy")
    made += V.plot_model_heatmap(summ, "brier", fig_dir / "heatmap_brier_config_model_strategy")
    made += V.plot_strategy_comparison(summ, fig_dir / "strategy_comparison_auc")
    made += V.plot_feature_stability(stab, fig_dir / "feature_selection_stability_heatmap")
    made += V.plot_feature_stability_top(stab, fig_dir / "feature_stability_top20")
    made += V.plot_hpo_history(trials, pm, ps, fig_dir / f"hpo_history_{tag}")
    for p in ("C",):
        if p in set(pd.Series([s for s in folds_full.best_params.head(50)]).map(lambda s: p in s)) or True:
            try:
                made += V.plot_hpo_param_distribution(folds_full.merge(
                    folds[["config", "model", "strategy", "repeat", "fold"]],
                    on=["config", "model", "strategy", "repeat", "fold"]), pm, ps, p, fig_dir / f"hpo_{p}_distribution_{tag}")
            except Exception as e:
                print("skip hpo param plot:", e)

    # per-model comparison on the largest configuration, all models/strategies
    all_cfg = list(dict.fromkeys(summ.config))[-1]
    for m in models:
        for s in strategies:
            d = V.select(preds, m, s)
            if d:
                made += V.plot_roc(d, f"ROC (out-of-fold): {m} / {s}", fig_dir / "per_model" / f"roc_{m}_{s}")

    # per-configuration PNG sets for the primary model (kept from the notebook's per-model export)
    for c, d in prim.items():
        base = fig_dir / "per_config" / V._slug(c)
        one = {c: d}
        made += V.plot_roc(one, f"ROC: {c} ({pm}/{ps})", base / "roc")
        made += V.plot_pr(one, f"PR: {c} ({pm}/{ps})", base / "pr")
        made += V.plot_confusion_grid(one, f"Confusion matrix: {c}", base / "confusion", ncols=1)
        made += V.plot_calibration(one, f"Calibration: {c}", base / "calibration", ncols=1)
        made += V.plot_probability_hist(d, f"Predicted probabilities: {c}", base / "prob_hist")
        made += V.plot_metric_bars_per_fold(folds, pm, ps, c, base / "per_fold")

    # statistical comparison between configurations (primary model/strategy only)
    pw = pairwise_comparison(prim, B=2000)
    pw.to_csv(tab_dir / "pairwise_config_comparison_primary.csv", index=False)
    ref = "DVH"
    vr = vs_reference(prim, ref, B=2000)
    vr.to_csv(tab_dir / "vs_reference_DVH_primary.csv", index=False)
    made += V.plot_pairwise_delta(pw, list(prim), fig_dir / "pairwise_delta_auc_primary")
    made += V.plot_delta_vs_reference(vr, ref, fig_dir / "delta_auc_vs_DVH_primary")

    # label-free descriptive figures (full data; outcome never used as input)
    ds = load_dataset(cfg)
    made += V.plot_correlation_heatmaps(ds, fig_dir / "correlation")
    if not a.skip_tsne:
        made += V.plot_tsne(ds, fig_dir / "tsne_all_features")
    print(f"{len(made)} files written under {fig_dir}")


if __name__ == "__main__":
    main()

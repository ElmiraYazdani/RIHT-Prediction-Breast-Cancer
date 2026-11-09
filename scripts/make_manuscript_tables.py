"""Tables used in the manuscript (descriptive; all derived from the data files and outputs/ of the finished runs).

Writes outputs/manuscript/: table1_characteristics.csv, table2_primary_performance.csv, table3_dosiomics_vs_dvh.csv,
table_s1_univariate_auc_apparent.csv, table_s2_all_models_auc.csv, table_s3_selected_features.csv, key_numbers.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import load_dataset


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    cfg = load_config()
    ds = load_dataset(cfg)
    out = resolve(cfg, "output_dir") / "manuscript"
    out.mkdir(parents=True, exist_ok=True)
    main_dir = resolve(cfg, "output_dir") / "main"
    X, y = ds.X, ds.y
    key = {}

    # ---- Table 1 (descriptive; unadjusted Mann-Whitney / Fisher, no multiplicity claims) ----
    rows = []
    c0, c1 = f"No hypothyroidism (n={(y == 0).sum()})", f"Hypothyroidism (n={y.sum()})"
    cont = [("CB_AGE", "Age, years"), ("CB_VOLCTCM3", "Thyroid volume, cm3"), ("DVH_Dmean", "Thyroid Dmean, Gy"),
            ("DVH_Dmax", "Thyroid Dmax, Gy"), ("DVH_V20", "V20, %"), ("DVH_V30", "V30, %"), ("DVH_V40", "V40, %")]
    q = lambda v: f"{v.median():.1f} ({v.quantile(.25):.1f}-{v.quantile(.75):.1f})"
    for c, lab in cont:
        a_, b_ = X.loc[y == 0, c], X.loc[y == 1, c]
        rows.append({"Variable": lab, c0: q(a_), c1: q(b_), "p (unadjusted)": fmt_p(stats.mannwhitneyu(a_, b_).pvalue),
                     "Statistic": "median (IQR); Mann-Whitney"})
    for c, lab in [("CB_DM", "Diabetes mellitus"), ("CB_HTN", "Hypertension"), ("CB_SMOKE", "Smoking")]:
        a_, b_ = X.loc[y == 0, c], X.loc[y == 1, c]
        tab = [[int(a_.sum()), int(len(a_) - a_.sum())], [int(b_.sum()), int(len(b_) - b_.sum())]]
        rows.append({"Variable": lab, c0: f"{int(a_.sum())} ({100 * a_.mean():.1f}%)",
                     c1: f"{int(b_.sum())} ({100 * b_.mean():.1f}%)",
                     "p (unadjusted)": fmt_p(stats.fisher_exact(tab)[1]), "Statistic": "n (%); Fisher exact"})
    t1 = pd.DataFrame(rows)
    t1.to_csv(out / "table1_characteristics.csv", index=False)

    # ---- univariate (apparent, full-data, NOT cross-validated) AUC of single features ----
    uni = []
    for c in X.columns:
        a = roc_auc_score(y, X[c])
        uni.append((ds.feature_to_group[c], c, max(a, 1 - a)))
    uni = pd.DataFrame(uni, columns=["group", "feature", "apparent_univariate_auc"]).sort_values(
        "apparent_univariate_auc", ascending=False)
    top = uni.groupby("group").head(3).sort_values(["group", "apparent_univariate_auc"], ascending=[True, False])
    top.to_csv(out / "table_s1_univariate_auc_apparent.csv", index=False)
    key["univariate_max"] = uni.groupby("group").apparent_univariate_auc.max().round(3).to_dict()
    key["univariate_median"] = uni.groupby("group").apparent_univariate_auc.median().round(3).to_dict()

    # ---- redundancy between dosiomics and DVH ----
    dfn = ds.feature_groups["DF"]
    cr = X[ds.feature_groups["DVH"] + dfn].corr(method="spearman").loc[ds.feature_groups["DVH"], dfn]
    key["spearman_DFmean_vs_Dmean"] = round(float(X["DF_firstorder Mean"].corr(X["DVH_Dmean"], method="spearman")), 3)
    key["spearman_DF_ClusterShade_vs_V30"] = round(float(X["DF_glcm ClusterShade"].corr(X["DVH_V30"], method="spearman")), 3)
    key["spearman_DF_Skewness_vs_V30"] = round(float(X["DF_firstorder Skewness"].corr(X["DVH_V30"], method="spearman")), 3)
    key["n_DF_features_abs_rho_ge_0.8_with_any_DVH"] = int((cr.abs().max(axis=0) >= 0.8).sum())
    key["n_DF_features"] = len(dfn)
    key["spearman_RF_volume_max_abs"] = round(float(max(abs(X[c].corr(pd.Series(ds.volume, index=X.index), method="spearman"))
                                                       for c in ds.feature_groups["RF"])), 3)

    # ---- primary performance table ----
    s = pd.read_csv(main_dir / "tables" / "metrics_summary.csv")
    p = s[(s.model == cfg["models"]["primary_model"]) & (s.strategy == cfg["models"]["primary_strategy"])].copy()
    f = lambda m: f"{p[m]:.3f}"
    t2 = pd.DataFrame({
        "Feature configuration": p.config,
        "ROC-AUC (95% CI)": p.apply(lambda r: f"{r.roc_auc:.3f} ({r.roc_auc_ci_low:.3f}-{r.roc_auc_ci_high:.3f})", axis=1),
        "PR-AUC": p.pr_auc.map("{:.3f}".format), "Sensitivity": p.sensitivity.map("{:.2f}".format),
        "Specificity": p.specificity.map("{:.2f}".format), "Precision": p.precision.map("{:.2f}".format),
        "F1": p.f1.map("{:.2f}".format), "Balanced accuracy": p.balanced_accuracy.map("{:.2f}".format),
        "Brier": p.brier.map("{:.3f}".format), "Calibration slope": p.calibration_slope.map("{:.2f}".format),
        "SD of AUC across repeats": p.roc_auc_sd_repeats.map("{:.3f}".format)})
    t2.to_csv(out / "table2_primary_performance.csv", index=False)

    # ---- dosiomics vs DVH across every model/strategy ----
    ext = pd.read_csv(resolve(cfg, "output_dir") / "extended" / "tables" / "metrics_summary.csv")
    allm = pd.concat([s, ext], ignore_index=True)
    rows = []
    for (m, st), g in allm.groupby(["model", "strategy"]):
        g = g.set_index("config")
        rows.append({"model": m, "strategy": st, "AUC_DVH": g.loc["DVH", "roc_auc"], "AUC_DFs": g.loc["DFs", "roc_auc"],
                     "AUC_DVH+DFs": g.loc["DVH+DFs", "roc_auc"], "AUC_CBs": g.loc["CBs", "roc_auc"],
                     "AUC_CT RFs": g.loc["CT RFs", "roc_auc"],
                     "dAUC_DFs_minus_DVH": g.loc["DFs", "roc_auc"] - g.loc["DVH", "roc_auc"],
                     "dAUC_DVH+DFs_minus_DVH": g.loc["DVH+DFs", "roc_auc"] - g.loc["DVH", "roc_auc"]})
    t3 = pd.DataFrame(rows)
    t3.to_csv(out / "table3_dosiomics_vs_dvh.csv", index=False)
    key["n_model_strategy_combinations"] = len(t3)
    key["DFs_beats_DVH_n"] = int((t3.dAUC_DFs_minus_DVH > 0).sum())
    key["DVH+DFs_beats_DVH_n"] = int((t3["dAUC_DVH+DFs_minus_DVH"] > 0).sum())
    key["dAUC_DFs_minus_DVH_range"] = [round(t3.dAUC_DFs_minus_DVH.min(), 3), round(t3.dAUC_DFs_minus_DVH.max(), 3)]
    key["dAUC_DVH+DFs_minus_DVH_range"] = [round(t3["dAUC_DVH+DFs_minus_DVH"].min(), 3),
                                           round(t3["dAUC_DVH+DFs_minus_DVH"].max(), 3)]
    key["median_dAUC_DVH+DFs_minus_DVH"] = round(float(t3["dAUC_DVH+DFs_minus_DVH"].median()), 3)
    key["median_dAUC_DFs_minus_DVH"] = round(float(t3.dAUC_DFs_minus_DVH.median()), 3)
    key["max_AUC_CBs_any_model"] = round(float(allm[allm.config == "CBs"].roc_auc.max()), 3)
    key["max_AUC_CTRFs_any_model"] = round(float(allm[allm.config == "CT RFs"].roc_auc.max()), 3)
    allm.pivot_table(index="config", columns=["model", "strategy"], values="roc_auc").round(3).to_csv(
        out / "table_s2_all_models_auc.csv")

    # ---- selected features (primary) ----
    st = pd.read_csv(main_dir / "tables" / "feature_selection_stability.csv")
    sel = st[st.config.isin(["DVH", "DFs", "DVH+DFs", "CBs", "CT RFs", "CBs+DVH+CT RFs+DFs"])]
    sel = sel[sel.selection_proportion > 0].sort_values(["config", "selection_proportion"], ascending=[True, False])
    sel[["config", "feature", "times_selected", "n_outer_fits", "selection_proportion"]].to_csv(
        out / "table_s3_selected_features.csv", index=False)

    # ---- misc facts ----
    folds = pd.read_csv(main_dir / "tables" / "per_fold_metrics.csv")
    key["events_per_outer_test_fold"] = [int(folds.n_pos_test.min()), int(folds.n_pos_test.max())]
    key["events_per_outer_train_fold"] = [int(folds.n_pos_train.min()), int(folds.n_pos_train.max())]
    key["mean_fold_test_auc_DVH_primary"] = round(float(folds[(folds.config == "DVH") & (folds.model == "logreg_l2") &
                                                               (folds.strategy == "class_weight")].test_auc.mean()), 3)
    stg = pd.read_csv(main_dir / "tables" / "feature_selection_stage_counts.csv")
    key["stability_threshold_median_final"] = round(float(stg.final_threshold.median()), 3)
    key["stability_threshold_max_final"] = round(float(stg.final_threshold.max()), 3)
    key["stability_fits"] = int(len(stg))
    hp = pd.read_csv(main_dir / "tables" / "optimized_hyperparameters.csv")
    h = hp[(hp.model == "logreg_l2") & (hp.strategy == "class_weight") & hp.config.isin(["DVH", "DFs"])]
    h = h.assign(C=h.best_params.map(lambda s: json.loads(s)["C"]))
    key["C_median_IQR"] = h.groupby("config").C.apply(lambda s: [round(s.median(), 3), round(s.quantile(.25), 3),
                                                                 round(s.quantile(.75), 3)]).to_dict()
    perm = pd.read_csv(main_dir / "tables" / "permutation_test_summary.csv")
    key["permutation"] = perm.round(3).to_dict(orient="records")
    with open(out / "key_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2, default=str)
    print(t1.to_string(index=False))
    print(json.dumps(key, indent=1, default=str))


if __name__ == "__main__":
    main()

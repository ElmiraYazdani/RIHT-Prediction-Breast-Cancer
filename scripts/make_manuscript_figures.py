"""Manuscript figures 2-5 (figure 1 = make_workflow_figure.py; figure 6 = outputs/main/figures stability heat-map).

Writes outputs/manuscript/figure{2,3,4,5}_*.png|pdf and dca_numbers.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
from thyroid_rihp import visualization as V
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.evaluation import calibration_bins, decision_curve
from thyroid_rihp.stats import mean_prob_by_patient

COL = {"CBs": "#6c757d", "CT RFs": "#e76f51", "DVH": "#2a9d8f", "DFs": "#b5179e", "DVH+DFs": "#1f4e79"}
LAB = {"CBs": "CBs", "CT RFs": "CT RFs", "DVH": "DVH", "DFs": "DFs", "DVH+DFs": "DVH + DFs"}
PANEL = dict(fontsize=13, fontweight="bold", va="top")


def main():
    cfg = load_config()
    root = resolve(cfg, "output_dir")
    main_dir, out = root / "main", root / "manuscript"
    out.mkdir(parents=True, exist_ok=True)
    pm, ps = cfg["models"]["primary_model"], cfg["models"]["primary_strategy"]
    preds = pd.read_csv(main_dir / "predictions" / "oof_predictions.csv")
    summ = pd.read_csv(main_dir / "tables" / "metrics_summary.csv")
    prim = V.select(preds, pm, ps)
    mp = {k: mean_prob_by_patient(prim[k]) for k in COL}

    # ---------------- Figure 2: ROC and PR ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for k, d in mp.items():
        fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
        r = summ[(summ.config == k) & (summ.model == pm) & (summ.strategy == ps)].iloc[0]
        axes[0].plot(fpr, tpr, color=COL[k], lw=2.2, label=f"{LAB[k]}  AUC {r.roc_auc:.3f}")
        pr, rc, _ = precision_recall_curve(d.y_true, d.y_prob)
        axes[1].plot(rc, pr, color=COL[k], lw=2.2, label=f"{LAB[k]}  AP {r.pr_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[1].axhline(mp["DVH"].y_true.mean(), ls="--", color="grey", lw=1)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    for ax, t in zip(axes, "ab"):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8.5)
        ax.text(-0.13, 1.06, t, transform=ax.transAxes, **PANEL)
    fig.tight_layout()
    V.save_fig(fig, out / "figure2_roc_pr")

    # ---------------- Figure 3: forest of AUC + delta vs DVH ----------------
    p = summ[(summ.model == pm) & (summ.strategy == ps)].sort_values("roc_auc")
    vr = pd.read_csv(main_dir / "tables" / "vs_reference_DVH_primary.csv").set_index("config")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(p))
    axes[0].errorbar(p.roc_auc, y, xerr=[p.roc_auc - p.roc_auc_ci_low, p.roc_auc_ci_high - p.roc_auc], fmt="o",
                     color="#1f4e79", ecolor="#7f9ec2", capsize=2.5)
    axes[0].set_yticks(y, p.config)
    axes[0].axvline(0.5, color="grey", ls="--", lw=1)
    axes[0].set(xlim=(0.2, 1.0), xlabel="Out-of-fold ROC-AUC (95% bootstrap CI)")
    d = vr.loc[[c for c in p.config if c in vr.index]]
    y2 = np.arange(len(d))
    axes[1].errorbar(d.delta_auc_vs_ref, y2, xerr=[d.delta_auc_vs_ref - d.lo, d.hi - d.delta_auc_vs_ref], fmt="o",
                     color="#b5179e", ecolor="#d9a3c9", capsize=2.5)
    axes[1].set_yticks(y2, d.index)
    axes[1].axvline(0, color="grey", ls="--", lw=1)
    axes[1].set(xlabel="AUC difference vs. DVH alone (paired bootstrap 95% CI)", xlim=(-0.85, 0.08))
    for ax, t in zip(axes, "ab"):
        ax.text(-0.02 if t == "a" else -0.02, 1.07, t, transform=ax.transAxes, **PANEL)
    fig.tight_layout()
    V.save_fig(fig, out / "figure3_configs_forest")

    # ---------------- Figure 4: dosiomics vs DVH across all model/strategy combinations ----------------
    t3 = pd.read_csv(out / "table3_dosiomics_vs_dvh.csv").sort_values("AUC_DVH")
    t3["label"] = t3.model.str.replace("_", " ") + " / " + t3.strategy.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    y = np.arange(len(t3))
    for i, (_, r) in enumerate(t3.iterrows()):
        ax.plot([r["AUC_DFs"], r["AUC_DVH"]], [i, i], color="#cccccc", lw=1.4, zorder=1)
    ax.scatter(t3.AUC_DVH, y, color=COL["DVH"], s=44, label="DVH", zorder=3)
    ax.scatter(t3.AUC_DFs, y, color=COL["DFs"], s=44, label="DFs", zorder=3)
    ax.scatter(t3["AUC_DVH+DFs"], y, color=COL["DVH+DFs"], marker="D", s=34, label="DVH + DFs", zorder=3)
    ax.set_yticks(y, t3.label, fontsize=8)
    ax.set(xlabel="Out-of-fold ROC-AUC", xlim=(0.78, 0.98))
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    V.save_fig(fig, out / "figure4_models_dvh_vs_df")

    # ---------------- Figure 5: calibration + decision curves ----------------
    sel = ["DVH", "DFs", "DVH+DFs"]
    th = np.linspace(0.02, 0.8, 79)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    rows = []
    for k in sel:
        d_ = mp[k]
        t = calibration_bins(d_.y_true, d_.y_prob, 5)
        axes[0].plot(t.mean_pred, t.obs_rate, "o-", color=COL[k], lw=1.8, label=LAB[k])
        dc = decision_curve(d_.y_true, d_.y_prob, th)
        axes[1].plot(dc.threshold, dc.net_benefit_model, color=COL[k], lw=2, label=LAB[k])
        for q in (0.1, 0.2, 0.3, 0.5):
            r = dc.iloc[(dc.threshold - q).abs().argmin()]
            rows.append({"config": k, "threshold": q, "net_benefit_model": r.net_benefit_model,
                         "net_benefit_treat_all": r.net_benefit_all})
    dc = decision_curve(mp["DVH"].y_true, mp["DVH"].y_prob, th)
    axes[1].plot(dc.threshold, dc.net_benefit_all, "--", color="black", lw=1.4, label="Treat all")
    axes[1].axhline(0, color="grey", lw=1.2, label="Treat none")
    axes[0].set(xlabel="Mean predicted probability (quintiles)", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0, 0.8), ylim=(-0.06, 0.26))
    for ax, t in zip(axes, "ab"):
        ax.legend(frameon=False, fontsize=8.5, loc="upper left" if t == "a" else "upper right")
        ax.text(-0.13, 1.06, t, transform=ax.transAxes, **PANEL)
    fig.tight_layout()
    V.save_fig(fig, out / "figure5_calibration_dca")
    pd.DataFrame(rows).round(3).to_csv(out / "dca_numbers.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

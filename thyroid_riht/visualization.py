"""Publication-quality figures (every figure is written as PNG @ 300 dpi and as vector PDF).

All curves use out-of-fold predictions only. Where several outer repeats exist, curves are drawn from the
patient-wise MEAN out-of-fold probability across repeats (each individual prediction is still out-of-fold);
tables report the mean of per-repeat metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .evaluation import calibration_bins, decision_curve
from .stats import mean_prob_by_patient

BEST_COLOR = "#e0218a"   # kept from the original notebook (best model highlighted in pink)
PALETTE = list(plt.get_cmap("tab20").colors[:15])


def set_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


set_style()


def save_fig(fig, path_base) -> list:
    """Save as <base>.png (300 dpi) and <base>.pdf; returns the two paths."""
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = Path(f"{path_base}.{ext}")      # not with_suffix(): names such as "..._thr0.5" would be truncated
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def select(preds: pd.DataFrame, model: str, strategy: str) -> dict:
    d = preds[(preds.model == model) & (preds.strategy == strategy)]
    return {c: g for c, g in d.groupby("config", sort=False)}


def _color_map(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _best(names, mp):
    aucs = {n: roc_auc_score(mp[n]["y_true"], mp[n]["y_prob"]) for n in names}
    return max(aucs, key=aucs.get), aucs


# --------------------------------------------------------------------------- curves
def plot_roc(oof: dict, title: str, path_base, summary: pd.DataFrame | None = None):
    mp = {n: mean_prob_by_patient(d) for n, d in oof.items()}
    cm = _color_map(list(mp))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for n, d in mp.items():
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        a = roc_auc_score(d["y_true"], d["y_prob"])
        ax.plot(fpr, tpr, color=cm[n], lw=1.8, label=f"{n} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="1 - Specificity", ylabel="Sensitivity", title=title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    ax.text(0, -0.13, "Curves: patient-wise mean out-of-fold probability over outer repeats (internal CV).",
            transform=ax.transAxes, fontsize=7, color="dimgrey")
    return save_fig(fig, path_base)


def plot_pr(oof: dict, title: str, path_base):
    mp = {n: mean_prob_by_patient(d) for n, d in oof.items()}
    cm = _color_map(list(mp))
    prev = next(iter(mp.values()))["y_true"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for n, d in mp.items():
        pr, rc, _ = precision_recall_curve(d["y_true"], d["y_prob"])
        ax.plot(rc, pr, color=cm[n], lw=1.8, label=f"{n} (AP {average_precision_score(d['y_true'], d['y_prob']):.3f})")
    ax.axhline(prev, ls="--", color="grey", lw=1, label=f"Prevalence ({prev:.2f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (sensitivity)", ylabel="Precision", title=title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return save_fig(fig, path_base)


def _confusion_mean(d: pd.DataFrame, col: str) -> np.ndarray:
    cms = []
    for _, g in d.groupby("repeat"):
        tn = ((g.y_true == 0) & (g[col] == 0)).sum()
        fp = ((g.y_true == 0) & (g[col] == 1)).sum()
        fn = ((g.y_true == 1) & (g[col] == 0)).sum()
        tp = ((g.y_true == 1) & (g[col] == 1)).sum()
        cms.append([[tn, fp], [fn, tp]])
    return np.mean(cms, axis=0)


def plot_confusion_grid(oof: dict, title: str, path_base, col: str = "y_pred", ncols: int = 5):
    names = list(oof)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, n in zip(axes, names):
        cm = _confusion_mean(oof[n], col)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() * 0.8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        sens = cm[1, 1] / max(cm[1].sum(), 1e-9)
        spec = cm[0, 0] / max(cm[0].sum(), 1e-9)
        ax.set_title(f"{n}\nSe {sens:.2f} | Sp {spec:.2f}", fontsize=8.5)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["True 0", "True 1"], fontsize=7)
        ax.grid(False)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, path_base)


def plot_calibration(oof: dict, title: str, path_base, n_bins: int = 5, ncols: int = 5):
    names = list(oof)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, n in zip(axes, names):
        d = mean_prob_by_patient(oof[n])
        t = calibration_bins(d["y_true"], d["y_prob"], n_bins)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(t.mean_pred, t.obs_rate, "o-", color="#1f4e79", lw=1.6)
        for _, r in t.iterrows():
            ax.annotate(f"{int(r.events)}/{int(r.n)}", (r.mean_pred, r.obs_rate), textcoords="offset points",
                        xytext=(3, -9), fontsize=6)
        brier = float(np.mean((d["y_prob"] - d["y_true"]) ** 2))
        ax.set_title(f"{n}\nBrier {brier:.3f}", fontsize=8.5)
        ax.set(xlim=(0, 1), ylim=(0, 1))
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.supxlabel("Mean predicted probability (quantile bins; labels = events/n)")
    fig.supylabel("Observed event rate")
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, path_base)


def plot_dca(oof: dict, title: str, path_base, thresholds):
    mp = {n: mean_prob_by_patient(d) for n, d in oof.items()}
    best, _ = _best(list(mp), mp)
    cm = _color_map(list(mp))
    fig, ax = plt.subplots(figsize=(8, 6.2))
    first = next(iter(mp.values()))
    for n, d in mp.items():
        dc = decision_curve(d["y_true"], d["y_prob"], thresholds)
        ax.plot(dc.threshold, dc.net_benefit_model, color=BEST_COLOR if n == best else cm[n],
                lw=3.6 if n == best else 1.3, alpha=1 if n == best else 0.75, zorder=10 if n == best else 3,
                label=f"{n}" + (" (highest AUC)" if n == best else ""))
    dc = decision_curve(first["y_true"], first["y_prob"], thresholds)
    ax.plot(dc.threshold, dc.net_benefit_all, "--", color="black", lw=1.6, label="Treat all")
    ax.axhline(0, color="grey", lw=1.4, label="Treat none")
    ax.set(xlim=(0, thresholds[-1]), ylim=(-0.08, first["y_true"].mean() + 0.05),
           xlabel="Threshold probability", ylabel="Net benefit", title=title)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(0.01, 0.01, "Out-of-fold predictions; probabilities are not recalibrated (class weighting/SMOTE shift them).",
            transform=ax.transAxes, fontsize=6.5, color="dimgrey")
    return save_fig(fig, path_base)


# --------------------------------------------------------------------------- per-fold / overfitting
def plot_fold_auc(folds: pd.DataFrame, model: str, strategy: str, path_base):
    """Test AUC per outer fold and train-vs-test AUC (kept from the original notebook; values now valid)."""
    d = folds[(folds.model == model) & (folds.strategy == strategy)].copy()
    d["fold_id"] = d["repeat"] * d["fold"].max() + d["fold"] + d["repeat"] + 1
    cfgs = list(dict.fromkeys(d.config))
    cm = _color_map(cfgs)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for c in cfgs:
        g = d[d.config == c].sort_values("fold_id")
        axes[0].plot(g.fold_id, g.test_auc, "o-", color=cm[c], lw=1.2, ms=3.5, label=c)
        axes[1].plot(g.fold_id, g.train_auc, "o-", color=cm[c], lw=1.2, ms=3.5)
    axes[0].set(title="Test AUC per outer fold", xlabel="Outer fold (repeats concatenated)", ylabel="AUC", ylim=(0.3, 1.02))
    axes[1].set(title="Train (apparent) AUC per outer fold", xlabel="Outer fold (repeats concatenated)", ylim=(0.3, 1.02))
    gap = d.groupby("config").apply(lambda g: (g.train_auc - g.test_auc).mean(), include_groups=False).reindex(cfgs)
    axes[2].barh(cfgs, gap.values, color=[cm[c] for c in cfgs])
    axes[2].set(title="Optimism gap: mean(train AUC - test AUC)", xlabel="AUC difference")
    axes[2].invert_yaxis()
    axes[0].legend(loc="center left", bbox_to_anchor=(0, -0.28), ncol=5, frameon=False, fontsize=7)
    fig.suptitle(f"{model} / {strategy}", y=1.02)
    fig.tight_layout()
    return save_fig(fig, path_base)


def plot_metric_bars_per_fold(folds: pd.DataFrame, model: str, strategy: str, config: str, path_base):
    d = folds[(folds.model == model) & (folds.strategy == strategy) & (folds.config == config)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, ttl) in zip(axes, [("test_auc", "AUC"), ("balanced_accuracy", "Balanced accuracy"),
                                     ("n_features", "Features selected")]):
        lab = [f"r{r + 1}f{f + 1}" for r, f in zip(d.repeat, d.fold)]
        ax.bar(lab, d[col], color="#1f4e79", edgecolor="black", alpha=0.85)
        ax.axhline(d[col].mean(), color="green", ls="--", label=f"mean {d[col].mean():.3f}")
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.legend(frameon=False)
    fig.suptitle(f"{config} | {model}/{strategy}: per outer fold", y=1.03)
    fig.tight_layout()
    return save_fig(fig, path_base)


# --------------------------------------------------------------------------- feature plots
def plot_correlation_heatmaps(ds, out_dir):
    """Label-free descriptive Spearman correlation maps of each feature group (full data, no outcome used)."""
    out = []
    X = ds.X
    small = ds.feature_groups["CB"] + ds.feature_groups["DVH"]
    c = X[small].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(c, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.3, annot=True, fmt=".2f",
                annot_kws={"size": 5.5}, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Clinical (CB_) and DVH (DVH_) features")
    ax.grid(False)
    out += save_fig(fig, Path(out_dir) / "correlation_CB_DVH")
    for g, ttl in (("RF", "CT radiomic features"), ("DF", "Dosiomic features")):
        c = X[ds.feature_groups[g]].corr(method="spearman").fillna(0)
        dist = np.clip(1 - np.abs(c.values), 0, None)
        np.fill_diagonal(dist, 0)
        link = hierarchy.linkage(hierarchy.distance.squareform(dist, checks=False), method="average")
        order = hierarchy.leaves_list(link)
        cc = c.iloc[order, order]
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cc, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, xticklabels=False, yticklabels=False,
                    cbar_kws={"label": "Spearman rho"})
        ax.set_title(f"{ttl} (n={c.shape[0]}, hierarchically ordered)")
        ax.grid(False)
        out += save_fig(fig, Path(out_dir) / f"correlation_{g}")
    # cross-group: DVH vs most-correlated dosiomic features
    cross = X[ds.feature_groups["DVH"] + ds.feature_groups["DF"]].corr(method="spearman").loc[
        ds.feature_groups["DVH"], ds.feature_groups["DF"]]
    top = cross.abs().max(axis=0).sort_values(ascending=False).head(25).index
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(cross[top], cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman rho"},
                xticklabels=[t.replace("DF_", "") for t in top])
    ax.set_title("DVH vs. dosiomic features (25 dosiomic features most correlated with any DVH parameter)")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.grid(False)
    out += save_fig(fig, Path(out_dir) / "correlation_DVH_vs_DF")
    return out


def plot_feature_stability(stab: pd.DataFrame, path_base, top_per_config: int = 3):
    """Heatmap: selection proportion across outer fits for the most frequently selected features."""
    keep = stab.groupby("config").head(top_per_config).feature.unique()
    pv = stab[stab.feature.isin(keep)].pivot(index="feature", columns="config", values="selection_proportion").fillna(0)
    pv = pv.loc[pv.max(axis=1).sort_values(ascending=False).index]
    pv.index = [i.replace("firstorder ", "fo ") for i in pv.index]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * pv.shape[1] + 3), max(4, 0.28 * pv.shape[0] + 2)))
    sns.heatmap(pv, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "Selection proportion (outer folds x repeats)"}, ax=ax)
    ax.set_title("Feature-selection stability across outer training folds")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    return save_fig(fig, path_base)


def plot_feature_stability_top(stab: pd.DataFrame, path_base, top_k: int = 20):
    """Original notebook figure: top-k most frequently selected features (pooled over configurations)."""
    t = stab.groupby("feature").agg(times=("times_selected", "sum"), fits=("n_outer_fits", "sum"))
    t["prop"] = t.times / t.fits
    t = t.sort_values("times", ascending=False).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(t.index, t.times, color="#1f4e79")
    ax.set_xlabel("Times selected (summed over configurations, outer folds and repeats)")
    ax.set_title(f"Top {top_k} most frequently selected features")
    return save_fig(fig, path_base)


# --------------------------------------------------------------------------- model comparison
def plot_config_auc(summary: pd.DataFrame, model: str, strategy: str, path_base):
    d = summary[(summary.model == model) & (summary.strategy == strategy)].sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(d))
    ax.errorbar(d.roc_auc, y, xerr=[d.roc_auc - d.roc_auc_ci_low, d.roc_auc_ci_high - d.roc_auc], fmt="o",
                color="#1f4e79", ecolor="#7f9ec2", capsize=2.5)
    ax.set_yticks(y, d.config)
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set(xlabel="Out-of-fold ROC-AUC (95% patient-bootstrap CI)", xlim=(0.4, 1.0),
           title=f"Feature-group configurations: {model} / {strategy}")
    ax.text(0.01, -0.12, "CI = test-sample uncertainty only (training variability not captured).",
            transform=ax.transAxes, fontsize=7, color="dimgrey")
    return save_fig(fig, path_base)


def plot_model_heatmap(summary: pd.DataFrame, metric: str, path_base, strategies=None):
    d = summary.copy()
    d["col"] = d.model + "\n" + d.strategy
    if strategies:
        d = d[d.strategy.isin(strategies)]
    pv = d.pivot(index="config", columns="col", values=metric)
    order = [c for c in dict.fromkeys(summary.config) if c in pv.index]
    pv = pv.loc[order]
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * pv.shape[1] + 2), 6.5))
    sns.heatmap(pv, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5 if "auc" in metric else None, ax=ax,
                annot_kws={"size": 6.5}, cbar_kws={"label": metric})
    ax.set_title(f"{metric} by feature configuration, model and imbalance strategy")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(False)
    return save_fig(fig, path_base)


def plot_strategy_comparison(summary: pd.DataFrame, path_base, metric: str = "roc_auc"):
    d = summary.copy()
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        g = d[d.model == m]
        sns.stripplot(data=g, x="strategy", y=metric, order=["none", "class_weight", "smote"], ax=ax,
                      color="#1f4e79", alpha=0.7, size=5)
        sns.pointplot(data=g, x="strategy", y=metric, order=["none", "class_weight", "smote"], ax=ax, color="black",
                      errorbar=None, markers="_", linestyles="", markersize=14)
        ax.set_title(m)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel(f"{metric} (one dot per feature configuration; bar = mean)")
    fig.suptitle("Imbalance-handling strategies", y=1.02)
    fig.tight_layout()
    return save_fig(fig, path_base)


def plot_pairwise_delta(pw: pd.DataFrame, names: list, path_base):
    n = len(names)
    M = np.full((n, n), np.nan)
    S = np.full((n, n), "", dtype=object)
    ix = {k: i for i, k in enumerate(names)}
    for _, r in pw.iterrows():
        i, j = ix[r.config_A], ix[r.config_B]
        M[i, j], M[j, i] = r.delta_auc, -r.delta_auc
        S[i, j] = S[j, i] = "*" if r.ci_excludes_0 else ""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.2f}{S[i, j]}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(n), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75, label="delta AUC (row - column)")
    ax.set_title("Paired patient-bootstrap delta-AUC between configurations\n"
                 "* = 95% bootstrap CI excludes 0 (descriptive; unadjusted, no formal significance claim)")
    ax.grid(False)
    return save_fig(fig, path_base)


def plot_delta_vs_reference(df: pd.DataFrame, reference: str, path_base):
    d = df.sort_values("delta_auc_vs_ref")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(d))
    ax.errorbar(d.delta_auc_vs_ref, y, xerr=[d.delta_auc_vs_ref - d.lo, d.hi - d.delta_auc_vs_ref], fmt="o",
                color="#1f4e79", ecolor="#7f9ec2", capsize=2.5)
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks(y, d.config)
    ax.set(xlabel=f"AUC difference vs. {reference} (paired bootstrap 95% CI)",
           title=f"Added value over the pre-specified reference: {reference}")
    return save_fig(fig, path_base)


# --------------------------------------------------------------------------- HPO
def plot_hpo_history(trials: pd.DataFrame, model: str, strategy: str, path_base, configs=None):
    d = trials[(trials.model == model) & (trials.strategy == strategy)]
    cfgs = configs or list(dict.fromkeys(d.config))
    cm = _color_map(cfgs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in cfgs:
        g = d[d.config == c]
        curves = []
        for _, h in g.groupby(["repeat", "fold"]):
            v = h.sort_values("trial").inner_score.cummax().to_numpy()
            curves.append(v)
        L = min(map(len, curves))
        A = np.array([v[:L] for v in curves])
        axes[0].plot(np.arange(1, L + 1), A.mean(axis=0), color=cm[c], lw=1.4, label=c)
    axes[0].set(xlabel="Optuna trial", ylabel="Best inner-CV AUC so far (mean over outer fits)",
                title=f"Optimisation history: {model} / {strategy}")
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    # parameter vs inner score
    pcols = sorted({k for s in d.params.head(500) for k in json.loads(s)})
    if pcols:
        pname = "C" if "C" in pcols else pcols[0]
        vals = d.params.map(lambda s: json.loads(s).get(pname, np.nan))
        samp = d.assign(p=vals).sample(min(len(d), 6000), random_state=0)
        sc = axes[1].scatter(samp.p, samp.inner_score, c=samp.trial, s=6, cmap="viridis", alpha=0.6)
        if pname == "C":
            axes[1].set_xscale("log")
        axes[1].set(xlabel=pname, ylabel="Inner-CV AUC", title=f"Search landscape ({pname}); colour = trial number")
        fig.colorbar(sc, ax=axes[1], label="trial")
    fig.tight_layout()
    return save_fig(fig, path_base)


def plot_hpo_param_distribution(folds: pd.DataFrame, model: str, strategy: str, param: str, path_base):
    d = folds[(folds.model == model) & (folds.strategy == strategy)].copy()
    d[param] = d.best_params.map(lambda s: json.loads(s).get(param, np.nan))
    fig, ax = plt.subplots(figsize=(8, 6))
    order = list(dict.fromkeys(d.config))
    sns.boxplot(data=d, y="config", x=param, order=order, ax=ax, color="#9ec5e8", fliersize=2)
    sns.stripplot(data=d, y="config", x=param, order=order, ax=ax, color="black", size=2.5, alpha=0.6)
    if d[param].max() / max(d[param].min(), 1e-12) > 50:
        ax.set_xscale("log")
    ax.set(title=f"Selected {param} across outer fits ({model}/{strategy})", xlabel=param, ylabel="")
    return save_fig(fig, path_base)


# --------------------------------------------------------------------------- misc
def plot_tsne(ds, path_base, seed: int = 42):
    """Descriptive t-SNE of all standardised candidate features. The outcome is NOT part of the input
    (the original notebook accidentally included the label column); it is used only for colouring."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(ds.X.to_numpy(dtype=float))
    emb = TSNE(n_components=2, perplexity=20, random_state=seed, init="pca", max_iter=1000).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for v, col, lab in ((0, "#ff7043", "No hypothyroidism (0)"), (1, "#1f4e79", "Hypothyroidism (1)")):
        m = ds.y == v
        ax.scatter(emb[m, 0], emb[m, 1], s=34, alpha=0.75, color=col, label=lab, edgecolor="none")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="t-SNE of all candidate features (label not an input)")
    ax.legend(frameon=False)
    ax.grid(False)
    return save_fig(fig, path_base)


def plot_probability_hist(oof: pd.DataFrame, title: str, path_base):
    d = mean_prob_by_patient(oof)
    fig, ax = plt.subplots(figsize=(6, 4))
    for v, col in ((0, "#ff7043"), (1, "#1f4e79")):
        ax.hist(d.y_prob[d.y_true == v], bins=20, range=(0, 1), alpha=0.65, color=col, label=f"class {v}")
    ax.set(xlabel="Out-of-fold predicted probability", ylabel="Patients", title=title)
    ax.legend(frameon=False)
    return save_fig(fig, path_base)

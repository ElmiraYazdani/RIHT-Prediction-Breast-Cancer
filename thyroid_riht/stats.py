"""Statistical comparison utilities.

IMPORTANT interpretation notes (also written to the reports)
-----------------------------------------------------------
* Out-of-fold predictions of different models come from models trained on overlapping data, and the
  predictions of one patient are not independent of those of others (shared training sets). DeLong's
  test assumes a fixed model evaluated on independent test cases, so p-values obtained from pooled
  out-of-fold predictions are ANTI-CONSERVATIVE and are provided as descriptive screening statistics only.
* A paired patient-level bootstrap of the AUC difference is provided as the preferred descriptive
  summary; it also ignores training-set variability, so intervals are optimistic.
* Selecting the best of many configurations and then testing it against the rest is circular. The
  comparisons here are therefore restricted to a pre-specified primary model/strategy.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .evaluation import auc_matrix, bootstrap_ci, bootstrap_indices


# --------------------------------- DeLong (fast, Sun & Xu 2014) ---------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def _fast_delong(preds_sorted: np.ndarray, m: int):
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx, ty, tz = (np.empty([k, m]), np.empty([k, n]), np.empty([k, m + n]))
    for r in range(k):
        tz[r] = _midrank(preds_sorted[r])
        tx[r] = _midrank(preds_sorted[r, :m])
        ty[r] = _midrank(preds_sorted[r, m:])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, p1, p2) -> dict:
    """Two-sided DeLong test for two correlated AUCs computed on the SAME cases."""
    y = np.asarray(y_true).astype(int)
    order = (-y).argsort(kind="stable")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc1": aucs[0], "auc2": aucs[1], "z": 0.0, "p": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mean_prob_by_patient(oof: pd.DataFrame) -> pd.DataFrame:
    """Average the out-of-fold probability of each patient over outer repeats."""
    g = oof.groupby("patient").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
    return g


def pairwise_comparison(oof_by_config: dict, B: int = 2000, seed: int = 11, level: float = 0.95) -> pd.DataFrame:
    """All pairs of configurations (same model/strategy): paired bootstrap dAUC + descriptive DeLong p (Holm)."""
    names = list(oof_by_config)
    base = {n: mean_prob_by_patient(d) for n, d in oof_by_config.items()}
    ref = base[names[0]]
    pats = ref.index
    y = ref["y_true"].to_numpy().astype(int)
    idx = bootstrap_indices(len(y), B, seed, y)
    probs = {n: base[n].loc[pats, "y_prob"].to_numpy() for n in names}
    aucs_b = {n: auc_matrix(y[idx], probs[n][idx]) for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        d = aucs_b[a] - aucs_b[b]
        lo, hi = bootstrap_ci(d, level)
        dl = delong_test(y, probs[a], probs[b])
        rows.append({"config_A": a, "config_B": b, "auc_A": dl["auc1"], "auc_B": dl["auc2"],
                     "delta_auc": dl["auc1"] - dl["auc2"], "boot_ci_low": lo, "boot_ci_high": hi,
                     "ci_excludes_0": bool(lo > 0 or hi < 0), "delong_z": dl["z"], "delong_p_unadjusted": dl["p"]})
    df = pd.DataFrame(rows)
    df["delong_p_holm"] = holm(df["delong_p_unadjusted"].to_numpy())
    df["note"] = "descriptive only: pooled OOF predictions violate DeLong independence; CI ignores training variability"
    return df


def vs_reference(oof_by_config: dict, reference: str, B: int = 2000, seed: int = 11, level: float = 0.95) -> pd.DataFrame:
    """Each config vs. a pre-specified reference config (e.g. DVH only): dAUC with paired bootstrap CI."""
    pw = pairwise_comparison(oof_by_config, B, seed, level)
    rows = []
    for _, r in pw.iterrows():
        if r.config_A == reference:
            rows.append({**r.to_dict(), "config": r.config_B, "reference": reference,
                         "delta_auc_vs_ref": -r.delta_auc, "lo": -r.boot_ci_high, "hi": -r.boot_ci_low})
        elif r.config_B == reference:
            rows.append({**r.to_dict(), "config": r.config_A, "reference": reference,
                         "delta_auc_vs_ref": r.delta_auc, "lo": r.boot_ci_low, "hi": r.boot_ci_high})
    return pd.DataFrame(rows)[["config", "reference", "delta_auc_vs_ref", "lo", "hi"]]

"""Training-fold-only feature selection.

``select_features`` is a pure function of (X_train, y_train, volume_train, config, seed): it never
sees validation/test rows, so calling it inside every training fold (outer AND inner) is leakage-free.

Stages (manuscript section 2.7):
  0. clean      : inf -> NaN, median imputation, zero-variance removal (train statistics only)
  1. volume     : drop RF/DF features with |Spearman rho| > 0.50 against thyroid volume
  2. redundancy : within each group, greedily keep the feature with the best univariate AUC and drop
                  partners with |Pearson r| > 0.60
  3. mRMR       : per group, greedy relevance (mutual information) minus mean |corr| to selected
  4. stability  : repeated elastic-net logistic regression on stratified subsamples; keep features
                  selected in >= threshold of iterations (adaptive fallback, capped at max_features)

Stages that the original notebook referenced but never defined (significance filter, VIF) are NOT part
of the manuscript and are intentionally omitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class SelectionResult:
    selected: list                                   # final feature names (ordered by stability frequency)
    stage_counts: dict = field(default_factory=dict)
    stability_freq: dict = field(default_factory=dict)
    final_threshold: float | None = None
    dropped_volume: list = field(default_factory=list)


def _clean(X: pd.DataFrame, std_thr: float):
    A = X.to_numpy(dtype=float, copy=True)
    A[~np.isfinite(A)] = np.nan
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    r, c = np.where(np.isnan(A))
    A[r, c] = med[c]
    keep = A.std(axis=0) > std_thr
    return pd.DataFrame(A[:, keep], columns=X.columns[keep], index=X.index)


def _auc_fast(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties handled by mid-ranks)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def univariate_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Direction-free univariate AUC max(AUC, 1-AUC), computed on the supplied (training) rows only."""
    out = {}
    for c in X.columns:
        a = _auc_fast(y, X[c].to_numpy())
        out[c] = max(a, 1 - a)
    return pd.Series(out)


def volume_filter(X: pd.DataFrame, volume: np.ndarray, groups_to_filter, feature_to_group, thr: float):
    drop = []
    for c in X.columns:
        if feature_to_group[c] not in groups_to_filter:
            continue
        rho, _ = spearmanr(X[c].to_numpy(), volume)
        if np.isfinite(rho) and abs(rho) > thr:
            drop.append(c)
    return [c for c in X.columns if c not in drop], drop


def redundancy_filter(X: pd.DataFrame, auc: pd.Series, thr: float) -> list:
    """Greedy: highest-AUC first; remove remaining features correlated above thr with a kept one."""
    cols = list(auc.sort_values(ascending=False, kind="stable").index)
    corr = X[cols].corr(method="pearson").abs().fillna(0.0)
    kept = []
    for c in cols:
        if all(corr.loc[c, k] <= thr for k in kept):
            kept.append(c)
    return kept


def mrmr(X: pd.DataFrame, y: np.ndarray, k: int, n_neighbors: int, seed: int) -> list:
    """Greedy mRMR (difference form): relevance = MI(feature, y); redundancy = mean |Pearson| to selected."""
    cols = list(X.columns)
    if len(cols) <= k:
        return cols
    rel = mutual_info_classif(X.to_numpy(), y, n_neighbors=n_neighbors, random_state=seed)
    corr = X.corr().abs().fillna(0.0).to_numpy()
    selected, remaining = [], list(range(len(cols)))
    while len(selected) < k and remaining:
        best, best_score = None, -np.inf
        for j in remaining:
            red = corr[j, selected].mean() if selected else 0.0
            score = rel[j] - red
            if score > best_score:
                best, best_score = j, score
        selected.append(best)
        remaining.remove(best)
    return [cols[j] for j in selected]


def stability_selection(X: pd.DataFrame, y: np.ndarray, scfg: dict, seed: int):
    """Repeated stratified-subsample elastic-net logistic regression; returns (freq Series, mean|coef| Series)."""
    cols = list(X.columns)
    A = X.to_numpy(dtype=float)
    rng = np.random.RandomState(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n0 = max(int(len(idx0) * scfg["subsample_ratio"]), 2)
    n1 = max(int(len(idx1) * scfg["subsample_ratio"]), 2)
    counts, coefs, ok = np.zeros(len(cols)), np.zeros(len(cols)), 0
    for it in range(int(scfg["n_iterations"])):
        sub = np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)])
        sc = StandardScaler().fit(A[sub])                  # scaler fitted on the subsample only
        Z = sc.transform(A[sub])
        l1 = float(rng.choice(scfg["l1_ratios"]))
        C = float(rng.choice(scfg["Cs"]))
        try:
            m = LogisticRegression(l1_ratio=l1, C=C, solver="saga", max_iter=3000, tol=1e-4,
                                   class_weight="balanced", random_state=seed + it)
            m.fit(Z, y[sub])
        except Exception:                                  # a failed iteration simply does not vote
            continue
        w = np.abs(m.coef_[0])
        counts += (w > scfg["coef_threshold"])
        coefs += w
        ok += 1
    if ok == 0:
        raise RuntimeError("Stability selection: all iterations failed")
    return pd.Series(counts / ok, index=cols), pd.Series(coefs / ok, index=cols)


def select_features(X_train: pd.DataFrame, y_train, volume_train, feature_to_group: dict,
                    cfg: dict, seed: int = 0) -> SelectionResult:
    """Run the full multi-stage selection on TRAINING rows only."""
    fs = cfg["feature_selection"]
    y = np.asarray(y_train).astype(int)
    res = SelectionResult(selected=[])
    X = _clean(X_train, fs["zero_variance_std"])
    res.stage_counts["input"] = int(X_train.shape[1])
    res.stage_counts["after_clean"] = int(X.shape[1])

    if fs["volume_filter"]["enabled"]:
        keep, drop = volume_filter(X, np.asarray(volume_train, dtype=float), fs["volume_filter"]["groups"],
                                   feature_to_group, fs["volume_filter"]["abs_spearman_max"])
        res.dropped_volume = drop
        if keep:
            X = X[keep]
    res.stage_counts["after_volume"] = int(X.shape[1])

    auc = univariate_auc(X, y)
    if fs["redundancy"]["enabled"]:
        thr = fs["redundancy"]["abs_pearson_max"]
        if fs["redundancy"]["scope"] == "global":
            kept = redundancy_filter(X, auc, thr)
        else:
            kept = []
            for g in sorted({feature_to_group[c] for c in X.columns}):
                gc = [c for c in X.columns if feature_to_group[c] == g]
                kept.extend(redundancy_filter(X[gc], auc[gc], thr))
        X = X[[c for c in X.columns if c in set(kept)]]
    res.stage_counts["after_redundancy"] = int(X.shape[1])

    if fs["mrmr"]["enabled"]:
        kept = []
        for g in sorted({feature_to_group[c] for c in X.columns}):
            gc = [c for c in X.columns if feature_to_group[c] == g]
            k = int(fs["mrmr"]["k_per_group"].get(g, 5))
            kept.extend(mrmr(X[gc], y, k, fs["mrmr"]["mi_neighbors"], seed))
        X = X[[c for c in X.columns if c in set(kept)]]
    res.stage_counts["after_mrmr"] = int(X.shape[1])

    if fs["stability"]["enabled"] and X.shape[1] > 1:
        sc = fs["stability"]
        freq, mcoef = stability_selection(X, y, sc, seed)
        thr = sc["threshold"]
        keep = freq[freq >= thr].index.tolist()
        min_f = int(sc["min_features"])
        while len(keep) < min_f and thr > sc["fallback_min_threshold"] + 1e-12:
            thr = round(thr - sc["fallback_step"], 4)
            keep = freq[freq >= thr].index.tolist()
        if len(keep) < min_f:                               # last resort: top-k by frequency
            order = sorted(freq.index, key=lambda c: (-freq[c], -mcoef[c]))
            keep = order[:min(min_f, len(order))]
            thr = float(freq[keep[-1]])
        order = sorted(keep, key=lambda c: (-freq[c], -mcoef[c]))
        keep = order[:int(sc["max_features"])]
        res.selected, res.final_threshold = keep, float(thr)
        res.stability_freq = freq.to_dict()
    else:
        res.selected = list(X.columns)[:int(fs["stability"]["max_features"])]
    res.stage_counts["final"] = len(res.selected)
    if not res.selected:
        raise RuntimeError("Feature selection returned no features")
    return res

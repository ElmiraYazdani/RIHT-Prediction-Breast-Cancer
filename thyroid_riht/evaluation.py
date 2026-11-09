"""Metrics, calibration, bootstrap uncertainty and decision-curve helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, roc_auc_score, roc_curve)

METRIC_COLS = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1",
               "balanced_accuracy", "accuracy", "brier"]


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 on the supplied predictions (training-side only)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k])
    return t if np.isfinite(t) else 0.5


def calibration_slope_intercept(y, p, eps: float = 1e-6):
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Returns (intercept a, slope b)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def compute_metrics(y, p, thr: float = 0.5, with_calibration: bool = False) -> dict:
    """All headline metrics. Threshold metrics use ``p >= thr``; precision/F1 are 0 when no positive is predicted."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    out = {
        "roc_auc": roc_auc_score(y, p) if two else np.nan,
        "pr_auc": average_precision_score(y, p) if two else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred) if two else np.nan,
        "accuracy": float((pred == y).mean()),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if with_calibration:
        a, b = calibration_slope_intercept(y, p)
        out["calibration_intercept"], out["calibration_slope"] = a, b
    return out


# ----------------------------- vectorised bootstrap helpers -----------------------------
def auc_matrix(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Rank-based AUC for every row of P (shape B x N) given a single label vector y or a B x N label matrix."""
    P = np.atleast_2d(P)
    Y = y if np.ndim(y) == 2 else np.broadcast_to(y, P.shape)
    R = rankdata(P, axis=1)
    n1 = Y.sum(axis=1)
    n0 = P.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((R * Y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc


def ap_vector(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def bootstrap_indices(n: int, B: int, seed: int, y: np.ndarray | None = None) -> np.ndarray:
    """Patient-level bootstrap indices, B x n (resampling patients, so repeats of the same patient stay together)."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(B, n))
    if y is not None:  # discard resamples with a single class by redrawing them
        bad = np.where(y[idx].sum(axis=1) % n == 0)[0]
        while len(bad):
            idx[bad] = rng.randint(0, n, size=(len(bad), n))
            bad = bad[y[idx[bad]].sum(axis=1) % n == 0]
    return idx


def bootstrap_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    v = values[np.isfinite(values)]
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def summarize_runs_with_ci(oof: pd.DataFrame, B: int, seed: int, level: float = 0.95) -> dict:
    """Metrics for one (config, model, strategy) across outer repeats.

    ``oof`` has columns patient, repeat, y_true, y_prob, y_pred (inner-threshold) and y_pred_05.
    Point estimate = mean over repeats of the per-repeat pooled out-of-fold metric.
    CI = percentile CI of the same quantity under patient-level bootstrap (resamples patients; every
    repeat is evaluated on the same resample). It reflects test-sample variability only, NOT variability of
    the training procedure, and is therefore optimistic for small data.
    """
    reps = sorted(oof["repeat"].unique())
    pats = np.array(sorted(oof["patient"].unique()))
    pos = {p: i for i, p in enumerate(pats)}
    n = len(pats)
    y = oof.drop_duplicates("patient").set_index("patient").loc[pats, "y_true"].to_numpy().astype(int)
    P = np.full((len(reps), n), np.nan)
    PR = np.full((len(reps), n), np.nan)
    PR05 = np.full((len(reps), n), np.nan)
    for ri, r in enumerate(reps):
        d = oof[oof["repeat"] == r]
        ix = d["patient"].map(pos).to_numpy()
        P[ri, ix], PR[ri, ix], PR05[ri, ix] = d["y_prob"].to_numpy(), d["y_pred"].to_numpy(), d["y_pred_05"].to_numpy()
    idx = bootstrap_indices(n, B, seed, y)
    res = {}
    # AUC
    pt = np.array([auc_matrix(y, P[ri])[0] for ri in range(len(reps))])
    boot = np.mean([auc_matrix(y[idx], P[ri][idx]) for ri in range(len(reps))], axis=0)
    res["roc_auc"] = (float(pt.mean()), *bootstrap_ci(boot, level), float(pt.std(ddof=1)) if len(pt) > 1 else np.nan)
    # Brier (probabilities)
    pt = np.array([np.mean((P[ri] - y) ** 2) for ri in range(len(reps))])
    boot = np.mean([np.mean((P[ri][idx] - y[idx]) ** 2, axis=1) for ri in range(len(reps))], axis=0)
    res["brier"] = (float(pt.mean()), *bootstrap_ci(boot, level), float(pt.std(ddof=1)) if len(pt) > 1 else np.nan)
    # PR-AUC (average precision) - loop bootstrap with fewer resamples for speed
    Bpr = min(B, 300)
    pt = np.array([ap_vector(y, P[ri]) for ri in range(len(reps))])
    boot = np.mean([[ap_vector(y[ix], P[ri][ix]) for ix in idx[:Bpr]] for ri in range(len(reps))], axis=0)
    res["pr_auc"] = (float(pt.mean()), *bootstrap_ci(boot, level), float(pt.std(ddof=1)) if len(pt) > 1 else np.nan)
    # threshold metrics (inner-selected threshold)
    for tag, M in (("", PR), ("@0.5", PR05)):
        def thr_metrics(Yv, Pv):
            tp = ((Pv == 1) & (Yv == 1)).sum(axis=-1)
            fn = ((Pv == 0) & (Yv == 1)).sum(axis=-1)
            tn = ((Pv == 0) & (Yv == 0)).sum(axis=-1)
            fp = ((Pv == 1) & (Yv == 0)).sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sens, spec = tp / (tp + fn), tn / (tn + fp)
                prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
                f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
            return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
                    "balanced_accuracy": (sens + spec) / 2, "accuracy": (tp + tn) / (tp + tn + fp + fn)}
        pts = [thr_metrics(y, M[ri]) for ri in range(len(reps))]
        boots = [thr_metrics(y[idx], M[ri][idx]) for ri in range(len(reps))]
        for k in ("sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "accuracy"):
            ptv = np.array([p[k] for p in pts])
            bv = np.mean([b[k] for b in boots], axis=0)
            res[k + tag] = (float(ptv.mean()), *bootstrap_ci(bv, level),
                            float(ptv.std(ddof=1)) if len(ptv) > 1 else np.nan)
    return res


def decision_curve(y, p, thresholds) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none (Vickers & Elkin)."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum()
        fp = ((pred) & (y == 0)).sum()
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit_model": tp / n - fp / n * w,
                     "net_benefit_all": prev - (1 - prev) * w, "net_benefit_none": 0.0})
    return pd.DataFrame(rows)


def calibration_bins(y, p, n_bins: int = 5) -> pd.DataFrame:
    """Quantile-binned reliability table (few bins because only ~27 events exist)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=n_bins, labels=False)
    g = d.groupby("bin").agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"), events=("y", "sum"))
    return g.reset_index()

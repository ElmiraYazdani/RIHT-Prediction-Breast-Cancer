"""Nested, repeated, stratified cross-validation with leakage-safe feature selection and tuning.

Information flow for ONE outer fold (train T, test V) of ONE feature configuration:

  outer feature selection : fitted on T only                         -> columns used for the final model
  inner CV (K folds of T) : for each inner split (A, B):
        feature selection fitted on A only (cached; model independent)
        Optuna proposes hyper-parameters; the pipeline is fitted on A and scored on B
  best hyper-parameters   : argmax of the mean inner score (outer fold never consulted)
  threshold               : Youden-optimal cut-off of the inner out-of-fold probabilities of the best trial
  final model             : pipeline refitted on all of T with the outer columns; predicts V once

Nothing computed from V (features, labels, predictions) can influence any choice made for V.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import Dataset, combo_name, feature_group_combinations, select_columns
from .evaluation import compute_metrics, youden_threshold
from .feature_selection import select_features
from .models import MODEL_REGISTRY, build_pipeline, is_applicable, predict_proba_pos, suggest_params

optuna.logging.set_verbosity(optuna.logging.ERROR)
logger = logging.getLogger("thyroid_rihp")


def make_outer_splits(y: np.ndarray, n_splits: int, repeat: int, seed: int, groups=None):
    """Shuffled stratified (grouped when patients repeat) splits; identical for every config/model (paired)."""
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
    return list(cv.split(np.zeros(len(y)), y))


def _objective_score(y, p, kind: str) -> float:
    if kind == "neg_brier":
        return -float(np.mean((np.clip(p, 0, 1) - y) ** 2))
    return float(roc_auc_score(y, p))


def run_fold_task(cfg: dict, X: pd.DataFrame, y: np.ndarray, volume: np.ndarray, ids: np.ndarray,
                  f2g: dict, groups, cols: list, combo_label: str, repeat: int, fold: int,
                  train_idx: np.ndarray, test_idx: np.ndarray, models: list, strategies: list) -> dict:
    """One (config, repeat, outer fold) unit of work. Pure function of its arguments (picklable for joblib)."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    base_seed = int(cfg["cv"]["seed"])
    seed = base_seed + 100 * repeat + fold
    Xtr_all, ytr, vtr = X.iloc[train_idx][cols], y[train_idx], volume[train_idx]
    Xte_all, yte = X.iloc[test_idx][cols], y[test_idx]

    # ---- outer feature selection: training rows only ----
    outer_sel = select_features(Xtr_all, ytr, vtr, f2g, cfg, seed=seed)
    sel = outer_sel.selected

    # ---- inner splits + cached, model-independent inner feature selection (inner-training rows only) ----
    k_in = int(cfg["cv"]["inner_folds"])
    if groups is not None:
        inner = list(StratifiedGroupKFold(n_splits=k_in, shuffle=True, random_state=seed)
                     .split(np.zeros(len(ytr)), ytr, groups[train_idx]))
    else:
        inner = list(StratifiedKFold(n_splits=k_in, shuffle=True, random_state=seed)
                     .split(np.zeros(len(ytr)), ytr))
    inner_sel = [select_features(Xtr_all.iloc[a], ytr[a], vtr[a], f2g, cfg, seed=seed + 17 * (i + 1)).selected
                 for i, (a, b) in enumerate(inner)]

    pred_rows, fold_rows, trial_rows = [], [], []
    ocfg = cfg["optuna"]
    for model_name in models:
        for strategy in strategies:
            if not is_applicable(model_name, strategy):
                continue

            def objective(trial, model_name=model_name, strategy=strategy):
                params = suggest_params(model_name, trial)
                scores, oof_idx, oof_p = [], [], []
                for (a, b), s_in in zip(inner, inner_sel):
                    pipe = build_pipeline(model_name, params, strategy, seed, ytr[a])
                    pipe.fit(Xtr_all.iloc[a][s_in].to_numpy(), ytr[a])
                    p = predict_proba_pos(pipe, Xtr_all.iloc[b][s_in].to_numpy())
                    scores.append(_objective_score(ytr[b], p, ocfg["objective"]))
                    oof_idx.append(b)
                    oof_p.append(p)
                trial.set_user_attr("oof_idx", np.concatenate(oof_idx).tolist())
                trial.set_user_attr("oof_p", np.concatenate(oof_p).tolist())
                return float(np.mean(scores))

            sampler = optuna.samplers.TPESampler(seed=seed + int(ocfg.get("sampler_seed_offset", 7)))
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=int(ocfg["n_trials"]), catch=(ValueError, np.linalg.LinAlgError))
            n_failed = sum(t.state.name != "COMPLETE" for t in study.trials)
            if not any(t.state.name == "COMPLETE" for t in study.trials):
                raise RuntimeError(f"All Optuna trials failed for {model_name}/{strategy}")
            best = study.best_trial
            oi, op = np.asarray(best.user_attrs["oof_idx"]), np.asarray(best.user_attrs["oof_p"])
            thr = youden_threshold(ytr[oi], op) if cfg["threshold"]["method"] == "youden_inner" \
                else float(cfg["threshold"]["fixed"])

            pipe = build_pipeline(model_name, best.params, strategy, seed, ytr)
            pipe.fit(Xtr_all[sel].to_numpy(), ytr)
            p_te = predict_proba_pos(pipe, Xte_all[sel].to_numpy())
            p_tr = predict_proba_pos(pipe, Xtr_all[sel].to_numpy())
            m_thr = compute_metrics(yte, p_te, thr)
            m_05 = compute_metrics(yte, p_te, float(cfg["threshold"]["fixed"]))
            row = {
                "config": combo_label, "model": model_name, "strategy": strategy, "repeat": repeat, "fold": fold,
                "n_train": len(train_idx), "n_test": len(test_idx), "n_pos_train": int(ytr.sum()),
                "n_pos_test": int(yte.sum()), "n_features": len(sel), "features": "|".join(sel),
                "best_params": json.dumps(best.params), "inner_cv_score": float(best.value),
                "threshold": float(thr), "n_failed_trials": int(n_failed),
                "train_auc": float(roc_auc_score(ytr, p_tr)),
                "test_auc": m_thr["roc_auc"],
            }
            row.update({k: v for k, v in m_thr.items() if k not in ("roc_auc",)})
            row.update({f"{k}@0.5": v for k, v in m_05.items() if k in
                        ("sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "accuracy")})
            fold_rows.append(row)
            for pid, yt, pp in zip(ids[test_idx], yte, p_te):
                pred_rows.append((combo_label, model_name, strategy, repeat, fold, pid, int(yt), float(pp),
                                  int(pp >= thr), int(pp >= float(cfg["threshold"]["fixed"])), float(thr)))
            for t in study.trials:
                if t.state.name == "COMPLETE":
                    trial_rows.append((combo_label, model_name, strategy, repeat, fold, t.number, t.value,
                                       json.dumps(t.params)))
    sel_set = set(sel)
    selection_rows = [(combo_label, repeat, fold, f, outer_sel.stability_freq.get(f, np.nan), f in sel_set)
                      for f in sorted(set(outer_sel.stability_freq) | sel_set)]
    return {"fold_rows": fold_rows, "pred_rows": pred_rows, "trial_rows": trial_rows,
            "selection_rows": selection_rows, "stage_counts": {**outer_sel.stage_counts,
            "config": combo_label, "repeat": repeat, "fold": fold, "final_threshold": outer_sel.final_threshold},
            "seconds": time.time() - t0}


def _cache_key(cfg, label, repeat, fold, models, strategies) -> str:
    blob = json.dumps({"cfg": {k: cfg[k] for k in ("cv", "feature_selection", "optuna", "threshold")},
                       "label": label, "r": repeat, "f": fold, "m": models, "s": strategies}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def run_nested_cv(cfg: dict, ds: Dataset, models: list, strategies: list, run_dir: Path,
                  combos=None, n_jobs: int | None = None, use_cache: bool = True, verbose: bool = True) -> dict:
    """Run every (config x repeat x outer fold) task in parallel and return concatenated result tables."""
    combos = combos or feature_group_combinations()
    n_jobs = n_jobs or int(cfg["runtime"]["n_jobs"])
    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels = cfg["data"]["group_labels"]
    f2g = ds.feature_to_group
    n_rep, n_out = int(cfg["cv"]["outer_repeats"]), int(cfg["cv"]["outer_folds"])

    splits = {r: make_outer_splits(ds.y, n_out, r, int(cfg["cv"]["seed"]), ds.cv_groups) for r in range(n_rep)}
    tasks = []
    for combo in combos:
        label = combo_name(combo, labels)
        cols = select_columns(ds, combo)
        for r in range(n_rep):
            for f, (tr, te) in enumerate(splits[r]):
                tasks.append((label, cols, r, f, tr, te))

    def work(label, cols, r, f, tr, te):
        key = _cache_key(cfg, label, r, f, models, strategies)
        cf = cache_dir / f"{key}.pkl"
        if use_cache and cf.exists():
            with open(cf, "rb") as fh:
                return pickle.load(fh)
        out = run_fold_task(cfg, ds.X, ds.y, ds.volume, ds.ids, f2g, ds.cv_groups, cols, label, r, f, tr, te,
                            models, strategies)
        if use_cache:
            with open(cf, "wb") as fh:
                pickle.dump(out, fh)
        return out

    if verbose:
        print(f"[run_nested_cv] {len(tasks)} fold-tasks, {len(models)} models x {len(strategies)} strategies, "
              f"{cfg['optuna']['n_trials']} Optuna trials, n_jobs={n_jobs}", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(delayed(work)(*t) for t in tasks)
    if verbose:
        print(f"[run_nested_cv] finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    folds = pd.DataFrame([r for out in results for r in out["fold_rows"]])
    preds = pd.DataFrame([r for out in results for r in out["pred_rows"]],
                         columns=["config", "model", "strategy", "repeat", "fold", "patient", "y_true", "y_prob",
                                  "y_pred", "y_pred_05", "threshold"])
    trials = pd.DataFrame([r for out in results for r in out["trial_rows"]],
                          columns=["config", "model", "strategy", "repeat", "fold", "trial", "inner_score", "params"])
    selection = pd.DataFrame([r for out in results for r in out["selection_rows"]],
                             columns=["config", "repeat", "fold", "feature", "stability_freq", "selected"])
    stages = pd.DataFrame([out["stage_counts"] for out in results])
    return {"folds": folds, "preds": preds, "trials": trials, "selection": selection, "stages": stages}

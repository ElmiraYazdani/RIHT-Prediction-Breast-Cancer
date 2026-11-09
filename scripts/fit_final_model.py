"""Fit ONE descriptive final model on all 115 patients for a chosen configuration (default: the pre-specified
primary model/strategy).

WARNING: features and hyper-parameters are selected on all data (inner CV only). The resulting model has NO unbiased
performance estimate - quote nested-CV results instead. It is written to outputs/final_model/ (git-ignored) only so
coefficients can be inspected; it is not externally validated and must not be used clinically.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import combo_name, feature_group_combinations, load_dataset, select_columns
from thyroid_rihp.evaluation import youden_threshold
from thyroid_rihp.feature_selection import select_features
from thyroid_rihp.models import build_pipeline, predict_proba_pos, suggest_params
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="DVH")
    a = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(cfg)
    labels = cfg["data"]["group_labels"]
    combos = {combo_name(c, labels): c for c in feature_group_combinations()}
    cols = select_columns(ds, combos[a.config])
    model, strategy = cfg["models"]["primary_model"], cfg["models"]["primary_strategy"]
    seed = int(cfg["cv"]["seed"])
    sel = select_features(ds.X[cols], ds.y, ds.volume, ds.feature_to_group, cfg, seed=seed)
    Xs = ds.X[sel.selected].to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ds.y))

    def objective(trial):
        params = suggest_params(model, trial)
        s = []
        for tr, va in splits:
            pipe = build_pipeline(model, params, strategy, seed, ds.y[tr]).fit(Xs[tr], ds.y[tr])
            s.append(roc_auc_score(ds.y[va], predict_proba_pos(pipe, Xs[va])))
        return float(np.mean(s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(objective, n_trials=int(cfg["optuna"]["n_trials"]))
    pipe = build_pipeline(model, study.best_params, strategy, seed, ds.y).fit(Xs, ds.y)
    out = resolve(cfg, "output_dir") / "final_model"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / f"final_{model}_{strategy}.joblib")
    coef = pd.DataFrame({"feature": sel.selected, "standardised_coef": pipe.named_steps["clf"].coef_[0]})
    coef.to_csv(out / "final_model_coefficients.csv", index=False)
    with open(out / "final_model_info.json", "w", encoding="utf-8") as fh:
        json.dump({"config": a.config, "model": model, "strategy": strategy, "features": sel.selected,
                   "best_params": study.best_params, "inner_cv_auc": study.best_value,
                   "WARNING": "descriptive model fitted on all data; no unbiased performance estimate; not externally validated"},
                  fh, indent=2)
    print(coef.to_string(index=False))


if __name__ == "__main__":
    main()

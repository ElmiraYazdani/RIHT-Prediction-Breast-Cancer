"""Label-permutation null distribution of the WHOLE nested pipeline (feature selection + tuning + fit).

For each permutation the outcome vector is shuffled across patients, new stratified outer folds are drawn and the
complete nested procedure is repeated for the primary model/strategy. The pooled out-of-fold AUC of the real labels
(single repeat, same design) is compared with this null. A label-permutation p-value is a sanity check against
pipeline leakage / optimistic bias, NOT a model-comparison test.

Usage: python scripts/run_permutation_test.py --n-perm 50 --configs "DVH" "CBs" "CT RFs"
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import combo_name, feature_group_combinations, load_dataset, select_columns
from thyroid_rihp.nested_cv import make_outer_splits, run_fold_task


def one_perm(cfg, ds, cols, label, k, y_perm, model, strategy):
    rows = []
    for f, (tr, te) in enumerate(make_outer_splits(y_perm, int(cfg["cv"]["outer_folds"]), 1000 + k, int(cfg["cv"]["seed"]))):
        out = run_fold_task(cfg, ds.X, y_perm, ds.volume, ds.ids, ds.feature_to_group, None, cols, label, 0, f, tr, te,
                            [model], [strategy])
        rows += [(r[5], r[6], r[7]) for r in out["pred_rows"]]
    d = pd.DataFrame(rows, columns=["patient", "y", "p"])
    return label, k, roc_auc_score(d.y, d.p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--configs", nargs="+", default=["CBs", "DVH", "CT RFs", "CBs+DVH+CT RFs+DFs"])
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(cfg)
    model, strategy = cfg["models"]["primary_model"], cfg["models"]["primary_strategy"]
    labels = cfg["data"]["group_labels"]
    combos = {combo_name(c, labels): c for c in feature_group_combinations()}
    rng = np.random.RandomState(int(cfg["cv"]["seed"]) + 99)
    tasks = []
    for name in a.configs:
        cols = select_columns(ds, combos[name])
        for k in range(a.n_perm):
            tasks.append((cfg, ds, cols, name, k, rng.permutation(ds.y), model, strategy))
    res = Parallel(n_jobs=a.n_jobs or int(cfg["runtime"]["n_jobs"]), verbose=5)(delayed(one_perm)(*t, ) for t in tasks)
    null = pd.DataFrame(res, columns=["config", "perm", "null_auc"])
    out_dir = resolve(cfg, "output_dir") / a.run_name / "tables"
    null.to_csv(out_dir / "permutation_null_auc.csv", index=False)

    preds = pd.read_csv(resolve(cfg, "output_dir") / a.run_name / "predictions" / "oof_predictions.csv")
    d = preds[(preds.model == model) & (preds.strategy == strategy) & (preds.repeat == 0)]
    rows = []
    for name, g in null.groupby("config"):
        obs_df = d[d.config == name]
        obs = roc_auc_score(obs_df.y_true, obs_df.y_prob)
        rows.append({"config": name, "observed_auc_repeat0": obs, "n_perm": len(g), "null_mean": g.null_auc.mean(),
                     "null_sd": g.null_auc.std(ddof=1), "null_q95": g.null_auc.quantile(.95),
                     "perm_p_value": (1 + (g.null_auc >= obs).sum()) / (1 + len(g))})
    pd.DataFrame(rows).to_csv(out_dir / "permutation_test_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()

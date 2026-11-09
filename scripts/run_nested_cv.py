"""Run the nested, repeated, stratified CV for all feature-group configurations.

Examples
--------
  python scripts/run_nested_cv.py --run-name main                      # 15 configs x 4 models x 3 strategies
  python scripts/run_nested_cv.py --run-name extended --models extended --strategies none
  python scripts/run_nested_cv.py --run-name smoke --smoke             # fast end-to-end check
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker; parallelism is across fold-tasks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import feature_group_combinations, load_dataset
from thyroid_rihp.nested_cv import run_nested_cv
from thyroid_rihp.reporting import save_run_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--models", default="main", help="'main', 'extended', 'all' or comma-separated names")
    ap.add_argument("--strategies", default=None, help="comma-separated subset of none,class_weight,smote")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--stability-iterations", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 configs, 1 repeat, 5 trials, 30 stability iterations")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value by dotted path, e.g. feature_selection.stability.threshold=0.5")
    a = ap.parse_args()

    ov = {}
    if a.repeats is not None:
        ov["cv"] = {"outer_repeats": a.repeats}
    if a.trials is not None:
        ov["optuna"] = {"n_trials": a.trials}
    if a.stability_iterations is not None:
        ov["feature_selection"] = {"stability": {"n_iterations": a.stability_iterations}}
    if a.smoke:
        ov = {"cv": {"outer_repeats": 1}, "optuna": {"n_trials": 5},
              "feature_selection": {"stability": {"n_iterations": 30}}, "evaluation": {"bootstrap_n": 100}}
    import yaml
    from thyroid_rihp.config import _deep_update
    for item in a.set:
        key, val = item.split("=", 1)
        d = yaml.safe_load(val)
        for part in reversed(key.split(".")):
            d = {part: d}
        ov = _deep_update(ov, d)
    cfg = load_config(a.config, ov)

    mc = cfg["models"]
    models = mc["main"] if a.models == "main" else mc["extended"] if a.models == "extended" \
        else mc["main"] + mc["extended"] if a.models == "all" else a.models.split(",")
    strategies = a.strategies.split(",") if a.strategies else mc["strategies"]

    ds = load_dataset(cfg)
    combos = feature_group_combinations()
    if a.smoke:
        combos = [("DVH",), ("CB", "DVH", "RF", "DF")]
        models, strategies = ["logreg_l2", "random_forest"], ["none", "smote"]
    out_dir = resolve(cfg, "output_dir") / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump({"config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "models": models,
                   "strategies": strategies, "n_configs": len(combos), "n_patients": len(ds.y),
                   "n_events": int(ds.y.sum())}, fh, indent=2, default=str)
    res = run_nested_cv(cfg, ds, models, strategies, out_dir, combos=combos, n_jobs=a.n_jobs,
                        use_cache=not a.no_cache)
    save_run_outputs(res, out_dir, cfg)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

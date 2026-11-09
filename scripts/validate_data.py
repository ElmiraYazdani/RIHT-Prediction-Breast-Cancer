"""Validate the four Excel files and write a (name-free) data-validation report.

Outputs: outputs/data_validation/data_validation.json and data_summary.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from thyroid_rihp.config import load_config, resolve
from thyroid_rihp.data import load_dataset


def main():
    cfg = load_config()
    ds = load_dataset(cfg)
    out = resolve(cfg, "output_dir") / "data_validation"
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(ds.validation)
    rep["feature_names_per_group_head"] = {g: f[:5] for g, f in ds.feature_groups.items()}
    rep["thyroid_volume_cm3"] = {"mean": float(ds.volume.mean()), "min": float(ds.volume.min()),
                                 "max": float(ds.volume.max())}
    rep["manuscript_vs_data"] = {
        "manuscript_cohort_n": 120, "manuscript_positives": 20, "manuscript_after_preprocessing_n": 115,
        "data_n": rep["n_patients"], "data_positives": rep["n_positive"],
        "manuscript_features_after_preprocessing": 107,
        "data_features_per_group": rep["n_features_per_group"],
    }
    with open(out / "data_validation.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    rows = []
    for g, fs in ds.feature_groups.items():
        sub = ds.X[fs]
        rows.append({"group": g, "n_features": len(fs), "missing": int(sub.isna().sum().sum()),
                     "constant": int((sub.nunique() <= 1).sum())})
    pd.DataFrame(rows).to_csv(out / "data_summary.csv", index=False)
    for k in ("n_patients", "n_positive", "n_negative", "prevalence", "n_features_per_group", "n_missing_values",
              "label_consistent_across_files", "ids_identical_across_files", "outcome_sorted_by_row_order",
              "rows_sharing_identical_name_in_CB", "name_mismatch_ids_vs_CB"):
        print(f"{k}: {rep.get(k)}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()

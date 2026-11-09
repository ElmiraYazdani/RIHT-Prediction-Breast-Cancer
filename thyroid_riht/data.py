"""Data loading, validation and feature-group construction.

Design rules
------------
* Patient names are read ONLY to cross-check that the four Excel files describe the same patients;
  they are never stored in the returned objects and never written to disk.
* Feature names are prefixed with their group (CB_, DVH_, RF_, DF_) because the radiomics and
  dosiomics files share identical column names.
* Files are merged on the patient ID (never by row position); the outcome label must agree across files.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import resolve

GROUP_ORDER = ["CB", "DVH", "RF", "DF"]


@dataclass
class Dataset:
    X: pd.DataFrame                    # all candidate features, prefixed names, index = patient ID
    y: np.ndarray                      # 0/1 outcome
    ids: np.ndarray                    # patient IDs (study codes, not names)
    feature_groups: dict               # group -> list of prefixed feature names
    volume: np.ndarray                 # thyroid volume (cm3), used only by the volume-dependence filter
    cv_groups: np.ndarray | None       # patient-level grouping vector when IDs repeat, else None
    validation: dict = field(default_factory=dict)

    @property
    def feature_to_group(self) -> dict:
        return {f: g for g, fs in self.feature_groups.items() for f in fs}


def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)


def feature_group_combinations(groups=GROUP_ORDER) -> list[tuple[str, ...]]:
    """All 15 non-empty combinations, ordered by size then canonical group order."""
    combos = []
    for r in range(1, len(groups) + 1):
        combos.extend(itertools.combinations(groups, r))
    return combos


def combo_name(combo, labels: dict | None = None) -> str:
    labels = labels or {}
    return "+".join(labels.get(g, g) for g in combo)


def assert_no_forbidden_features(columns, cfg) -> None:
    """Raise if any candidate feature name looks like an identifier or a post-outcome variable."""
    pats = [p.lower() for p in cfg["data"]["forbidden_patterns"]]
    exempt = set(cfg["data"].get("forbidden_exempt", []))
    bad = []
    for c in columns:
        base = re.sub(r"^(CB|DVH|RF|DF)_", "", str(c)).lower()
        if c in exempt:
            continue
        if any(p in base for p in pats):
            bad.append(c)
    if bad:
        raise ValueError(f"Forbidden (identifier / outcome-related) columns in feature matrix: {bad}")


def load_dataset(cfg: dict, verbose: bool = False) -> Dataset:
    data_dir = resolve(cfg, "data_dir")
    dcfg = cfg["data"]
    id_col, label_col, name_cols = dcfg["id_col"], dcfg["label_col"], dcfg["name_cols"]

    raw, labels, names, report = {}, {}, {}, {"files": {}}
    for g, spec in dcfg["files"].items():
        path = data_dir / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing data file for group {g}: {path}")
        df = pd.read_excel(path, sheet_name=spec.get("sheet", 0))
        report["files"][g] = {"file": spec["file"], "rows": int(df.shape[0]), "columns": int(df.shape[1])}
        if id_col not in df.columns:
            raise ValueError(f"{spec['file']}: id column '{id_col}' not found")
        if df[id_col].isna().any():
            raise ValueError(f"{spec['file']}: missing patient IDs")
        df[id_col] = df[id_col].astype(str)
        nm = [c for c in name_cols if c in df.columns]
        names[g] = _norm_name(df[nm[0]]).set_axis(df[id_col].values) if nm else None
        if label_col in df.columns:
            labels[g] = pd.Series(df[label_col].values, index=df[id_col].values)
        raw[g] = df.drop(columns=[c for c in name_cols + [label_col] if c in df.columns])

    # ---- identifier consistency across the four files ----
    id_sets = {g: set(df[id_col]) for g, df in raw.items()}
    ref_ids = id_sets[GROUP_ORDER[0]]
    report["ids_identical_across_files"] = all(s == ref_ids for s in id_sets.values())
    if not report["ids_identical_across_files"]:
        common = set.intersection(*id_sets.values())
        only = sorted(set.union(*id_sets.values()) - common)
        raise ValueError(f"Patient IDs differ between files (not present in all files: {only})")
    report["duplicated_ids_per_file"] = {g: int(df[id_col].duplicated().sum()) for g, df in raw.items()}
    if any(report["duplicated_ids_per_file"].values()):
        raise ValueError("Duplicated IDs prevent a safe ID-based merge of the four files")

    base_ids = raw["CB"][id_col].tolist()
    report["row_order_identical_across_files"] = all(df[id_col].tolist() == base_ids for df in raw.values())

    # ---- align every file to the CB order via the ID (never by position) ----
    feats, feature_groups = [], {}
    for g in GROUP_ORDER:
        df = raw[g].set_index(id_col).loc[base_ids]
        num = df.select_dtypes(include=[np.number])
        dropped = [c for c in df.columns if c not in num.columns]
        if dropped:
            report.setdefault("non_numeric_dropped", {})[g] = dropped
        num = num.rename(columns={c: f"{g}_{c}" for c in num.columns})
        feature_groups[g] = list(num.columns)
        feats.append(num)
    X = pd.concat(feats, axis=1)
    X.index = pd.Index(base_ids, name=id_col)
    assert_no_forbidden_features(X.columns, cfg)

    # ---- labels must agree across files ----
    lab_df = pd.DataFrame({g: s.loc[base_ids] for g, s in labels.items()})
    if lab_df.empty:
        raise ValueError("No outcome label found in any file")
    report["label_consistent_across_files"] = bool((lab_df.nunique(axis=1) == 1).all())
    if not report["label_consistent_across_files"]:
        raise ValueError("Outcome labels disagree between files")
    y_raw = lab_df.iloc[:, 0]
    if y_raw.isna().any():
        raise ValueError("Missing outcome labels")
    if not set(np.unique(y_raw)).issubset({0, 1}):
        raise ValueError(f"Outcome must be binary 0/1, found {np.unique(y_raw)}")
    y = y_raw.astype(int).values

    # ---- name-based cross-check (names never leave this function) ----
    mism = {}
    if names.get("CB") is not None:
        for g in GROUP_ORDER:
            if names.get(g) is not None:
                diff = names[g].loc[base_ids].values != names["CB"].loc[base_ids].values
                mism[g] = [base_ids[i] for i in np.where(diff)[0]]
        nm_cb = names["CB"].loc[base_ids]
        report["rows_sharing_identical_name_in_CB"] = sorted(nm_cb.index[nm_cb.duplicated(keep=False)].tolist())
    report["name_mismatch_ids_vs_CB"] = mism
    del names

    # ---- volume used by the volume-dependence filter ----
    vs = dcfg["volume_source"]
    vcol = f"{vs['group']}_{vs['column']}"
    if vcol not in X.columns:
        raise ValueError(f"Volume column {vcol} not found")
    volume = X[vcol].values.astype(float)

    # ---- structural checks ----
    n_pos = int(y.sum())
    report.update(
        n_patients=int(len(y)), n_positive=n_pos, n_negative=int(len(y) - n_pos),
        prevalence=float(n_pos / len(y)),
        n_features_per_group={g: len(f) for g, f in feature_groups.items()},
        n_missing_values=int(X.isna().sum().sum()),
        n_infinite_values=int(np.isinf(X.to_numpy(dtype=float)).sum()),
        constant_features=[c for c in X.columns if X[c].nunique(dropna=True) <= 1],
        outcome_sorted_by_row_order=bool(np.all(np.diff(y) <= 0) or np.all(np.diff(y) >= 0)),
        id_is_unique=True,
    )
    cv_groups = None  # one row per patient (IDs unique) -> no patient-level grouping needed
    report["patient_grouping_needed"] = False
    if verbose:
        print({k: v for k, v in report.items() if k != "files"})
    return Dataset(X=X, y=y, ids=np.asarray(base_ids), feature_groups=feature_groups,
                   volume=volume, cv_groups=cv_groups, validation=report)


def select_columns(ds: Dataset, combo) -> list[str]:
    cols = []
    for g in combo:
        cols.extend(ds.feature_groups[g])
    return cols

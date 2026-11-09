import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from thyroid_rihp.config import load_config
from thyroid_rihp.data import Dataset

FAST = {"cv": {"outer_repeats": 1, "outer_folds": 4, "inner_folds": 3},
        "optuna": {"n_trials": 3},
        "feature_selection": {"stability": {"n_iterations": 25},
                              "mrmr": {"k_per_group": {"CB": 3, "DVH": 3, "RF": 4, "DF": 4}}},
        "evaluation": {"bootstrap_n": 60}}


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides=FAST)


def make_synthetic(n=64, n_pos=16, seed=0, signal=True):
    """Synthetic cohort with the same group structure as the real data (prefixed names)."""
    rng = np.random.RandomState(seed)
    y = np.array([1] * n_pos + [0] * (n - n_pos))      # sorted by outcome on purpose, like the real files
    vol = rng.gamma(9, 1.1, n)
    cols = {}
    for i in range(3):
        cols[f"CB_c{i}"] = rng.normal(size=n)
    cols["CB_VOLCTCM3"] = vol
    for i in range(4):
        cols[f"DVH_d{i}"] = rng.normal(size=n) + (1.5 * y if (signal and i == 0) else 0)
    for g in ("RF", "DF"):
        for i in range(12):
            cols[f"{g}_f{i}"] = (rng.normal(size=n) + (vol * 0.5 if i == 0 else 0)
                                 + (0.8 * y if (signal and g == "DF" and i == 1) else 0))
    X = pd.DataFrame(cols, index=[f"P{i + 1}" for i in range(n)])
    groups = {g: [c for c in X.columns if c.startswith(g + "_")] for g in ("CB", "DVH", "RF", "DF")}
    return Dataset(X=X, y=y, ids=X.index.to_numpy(), feature_groups=groups, volume=vol, cv_groups=None,
                   validation={})


@pytest.fixture(scope="session")
def synth():
    return make_synthetic()


def write_excel_set(dir_, ds, names=True):
    """Write the four Excel files for a synthetic dataset (names are fake)."""
    for g, fname, nc in (("CB", "CB_Thyroid.xlsx", "Unnamed: 0"), ("DVH", "DVH_Thyroid.xlsx", "Unnamed: 0"),
                         ("RF", "Radiomics_Thyroid.xlsx", "Patient name"),
                         ("DF", "Dosiomics_Thyroid.xlsx", "Patient name")):
        df = pd.DataFrame({c[len(g) + 1:]: ds.X[c].values for c in ds.feature_groups[g]})
        df.insert(0, "ID", ds.ids)
        if names:
            df.insert(0, nc, [f"FAKE,NAME{i}" for i in range(len(ds.ids))])
        df["group"] = ds.y
        df.to_excel(os.path.join(dir_, fname), index=False)

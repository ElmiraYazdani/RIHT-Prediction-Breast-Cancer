import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic, write_excel_set
from thyroid_rihp.config import PROJECT_ROOT, load_config
from thyroid_rihp.data import (assert_no_forbidden_features, combo_name, feature_group_combinations, load_dataset,
                               select_columns)

ALL_FILES = ("CB_Thyroid.xlsx", "DVH_Thyroid.xlsx", "Radiomics_Thyroid.xlsx", "Dosiomics_Thyroid.xlsx")


def _cfg_for(tmp_path):
    cfg = load_config()
    cfg["paths"]["data_dir"] = str(tmp_path)
    return cfg


def _rewrite(path, fn):
    df = pd.read_excel(path)
    fn(df)
    df.to_excel(path, index=False)


def test_loader_roundtrip_and_no_names(tmp_path):
    ds0 = make_synthetic()
    write_excel_set(tmp_path, ds0)
    ds = load_dataset(_cfg_for(tmp_path))
    assert ds.X.shape[0] == 64 and ds.y.sum() == 16
    assert all(c.split("_")[0] in ("CB", "DVH", "RF", "DF") for c in ds.X.columns)
    assert ds.X.select_dtypes(exclude=[np.number]).shape[1] == 0       # nothing non-numeric (names) leaked in
    assert not any(c.endswith("_group") for c in ds.X.columns)         # label not among the features


def test_merge_is_by_id_not_position(tmp_path):
    ds0 = make_synthetic()
    write_excel_set(tmp_path, ds0)
    p = tmp_path / "DVH_Thyroid.xlsx"
    pd.read_excel(p).iloc[::-1].to_excel(p, index=False)               # reverse row order in one file
    ds = load_dataset(_cfg_for(tmp_path))
    assert ds.validation["row_order_identical_across_files"] is False
    assert np.allclose(ds.X["DVH_d0"].values, ds0.X["DVH_d0"].values)  # still aligned by ID


def test_label_disagreement_raises(tmp_path):
    write_excel_set(tmp_path, make_synthetic())
    _rewrite(tmp_path / "DVH_Thyroid.xlsx", lambda d: d.__setitem__("group", 1 - d["group"]))
    with pytest.raises(ValueError, match="disagree"):
        load_dataset(_cfg_for(tmp_path))


def test_id_mismatch_raises(tmp_path):
    write_excel_set(tmp_path, make_synthetic())
    _rewrite(tmp_path / "Dosiomics_Thyroid.xlsx", lambda d: d.__setitem__("ID", d["ID"].replace("P1", "PX")))
    with pytest.raises(ValueError, match="IDs differ"):
        load_dataset(_cfg_for(tmp_path))


def test_duplicate_id_raises(tmp_path):
    write_excel_set(tmp_path, make_synthetic())
    for f in ALL_FILES:
        _rewrite(tmp_path / f, lambda d: d.__setitem__("ID", d["ID"].replace("P2", "P1")))
    with pytest.raises(ValueError):
        load_dataset(_cfg_for(tmp_path))


def test_non_binary_label_raises(tmp_path):
    write_excel_set(tmp_path, make_synthetic())
    for f in ALL_FILES:
        _rewrite(tmp_path / f, lambda d: d.__setitem__("group", d["group"].replace(1, 2)))
    with pytest.raises(ValueError, match="binary"):
        load_dataset(_cfg_for(tmp_path))


@pytest.mark.parametrize("bad", ["CB_TSH_followup", "DVH_outcome_flag", "RF_Patient name", "DF_group",
                                 "CB_Unnamed: 0"])
def test_forbidden_columns_blocked(bad):
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_forbidden_features(["CB_AGE", bad], load_config())


def test_allowed_columns_pass():
    assert_no_forbidden_features(["CB_AGE", "CB_VOLCTCM3", "DVH_V30", "RF_glcm Idm", "DF_firstorder Mean"],
                                 load_config())


def test_fifteen_unique_nonempty_combinations():
    combos = feature_group_combinations()
    assert len(combos) == 15 and len(set(combos)) == 15 and all(len(c) >= 1 for c in combos)
    assert sorted(len(c) for c in combos) == [1] * 4 + [2] * 6 + [3] * 4 + [4]
    names = [combo_name(c, load_config()["data"]["group_labels"]) for c in combos]
    assert names[0] == "CBs" and names[-1] == "CBs+DVH+CT RFs+DFs" and len(set(names)) == 15


def test_combo_columns_are_group_unions():
    ds = make_synthetic()
    for combo in feature_group_combinations():
        cols = select_columns(ds, combo)
        assert len(cols) == len(set(cols)) == sum(len(ds.feature_groups[g]) for g in combo)
        assert {c.split("_")[0] for c in cols} == set(combo)


REAL = PROJECT_ROOT / "data" / "CB_Thyroid.xlsx"


@pytest.mark.skipif(not REAL.exists(), reason="confidential data not present")
def test_real_data_schema():
    ds = load_dataset(load_config())
    v = ds.validation
    assert v["n_patients"] == 115 and v["n_positive"] == 27       # NOT 120/20 as in the manuscript abstract
    assert v["n_features_per_group"] == {"CB": 5, "DVH": 12, "RF": 107, "DF": 107}
    assert v["label_consistent_across_files"] and v["ids_identical_across_files"]
    assert v["n_missing_values"] == 0 and v["n_infinite_values"] == 0
    assert not any("tsh" in c.lower() for c in ds.X.columns)      # follow-up TSH is not in the feature matrix

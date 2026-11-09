"""Leakage-prevention tests: nothing derived from the outer test rows may change any training-side decision."""
import numpy as np

from thyroid_rihp.data import select_columns
from thyroid_rihp.feature_selection import select_features
from thyroid_rihp.models import build_pipeline, predict_proba_pos
from thyroid_rihp.nested_cv import make_outer_splits, run_fold_task

TRAIN_SIDE = ("features", "best_params", "inner_cv_score", "threshold", "train_auc")


def _task(cfg, ds, X, y, tr, te, models=("logreg_l2",), strategies=("smote", "class_weight")):
    cols = select_columns(ds, ("CB", "DVH", "RF", "DF"))
    return run_fold_task(cfg, X, y, ds.volume, ds.ids, ds.feature_to_group, None, cols, "all", 0, 0, tr, te,
                         list(models), list(strategies))


def test_test_labels_do_not_influence_selection_tuning_or_threshold(cfg, synth):
    ds = synth
    tr, te = make_outer_splits(ds.y, 4, 0, 111)[0]
    base = _task(cfg, ds, ds.X, ds.y, tr, te)
    y2 = ds.y.copy()
    y2[te] = np.random.RandomState(1).permutation(y2[te])            # scramble ONLY the outer-test labels
    alt = _task(cfg, ds, ds.X, y2, tr, te)
    for a, b in zip(base["fold_rows"], alt["fold_rows"]):
        for k in TRAIN_SIDE:
            assert a[k] == b[k], f"{k} depends on outer-test labels"
    assert [r[7] for r in base["pred_rows"]] == [r[7] for r in alt["pred_rows"]]   # predictions identical too


def test_test_features_do_not_influence_selection_or_fit(cfg, synth):
    ds = synth
    tr, te = make_outer_splits(ds.y, 4, 0, 111)[1]
    base = _task(cfg, ds, ds.X, ds.y, tr, te)
    X2 = ds.X.copy()
    X2.iloc[te] = X2.iloc[te] * 50 + 1000                            # wreck the test features only
    alt = _task(cfg, ds, X2, ds.y, tr, te)
    for a, b in zip(base["fold_rows"], alt["fold_rows"]):
        for k in TRAIN_SIDE:
            assert a[k] == b[k], f"{k} depends on outer-test features"


def test_one_test_row_does_not_influence_another_test_prediction(cfg, synth):
    ds = synth
    tr, te = make_outer_splits(ds.y, 4, 0, 111)[2]
    base = _task(cfg, ds, ds.X, ds.y, tr, te, strategies=("none",))
    X2 = ds.X.copy()
    X2.iloc[te[0]] = X2.iloc[te[0]] * 99
    alt = _task(cfg, ds, X2, ds.y, tr, te, strategies=("none",))
    p0, p1 = [r[7] for r in base["pred_rows"]], [r[7] for r in alt["pred_rows"]]
    assert p0[1:] == p1[1:]                                          # only the modified row's prediction may move


def test_selection_is_deterministic_function_of_training_rows(cfg, synth):
    ds = synth
    cols = select_columns(ds, ("DVH", "RF", "DF"))
    tr = np.arange(0, 48)
    r1 = select_features(ds.X.iloc[tr][cols], ds.y[tr], ds.volume[tr], ds.feature_to_group, cfg, seed=3)
    r2 = select_features(ds.X.iloc[tr][cols].copy(), ds.y[tr].copy(), ds.volume[tr].copy(),
                         ds.feature_to_group, cfg, seed=3)
    assert r1.selected == r2.selected


def test_volume_filter_removes_volume_proxy(cfg, synth):
    ds = synth
    cols = select_columns(ds, ("RF", "DF"))
    r = select_features(ds.X[cols], ds.y, ds.volume, ds.feature_to_group, cfg, seed=0)
    assert "RF_f0" in r.dropped_volume and "DF_f0" in r.dropped_volume   # built to track volume
    assert "RF_f0" not in r.selected and "DF_f0" not in r.selected


def test_scaler_and_imputer_fit_on_training_rows_only(synth):
    ds = synth
    X = ds.X[["DVH_d0", "DVH_d1", "DVH_d2"]].to_numpy()
    tr, te = np.arange(0, 40), np.arange(40, 64)
    pipe = build_pipeline("logreg_l2", {"C": 1.0}, "smote", 0, ds.y[tr])
    pipe.fit(X[tr], ds.y[tr])
    assert np.allclose(pipe.named_steps["scaler"].mean_, X[tr].mean(axis=0))          # train statistics only
    assert not np.allclose(pipe.named_steps["scaler"].mean_, X.mean(axis=0))
    assert len(predict_proba_pos(pipe, X[te])) == len(te)                              # no resampling at predict time


def test_smote_is_applied_only_inside_fit(synth):
    ds = synth
    X, y = ds.X[["DVH_d0", "DVH_d1"]].to_numpy(), ds.y
    pipe = build_pipeline("logreg_l2", {"C": 1.0}, "smote", 0, y)
    seen = {}
    smote = pipe.named_steps["smote"]
    orig = smote.fit_resample

    def spy(Xa, ya):
        seen["n"], seen["pos"] = len(ya), int(ya.sum())
        return orig(Xa, ya)
    smote.fit_resample = spy
    pipe.fit(X, y)
    assert seen["n"] == 64 and seen["pos"] == 16            # sees exactly the (imbalanced) training rows
    pipe.predict_proba(X[:10])
    assert seen["n"] == 64                                  # untouched by prediction


def test_outer_folds_are_disjoint_stratified_and_paired(synth):
    y = synth.y
    s0 = make_outer_splits(y, 4, 0, 111)
    s0b = make_outer_splits(y, 4, 0, 111)
    s1 = make_outer_splits(y, 4, 1, 111)
    te_all = np.concatenate([te for _, te in s0])
    assert sorted(te_all) == list(range(len(y)))                     # every patient tested exactly once
    for tr, te in s0:
        assert not set(tr) & set(te)
        assert abs(y[te].mean() - y.mean()) < 0.1                    # stratified
    assert all((a[1] == b[1]).all() for a, b in zip(s0, s0b))        # reproducible
    assert not all((a[1] == b[1]).all() for a, b in zip(s0, s1))     # repeats differ
    assert y[s0[0][1]].min() == 0 and y[s0[0][1]].max() == 1         # shuffled: data are sorted by outcome


def test_group_aware_split_never_splits_a_patient():
    y = np.array([1] * 12 + [0] * 36)
    groups = np.repeat(np.arange(24), 2)
    for tr, te in make_outer_splits(y, 4, 0, 5, groups=groups):
        assert not set(groups[tr]) & set(groups[te])

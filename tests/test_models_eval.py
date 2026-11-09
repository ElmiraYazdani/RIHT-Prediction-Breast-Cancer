import numpy as np
import optuna
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from conftest import make_synthetic
from thyroid_rihp.evaluation import (auc_matrix, bootstrap_indices, calibration_bins, compute_metrics,
                                     decision_curve, youden_threshold)
from thyroid_rihp.models import (MODEL_REGISTRY, STRATEGIES, SafeSMOTE, build_pipeline, is_applicable,
                                 predict_proba_pos, suggest_params)
from thyroid_rihp.stats import delong_test, holm

optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.mark.parametrize("model", list(MODEL_REGISTRY))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_model_strategy_fits_and_predicts(model, strategy):
    if not is_applicable(model, strategy):
        pytest.skip("not applicable")
    ds = make_synthetic(60, 14)
    X, y = ds.X[["DVH_d0", "DVH_d1", "DVH_d2"]].to_numpy(), ds.y
    params = suggest_params(model, optuna.create_study().ask())
    pipe = build_pipeline(model, params, strategy, 0, y)
    pipe.fit(X, y)
    p = predict_proba_pos(pipe, X)
    assert p.shape == (60,) and np.all((p >= 0) & (p <= 1))
    # (randomly sampled hyper-parameters may legitimately shrink a linear model to a constant, so only
    #  validity of the probabilities is asserted here; learning ability is checked below)


@pytest.mark.parametrize("model,params", [("logreg_l2", {"C": 1.0}), ("logreg_enet", {"C": 1.0, "l1_ratio": 0.3}),
                                          ("svm_linear", {"C": 1.0}), ("naive_bayes", {"var_smoothing": 1e-6})])
def test_models_learn_a_clear_signal(model, params):
    ds = make_synthetic(60, 14)
    X, y = ds.X[["DVH_d0", "DVH_d1", "DVH_d2"]].to_numpy(), ds.y
    pipe = build_pipeline(model, params, "none", 0, y).fit(X, y)
    assert roc_auc_score(y, predict_proba_pos(pipe, X)) > 0.8


def test_class_weight_not_applicable_where_unsupported():
    assert not is_applicable("knn", "class_weight") and not is_applicable("naive_bayes", "class_weight")
    assert is_applicable("logreg_l2", "class_weight") and is_applicable("knn", "smote")


def test_safe_smote_handles_tiny_minority():
    X = np.random.RandomState(0).normal(size=(20, 3))
    y = np.array([1, 1, 1] + [0] * 17)
    _, yr = SafeSMOTE(random_state=0).fit_resample(X, y)
    assert yr.sum() == 17
    _, y1 = SafeSMOTE(random_state=0).fit_resample(X, np.array([1] + [0] * 19))
    assert len(y1) == 20                                             # one minority sample: returned unchanged


def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.1, 0.05])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))


def test_no_positive_predictions_gives_zero_precision_not_error():
    m = compute_metrics(np.array([1, 0, 0, 1]), np.array([0.1, 0.1, 0.2, 0.3]), 0.9)
    assert m["precision"] == 0 and m["sensitivity"] == 0 and m["specificity"] == 1


def test_youden_uses_only_given_data():
    y = np.array([0] * 10 + [1] * 10)
    p = np.r_[np.linspace(0.0, 0.4, 10), np.linspace(0.6, 1.0, 10)]
    assert 0.4 < youden_threshold(y, p) <= 0.6
    assert youden_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.5


def test_auc_matrix_matches_sklearn_and_bootstrap_shape():
    rng = np.random.RandomState(1)
    y = np.r_[np.ones(15), np.zeros(45)].astype(int)
    p = rng.rand(60) + 0.4 * y
    assert auc_matrix(y, p)[0] == pytest.approx(roc_auc_score(y, p))
    idx = bootstrap_indices(60, 50, 0, y)
    assert idx.shape == (50, 60) and (y[idx].sum(axis=1) > 0).all() and (y[idx].sum(axis=1) < 60).all()
    a = auc_matrix(y[idx], p[idx])
    assert a.shape == (50,) and 0.5 < a.mean() < 1


def test_decision_curve_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1])
    d = decision_curve(y, p, [0.5])
    assert d.net_benefit_model[0] == pytest.approx(2 / 4)               # 2 TP, 0 FP at 0.5
    assert d.net_benefit_all[0] == pytest.approx(0.5 - 0.5 * 1.0)
    assert d.net_benefit_none[0] == 0


def test_calibration_bins_counts():
    rng = np.random.RandomState(0)
    p = rng.rand(100)
    y = (rng.rand(100) < p).astype(int)
    t = calibration_bins(y, p, 5)
    assert t.n.sum() == 100 and t.events.sum() == y.sum()


def test_delong_matches_auc_and_is_null_for_identical_scores():
    rng = np.random.RandomState(2)
    y = np.r_[np.ones(20), np.zeros(60)].astype(int)
    p1 = rng.rand(80) + 0.5 * y
    r = delong_test(y, p1, p1)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, p1)) and r["p"] == pytest.approx(1.0)
    p2 = rng.rand(80) + 1.5 * y
    r2 = delong_test(y, p1, p2)
    assert r2["auc2"] > r2["auc1"] and r2["p"] < 0.05


def test_holm_is_monotone_and_bounded():
    raw = np.array([0.01, 0.04, 0.03, 0.5])
    adj = holm(raw)
    assert np.all(adj >= raw) and adj.max() <= 1
    assert adj[0] == pytest.approx(0.04) and adj[2] == pytest.approx(0.09) and adj[1] == pytest.approx(0.09)

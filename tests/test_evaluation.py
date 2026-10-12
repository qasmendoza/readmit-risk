"""Metrics, DeLong test, fairness audit, and decision analysis."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from readmit.evaluation.decision import capacity_threshold, decision_curve, operating_point
from readmit.evaluation.delong import delong_test
from readmit.evaluation.fairness import fairness_report, max_auroc_gap, subgroup_audit
from readmit.evaluation.metrics import (
    calibration_intercept_slope,
    classification_metrics,
    cluster_bootstrap_ci,
    expected_calibration_error,
)


@pytest.fixture(scope="module")
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 4000)
    y = (rng.random(4000) < p).astype(int)
    return y, p


def test_well_calibrated_predictions(calibrated):
    y, p = calibrated
    intercept, slope = calibration_intercept_slope(y, p)
    assert abs(intercept) < 0.15 and 0.85 < slope < 1.15
    assert expected_calibration_error(y, p) < 0.03
    metrics = classification_metrics(y, p)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert metrics["hosmer_lemeshow_p"] > 0.001


def test_overconfident_predictions_have_small_slope(calibrated):
    y, p = calibrated
    logit = np.log(p / (1 - p))
    overconfident = 1 / (1 + np.exp(-3 * logit))
    assert calibration_intercept_slope(y, overconfident)[1] < 0.5


def test_cluster_bootstrap(calibrated):
    y, p = calibrated
    clusters = np.repeat(np.arange(1000), 4)
    ci = cluster_bootstrap_ci(y, p, clusters, "roc_auc", n_boot=200, seed=1)
    assert ci["low"] < ci["estimate"] < ci["high"]
    assert ci["n_boot"] > 150


def test_delong():
    rng = np.random.default_rng(3)
    y = (rng.random(600) < 0.25).astype(int)
    strong = y + rng.normal(0, 0.7, 600)
    weak = y + rng.normal(0, 2.5, 600)
    result = delong_test(y, strong, weak)
    assert result["auc_a"] == pytest.approx(roc_auc_score(y, strong))
    assert result["difference"] > 0 and result["p_value"] < 0.01
    assert delong_test(y, strong, strong)["p_value"] == 1.0
    with pytest.raises(ValueError):
        delong_test(np.zeros(10), strong[:10], weak[:10])


def test_fairness_audit_flags_small_groups(calibrated):
    y, p = calibrated
    groups = np.where(np.arange(len(y)) < 30, "tiny", np.where(np.arange(len(y)) % 2, "a", "b"))
    frame = pd.DataFrame({"y": y, "p": p, "group": groups})
    table, summary = subgroup_audit(frame, "group", threshold=0.4, min_group_size=50)
    assert "insufficient" in table.loc[table["group"] == "tiny", "note"].iloc[0]
    assert summary["groups_compared"] == 2
    assert 0 <= summary["auroc_gap"] < 0.1
    _combined, summaries = fairness_report(frame, ["group", "absent"], 0.4, 50)
    assert len(summaries) == 1 and max_auroc_gap(summaries) == summary["auroc_gap"]


def test_decision_analysis(calibrated):
    y, p = calibrated
    threshold = capacity_threshold(p, 0.1)
    op = operating_point(y, p, threshold)
    assert op["flag_rate"] == pytest.approx(0.1, abs=0.01)
    assert op["tp"] + op["fp"] + op["fn"] + op["tn"] == len(y)
    assert op["ppv"] > y.mean()
    curve = decision_curve(y, p)
    assert (curve["net_benefit_treat_none"] == 0).all()
    first = curve.iloc[0]
    assert first["net_benefit_treat_all"] == pytest.approx(
        y.mean() - (1 - y.mean()) * first["threshold"] / (1 - first["threshold"])
    )
    with pytest.raises(ValueError):
        capacity_threshold(p, 1.5)

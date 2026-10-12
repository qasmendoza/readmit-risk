"""Drift detection and label-delayed performance monitoring."""

import numpy as np
import pandas as pd

from readmit.monitoring.drift import categorical_psi, detect_drift, population_stability_index
from readmit.monitoring.performance import performance_check


def test_psi():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert population_stability_index(reference, rng.normal(0, 1, 5000)) < 0.02
    assert population_stability_index(reference, rng.normal(1.0, 1, 5000)) > 0.2
    assert categorical_psi(pd.Series(list("aabb") * 100), pd.Series(list("aaab") * 100)) > 0.05


def test_detect_drift_flags_shifted_features():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame(
        {
            "ed_visits_6m": rng.poisson(0.5, 3000),
            "risk": rng.beta(2, 10, 3000),
            "sex": rng.choice(["male", "female"], 3000),
        }
    )
    stable = detect_drift(reference, reference.sample(1500, random_state=2))
    assert not stable.drifted
    shifted = reference.assign(ed_visits_6m=rng.poisson(1.8, 3000), risk=rng.beta(4, 8, 3000))
    report = detect_drift(reference, shifted)
    assert report.drifted and {"ed_visits_6m", "risk"} <= set(report.drifted_features)
    assert "DRIFT DETECTED" in report.to_markdown()
    assert report.to_dict()["drift_detected"] is True


def test_performance_check():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.6, 3000)
    y = (rng.random(3000) < p).astype(int)
    healthy = performance_check(y, p, baseline_auroc=0.70)
    assert healthy["evaluated"] and not healthy["retrain_recommended"]
    degraded = performance_check(y, rng.permutation(p), baseline_auroc=0.70)
    assert degraded["retrain_recommended"] and "AUROC" in degraded["reasons"][0]
    assert not performance_check([0, 0], [0.1, 0.2], 0.7)["evaluated"]

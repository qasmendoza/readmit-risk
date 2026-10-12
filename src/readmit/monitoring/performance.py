"""Realized performance once labels mature.

Readmission labels arrive 30 days after discharge, so performance is monitored on a delayed window:
score the stays whose follow-up has completed and compare discrimination and calibration with the
values recorded at training time. A large AUROC drop or a calibration shift recommends
recalibration or retraining (the pipeline's quality gate then decides whether the new model ships).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from readmit.evaluation.metrics import classification_metrics


def performance_check(
    y_true,
    risk,
    baseline_auroc: float,
    max_auroc_drop: float = 0.05,
    max_calibration_shift: float = 0.25,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(risk, dtype=float)
    if len(y) == 0 or y.min() == y.max():
        return {
            "evaluated": False,
            "reason": "need matured labels from both classes",
            "n": len(y),
            "retrain_recommended": False,
        }
    metrics = classification_metrics(y, p)
    auroc_drop = baseline_auroc - metrics["roc_auc"]
    reasons = []
    if auroc_drop > max_auroc_drop:
        reasons.append(f"AUROC fell {auroc_drop:.3f} (> {max_auroc_drop})")
    if abs(metrics["calibration_intercept"]) > max_calibration_shift:
        reasons.append(f"calibration-in-the-large {metrics['calibration_intercept']:+.2f}")
    return {
        "evaluated": True,
        "n": metrics["n"],
        "prevalence": metrics["prevalence"],
        "roc_auc": metrics["roc_auc"],
        "baseline_roc_auc": baseline_auroc,
        "auroc_drop": float(auroc_drop),
        "average_precision": metrics["average_precision"],
        "brier": metrics["brier"],
        "calibration_intercept": metrics["calibration_intercept"],
        "calibration_slope": metrics["calibration_slope"],
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
    }

"""Turning risk scores into care-management decisions.

Care-management programs have finite capacity, so the operating point is defined by capacity
(e.g. "the team can follow up with the top 10% of discharges") rather than by an arbitrary 0.5
cutoff. Thresholds are set on training-period scores and then applied unchanged to the test
period, exactly as a deployed model would be. Decision-curve analysis (Vickers & Elkin, 2006)
reports net benefit across threshold probabilities against treat-all and treat-none strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def capacity_threshold(scores, fraction: float) -> float:
    """Score threshold that flags the top ``fraction`` of a score distribution."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - fraction))


def operating_point(y, p, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(p, dtype=float) >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "threshold": float(threshold),
        "flag_rate": float(flagged.mean()),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": ppv,
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "number_needed_to_evaluate": 1 / ppv if ppv and not np.isnan(ppv) else float("nan"),
        "lift_over_prevalence": ppv / y.mean()
        if y.mean() > 0 and not np.isnan(ppv)
        else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def decision_curve(y, p, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Net benefit of the model, treat-all, and treat-none across threshold probabilities."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.61, 0.02), 2)
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        flagged = p >= t
        tp = np.sum(flagged & (y == 1))
        fp = np.sum(flagged & (y == 0))
        odds = t / (1 - t)
        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(tp / n - fp / n * odds),
                "net_benefit_treat_all": float(prevalence - (1 - prevalence) * odds),
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)

"""Subgroup performance audit (sex, race, ethnicity, age band).

Race and ethnicity are *not* model inputs; they are used only to audit whether the model performs
and allocates care-management slots equitably. Reported per group: AUROC, calibration-in-the-large
(observed/expected ratio), flag rate, sensitivity (equal opportunity), false-positive rate, PPV.
Groups smaller than ``min_group_size`` are reported as insufficient rather than over-interpreted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def _group_row(y: np.ndarray, p: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
    positives = y == 1
    negatives = ~positives
    row: dict[str, Any] = {
        "n": len(y),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "observed_to_expected": float(y.mean() / p.mean()) if p.mean() > 0 else float("nan"),
        "flag_rate": float(flagged.mean()),
        "sensitivity": float(flagged[positives].mean()) if positives.any() else float("nan"),
        "false_positive_rate": float(flagged[negatives].mean())
        if negatives.any()
        else float("nan"),
        "ppv": float(y[flagged].mean()) if flagged.any() else float("nan"),
    }
    row["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    return row


def subgroup_audit(
    frame: pd.DataFrame,
    attribute: str,
    threshold: float,
    min_group_size: int = 50,
    y_col: str = "y",
    p_col: str = "p",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-group metrics and a disparity summary for one attribute."""
    rows = []
    for group, data in frame.groupby(attribute, dropna=False):
        y = data[y_col].to_numpy(dtype=int)
        p = data[p_col].to_numpy(dtype=float)
        row = {"attribute": attribute, "group": str(group)}
        if len(data) < min_group_size:
            row.update(n=len(data), note=f"n < {min_group_size}: insufficient for comparison")
        else:
            row.update(_group_row(y, p, p >= threshold), note="")
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table["note"] == ""]
    summary: dict[str, Any] = {"attribute": attribute, "groups_compared": len(eligible)}
    if len(eligible) >= 2:
        summary.update(
            auroc_gap=float(eligible["roc_auc"].max() - eligible["roc_auc"].min()),
            sensitivity_gap=float(eligible["sensitivity"].max() - eligible["sensitivity"].min()),
            flag_rate_ratio=float(
                eligible["flag_rate"].min() / eligible["flag_rate"].max()
                if eligible["flag_rate"].max() > 0
                else float("nan")
            ),
            oe_ratio_range=[
                float(eligible["observed_to_expected"].min()),
                float(eligible["observed_to_expected"].max()),
            ],
        )
        # Equal opportunity test: are readmitted patients flagged at different rates by group?
        positives = frame[frame[y_col] == 1]
        positives = positives[positives[attribute].astype(str).isin(eligible["group"])]
        contingency = pd.crosstab(positives[attribute].astype(str), positives[p_col] >= threshold)
        if contingency.shape == (len(eligible), 2) and (contingency.to_numpy() > 0).all():
            chi2, p_value, _, _ = stats.chi2_contingency(contingency.to_numpy())
            summary.update(sensitivity_chi2=float(chi2), sensitivity_p_value=float(p_value))
    return table, summary


def fairness_report(
    frame: pd.DataFrame, attributes: list[str], threshold: float, min_group_size: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables, summaries = [], []
    for attribute in attributes:
        if attribute not in frame:
            continue
        table, summary = subgroup_audit(frame, attribute, threshold, min_group_size)
        tables.append(table)
        summaries.append(summary)
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return combined, summaries


def max_auroc_gap(summaries: list[dict[str, Any]]) -> float:
    gaps = [s["auroc_gap"] for s in summaries if "auroc_gap" in s and not np.isnan(s["auroc_gap"])]
    return float(max(gaps)) if gaps else 0.0

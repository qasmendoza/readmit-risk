"""Cohort characteristics by outcome ("Table 1") with hypothesis tests.

Continuous variables: median [IQR] with Mann-Whitney U tests (distributions are skewed).
Binary variables: n (%) with chi-square tests (Fisher's exact test when expected counts < 5).
Categorical variables: chi-square test of independence.
Every row reports a standardized mean difference (SMD); |SMD| > 0.1 is a conventional marker of
a meaningful imbalance. p-values are Holm-adjusted across all rows to control the family-wise
error rate. This is exploratory analysis: associations here are not causal effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from readmit.features import SPEC_BY_NAME

TABLE1_VARIABLES = [
    "age_years",
    "male",
    "los_days",
    "emergency_admission",
    "prior_admissions_12m",
    "ed_visits_6m",
    "charlson_score",
    "chf",
    "chronic_pulmonary",
    "renal_disease",
    "diabetes_complicated",
    "dementia",
    "malignancy",
    "sdoh_count",
    "polypharmacy",
    "last_egfr",
    "last_sodium",
    "lace_score",
]
CATEGORICAL = ["race", "ethnicity"]


def _smd_continuous(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _smd_binary(p1: float, p0: float) -> float:
    pooled = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return float((p1 - p0) / pooled) if pooled > 0 else 0.0


def _median_iqr(values: np.ndarray) -> str:
    q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
    return f"{median:.1f} [{q1:.1f}–{q3:.1f}]"


def table_one(
    frame: pd.DataFrame, outcome: str = "readmitted_30d"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (table, metadata). ``frame`` holds features, demographics, and the outcome."""
    yes = frame[frame[outcome] == 1]
    no = frame[frame[outcome] == 0]
    rows: list[dict[str, Any]] = []
    for name in TABLE1_VARIABLES:
        if name not in frame:
            continue
        label = SPEC_BY_NAME[name].description if name in SPEC_BY_NAME else "LACE index"
        a = yes[name].dropna().to_numpy(dtype=float)
        b = no[name].dropna().to_numpy(dtype=float)
        binary = set(np.unique(frame[name].dropna())) <= {0, 1}
        if binary:
            table = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
            expected = stats.contingency.expected_freq(table)
            if (expected < 5).any():
                _, p_value = stats.fisher_exact(table)
                test = "Fisher"
            else:
                _, p_value, _, _ = stats.chi2_contingency(table)
                test = "chi-square"
            rows.append(
                {
                    "variable": label,
                    "readmitted": f"{int(a.sum()):,} ({a.mean() * 100:.1f}%)",
                    "not_readmitted": f"{int(b.sum()):,} ({b.mean() * 100:.1f}%)",
                    "test": test,
                    "p_value": float(p_value),
                    "smd": _smd_binary(a.mean(), b.mean()),
                    "missing_pct": float(frame[name].isna().mean() * 100),
                }
            )
        else:
            _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "variable": label,
                    "readmitted": _median_iqr(a),
                    "not_readmitted": _median_iqr(b),
                    "test": "Mann-Whitney U",
                    "p_value": float(p_value),
                    "smd": _smd_continuous(a, b),
                    "missing_pct": float(frame[name].isna().mean() * 100),
                }
            )
    for name in CATEGORICAL:
        if name not in frame:
            continue
        contingency = pd.crosstab(frame[name], frame[outcome])
        _, p_value, _, _ = stats.chi2_contingency(contingency.to_numpy())
        shares = contingency.div(contingency.sum(axis=0), axis=1)
        smd = (
            max(
                abs(_smd_binary(shares.loc[level, 1], shares.loc[level, 0]))
                for level in shares.index
            )
            if {0, 1} <= set(shares.columns)
            else 0.0
        )
        rows.append(
            {
                "variable": f"{name.title()} (max level SMD)",
                "readmitted": f"{len(contingency)} levels",
                "not_readmitted": "",
                "test": "chi-square",
                "p_value": float(p_value),
                "smd": float(smd),
                "missing_pct": 0.0,
            }
        )
    table = pd.DataFrame(rows)
    _, adjusted, _, _ = multipletests(table["p_value"], method="holm")
    table["p_holm"] = adjusted
    meta = {
        "n_readmitted": len(yes),
        "n_not_readmitted": len(no),
        "imbalanced_variables": table.loc[table["smd"].abs() > 0.1, "variable"].tolist(),
    }
    return table, meta

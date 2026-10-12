"""Model explanations.

* Global: permutation importance on the held-out test set (drop in AUPRC when a feature is
  shuffled), which works for any model and reflects out-of-sample reliance.
* Local: for one patient, each feature is replaced by its typical (training median) value and the
  change in predicted risk is recorded. Features whose actual value raises risk the most are the
  patient's "key drivers". This occlusion approach is model-agnostic and fast (one batched
  prediction), at the cost of ignoring feature interactions; it is labelled as approximate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from readmit.features import FEATURE_NAMES, SPEC_BY_NAME


def format_value(name: str, value: float) -> str:
    spec = SPEC_BY_NAME.get(name)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not recorded"
    if spec is None:
        return f"{value:.2f}"
    if spec.kind == "binary":
        return "yes" if value >= 0.5 else "no"
    if spec.kind in ("count", "score"):
        return f"{round(value)}"
    return f"{value:.1f} {spec.unit}".strip()


def permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X[columns],
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    table = table[table["feature"].isin(FEATURE_NAMES)]
    table["description"] = table["feature"].map(lambda f: SPEC_BY_NAME[f].description)
    return table.sort_values("importance", ascending=False, ignore_index=True)


def local_explanation(
    predict: Any,
    row: pd.DataFrame,
    reference: dict[str, float],
    top_k: int = 5,
    min_contribution: float = 0.005,
) -> list[dict[str, Any]]:
    """Top features pushing this patient's risk above a typical patient's."""
    if len(row) != 1:
        raise ValueError("expected exactly one row")
    base = float(predict(row)[0])
    names = [
        f
        for f in FEATURE_NAMES
        if pd.notna(row.iloc[0][f]) and not np.isclose(row.iloc[0][f], reference[f])
    ]
    if not names:
        return []
    variants = pd.concat([row] * len(names), ignore_index=True)
    for i, name in enumerate(names):
        variants.loc[i, name] = reference[name]
    contributions = base - np.asarray(predict(variants), dtype=float)
    factors = []
    for name, delta in sorted(zip(names, contributions, strict=True), key=lambda t: -t[1]):
        if delta < min_contribution or len(factors) >= top_k:
            break
        value = float(row.iloc[0][name])
        factors.append(
            {
                "feature": name,
                "description": SPEC_BY_NAME[name].description,
                "value": value,
                "typical": float(reference[name]),
                "contribution": round(float(delta), 4),
                "text": f"{SPEC_BY_NAME[name].description}: {format_value(name, value)} "
                f"(typical {format_value(name, reference[name])})",
            }
        )
    return factors

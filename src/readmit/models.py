"""Candidate models, leakage-safe splitting, tuning, and probability calibration.

* ``lace``: the LACE index, a validated clinical readmission score, mapped to probabilities with a
  one-feature logistic model. Any ML model must beat it to be worth deploying.
* ``logistic``: L2-regularized logistic regression with median imputation + missingness
  indicators and standardization. Interpretable and a strong tabular baseline.
* ``hist_gbm``: histogram gradient boosting (native missing-value handling) with optional
  *monotonic constraints*, so predicted risk can never fall as prior admissions, ED visits,
  comorbidity burden, or length of stay rise. That keeps the model clinically sensible and robust.

Splitting: train on earlier discharges and test on the most recent ones (temporal validation),
with a 30-day embargo before the cutoff so no training label depends on test-period events.
Hyperparameters are tuned with patient-grouped K-fold CV so one patient's stays never appear on
both sides of a fold. Final models are calibrated with grouped cross-validated Platt scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from readmit.features import FEATURE_NAMES, LACE_COLUMN, SPEC_BY_NAME

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier_score": "neg_brier_score",
}
INPUT_COLUMNS = [*FEATURE_NAMES, LACE_COLUMN]


@dataclass
class TemporalSplit:
    train_index: np.ndarray
    test_index: np.ndarray
    cutoff: pd.Timestamp
    embargo_days: int
    embargoed: int


def temporal_split(discharges: pd.Series, test_fraction: float, embargo_days: int) -> TemporalSplit:
    """Hold out the most recent discharges; drop training stays inside the embargo window."""
    discharges = pd.Series(pd.to_datetime(discharges)).reset_index(drop=True)
    cutoff = discharges.quantile(1 - test_fraction)
    test = discharges >= cutoff
    train = discharges < cutoff - pd.Timedelta(days=embargo_days)
    return TemporalSplit(
        train_index=np.flatnonzero(train.to_numpy()),
        test_index=np.flatnonzero(test.to_numpy()),
        cutoff=pd.Timestamp(cutoff),
        embargo_days=embargo_days,
        embargoed=int((~train & ~test).sum()),
    )


def make_pipeline(name: str, monotonic: bool = True, seed: int = 42) -> tuple[Pipeline, dict]:
    if name == "lace":
        pipe = Pipeline(
            [
                ("select", ColumnTransformer([("lace", "passthrough", [LACE_COLUMN])])),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        return pipe, {}
    if name == "logistic":
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        pipe = Pipeline(
            [
                ("prep", ColumnTransformer([("num", numeric, FEATURE_NAMES)])),
                ("clf", LogisticRegression(max_iter=3000)),
            ]
        )
        return pipe, {"clf__C": [0.03, 0.3, 3.0]}
    if name == "hist_gbm":
        constraints = [SPEC_BY_NAME[f].monotonic for f in FEATURE_NAMES] if monotonic else None
        pipe = Pipeline(
            [
                ("select", ColumnTransformer([("num", "passthrough", FEATURE_NAMES)])),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        l2_regularization=1.0,
                        monotonic_cst=constraints,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return pipe, {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]}
    raise ValueError(f"unknown model '{name}'")


@dataclass
class CandidateResult:
    name: str
    best_params: dict[str, Any]
    cv: dict[str, dict[str, float]]
    model: Any = None
    test_proba: np.ndarray | None = None
    train_proba: np.ndarray | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)


def grouped_splits(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int) -> list:
    return list(GroupKFold(n_splits=folds).split(X, y, groups))


def tune_candidate(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    selection_metric: str,
    monotonic: bool = True,
    seed: int = 42,
) -> CandidateResult:
    """Grid search with patient-grouped CV; returns best params and CV score summaries."""
    pipe, grid = make_pipeline(name, monotonic, seed)
    splits = grouped_splits(X, y, groups, folds)
    if grid:
        search = GridSearchCV(pipe, grid, scoring=SCORING, refit=False, cv=splits, n_jobs=1)
        search.fit(X[INPUT_COLUMNS], y)
        results = search.cv_results_
        best = int(np.argmax(results[f"mean_test_{selection_metric}"]))
        params = results["params"][best]
        cv = {
            metric: {
                "mean": float(results[f"mean_test_{metric}"][best]),
                "std": float(results[f"std_test_{metric}"][best]),
            }
            for metric in SCORING
        }
    else:
        scores = cross_validate(pipe, X[INPUT_COLUMNS], y, cv=splits, scoring=SCORING)
        params = {}
        cv = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in SCORING
        }
    return CandidateResult(name=name, best_params=params, cv=cv)


def fit_calibrated(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    method: str,
    monotonic: bool = True,
    seed: int = 42,
) -> CalibratedClassifierCV:
    """Fit the tuned pipeline with grouped cross-validated probability calibration."""
    pipe, _ = make_pipeline(name, monotonic, seed)
    pipe.set_params(**params)
    model = CalibratedClassifierCV(
        pipe, method=method, cv=grouped_splits(X, y, groups, folds), ensemble=True
    )
    return model.fit(X[INPUT_COLUMNS], y)

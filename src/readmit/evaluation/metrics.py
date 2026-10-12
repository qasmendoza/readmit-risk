"""Discrimination and calibration metrics for clinical risk models.

Follows TRIPOD-style reporting: AUROC and AUPRC (discrimination), Brier score, calibration
intercept and slope (logistic recalibration), expected calibration error, and the
Hosmer-Lemeshow test (reported with the usual caveat that it is sensitive to sample size).
Confidence intervals use a *patient-level cluster bootstrap*, because one patient can contribute
several hospitalizations and episodes from the same patient are not independent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit ``logit P(y=1) = a + b * logit(p)``; ideal calibration is a = 0, b = 1."""
    x = _logit(np.asarray(p, dtype=float)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    slope = float(model.coef_[0][0])
    # Calibration-in-the-large: intercept with the slope fixed at 1 (offset logistic model).
    offset = x.ravel()
    intercept = 0.0
    for _ in range(50):  # Newton-Raphson for a single parameter
        mu = 1 / (1 + np.exp(-(intercept + offset)))
        gradient = np.sum(y - mu)
        hessian = np.sum(mu * (1 - mu))
        step = gradient / max(hessian, _EPS)
        intercept += step
        if abs(step) < 1e-8:
            break
    return float(intercept), slope


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-frequency bins."""
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    total = len(p)
    return float(sum(len(g) / total * abs(y[g].mean() - p[g].mean()) for g in groups if len(g)))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> tuple[float, float]:
    order = np.argsort(p)
    statistic = 0.0
    for g in np.array_split(order, groups):
        observed, expected, n = y[g].sum(), p[g].sum(), len(g)
        if n == 0:
            continue
        mean_p = expected / n
        variance = n * mean_p * (1 - mean_p)
        if variance > 0:
            statistic += (observed - expected) ** 2 / variance
    return float(statistic), float(stats.chi2.sf(statistic, groups - 2))


def calibration_curve_points(
    y: np.ndarray, p: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    order = np.argsort(p)
    return [
        {
            "mean_predicted": float(p[g].mean()),
            "observed_rate": float(y[g].mean()),
            "n": len(g),
        }
        for g in np.array_split(order, bins)
        if len(g)
    ]


def classification_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    intercept, slope = calibration_intercept_slope(y, p)
    hl_stat, hl_p = hosmer_lemeshow(y, p)
    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, _EPS, 1 - _EPS))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": expected_calibration_error(y, p),
        "hosmer_lemeshow_stat": hl_stat,
        "hosmer_lemeshow_p": hl_p,
    }


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "brier": brier_score_loss,
}


def cluster_bootstrap_ci(
    y,
    p,
    clusters,
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "roc_auc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CI from resampling clusters (patients) with replacement."""
    fn = METRIC_FUNCTIONS[metric] if isinstance(metric, str) else metric
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    clusters = np.asarray(clusters)
    _, codes = np.unique(clusters, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[c] for c in chosen])
        if y[idx].min() == y[idx].max():
            continue
        estimates.append(fn(y[idx], p[idx]))
    values = np.asarray(estimates)
    return {
        "estimate": float(fn(y, p)),
        "low": float(np.quantile(values, alpha / 2)),
        "high": float(np.quantile(values, 1 - alpha / 2)),
        "n_boot": len(values),
    }

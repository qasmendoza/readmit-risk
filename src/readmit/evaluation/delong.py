"""DeLong's test for two correlated ROC curves (DeLong, DeLong & Clarke-Pearson, 1988).

Implements the fast midrank algorithm of Sun & Xu (2014). Used to test whether the selected model
discriminates significantly better than the LACE clinical baseline when both are scored on the
same held-out hospitalizations (paired comparison).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _structural_components(scores: np.ndarray, n_pos: int) -> tuple[float, np.ndarray, np.ndarray]:
    positives, negatives = scores[:n_pos], scores[n_pos:]
    n_neg = len(negatives)
    rank_all = stats.rankdata(scores)
    rank_pos = stats.rankdata(positives)
    rank_neg = stats.rankdata(negatives)
    auc = (rank_all[:n_pos].sum() / n_pos - (n_pos + 1) / 2) / n_neg
    v10 = (rank_all[:n_pos] - rank_pos) / n_neg  # placement values of positives
    v01 = 1.0 - (rank_all[n_pos:] - rank_neg) / n_pos  # placement values of negatives
    return float(auc), v10, v01


def delong_test(y_true, scores_a, scores_b) -> dict[str, float]:
    """Two-sided test of H0: AUC_a == AUC_b on the same cases."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("inputs must have equal length")
    if y.min() == y.max():
        raise ValueError("both classes are required")
    order = np.argsort(-y, kind="stable")  # positives first
    y, a, b = y[order], a[order], b[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    auc_a, v10_a, v01_a = _structural_components(a, n_pos)
    auc_b, v10_b, v01_b = _structural_components(b, n_pos)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / n_pos + s01 / n_neg
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    difference = auc_a - auc_b
    if variance <= 1e-15:
        z, p_value = 0.0, 1.0
    else:
        z = difference / np.sqrt(variance)
        p_value = float(2 * stats.norm.sf(abs(z)))
    half_width = stats.norm.ppf(0.975) * np.sqrt(max(variance, 0.0))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "difference": float(difference),
        "ci_low": float(difference - half_width),
        "ci_high": float(difference + half_width),
        "z": float(z),
        "p_value": float(p_value),
        "se_a": float(np.sqrt(cov[0, 0])),
        "se_b": float(np.sqrt(cov[1, 1])),
    }

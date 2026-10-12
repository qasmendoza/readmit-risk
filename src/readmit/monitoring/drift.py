"""Data and prediction drift between the training reference profile and current traffic.

Decisions use effect sizes, because with thousands of rows KS and chi-square p-values flag even
trivial shifts. Two complementary measures are used:

* Population Stability Index (PSI): < 0.1 stable, 0.1-0.2 moderate, >= 0.2 significant. These
  conventions come from credit scoring on continuous scores.
* Standardized mean difference (SMD): |SMD| >= 0.2 significant, >= 0.1 moderate.

Why both: PSI's quantile bins collapse on zero-inflated counts. In this project's drift simulation,
ED visits per stay rose from 0.57 to 1.01 (+77%) while PSI was only 0.08; the SMD (about 0.4)
caught it. A feature is flagged when either measure crosses its threshold. The predicted-risk
distribution is monitored too: it moves as soon as inputs move, long before labels arrive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from readmit.features import FEATURE_NAMES

DEFAULT_NUMERIC = (*FEATURE_NAMES, "risk")
DEFAULT_CATEGORICAL = ("sex", "age_band")
_EPS = 1e-6


def population_stability_index(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))[1:-1]
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    p, q = np.clip(p, _EPS, None), np.clip(q, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    p = reference.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    q = current.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    p, q = np.clip(p, _EPS, None), np.clip(q, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def standardized_mean_difference(reference, current) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    pooled = np.sqrt((np.nanvar(ref, ddof=1) + np.nanvar(cur, ddof=1)) / 2)
    return float((np.nanmean(cur) - np.nanmean(ref)) / pooled) if pooled > 0 else 0.0


def severity(psi: float, smd: float = 0.0) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi >= 0.2 or abs(smd) >= 0.2:
        return "significant"
    if psi >= 0.1 or abs(smd) >= 0.1:
        return "moderate"
    return "stable"


@dataclass
class FeatureDrift:
    feature: str
    kind: str
    psi: float
    severity: str
    test: str
    statistic: float
    p_value: float
    reference_mean: float | None = None
    current_mean: float | None = None
    smd: float | None = None


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    psi_threshold: float
    n_reference: int
    n_current: int
    smd_threshold: float = 0.2
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def drifted_features(self) -> list[str]:
        return [
            f.feature
            for f in self.features
            if (not np.isnan(f.psi) and f.psi >= self.psi_threshold)
            or (f.smd is not None and abs(f.smd) >= self.smd_threshold)
        ]

    @property
    def drifted(self) -> bool:
        return bool(self.drifted_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "drift_detected": self.drifted,
            "drifted_features": self.drifted_features,
            "psi_threshold": self.psi_threshold,
            "smd_threshold": self.smd_threshold,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "features": [asdict(f) for f in self.features],
        }

    def to_markdown(self) -> str:
        status = "DRIFT DETECTED" if self.drifted else "No significant drift"
        lines = [
            "# Drift report",
            "",
            f"**{status}** (thresholds: PSI {self.psi_threshold}, |SMD| {self.smd_threshold}) · "
            f"reference n={self.n_reference:,} · current n={self.n_current:,} · {self.created_at}",
            "",
            "| Feature | PSI | SMD | Severity | Test | Statistic | p | Ref. mean | Cur. mean |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for f in sorted(self.features, key=lambda x: -np.nan_to_num(x.psi)):
            ref = "–" if f.reference_mean is None else f"{f.reference_mean:,.3f}"
            cur = "–" if f.current_mean is None else f"{f.current_mean:,.3f}"
            smd = "–" if f.smd is None else f"{f.smd:+.2f}"
            lines.append(
                f"| {f.feature} | {f.psi:.3f} | {smd} | {f.severity} | {f.test} | "
                f"{f.statistic:.3f} | {f.p_value:.2g} | {ref} | {cur} |"
            )
        lines += [
            "",
            "PSI < 0.1 stable · 0.1–0.2 moderate · ≥ 0.2 significant; |SMD| uses 0.1 / 0.2.",
        ]
        return "\n".join(lines)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: tuple[str, ...] = DEFAULT_NUMERIC,
    categorical: tuple[str, ...] = DEFAULT_CATEGORICAL,
    psi_threshold: float = 0.2,
    smd_threshold: float = 0.2,
) -> DriftReport:
    results: list[FeatureDrift] = []
    for name in numeric:
        if name not in reference or name not in current:
            continue
        ref = reference[name].dropna().to_numpy(dtype=float)
        cur = current[name].dropna().to_numpy(dtype=float)
        if len(ref) == 0 or len(cur) == 0:
            continue
        ks = stats.ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        smd = standardized_mean_difference(ref, cur)
        results.append(
            FeatureDrift(
                name,
                "numeric",
                psi,
                severity(psi, smd),
                "Kolmogorov-Smirnov",
                float(ks.statistic),
                float(ks.pvalue),
                float(ref.mean()),
                float(cur.mean()),
                smd,
            )
        )
    for name in categorical:
        if name not in reference or name not in current:
            continue
        levels = sorted(
            set(reference[name].dropna().astype(str)) | set(current[name].dropna().astype(str))
        )
        table = np.vstack(
            [
                reference[name].astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(),
                current[name].astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(),
            ]
        )
        table = table[:, table.sum(axis=0) > 0]
        chi2, p_value = stats.chi2_contingency(table)[:2] if table.shape[1] > 1 else (0.0, 1.0)
        psi = categorical_psi(reference[name], current[name])
        results.append(
            FeatureDrift(
                name, "categorical", psi, severity(psi), "chi-square", float(chi2), float(p_value)
            )
        )
    return DriftReport(results, psi_threshold, len(reference), len(current), smd_threshold)

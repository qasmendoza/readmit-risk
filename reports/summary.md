**Data:** synthetic FHIR R4 (readmit generator) · 6,570 index hospitalizations · 30-day readmission rate 14.6% · model `20261011.211408-294a166`

| Held-out metric (95% CI) | Selected: `logistic` | Baseline: `lace` |
|---|---|---|
| AUROC | 0.803 (0.758–0.839) | 0.752 |
| AUPRC | 0.446 (0.325–0.543) | 0.340 |
| Brier score | 0.099 (0.084–0.114) | 0.107 |
| Calibration slope | 1.05 | 0.99 |

AUROC vs. LACE: +0.051 (95% CI +0.024 to +0.078, DeLong p < 0.001). Flagging the top 10% of discharges (care-management capacity) catches **34.6%** of readmissions with PPV 52.5% (3.7× the base rate).

> Synthetic data with a known, relatively clean mechanism: these numbers demonstrate the pipeline, not clinical performance. Published readmission models typically reach C-statistics of about 0.6-0.7 on real data.

Details: [evaluation report](reports/REPORT.md) · [model card](reports/MODEL_CARD.md) · [figures](reports/figures)
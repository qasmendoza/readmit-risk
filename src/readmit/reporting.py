"""Evaluation report, model card, figures, and README result injection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from readmit.pipeline import PipelineResult

logger = logging.getLogger(__name__)
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def md_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _p(value: float) -> str:
    if value is None or np.isnan(value):
        return "–"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _ci(ci: dict[str, float], digits: int = 3) -> str:
    return f"{ci['estimate']:.{digits}f} ({ci['low']:.{digits}f}–{ci['high']:.{digits}f})"


def _pct(value: float) -> str:
    return "–" if value is None or np.isnan(value) else f"{value * 100:.1f}%"


def build_summary(result: PipelineResult) -> str:
    cfg = result.config
    ci = result.confidence_intervals
    op = result.operating_point
    baseline = cfg.gate.baseline
    base = result.candidates.get(baseline)

    def base_metric(name: str, digits: int = 3) -> str:
        return f"{base.test_metrics[name]:.{digits}f}" if base is not None else "–"

    lines = [
        f"**Data:** {result.data_source} · {len(result.dataset):,} index hospitalizations · "
        f"30-day readmission rate {result.cohort.readmission_rate:.1%} · model "
        f"`{result.bundle.version}`",
        "",
        md_table(
            [
                ["AUROC", _ci(ci["roc_auc"]), base_metric("roc_auc")],
                ["AUPRC", _ci(ci["average_precision"]), base_metric("average_precision")],
                ["Brier score", _ci(ci["brier"]), base_metric("brier")],
                [
                    "Calibration slope",
                    f"{result.test_metrics['calibration_slope']:.2f}",
                    base_metric("calibration_slope", 2),
                ],
            ],
            [
                "Held-out metric (95% CI)",
                f"Selected: `{result.best_model}`",
                f"Baseline: `{baseline}`",
            ],
        ),
        "",
    ]
    if baseline in result.comparisons:
        comparison = result.comparisons[baseline]
        lines += [
            f"AUROC vs. LACE: {comparison['difference']:+.3f} (95% CI "
            f"{comparison['ci_low']:+.3f} to {comparison['ci_high']:+.3f}, DeLong p "
            f"{_p(comparison['p_value'])}). Flagging the top "
            f"{cfg.evaluation.capacity_fraction:.0%} of discharges (care-management capacity) "
            f"catches **{_pct(op['sensitivity'])}** of readmissions with PPV {_pct(op['ppv'])} "
            f"({op['lift_over_prevalence']:.1f}× the base rate).",
            "",
        ]
    lines += [
        "> Synthetic data with a known, relatively clean mechanism: these numbers demonstrate the "
        "pipeline, not clinical performance. Published readmission models typically reach "
        "C-statistics of about 0.6-0.7 on real data.",
        "",
        "Details: [evaluation report](reports/REPORT.md) · [model card](reports/MODEL_CARD.md) · "
        "[figures](reports/figures)",
    ]
    return "\n".join(lines)


def build_report(result: PipelineResult, figures: list[str]) -> str:
    cfg = result.config
    meta = result.bundle.metadata
    split = meta["split"]
    t1 = result.table1
    out = [
        "# Readmission risk model: evaluation report",
        "",
        f"Model version `{result.bundle.version}` · data source **{result.data_source}** · "
        f"generated {meta['created_at']}",
        "",
        "## 1. Cohort",
        "",
        md_table(
            [
                [f["step"], f"{f['n']:,}" + (f" ({f['positives']:,})" if "positives" in f else "")]
                for f in result.cohort.flow
            ],
            ["Step", "n"],
        ),
        "",
        f"30-day readmission rate: **{result.cohort.readmission_rate:.1%}**.",
        "",
        "## 2. Cohort characteristics (Table 1)",
        "",
        "One randomly selected index hospitalization per patient, because stays from the same "
        "patient are not independent and pooling them inflates significance. "
        f"Readmitted n = {result.table1_meta['n_readmitted']:,}; not readmitted n = "
        f"{result.table1_meta['n_not_readmitted']:,}. Continuous variables: median [IQR]; binary: "
        "n (%). p-values are Holm-adjusted; |SMD| > 0.1 indicates meaningful imbalance.",
        "",
        md_table(
            [
                [r.variable, r.readmitted, r.not_readmitted, r.test, _p(r.p_holm), f"{r.smd:+.2f}"]
                for r in t1.itertuples()
            ],
            ["Variable", "Readmitted", "Not readmitted", "Test", "p (Holm)", "SMD"],
        ),
        "",
        "## 3. Model development",
        "",
        f"Temporal split at {split['cutoff'][:10]}: {split['train']:,} training and "
        f"{split['test']:,} test hospitalizations; {split['embargoed']:,} stays in the "
        f"{split['embargo_days']}-day embargo before the cutoff were excluded so no training label "
        f"depends on test-period events. Hyperparameters were tuned with {cfg.split.cv_folds}-fold "
        "patient-grouped cross-validation; final models use grouped cross-validated "
        f"{cfg.models.calibration} calibration.",
        "",
        md_table(
            [
                [
                    f"**{n}**" if n == result.best_model else n,
                    f"{r.cv['average_precision']['mean']:.3f} ± {r.cv['average_precision']['std']:.3f}",
                    f"{r.cv['roc_auc']['mean']:.3f} ± {r.cv['roc_auc']['std']:.3f}",
                    f"{-r.cv['neg_brier_score']['mean']:.4f}",
                    ", ".join(f"{k.split('__')[-1]}={v}" for k, v in r.best_params.items()) or "–",
                ]
                for n, r in result.candidates.items()
            ],
            ["Model", "CV AUPRC", "CV AUROC", "CV Brier", "Selected hyperparameters"],
        ),
        "",
        f"Selected **{result.best_model}** by cross-validated {cfg.models.selection_metric}.",
        "",
        "## 4. Held-out performance (most recent discharges)",
        "",
        md_table(
            [
                [
                    f"**{n}**" if n == result.best_model else n,
                    f"{m['roc_auc']:.3f}",
                    f"{m['average_precision']:.3f}",
                    f"{m['brier']:.4f}",
                    f"{m['calibration_slope']:.2f}",
                    f"{m['calibration_intercept']:+.2f}",
                    f"{m['ece']:.3f}",
                ]
                for n, m in ((n, r.test_metrics) for n, r in result.candidates.items())
            ],
            ["Model", "AUROC", "AUPRC", "Brier", "Cal. slope", "Cal. intercept", "ECE"],
        ),
        "",
        f"Selected model with 95% patient-level cluster-bootstrap CIs "
        f"({cfg.evaluation.bootstrap_iterations:,} resamples): AUROC "
        f"{_ci(result.confidence_intervals['roc_auc'])}, AUPRC "
        f"{_ci(result.confidence_intervals['average_precision'])}, Brier "
        f"{_ci(result.confidence_intervals['brier'], 4)}. Test prevalence "
        f"{split['test_prevalence']:.1%}.",
        "",
        "Paired DeLong tests (selected model minus comparator, same test hospitalizations):",
        "",
        md_table(
            [
                [
                    other,
                    f"{c['difference']:+.3f}",
                    f"{c['ci_low']:+.3f} to {c['ci_high']:+.3f}",
                    _p(c["p_value"]),
                ]
                for other, c in result.comparisons.items()
            ],
            ["Comparator", "ΔAUROC", "95% CI", "p"],
        ),
        "",
        "## 5. Calibration",
        "",
        f"Calibration slope {result.test_metrics['calibration_slope']:.2f} (ideal 1), "
        f"calibration-in-the-large {result.test_metrics['calibration_intercept']:+.2f} (ideal 0), "
        f"ECE {result.test_metrics['ece']:.3f}. Hosmer–Lemeshow χ² = "
        f"{result.test_metrics['hosmer_lemeshow_stat']:.1f}, p = "
        f"{_p(result.test_metrics['hosmer_lemeshow_p'])} (sensitive to sample size; shown for "
        "completeness, the slope and intercept are the primary calibration measures).",
        "",
        "## 6. Decision analysis",
        "",
    ]
    op = result.operating_point
    out += [
        f"Threshold set on training-period scores to flag the top "
        f"{cfg.evaluation.capacity_fraction:.0%} (risk ≥ {op['threshold']:.3f}), then applied "
        "unchanged to the test period:",
        "",
        md_table(
            [
                [
                    _pct(op["flag_rate"]),
                    _pct(op["sensitivity"]),
                    _pct(op["specificity"]),
                    _pct(op["ppv"]),
                    f"{op['number_needed_to_evaluate']:.1f}",
                    f"{op['lift_over_prevalence']:.2f}×",
                ]
            ],
            [
                "Flag rate",
                "Sensitivity",
                "Specificity",
                "PPV",
                "Patients per readmission found",
                "Lift",
            ],
        ),
        "",
    ]
    curve = result.decision_curve
    useful = curve[
        (curve["net_benefit_model"] > curve["net_benefit_treat_all"])
        & (curve["net_benefit_model"] > 0)
    ]
    if len(useful):
        out.append(
            f"Decision curve: the model offers higher net benefit than treating everyone or "
            f"no one for threshold probabilities between {useful['threshold'].min():.2f} and "
            f"{useful['threshold'].max():.2f}."
        )
    out += [
        "",
        "## 7. Fairness audit",
        "",
        "Race and ethnicity are not model inputs; they are used only for this audit. Groups "
        f"smaller than {cfg.evaluation.fairness_min_group_size} are not compared.",
        "",
    ]
    ft = result.fairness_table
    if len(ft):
        out.append(
            md_table(
                [
                    [
                        r.attribute,
                        r.group,
                        f"{int(r.n):,}",
                        "–" if r.note else f"{r.roc_auc:.3f}",
                        "–" if r.note else f"{r.observed_to_expected:.2f}",
                        "–" if r.note else _pct(r.flag_rate),
                        "–" if r.note else _pct(r.sensitivity),
                        r.note or "",
                    ]
                    for r in ft.itertuples()
                ],
                ["Attribute", "Group", "n", "AUROC", "O/E", "Flag rate", "Sensitivity", "Note"],
            )
        )
    for summary in result.fairness_summary:
        if "auroc_gap" in summary:
            extra = (
                f", equal-opportunity χ² p = {_p(summary['sensitivity_p_value'])}"
                if "sensitivity_p_value" in summary
                else ""
            )
            out.append(
                f"\n- **{summary['attribute']}**: AUROC gap {summary['auroc_gap']:.3f}, "
                f"sensitivity gap {summary['sensitivity_gap'] * 100:.1f} pp, flag-rate ratio "
                f"{summary['flag_rate_ratio']:.2f}{extra}"
            )
    out += [
        "",
        "## 8. Explainability (permutation importance on the test set)",
        "",
        md_table(
            [
                [r.description, f"{r.importance:.4f}", f"{r.std:.4f}"]
                for r in result.importance.head(12).itertuples()
            ],
            ["Feature", "Mean drop in AUPRC", "SD"],
        ),
        "",
        "## 9. Quality gate",
        "",
        md_table(
            [
                [c["name"], "✅ pass" if c["passed"] else "❌ fail", c["detail"]]
                for c in result.gate["checks"]
            ],
            ["Check", "Result", "Detail"],
        ),
        "",
        f"Overall: **{'PASSED' if result.gate['passed'] else 'FAILED'}** "
        f"({'enforced' if result.gate['enforced'] else 'not enforced'}).",
        "",
        "## 10. Copilot retrieval evaluation",
        "",
        f"Guidance retrieval hit rate@{result.retrieval_eval['k']}: "
        f"{result.retrieval_eval['hit_rate']:.0%}, MRR {result.retrieval_eval['mrr']:.2f} "
        f"over {result.retrieval_eval['n']} labelled queries.",
        "",
    ]
    if figures:
        out += ["## Figures", ""] + [f"![{Path(f).stem}](figures/{Path(f).name})" for f in figures]
        out.append("")
    out += [
        "## Limitations",
        "",
        "Synthetic data; CMS planned-readmission logic not applied; readmissions to facilities "
        "outside the data source are invisible; a utilization label can encode differences in "
        "access to care. See the model card for intended use.",
        "",
    ]
    return "\n".join(out)


def build_model_card(result: PipelineResult) -> str:
    cfg = result.config
    meta = result.bundle.metadata
    ci = result.confidence_intervals
    op = result.operating_point
    gaps = [
        f"{s['attribute']} AUROC gap {s['auroc_gap']:.3f}"
        for s in result.fairness_summary
        if "auroc_gap" in s
    ]
    return "\n".join(
        [
            "# Model card: 30-day hospital readmission risk",
            "",
            "Format follows Mitchell et al., *Model Cards for Model Reporting* (2019). Numbers are "
            "generated by the training pipeline for this exact model version.",
            "",
            "## Model details",
            "",
            f"- **Version:** `{result.bundle.version}` (created {meta['created_at']})",
            f"- **Type:** {result.best_model} with grouped cross-validated {cfg.models.calibration} "
            "calibration; selected over the LACE index and alternative models by cross-validated "
            f"{cfg.models.selection_metric}",
            f"- **Inputs:** {len(meta['features'])} point-in-time features derived from HL7 FHIR R4 "
            "Patient, Encounter, Condition, Observation, and MedicationRequest resources",
            "- **Output:** probability of an acute readmission within 30 days of discharge, a risk "
            "tier, and the main contributing factors",
            "- **License:** MIT",
            "",
            "## Intended use",
            "",
            "- **Primary use:** prioritizing post-discharge outreach by care managers and transitional "
            "care teams in a value-based care program.",
            "- **Users:** care managers and clinicians, as decision support with human review.",
            "- **Out of scope:** diagnosis or treatment decisions; denying or limiting care, coverage, "
            "or benefits; pediatric patients; any use on real patients before local validation, "
            "governance approval, and prospective monitoring.",
            "",
            "## Training and evaluation data",
            "",
            f"- **Source:** {result.data_source}. No real patient data was used.",
            f"- **Cohort:** {len(result.dataset):,} index hospitalizations (adults, alive at discharge, "
            f"full 30-day follow-up); readmission rate {result.cohort.readmission_rate:.1%}.",
            f"- **Split:** temporal; trained on discharges before {meta['split']['cutoff'][:10]} "
            f"(with a {meta['split']['embargo_days']}-day embargo), tested on later discharges.",
            "",
            "## Performance (held-out, most recent discharges)",
            "",
            f"- AUROC {_ci(ci['roc_auc'])}; AUPRC {_ci(ci['average_precision'])}; Brier "
            f"{_ci(ci['brier'], 4)}",
            f"- Calibration slope {result.test_metrics['calibration_slope']:.2f}, intercept "
            f"{result.test_metrics['calibration_intercept']:+.2f}",
            f"- At {cfg.evaluation.capacity_fraction:.0%} capacity: sensitivity "
            f"{_pct(op['sensitivity'])}, PPV {_pct(op['ppv'])}",
            "",
            "## Fairness",
            "",
            "Race and ethnicity are excluded from model inputs and used only for auditing. "
            + ("Largest observed gaps: " + "; ".join(gaps) + ". " if gaps else "")
            + "See section 7 of the evaluation report for per-group metrics.",
            "",
            "## Ethical considerations",
            "",
            "- **Label choice:** readmission is a utilization outcome, and utilization reflects access "
            "to care as well as need. Obermeyer et al. (Science, 2019) showed how a utilization-type "
            "label (cost) can encode racial bias in a population-health algorithm; subgroup calibration "
            "should be monitored for this reason.",
            "- **Human oversight:** outputs prioritize outreach; they never replace clinical judgment.",
            "- **Privacy:** the copilot receives only minimum-necessary, de-identified context.",
            "",
            "## Caveats and recommendations",
            "",
            "- Trained on synthetic data; performance on real populations is unknown.",
            "- The CMS planned-readmission algorithm is not applied.",
            "- Monitor feature drift and realized performance; recalibrate or retrain when the quality "
            "gate's criteria are no longer met.",
            "",
        ]
    )


def make_figures(result: PipelineResult, directory: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures")
        return []
    directory.mkdir(parents=True, exist_ok=True)
    y = result.dataset.iloc[result.split.test_index]["readmitted_30d"].to_numpy()
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, r in result.candidates.items():
        fpr, tpr, _ = roc_curve(y, r.test_proba)
        precision, recall, _ = precision_recall_curve(y, r.test_proba)
        style = {"linewidth": 2.4 if name == result.best_model else 1.2}
        axes[0].plot(fpr, tpr, label=f"{name} ({r.test_metrics['roc_auc']:.3f})", **style)
        axes[1].plot(
            recall, precision, label=f"{name} ({r.test_metrics['average_precision']:.3f})", **style
        )
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle=":")
    axes[1].axhline(y.mean(), color="grey", linestyle=":", label=f"prevalence ({y.mean():.3f})")
    axes[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(str(directory / "roc_pr.png"))
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    points = result.calibration_points
    upper = max(max(p["mean_predicted"] for p in points), max(p["observed_rate"] for p in points))
    ax.plot([0, upper], [0, upper], color="grey", linestyle=":")
    ax.plot([p["mean_predicted"] for p in points], [p["observed_rate"] for p in points], marker="o")
    ax.set(
        title=f"Calibration ({result.best_model}, test deciles)",
        xlabel="Mean predicted risk",
        ylabel="Observed readmission rate",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(str(directory / "calibration.png"))
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    curve = result.decision_curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["threshold"], curve["net_benefit_model"], label=result.best_model, linewidth=2)
    ax.plot(curve["threshold"], curve["net_benefit_treat_all"], label="Flag everyone")
    ax.plot(curve["threshold"], curve["net_benefit_treat_none"], label="Flag no one", color="grey")
    ax.set_ylim(
        min(-0.02, curve["net_benefit_model"].min()),
        max(curve["net_benefit_model"].max(), y.mean()) * 1.15,
    )
    ax.set(title="Decision curve (test)", xlabel="Threshold probability", ylabel="Net benefit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(str(directory / "decision_curve.png"))
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    top = result.importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top["description"], top["importance"], xerr=top["std"], color="#2a9d8f")
    ax.set(title="Permutation importance (drop in test AUPRC)", xlabel="Importance")
    fig.tight_layout()
    paths.append(str(directory / "importance.png"))
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def write_reports(result: PipelineResult, make_plots: bool = True) -> dict[str, Path]:
    reports = result.config.output.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    figures = make_figures(result, reports / "figures") if make_plots else []
    paths = {
        "report": reports / "REPORT.md",
        "model_card": reports / "MODEL_CARD.md",
        "summary": reports / "summary.md",
    }
    paths["report"].write_text(build_report(result, figures), encoding="utf-8")
    paths["model_card"].write_text(build_model_card(result), encoding="utf-8")
    paths["summary"].write_text(build_summary(result), encoding="utf-8")
    return paths


def update_readme(readme_path: str | Path, summary: str) -> bool:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{summary}\n{README_END}{tail}", encoding="utf-8")
    return True

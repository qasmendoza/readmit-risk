"""End-to-end pipeline behaviour, artifacts, and the deployment-blocking quality gate."""

import json

import pytest

from readmit.bundle import BUNDLE_FILE, REFERENCE_FILE, ModelBundle
from readmit.fhir.synthetic import write_population
from readmit.pipeline import QualityGateError, run_pipeline
from readmit.reporting import README_END, README_START, update_readme
from tests.conftest import make_config


def test_pipeline_outputs(trained):
    cfg = trained.config
    assert trained.gate["passed"]
    assert trained.best_model in cfg.models.candidates
    for name in ("roc_auc", "average_precision", "brier"):
        ci = trained.confidence_intervals[name]
        assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert 0.08 < trained.cohort.readmission_rate < 0.25
    model_dir = cfg.model_dir
    assert (model_dir / BUNDLE_FILE).exists() and (model_dir / REFERENCE_FILE).exists()
    metrics = json.loads((cfg.output.artifacts_dir / "metrics.json").read_text())
    assert metrics["best_model"] == trained.best_model
    report = (cfg.output.reports_dir / "REPORT.md").read_text()
    for section in ("## 1. Cohort", "Table 1", "DeLong", "## 7. Fairness audit", "Quality gate"):
        assert section in report
    card = (cfg.output.reports_dir / "MODEL_CARD.md").read_text()
    assert "Out of scope" in card and trained.bundle.version in card
    assert len(list((cfg.output.reports_dir / "figures").glob("*.png"))) == 4


def test_bundle_roundtrip_and_tiers(trained):
    bundle = ModelBundle.load(trained.config.model_dir)
    assert bundle.version == trained.bundle.version
    assert bundle.thresholds["high"] > bundle.thresholds["moderate"]
    assert bundle.tier(bundle.thresholds["high"]) == "high"
    assert bundle.tier(0.0) == "low"
    scored = bundle.score(
        {
            "age_years": 85,
            "prior_admissions_12m": 4,
            "ed_visits_6m": 4,
            "charlson_score": 6,
            "chf": 1,
            "los_days": 9,
        }
    )
    assert 0 <= scored["risk"] <= 1 and scored["factors"]
    assert scored["factors"] == sorted(scored["factors"], key=lambda f: -f["contribution"])


def test_quality_gate_blocks_deployment(tmp_path):
    write_population(tmp_path / "fhir", 500, seed=13)
    cfg = make_config(
        tmp_path,
        **{
            "gate.min_auroc": 0.99,
            "models.candidates": "[lace, logistic]",
            "evaluation.bootstrap_iterations": 100,
        },
    )
    with pytest.raises(QualityGateError, match=r"required >= 0\.99"):
        run_pipeline(cfg, data_source="test", make_plots=False, track=False)
    assert not (cfg.model_dir / BUNDLE_FILE).exists()  # blocked model never saved
    assert (cfg.output.reports_dir / "REPORT.md").exists()  # but the failure is documented


def test_tiny_cohort_is_rejected(tmp_path):
    write_population(tmp_path / "fhir", 20, seed=1)
    with pytest.raises(ValueError, match="too small"):
        run_pipeline(make_config(tmp_path), track=False, make_plots=False)


def test_readme_injection(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")
    assert update_readme(readme, "NEW RESULTS")
    text = readme.read_text()
    assert "NEW RESULTS" in text and "old" not in text and text.endswith("outro\n")
    plain = tmp_path / "plain.md"
    plain.write_text("no markers")
    assert not update_readme(plain, "x")

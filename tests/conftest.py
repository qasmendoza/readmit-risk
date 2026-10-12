"""Shared fixtures. Expensive objects (synthetic data, a trained pipeline) are session-scoped."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from readmit.config import Config, load_config
from readmit.fhir.flatten import flatten_path
from readmit.fhir.synthetic import write_population

ROOT = Path(__file__).resolve().parents[1]
SYNTHEA_DIR = ROOT / "tests" / "fixtures" / "synthea"


def make_config(base: Path, **overrides: object) -> Config:
    values = [
        f"data.input_dir={base / 'fhir'}",
        f"data.tables_dir={base / 'tables'}",
        f"output.artifacts_dir={base / 'artifacts'}",
        f"output.reports_dir={base / 'reports'}",
        f"copilot.knowledge_dir={ROOT / 'knowledge'}",
        "tracking.enabled=false",
        "evaluation.bootstrap_iterations=200",
        "split.cv_folds=3",
    ]
    values += [f"{k}={v}" for k, v in overrides.items()]
    return load_config(ROOT / "configs" / "config.yaml", values)


@pytest.fixture(scope="session")
def synthea_dir() -> Path:
    return SYNTHEA_DIR


@pytest.fixture(scope="session")
def synthea_tables(synthea_dir):
    return flatten_path(synthea_dir)


@pytest.fixture(scope="session")
def synthea_patient_bundle(synthea_dir) -> dict:
    """Real Synthea record for an adult with an ICU transfer chain."""
    return json.loads((synthea_dir / "synthea_patient_2.json").read_text())


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "fhir"
    write_population(path, 400, seed=7)
    return path


@pytest.fixture(scope="session")
def synthetic_tables(synthetic_dir):
    return flatten_path(synthetic_dir)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One full pipeline run shared by pipeline, API, monitoring, and CLI tests."""
    from readmit.pipeline import run_pipeline

    base = tmp_path_factory.mktemp("trained")
    write_population(base / "fhir", 900, seed=11)
    cfg = make_config(base)
    return run_pipeline(cfg, data_source="synthetic (test)", make_plots=True, track=False)


@pytest.fixture(scope="session")
def synthetic_patient_bundle(tmp_path_factory) -> dict:
    """A synthetic patient (Bundle format) with at least one acute inpatient stay."""
    path = tmp_path_factory.mktemp("bundles")
    write_population(path, 30, seed=5, fmt="bundle")
    for file in sorted(path.glob("patient_*.json")):
        bundle = json.loads(file.read_text())
        kinds = [
            e["resource"].get("class", {}).get("code")
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Encounter"
        ]
        if kinds.count("IMP") >= 2:
            return bundle
    raise RuntimeError("no suitable synthetic patient found")

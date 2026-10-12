"""API behaviour: FHIR in/out, validation, auth, limits, SageMaker contract, metrics, audit."""

import logging

import pytest
from fastapi.testclient import TestClient

from readmit.fhir.risk_assessment import RISK_PROBABILITY_SYSTEM, build_risk_assessment
from readmit.serving.app import create_app


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(model_dir=trained.config.model_dir, api_key="")) as c:
        yield c


def test_health_ready_and_model_info(client, trained):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_version"] == trained.bundle.version
    info = client.get("/v1/model").json()
    assert info["model_name"] == trained.best_model
    assert {f["name"] for f in info["features"]} >= {"charlson_score", "prior_admissions_12m"}


def test_predict_from_features(client):
    response = client.post(
        "/v1/predict",
        json={
            "features": {
                "age_years": 86,
                "prior_admissions_12m": 4,
                "ed_visits_6m": 4,
                "charlson_score": 6,
                "chf": 1,
                "los_days": 10,
                "n_medications_12m": 9,
                "polypharmacy": 1,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk"] <= 1 and body["tier"] in {"low", "moderate", "high"}
    assert client.post("/v1/predict", json={"features": {"not_a_feature": 1}}).status_code == 422


def test_fhir_risk_assessment_from_real_synthea_record(client, synthea_patient_bundle):
    response = client.post("/v1/fhir/risk-assessment", json=synthea_patient_bundle)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/fhir+json")
    resource = response.json()
    assert resource["resourceType"] == "RiskAssessment" and resource["status"] == "final"
    assert resource["subject"]["reference"].startswith("Patient/")
    prediction = resource["prediction"][0]
    assert 0 <= prediction["probabilityDecimal"] <= 1
    assert prediction["qualitativeRisk"]["coding"][0]["system"] == RISK_PROBABILITY_SYSTEM


def test_fhir_errors(client, synthea_patient_bundle):
    assert (
        client.post("/v1/fhir/risk-assessment", json={"resourceType": "Patient"}).status_code == 422
    )
    missing = client.post("/v1/fhir/risk-assessment?encounter_id=nope", json=synthea_patient_bundle)
    assert missing.status_code == 404
    empty = {"resourceType": "Bundle", "entry": []}
    assert client.post("/v1/fhir/risk-assessment", json=empty).status_code == 422


def test_copilot_brief(client, synthetic_patient_bundle):
    response = client.post("/v1/copilot/brief", json=synthetic_patient_bundle)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "template" and body["citations"] and body["guidance"]
    assert body["invalid_citations"] == []
    assert "Decision support only" in body["brief"]


def test_sagemaker_contract(client, synthea_patient_bundle):
    assert client.get("/ping").status_code == 200
    by_features = client.post("/invocations", json={"features": {"age_years": 70}})
    assert by_features.status_code == 200 and "risk" in by_features.json()
    by_bundle = client.post("/invocations", json=synthea_patient_bundle)
    assert by_bundle.json()["resourceType"] == "RiskAssessment"
    assert client.post("/invocations", json={"oops": 1}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/v1/predict", json={"features": {"age_years": 60}})
    text = client.get("/metrics").text
    assert "readmit_predictions_total" in text and "readmit_request_latency_seconds" in text
    assert 'route="/v1/predict"' in text


def test_api_key_and_body_limit(trained):
    app = create_app(model_dir=trained.config.model_dir, api_key="s3cret", max_body_mb=0.001)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # probes stay open
        assert c.get("/v1/model").status_code == 401
        assert c.get("/v1/model", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/model", headers={"X-API-Key": "s3cret"}).status_code == 200
        big = {"features": {"age_years": 70}, "padding": "x" * 5000}
        assert c.post("/v1/predict", json=big, headers={"X-API-Key": "s3cret"}).status_code == 413


def test_not_ready_without_model(tmp_path):
    with TestClient(create_app(model_dir=tmp_path / "missing")) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.get("/ping").status_code == 503


def test_audit_log_pseudonymizes_identifiers(client, synthea_patient_bundle, caplog):
    patient_id = next(
        e["resource"]["id"]
        for e in synthea_patient_bundle["entry"]
        if e["resource"]["resourceType"] == "Patient"
    )
    with caplog.at_level(logging.INFO, logger="readmit.audit"):
        client.post("/v1/fhir/risk-assessment", json=synthea_patient_bundle)
    audit_lines = [r.getMessage() for r in caplog.records if r.name == "readmit.audit"]
    assert audit_lines and all(patient_id not in line for line in audit_lines)


def test_risk_assessment_validation():
    with pytest.raises(ValueError):
        build_risk_assessment(
            patient_id="p",
            encounter_id="e",
            risk=1.4,
            tier="high",
            model_name="m",
            model_version="v",
            factors=[],
        )
    with pytest.raises(ValueError):
        build_risk_assessment(
            patient_id="p",
            encounter_id="e",
            risk=0.4,
            tier="extreme",
            model_name="m",
            model_version="v",
            factors=[],
        )

"""Copilot retrieval, privacy of the patient context, and LLM guardrails."""

from types import SimpleNamespace

import pytest

from readmit.copilot.assistant import CareCopilot, validate_citations
from readmit.copilot.knowledge import load_guidance
from readmit.copilot.retriever import TfidfRetriever, evaluate_retrieval
from readmit.service import ScoringService
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def guidance():
    return load_guidance(ROOT / "knowledge")


class FakeLLM:
    """Stands in for anthropic.Anthropic(); records the request and returns canned text."""

    def __init__(self, text=None, fail=False):
        self.text, self.fail, self.calls = text, fail, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("API unavailable")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_guidance_loads_with_metadata(guidance):
    assert len(guidance) >= 10
    assert all(doc["title"] and doc["references"] for doc in guidance)


def test_retrieval_quality(guidance):
    result = evaluate_retrieval(TfidfRetriever(guidance))
    assert result["hit_rate"] >= 0.9 and result["mrr"] >= 0.8


def test_citation_validation():
    text, valid, invalid = validate_citations(
        "Risk is high [Encounter/1, Condition/made-up]. See [guide:follow-up-visit].",
        {"Encounter/1", "guide:follow-up-visit"},
    )
    assert valid == ["Encounter/1", "guide:follow-up-visit"] and invalid == ["Condition/made-up"]
    assert "made-up" not in text and "[Encounter/1]" in text


def _service(trained, llm=None):
    return ScoringService(trained.bundle, llm_client=llm)


def test_context_is_minimum_necessary(trained, synthea_patient_bundle):
    service = _service(trained)
    scored = service.score_bundle(synthea_patient_bundle)
    brief = service.brief(scored)
    patient = next(
        e["resource"]
        for e in synthea_patient_bundle["entry"]
        if e["resource"]["resourceType"] == "Patient"
    )
    for secret in (
        patient["name"][0]["family"],
        patient["birthDate"],
        patient["address"][0]["city"],
    ):
        assert secret not in brief.text
    assert brief.mode == "template" and brief.citations


def test_llm_brief_strips_fabricated_citations(trained, synthea_patient_bundle):
    service = _service(trained)
    scored = service.score_bundle(synthea_patient_bundle)
    fake = FakeLLM(f"Risk summary [Encounter/{scored.encounter_id}]. Fabricated [Condition/xyz].")
    service.copilot.llm_client = fake
    brief = service.brief(scored)
    assert brief.mode == "llm"
    assert brief.invalid_citations == ["Condition/xyz"] and "Condition/xyz" not in brief.text
    request = fake.calls[0]
    assert "Do not diagnose" in request["system"]
    assert patient_name_not_sent(request, synthea_patient_bundle)


def patient_name_not_sent(request, bundle):
    patient = next(
        e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient"
    )
    return patient["name"][0]["family"] not in request["messages"][0]["content"]


def test_llm_failure_falls_back_to_template(trained, synthea_patient_bundle):
    service = _service(trained, llm=FakeLLM(fail=True))
    brief = service.brief(service.score_bundle(synthea_patient_bundle))
    assert brief.mode == "template" and "Caveats" in brief.text


def test_copilot_requires_documents():
    with pytest.raises(ValueError):
        CareCopilot([])

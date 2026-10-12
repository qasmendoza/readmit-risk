"""Express a prediction as an HL7 FHIR R4 ``RiskAssessment`` resource.

Returning a standard resource (instead of bespoke JSON) lets an EHR or FHIR server store and display
the score natively. ``qualitativeRisk`` uses the HL7 ``risk-probability`` code system, ``basis``
references the index encounter, and the main contributing factors are carried in ``note``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

RISK_PROBABILITY_SYSTEM = "http://terminology.hl7.org/CodeSystem/risk-probability"
_TIER_DISPLAY = {
    "low": "Low likelihood",
    "moderate": "Moderate likelihood",
    "high": "High likelihood",
}


def build_risk_assessment(
    *,
    patient_id: str,
    encounter_id: str,
    risk: float,
    tier: str,
    model_name: str,
    model_version: str,
    factors: list[dict[str, Any]],
    occurrence: datetime | None = None,
) -> dict[str, Any]:
    if tier not in _TIER_DISPLAY:
        raise ValueError(f"unknown tier '{tier}'")
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be a probability")
    moment = (occurrence or datetime.now(UTC)).isoformat(timespec="seconds")
    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "occurrenceDateTime": moment,
        "method": {"text": f"readmit {model_name} model, version {model_version}"},
        "code": {"text": "30-day hospital readmission risk"},
        "basis": [{"reference": f"Encounter/{encounter_id}"}],
        "prediction": [
            {
                "outcome": {"text": "Acute hospital readmission within 30 days of discharge"},
                "probabilityDecimal": round(float(risk), 4),
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": RISK_PROBABILITY_SYSTEM,
                            "code": tier,
                            "display": _TIER_DISPLAY[tier],
                        }
                    ]
                },
                "whenRange": {
                    "low": {
                        "value": 0,
                        "unit": "days",
                        "system": "http://unitsofmeasure.org",
                        "code": "d",
                    },
                    "high": {
                        "value": 30,
                        "unit": "days",
                        "system": "http://unitsofmeasure.org",
                        "code": "d",
                    },
                },
            }
        ],
        "mitigation": "Decision support for care-transition outreach; not a diagnosis.",
    }
    if factors:
        resource["note"] = [
            {"text": "Main contributing factors: " + "; ".join(f["text"] for f in factors)}
        ]
    return resource

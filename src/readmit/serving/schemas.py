"""Request and response models for the ReadmitRisk API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from readmit.features import FEATURE_NAMES, LACE_COLUMN

_ALLOWED = {*FEATURE_NAMES, LACE_COLUMN}


class FeaturesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": {
                        "age_years": 81,
                        "male": 0,
                        "los_days": 6.5,
                        "emergency_admission": 1,
                        "prior_admissions_12m": 2,
                        "ed_visits_6m": 3,
                        "charlson_score": 4,
                        "chf": 1,
                        "n_medications_12m": 8,
                        "polypharmacy": 1,
                        "last_egfr": 41,
                        "last_sodium": 133,
                    }
                }
            ]
        }
    )

    features: dict[str, float | int | None] = Field(
        description="Precomputed features; omitted ones are treated as missing."
    )

    @field_validator("features")
    @classmethod
    def _known(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(value) - _ALLOWED)
        if unknown:
            raise ValueError(f"unknown feature(s): {unknown}")
        return value


class Factor(BaseModel):
    feature: str
    description: str
    value: float
    typical: float
    contribution: float
    text: str


class PredictResponse(BaseModel):
    model_version: str
    model_name: str
    risk: float
    tier: Literal["low", "moderate", "high"]
    thresholds: dict[str, float]
    factors: list[Factor]


class GuidanceRef(BaseModel):
    id: str
    title: str
    score: float
    references: str = ""


class BriefResponse(BaseModel):
    model_version: str
    encounter_id: str
    risk: float
    tier: Literal["low", "moderate", "high"]
    factors: list[Factor]
    brief: str
    mode: Literal["llm", "template"]
    citations: list[str]
    invalid_citations: list[str]
    guidance: list[GuidanceRef]


class ModelInfo(BaseModel):
    version: str
    model_name: str
    created_at: str | None
    data_source: str | None
    test_metrics: dict[str, float]
    confidence_intervals: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    operating_point: dict[str, Any]
    gate: dict[str, Any]
    features: list[dict[str, Any]]

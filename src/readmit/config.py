"""Typed, validated configuration (YAML + dotted CLI overrides).

Example: ``readmit run --set evaluation.capacity_fraction=0.05 --set "models.candidates=[lace, hist_gbm]"``
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
MODEL_NAMES = ("lace", "logistic", "hist_gbm")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    input_dir: Path = Path("data/fhir")
    input_format: Literal["auto", "ndjson", "bundle"] = "auto"
    tables_dir: Path = Path("data/tables")
    synthetic_patients: int = Field(3000, ge=50)


class CohortConfig(_Strict):
    min_age: int = Field(18, ge=0)
    followup_days: int = Field(30, ge=1)
    transfer_gap_hours: float = Field(24.0, ge=0)
    exclude_encounter_types: list[str] = Field(
        default_factory=lambda: [
            "skilled nursing",
            "rehabilitation",
            "detoxification",
            "hospice",
            "psychiatric",
        ]
    )
    as_of_date: date | None = None


class FeatureConfig(_Strict):
    utilization_lookback_days: int = Field(365, ge=30)
    ed_lookback_days: int = Field(182, ge=30)
    lab_lookback_days: int = Field(365, ge=1)
    medication_lookback_days: int = Field(365, ge=30)
    condition_active_window_days: int = Field(365, ge=0)
    polypharmacy_threshold: int = Field(5, ge=2)


class SplitConfig(_Strict):
    test_fraction: float = Field(0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(5, ge=3, le=10)


class ModelsConfig(_Strict):
    candidates: list[str] = Field(default_factory=lambda: list(MODEL_NAMES))
    selection_metric: Literal["average_precision", "roc_auc", "neg_brier_score"] = (
        "average_precision"
    )
    calibration: Literal["sigmoid", "isotonic"] = "sigmoid"
    monotonic_constraints: bool = True

    @field_validator("candidates")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(MODEL_NAMES))
        if unknown or not value:
            raise ValueError(f"candidates must be a non-empty subset of {list(MODEL_NAMES)}")
        return value


class EvaluationConfig(_Strict):
    bootstrap_iterations: int = Field(1000, ge=100)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    capacity_fraction: float = Field(0.10, gt=0, lt=1)
    moderate_fraction: float = Field(0.30, gt=0, lt=1)
    fairness_attributes: list[str] = Field(
        default_factory=lambda: ["sex", "race", "ethnicity", "age_band"]
    )
    fairness_min_group_size: int = Field(50, ge=10)

    @model_validator(mode="after")
    def _tiers(self) -> EvaluationConfig:
        if self.moderate_fraction <= self.capacity_fraction:
            raise ValueError("moderate_fraction must exceed capacity_fraction")
        return self


class GateConfig(_Strict):
    enforce: bool = True
    min_auroc: float = Field(0.65, ge=0.5, le=1)
    require_beats_baseline: bool = True
    baseline: str = "lace"
    max_fairness_auroc_gap: float | None = None


class CopilotConfig(_Strict):
    knowledge_dir: Path = Path("knowledge")
    llm_model: str = "claude-haiku-5-5"
    max_tokens: int = Field(700, ge=100)
    top_k_guidance: int = Field(3, ge=1, le=6)


class TrackingConfig(_Strict):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "readmission-risk"
    register_model: bool = False
    model_name: str = "readmission-risk"


class OutputConfig(_Strict):
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


class Config(_Strict):
    project: str = "readmit-risk"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    copilot: CopilotConfig = Field(default_factory=CopilotConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def model_dir(self) -> Path:
        return self.output.artifacts_dir / "model"


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override '{dotted_key}': '{key}' is not a section")
    node[keys[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key=value")
        key, raw = item.split("=", 1)
        _set_dotted(result, key.strip(), yaml.safe_load(raw))
    return result


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    elif path is not None:
        raise FileNotFoundError(f"config file not found: {config_path}")
    return Config.model_validate(_deep_merge(raw, parse_overrides(overrides)))

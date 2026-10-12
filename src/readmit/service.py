"""Scoring service shared by the API, SageMaker handler, and CLI.

FHIR Bundle -> flatten -> index hospitalization -> point-in-time features -> calibrated risk, tier,
and drivers. Cohort and feature settings are read from the model bundle's metadata, so serving
always uses exactly the definitions the model was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from readmit.bundle import ModelBundle
from readmit.cohort import build_episodes, scoring_episode
from readmit.config import CohortConfig, CopilotConfig, FeatureConfig
from readmit.copilot.assistant import Brief, CareCopilot
from readmit.copilot.context import build_patient_context
from readmit.features import compute_features
from readmit.fhir.flatten import FhirTables, flatten_bundle
from readmit.fhir.risk_assessment import build_risk_assessment


@dataclass
class ScoredEncounter:
    patient_id: str
    encounter_id: str
    risk: float
    tier: str
    factors: list[dict[str, Any]]
    features: dict[str, float]
    tables: FhirTables = field(repr=False)
    episode: pd.Series = field(repr=False)
    all_episodes: pd.DataFrame = field(repr=False)


class ScoringService:
    def __init__(
        self, bundle: ModelBundle, llm_client: Any | None = None, llm_model: str | None = None
    ) -> None:
        self.bundle = bundle
        meta = bundle.metadata
        self.cohort_cfg = CohortConfig.model_validate(meta.get("cohort_config", {}))
        self.feature_cfg = FeatureConfig.model_validate(meta.get("feature_config", {}))
        copilot_defaults = CopilotConfig()
        self.copilot = CareCopilot(
            bundle.guidance_docs,
            llm_client=llm_client,
            model=llm_model,
            max_tokens=copilot_defaults.max_tokens,
            top_k=copilot_defaults.top_k_guidance,
        )

    def score_bundle(
        self, fhir_bundle: dict[str, Any], encounter_id: str | None = None, top_k: int = 5
    ) -> ScoredEncounter:
        tables = flatten_bundle(fhir_bundle)
        if tables.patients.empty:
            raise ValueError("the Bundle contains no Patient resource")
        episode = scoring_episode(tables, self.cohort_cfg, encounter_id)
        all_episodes = build_episodes(
            tables.encounters,
            self.cohort_cfg.exclude_encounter_types,
            self.cohort_cfg.transfer_gap_hours,
        )
        features = compute_features(
            tables, episode, self.feature_cfg, self.cohort_cfg, all_episodes
        )
        scored = self.bundle.score(features.reset_index(drop=True), top_k=top_k)
        row = episode.iloc[0]
        return ScoredEncounter(
            patient_id=str(row["patient_id"]),
            encounter_id=str(row["episode_id"]),
            risk=scored["risk"],
            tier=scored["tier"],
            factors=scored["factors"],
            features={k: (None if pd.isna(v) else float(v)) for k, v in features.iloc[0].items()},
            tables=tables,
            episode=row,
            all_episodes=all_episodes,
        )

    def risk_assessment(self, scored: ScoredEncounter) -> dict[str, Any]:
        return build_risk_assessment(
            patient_id=scored.patient_id,
            encounter_id=scored.encounter_id,
            risk=scored.risk,
            tier=scored.tier,
            model_name=self.bundle.model_name,
            model_version=self.bundle.version,
            factors=scored.factors,
        )

    def brief(self, scored: ScoredEncounter) -> Brief:
        context = build_patient_context(
            scored.tables,
            scored.episode,
            pd.Series(scored.features),
            scored.all_episodes,
            self.feature_cfg,
        )
        return self.copilot.brief(context, scored.risk, scored.tier, scored.factors)

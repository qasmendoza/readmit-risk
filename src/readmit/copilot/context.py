"""Build a minimum-necessary, citable patient context for the copilot.

Privacy-by-design choices, inspired by the HIPAA minimum-necessary standard and Safe Harbor
de-identification:

* no names, addresses, contact details, or record identifiers beyond FHIR resource ids;
* ages are reported as bands, with everyone 85 and over grouped together;
* calendar dates are converted to relative times ("12 days before discharge").

Each fact carries the FHIR resource id it came from (e.g. ``Condition/3f2a...``) so generated text
can cite its evidence, and the features it supports so model drivers can be linked to evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from readmit.config import FeatureConfig
from readmit.explain import format_value
from readmit.fhir.codes import LAB_FEATURES, NON_DISEASE_PATTERN, SDOH_PATTERN, charlson_categories
from readmit.fhir.flatten import FhirTables

_CATEGORY_FEATURES = {
    "congestive_heart_failure": "chf",
    "chronic_pulmonary_disease": "chronic_pulmonary",
    "renal_disease": "renal_disease",
    "diabetes_complicated": "diabetes_complicated",
    "dementia": "dementia",
    "any_malignancy": "malignancy",
    "metastatic_solid_tumor": "malignancy",
}


@dataclass
class Fact:
    id: str
    text: str
    kind: str
    features: list[str] = field(default_factory=list)


@dataclass
class PatientContext:
    facts: list[Fact]
    age_band: str
    sex: str

    def facts_for(self, feature: str, limit: int = 2) -> list[Fact]:
        return [f for f in self.facts if feature in f.features][:limit]

    def render(self) -> str:
        return "\n".join(f"[{f.id}] {f.text}" for f in self.facts)

    @property
    def ids(self) -> set[str]:
        return {f.id for f in self.facts}


def _days_before(moment: pd.Timestamp, anchor: pd.Timestamp) -> str:
    days = (anchor - moment).total_seconds() / 86400
    if days < 1:
        return "on the day of discharge"
    if days < 60:
        return f"{days:.0f} days before discharge"
    return (
        f"about {days / 365.25:.1f} years before discharge"
        if days >= 365
        else f"{days / 30.4:.0f} months before discharge"
    )


def build_patient_context(
    tables: FhirTables,
    episode: pd.Series,
    features: pd.Series,
    all_episodes: pd.DataFrame,
    cfg: FeatureConfig,
) -> PatientContext:
    discharge, admit = episode["discharge"], episode["admit"]
    facts = [
        Fact(
            id=f"Encounter/{episode['episode_id']}",
            text=(
                f"Index hospitalization: {features['los_days']:.1f}-day stay, "
                f"{'admitted through the emergency department' if features['emergency_admission'] else 'not admitted through the ED'}"
                f"{', discharged on a weekend' if features['weekend_discharge'] else ''}."
            ),
            kind="encounter",
            features=["los_days", "emergency_admission", "weekend_discharge"],
        ),
        Fact(
            id="Patient",
            text=f"Age band {episode['age_band']}, sex {episode['sex']}.",
            kind="demographics",
            features=["age_years", "male"],
        ),
    ]
    prior = all_episodes[
        (all_episodes["patient_id"] == episode["patient_id"])
        & (all_episodes["discharge"] < admit)
        & (all_episodes["discharge"] >= admit - pd.Timedelta(days=cfg.utilization_lookback_days))
    ]
    for row in prior.sort_values("discharge", ascending=False).head(3).itertuples():
        stay = (row.discharge - row.admit).total_seconds() / 86400
        facts.append(
            Fact(
                f"Encounter/{row.episode_id}",
                f"Prior hospitalization ({stay:.1f} days) ended {_days_before(row.discharge, discharge)}.",
                "prior_admission",
                ["prior_admissions_12m", "days_since_last_discharge"],
            )
        )
    encounters = tables.encounters
    ed = encounters[
        (encounters["class_code"] == "EMER")
        & (encounters["start"] >= admit - pd.Timedelta(days=cfg.ed_lookback_days))
        & (encounters["start"] < admit - pd.Timedelta(hours=24))
    ]
    for row in ed.sort_values("start", ascending=False).head(3).itertuples():
        facts.append(
            Fact(
                f"Encounter/{row.encounter_id}",
                f"Emergency department visit {_days_before(row.start, discharge)}.",
                "ed_visit",
                ["ed_visits_6m"],
            )
        )

    cond = tables.conditions.assign(available=lambda c: c["recorded"].fillna(c["onset"]))
    active = cond[
        (cond["available"] <= discharge)
        & (cond["abatement"].isna() | (cond["abatement"] > discharge))
    ]
    added = 0
    for row in active.sort_values("available").itertuples():
        display = row.display or ""
        categories = charlson_categories(display)
        sdoh = bool(SDOH_PATTERN.search(display))
        if not categories and not sdoh and NON_DISEASE_PATTERN.search(display):
            continue
        supported = ["n_active_conditions"] if not sdoh else ["sdoh_count"]
        supported += [_CATEGORY_FEATURES[c] for c in categories if c in _CATEGORY_FEATURES]
        supported += ["charlson_score"] if categories else []
        facts.append(
            Fact(
                f"Condition/{row.condition_id}",
                f"{display} (documented {_days_before(row.available, discharge)}).",
                "social_need" if sdoh else "condition",
                supported,
            )
        )
        added += 1
        if added >= 10:
            break

    meds = tables.medications[
        (tables.medications["authored_on"] <= discharge)
        & (
            tables.medications["authored_on"]
            >= discharge - pd.Timedelta(days=cfg.medication_lookback_days)
        )
    ]
    distinct = meds.sort_values("authored_on", ascending=False).drop_duplicates("display")
    for row in distinct.head(6).itertuples():
        facts.append(
            Fact(
                f"MedicationRequest/{row.medication_request_id}",
                f"Prescribed: {row.display}.",
                "medication",
                ["n_medications_12m", "polypharmacy"],
            )
        )
    if len(distinct) > 6:
        facts.append(
            Fact(
                "MedicationRequest/summary",
                f"{len(distinct)} distinct medications prescribed in the prior 12 months.",
                "medication",
                ["n_medications_12m", "polypharmacy"],
            )
        )

    obs = tables.observations
    for feature, codes in LAB_FEATURES.items():
        candidates = obs[obs["code"].isin(codes) & (obs["effective"] <= discharge)]
        if candidates.empty or pd.isna(features.get(feature)):
            continue
        latest = candidates.sort_values("effective").iloc[-1]
        facts.append(
            Fact(
                f"Observation/{latest['observation_id']}",
                f"{latest['display'] or feature}: {format_value(feature, float(latest['value']))} "
                f"({_days_before(latest['effective'], discharge)}).",
                "lab",
                [feature],
            )
        )
    return PatientContext(facts=facts, age_band=str(episode["age_band"]), sex=str(episode["sex"]))

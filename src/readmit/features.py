"""Point-in-time feature engineering for index hospitalizations.

Every feature is computed *as of the discharge timestamp* using only information that existed at
that moment, which is what the model will see in production:

* utilization windows end at the index admission, and the triggering ED visit is not counted as
  a "prior" visit;
* conditions count only if recorded (``recordedDate``, falling back to onset) by discharge, so a
  diagnosis documented after discharge with a back-dated onset cannot leak into training;
* medications are those *prescribed* in the lookback window. ``MedicationRequest.status`` is
  deliberately ignored because it reflects the state at export time, not at discharge;
* labs and vitals are the most recent values on or before discharge.

The same function computes features for model training and for real-time API requests, so there
is no training/serving skew.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from readmit.cohort import build_episodes
from readmit.config import CohortConfig, FeatureConfig
from readmit.fhir.codes import (
    AMBULATORY_CLASS,
    EMERGENCY_CLASS,
    LAB_FEATURES,
    NON_DISEASE_PATTERN,
    SDOH_PATTERN,
    charlson_categories,
    charlson_score,
)
from readmit.fhir.flatten import FhirTables


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    kind: Literal["continuous", "count", "binary", "score"]
    unit: str = ""
    source: str = ""
    monotonic: int = 0  # +1: risk may only rise with the feature; -1: only fall; 0: unconstrained

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("age_years", "Age at admission", "continuous", "years", "Patient.birthDate"),
    FeatureSpec("male", "Male sex", "binary", "", "Patient.gender"),
    FeatureSpec("los_days", "Length of stay", "continuous", "days", "Encounter.period", 1),
    FeatureSpec(
        "emergency_admission",
        "Admitted through the emergency department",
        "binary",
        "",
        "Encounter (EMER within 24 h before admission)",
        1,
    ),
    FeatureSpec(
        "prior_admissions_12m",
        "Hospitalizations in the prior 12 months",
        "count",
        "",
        "Encounter (acute IMP)",
        1,
    ),
    FeatureSpec(
        "ed_visits_6m",
        "Emergency department visits in the prior 6 months",
        "count",
        "",
        "Encounter (EMER)",
        1,
    ),
    FeatureSpec(
        "outpatient_visits_12m",
        "Outpatient visits in the prior 12 months",
        "count",
        "",
        "Encounter (AMB)",
    ),
    FeatureSpec(
        "days_since_last_discharge",
        "Days since previous discharge (capped at 365)",
        "continuous",
        "days",
        "Encounter (acute IMP)",
        -1,
    ),
    FeatureSpec("charlson_score", "Charlson Comorbidity Index", "score", "", "Condition", 1),
    FeatureSpec("chf", "Congestive heart failure", "binary", "", "Condition", 1),
    FeatureSpec("chronic_pulmonary", "Chronic pulmonary disease", "binary", "", "Condition", 1),
    FeatureSpec(
        "renal_disease", "Moderate or severe chronic kidney disease", "binary", "", "Condition", 1
    ),
    FeatureSpec(
        "diabetes_complicated", "Diabetes with complications", "binary", "", "Condition", 1
    ),
    FeatureSpec("dementia", "Dementia", "binary", "", "Condition"),
    FeatureSpec(
        "malignancy", "Cancer (any malignancy, including metastatic)", "binary", "", "Condition"
    ),
    FeatureSpec("n_active_conditions", "Active chronic conditions", "count", "", "Condition"),
    FeatureSpec(
        "sdoh_count", "Documented social needs", "count", "", "Condition (SDOH findings)", 1
    ),
    FeatureSpec(
        "n_medications_12m",
        "Distinct medications prescribed in the prior 12 months",
        "count",
        "",
        "MedicationRequest",
        1,
    ),
    FeatureSpec("polypharmacy", "Polypharmacy", "binary", "", "MedicationRequest", 1),
    FeatureSpec("last_bmi", "Most recent BMI", "continuous", "kg/m²", "Observation 39156-5"),
    FeatureSpec(
        "last_systolic_bp",
        "Most recent systolic blood pressure",
        "continuous",
        "mmHg",
        "Observation 8480-6",
    ),
    FeatureSpec(
        "last_heart_rate", "Most recent heart rate", "continuous", "/min", "Observation 8867-4"
    ),
    FeatureSpec(
        "last_egfr", "Most recent eGFR", "continuous", "mL/min/1.73m²", "Observation 33914-3", -1
    ),
    FeatureSpec(
        "last_creatinine", "Most recent creatinine", "continuous", "mg/dL", "Observation 38483-4"
    ),
    FeatureSpec("last_hba1c", "Most recent HbA1c", "continuous", "%", "Observation 4548-4"),
    FeatureSpec("last_glucose", "Most recent glucose", "continuous", "mg/dL", "Observation 2339-0"),
    FeatureSpec("last_sodium", "Most recent sodium", "continuous", "mmol/L", "Observation 2947-0"),
    FeatureSpec(
        "weekend_discharge", "Discharged on a weekend", "binary", "", "Encounter.period.end"
    ),
)
FEATURE_NAMES: list[str] = [spec.name for spec in FEATURE_SPECS]
SPEC_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}
LACE_COLUMN = "lace_score"
_CODE_TO_LAB = {code: name for name, codes in LAB_FEATURES.items() for code in codes}


def lace_score(los_days, emergency, charlson, ed_visits_6m) -> np.ndarray:
    """LACE index (van Walraven et al., 2010): Length of stay, Acuity, Comorbidity, ED visits."""
    days = np.floor(np.asarray(los_days, dtype=float))
    length = np.select(
        [days < 1, days == 1, days == 2, days == 3, days <= 6, days <= 13], [0, 1, 2, 3, 4, 5], 7
    )
    acuity = 3 * np.asarray(emergency, dtype=int)
    cci = np.asarray(charlson, dtype=int)
    comorbidity = np.select([cci == 0, cci == 1, cci == 2, cci == 3], [0, 1, 2, 3], 5)
    ed = np.minimum(np.asarray(ed_visits_6m, dtype=int), 4)
    return length + acuity + comorbidity + ed


def _windowed(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    time_column: str,
    anchor: str,
    start_offset: pd.Timedelta,
    end_offset: pd.Timedelta,
    inclusive_end: bool = False,
) -> pd.DataFrame:
    """Join events to episodes and keep those in [anchor + start, anchor + end)."""
    merged = episodes[["episode_id", "patient_id", anchor]].merge(events, on="patient_id")
    low = merged[anchor] + start_offset
    high = merged[anchor] + end_offset
    in_window = (merged[time_column] >= low) & (
        merged[time_column] <= high if inclusive_end else merged[time_column] < high
    )
    return merged[in_window]


def _count(frame: pd.DataFrame, index: pd.Index, column: str | None = None) -> pd.Series:
    if column is None:
        counts = frame.groupby("episode_id").size()
    else:
        counts = frame.groupby("episode_id")[column].nunique()
    return counts.reindex(index, fill_value=0).astype(int)


def compute_features(
    tables: FhirTables,
    episodes: pd.DataFrame,
    cfg: FeatureConfig,
    cohort_cfg: CohortConfig,
    all_episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix (one row per episode_id) plus the LACE baseline score."""
    ep = episodes[
        ["episode_id", "patient_id", "admit", "discharge", "los_days", "age", "sex"]
    ].copy()
    index = pd.Index(ep["episode_id"], name="episode_id")
    out = pd.DataFrame(index=index)
    out["age_years"] = ep["age"].to_numpy()
    out["male"] = (ep["sex"] == "male").astype(int).to_numpy()
    out["los_days"] = ep["los_days"].to_numpy()
    out["weekend_discharge"] = (ep["discharge"].dt.weekday >= 5).astype(int).to_numpy()

    if all_episodes is None:
        all_episodes = build_episodes(
            tables.encounters, cohort_cfg.exclude_encounter_types, cohort_cfg.transfer_gap_hours
        )
    previous = all_episodes[["patient_id", "discharge"]].rename(
        columns={"discharge": "prev_discharge"}
    )
    lookback = pd.Timedelta(days=cfg.utilization_lookback_days)
    prior = _windowed(ep, previous, "prev_discharge", "admit", -lookback, pd.Timedelta(0))
    out["prior_admissions_12m"] = _count(prior, index)
    last = _windowed(
        ep, previous, "prev_discharge", "admit", -pd.Timedelta(days=36500), pd.Timedelta(0)
    )
    last_discharge = last.groupby("episode_id")["prev_discharge"].max().reindex(index)
    gap_days = (ep.set_index("episode_id")["admit"] - last_discharge).dt.total_seconds() / 86400
    out["days_since_last_discharge"] = gap_days.clip(upper=365).fillna(365.0)

    encounters = tables.encounters[["patient_id", "class_code", "start"]]
    emergency = encounters[encounters["class_code"] == EMERGENCY_CLASS]
    out["emergency_admission"] = (
        _count(
            _windowed(
                ep,
                emergency,
                "start",
                "admit",
                -pd.Timedelta(hours=24),
                pd.Timedelta(0),
                inclusive_end=True,
            ),
            index,
        )
        > 0
    ).astype(int)
    out["ed_visits_6m"] = _count(
        _windowed(
            ep,
            emergency,
            "start",
            "admit",
            -pd.Timedelta(days=cfg.ed_lookback_days),
            -pd.Timedelta(hours=24),
        ),
        index,
    )
    ambulatory = encounters[encounters["class_code"] == AMBULATORY_CLASS]
    out["outpatient_visits_12m"] = _count(
        _windowed(ep, ambulatory, "start", "admit", -lookback, pd.Timedelta(0)), index
    )

    out = out.join(_condition_features(tables.conditions, ep, cfg, index))
    meds = tables.medications[["patient_id", "code", "display", "authored_on"]].assign(
        med_key=lambda m: m["code"].fillna(m["display"])
    )
    recent_meds = _windowed(
        ep,
        meds,
        "authored_on",
        "discharge",
        -pd.Timedelta(days=cfg.medication_lookback_days),
        pd.Timedelta(0),
        inclusive_end=True,
    )
    out["n_medications_12m"] = _count(recent_meds, index, "med_key")
    out["polypharmacy"] = (out["n_medications_12m"] >= cfg.polypharmacy_threshold).astype(int)
    out = out.join(_lab_features(tables.observations, ep, cfg, index))

    out[LACE_COLUMN] = lace_score(
        out["los_days"], out["emergency_admission"], out["charlson_score"], out["ed_visits_6m"]
    )
    return out[[*FEATURE_NAMES, LACE_COLUMN]]


def _condition_features(
    conditions: pd.DataFrame, ep: pd.DataFrame, cfg: FeatureConfig, index: pd.Index
) -> pd.DataFrame:
    cond = conditions[["patient_id", "code", "display", "onset", "recorded", "abatement"]].copy()
    cond["available"] = cond["recorded"].fillna(cond["onset"])
    merged = ep[["episode_id", "patient_id", "discharge"]].merge(cond, on="patient_id")
    known = merged["available"].notna() & (merged["available"] <= merged["discharge"])
    window = pd.Timedelta(days=cfg.condition_active_window_days)
    recent = merged["abatement"].isna() | (merged["abatement"] >= merged["discharge"] - window)
    active_now = merged["abatement"].isna() | (merged["abatement"] > merged["discharge"])
    relevant = merged[known & recent]

    displays = relevant["display"].fillna("").unique()
    category_map = {d: charlson_categories(d) for d in displays}
    categories = (
        relevant[["episode_id", "display"]]
        .assign(category=relevant["display"].map(category_map))
        .explode("category")
        .dropna(subset=["category"])
    )
    per_episode = categories.groupby("episode_id")["category"].agg(set)
    sets = per_episode.reindex(index).map(lambda v: v if isinstance(v, set) else set())

    current = merged[known & active_now]
    is_sdoh = current["display"].fillna("").str.contains(SDOH_PATTERN)
    is_disease = ~current["display"].fillna("").str.contains(NON_DISEASE_PATTERN)
    frame = pd.DataFrame(index=index)
    frame["charlson_score"] = sets.map(charlson_score).astype(int)
    frame["chf"] = sets.map(lambda s: "congestive_heart_failure" in s).astype(int)
    frame["chronic_pulmonary"] = sets.map(lambda s: "chronic_pulmonary_disease" in s).astype(int)
    frame["renal_disease"] = sets.map(lambda s: "renal_disease" in s).astype(int)
    frame["diabetes_complicated"] = sets.map(lambda s: "diabetes_complicated" in s).astype(int)
    frame["dementia"] = sets.map(lambda s: "dementia" in s).astype(int)
    frame["malignancy"] = sets.map(
        lambda s: bool({"any_malignancy", "metastatic_solid_tumor"} & s)
    ).astype(int)
    frame["n_active_conditions"] = _count(current[is_disease], index, "code")
    frame["sdoh_count"] = _count(current[is_sdoh], index, "display")
    return frame


def _lab_features(
    observations: pd.DataFrame, ep: pd.DataFrame, cfg: FeatureConfig, index: pd.Index
) -> pd.DataFrame:
    obs = observations[observations["code"].isin(_CODE_TO_LAB)][
        ["patient_id", "code", "value", "effective"]
    ]
    obs = obs.assign(feature=obs["code"].map(_CODE_TO_LAB)).dropna(subset=["value"])
    recent = _windowed(
        ep,
        obs,
        "effective",
        "discharge",
        -pd.Timedelta(days=cfg.lab_lookback_days),
        pd.Timedelta(0),
        inclusive_end=True,
    )
    latest = (
        recent.sort_values("effective").groupby(["episode_id", "feature"])["value"].last().unstack()
    )
    return latest.reindex(index=index, columns=list(LAB_FEATURES)).astype(float)

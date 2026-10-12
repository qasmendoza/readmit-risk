"""Index hospitalizations and 30-day readmission labels.

Definitions (simplified from the CMS Hospital Readmissions Reduction Program methodology):

* **Acute hospitalization**: a finished inpatient (``IMP``) encounter whose type is not a
  non-acute setting (skilled nursing, rehabilitation, detox, hospice, psychiatric).
* **Transfer chains**: acute encounters that start within ``transfer_gap_hours`` of the previous
  one ending are merged into a single hospitalization. Synthea, for example, models an ICU stay as
  admission -> ICU transfer -> ward; counting those as readmissions would corrupt the labels.
* **Index exclusions**: age < ``min_age`` at admission; death during the stay; discharge too close
  to the data extraction date to observe the full follow-up window (right-censoring).
* **Label**: another acute hospitalization beginning within ``followup_days`` after discharge.
  Readmissions can themselves be index stays, as in the CMS measure.

Known simplification: CMS's planned-readmission algorithm (which excludes e.g. scheduled
chemotherapy) is not implemented; all qualifying readmissions count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from readmit.config import CohortConfig
from readmit.fhir.codes import INPATIENT_CLASS
from readmit.fhir.flatten import FhirTables

AGE_BANDS = [
    (18, 45, "18-44"),
    (45, 65, "45-64"),
    (65, 75, "65-74"),
    (75, 85, "75-84"),
    (85, 200, "85+"),
]


@dataclass
class Cohort:
    index: pd.DataFrame
    flow: list[dict[str, object]] = field(default_factory=list)
    as_of: pd.Timestamp | None = None

    @property
    def readmission_rate(self) -> float:
        return float(self.index["readmitted_30d"].mean()) if len(self.index) else float("nan")


def age_band(age: float) -> str:
    for low, high, label in AGE_BANDS:
        if low <= age < high:
            return label
    return "<18"


def acute_inpatient(encounters: pd.DataFrame, exclude_types: list[str]) -> pd.DataFrame:
    frame = encounters[
        (encounters["class_code"] == INPATIENT_CLASS)
        & encounters["end"].notna()
        & encounters["start"].notna()
        & (encounters["status"].fillna("finished") == "finished")
    ]
    if exclude_types:
        pattern = "|".join(exclude_types)
        frame = frame[
            ~frame["type_display"].fillna("").str.contains(pattern, case=False, regex=True)
        ]
    return frame


def build_episodes(
    encounters: pd.DataFrame, exclude_types: list[str], transfer_gap_hours: float
) -> pd.DataFrame:
    """Merge contiguous acute inpatient encounters into hospitalization episodes."""
    acute = acute_inpatient(encounters, exclude_types).sort_values(["patient_id", "start", "end"])
    columns = ["episode_id", "patient_id", "admit", "discharge", "n_encounters", "encounter_ids"]
    if acute.empty:
        return pd.DataFrame(columns=columns)
    gap = pd.Timedelta(hours=transfer_gap_hours)
    episodes = []
    for patient_id, group in acute.groupby("patient_id", sort=False):
        current: dict | None = None
        for row in group.itertuples(index=False):
            if current is not None and row.start <= current["discharge"] + gap:
                current["discharge"] = max(current["discharge"], row.end)
                current["encounter_ids"].append(row.encounter_id)
            else:
                if current is not None:
                    episodes.append(current)
                current = {
                    "episode_id": row.encounter_id,
                    "patient_id": patient_id,
                    "admit": row.start,
                    "discharge": row.end,
                    "encounter_ids": [row.encounter_id],
                }
        if current is not None:
            episodes.append(current)
    frame = pd.DataFrame(episodes)
    frame["n_encounters"] = frame["encounter_ids"].str.len()
    return frame[columns].reset_index(drop=True)


def label_readmissions(episodes: pd.DataFrame, followup_days: int) -> pd.DataFrame:
    """Flag episodes followed by another episode starting within ``followup_days``."""
    frame = episodes.sort_values(["patient_id", "admit"]).copy()
    next_admit = frame.groupby("patient_id")["admit"].shift(-1)
    delta = next_admit - frame["discharge"]
    window = pd.Timedelta(days=followup_days)
    frame["days_to_readmission"] = delta.dt.total_seconds() / 86400
    frame["readmitted_30d"] = ((delta > pd.Timedelta(0)) & (delta <= window)).astype(int)
    frame.loc[frame["readmitted_30d"] == 0, "days_to_readmission"] = np.nan
    return frame


def build_cohort(
    tables: FhirTables, cfg: CohortConfig, as_of: date | pd.Timestamp | None = None
) -> Cohort:
    """Index hospitalizations with labels, demographics, and a CONSORT-style exclusion flow."""
    flow: list[dict[str, object]] = []
    inpatient = tables.encounters[tables.encounters["class_code"] == INPATIENT_CLASS]
    flow.append({"step": "Inpatient encounters", "n": len(inpatient)})
    acute = acute_inpatient(tables.encounters, cfg.exclude_encounter_types)
    flow.append(
        {"step": "Acute inpatient encounters (non-acute settings removed)", "n": len(acute)}
    )
    episodes = build_episodes(
        tables.encounters, cfg.exclude_encounter_types, cfg.transfer_gap_hours
    )
    flow.append({"step": "Hospitalizations after merging transfer chains", "n": len(episodes)})
    if episodes.empty:
        return Cohort(index=episodes.assign(readmitted_30d=pd.Series(dtype=int)), flow=flow)

    episodes = enrich_episodes(label_readmissions(episodes, cfg.followup_days), tables.patients)

    adult = episodes["age"] >= cfg.min_age
    episodes = episodes[adult]
    flow.append({"step": f"Age >= {cfg.min_age} at admission", "n": len(episodes)})

    died = episodes["deceased_at"].notna() & (
        episodes["deceased_at"] <= episodes["discharge"] + pd.Timedelta(days=1)
    )
    episodes = episodes[~died]
    flow.append({"step": "Alive at discharge", "n": len(episodes)})

    if as_of is None:
        as_of = cfg.as_of_date
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else tables.encounters["end"].max()
    complete = episodes["discharge"] <= as_of_ts - pd.Timedelta(days=cfg.followup_days)
    episodes = episodes[complete]
    flow.append({"step": f"Full {cfg.followup_days}-day follow-up observable", "n": len(episodes)})

    episodes = episodes.sort_values("discharge", kind="stable")
    keep = [
        "episode_id",
        "patient_id",
        "admit",
        "discharge",
        "los_days",
        "n_encounters",
        "encounter_ids",
        "age",
        "age_band",
        "sex",
        "race",
        "ethnicity",
        "readmitted_30d",
        "days_to_readmission",
    ]
    index = episodes[keep].reset_index(drop=True)
    flow.append(
        {
            "step": "Index hospitalizations (readmitted within 30 days)",
            "n": len(index),
            "positives": int(index["readmitted_30d"].sum()),
        }
    )
    return Cohort(index=index, flow=flow, as_of=as_of_ts)


def enrich_episodes(episodes: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Attach demographics, age at admission, length of stay, and age band."""
    demographics = patients.drop_duplicates("patient_id").set_index("patient_id")[
        ["birth_date", "sex", "race", "ethnicity", "deceased_at"]
    ]
    frame = episodes.join(demographics, on="patient_id")
    frame["age"] = (frame["admit"] - frame["birth_date"]).dt.days / 365.25
    frame["los_days"] = (frame["discharge"] - frame["admit"]).dt.total_seconds() / 86400
    frame["age_band"] = frame["age"].map(age_band)
    return frame


class EncounterNotFoundError(LookupError):
    pass


def scoring_episode(
    tables: FhirTables, cfg: CohortConfig, encounter_id: str | None = None
) -> pd.DataFrame:
    """The hospitalization to score for one patient's record (used by the API).

    Uses the episode containing ``encounter_id`` if given, otherwise the most recent acute
    hospitalization. Returns a one-row frame shaped like ``Cohort.index`` (without labels).
    """
    episodes = build_episodes(
        tables.encounters, cfg.exclude_encounter_types, cfg.transfer_gap_hours
    )
    if episodes.empty:
        raise EncounterNotFoundError("no finished acute inpatient encounter found in the record")
    if encounter_id is not None:
        match = episodes[episodes["encounter_ids"].map(lambda ids: encounter_id in ids)]
        if match.empty:
            raise EncounterNotFoundError(
                f"encounter {encounter_id} is not a finished acute inpatient encounter"
            )
        episodes = match
    episode = episodes.sort_values("discharge").tail(1)
    return enrich_episodes(episode, tables.patients).reset_index(drop=True)

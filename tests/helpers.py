"""Builders for small hand-made FHIR tables used in unit tests."""

from __future__ import annotations

import pandas as pd

from readmit.fhir.flatten import DATETIME_COLUMNS, TABLE_COLUMNS, FhirTables


def _frame(name: str, rows) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=TABLE_COLUMNS[name])
    for column in DATETIME_COLUMNS[name]:
        frame[column] = pd.to_datetime(frame[column], format="ISO8601")
    if name == "observations":
        frame["value"] = pd.to_numeric(frame["value"])
    return frame


def tables_from(patients=(), encounters=(), conditions=(), observations=(), medications=()):
    return FhirTables(
        patients=_frame("patients", patients),
        encounters=_frame("encounters", encounters),
        conditions=_frame("conditions", conditions),
        observations=_frame("observations", observations),
        medications=_frame("medications", medications),
    )


def patient(pid="p1", birth="1940-01-01", sex="female", deceased=None):
    return {
        "patient_id": pid,
        "birth_date": birth,
        "sex": sex,
        "race": "White",
        "ethnicity": "Not Hispanic or Latino",
        "deceased_at": deceased,
    }


def encounter(eid, start, end, cls="IMP", display="Hospital admission (procedure)", pid="p1"):
    return {
        "encounter_id": eid,
        "patient_id": pid,
        "class_code": cls,
        "type_code": "x",
        "type_display": display,
        "status": "finished",
        "start": start,
        "end": end,
    }


def condition(cid, display, onset, recorded=None, abatement=None, pid="p1"):
    return {
        "condition_id": cid,
        "patient_id": pid,
        "encounter_id": None,
        "system": "snomed",
        "code": cid,
        "display": display,
        "clinical_status": "active",
        "onset": onset,
        "recorded": recorded or onset,
        "abatement": abatement,
    }


def truncate(tables: FhirTables, patient_id: str, moment: pd.Timestamp) -> FhirTables:
    """One patient's record as it existed at ``moment`` (future events removed)."""
    t = tables.for_patient(patient_id)
    encounters = t.encounters[t.encounters["start"] <= moment].copy()
    conditions = t.conditions[
        t.conditions["recorded"].fillna(t.conditions["onset"]) <= moment
    ].copy()
    conditions.loc[conditions["abatement"] > moment, "abatement"] = pd.NaT
    patients = t.patients.copy()
    patients.loc[patients["deceased_at"] > moment, "deceased_at"] = pd.NaT
    return FhirTables(
        patients=patients,
        encounters=encounters,
        conditions=conditions,
        observations=t.observations[t.observations["effective"] <= moment],
        medications=t.medications[t.medications["authored_on"] <= moment],
    )

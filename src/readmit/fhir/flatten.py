"""Flatten HL7 FHIR R4 resources into analysis-ready tables.

Inputs supported:
* FHIR Bulk Data (Flat FHIR) exports: one ``<ResourceType>.ndjson`` (optionally ``.gz``) file
  per resource type, one resource per line. This is the format population-scale sources such as
  payer and EHR bulk APIs produce.
* Bundles: JSON files containing a ``Bundle`` (Synthea's default output) or single resources.
  The API also uses this path to score one patient's record.

References may be ``urn:uuid:<id>`` (Bundles) or ``<Type>/<id>`` (bulk exports); both resolve to
the bare logical id. Timestamps are converted to naive UTC.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from readmit.fhir.codes import BLOOD_PRESSURE_PANEL

SUPPORTED_TYPES = ("Patient", "Encounter", "Condition", "Observation", "MedicationRequest")
RACE_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
ETHNICITY_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"

TABLE_COLUMNS = {
    "patients": ["patient_id", "birth_date", "sex", "race", "ethnicity", "deceased_at"],
    "encounters": [
        "encounter_id",
        "patient_id",
        "class_code",
        "type_code",
        "type_display",
        "status",
        "start",
        "end",
    ],
    "conditions": [
        "condition_id",
        "patient_id",
        "encounter_id",
        "system",
        "code",
        "display",
        "clinical_status",
        "onset",
        "recorded",
        "abatement",
    ],
    "observations": [
        "observation_id",
        "patient_id",
        "encounter_id",
        "code",
        "display",
        "value",
        "unit",
        "effective",
    ],
    "medications": [
        "medication_request_id",
        "patient_id",
        "encounter_id",
        "code",
        "display",
        "status",
        "authored_on",
    ],
}
DATETIME_COLUMNS = {
    "patients": ["birth_date", "deceased_at"],
    "encounters": ["start", "end"],
    "conditions": ["onset", "recorded", "abatement"],
    "observations": ["effective"],
    "medications": ["authored_on"],
}


def ref_id(reference: dict[str, Any] | str | None) -> str | None:
    """Resolve ``{"reference": "urn:uuid:x"}`` / ``"Patient/x"`` to the bare id ``x``."""
    if reference is None:
        return None
    value = reference.get("reference") if isinstance(reference, dict) else reference
    if not value:
        return None
    if value.startswith("urn:uuid:"):
        return value[len("urn:uuid:") :]
    return value.rsplit("/", 1)[-1]


def _coding(concept: dict[str, Any] | None) -> dict[str, Any]:
    if not concept:
        return {}
    codings = concept.get("coding") or [{}]
    first = dict(codings[0])
    first.setdefault("display", concept.get("text"))
    return first


def _us_core_category(resource: dict[str, Any], url: str) -> str | None:
    for extension in resource.get("extension", []):
        if extension.get("url") != url:
            continue
        text = None
        for sub in extension.get("extension", []):
            if sub.get("url") == "ombCategory":
                return sub.get("valueCoding", {}).get("display")
            if sub.get("url") == "text":
                text = sub.get("valueString")
        return text
    return None


def _patient(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": r["id"],
        "birth_date": r.get("birthDate"),
        "sex": r.get("gender", "unknown"),
        "race": _us_core_category(r, RACE_URL) or "Unknown",
        "ethnicity": _us_core_category(r, ETHNICITY_URL) or "Unknown",
        "deceased_at": r.get("deceasedDateTime"),
    }


def _encounter(r: dict[str, Any]) -> dict[str, Any]:
    encounter_type = _coding((r.get("type") or [None])[0])
    period = r.get("period", {})
    return {
        "encounter_id": r["id"],
        "patient_id": ref_id(r.get("subject")),
        "class_code": (r.get("class") or {}).get("code"),
        "type_code": encounter_type.get("code"),
        "type_display": encounter_type.get("display") or "",
        "status": r.get("status"),
        "start": period.get("start"),
        "end": period.get("end"),
    }


def _condition(r: dict[str, Any]) -> dict[str, Any]:
    code = _coding(r.get("code"))
    return {
        "condition_id": r["id"],
        "patient_id": ref_id(r.get("subject")),
        "encounter_id": ref_id(r.get("encounter")),
        "system": code.get("system"),
        "code": code.get("code"),
        "display": code.get("display") or "",
        "clinical_status": _coding(r.get("clinicalStatus")).get("code"),
        "onset": r.get("onsetDateTime") or r.get("recordedDate"),
        "recorded": r.get("recordedDate") or r.get("onsetDateTime"),
        "abatement": r.get("abatementDateTime"),
    }


def _observations(r: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per numeric value; blood-pressure panels expand into their components."""
    code = _coding(r.get("code"))
    base = {
        "patient_id": ref_id(r.get("subject")),
        "encounter_id": ref_id(r.get("encounter")),
        "effective": r.get("effectiveDateTime") or r.get("issued"),
    }
    rows = []
    if "valueQuantity" in r:
        rows.append(
            {
                "observation_id": r["id"],
                "code": code.get("code"),
                "display": code.get("display"),
                "value": r["valueQuantity"].get("value"),
                "unit": r["valueQuantity"].get("unit"),
                **base,
            }
        )
    if code.get("code") == BLOOD_PRESSURE_PANEL or "component" in r:
        for component in r.get("component", []):
            if "valueQuantity" not in component:
                continue
            component_code = _coding(component.get("code"))
            rows.append(
                {
                    "observation_id": r["id"],
                    "code": component_code.get("code"),
                    "display": component_code.get("display"),
                    "value": component["valueQuantity"].get("value"),
                    "unit": component["valueQuantity"].get("unit"),
                    **base,
                }
            )
    return rows


def _medication_request(
    r: dict[str, Any], medications: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if "medicationCodeableConcept" in r:
        code = _coding(r["medicationCodeableConcept"])
    else:
        reference = r.get("medicationReference") or {}
        code = medications.get(ref_id(reference) or "", {}) or {"display": reference.get("display")}
    return {
        "medication_request_id": r["id"],
        "patient_id": ref_id(r.get("subject")),
        "encounter_id": ref_id(r.get("encounter")),
        "code": code.get("code"),
        "display": code.get("display") or "",
        "status": r.get("status"),
        "authored_on": r.get("authoredOn"),
    }


@dataclass
class FhirTables:
    patients: pd.DataFrame
    encounters: pd.DataFrame
    conditions: pd.DataFrame
    observations: pd.DataFrame
    medications: pd.DataFrame

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            getattr(self, f.name).to_parquet(directory / f"{f.name}.parquet", index=False)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> FhirTables:
        """Load Parquet tables written by ``save`` or by the Spark job (normalized to naive UTC)."""
        directory = Path(directory)
        tables = {}
        for f in fields(cls):
            frame = pd.read_parquet(directory / f"{f.name}.parquet")
            for column in DATETIME_COLUMNS[f.name]:
                if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                    frame[column] = frame[column].dt.tz_convert("UTC").dt.tz_localize(None)
            tables[f.name] = frame[TABLE_COLUMNS[f.name]]
        return cls(**tables)

    def for_patient(self, patient_id: str) -> FhirTables:
        return FhirTables(
            **{
                f.name: getattr(self, f.name)[getattr(self, f.name)["patient_id"] == patient_id]
                for f in fields(self)
            }
        )

    def summary(self) -> dict[str, int]:
        return {f.name: len(getattr(self, f.name)) for f in fields(self)}


def _read_json_file(path: Path) -> Iterator[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry", []):
            if "resource" in entry:
                yield entry["resource"]
    else:
        yield data


def _read_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


NEEDED_TYPES = {*SUPPORTED_TYPES, "Medication"}


def _skippable_bulk_file(file: Path) -> bool:
    """Bulk exports name files by resource type (``Claim.ndjson``, ``Claim.001.ndjson``).

    Files for resource types the pipeline never uses are skipped without parsing, which matters
    for real exports where claims and provenance can dwarf the clinical resources.
    """
    token = file.name.split(".")[0]
    return token[:1].isupper() and token.isalpha() and token not in NEEDED_TYPES


def iter_resources(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield resources from an NDJSON/JSON file or a directory of them."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FHIR input not found: {path}")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        name = file.name.lower()
        if name.endswith((".ndjson", ".ndjson.gz")):
            if _skippable_bulk_file(file):
                continue
            yield from _read_ndjson(file)
        elif name.endswith(".json"):
            yield from _read_json_file(file)


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=TABLE_COLUMNS[name])
    for column in DATETIME_COLUMNS[name]:
        parsed = pd.to_datetime(frame[column], utc=True, format="ISO8601", errors="coerce")
        frame[column] = parsed.dt.tz_convert(None)
    if name == "observations":
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def flatten_resources(resources: Iterable[dict[str, Any]]) -> FhirTables:
    """Flatten an iterable of FHIR resources into tables."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_COLUMNS}
    medication_catalog: dict[str, dict[str, Any]] = {}
    pending_requests: list[dict[str, Any]] = []
    for resource in resources:
        kind = resource.get("resourceType")
        if kind == "Patient":
            rows["patients"].append(_patient(resource))
        elif kind == "Encounter":
            rows["encounters"].append(_encounter(resource))
        elif kind == "Condition":
            rows["conditions"].append(_condition(resource))
        elif kind == "Observation":
            rows["observations"].extend(_observations(resource))
        elif kind == "Medication":
            medication_catalog[resource["id"]] = _coding(resource.get("code"))
        elif kind == "MedicationRequest":
            pending_requests.append(resource)
    rows["medications"] = [_medication_request(r, medication_catalog) for r in pending_requests]
    tables = {name: _frame(name, data) for name, data in rows.items()}
    return FhirTables(**tables)


def flatten_path(path: str | Path) -> FhirTables:
    return flatten_resources(iter_resources(path))


def flatten_bundle(bundle: dict[str, Any]) -> FhirTables:
    """Flatten an in-memory Bundle (used by the API for single-patient scoring)."""
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("expected a FHIR Bundle resource")
    return flatten_resources(e["resource"] for e in bundle.get("entry", []) if "resource" in e)

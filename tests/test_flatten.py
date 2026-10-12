"""FHIR parsing contract tests against real Synthea output and the synthetic generator."""

import gzip
import json

import pandas as pd
import pytest

from readmit.fhir.flatten import FhirTables, flatten_bundle, flatten_path, ref_id
from readmit.fhir.synthetic import write_population


def test_reference_forms():
    assert ref_id({"reference": "urn:uuid:abc-1"}) == "abc-1"
    assert ref_id({"reference": "Patient/abc-1"}) == "abc-1"
    assert ref_id("Encounter/xyz") == "xyz"
    assert ref_id(None) is None


def test_synthea_contract(synthea_tables):
    t = synthea_tables
    assert t.summary()["patients"] == 3
    assert set(t.encounters["class_code"]) >= {"AMB", "EMER", "IMP"}
    assert t.patients["race"].ne("Unknown").all()  # US Core race extension parsed
    # blood-pressure panels expand into systolic/diastolic component rows
    assert {"8480-6", "8462-4"} <= set(t.observations["code"])
    for column in ("start", "end"):
        assert pd.api.types.is_datetime64_any_dtype(t.encounters[column])
    assert t.encounters["patient_id"].isin(t.patients["patient_id"]).all()


def test_ndjson_and_bundle_formats_agree(tmp_path):
    write_population(tmp_path / "nd", 25, seed=3, fmt="ndjson")
    write_population(tmp_path / "bd", 25, seed=3, fmt="bundle")
    a, b = flatten_path(tmp_path / "nd"), flatten_path(tmp_path / "bd")
    assert a.summary() == b.summary()
    pd.testing.assert_series_equal(
        a.encounters["patient_id"].value_counts().sort_index(),
        b.encounters["patient_id"].value_counts().sort_index(),
    )


def test_gzip_ndjson(tmp_path):
    write_population(tmp_path / "plain", 5, seed=1)
    gz_dir = tmp_path / "gz"
    gz_dir.mkdir()
    for path in (tmp_path / "plain").glob("*.ndjson"):
        with gzip.open(gz_dir / f"{path.name}.gz", "wt", encoding="utf-8") as handle:
            handle.write(path.read_text())
    assert flatten_path(gz_dir).summary() == flatten_path(tmp_path / "plain").summary()


def test_medication_reference_resolution():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "gender": "female",
                    "birthDate": "1950-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "m1",
                    "code": {"coding": [{"system": "rxnorm", "code": "123", "display": "Drug X"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "mr1",
                    "status": "active",
                    "subject": {"reference": "urn:uuid:p1"},
                    "medicationReference": {"reference": "urn:uuid:m1"},
                    "authoredOn": "2024-01-01T00:00:00+00:00",
                }
            },
        ],
    }
    meds = flatten_bundle(bundle).medications
    assert meds.loc[0, "code"] == "123" and meds.loc[0, "display"] == "Drug X"


def test_rejects_non_bundle_and_missing_path(tmp_path):
    with pytest.raises(ValueError):
        flatten_bundle({"resourceType": "Patient"})
    with pytest.raises(FileNotFoundError):
        flatten_path(tmp_path / "missing")


def test_save_load_roundtrip(tmp_path, synthea_tables):
    loaded = FhirTables.load(synthea_tables.save(tmp_path / "tables"))
    assert loaded.summary() == synthea_tables.summary()
    pd.testing.assert_frame_equal(
        loaded.encounters.reset_index(drop=True), synthea_tables.encounters.reset_index(drop=True)
    )


def test_manifest_is_ignored(tmp_path):
    write_population(tmp_path, 5, seed=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["patients"] == 5
    assert flatten_path(tmp_path).summary()["patients"] == 5


def test_unused_bulk_files_are_skipped(tmp_path):
    write_population(tmp_path, 5, seed=2)
    (tmp_path / "Claim.ndjson").write_text("not valid json\n")  # would raise if it were parsed
    (tmp_path / "Provenance.001.ndjson").write_text("{broken\n")
    assert flatten_path(tmp_path).summary()["patients"] == 5

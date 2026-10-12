"""Synthetic HL7 FHIR R4 population with a documented 30-day readmission mechanism.

Resource structure mirrors Synthea output (same resource types, code systems, codes, displays,
and field layout); the parser is contract-tested against real Synthea bundles in
``tests/fixtures/synthea``. Synthea is excellent for interoperability testing, but its 30-day
readmission rate is only ~4% and is not driven by clinical risk factors. Here readmissions
follow factors reported in the readmission literature (age, comorbidity burden, heart failure,
COPD, CKD, prior utilization, length of stay, polypharmacy, social needs, emergency admission)
plus an *unobserved* frailty term, so no model can be perfect. The ground-truth coefficients
are written to ``manifest.json``.

The data also exercises real-world edge cases: ICU transfer chains (contiguous inpatient
encounters), skilled-nursing stays after discharge, in-hospital and post-discharge deaths,
conditions that begin mid-history, mixed date formats, and missing labs.

Everything is synthetic. No real patient data is used or produced.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from readmit.fhir.codes import ICD10_SYSTEM, LOINC_SYSTEM, RXNORM_SYSTEM, SNOMED_SYSTEM

WINDOW_START = datetime(2016, 1, 1)
WINDOW_END = datetime(2025, 12, 31, 23, 59)
GENERATOR_VERSION = "1.0"

RACES = [
    ("2106-3", "White", 0.62),
    ("2054-5", "Black or African American", 0.14),
    ("2028-9", "Asian", 0.06),
    ("1002-5", "American Indian or Alaska Native", 0.01),
    ("2076-8", "Native Hawaiian or Other Pacific Islander", 0.005),
    ("2131-1", "Other", 0.165),
]
ETHNICITIES = [("2135-2", "Hispanic or Latino", 0.18), ("2186-5", "Not Hispanic or Latino", 0.82)]

CONDITIONS: dict[str, tuple[str, str, str]] = {
    "hypertension": (SNOMED_SYSTEM, "59621000", "Essential hypertension (disorder)"),
    "diabetes": (SNOMED_SYSTEM, "44054006", "Diabetes mellitus type 2 (disorder)"),
    "diabetic_neuropathy": (
        SNOMED_SYSTEM,
        "368581000119106",
        "Neuropathy due to type 2 diabetes mellitus (disorder)",
    ),
    "diabetic_kidney": (
        SNOMED_SYSTEM,
        "127013003",
        "Disorder of kidney due to diabetes mellitus (disorder)",
    ),
    "chf": (SNOMED_SYSTEM, "88805009", "Chronic congestive heart failure (disorder)"),
    "copd": (SNOMED_SYSTEM, "185086009", "Chronic obstructive bronchitis (disorder)"),
    "emphysema": (SNOMED_SYSTEM, "87433001", "Pulmonary emphysema (disorder)"),
    "ckd2": (SNOMED_SYSTEM, "431856006", "Chronic kidney disease stage 2 (disorder)"),
    "ckd3": (SNOMED_SYSTEM, "433144002", "Chronic kidney disease stage 3 (disorder)"),
    "ckd4": (SNOMED_SYSTEM, "431857002", "Chronic kidney disease stage 4 (disorder)"),
    "esrd": (SNOMED_SYSTEM, "46177005", "End-stage renal disease (disorder)"),
    "mi": (SNOMED_SYSTEM, "22298006", "Myocardial infarction (disorder)"),
    "stroke": (SNOMED_SYSTEM, "230690007", "Stroke"),
    "alzheimers": (SNOMED_SYSTEM, "26929004", "Alzheimer's disease (disorder)"),
    "lung_cancer": (SNOMED_SYSTEM, "254637007", "Non-small cell lung cancer (disorder)"),
    "breast_cancer": (SNOMED_SYSTEM, "254837009", "Malignant neoplasm of breast (disorder)"),
    "colon_cancer": (SNOMED_SYSTEM, "363406005", "Malignant tumor of colon"),
    "metastatic": (ICD10_SYSTEM, "C79.89", "Secondary malignant neoplasm of other specified sites"),
    "depression": (SNOMED_SYSTEM, "370143000", "Major depression disorder"),
    "obesity": (SNOMED_SYSTEM, "162864005", "Body mass index 30+ - obesity (finding)"),
    "anemia": (SNOMED_SYSTEM, "271737000", "Anemia (disorder)"),
    "social_isolation": (SNOMED_SYSTEM, "422650009", "Social isolation (finding)"),
    "limited_social_contact": (SNOMED_SYSTEM, "423315002", "Limited social contact (finding)"),
    "unemployed": (SNOMED_SYSTEM, "73438004", "Unemployed (finding)"),
    "housing": (SNOMED_SYSTEM, "105531004", "Housing unsatisfactory (finding)"),
    "transportation": (SNOMED_SYSTEM, "713458007", "Lack of access to transportation (finding)"),
    "stress": (SNOMED_SYSTEM, "73595000", "Stress (finding)"),
    "pneumonia": (SNOMED_SYSTEM, "233604007", "Pneumonia (disorder)"),
    "sinusitis": (SNOMED_SYSTEM, "444814009", "Viral sinusitis (disorder)"),
    "bronchitis": (SNOMED_SYSTEM, "10509002", "Acute bronchitis (disorder)"),
}
SDOH_KEYS = (
    "social_isolation",
    "limited_social_contact",
    "unemployed",
    "housing",
    "transportation",
)

MEDICATIONS: dict[str, tuple[str, str]] = {
    "lisinopril": ("314076", "lisinopril 10 MG Oral Tablet"),
    "amlodipine": ("308136", "amLODIPine 2.5 MG Oral Tablet"),
    "hctz": ("310798", "Hydrochlorothiazide 25 MG Oral Tablet"),
    "metformin": ("860975", "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet"),
    "insulin": (
        "106892",
        "insulin isophane, human 70 UNT/ML / insulin, regular, human 30 UNT/ML Injectable "
        "Suspension [Humulin]",
    ),
    "furosemide": ("313988", "Furosemide 40 MG Oral Tablet"),
    "carvedilol": ("200033", "carvedilol 25 MG Oral Tablet"),
    "metoprolol": ("866412", "24 HR metoprolol succinate 100 MG Extended Release Oral Tablet"),
    "atorvastatin": ("617311", "atorvastatin 40 MG Oral Tablet"),
    "aspirin": ("243670", "Aspirin 81 MG Oral Tablet"),
    "clopidogrel": ("309362", "Clopidogrel 75 MG Oral Tablet"),
    "albuterol": (
        "859088",
        "NDA020983 200 ACTUAT albuterol 0.09 MG/ACTUAT Metered Dose Inhaler [Ventolin]",
    ),
    "fluticasone_salmeterol": (
        "896209",
        "60 ACTUAT Fluticasone propionate 0.25 MG/ACTUAT / salmeterol 0.05 MG/ACTUAT Dry Powder "
        "Inhaler",
    ),
    "sertraline": ("312938", "Sertraline 100 MG Oral Tablet"),
    "donepezil": ("997223", "Donepezil hydrochloride 10 MG Oral Tablet"),
    "losartan": ("979492", "losartan potassium 50 MG Oral Tablet"),
    "amox_clav": ("562251", "Amoxicillin 250 MG / Clavulanate 125 MG Oral Tablet"),
}
CONDITION_MEDS = {
    "hypertension": ("lisinopril", "amlodipine", "hctz"),
    "diabetes": ("metformin", "insulin"),
    "chf": ("furosemide", "carvedilol", "lisinopril"),
    "copd": ("albuterol", "fluticasone_salmeterol"),
    "mi": ("aspirin", "clopidogrel", "atorvastatin", "metoprolol"),
    "stroke": ("aspirin", "atorvastatin"),
    "depression": ("sertraline",),
    "alzheimers": ("donepezil",),
    "ckd": ("losartan",),
}
ENCOUNTER_TYPES = {
    "checkup": ("AMB", "185349003", "Encounter for check up (procedure)"),
    "problem": ("AMB", "185347001", "Encounter for problem (procedure)"),
    "ed": ("EMER", "50849002", "Emergency room admission (procedure)"),
    "admission": ("IMP", "32485007", "Hospital admission (procedure)"),
    "icu": ("IMP", "397821002", "Patient transfer to intensive care unit (procedure)"),
    "ward": ("IMP", "305342007", "Admission to ward (procedure)"),
    "snf": ("IMP", "449411000124106", "Admission to skilled nursing facility (procedure)"),
}

# Ground-truth readmission mechanism (log-odds scale). Documented in manifest.json and the docs.
READMISSION_COEFFICIENTS: dict[str, float] = {
    "intercept": -4.4,
    "age_per_year_over_65": 0.022,
    "charlson_point": 0.10,
    "heart_failure": 0.38,
    "copd": 0.25,
    "ckd_stage3_plus": 0.28,
    "diabetes_complications": 0.20,
    "log1p_length_of_stay": 0.32,
    "prior_admissions_12m_capped4": 0.34,
    "ed_visits_6m_capped4": 0.17,
    "polypharmacy": 0.22,
    "social_needs": 0.30,
    "emergency_admission": 0.20,
    "weekend_discharge": 0.10,
    "egfr_below_45": 0.25,
    "sodium_below_135": 0.22,
    "heart_failure_x_age_75_plus": 0.45,
    "unobserved_frailty": 0.55,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S+00:00")


@dataclass
class _Patient:
    pid: str
    birth: datetime
    sex: str
    race: tuple[str, str, float]
    ethnicity: tuple[str, str, float]
    frailty: float
    onsets: dict[str, datetime] = field(default_factory=dict)
    med_lists: dict[str, list[str]] = field(default_factory=dict)
    deceased: datetime | None = None

    def age(self, moment: datetime) -> float:
        return (moment - self.birth).days / 365.25

    def active(self, key: str, moment: datetime) -> bool:
        onset = self.onsets.get(key)
        return onset is not None and onset <= moment

    def ckd_stage(self, moment: datetime) -> int:
        for stage, key in ((5, "esrd"), (4, "ckd4"), (3, "ckd3"), (2, "ckd2")):
            if self.active(key, moment):
                return stage
        return 0

    def charlson(self, moment: datetime) -> int:
        a = lambda k: self.active(k, moment)  # noqa: E731
        score = 0
        score += a("mi") + a("chf") + a("stroke") + a("alzheimers") + (a("copd") or a("emphysema"))
        if a("diabetic_neuropathy") or a("diabetic_kidney"):
            score += 2
        elif a("diabetes"):
            score += 1
        score += 2 * (self.ckd_stage(moment) >= 3)
        if a("metastatic"):
            score += 6
        elif a("lung_cancer") or a("breast_cancer") or a("colon_cancer"):
            score += 2
        return int(score)

    def social_needs(self, moment: datetime) -> bool:
        return any(self.active(k, moment) for k in SDOH_KEYS)

    def meds_at(self, moment: datetime) -> list[str]:
        meds: list[str] = []
        for key, chosen in self.med_lists.items():
            if key == "ckd":
                present = self.ckd_stage(moment) >= 3
            elif key == "copd":
                present = self.active("copd", moment) or self.active("emphysema", moment)
            else:
                present = self.active(key, moment)
            if present:
                meds.extend(chosen)
        return list(dict.fromkeys(meds))


class _Builder:
    """Accumulates FHIR resources for one patient in NDJSON or Bundle reference style."""

    def __init__(self, rng: np.random.Generator, mode: Literal["ndjson", "bundle"]) -> None:
        self.rng = rng
        self.mode = mode
        self.resources: list[dict[str, Any]] = []

    def new_id(self) -> str:
        return str(uuid.UUID(bytes=self.rng.bytes(16), version=4))

    def ref(self, kind: str, rid: str) -> dict[str, str]:
        return {"reference": f"urn:uuid:{rid}" if self.mode == "bundle" else f"{kind}/{rid}"}

    def add(self, resource: dict[str, Any]) -> str:
        self.resources.append(resource)
        return resource["id"]

    def encounter(self, pid: str, kind: str, start: datetime, end: datetime) -> str:
        cls, code, display = ENCOUNTER_TYPES[kind]
        return self.add(
            {
                "resourceType": "Encounter",
                "id": self.new_id(),
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": cls,
                },
                "type": [
                    {
                        "coding": [{"system": SNOMED_SYSTEM, "code": code, "display": display}],
                        "text": display,
                    }
                ],
                "subject": self.ref("Patient", pid),
                "period": {"start": _ts(start), "end": _ts(end)},
            }
        )

    def condition(
        self,
        pid: str,
        key: str,
        onset: datetime,
        eid: str | None,
        abatement: datetime | None = None,
        recorded: datetime | None = None,
    ) -> None:
        system, code, display = CONDITIONS[key]
        resource: dict[str, Any] = {
            "resourceType": "Condition",
            "id": self.new_id(),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "resolved" if abatement else "active",
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                    }
                ]
            },
            "code": {
                "coding": [{"system": system, "code": code, "display": display}],
                "text": display,
            },
            "subject": self.ref("Patient", pid),
            "onsetDateTime": _ts(onset),
            "recordedDate": _ts(recorded or onset),
        }
        if eid:
            resource["encounter"] = self.ref("Encounter", eid)
        if abatement:
            resource["abatementDateTime"] = _ts(abatement)
        self.add(resource)

    def observation(
        self,
        pid: str,
        eid: str,
        moment: datetime,
        code: str,
        display: str,
        value: float,
        unit: str,
        category: str = "vital-signs",
    ) -> None:
        self.add(
            {
                "resourceType": "Observation",
                "id": self.new_id(),
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": category,
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [{"system": LOINC_SYSTEM, "code": code, "display": display}],
                    "text": display,
                },
                "subject": self.ref("Patient", pid),
                "encounter": self.ref("Encounter", eid),
                "effectiveDateTime": _ts(moment),
                "valueQuantity": {
                    "value": round(float(value), 1),
                    "unit": unit,
                    "system": "http://unitsofmeasure.org",
                    "code": unit,
                },
            }
        )

    def blood_pressure(
        self, pid: str, eid: str, moment: datetime, systolic: float, diastolic: float
    ) -> None:
        def component(code: str, display: str, value: float) -> dict[str, Any]:
            return {
                "code": {"coding": [{"system": LOINC_SYSTEM, "code": code, "display": display}]},
                "valueQuantity": {
                    "value": round(value),
                    "unit": "mm[Hg]",
                    "system": "http://unitsofmeasure.org",
                    "code": "mm[Hg]",
                },
            }

        self.add(
            {
                "resourceType": "Observation",
                "id": self.new_id(),
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "vital-signs",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": LOINC_SYSTEM,
                            "code": "85354-9",
                            "display": "Blood pressure panel with all children optional",
                        }
                    ]
                },
                "subject": self.ref("Patient", pid),
                "encounter": self.ref("Encounter", eid),
                "effectiveDateTime": _ts(moment),
                "component": [
                    component("8462-4", "Diastolic Blood Pressure", diastolic),
                    component("8480-6", "Systolic Blood Pressure", systolic),
                ],
            }
        )

    def medication(self, pid: str, eid: str, moment: datetime, key: str, status: str) -> None:
        code, display = MEDICATIONS[key]
        self.add(
            {
                "resourceType": "MedicationRequest",
                "id": self.new_id(),
                "status": status,
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{"system": RXNORM_SYSTEM, "code": code, "display": display}],
                    "text": display,
                },
                "subject": self.ref("Patient", pid),
                "encounter": self.ref("Encounter", eid),
                "authoredOn": _ts(moment),
            }
        )


class PopulationGenerator:
    def __init__(
        self, seed: int = 42, drift: bool = False, mode: Literal["ndjson", "bundle"] = "ndjson"
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.mode = mode
        self.stats: dict[str, int] = defaultdict(int)

    def _p(self, base: float, slope: float, age: float, frailty: float) -> float:
        return _sigmoid(base + slope * (age - 65) + 0.3 * frailty)

    def _onset(self, birth: datetime) -> datetime:
        rng = self.rng
        if rng.random() < 0.6:
            onset = WINDOW_START - timedelta(days=float(rng.uniform(30, 8 * 365)))
        else:
            onset = WINDOW_START + timedelta(days=float(rng.uniform(0, 9 * 365)))
        return max(onset, birth + timedelta(days=18 * 365))

    def _new_patient(self, builder: _Builder) -> _Patient:
        rng = self.rng
        band = rng.choice(
            4, p=[0.15, 0.25, 0.35, 0.25] if not self.drift else [0.08, 0.2, 0.37, 0.35]
        )
        low, high = [(18, 45), (45, 65), (65, 80), (80, 95)][band]
        age0 = rng.uniform(low, high)
        birth = WINDOW_START - timedelta(days=float(age0 * 365.25))
        race = RACES[rng.choice(len(RACES), p=[r[2] for r in RACES])]
        ethnicity = ETHNICITIES[rng.choice(2, p=[e[2] for e in ETHNICITIES])]
        frailty = float(rng.normal(0.5 if self.drift else 0.0, 1.0))
        patient = _Patient(
            builder.new_id(),
            birth,
            "female" if rng.random() < 0.52 else "male",
            race,
            ethnicity,
            frailty,
        )

        def maybe(key: str, prob: float) -> None:
            if rng.random() < prob:
                patient.onsets[key] = self._onset(birth)

        f = frailty
        maybe("hypertension", self._p(0.3, 0.05, age0, f))
        maybe("diabetes", self._p(-1.0, 0.03, age0, f))
        if "diabetes" in patient.onsets and rng.random() < 0.4:
            comp = "diabetic_neuropathy" if rng.random() < 0.5 else "diabetic_kidney"
            patient.onsets[comp] = patient.onsets["diabetes"] + timedelta(
                days=float(rng.uniform(200, 2000))
            )
        maybe("chf", self._p(-2.4, 0.06, age0, f))
        maybe("copd" if rng.random() < 0.6 else "emphysema", self._p(-2.2, 0.03, age0, f))
        if rng.random() < self._p(-1.6, 0.05, age0, f):
            stage = rng.choice(["ckd2", "ckd3", "ckd4", "esrd"], p=[0.35, 0.45, 0.15, 0.05])
            patient.onsets[str(stage)] = self._onset(birth)
        maybe("mi", self._p(-2.6, 0.04, age0, f))
        maybe("stroke", self._p(-3.0, 0.05, age0, f))
        maybe("alzheimers", self._p(-4.5, 0.12, age0, f))
        if rng.random() < self._p(-3.0, 0.03, age0, f):
            options = ["lung_cancer", "colon_cancer"] + (
                ["breast_cancer"] if patient.sex == "female" else []
            )
            cancer = str(rng.choice(options))
            patient.onsets[cancer] = self._onset(birth)
            if rng.random() < 0.25:
                patient.onsets["metastatic"] = patient.onsets[cancer] + timedelta(
                    days=float(rng.uniform(100, 900))
                )
        maybe("depression", 0.12)
        maybe("obesity", 0.30)
        maybe("anemia", 0.10)
        sdoh_shift = 1.4 if self.drift else 1.0
        maybe("social_isolation", min(0.10 * sdoh_shift + 0.04 * (age0 > 75), 0.9))
        maybe("limited_social_contact", 0.08 * sdoh_shift)
        if age0 < 62:
            maybe("unemployed", 0.10 * sdoh_shift)
        maybe("housing", 0.04 * sdoh_shift)
        maybe("transportation", 0.05 * sdoh_shift)
        maybe("stress", 0.20)

        for key, options in CONDITION_MEDS.items():
            present = (
                key in patient.onsets
                or (key == "ckd" and patient.ckd_stage(WINDOW_END) >= 3)
                or (key == "copd" and "emphysema" in patient.onsets)
            )
            if present:
                chosen = [m for m in options if rng.random() < 0.7] or [options[0]]
                patient.med_lists[key] = chosen
        return patient

    def _vitals_and_labs(
        self, b: _Builder, p: _Patient, eid: str, moment: datetime, labs: bool
    ) -> dict[str, float]:
        rng = self.rng
        systolic = rng.normal(126 + 10 * p.active("hypertension", moment), 14)
        b.blood_pressure(p.pid, eid, moment, systolic, rng.normal(80, 9))
        b.observation(
            p.pid,
            eid,
            moment,
            "8867-4",
            "Heart rate",
            rng.normal(76 + 8 * p.active("chf", moment), 11),
            "/min",
        )
        b.observation(
            p.pid,
            eid,
            moment,
            "39156-5",
            "Body mass index (BMI) [Ratio]",
            rng.normal(34 if p.active("obesity", moment) else 26.5, 3.5),
            "kg/m2",
        )
        values: dict[str, float] = {}
        if not labs:
            return values
        stage = p.ckd_stage(moment)
        egfr = max(
            rng.normal(
                {0: 86, 2: 74, 3: 44, 4: 22, 5: 10}[stage], {0: 12, 2: 8, 3: 8, 4: 4, 5: 3}[stage]
            ),
            4,
        )
        creatinine = max(0.9 * (90 / egfr) ** 0.9 + rng.normal(0, 0.08), 0.4)
        diabetic = p.active("diabetes", moment)
        sodium = rng.normal(135.5 if p.active("chf", moment) else 139.5, 3.5)
        b.observation(
            p.pid,
            eid,
            moment,
            "33914-3",
            "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum or "
            "Plasma by Creatinine-based formula (MDRD)",
            egfr,
            "mL/min/{1.73_m2}",
            "laboratory",
        )
        b.observation(
            p.pid,
            eid,
            moment,
            "38483-4",
            "Creatinine [Mass/volume] in Blood",
            creatinine,
            "mg/dL",
            "laboratory",
        )
        b.observation(
            p.pid,
            eid,
            moment,
            "2339-0",
            "Glucose [Mass/volume] in Blood",
            rng.normal(165 if diabetic else 98, 35 if diabetic else 12),
            "mg/dL",
            "laboratory",
        )
        b.observation(
            p.pid,
            eid,
            moment,
            "2947-0",
            "Sodium [Moles/volume] in Blood",
            sodium,
            "mmol/L",
            "laboratory",
        )
        if diabetic or rng.random() < 0.15:
            b.observation(
                p.pid,
                eid,
                moment,
                "4548-4",
                "Hemoglobin A1c/Hemoglobin.total in Blood",
                rng.normal(7.9 if diabetic else 5.4, 1.1 if diabetic else 0.3),
                "%",
                "laboratory",
            )
        values.update(egfr=egfr, sodium=sodium)
        return values

    def _simulate(self, b: _Builder) -> _Patient:
        rng = self.rng
        p = self._new_patient(b)
        events: list[tuple[datetime, str]] = []
        year = WINDOW_START
        while year < WINDOW_END:
            if rng.random() < 0.85:
                events.append((year + timedelta(days=float(rng.uniform(0, 364))), "checkup"))
            year = year.replace(year=year.year + 1)
        n_major = sum(
            k in p.onsets for k in ("chf", "copd", "emphysema", "diabetes", "mi", "stroke")
        )
        span_years = (WINDOW_END - WINDOW_START).days / 365.25
        for _ in range(rng.poisson((0.4 + 0.3 * n_major) * span_years)):
            events.append(
                (
                    WINDOW_START + timedelta(days=float(rng.uniform(0, span_years * 365.25))),
                    "problem",
                )
            )
        ed_rate = math.exp(
            -1.7 + 0.22 * n_major + 0.5 * p.frailty + 0.35 * p.social_needs(WINDOW_END)
        )
        ed_rate *= 1.5 if self.drift else 1.0
        ed_times = sorted(
            WINDOW_START + timedelta(days=float(rng.uniform(0, span_years * 365.25)))
            for _ in range(rng.poisson(ed_rate * span_years))
        )
        events.extend((t, "ed") for t in ed_times)

        admissions: list[tuple[datetime, datetime]] = []
        t, forced = WINDOW_START, None
        c = READMISSION_COEFFICIENTS
        while True:
            if forced is not None:
                admit, via_ed, forced = forced, rng.random() < 0.75, None
            else:
                rate = math.exp(
                    -2.0 + 0.22 * p.charlson(t) + 0.02 * (p.age(t) - 65) + 0.55 * p.frailty
                )
                admit = t + timedelta(days=float(rng.exponential(365.25 / rate)))
                via_ed = rng.random() < 0.6
            if admit > WINDOW_END - timedelta(days=1):
                break
            charlson = p.charlson(admit)
            los = float(
                np.clip(
                    rng.lognormal(math.log(2.8) + 0.07 * charlson + 0.18 * p.frailty, 0.55), 0.3, 30
                )
            )
            discharge = admit + timedelta(days=los)
            if via_ed:
                ed_start = admit - timedelta(hours=float(rng.uniform(2, 10)))
                events.append((ed_start, "ed_admit"))
                ed_times.append(ed_start)
            events.append((admit, f"admission:{los}"))
            self.stats["acute_admissions"] += 1
            if rng.random() < _sigmoid(
                -4.4 + 0.03 * (p.age(admit) - 65) + 0.15 * charlson + 0.4 * p.frailty
            ):
                p.deceased = discharge
                self.stats["in_hospital_deaths"] += 1
                admissions.append((admit, discharge))
                break
            prior = sum(1 for _, d in admissions if admit - timedelta(days=365) <= d < admit)
            ed6 = sum(
                1
                for e in ed_times
                if admit - timedelta(days=182) <= e < admit - timedelta(hours=24)
            )
            age = p.age(admit)
            chf = p.active("chf", admit)
            stage = p.ckd_stage(admit)
            egfr = {0: 86, 2: 74, 3: 44, 4: 22, 5: 10}[stage] + rng.normal(0, 8)
            sodium = (135.5 if chf else 139.5) + rng.normal(0, 3.5)
            logit = (
                c["intercept"]
                + c["age_per_year_over_65"] * (age - 65)
                + c["charlson_point"] * charlson
                + c["heart_failure"] * chf
                + c["copd"] * (p.active("copd", admit) or p.active("emphysema", admit))
                + c["ckd_stage3_plus"] * (stage >= 3)
                + c["diabetes_complications"]
                * (p.active("diabetic_neuropathy", admit) or p.active("diabetic_kidney", admit))
                + c["log1p_length_of_stay"] * math.log1p(los)
                + c["prior_admissions_12m_capped4"] * min(prior, 4)
                + c["ed_visits_6m_capped4"] * min(ed6, 4)
                + c["polypharmacy"] * (len(p.meds_at(admit)) >= 5)
                + c["social_needs"] * p.social_needs(admit)
                + c["emergency_admission"] * via_ed
                + c["weekend_discharge"] * (discharge.weekday() >= 5)
                + c["egfr_below_45"] * (egfr < 45)
                + c["sodium_below_135"] * (sodium < 135)
                + c["heart_failure_x_age_75_plus"] * (chf and age >= 75)
                + c["unobserved_frailty"] * p.frailty
            )
            readmit = rng.random() < _sigmoid(logit)
            admissions.append((admit, discharge))
            if readmit:
                forced = discharge + timedelta(days=1.5 + min(float(rng.exponential(8)), 28.4))
                self.stats["readmissions"] += 1
            elif rng.random() < _sigmoid(
                -4.6 + 0.03 * (age - 65) + 0.1 * charlson + 0.3 * p.frailty
            ):
                p.deceased = discharge + timedelta(days=float(rng.uniform(2, 30)))
                self.stats["post_discharge_deaths"] += 1
                break
            if age >= 70 and rng.random() < 0.12:
                snf_end = discharge + timedelta(days=float(rng.uniform(7, 30)))
                if forced is not None:
                    snf_end = min(snf_end, forced - timedelta(hours=2))
                if snf_end > discharge + timedelta(hours=3):
                    events.append(
                        (
                            discharge + timedelta(hours=1),
                            f"snf:{(snf_end - discharge).total_seconds() / 86400}",
                        )
                    )
            t = forced if forced is not None else discharge + timedelta(days=31)

        self._emit(b, p, sorted(events, key=lambda e: e[0]))
        return p

    def _emit(self, b: _Builder, p: _Patient, events: list[tuple[datetime, str]]) -> None:
        rng = self.rng
        gender = p.sex
        resource: dict[str, Any] = {
            "resourceType": "Patient",
            "id": p.pid,
            "extension": [
                {
                    "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                    "extension": [
                        {
                            "url": "ombCategory",
                            "valueCoding": {
                                "system": "urn:oid:2.16.840.1.113883.6.238",
                                "code": p.race[0],
                                "display": p.race[1],
                            },
                        },
                        {"url": "text", "valueString": p.race[1]},
                    ],
                },
                {
                    "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                    "extension": [
                        {
                            "url": "ombCategory",
                            "valueCoding": {
                                "system": "urn:oid:2.16.840.1.113883.6.238",
                                "code": p.ethnicity[0],
                                "display": p.ethnicity[1],
                            },
                        },
                        {"url": "text", "valueString": p.ethnicity[1]},
                    ],
                },
            ],
            "gender": gender,
            "birthDate": p.birth.strftime("%Y-%m-%d"),
        }
        if p.deceased:
            resource["deceasedDateTime"] = _ts(p.deceased)
        b.resources.insert(0, resource)

        recorded: set[str] = set()
        last_med_renewal: dict[str, datetime] = {}
        for key, onset in p.onsets.items():
            if onset < WINDOW_START:
                b.condition(p.pid, key, onset, None)
                recorded.add(key)
        for moment, kind in events:
            if p.deceased and moment > p.deceased:
                break
            if kind.startswith("admission:"):
                los = float(kind.split(":")[1])
                end = moment + timedelta(days=los)
                if rng.random() < 0.06 and los > 1.5:
                    cut1, cut2 = (
                        moment + timedelta(days=los * 0.3),
                        moment + timedelta(days=los * 0.6),
                    )
                    eid = b.encounter(p.pid, "admission", moment, cut1)
                    b.encounter(p.pid, "icu", cut1 - timedelta(minutes=20), cut2)
                    b.encounter(p.pid, "ward", cut2 - timedelta(minutes=15), end)
                else:
                    eid = b.encounter(p.pid, "admission", moment, end)
                self._vitals_and_labs(
                    b, p, eid, moment + timedelta(hours=1), labs=rng.random() < 0.92
                )
                if rng.random() < 0.25:
                    b.condition(p.pid, "pneumonia", moment, eid, abatement=end + timedelta(days=10))
                b.medication(p.pid, eid, end, "amox_clav", "completed")
                for med in p.meds_at(end):
                    if (end - last_med_renewal.get(med, datetime.min)).days > 300:
                        b.medication(p.pid, eid, end, med, "active")
                        last_med_renewal[med] = end
                continue
            if kind.startswith("snf:"):
                b.encounter(
                    p.pid, "snf", moment, moment + timedelta(days=float(kind.split(":")[1]))
                )
                continue
            encounter_kind = "ed" if kind in ("ed", "ed_admit") else kind
            minutes = {"checkup": 25, "problem": 20, "ed": 240}[encounter_kind]
            eid = b.encounter(p.pid, encounter_kind, moment, moment + timedelta(minutes=minutes))
            for key, onset in p.onsets.items():
                if key not in recorded and onset <= moment:
                    # Onset may predate the visit where it is first documented (recordedDate).
                    b.condition(p.pid, key, onset, eid, recorded=moment)
                    recorded.add(key)
            if encounter_kind in ("checkup", "ed"):
                chronic = p.charlson(moment) > 0 or p.active("hypertension", moment)
                self._vitals_and_labs(
                    b,
                    p,
                    eid,
                    moment,
                    labs=encounter_kind == "checkup" and chronic and rng.random() < 0.8,
                )
            if encounter_kind == "checkup":
                for med in p.meds_at(moment):
                    if (moment - last_med_renewal.get(med, datetime.min)).days > 300:
                        b.medication(p.pid, eid, moment, med, "active")
                        last_med_renewal[med] = moment
            elif encounter_kind == "problem" and rng.random() < 0.3:
                b.condition(
                    p.pid,
                    str(rng.choice(["sinusitis", "bronchitis"])),
                    moment,
                    eid,
                    abatement=moment + timedelta(days=float(rng.uniform(7, 21))),
                )

    def generate(self, n_patients: int) -> list[list[dict[str, Any]]]:
        """Return one list of resources per patient."""
        patients = []
        for _ in range(n_patients):
            builder = _Builder(self.rng, self.mode)
            self._simulate(builder)
            patients.append(builder.resources)
        self.stats["patients"] = n_patients
        return patients


def write_population(
    out_dir: str | Path,
    n_patients: int,
    seed: int = 42,
    fmt: Literal["ndjson", "bundle"] = "ndjson",
    drift: bool = False,
) -> dict[str, Any]:
    """Generate a population and write it as FHIR Bulk NDJSON or one Bundle per patient."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generator = PopulationGenerator(seed=seed, drift=drift, mode=fmt)
    patients = generator.generate(n_patients)
    counts: dict[str, int] = defaultdict(int)
    if fmt == "ndjson":
        handles: dict[str, Any] = {}
        try:
            for resources in patients:
                for resource in resources:
                    kind = resource["resourceType"]
                    if kind not in handles:
                        handles[kind] = open(out / f"{kind}.ndjson", "w", encoding="utf-8")  # noqa: SIM115
                    handles[kind].write(json.dumps(resource, separators=(",", ":")) + "\n")
                    counts[kind] += 1
        finally:
            for handle in handles.values():
                handle.close()
    else:
        for resources in patients:
            bundle = {
                "resourceType": "Bundle",
                "type": "transaction",
                "entry": [
                    {
                        "fullUrl": f"urn:uuid:{r['id']}",
                        "resource": r,
                        "request": {"method": "POST", "url": r["resourceType"]},
                    }
                    for r in resources
                ],
            }
            for r in resources:
                counts[r["resourceType"]] += 1
            (out / f"patient_{resources[0]['id']}.json").write_text(
                json.dumps(bundle, separators=(",", ":")), encoding="utf-8"
            )
    stats = dict(generator.stats)
    manifest = {
        "generator": "readmit synthetic FHIR R4 population",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "format": fmt,
        "drift": drift,
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "patients": n_patients,
        "resource_counts": dict(counts),
        "simulation_stats": stats,
        "simulated_readmission_rate": round(
            stats.get("readmissions", 0) / max(stats.get("acute_admissions", 1), 1), 4
        ),
        "readmission_log_odds_coefficients": READMISSION_COEFFICIENTS,
        "note": "Synthetic data only. Coefficients are the generator's ground truth, not "
        "clinical estimates.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

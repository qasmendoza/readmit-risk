"""Charlson comorbidity rules, checked against condition displays that Synthea actually emits."""

import pytest

from readmit.fhir.codes import SDOH_PATTERN, charlson_categories, charlson_score


@pytest.mark.parametrize(
    ("display", "expected"),
    [
        ("Chronic congestive heart failure (disorder)", ["congestive_heart_failure"]),
        ("History of myocardial infarction (situation)", ["myocardial_infarction"]),
        ("Non-small cell lung cancer (disorder)", ["any_malignancy"]),
        ("Malignant neoplasm of breast (disorder)", ["any_malignancy"]),
        ("Metastasis from malignant tumor of prostate (disorder)", ["metastatic_solid_tumor"]),
        (
            "Secondary and unspecified malignant neoplasm of lymph nodes of head, face and neck",
            ["metastatic_solid_tumor"],
        ),
        ("Neuropathy due to type 2 diabetes mellitus (disorder)", ["diabetes_complicated"]),
        ("Diabetes mellitus type 2 (disorder)", ["diabetes_uncomplicated"]),
        ("Chronic kidney disease stage 3 (disorder)", ["renal_disease"]),
        ("End-stage renal disease (disorder)", ["renal_disease"]),
        ("Pulmonary emphysema (disorder)", ["chronic_pulmonary_disease"]),
        ("Alzheimer's disease (disorder)", ["dementia"]),
        ("Stroke", ["cerebrovascular_disease"]),
    ],
)
def test_categories_match(display, expected):
    assert charlson_categories(display) == expected


@pytest.mark.parametrize(
    "display",
    [
        "Suspected lung cancer (situation)",  # not a confirmed diagnosis
        "Carcinoma in situ of prostate (disorder)",  # in situ is not invasive malignancy
        "Squamous cell carcinoma of skin of scalp and neck",  # non-melanoma skin cancer excluded
        "Benign neoplasm of tongue",
        "Neoplasm of prostate",  # unspecified behaviour: conservative
        "Chronic kidney disease stage 2 (disorder)",  # mild CKD is not Charlson renal disease
        "Prediabetes",
        "Encounter for screening for diabetes mellitus",
        "Essential hypertension (disorder)",
        "Hearing aids user",  # regression: once matched AIDS/HIV
    ],
)
def test_non_matches(display):
    assert charlson_categories(display) == []


def test_hierarchy_rules():
    # complicated diabetes replaces uncomplicated; metastatic replaces malignancy
    categories = {
        "diabetes_uncomplicated",
        "diabetes_complicated",
        "any_malignancy",
        "metastatic_solid_tumor",
        "congestive_heart_failure",
    }
    assert charlson_score(categories) == 2 + 6 + 1
    assert charlson_score(set()) == 0
    assert charlson_score({"aids_hiv"}) == 6


def test_sdoh_pattern():
    assert SDOH_PATTERN.search("Social isolation (finding)")
    assert SDOH_PATTERN.search("Lack of access to transportation (finding)")
    assert not SDOH_PATTERN.search("Full-time employment (finding)")

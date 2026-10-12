"""Clinical terminology used by the pipeline.

Charlson Comorbidity Index (Charlson et al., 1987) categories are normally defined on ICD codes
(e.g. Quan et al., 2005). Synthea and many FHIR sources code conditions in SNOMED CT (and
sometimes ICD-10), so categories here are matched on the coded display text with explicit
include/exclude rules. The rules were checked against the conditions Synthea actually emits:
"Suspected lung cancer" and "Carcinoma in situ" do not count as malignancy, non-melanoma skin
cancers are excluded (as in Quan's ICD-10 algorithm), CKD stages 1-2 are not "renal disease",
and "History of myocardial infarction" does count. A production deployment would replace these
rules with validated value sets (e.g. from the NLM Value Set Authority Center).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"

# Observation codes used as features: feature name -> accepted LOINC codes (first = preferred).
LAB_FEATURES: dict[str, tuple[str, ...]] = {
    "last_bmi": ("39156-5",),
    "last_systolic_bp": ("8480-6",),
    "last_heart_rate": ("8867-4",),
    "last_egfr": ("33914-3",),
    "last_creatinine": ("38483-4", "2160-0"),
    "last_hba1c": ("4548-4",),
    "last_glucose": ("2339-0", "2345-7"),
    "last_sodium": ("2947-0",),
}
BLOOD_PRESSURE_PANEL = "85354-9"
SYSTOLIC_BP = "8480-6"
DIASTOLIC_BP = "8462-4"

INPATIENT_CLASS = "IMP"
EMERGENCY_CLASS = "EMER"
AMBULATORY_CLASS = "AMB"


@dataclass(frozen=True)
class CharlsonCategory:
    name: str
    weight: int
    include: str
    exclude: str | None = None
    supersedes: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, display: str) -> bool:
        text = display.lower()
        if not re.search(self.include, text):
            return False
        return not (self.exclude and re.search(self.exclude, text))


CHARLSON_CATEGORIES: tuple[CharlsonCategory, ...] = (
    CharlsonCategory("myocardial_infarction", 1, r"myocardial infarction"),
    CharlsonCategory("congestive_heart_failure", 1, r"heart failure"),
    CharlsonCategory(
        "peripheral_vascular_disease",
        1,
        r"peripheral (vascular|arterial) disease|aortic aneurysm|intermittent claudication",
    ),
    CharlsonCategory(
        "cerebrovascular_disease",
        1,
        r"\bstroke\b|cerebrovascular|cerebral infarction|transient ischemic attack",
    ),
    CharlsonCategory("dementia", 1, r"dementia|alzheimer"),
    CharlsonCategory(
        "chronic_pulmonary_disease",
        1,
        r"chronic obstructive|emphysema|asthma|chronic bronchitis|pulmonary fibrosis",
    ),
    CharlsonCategory(
        "rheumatic_disease",
        1,
        r"rheumatoid arthritis|systemic lupus|polymyalgia rheumatica|polymyositis|dermatomyositis",
    ),
    CharlsonCategory("peptic_ulcer_disease", 1, r"peptic ulcer|gastric ulcer|duodenal ulcer"),
    CharlsonCategory(
        "mild_liver_disease", 1, r"chronic hepatitis|cirrhosis|fatty liver|chronic liver disease"
    ),
    CharlsonCategory(
        "diabetes_uncomplicated",
        1,
        r"diabetes mellitus",
        exclude=r"prediabetes|screening|due to|gestational",
    ),
    CharlsonCategory(
        "diabetes_complicated",
        2,
        r"due to (type (1|2|i|ii) )?diabetes|diabetic (retinopathy|nephropathy|neuropathy|foot)",
        supersedes=("diabetes_uncomplicated",),
    ),
    CharlsonCategory("hemiplegia_paraplegia", 2, r"hemipleg|parapleg|quadripleg|tetrapleg"),
    CharlsonCategory(
        "renal_disease",
        2,
        r"chronic kidney disease stage [345]|end[- ]stage renal|renal transplant|kidney transplant"
        r"|dialysis|awaiting transplantation of kidney",
    ),
    CharlsonCategory(
        "any_malignancy",
        2,
        r"malignant|carcinoma|cancer|lymphoma|leukemia|sarcoma|melanoma|myeloma",
        exclude=(
            r"suspected|in situ|benign|screening|uncertain behavior|metasta|secondary"
            r"|carcinoma of skin|basal cell|squamous cell carcinoma of skin"
        ),
    ),
    CharlsonCategory(
        "severe_liver_disease",
        3,
        r"esophageal varices|portal hypertension|hepatic failure|hepatorenal|hepatic encephalopathy",
        supersedes=("mild_liver_disease",),
    ),
    CharlsonCategory(
        "metastatic_solid_tumor",
        6,
        r"metasta|secondary (and unspecified )?malignant",
        supersedes=("any_malignancy",),
    ),
    CharlsonCategory(
        "aids_hiv",
        6,
        r"\bhiv\b|human immunodeficiency virus|acquired immunodeficiency|\baids\b",
        exclude=r"(hearing|walking|mobility|visual|memory) aids?\b|aids user",
    ),
)
CHARLSON_BY_NAME = {c.name: c for c in CHARLSON_CATEGORIES}


def charlson_categories(display: str) -> list[str]:
    """Charlson categories matched by a single condition display."""
    return [c.name for c in CHARLSON_CATEGORIES if c.matches(display)]


def charlson_score(categories: set[str]) -> int:
    """Weighted Charlson score, applying hierarchy rules (e.g. metastatic replaces malignancy)."""
    present = set(categories)
    for category in CHARLSON_CATEGORIES:
        if category.name in present:
            present -= set(category.supersedes)
    return sum(CHARLSON_BY_NAME[name].weight for name in present)


# Social determinants of health (SDOH) recorded as SNOMED "(finding)" conditions in Synthea.
SDOH_PATTERN = re.compile(
    r"social isolation|limited social contact|unemployed|housing unsatisfactory|homeless"
    r"|lack of access to transportation|transport problem|victim of intimate partner abuse"
    r"|reports of violence in the environment",
    re.IGNORECASE,
)
# Conditions that describe circumstances rather than diseases; excluded from condition counts.
NON_DISEASE_PATTERN = re.compile(r"\((?:finding|situation)\)", re.IGNORECASE)

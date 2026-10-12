# Methodology

This document describes how ReadmitRisk defines its cohort and outcome, builds features, validates
models, and evaluates them. Section numbers in the generated [evaluation report](../reports/REPORT.md)
follow the same order.

## 1. Data and interoperability

Input is HL7 FHIR R4, either as **Bulk Data (Flat FHIR) NDJSON**, one file per resource type as produced by
payer and EHR bulk `$export` APIs, or as **Bundles** (Synthea's default output, and the API's input format).
Five resource types are used: `Patient` (with US Core race and ethnicity extensions), `Encounter`, `Condition`,
`Observation` (blood-pressure panels are expanded into their systolic and diastolic components), and
`MedicationRequest` (inline codes or `medicationReference` resolved through `Medication` resources).
References in either `urn:uuid:` or `Type/id` form resolve to logical ids. Files for unused resource types in bulk
exports (claims, provenance, and so on) are skipped without being parsed.

The parser is contract-tested against trimmed real Synthea bundles in `tests/fixtures/synthea/`, and the full
ingestion path was run on a 250-patient Synthea v3.3.0 bulk export. The PySpark implementation must match the
pandas implementation column for column (enforced in CI).

## 2. Cohort and outcome

Simplified from the CMS Hospital Readmissions Reduction Program (HRRP) measure methodology:

1. **Acute hospitalizations.** Finished inpatient (`IMP`) encounters, excluding non-acute settings (skilled
   nursing, rehabilitation, detoxification, hospice, psychiatric), which are neither index stays nor readmissions.
2. **Transfer chains.** Acute encounters that start within 24 hours of the previous one ending are merged into one
   hospitalization. Synthea models an ICU stay as admission, ICU transfer, and ward encounters; without merging,
   each transfer would be mislabeled as a readmission.
3. **Exclusions.** Age under 18 at admission; death during the stay; discharge within 30 days of the data
   extraction date (follow-up not fully observable).
4. **Outcome.** Another acute hospitalization beginning within 30 days after discharge. As in HRRP, a readmission
   can itself be an index stay.

Not implemented: the CMS planned-readmission algorithm, so scheduled readmissions (for example, chemotherapy) count.
The generated report shows a CONSORT-style flow with counts at every step.

## 3. Features

All features are computed **as of the discharge timestamp**, using only information that existed then:

- utilization windows end at the index admission, and the ED visit that triggered the admission (within 24 hours
  before it) is not counted as a prior visit;
- conditions count only once recorded (`recordedDate`, falling back to onset), so a diagnosis documented after
  discharge with a back-dated onset cannot leak into training;
- medications are those **prescribed** in the prior 12 months; `MedicationRequest.status` is deliberately ignored
  because it describes the record at export time, not at discharge;
- labs and vitals are the most recent values on or before discharge (12-month lookback).

`tests/test_features.py::test_features_never_use_future_data` truncates real records at the discharge timestamp and
asserts that the features are unchanged. The same function computes features for training and for live API
requests, with settings read from the model bundle, so training and serving cannot diverge.

**Charlson Comorbidity Index** categories (Charlson 1987; ICD-10 coding per Quan 2005) are matched on SNOMED CT or
ICD-10 display text with explicit include and exclude rules, checked against the conditions Synthea emits: suspected
diagnoses, carcinoma in situ, benign neoplasms, and non-melanoma skin cancer are not malignancy; CKD stages 1-2 are
not renal disease; hierarchy rules apply (metastatic replaces malignancy, complicated replaces uncomplicated
diabetes). A condition counts if it is active at discharge or was active within the prior year. Production use would
replace these rules with validated value sets (for example, from the NLM Value Set Authority Center).

**LACE index** (van Walraven 2010): length of stay, acuity (emergency admission), Charlson score, and ED visits in
the prior 6 months, scored 0-19.

### Feature dictionary

Monotonic direction is enforced in the gradient-boosting model (risk may only rise, or only fall, with the feature).

| Feature | Description | Type | Unit | FHIR source | Monotonic |
|---|---|---|---|---|---|
| `age_years` | Age at admission | continuous | years | Patient.birthDate | – |
| `male` | Male sex | binary | – | Patient.gender | – |
| `los_days` | Length of stay | continuous | days | Encounter.period | ↑ only |
| `emergency_admission` | Admitted through the emergency department | binary | – | Encounter (EMER within 24 h before admission) | ↑ only |
| `prior_admissions_12m` | Hospitalizations in the prior 12 months | count | – | Encounter (acute IMP) | ↑ only |
| `ed_visits_6m` | Emergency department visits in the prior 6 months | count | – | Encounter (EMER) | ↑ only |
| `outpatient_visits_12m` | Outpatient visits in the prior 12 months | count | – | Encounter (AMB) | – |
| `days_since_last_discharge` | Days since previous discharge (capped at 365) | continuous | days | Encounter (acute IMP) | ↓ only |
| `charlson_score` | Charlson Comorbidity Index | score | – | Condition | ↑ only |
| `chf` | Congestive heart failure | binary | – | Condition | ↑ only |
| `chronic_pulmonary` | Chronic pulmonary disease | binary | – | Condition | ↑ only |
| `renal_disease` | Moderate or severe chronic kidney disease | binary | – | Condition | ↑ only |
| `diabetes_complicated` | Diabetes with complications | binary | – | Condition | ↑ only |
| `dementia` | Dementia | binary | – | Condition | – |
| `malignancy` | Cancer (any malignancy, including metastatic) | binary | – | Condition | – |
| `n_active_conditions` | Active chronic conditions | count | – | Condition | – |
| `sdoh_count` | Documented social needs | count | – | Condition (SDOH findings) | ↑ only |
| `n_medications_12m` | Distinct medications prescribed in the prior 12 months | count | – | MedicationRequest | ↑ only |
| `polypharmacy` | Polypharmacy | binary | – | MedicationRequest | ↑ only |
| `last_bmi` | Most recent BMI | continuous | kg/m² | Observation 39156-5 | – |
| `last_systolic_bp` | Most recent systolic blood pressure | continuous | mmHg | Observation 8480-6 | – |
| `last_heart_rate` | Most recent heart rate | continuous | /min | Observation 8867-4 | – |
| `last_egfr` | Most recent eGFR | continuous | mL/min/1.73m² | Observation 33914-3 | ↓ only |
| `last_creatinine` | Most recent creatinine | continuous | mg/dL | Observation 38483-4 | – |
| `last_hba1c` | Most recent HbA1c | continuous | % | Observation 4548-4 | – |
| `last_glucose` | Most recent glucose | continuous | mg/dL | Observation 2339-0 | – |
| `last_sodium` | Most recent sodium | continuous | mmol/L | Observation 2947-0 | – |
| `weekend_discharge` | Discharged on a weekend | binary | – | Encounter.period.end | – |

Race and ethnicity are **not** model inputs. They are used only for the fairness audit.

## 4. Models

| Candidate | Description |
|---|---|
| `lace` | LACE score mapped to a probability with a one-feature logistic model: the clinical baseline |
| `logistic` | L2-regularized logistic regression; median imputation with missingness indicators; standardized inputs |
| `hist_gbm` | Histogram gradient boosting (native missing-value handling) with monotonic constraints |

Every candidate's final model is calibrated with patient-grouped cross-validated Platt scaling (`sigmoid`;
`isotonic` is configurable).

## 5. Validation design

- **Temporal split.** The most recent 20% of discharges form the test set; models never see the future.
- **Embargo.** Training stays discharged within 30 days before the cutoff are dropped, because their labels depend on
  events in the test period.
- **Patient-grouped cross-validation** (5 folds) on the training period for hyperparameter tuning and model
  selection, so one patient's stays never fall on both sides of a fold.
- **Selection by cross-validated AUPRC**, which suits an imbalanced outcome. The test set is used once, for the
  final report.

## 6. Statistical analysis

- **Table 1** compares readmitted and non-readmitted patients with Mann-Whitney U (continuous), χ² or Fisher's exact
  (binary), and χ² (categorical) tests, Holm-adjusted across rows, plus standardized mean differences. Because
  patients contribute several stays, Table 1 uses **one randomly selected stay per patient**: in development, an
  episode-level analysis produced spurious sex and race associations (p < 0.001) that disappeared (p = 0.67 and 0.087) once the
  independence assumption was respected.
- **Confidence intervals** use a patient-level cluster bootstrap (1,000 resamples).
- **Model comparison** uses DeLong's test for correlated ROC curves (fast midrank algorithm of Sun & Xu 2014),
  validated against scikit-learn AUCs and bootstrap standard errors in the test suite.
- **Calibration** is summarized by the calibration slope and calibration-in-the-large (logistic recalibration),
  expected calibration error, and the Hosmer-Lemeshow test (reported with its known sample-size sensitivity).

## 7. Decision analysis

Care-management programs have finite capacity, so the high-risk threshold flags the **top 10% of training-period
scores**; it is then applied unchanged to the test period, as a deployed model would be. The report gives
sensitivity, PPV, the number of patients evaluated per readmission found, and lift over the base rate. Decision-curve
analysis (Vickers & Elkin 2006) compares net benefit with flag-everyone and flag-no-one strategies.

## 8. Fairness and explainability

The audit reports AUROC, calibration-in-the-large (observed/expected ratio), flag rate, and sensitivity by sex,
race, ethnicity, and age band, skipping groups with fewer than 50 stays, and tests equal opportunity with a χ² test.
A maximum subgroup AUROC gap can be made a deployment requirement (`gate.max_fairness_auroc_gap`).
Global explanations are permutation importances (drop in test AUPRC). Per-patient drivers replace each feature with
its typical (median) value and report the change in risk; this occlusion approach is fast and model-agnostic but
ignores interactions, so it is presented as approximate.

## 9. Quality gate

The pipeline fails, and the model bundle is not written, unless the selected model reaches the minimum test AUROC
(default 0.65) and outperforms the LACE baseline; an optional subgroup-gap limit can be added. The same gate runs in
CI and inside the Docker build.

## 10. Monitoring

Drift is measured on every feature and on the predicted risk itself. A feature is flagged when either its
Population Stability Index reaches 0.2 or its absolute standardized mean difference reaches 0.2. Both are needed: in
the drift simulation, ED visits per stay rose 77% while PSI was only 0.08, and PSI is identically 0 for binary
features because all quantile bins collapse. Readmission labels arrive 30 days late, so performance monitoring scores
matured stays and recommends recalibration or retraining if AUROC falls by more than 0.05 or calibration shifts.

## 11. Synthetic data

Synthea is excellent for interoperability testing, but its 30-day readmission rate is about 4-6% and readmissions are
not driven by clinical risk factors. The repository's generator (`src/readmit/fhir/synthetic.py`) produces
Synthea-shaped FHIR (same codes, systems, and structure, including transfer chains, skilled-nursing stays, deaths,
mid-history diagnoses, and missing labs) with readmissions drawn from the logistic mechanism below plus an
**unobserved frailty term**, so no model can be perfect. The overall readmission rate is about 14-15%, in line with
the US all-cause 30-day rate.

| Term (log-odds) | Coefficient |
|---|---|
| `intercept` | -4.400 |
| `age_per_year_over_65` | +0.022 |
| `charlson_point` | +0.100 |
| `heart_failure` | +0.380 |
| `copd` | +0.250 |
| `ckd_stage3_plus` | +0.280 |
| `diabetes_complications` | +0.200 |
| `log1p_length_of_stay` | +0.320 |
| `prior_admissions_12m_capped4` | +0.340 |
| `ed_visits_6m_capped4` | +0.170 |
| `polypharmacy` | +0.220 |
| `social_needs` | +0.300 |
| `emergency_admission` | +0.200 |
| `weekend_discharge` | +0.100 |
| `egfr_below_45` | +0.250 |
| `sodium_below_135` | +0.220 |
| `heart_failure_x_age_75_plus` | +0.450 |
| `unobserved_frailty` | +0.550 |

These are the generator's ground truth, chosen to be directionally consistent with the readmission literature; they
are not clinical estimates. Because the mechanism is known and relatively clean, discrimination here is higher than
real-world readmission models typically achieve (a systematic review found C-statistics mostly between 0.6 and 0.7;
Kansagara 2011). Results demonstrate the pipeline, not clinical performance.

## References

- Charlson ME, et al. A new method of classifying prognostic comorbidity in longitudinal studies. *J Chronic Dis.* 1987;40(5):373-383.
- Quan H, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. *Med Care.* 2005;43(11):1130-1139.
- van Walraven C, et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. *CMAJ.* 2010;182(6):551-557.
- Kansagara D, et al. Risk prediction models for hospital readmission: a systematic review. *JAMA.* 2011;306(15):1688-1698.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics.* 1988;44(3):837-845.
- Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Process Lett.* 2014;21(11):1389-1393.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making.* 2006;26(6):565-574.
- Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17:230.
- Collins GS, et al. TRIPOD+AI statement. *BMJ.* 2024;385:e078378.
- Mitchell M, et al. Model cards for model reporting. *FAT\* 2019.*
- Obermeyer Z, et al. Dissecting racial bias in an algorithm used to manage the health of populations. *Science.* 2019;366(6464):447-453.
- Walonoski J, et al. Synthea: an approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. *JAMIA.* 2018;25(3):230-238.

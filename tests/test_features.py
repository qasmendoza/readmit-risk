"""Point-in-time feature engineering, including leakage property tests."""

import numpy as np
import pandas as pd
import pytest

from readmit.cohort import build_cohort, build_episodes
from readmit.config import CohortConfig, FeatureConfig
from readmit.features import FEATURE_NAMES, LACE_COLUMN, compute_features, lace_score
from tests.helpers import condition, encounter, patient, tables_from, truncate

CCFG, FCFG = CohortConfig(), FeatureConfig()


@pytest.mark.parametrize(
    ("los", "emergency", "charlson", "ed", "expected"),
    [
        (0.5, 0, 0, 0, 0),
        (1.5, 0, 0, 0, 1),
        (5.0, 1, 2, 1, 10),
        (9.0, 0, 3, 2, 10),
        (20.0, 1, 7, 9, 19),
    ],
)
def test_lace_score(los, emergency, charlson, ed, expected):
    assert lace_score([los], [emergency], [charlson], [ed])[0] == expected


def _features(tables, as_of="2025-06-01"):
    cohort = build_cohort(tables, CCFG, as_of=pd.Timestamp(as_of))
    return compute_features(tables, cohort.index, FCFG, CCFG), cohort


def test_emergency_admission_and_prior_ed_visits():
    tables = tables_from(
        [patient()],
        [
            encounter("ed0", "2024-02-01 10:00", "2024-02-01 14:00", cls="EMER"),
            encounter("ed1", "2024-03-01 03:00", "2024-03-01 07:00", cls="EMER"),
            encounter("a1", "2024-03-01 08:00", "2024-03-04 08:00"),
        ],
    )
    features, _ = _features(tables)
    row = features.iloc[0]
    assert row["emergency_admission"] == 1  # ED visit 5 h before admission
    assert row["ed_visits_6m"] == 1  # the triggering ED visit is not a "prior" visit
    assert row["los_days"] == pytest.approx(3.0)


def test_condition_recorded_after_discharge_is_not_used():
    base = [patient()], [encounter("a1", "2024-03-01", "2024-03-05")]
    late = tables_from(
        *base,
        conditions=[
            condition(
                "c1",
                "Chronic congestive heart failure (disorder)",
                onset="2023-01-01",
                recorded="2024-04-01",
            )
        ],
    )  # back-dated onset, documented after discharge
    timely = tables_from(
        *base,
        conditions=[
            condition("c1", "Chronic congestive heart failure (disorder)", onset="2023-01-01")
        ],
    )
    assert _features(late)[0]["chf"].iloc[0] == 0
    assert _features(timely)[0]["chf"].iloc[0] == 1
    assert _features(timely)[0]["charlson_score"].iloc[0] == 1


def test_features_never_use_future_data(synthetic_tables):
    """Features on the full record equal features on the record truncated at discharge."""
    cohort = build_cohort(synthetic_tables, CCFG)
    sample = cohort.index.sample(12, random_state=0)
    for _, episode in sample.iterrows():
        full_tables = synthetic_tables.for_patient(episode["patient_id"])
        past_tables = truncate(synthetic_tables, episode["patient_id"], episode["discharge"])
        one = episode.to_frame().T.infer_objects()
        full = compute_features(full_tables, one, FCFG, CCFG)
        past = compute_features(
            past_tables,
            one,
            FCFG,
            CCFG,
            build_episodes(
                past_tables.encounters, CCFG.exclude_encounter_types, CCFG.transfer_gap_hours
            ),
        )
        pd.testing.assert_frame_equal(full, past, check_dtype=False)


def test_feature_matrix_shape_and_ranges(synthetic_tables):
    cohort = build_cohort(synthetic_tables, CCFG)
    features = compute_features(synthetic_tables, cohort.index, FCFG, CCFG)
    assert list(features.columns) == [*FEATURE_NAMES, LACE_COLUMN]
    assert len(features) == len(cohort.index)
    assert features["charlson_score"].between(0, 40).all()
    assert features["days_since_last_discharge"].between(0, 365).all()
    assert set(np.unique(features["polypharmacy"])) <= {0, 1}
    assert (features["polypharmacy"] == (features["n_medications_12m"] >= 5)).all()

"""Cohort definition: transfer chains, exclusions, labels, and scoring-episode selection."""

import pandas as pd
import pytest

from readmit.cohort import EncounterNotFoundError, build_cohort, build_episodes, scoring_episode
from readmit.config import CohortConfig
from tests.helpers import encounter, patient, tables_from

CFG = CohortConfig()


def test_synthea_icu_transfer_chain_is_one_hospitalization(synthea_tables):
    episodes = build_episodes(
        synthea_tables.encounters, CFG.exclude_encounter_types, CFG.transfer_gap_hours
    )
    assert episodes["n_encounters"].max() == 3  # admission -> ICU transfer -> ward
    assert not episodes["encounter_ids"].map(len).eq(0).any()


def test_synthea_real_readmission_is_labelled(synthea_tables):
    cohort = build_cohort(
        synthea_tables, CFG.model_copy(update={"min_age": 0}), as_of=pd.Timestamp("2026-01-01")
    )
    readmitted = cohort.index[cohort.index["readmitted_30d"] == 1]
    assert len(readmitted) == 1
    assert 6.9 < readmitted["days_to_readmission"].iloc[0] < 7.1


def test_pediatric_stays_excluded_by_default(synthea_tables):
    cohort = build_cohort(synthea_tables, CFG, as_of=pd.Timestamp("2026-01-01"))
    assert (cohort.index["age"] >= 18).all()
    steps = {f["step"]: f["n"] for f in cohort.flow}
    assert steps["Age >= 18 at admission"] < steps["Hospitalizations after merging transfer chains"]


def test_skilled_nursing_is_neither_index_nor_readmission():
    tables = tables_from(
        [patient()],
        [
            encounter("e1", "2024-01-01", "2024-01-05"),
            encounter(
                "snf",
                "2024-01-05 02:00",
                "2024-01-25",
                display="Admission to skilled nursing facility (procedure)",
            ),
        ],
    )
    cohort = build_cohort(tables, CFG, as_of=pd.Timestamp("2025-01-01"))
    assert list(cohort.index["episode_id"]) == ["e1"]
    assert cohort.index["readmitted_30d"].iloc[0] == 0


def test_readmission_window_and_exclusions():
    tables = tables_from(
        [patient("p1"), patient("p2", deceased="2024-03-04"), patient("p3")],
        [
            encounter("a1", "2024-01-01", "2024-01-04", pid="p1"),
            encounter("a2", "2024-01-20", "2024-01-22", pid="p1"),  # 16 days later: readmission
            encounter("a3", "2024-04-01", "2024-04-02", pid="p1"),  # 69 days later: not one
            encounter("b1", "2024-03-01", "2024-03-04", pid="p2"),  # died in hospital
            encounter("c1", "2024-12-20", "2024-12-22", pid="p3"),  # follow-up incomplete
        ],
    )
    cohort = build_cohort(tables, CFG, as_of=pd.Timestamp("2025-01-01"))
    labels = dict(zip(cohort.index["episode_id"], cohort.index["readmitted_30d"], strict=True))
    assert labels == {"a1": 1, "a2": 0, "a3": 0}


def test_scoring_episode_selection(synthea_tables):
    one = synthea_tables.for_patient("3ddadc27-f390-02a2-5c25-368d4d8d036f")
    latest = scoring_episode(one, CFG)
    assert latest["n_encounters"].iloc[0] == 3
    inner_id = latest["encounter_ids"].iloc[0][1]  # the ICU leg resolves to the same stay
    assert scoring_episode(one, CFG, inner_id)["episode_id"].iloc[0] == latest["episode_id"].iloc[0]
    with pytest.raises(EncounterNotFoundError):
        scoring_episode(one, CFG, "not-an-encounter")

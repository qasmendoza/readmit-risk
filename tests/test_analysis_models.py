"""Table 1 statistics, temporal splitting, and model construction."""

import numpy as np
import pandas as pd
import pytest

from readmit.analysis import table_one
from readmit.features import FEATURE_NAMES, SPEC_BY_NAME
from readmit.models import make_pipeline, temporal_split


def test_table_one(trained):
    table, meta = table_one(trained.dataset.groupby("patient_id").head(1))
    assert {"variable", "readmitted", "not_readmitted", "p_value", "p_holm", "smd"} <= set(table)
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()  # Holm never shrinks p-values
    lace = table[table["variable"] == "LACE index"].iloc[0]
    assert lace["smd"] > 0  # readmitted patients have higher LACE scores
    assert meta["n_readmitted"] > 0


def test_temporal_split_respects_embargo():
    discharges = pd.Series(pd.date_range("2020-01-01", periods=500, freq="D"))
    split = temporal_split(discharges, test_fraction=0.2, embargo_days=30)
    train, test = discharges.iloc[split.train_index], discharges.iloc[split.test_index]
    assert train.max() < split.cutoff - pd.Timedelta(days=30)
    assert test.min() >= split.cutoff
    assert split.embargoed == 500 - len(train) - len(test) > 0


def test_monotonic_constraints_follow_feature_specs():
    pipe, grid = make_pipeline("hist_gbm", monotonic=True)
    constraints = pipe.named_steps["clf"].monotonic_cst
    assert constraints == [SPEC_BY_NAME[f].monotonic for f in FEATURE_NAMES]
    assert SPEC_BY_NAME["prior_admissions_12m"].monotonic == 1
    assert grid
    assert make_pipeline("hist_gbm", monotonic=False)[0].named_steps["clf"].monotonic_cst is None
    with pytest.raises(ValueError):
        make_pipeline("random_forest")


def test_selected_model_respects_monotonicity(trained):
    """Raising prior admissions must never lower predicted risk."""
    bundle = trained.bundle
    row = trained.dataset.iloc[trained.split.test_index].head(25)
    base = bundle.predict_proba(row)
    more = bundle.predict_proba(row.assign(prior_admissions_12m=row["prior_admissions_12m"] + 3))
    assert np.all(more >= base - 1e-9)

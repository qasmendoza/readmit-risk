"""PySpark flattening must match the pandas implementation exactly."""

import shutil

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark
if shutil.which("java") is None:  # pragma: no cover
    pytest.skip("Java runtime not available", allow_module_level=True)


def test_spark_matches_pandas(tmp_path):
    from readmit.fhir.flatten import TABLE_COLUMNS, FhirTables, flatten_path
    from readmit.fhir.spark_flatten import flatten_ndjson, spark_session
    from readmit.fhir.synthetic import write_population

    write_population(tmp_path / "fhir", 40, seed=3)
    spark = spark_session(master="local[1]")
    try:
        counts = flatten_ndjson(spark, tmp_path / "fhir", tmp_path / "tables")
    finally:
        spark.stop()
    expected = flatten_path(tmp_path / "fhir")
    actual = FhirTables.load(tmp_path / "tables")
    assert counts == expected.summary()
    for name in TABLE_COLUMNS:
        left, right = getattr(expected, name), getattr(actual, name)
        keys = list(left.columns[:2]) + (["code"] if "code" in left else [])
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
        for column in left.columns:
            a, b = left[column], right[column]
            if a.dtype.kind == "M":  # pandas and Spark may use different time units
                a, b = a.astype("datetime64[ns]"), b.astype("datetime64[ns]")
            if a.dtype.kind in "fM":
                pd.testing.assert_series_equal(a, b, check_dtype=False, check_names=False)
            else:
                assert (a.astype(str).fillna("") == b.astype(str).fillna("")).all(), (name, column)

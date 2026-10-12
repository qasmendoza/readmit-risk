"""Distributed flattening of FHIR Bulk Data (NDJSON) with PySpark.

Produces the same five tables (and columns) as the pandas flattener, as Parquet, so the rest of the
pipeline is unchanged when ingestion moves to a cluster (e.g. a Databricks job reading a bulk
``$export`` from cloud storage). Parity with the pandas implementation is enforced by a CI test.

Robustness: optional fields that are absent from a given export (so missing from Spark's inferred
schema) become typed nulls, and ``try_element_at`` avoids ANSI-mode index errors on empty arrays.

Usage::

    spark-submit -m readmit.fhir.spark_flatten <input_dir> <output_dir>
    python -m readmit.fhir.spark_flatten data/fhir data/tables_spark
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from readmit.fhir.flatten import ETHNICITY_URL, RACE_URL, TABLE_COLUMNS


def spark_session(app_name: str = "readmit-fhir-flatten", master: str = "local[*]") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _has(schema: StructType, path: str) -> bool:
    node = schema
    for part in path.split("."):
        if not isinstance(node, StructType) or part not in node.fieldNames():
            return False
        node = node[part].dataType
        while hasattr(node, "elementType"):
            node = node.elementType
    return True


def _col(df: DataFrame, path: str) -> Column:
    return F.col(path) if _has(df.schema, path) else F.lit(None).cast("string")


def _first(column: Column) -> Column:
    return F.try_element_at(column, F.lit(1))


def _ref(column: Column) -> Column:
    return F.regexp_extract(column, r"([^/:]+)$", 1)


def _ts(column: Column) -> Column:
    return F.to_timestamp(column)


def _coding(df: DataFrame, concept: str, field: str) -> Column:
    if not _has(df.schema, f"{concept}.coding"):
        return F.lit(None).cast("string")
    return _first(F.col(f"{concept}.coding"))[field]


def _us_core(df: DataFrame, url: str) -> Column:
    if not _has(df.schema, "extension.extension.valueCoding"):
        return F.lit("Unknown")
    outer = _first(F.filter("extension", lambda e: e["url"] == url))
    omb = _first(F.filter(outer["extension"], lambda s: s["url"] == "ombCategory"))
    return F.coalesce(omb["valueCoding"]["display"], F.lit("Unknown"))


def patients(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("id").alias("patient_id"),
        _ts(_col(df, "birthDate")).alias("birth_date"),
        F.coalesce(_col(df, "gender"), F.lit("unknown")).alias("sex"),
        _us_core(df, RACE_URL).alias("race"),
        _us_core(df, ETHNICITY_URL).alias("ethnicity"),
        _ts(_col(df, "deceasedDateTime")).alias("deceased_at"),
    )


def encounters(df: DataFrame) -> DataFrame:
    encounter_type = _first(F.col("type")) if _has(df.schema, "type") else None
    type_coding = _first(encounter_type["coding"]) if encounter_type is not None else None
    return df.select(
        F.col("id").alias("encounter_id"),
        _ref(F.col("subject.reference")).alias("patient_id"),
        _col(df, "class.code").alias("class_code"),
        (type_coding["code"] if type_coding is not None else F.lit(None)).alias("type_code"),
        F.coalesce(
            type_coding["display"] if type_coding is not None else F.lit(None), F.lit("")
        ).alias("type_display"),
        _col(df, "status").alias("status"),
        _ts(_col(df, "period.start")).alias("start"),
        _ts(_col(df, "period.end")).alias("end"),
    )


def conditions(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("id").alias("condition_id"),
        _ref(F.col("subject.reference")).alias("patient_id"),
        _ref(_col(df, "encounter.reference")).alias("encounter_id"),
        _coding(df, "code", "system").alias("system"),
        _coding(df, "code", "code").alias("code"),
        F.coalesce(_coding(df, "code", "display"), F.lit("")).alias("display"),
        _coding(df, "clinicalStatus", "code").alias("clinical_status"),
        _ts(F.coalesce(_col(df, "onsetDateTime"), _col(df, "recordedDate"))).alias("onset"),
        _ts(F.coalesce(_col(df, "recordedDate"), _col(df, "onsetDateTime"))).alias("recorded"),
        _ts(_col(df, "abatementDateTime")).alias("abatement"),
    )


def observations(df: DataFrame) -> DataFrame:
    base = [
        _ref(F.col("subject.reference")).alias("patient_id"),
        _ref(_col(df, "encounter.reference")).alias("encounter_id"),
        _ts(F.coalesce(_col(df, "effectiveDateTime"), _col(df, "issued"))).alias("effective"),
    ]
    parts = []
    if _has(df.schema, "valueQuantity"):
        parts.append(
            df.where(F.col("valueQuantity").isNotNull()).select(
                F.col("id").alias("observation_id"),
                _coding(df, "code", "code").alias("code"),
                _coding(df, "code", "display").alias("display"),
                F.col("valueQuantity.value").cast("double").alias("value"),
                F.col("valueQuantity.unit").alias("unit"),
                *base,
            )
        )
    if _has(df.schema, "component.valueQuantity"):
        exploded = df.select("*", F.explode("component").alias("comp")).where(
            F.col("comp.valueQuantity").isNotNull()
        )
        parts.append(
            exploded.select(
                F.col("id").alias("observation_id"),
                _first(F.col("comp.code.coding"))["code"].alias("code"),
                _first(F.col("comp.code.coding"))["display"].alias("display"),
                F.col("comp.valueQuantity.value").cast("double").alias("value"),
                F.col("comp.valueQuantity.unit").alias("unit"),
                *[
                    _ref(F.col("subject.reference")).alias("patient_id"),
                    _ref(_col(exploded, "encounter.reference")).alias("encounter_id"),
                    _ts(
                        F.coalesce(_col(exploded, "effectiveDateTime"), _col(exploded, "issued"))
                    ).alias("effective"),
                ],
            )
        )
    result = parts[0]
    for part in parts[1:]:
        result = result.unionByName(part)
    return result


def medications(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("id").alias("medication_request_id"),
        _ref(F.col("subject.reference")).alias("patient_id"),
        _ref(_col(df, "encounter.reference")).alias("encounter_id"),
        _coding(df, "medicationCodeableConcept", "code").alias("code"),
        F.coalesce(
            _coding(df, "medicationCodeableConcept", "display"),
            _col(df, "medicationReference.display"),
            F.lit(""),
        ).alias("display"),
        _col(df, "status").alias("status"),
        _ts(_col(df, "authoredOn")).alias("authored_on"),
    )


BUILDERS = {
    "patients": ("Patient", patients),
    "encounters": ("Encounter", encounters),
    "conditions": ("Condition", conditions),
    "observations": ("Observation", observations),
    "medications": ("MedicationRequest", medications),
}


def flatten_ndjson(
    spark: SparkSession, input_dir: str | Path, output_dir: str | Path
) -> dict[str, int]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    counts: dict[str, int] = {}
    for table, (resource_type, builder) in BUILDERS.items():
        source = input_dir / f"{resource_type}.ndjson"
        if not source.exists():
            raise FileNotFoundError(f"missing {source}; expected FHIR Bulk Data NDJSON files")
        frame = builder(spark.read.json(str(source))).select(*TABLE_COLUMNS[table])
        frame.write.mode("overwrite").parquet(str(output_dir / f"{table}.parquet"))
        counts[table] = frame.count()
    return counts


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised via CI job
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m readmit.fhir.spark_flatten <input_dir> <output_dir>")
        return 1
    spark = spark_session()
    try:
        print(flatten_ndjson(spark, args[0], args[1]))
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

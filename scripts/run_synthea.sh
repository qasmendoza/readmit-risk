#!/usr/bin/env bash
# Generate a Synthea population as FHIR Bulk Data NDJSON and evaluate the pipeline on it.
# Synthea (https://github.com/synthetichealth/synthea) needs Java 17+. Its 30-day readmissions
# are rare (~4-6%) and are not driven by clinical risk factors, so the quality gate is not
# enforced here: this run validates interoperability, not clinical signal.
set -euo pipefail
PATIENTS="${PATIENTS:-3000}"
VERSION="${SYNTHEA_VERSION:-v3.3.0}"
OUT="data/synthea"
JAR="tools/synthea-${VERSION}.jar"

mkdir -p tools "$OUT"
if [[ ! -f "$JAR" ]]; then
  curl -fsSL -o "$JAR" \
    "https://github.com/synthetichealth/synthea/releases/download/${VERSION}/synthea-with-dependencies.jar"
fi
java -Xmx4g -jar "$JAR" -p "$PATIENTS" -s 42 -cs 42 \
  --exporter.baseDirectory="$OUT" --exporter.years_of_history=10 \
  --exporter.fhir.bulk_data=true \
  --exporter.hospital.fhir.export=false --exporter.practitioner.fhir.export=false \
  Massachusetts
readmit run --input "$OUT/fhir" --source-label "Synthea ${VERSION} (${PATIENTS} patients)" \
  --no-tracking --set gate.enforce=false \
  --set output.artifacts_dir=artifacts/synthea --set output.reports_dir=reports/synthea

#!/usr/bin/env bash
# Smoke-test a running ReadmitRisk API: readiness, FHIR scoring of a real Synthea record,
# the copilot, metrics, and the SageMaker contract. Usage: scripts/smoke_test.sh [base-url]
set -euo pipefail
BASE="${1:-http://localhost:8080}"
FIXTURE="${2:-tests/fixtures/synthea/synthea_patient_2.json}"
AUTH=()
if [[ -n "${READMIT_API_KEY:-}" ]]; then AUTH=(-H "X-API-Key: ${READMIT_API_KEY}"); fi

for _ in $(seq 1 45); do
  curl -fsS "$BASE/ready" >/dev/null 2>&1 && break
  sleep 2
done
echo "ready:  $(curl -fsS "$BASE/ready")"
curl -fsS -o /dev/null "$BASE/ping" && echo "ping:   ok"

curl -fsS ${AUTH[@]+"${AUTH[@]}"} -X POST "$BASE/v1/fhir/risk-assessment" \
  -H "Content-Type: application/fhir+json" --data @"$FIXTURE" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r["resourceType"] == "RiskAssessment", r
p = r["prediction"][0]
assert 0 <= p["probabilityDecimal"] <= 1
print("fhir:   RiskAssessment", p["probabilityDecimal"], p["qualitativeRisk"]["coding"][0]["code"])'

curl -fsS ${AUTH[@]+"${AUTH[@]}"} -X POST "$BASE/v1/copilot/brief" \
  -H "Content-Type: application/json" --data @"$FIXTURE" | python3 -c '
import json, sys
b = json.load(sys.stdin)
assert b["citations"] and not b["invalid_citations"], b
print("brief: ", b["mode"], "with", len(b["citations"]), "citations")'

curl -fsS -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"features": {"age_years": 84, "prior_admissions_12m": 3, "ed_visits_6m": 3, "chf": 1}}' |
  python3 -c 'import json, sys; r = json.load(sys.stdin); print("invoke:", round(r["risk"], 3), r["tier"])'
curl -fsS "$BASE/metrics" | grep -q readmit_predictions_total && echo "metrics: ok"
echo "smoke test passed"

# data/

Local working directory for FHIR inputs and derived tables. Everything here except this file is git-ignored.
**Never commit real patient data.**

| Path | Contents | Created by |
|---|---|---|
| `fhir/` | Synthetic FHIR R4 Bulk Data NDJSON (`Patient.ndjson`, `Encounter.ndjson`, ...) and `manifest.json` | `readmit synth` or `readmit run --synthetic` |
| `fhir_drifted/` | A deliberately shifted population for the drift demo | `make drift-demo` |
| `synthea/` | A Synthea v3.3.0 bulk export | `make synthea` |

Any FHIR R4 source works: a directory of Bulk Data NDJSON files (optionally `.ndjson.gz`) or Bundle JSON files.
Point the pipeline at it with `readmit run --input <dir> --source-label "<description>"`.

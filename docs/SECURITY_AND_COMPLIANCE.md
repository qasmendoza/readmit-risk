# Security and compliance

**Scope.** This is a portfolio project that uses synthetic data only. It is not HIPAA-certified or HITRUST-assessed,
and nothing here constitutes legal or compliance advice. It demonstrates engineering patterns that a team handling
protected health information (PHI) would build on. A real deployment would also need a documented risk analysis,
Business Associate Agreements (BAAs) with every vendor that touches PHI, organizational policies, and workforce
training.

## Data handling

- No real patient data exists in the repository, images, or CI. `data/` is git-ignored apart from its README.
- The API returns only risk outputs; it never echoes the submitted record (minimum necessary).
- **Model artifacts are PHI-derived when trained on real data.** `reference_sample.parquet` (the drift-monitoring
  profile) holds patient-level feature rows without identifiers. With real data, treat bundles as sensitive: keep them
  in access-controlled registries and storage, or replace the sample with aggregated histograms.

## Application controls

| Control | Implementation |
|---|---|
| Authentication | Optional API key (`X-API-Key`, constant-time comparison); empty keys never count as credentials |
| Request limits | Configurable body-size limit (HTTP 413) |
| Logging | Access logs record route templates, status, latency, and request ids only, never bodies or query strings |
| Audit trail | One JSON audit event per prediction with keyed-HMAC pseudonymized patient and encounter ids |
| Input validation | Pydantic schemas; unknown features rejected; non-Bundle FHIR input rejected (HTTP 422) |
| Failure behavior | Missing model -> not ready (HTTP 503); LLM failure -> deterministic template brief |

## Infrastructure controls

- Image: multi-stage build, non-root user (UID 10001), no build tools in the runtime stage, pinned dependencies,
  hadolint in CI, Dependabot for base images and actions.
- Kubernetes: Pod Security Standards "restricted" enforced on the namespace; read-only root filesystem; all Linux
  capabilities dropped; no privilege escalation; seccomp `RuntimeDefault`; service-account token not mounted;
  default-deny ingress NetworkPolicy; secrets from a Kubernetes Secret, never from images or git.
- AWS: model archives uploaded with server-side encryption (SSE-KMS when a key is supplied); ECR scan on push.
- Recommended in production: TLS at the ingress, mTLS inside the cluster, Kubernetes Secret encryption at rest with a
  KMS provider, and log shipping to an append-only store with a defined retention period.

## Generative AI controls

- The copilot receives minimum-necessary, de-identified context: age bands (85+ grouped), sex, relative times
  ("12 days before discharge"), coded diagnoses, medications, and labs. Names, addresses, birth dates, and free-text
  notes are never sent.
- Coded displays are used instead of free text, which also limits prompt-injection exposure from clinical notes.
- The system prompt forbids diagnoses and medication changes; every citation is validated against supplied evidence,
  unsupported citations are removed, and their count is exported as a metric.
- With real PHI, only use an LLM endpoint covered by a BAA and your organization's policies.

## Model governance

- Lineage: each bundle records its version, data fingerprint, configuration, hyperparameters, and metrics; MLflow
  stores runs and registered versions.
- Release control: the quality gate blocks models that miss the AUROC threshold or fail to beat LACE (optionally also
  a fairness threshold), in CI and in the image build.
- Transparency: a generated model card and evaluation report accompany every model version.
- Post-deployment: drift and matured-label performance monitoring with explicit retraining triggers.

## Threat model (summary)

| Threat | Mitigation | Status |
|---|---|---|
| PHI exposure in logs | Bodies and query strings never logged; pseudonymized audit ids | Implemented |
| Unauthorized API use | API key; network policies | Implemented (basic) |
| Oversized payload denial of service | Body-size limit | Implemented |
| Container escape or privilege escalation | Non-root, read-only filesystem, dropped capabilities, restricted PSA | Implemented |
| LLM fabricating evidence | Citation validation, grounded prompt, template fallback | Implemented |
| PHI sent to third-party AI | Minimum-necessary context; BAA requirement documented | Partially (policy-dependent) |
| Deploying a degraded model | Quality gate, monitoring | Implemented |
| Supply-chain compromise | Pinned dependencies, Dependabot, hadolint | Partial (no image signing or SBOM yet) |

## Known gaps

Not implemented: OAuth 2.0 / SMART on FHIR backend-services authorization, rate limiting, consent management,
immutable audit-log storage, image signing (for example Sigstore cosign) and SBOM publication, and a formal HITRUST
control mapping.

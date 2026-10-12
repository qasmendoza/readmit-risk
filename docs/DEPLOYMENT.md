# Deployment

One container image serves every runtime: `uvicorn` by default (local Docker and Kubernetes) and `serve` for
Amazon SageMaker. The image build trains a demo model on synthetic FHIR data and **fails if that model does not pass
the quality gate**, so every image that exists is deployable.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `READMIT_MODEL_PATH` | Model bundle directory | `/opt/ml/model`, then `artifacts/model` |
| `READMIT_MODEL_URI` | Load from the MLflow registry instead, e.g. `models:/readmission-risk/3` | unset |
| `READMIT_TRACKING_URI` | MLflow tracking server for `READMIT_MODEL_URI` | unset |
| `READMIT_API_KEY` | Require `X-API-Key` on `/v1/*` | unset (open) |
| `READMIT_AUDIT_SALT` | Key for pseudonymizing identifiers in audit logs | development value (**change it**) |
| `READMIT_MAX_BODY_MB` | Request size limit | `10` |
| `READMIT_LOG_LEVEL` | Log level | `INFO` |
| `ANTHROPIC_API_KEY` | Enable LLM-written copilot briefs | unset (template briefs) |
| `READMIT_LLM_MODEL` | Claude model for the copilot | `claude-haiku-5-5` |

## Local

```bash
make install && make train && make serve       # http://localhost:8080/docs
make smoke                                      # in another terminal
```

## Docker and Compose

```bash
make docker-build && make docker-run            # read-only filesystem, all capabilities dropped
docker compose up --build                       # API only
docker compose --profile tracking up --build    # API + MLflow server on :5000
readmit run --synthetic --set tracking.tracking_uri=http://localhost:5000
```

## MLflow model registry

```bash
readmit run --synthetic --set tracking.register_model=true
mlflow ui --backend-store-uri sqlite:///mlflow.db          # inspect runs, metrics, artifacts
READMIT_MODEL_URI=models:/readmission-risk/1 READMIT_TRACKING_URI=sqlite:///mlflow.db readmit serve
```

Loading from the registry requires MLflow in the serving environment (`pip install "readmit[serve,track]"`); the
default runtime image deliberately excludes it and loads the bundle from disk.

## Kubernetes (local walkthrough with kind)

The manifests are Kustomize overlays (`deploy/k8s/overlays/dev` and `prod`). The namespace enforces the Pod Security
Standards **restricted** profile; pods run non-root with a read-only root filesystem and no capabilities, ingress is
default-deny, and a weekly CronJob runs drift monitoring.

```bash
kind create cluster --name readmit
docker build -t readmit-risk:local .
kind load docker-image readmit-risk:local --name readmit
kubectl apply -k deploy/k8s/overlays/dev
kubectl -n readmit create secret generic readmit-secrets \
  --from-literal=READMIT_API_KEY="$(openssl rand -hex 16)" \
  --from-literal=READMIT_AUDIT_SALT="$(openssl rand -hex 32)"
kubectl -n readmit rollout restart deployment/readmit-api
kubectl -n readmit rollout status deployment/readmit-api
kubectl -n readmit port-forward svc/readmit-api 8080:80
READMIT_API_KEY="$(kubectl -n readmit get secret readmit-secrets -o jsonpath='{.data.READMIT_API_KEY}' | base64 -d)" \
  bash scripts/smoke_test.sh
```

Notes:

- The HPA needs metrics-server (not installed in kind by default).
- NetworkPolicies are enforced only by CNIs that support them (Calico, Cilium). To admit traffic from an ingress
  controller or Prometheus, label its namespace `readmit.io/allow-ingress=true`.
- The `prod` overlay pins the image tag, runs three replicas, and makes the `readmit-secrets` Secret mandatory.
- The drift CronJob reads the latest FHIR extract from `/data/fhir/current` on the `readmit-data` volume and writes
  reports to `/data/reports/monitoring`. It exits with code 2 on drift, so a failed Job is the alert signal
  (for example via kube-state-metrics `kube_job_status_failed`). Run it on demand with
  `kubectl -n readmit create job drift-now --from=cronjob/readmit-drift-monitor`.

## Amazon SageMaker Serverless Inference

The image already implements SageMaker's bring-your-own-container contract (`GET /ping`, `POST /invocations` on
port 8080, model files in `/opt/ml/model`). Serverless endpoints bill per request and scale to zero; still, **tear
everything down when you are done**.

```bash
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/readmit-risk:0.1.0
BUCKET=readmit-sagemaker-$ACCOUNT-$REGION    # "sagemaker" in the name lets the managed policy read it

# 1. Push the image (SageMaker needs linux/amd64 and a plain image manifest, hence --provenance=false)
aws ecr create-repository --repository-name readmit-risk --region $REGION \
  --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region $REGION |
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker buildx build --platform linux/amd64 --provenance=false -t $IMAGE --load .
docker push $IMAGE

# 2. Execution role (for a demo account; scope it down for anything real)
cat > trust.json <<'JSON'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
JSON
aws iam create-role --role-name readmit-sagemaker-execution --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name readmit-sagemaker-execution \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# 3. Package the trained bundle, upload it encrypted, create model + serverless endpoint
pip install -e ".[aws]"
aws s3 mb s3://$BUCKET --region $REGION
readmit sagemaker deploy --image-uri $IMAGE --bucket $BUCKET --region $REGION \
  --role-arn arn:aws:iam::$ACCOUNT:role/readmit-sagemaker-execution

# 4. Score a real Synthea patient (returns a FHIR RiskAssessment); the first call includes a cold start
readmit sagemaker invoke --endpoint readmission-risk --region $REGION \
  --payload tests/fixtures/synthea/synthea_patient_2.json

# 5. Clean up
readmit sagemaker teardown --endpoint readmission-risk --region $REGION
aws s3 rb s3://$BUCKET --force
aws ecr delete-repository --repository-name readmit-risk --region $REGION --force
```

Pass `--kms-key-id` to `deploy` to upload the model archive with SSE-KMS instead of SSE-S3.

## Container registry releases

`.github/workflows/release.yml` pushes `ghcr.io/<owner>/readmit-risk` on every push to `main` (`latest`, `sha-…`) and
on version tags (`git tag v0.1.0 && git push --tags`). New GHCR packages may be private; change the visibility in the
package settings to make the image public.

## Observability

Prometheus scrapes `/metrics` (pod annotations are set). Useful series: `readmit_request_latency_seconds`,
`readmit_requests_total{status}`, `readmit_predictions_total{tier}`, `readmit_predicted_risk` (a live view of
prediction drift), `readmit_copilot_briefs_total{mode}`, and `readmit_copilot_invalid_citations_total`, which should
stay at zero. Sensible alerts: p95 latency, 5xx rate, any invalid citations, a shift in the share of high-risk
predictions, and a failed drift-monitor Job.

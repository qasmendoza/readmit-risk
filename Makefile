.DEFAULT_GOAL := help
PYTHON ?= python
IMAGE ?= readmit-risk:local
PATIENTS ?= 3000

.PHONY: help install lint format test data train results synthea drift-demo serve mlflow-ui \
        docker-build docker-run smoke k8s-validate clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install with the exact dependency versions the tests passed with
	$(PYTHON) -m pip install -c constraints.txt -e ".[dev]"

lint: ## Lint and check formatting
	ruff check . && ruff format --check .

format: ## Auto-format and apply safe lint fixes
	ruff format . && ruff check --fix .

test: ## Run the test suite with coverage
	pytest --cov=readmit --cov-report=term-missing

data: ## Generate a synthetic FHIR R4 population (Bulk Data NDJSON)
	readmit synth --patients $(PATIENTS) --out data/fhir

train: ## Train, evaluate, and quality-gate the model on synthetic FHIR data
	readmit run --synthetic --patients $(PATIENTS)

results: ## Train and refresh the Results section of the README
	readmit run --synthetic --patients $(PATIENTS) --no-tracking --update-readme

synthea: ## Generate a Synthea bulk export (needs Java 17+) and evaluate on it
	bash scripts/run_synthea.sh

drift-demo: ## Simulate a shifted population and run drift monitoring
	readmit synth --patients 1000 --drift --out data/fhir_drifted
	readmit monitor --input data/fhir_drifted --output reports/monitoring

serve: ## Run the API on http://localhost:8080 (docs at /docs)
	readmit serve

mlflow-ui: ## Browse local MLflow runs
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build: ## Build the image (trains and quality-gates a demo model during the build)
	docker build -t $(IMAGE) .

docker-run: ## Run the image hardened: read-only filesystem, no capabilities
	docker run --rm -p 8080:8080 --read-only --tmpfs /tmp --cap-drop ALL $(IMAGE)

smoke: ## Smoke-test a running API
	bash scripts/smoke_test.sh http://localhost:8080

k8s-validate: ## Render and schema-validate the Kubernetes overlays
	for overlay in dev prod; do \
		kustomize build deploy/k8s/overlays/$$overlay | kubeconform -strict -summary - || exit 1; \
	done

clean: ## Remove generated data, models, and caches
	rm -rf data/fhir data/fhir_drifted data/synthea data/tables artifacts mlruns mlflow.db \
		reports/monitoring .pytest_cache .ruff_cache .coverage coverage.xml

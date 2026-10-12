# syntax=docker/dockerfile:1
# One image for Kubernetes and Amazon SageMaker (BYOC): `uvicorn` by default, `serve` for SageMaker.
ARG PYTHON_VERSION=3.12

# ---- build: create the wheel and train a quality-gated demo model on synthetic FHIR data -------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
# Versions are pinned through constraints.txt (the exact set the test suite passed with).
# hadolint ignore=DL3013
RUN pip wheel --no-deps --wheel-dir /wheels . \
 && pip install --constraint constraints.txt .
COPY configs ./configs
COPY knowledge ./knowledge
ARG SYNTHETIC_PATIENTS=1500
# The build fails if the model does not pass the quality gate (exit code 3).
RUN readmit run --synthetic --patients "${SYNTHETIC_PATIENTS}" --no-tracking --no-plots \
 && rm -rf data reports

# ---- runtime: slim, non-root, model baked in (override with READMIT_MODEL_PATH/URI) -----------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 READMIT_MODEL_PATH=/app/artifacts/model
WORKDIR /app
COPY constraints.txt ./
COPY --from=builder /wheels /wheels
# hadolint ignore=DL3013
RUN pip install --constraint constraints.txt --find-links /wheels "readmit[serve]" \
 && rm -rf /wheels \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=builder /build/artifacts/model ./artifacts/model
COPY --chmod=0755 docker/serve /usr/local/bin/serve
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
CMD ["uvicorn", "readmit.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]

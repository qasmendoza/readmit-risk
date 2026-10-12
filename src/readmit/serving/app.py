"""ReadmitRisk API.

Routes
------
GET  /health                        liveness
GET  /ready                         readiness (model loaded)
GET  /v1/model                      model version, metrics, thresholds, features
POST /v1/predict                    score precomputed features
POST /v1/fhir/risk-assessment       FHIR Bundle in -> FHIR RiskAssessment out
POST /v1/copilot/brief              FHIR Bundle in -> grounded care-transition brief
GET  /metrics                       Prometheus metrics
GET  /ping, POST /invocations       Amazon SageMaker hosting contract (same image as Kubernetes)

Environment
-----------
READMIT_MODEL_PATH   bundle directory (defaults: /opt/ml/model, then artifacts/model)
READMIT_MODEL_URI    optional MLflow model URI, e.g. models:/readmission-risk/3
READMIT_API_KEY      if set, /v1/* routes require the X-API-Key header
READMIT_AUDIT_SALT   key for pseudonymizing identifiers in audit logs
READMIT_MAX_BODY_MB  request size limit (default 10)
ANTHROPIC_API_KEY    enables LLM-generated briefs (template briefs otherwise)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

import readmit
from readmit.bundle import ModelBundle, find_bundle_dir
from readmit.cohort import EncounterNotFoundError
from readmit.copilot.assistant import make_llm_client
from readmit.service import ScoringService
from readmit.serving.schemas import BriefResponse, FeaturesRequest, ModelInfo, PredictResponse
from readmit.serving.security import api_key_guard, audit, pseudonymize

logger = logging.getLogger("readmit.api")
FHIR_JSON = "application/fhir+json"


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "readmit_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "readmit_request_latency_seconds", "Request latency", ["route"], registry=self.registry
        )
        self.predictions = Counter(
            "readmit_predictions_total",
            "Predictions by tier",
            ["endpoint", "tier"],
            registry=self.registry,
        )
        self.risk = Histogram(
            "readmit_predicted_risk",
            "Predicted risk distribution",
            buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
            registry=self.registry,
        )
        self.briefs = Counter(
            "readmit_copilot_briefs_total", "Copilot briefs", ["mode"], registry=self.registry
        )
        self.citation_violations = Counter(
            "readmit_copilot_invalid_citations_total",
            "Unsupported citations removed from LLM output",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "readmit_model_info", "Loaded model", ["version", "model"], registry=self.registry
        )


def resolve_model_dir(explicit: str | Path | None = None) -> Path | None:
    """Locate the model bundle.

    An explicitly configured location (argument, then ``READMIT_MODEL_PATH``) is used exclusively:
    if it holds no bundle the service reports not-ready rather than silently serving some other
    model. The default search path applies only when nothing is configured.
    """
    if explicit is not None:
        return find_bundle_dir([explicit])
    configured = os.environ.get("READMIT_MODEL_PATH")
    if configured:
        return find_bundle_dir([configured])
    return find_bundle_dir(["/opt/ml/model", "artifacts/model", "/app/artifacts/model"])


def load_bundle(model_dir: str | Path | None = None) -> ModelBundle:
    uri = os.environ.get("READMIT_MODEL_URI")
    if uri and model_dir is None:
        from readmit.tracking import load_bundle_from_mlflow

        return load_bundle_from_mlflow(uri, os.environ.get("READMIT_TRACKING_URI"))
    directory = resolve_model_dir(model_dir)
    if directory is None:
        raise FileNotFoundError(
            "no model bundle found; set READMIT_MODEL_PATH or run `readmit run`"
        )
    return ModelBundle.load(directory)


def create_app(
    model_dir: str | Path | None = None,
    llm_client: Any | None = None,
    api_key: str | None = None,
    max_body_mb: float | None = None,
) -> FastAPI:
    metrics = Metrics()
    state: dict[str, Any] = {"service": None, "error": None}
    # An empty string never counts as a credential: treat it as "authentication disabled".
    key = (api_key if api_key is not None else os.environ.get("READMIT_API_KEY")) or None
    limit = int((max_body_mb or float(os.environ.get("READMIT_MAX_BODY_MB", "10"))) * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            bundle = load_bundle(model_dir)
            client = llm_client if llm_client is not None else make_llm_client()
            state["service"] = ScoringService(bundle, llm_client=client)
            metrics.model_info.labels(bundle.version, bundle.model_name).set(1)
            logger.info("loaded model %s (%s)", bundle.version, bundle.model_name)
        except Exception as exc:  # keep serving /health; /ready reports the failure
            state["error"] = str(exc)
            logger.error("model failed to load: %s", exc)
        yield

    app = FastAPI(
        title="ReadmitRisk API",
        version=readmit.__version__,
        description="FHIR-native 30-day readmission risk scoring with a grounded care-transition "
        "copilot. Decision support only; developed on synthetic data.",
        lifespan=lifespan,
    )
    guard = Depends(api_key_guard(key))

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > limit:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            template = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - started
            metrics.requests.labels(request.method, template, str(status_code)).inc()
            metrics.latency.labels(template).observe(elapsed)
            logger.info(
                "%s %s %s %.1fms id=%s",
                request.method,
                template,
                status_code,
                elapsed * 1000,
                request_id,
            )

    def service() -> ScoringService:
        if state["service"] is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")
        return state["service"]

    def record(
        endpoint: str,
        request: Request,
        tier: str,
        risk: float,
        patient_id: str | None = None,
        encounter_id: str | None = None,
    ) -> None:
        metrics.predictions.labels(endpoint, tier).inc()
        metrics.risk.observe(risk)
        audit(
            "prediction",
            endpoint=endpoint,
            request_id=getattr(request.state, "request_id", None),
            patient=pseudonymize(patient_id),
            encounter=pseudonymize(encounter_id),
            model_version=service().bundle.version,
            tier=tier,
        )

    def score_fhir(payload: dict[str, Any], encounter_id: str | None):
        if payload.get("resourceType") != "Bundle":
            raise HTTPException(422, "body must be a FHIR Bundle resource")
        try:
            return service().score_bundle(payload, encounter_id)
        except EncounterNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"could not process Bundle: {exc}") from exc

    def predict_features(request_body: FeaturesRequest) -> PredictResponse:
        svc = service()
        scored = svc.bundle.score(request_body.features)
        return PredictResponse(
            model_version=svc.bundle.version,
            model_name=svc.bundle.model_name,
            risk=scored["risk"],
            tier=scored["tier"],
            thresholds=svc.bundle.thresholds,
            factors=scored["factors"],
        )

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    def ready() -> JSONResponse:
        if state["service"] is None:
            return JSONResponse({"status": "not ready", "error": state["error"]}, status_code=503)
        return JSONResponse({"status": "ready", "model_version": state["service"].bundle.version})

    @app.get("/metrics", tags=["operations"])
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", response_model=ModelInfo, dependencies=[guard], tags=["model"])
    def model_info() -> ModelInfo:
        bundle = service().bundle
        meta = bundle.metadata
        return ModelInfo(
            version=bundle.version,
            model_name=bundle.model_name,
            created_at=meta.get("created_at"),
            data_source=meta.get("data_source"),
            test_metrics={k: float(v) for k, v in meta.get("test_metrics", {}).items()},
            confidence_intervals=meta.get("confidence_intervals", {}),
            thresholds=bundle.thresholds,
            operating_point=meta.get("operating_point", {}),
            gate=meta.get("gate", {}),
            features=meta.get("features", []),
        )

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[guard], tags=["scoring"])
    def predict(body: FeaturesRequest, request: Request) -> PredictResponse:
        response = predict_features(body)
        record("predict", request, response.tier, response.risk)
        return response

    @app.post(
        "/v1/fhir/risk-assessment",
        dependencies=[guard],
        tags=["scoring"],
        responses={200: {"content": {FHIR_JSON: {}}}},
    )
    def fhir_risk_assessment(
        request: Request,
        payload: dict[str, Any] = Body(...),
        encounter_id: str | None = Query(default=None),
    ) -> JSONResponse:
        scored = score_fhir(payload, encounter_id)
        record("fhir", request, scored.tier, scored.risk, scored.patient_id, scored.encounter_id)
        return JSONResponse(service().risk_assessment(scored), media_type=FHIR_JSON)

    @app.post(
        "/v1/copilot/brief", response_model=BriefResponse, dependencies=[guard], tags=["copilot"]
    )
    def copilot_brief(
        request: Request,
        payload: dict[str, Any] = Body(...),
        encounter_id: str | None = Query(default=None),
    ) -> BriefResponse:
        scored = score_fhir(payload, encounter_id)
        brief = service().brief(scored)
        record("copilot", request, scored.tier, scored.risk, scored.patient_id, scored.encounter_id)
        metrics.briefs.labels(brief.mode).inc()
        if brief.invalid_citations:
            metrics.citation_violations.inc(len(brief.invalid_citations))
        return BriefResponse(
            model_version=service().bundle.version,
            encounter_id=scored.encounter_id,
            risk=scored.risk,
            tier=scored.tier,
            factors=scored.factors,
            brief=brief.text,
            mode=brief.mode,
            citations=brief.citations,
            invalid_citations=brief.invalid_citations,
            guidance=brief.guidance,
        )

    @app.get("/ping", tags=["sagemaker"])
    def ping() -> Response:
        return Response(status_code=200 if state["service"] is not None else 503)

    @app.post("/invocations", tags=["sagemaker"])
    def invocations(request: Request, payload: dict[str, Any] = Body(...)) -> JSONResponse:
        if payload.get("resourceType") == "Bundle":
            scored = score_fhir(payload, payload.get("encounter_id"))
            record(
                "sagemaker",
                request,
                scored.tier,
                scored.risk,
                scored.patient_id,
                scored.encounter_id,
            )
            return JSONResponse(service().risk_assessment(scored), media_type=FHIR_JSON)
        try:
            body = FeaturesRequest.model_validate(payload)
        except Exception as exc:
            raise HTTPException(
                422, f"expected a FHIR Bundle or {{'features': {{...}}}}: {exc}"
            ) from exc
        response = predict_features(body)
        record("sagemaker", request, response.tier, response.risk)
        return JSONResponse(response.model_dump())

    return app


logging.basicConfig(
    level=os.environ.get("READMIT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = create_app()

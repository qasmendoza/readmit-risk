"""MLflow experiment tracking and model registry.

Each run logs configuration, cross-validation and held-out metrics for every candidate, bootstrap
confidence intervals, fairness gaps, the quality-gate result, reports, and the model bundle as an
MLflow ``pyfunc`` model (input: feature table; output: calibrated risk and tier). With
``tracking.register_model: true`` the model is registered so deployments can reference
``models:/readmission-risk/<version>``.
"""

from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

import readmit
from readmit.bundle import ModelBundle
from readmit.features import FEATURE_SPECS, LACE_COLUMN

if TYPE_CHECKING:  # pragma: no cover
    from readmit.pipeline import PipelineResult

logger = logging.getLogger(__name__)
_RUNTIME = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "joblib",
    "pyarrow",
    "pydantic",
    "pyyaml",
]

# Counts, flags, and scores are integers; measurements are doubles. Typing them this way lets
# JSON clients send natural values (3 ED visits, 72.5 years) without MLflow rejecting the payload.
SIGNATURE = ModelSignature(
    inputs=Schema(
        [
            ColSpec("double" if s.kind == "continuous" else "long", s.name, required=False)
            for s in FEATURE_SPECS
        ]
        + [ColSpec("long", LACE_COLUMN, required=False)]
    ),
    outputs=Schema([ColSpec("double", "risk"), ColSpec("string", "tier")]),
)


class ReadmissionPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle_dir"])

    def predict(self, context, model_input, params=None):
        risk = self.bundle.predict_proba(pd.DataFrame(model_input))
        return pd.DataFrame({"risk": risk, "tier": [self.bundle.tier(float(p)) for p in risk]})


def _requirements() -> list[str]:
    pins = []
    for package in _RUNTIME:
        try:
            pins.append(f"{package}=={importlib_metadata.version(package)}")
        except importlib_metadata.PackageNotFoundError:
            pins.append(package)
    return pins


def _metrics(result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, candidate in result.candidates.items():
        for key, value in candidate.cv.items():
            metrics[f"cv.{name}.{key}"] = value["mean"]
        for key in ("roc_auc", "average_precision", "brier", "calibration_slope", "ece"):
            metrics[f"test.{name}.{key}"] = candidate.test_metrics[key]
    for key, ci in result.confidence_intervals.items():
        metrics[f"best.{key}"] = ci["estimate"]
        metrics[f"best.{key}.ci_low"] = ci["low"]
        metrics[f"best.{key}.ci_high"] = ci["high"]
    for key in ("sensitivity", "ppv", "flag_rate"):
        metrics[f"operating_point.{key}"] = float(result.operating_point[key])
    for summary in result.fairness_summary:
        if "auroc_gap" in summary:
            metrics[f"fairness.{summary['attribute']}.auroc_gap"] = summary["auroc_gap"]
    metrics["copilot.retrieval_hit_rate"] = result.retrieval_eval["hit_rate"]
    return {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN


def log_run(result: PipelineResult) -> str:
    cfg = result.config
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment)
    with mlflow.start_run(run_name=f"{result.best_model}-{result.bundle.version}") as run:
        mlflow.set_tags(
            {
                "readmit.version": readmit.__version__,
                "model.version": result.bundle.version,
                "model.selected": result.best_model,
                "data.source": result.data_source,
                "data.fingerprint": result.bundle.metadata["data_fingerprint"],
                "gate.passed": str(result.gate["passed"]),
            }
        )
        params: dict[str, Any] = {
            "candidates": ",".join(cfg.models.candidates),
            "selection_metric": cfg.models.selection_metric,
            "calibration": cfg.models.calibration,
            "monotonic_constraints": cfg.models.monotonic_constraints,
            "cv_folds": cfg.split.cv_folds,
            "test_fraction": cfg.split.test_fraction,
            "capacity_fraction": cfg.evaluation.capacity_fraction,
            "index_stays": len(result.dataset),
            "readmission_rate": round(result.cohort.readmission_rate, 4),
            **{f"best.{k}": v for k, v in result.candidates[result.best_model].best_params.items()},
        }
        mlflow.log_params(params)
        mlflow.log_metrics(_metrics(result))
        if cfg.output.reports_dir.exists():
            mlflow.log_artifacts(str(cfg.output.reports_dir), artifact_path="reports")
        metrics_file = cfg.output.artifacts_dir / "metrics.json"
        if metrics_file.exists():
            mlflow.log_artifact(str(metrics_file), artifact_path="outputs")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=ReadmissionPyfunc(),
            artifacts={"bundle_dir": str(cfg.model_dir)},
            code_paths=[str(Path(readmit.__file__).parent)],
            signature=SIGNATURE,
            pip_requirements=_requirements(),
            registered_model_name=cfg.tracking.model_name if cfg.tracking.register_model else None,
        )
        logger.info("logged MLflow run %s", run.info.run_id)
        return run.info.run_id


def load_bundle_from_mlflow(model_uri: str, tracking_uri: str | None = None) -> ModelBundle:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().bundle

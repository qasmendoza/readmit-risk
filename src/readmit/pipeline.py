"""End-to-end training pipeline.

FHIR -> tables -> cohort + labels -> point-in-time features -> Table 1 -> temporal split with
embargo -> patient-grouped CV tuning of every candidate -> calibrated fits -> held-out evaluation
(cluster-bootstrap CIs, DeLong tests, calibration, decision curve, fairness audit, permutation
importance) -> quality gate -> model bundle, reports, and (optionally) MLflow.

If the quality gate fails, reports are still written so the failure can be diagnosed, but the
model bundle is *not* saved, so a blocked model can never be deployed by accident.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import readmit
from readmit.analysis import table_one
from readmit.bundle import REFERENCE_FILE, ModelBundle
from readmit.cohort import Cohort, build_cohort, build_episodes
from readmit.config import Config
from readmit.copilot.knowledge import load_guidance
from readmit.copilot.retriever import TfidfRetriever, evaluate_retrieval
from readmit.evaluation.decision import capacity_threshold, decision_curve, operating_point
from readmit.evaluation.delong import delong_test
from readmit.evaluation.fairness import fairness_report, max_auroc_gap
from readmit.evaluation.metrics import (
    calibration_curve_points,
    classification_metrics,
    cluster_bootstrap_ci,
)
from readmit.explain import permutation_importance_table
from readmit.features import FEATURE_NAMES, FEATURE_SPECS, compute_features
from readmit.fhir.flatten import FhirTables, flatten_path
from readmit.models import (
    INPUT_COLUMNS,
    CandidateResult,
    TemporalSplit,
    fit_calibrated,
    temporal_split,
    tune_candidate,
)

logger = logging.getLogger(__name__)
MIN_INDEX_STAYS = 200
MIN_POSITIVES = 30


class QualityGateError(RuntimeError):
    """Raised when the selected model fails a deployment requirement."""


@dataclass
class PipelineResult:
    config: Config
    data_source: str
    tables_summary: dict[str, int]
    cohort: Cohort
    dataset: pd.DataFrame
    split: TemporalSplit
    table1: pd.DataFrame
    table1_meta: dict[str, Any]
    candidates: dict[str, CandidateResult]
    best_model: str
    test_metrics: dict[str, Any]
    confidence_intervals: dict[str, dict[str, float]]
    comparisons: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    operating_point: dict[str, Any]
    decision_curve: pd.DataFrame
    calibration_points: list[dict[str, float]]
    fairness_table: pd.DataFrame
    fairness_summary: list[dict[str, Any]]
    importance: pd.DataFrame
    retrieval_eval: dict[str, float]
    gate: dict[str, Any]
    bundle: ModelBundle
    paths: dict[str, Path] = field(default_factory=dict)
    run_id: str | None = None


def fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def evaluate_gate(
    cfg: Config,
    best: str,
    candidates: dict[str, CandidateResult],
    comparisons: dict[str, dict[str, float]],
    fairness_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = []
    auroc = candidates[best].test_metrics["roc_auc"]
    checks.append(
        {
            "name": "minimum_auroc",
            "passed": bool(auroc >= cfg.gate.min_auroc),
            "detail": f"test AUROC {auroc:.3f} (required >= {cfg.gate.min_auroc})",
        }
    )
    baseline = cfg.gate.baseline
    if cfg.gate.require_beats_baseline and baseline in candidates:
        if best == baseline:
            checks.append(
                {
                    "name": "beats_baseline",
                    "passed": False,
                    "detail": f"the selected model is the {baseline} baseline itself",
                }
            )
        else:
            comparison = comparisons[baseline]
            checks.append(
                {
                    "name": "beats_baseline",
                    "passed": bool(comparison["difference"] > 0),
                    "detail": f"AUROC {comparison['difference']:+.3f} vs {baseline} "
                    f"(DeLong p = {comparison['p_value']:.3g})",
                }
            )
    if cfg.gate.max_fairness_auroc_gap is not None:
        gap = max_auroc_gap(fairness_summary)
        checks.append(
            {
                "name": "fairness_auroc_gap",
                "passed": bool(gap <= cfg.gate.max_fairness_auroc_gap),
                "detail": f"largest subgroup AUROC gap {gap:.3f} "
                f"(allowed <= {cfg.gate.max_fairness_auroc_gap})",
            }
        )
    return {
        "passed": all(c["passed"] for c in checks),
        "enforced": cfg.gate.enforce,
        "checks": checks,
    }


def run_pipeline(
    cfg: Config,
    input_path: str | Path | None = None,
    tables: FhirTables | None = None,
    data_source: str = "real",
    make_plots: bool = True,
    track: bool | None = None,
) -> PipelineResult:
    if tables is None:
        source = Path(input_path or cfg.data.input_dir)
        logger.info("flattening FHIR resources from %s", source)
        tables = flatten_path(source)
    logger.info("tables: %s", tables.summary())

    cohort = build_cohort(tables, cfg.cohort)
    n_positive = int(cohort.index["readmitted_30d"].sum()) if len(cohort.index) else 0
    if len(cohort.index) < MIN_INDEX_STAYS or n_positive < MIN_POSITIVES:
        raise ValueError(
            f"cohort too small to train ({len(cohort.index)} index stays, {n_positive} "
            f"readmissions; need >= {MIN_INDEX_STAYS} and >= {MIN_POSITIVES})"
        )
    all_episodes = build_episodes(
        tables.encounters, cfg.cohort.exclude_encounter_types, cfg.cohort.transfer_gap_hours
    )
    features = compute_features(tables, cohort.index, cfg.features, cfg.cohort, all_episodes)
    overlap = [c for c in features.columns if c in cohort.index.columns]
    dataset = cohort.index.drop(columns=overlap).set_index("episode_id").join(features)
    # Hypothesis tests assume independent observations, but patients contribute several stays.
    # Table 1 uses one randomly selected stay per patient (first stays would be biased toward
    # low risk because they have the least prior history).
    one_per_patient = dataset.sample(frac=1.0, random_state=cfg.seed).groupby("patient_id").head(1)
    table1, table1_meta = table_one(one_per_patient)

    split = temporal_split(dataset["discharge"], cfg.split.test_fraction, cfg.cohort.followup_days)
    train, test = dataset.iloc[split.train_index], dataset.iloc[split.test_index]
    for name, part in (("train", train), ("test", test)):
        if part["readmitted_30d"].nunique() < 2:
            raise ValueError(f"{name} split contains a single class; use more data")
    X_train, y_train = train[INPUT_COLUMNS], train["readmitted_30d"].to_numpy()
    X_test, y_test = test[INPUT_COLUMNS], test["readmitted_30d"].to_numpy()
    groups = train["patient_id"].to_numpy()

    candidates: dict[str, CandidateResult] = {}
    for name in cfg.models.candidates:
        logger.info("tuning %s with %d-fold patient-grouped CV", name, cfg.split.cv_folds)
        result = tune_candidate(
            name,
            X_train,
            y_train,
            groups,
            cfg.split.cv_folds,
            cfg.models.selection_metric,
            cfg.models.monotonic_constraints,
            cfg.seed,
        )
        result.model = fit_calibrated(
            name,
            result.best_params,
            X_train,
            y_train,
            groups,
            cfg.split.cv_folds,
            cfg.models.calibration,
            cfg.models.monotonic_constraints,
            cfg.seed,
        )
        result.train_proba = result.model.predict_proba(X_train)[:, 1]
        result.test_proba = result.model.predict_proba(X_test)[:, 1]
        result.test_metrics = classification_metrics(y_test, result.test_proba)
        candidates[name] = result

    best = max(candidates, key=lambda n: candidates[n].cv[cfg.models.selection_metric]["mean"])
    chosen = candidates[best]
    logger.info(
        "selected %s (CV %s = %.4f)",
        best,
        cfg.models.selection_metric,
        chosen.cv[cfg.models.selection_metric]["mean"],
    )

    clusters = test["patient_id"].to_numpy()
    confidence_intervals = {
        metric: cluster_bootstrap_ci(
            y_test,
            chosen.test_proba,
            clusters,
            metric,
            cfg.evaluation.bootstrap_iterations,
            cfg.evaluation.alpha,
            cfg.seed,
        )
        for metric in ("roc_auc", "average_precision", "brier")
    }
    comparisons = {
        other: delong_test(y_test, chosen.test_proba, result.test_proba)
        for other, result in candidates.items()
        if other != best
    }

    thresholds = {
        "high": capacity_threshold(chosen.train_proba, cfg.evaluation.capacity_fraction),
        "moderate": capacity_threshold(chosen.train_proba, cfg.evaluation.moderate_fraction),
    }
    op = operating_point(y_test, chosen.test_proba, thresholds["high"])
    curve = decision_curve(y_test, chosen.test_proba)
    audit_frame = test[["sex", "race", "ethnicity", "age_band"]].assign(
        y=y_test, p=chosen.test_proba
    )
    fairness_table, fairness_summary = fairness_report(
        audit_frame,
        cfg.evaluation.fairness_attributes,
        thresholds["high"],
        cfg.evaluation.fairness_min_group_size,
    )
    importance = permutation_importance_table(
        chosen.model, X_test, y_test, INPUT_COLUMNS, seed=cfg.seed
    )
    guidance = load_guidance(cfg.copilot.knowledge_dir)
    retrieval = evaluate_retrieval(TfidfRetriever(guidance))
    gate = evaluate_gate(cfg, best, candidates, comparisons, fairness_summary)

    created = datetime.now(UTC)
    data_hash = fingerprint(dataset[[*INPUT_COLUMNS, "readmitted_30d"]])
    metadata: dict[str, Any] = {
        "version": f"{created:%Y%m%d.%H%M%S}-{data_hash[:7]}",
        "created_at": created.isoformat(timespec="seconds"),
        "readmit_version": readmit.__version__,
        "python_version": platform.python_version(),
        "model_name": best,
        "best_params": chosen.best_params,
        "calibration": cfg.models.calibration,
        "selection_metric": cfg.models.selection_metric,
        "cv": {name: r.cv for name, r in candidates.items()},
        "test_metrics": chosen.test_metrics,
        "confidence_intervals": confidence_intervals,
        "comparisons": comparisons,
        "thresholds": thresholds,
        "operating_point": op,
        "capacity_fraction": cfg.evaluation.capacity_fraction,
        "fairness_summary": fairness_summary,
        "gate": gate,
        "data_source": data_source,
        "data_fingerprint": data_hash,
        "cohort_flow": cohort.flow,
        "split": {
            "cutoff": str(split.cutoff),
            "embargo_days": split.embargo_days,
            "train": len(train),
            "test": len(test),
            "embargoed": split.embargoed,
            "train_prevalence": float(y_train.mean()),
            "test_prevalence": float(y_test.mean()),
        },
        "features": [spec.to_dict() for spec in FEATURE_SPECS],
        "cohort_config": cfg.cohort.model_dump(mode="json"),
        "feature_config": cfg.features.model_dump(mode="json"),
        "retrieval_eval": retrieval,
    }
    bundle = ModelBundle(
        model=chosen.model,
        model_name=best,
        reference_values={
            k: float(v) for k, v in X_train[FEATURE_NAMES].median().fillna(0.0).items()
        },
        thresholds=thresholds,
        metadata=metadata,
        guidance_docs=guidance,
    )
    result = PipelineResult(
        config=cfg,
        data_source=data_source,
        tables_summary=tables.summary(),
        cohort=cohort,
        dataset=dataset,
        split=split,
        table1=table1,
        table1_meta=table1_meta,
        candidates=candidates,
        best_model=best,
        test_metrics=chosen.test_metrics,
        confidence_intervals=confidence_intervals,
        comparisons=comparisons,
        thresholds=thresholds,
        operating_point=op,
        decision_curve=curve,
        calibration_points=calibration_curve_points(y_test, chosen.test_proba),
        fairness_table=fairness_table,
        fairness_summary=fairness_summary,
        importance=importance,
        retrieval_eval=retrieval,
        gate=gate,
        bundle=bundle,
    )

    result.paths = save_outputs(result, save_bundle=gate["passed"] or not cfg.gate.enforce)
    from readmit.reporting import write_reports

    result.paths.update(write_reports(result, make_plots=make_plots))
    if not gate["passed"] and cfg.gate.enforce:
        failed = "; ".join(c["detail"] for c in gate["checks"] if not c["passed"])
        raise QualityGateError(f"quality gate failed, model not saved: {failed}")
    if cfg.tracking.enabled if track is None else track:
        try:
            from readmit.tracking import log_run
        except ImportError:
            logger.warning("mlflow is not installed; skipping experiment tracking")
        else:
            result.run_id = log_run(result)
    return result


def save_outputs(result: PipelineResult, save_bundle: bool = True) -> dict[str, Path]:
    cfg = result.config
    artifacts = cfg.output.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if save_bundle:
        model_dir = result.bundle.save(cfg.model_dir)
        train = result.dataset.iloc[result.split.train_index]
        reference = train[[*FEATURE_NAMES, "sex", "age_band"]].assign(
            risk=result.candidates[result.best_model].train_proba
        )
        if len(reference) > 20_000:
            reference = reference.sample(20_000, random_state=cfg.seed)
        reference.to_parquet(model_dir / REFERENCE_FILE)
        paths["model_dir"] = model_dir
    test = result.dataset.iloc[result.split.test_index]
    predictions = pd.DataFrame(
        {
            "episode_id": test.index,
            "discharge": test["discharge"].to_numpy(),
            "readmitted_30d": test["readmitted_30d"].to_numpy(),
            **{f"risk_{name}": r.test_proba for name, r in result.candidates.items()},
        }
    )
    outputs = {
        "test_predictions.csv": predictions,
        "table1.csv": result.table1,
        "fairness.csv": result.fairness_table,
        "permutation_importance.csv": result.importance,
        "decision_curve.csv": result.decision_curve,
    }
    for name, frame in outputs.items():
        frame.to_csv(artifacts / name, index=False)
        paths[name] = artifacts / name
    summary = {
        "best_model": result.best_model,
        "test_metrics": result.test_metrics,
        "confidence_intervals": result.confidence_intervals,
        "comparisons": result.comparisons,
        "operating_point": result.operating_point,
        "thresholds": result.thresholds,
        "fairness_summary": result.fairness_summary,
        "cv": {n: r.cv for n, r in result.candidates.items()},
        "candidate_test_metrics": {n: r.test_metrics for n, r in result.candidates.items()},
        "cohort_flow": result.cohort.flow,
        "retrieval_eval": result.retrieval_eval,
        "gate": result.gate,
    }
    (artifacts / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    paths["metrics.json"] = artifacts / "metrics.json"
    return paths

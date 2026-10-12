"""Command-line interface: ``readmit <command>``.

  synth      generate a synthetic FHIR R4 population (Bulk NDJSON or Bundles)
  run        train and evaluate (quality-gated); writes artifacts, reports, MLflow run
  monitor    drift and matured-label performance checks against the deployed model
  score      score one patient Bundle (FHIR RiskAssessment, or a copilot brief)
  serve      start the API
  sagemaker  package / deploy / invoke / teardown a SageMaker serverless endpoint

Exit codes: 0 success, 1 error, 2 drift detected (with --fail-on-drift), 3 quality gate failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("readmit")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("matplotlib", "alembic", "urllib3", "httpx", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _has_fhir(path: Path) -> bool:
    return path.exists() and any(
        p.suffix in (".ndjson", ".json", ".gz")
        for p in path.rglob("*")
        if p.name != "manifest.json"
    )


def cmd_synth(args: argparse.Namespace) -> int:
    from readmit.fhir.synthetic import write_population

    manifest = write_population(
        args.out, args.patients, seed=args.seed, fmt=args.format, drift=args.drift
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("patients", "resource_counts", "simulated_readmission_rate")},
            indent=2,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from readmit.config import load_config
    from readmit.pipeline import QualityGateError, run_pipeline
    from readmit.reporting import update_readme

    cfg = load_config(args.config, args.set)
    input_dir = Path(args.input or cfg.data.input_dir)
    label = args.source_label
    if args.synthetic:
        if not _has_fhir(input_dir):
            from readmit.fhir.synthetic import write_population

            patients = args.patients or cfg.data.synthetic_patients
            logger.info("generating %d synthetic patients into %s", patients, input_dir)
            write_population(input_dir, patients, seed=cfg.seed)
        label = label or "synthetic FHIR R4 (readmit generator)"
    try:
        result = run_pipeline(
            cfg,
            input_path=input_dir,
            data_source=label or f"FHIR R4: {input_dir}",
            make_plots=not args.no_plots,
            track=False if args.no_tracking else None,
        )
    except QualityGateError as exc:
        logger.error("%s", exc)
        return 3
    print(
        json.dumps(
            {
                "model": result.best_model,
                "version": result.bundle.version,
                "index_stays": len(result.dataset),
                "readmission_rate": round(result.cohort.readmission_rate, 4),
                "test_auroc": round(result.test_metrics["roc_auc"], 4),
                "test_auprc": round(result.test_metrics["average_precision"], 4),
                "gate_passed": result.gate["passed"],
                "report": str(result.paths.get("report")),
                "mlflow_run": result.run_id,
            },
            indent=2,
        )
    )
    if args.update_readme:
        summary = result.paths["summary"].read_text(encoding="utf-8")
        updated = update_readme(args.readme, summary)
        logger.info("README %s", "updated" if updated else "markers not found; not updated")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    import pandas as pd

    from readmit.bundle import REFERENCE_FILE, ModelBundle
    from readmit.cohort import build_cohort, build_episodes
    from readmit.config import CohortConfig, FeatureConfig
    from readmit.features import compute_features
    from readmit.fhir.flatten import flatten_path
    from readmit.monitoring.drift import detect_drift
    from readmit.monitoring.performance import performance_check

    bundle = ModelBundle.load(args.model_dir)
    cohort_cfg = CohortConfig.model_validate(bundle.metadata.get("cohort_config", {}))
    feature_cfg = FeatureConfig.model_validate(bundle.metadata.get("feature_config", {}))
    tables = flatten_path(args.input)
    cohort = build_cohort(tables, cohort_cfg)
    episodes = build_episodes(
        tables.encounters, cohort_cfg.exclude_encounter_types, cohort_cfg.transfer_gap_hours
    )
    features = compute_features(tables, cohort.index, feature_cfg, cohort_cfg, episodes)
    current = features.reset_index(drop=True).assign(
        risk=bundle.predict_proba(features.reset_index(drop=True)),
        sex=cohort.index["sex"].to_numpy(),
        age_band=cohort.index["age_band"].to_numpy(),
    )
    reference = pd.read_parquet(Path(args.model_dir) / REFERENCE_FILE)
    report = detect_drift(reference, current, psi_threshold=args.psi_threshold)
    performance = performance_check(
        cohort.index["readmitted_30d"], current["risk"], bundle.metadata["test_metrics"]["roc_auc"]
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "drift_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / "performance.json").write_text(
        json.dumps(performance, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "drift_detected": report.drifted,
                "drifted_features": report.drifted_features,
                "matured_label_auroc": performance.get("roc_auc"),
                "retrain_recommended": performance["retrain_recommended"],
                "report": str(out / "drift_report.md"),
            },
            indent=2,
        )
    )
    return 2 if report.drifted and args.fail_on_drift else 0


def cmd_score(args: argparse.Namespace) -> int:
    from readmit.bundle import ModelBundle
    from readmit.copilot.assistant import make_llm_client
    from readmit.service import ScoringService

    service = ScoringService(ModelBundle.load(args.model_dir), llm_client=make_llm_client())
    payload = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    scored = service.score_bundle(payload, args.encounter_id)
    if args.brief:
        brief = service.brief(scored)
        print(brief.text)
    else:
        print(json.dumps(service.risk_assessment(scored), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("readmit.serving.app:app", host=args.host, port=args.port, workers=args.workers)
    return 0


def cmd_sagemaker(args: argparse.Namespace) -> int:
    from readmit.cloud import sagemaker

    if args.action == "package":
        print(sagemaker.package_model(args.model_dir, args.output))
    elif args.action == "deploy":
        print(
            json.dumps(
                sagemaker.deploy(
                    args.model_dir,
                    args.image_uri,
                    args.role_arn,
                    args.bucket,
                    args.endpoint,
                    region=args.region,
                    memory_mb=args.memory_mb,
                    kms_key_id=args.kms_key_id,
                ),
                indent=2,
            )
        )
    elif args.action == "invoke":
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
        print(json.dumps(sagemaker.invoke(args.endpoint, payload, args.region), indent=2))
    elif args.action == "teardown":
        print("\n".join(sagemaker.teardown(args.endpoint, args.region)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readmit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate synthetic FHIR R4 data")
    p.add_argument("--patients", type=int, default=3000)
    p.add_argument("--out", default="data/fhir")
    p.add_argument("--format", choices=["ndjson", "bundle"], default="ndjson")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", action="store_true", help="simulate a shifted population")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("run", help="train and evaluate")
    p.add_argument("--config", default=None)
    p.add_argument("--input", default=None, help="FHIR NDJSON/Bundle directory")
    p.add_argument("--synthetic", action="store_true", help="generate data if the input is empty")
    p.add_argument("--patients", type=int, default=None)
    p.add_argument("--source-label", default=None)
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("monitor", help="drift and performance monitoring")
    p.add_argument("--input", required=True)
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--output", default="reports/monitoring")
    p.add_argument("--psi-threshold", type=float, default=0.2)
    p.add_argument("--fail-on-drift", action="store_true")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("score", help="score one FHIR Bundle")
    p.add_argument("--bundle", required=True)
    p.add_argument("--encounter-id", default=None)
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--brief", action="store_true", help="print a copilot brief instead")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("sagemaker", help="SageMaker serverless deployment")
    p.add_argument("action", choices=["package", "deploy", "invoke", "teardown"])
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--output", default="artifacts/model.tar.gz")
    p.add_argument("--image-uri")
    p.add_argument("--role-arn")
    p.add_argument("--bucket")
    p.add_argument("--endpoint", default="readmission-risk")
    p.add_argument("--region", default=None)
    p.add_argument("--memory-mb", type=int, default=2048)
    p.add_argument("--kms-key-id", default=None)
    p.add_argument("--payload", default=None)
    p.set_defaults(func=cmd_sagemaker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "sagemaker" and args.action == "deploy":
        missing = [f for f in ("image_uri", "role_arn", "bucket") if not getattr(args, f)]
        if missing:
            logger.error("deploy requires --%s", ", --".join(m.replace("_", "-") for m in missing))
            return 1
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

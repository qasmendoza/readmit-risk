"""CLI commands, SageMaker request builders, and configuration validation."""

import json
import tarfile

import pydantic
import pytest

from readmit.cli import main
from readmit.cloud import sagemaker
from readmit.config import load_config, parse_overrides
from readmit.fhir.synthetic import write_population
from tests.conftest import ROOT


def test_cli_synth(tmp_path, capsys):
    assert main(["synth", "--patients", "15", "--out", str(tmp_path / "fhir")]) == 0
    assert json.loads(capsys.readouterr().out)["patients"] == 15


def test_cli_run_end_to_end(tmp_path, capsys):
    args = [
        "run",
        "--synthetic",
        "--patients",
        "500",
        "--input",
        str(tmp_path / "fhir"),
        "--no-tracking",
        "--no-plots",
        "--set",
        f"output.artifacts_dir={tmp_path / 'artifacts'}",
        "--set",
        f"output.reports_dir={tmp_path / 'reports'}",
        "--set",
        f"copilot.knowledge_dir={ROOT / 'knowledge'}",
        "--set",
        "models.candidates=[lace, logistic]",
        "--set",
        "gate.require_beats_baseline=false",  # small data: tests the CLI flow only
        "--set",
        "evaluation.bootstrap_iterations=100",
        "--set",
        "split.cv_folds=3",
    ]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["gate_passed"] is True
    assert (tmp_path / "artifacts" / "model" / "bundle.joblib").exists()


def test_cli_quality_gate_exit_code(tmp_path):
    write_population(tmp_path / "fhir", 400, seed=2)
    code = main(
        [
            "run",
            "--input",
            str(tmp_path / "fhir"),
            "--no-tracking",
            "--no-plots",
            "--set",
            f"output.artifacts_dir={tmp_path / 'a'}",
            "--set",
            f"output.reports_dir={tmp_path / 'r'}",
            "--set",
            f"copilot.knowledge_dir={ROOT / 'knowledge'}",
            "--set",
            "gate.min_auroc=0.999",
            "--set",
            "models.candidates=[lace]",
            "--set",
            "gate.require_beats_baseline=false",
            "--set",
            "evaluation.bootstrap_iterations=100",
        ]
    )
    assert code == 3


def test_cli_monitor_detects_drift(trained, tmp_path, capsys):
    write_population(tmp_path / "drifted", 500, seed=99, drift=True)
    code = main(
        [
            "monitor",
            "--input",
            str(tmp_path / "drifted"),
            "--model-dir",
            str(trained.config.model_dir),
            "--output",
            str(tmp_path / "monitoring"),
            "--fail-on-drift",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 2 and result["drift_detected"]
    assert (tmp_path / "monitoring" / "drift_report.md").exists()


def test_cli_score(trained, synthea_dir, capsys):
    bundle = str(synthea_dir / "synthea_patient_2.json")
    assert main(["score", "--bundle", bundle, "--model-dir", str(trained.config.model_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["resourceType"] == "RiskAssessment"
    assert (
        main(["score", "--bundle", bundle, "--model-dir", str(trained.config.model_dir), "--brief"])
        == 0
    )
    assert "Risk summary" in capsys.readouterr().out


def test_cli_missing_model_dir(tmp_path):
    assert main(["score", "--bundle", "x.json", "--model-dir", str(tmp_path)]) == 1


def test_sagemaker_package_and_requests(trained, tmp_path):
    archive = sagemaker.package_model(trained.config.model_dir, tmp_path / "model.tar.gz")
    with tarfile.open(archive) as tar:
        assert "bundle.joblib" in tar.getnames()  # files at the archive root
    model = sagemaker.model_request("m", "123.dkr.ecr/img:1", "s3://b/k", "arn:role")
    assert model["PrimaryContainer"]["Environment"]["READMIT_MODEL_PATH"] == "/opt/ml/model"
    config = sagemaker.endpoint_config_request("c", "m", memory_mb=3072, max_concurrency=5)
    assert config["ProductionVariants"][0]["ServerlessConfig"] == {
        "MemorySizeInMB": 3072,
        "MaxConcurrency": 5,
    }
    with pytest.raises(ValueError):
        sagemaker.endpoint_config_request("c", "m", memory_mb=1500)
    assert sagemaker.upload_args("kms-key")["ServerSideEncryption"] == "aws:kms"
    assert sagemaker.upload_args(None) == {"ServerSideEncryption": "AES256"}
    assert main(["sagemaker", "deploy"]) == 1  # required arguments enforced


def test_config_overrides_and_validation():
    assert parse_overrides(["a.b=1", "c=[x, y]"]) == {"a": {"b": 1}, "c": ["x", "y"]}
    cfg = load_config(ROOT / "configs" / "config.yaml", ["evaluation.capacity_fraction=0.05"])
    assert cfg.evaluation.capacity_fraction == 0.05
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "configs" / "config.yaml", ["models.candidates=[xgboost]"])
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "configs" / "config.yaml", ["evaluation.moderate_fraction=0.05"])
    with pytest.raises(ValueError):
        parse_overrides(["no-equals-sign"])
    with pytest.raises(FileNotFoundError):
        load_config("missing.yaml")

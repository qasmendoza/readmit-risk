"""Deploy the same container image to Amazon SageMaker Serverless Inference.

Bring-your-own-container: the API image already implements SageMaker's hosting contract
(``GET /ping``, ``POST /invocations`` on port 8080, model files in ``/opt/ml/model``), so one image
runs on Kubernetes and SageMaker. This module packages the bundle as ``model.tar.gz``, uploads it to
S3 with server-side encryption (SSE-KMS when a key is given), and creates the SageMaker model,
serverless endpoint configuration, and endpoint with boto3.

Request builders are pure functions so they can be unit-tested without AWS credentials.
Costs: serverless endpoints bill per request and scale to zero; always run ``teardown`` when done.
"""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from typing import Any

SERVERLESS_MEMORY_OPTIONS = (1024, 2048, 3072, 4096, 5120, 6144)


def package_model(model_dir: str | Path, output: str | Path) -> Path:
    """Create model.tar.gz with the bundle files at the archive root."""
    model_dir, output = Path(model_dir), Path(output)
    if not (model_dir / "bundle.joblib").exists():
        raise FileNotFoundError(f"no bundle.joblib in {model_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(model_dir.iterdir()):
            archive.add(path, arcname=path.name)
    return output


def model_request(
    name: str,
    image_uri: str,
    model_data_url: str,
    role_arn: str,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ModelName": name,
        "ExecutionRoleArn": role_arn,
        "PrimaryContainer": {
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {"READMIT_MODEL_PATH": "/opt/ml/model", **(environment or {})},
        },
    }


def endpoint_config_request(
    name: str, model_name: str, memory_mb: int = 2048, max_concurrency: int = 2
) -> dict[str, Any]:
    if memory_mb not in SERVERLESS_MEMORY_OPTIONS:
        raise ValueError(f"memory_mb must be one of {SERVERLESS_MEMORY_OPTIONS}")
    if not 1 <= max_concurrency <= 200:
        raise ValueError("max_concurrency must be between 1 and 200")
    return {
        "EndpointConfigName": name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_mb,
                    "MaxConcurrency": max_concurrency,
                },
            }
        ],
    }


def upload_args(kms_key_id: str | None) -> dict[str, str]:
    if kms_key_id:
        return {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key_id}
    return {"ServerSideEncryption": "AES256"}


def deploy(
    model_dir: str | Path,
    image_uri: str,
    role_arn: str,
    bucket: str,
    endpoint_name: str = "readmission-risk",
    prefix: str = "readmit",
    region: str | None = None,
    memory_mb: int = 2048,
    max_concurrency: int = 2,
    kms_key_id: str | None = None,
    wait: bool = True,
) -> dict[str, str]:
    import boto3  # optional dependency: pip install "readmit[aws]"

    session = boto3.session.Session(region_name=region)
    s3, sm = session.client("s3"), session.client("sagemaker")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = package_model(model_dir, Path("artifacts") / "model.tar.gz")
    key = f"{prefix}/models/{stamp}/model.tar.gz"
    s3.upload_file(str(archive), bucket, key, ExtraArgs=upload_args(kms_key_id))
    model_name, config_name = f"{endpoint_name}-{stamp}", f"{endpoint_name}-cfg-{stamp}"
    sm.create_model(**model_request(model_name, image_uri, f"s3://{bucket}/{key}", role_arn))
    sm.create_endpoint_config(
        **endpoint_config_request(config_name, model_name, memory_mb, max_concurrency)
    )
    existing = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
    if any(e["EndpointName"] == endpoint_name for e in existing):
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    if wait:
        sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    return {
        "endpoint": endpoint_name,
        "model": model_name,
        "config": config_name,
        "model_data": f"s3://{bucket}/{key}",
    }


def invoke(
    endpoint_name: str, payload: dict[str, Any], region: str | None = None
) -> dict[str, Any]:
    import boto3

    runtime = boto3.session.Session(region_name=region).client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=json.dumps(payload)
    )
    return json.loads(response["Body"].read())


def teardown(endpoint_name: str = "readmission-risk", region: str | None = None) -> list[str]:
    """Delete the endpoint, its configuration, and its model to stop all charges."""
    import boto3

    sm = boto3.session.Session(region_name=region).client("sagemaker")
    removed: list[str] = []
    endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
    config_name = endpoint["EndpointConfigName"]
    config = sm.describe_endpoint_config(EndpointConfigName=config_name)
    sm.delete_endpoint(EndpointName=endpoint_name)
    removed.append(f"endpoint {endpoint_name}")
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    removed.append(f"endpoint config {config_name}")
    for variant in config["ProductionVariants"]:
        sm.delete_model(ModelName=variant["ModelName"])
        removed.append(f"model {variant['ModelName']}")
    return removed

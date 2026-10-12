"""The deployable artifact: calibrated model + everything needed to serve and monitor it.

A bundle directory holds ``bundle.joblib`` (model, thresholds, reference values, guidance documents
for the copilot), ``metadata.json`` (lineage and metrics), and ``reference_sample.parquet`` (the
training feature profile for drift monitoring). The same directory is loaded by the API, logged to
MLflow, and packaged as ``model.tar.gz`` for SageMaker.

Security note: joblib uses pickle. Only load bundles produced by this project's pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from readmit.explain import local_explanation
from readmit.features import FEATURE_NAMES, LACE_COLUMN

BUNDLE_FILE = "bundle.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference_sample.parquet"
INPUT_COLUMNS = [*FEATURE_NAMES, LACE_COLUMN]


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    reference_values: dict[str, float]
    thresholds: dict[str, float]
    metadata: dict[str, Any]
    guidance_docs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    def _frame(self, features: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
        missing = [c for c in INPUT_COLUMNS if c not in frame]
        for column in missing:
            frame[column] = np.nan
        return frame[INPUT_COLUMNS].astype(float)

    def predict_proba(self, features: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        return self.model.predict_proba(self._frame(features))[:, 1]

    def tier(self, probability: float) -> str:
        if probability >= self.thresholds["high"]:
            return "high"
        if probability >= self.thresholds["moderate"]:
            return "moderate"
        return "low"

    def score(self, features: pd.DataFrame | dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        frame = self._frame(features)
        if len(frame) != 1:
            raise ValueError("score() expects a single patient")
        probability = float(self.predict_proba(frame)[0])
        return {
            "risk": probability,
            "tier": self.tier(probability),
            "factors": local_explanation(self.predict_proba, frame, self.reference_values, top_k),
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / BUNDLE_FILE, compress=3)
        (directory / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        path = Path(directory) / BUNDLE_FILE
        if not path.exists():
            raise FileNotFoundError(f"no model bundle at {path}; run `readmit run` first")
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle


def find_bundle_dir(candidates: list[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate and (Path(candidate) / BUNDLE_FILE).exists():
            return Path(candidate)
    return None

"""MLflow logging: the logged pyfunc model reproduces the bundle's predictions."""

import numpy as np
import pytest

pytest.importorskip("mlflow")


@pytest.mark.slow
def test_log_and_reload(trained, tmp_path, monkeypatch):
    import mlflow

    from readmit.tracking import load_bundle_from_mlflow, log_run

    monkeypatch.chdir(tmp_path)  # keep mlruns/ and mlflow.db inside the temp dir
    cfg = trained.config.model_copy(deep=True)
    cfg.tracking.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg.tracking.register_model = True
    trained.config, original = cfg, trained.config
    try:
        run_id = log_run(trained)
    finally:
        trained.config = original
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    sample = trained.dataset.iloc[trained.split.test_index].head(10)
    output = model.predict(sample[list(model.metadata.get_input_schema().input_names())])
    assert list(output.columns) == ["risk", "tier"]
    np.testing.assert_allclose(output["risk"], trained.bundle.predict_proba(sample), rtol=1e-9)
    bundle = load_bundle_from_mlflow(f"models:/{cfg.tracking.model_name}/1")
    assert bundle.version == trained.bundle.version

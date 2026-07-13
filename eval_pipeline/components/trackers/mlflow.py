from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval_pipeline.components.trackers.local import LocalExperimentTracker
from eval_pipeline.components.trackers.utils import as_number, json_safe, tracking_key
from eval_pipeline.core.config import ConfigError, ExperimentConfig
from eval_pipeline.interfaces.tracking import ExperimentTracker
from eval_pipeline.registry import register_component

if TYPE_CHECKING:
    from eval_pipeline.core.context import ExperimentPaths


@register_component("mlflow", category="tracking")
class MlflowExperimentTracker(ExperimentTracker):
    def start(self, config: ExperimentConfig, paths: ExperimentPaths) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ConfigError("MLflow tracking requires installing the optional 'mlflow' dependency.") from exc

        self.config = config
        self.local = LocalExperimentTracker()
        self.local.start(config, paths)
        tracking_uri = self.params.get("tracking_uri")
        _allow_file_store_if_needed(tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(str(self.params.get("experiment_name", config.name)))
        self.run = mlflow.start_run(run_name=str(self.params.get("run_name", config.name)))
        atexit.register(self.close)
        self.log_config(config)
        run_params = self.params.get("run_params", {})
        if not isinstance(run_params, dict):
            raise ConfigError("[tracking.params.run_params] must be a table/object when provided.")
        if run_params:
            mlflow.log_params({key: json_safe(value) for key, value in run_params.items()})

    def log_config(self, config: ExperimentConfig) -> None:
        self.local.log_config(config)
        self.mlflow.log_artifact(str(self.local.resolved_config_path), artifact_path="config")
        if self.local.source_config_path.exists():
            self.mlflow.log_artifact(str(self.local.source_config_path), artifact_path="config")

    def log_metric(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        self.local.log_metric(name, value, step=step, stage=stage)
        numeric_value = as_number(value)
        if numeric_value is not None:
            self.mlflow.log_metric(tracking_key("metric", name, stage), numeric_value, step=step)

    def log_loss(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        self.local.log_loss(name, value, step=step, stage=stage)
        numeric_value = as_number(value)
        if numeric_value is not None:
            self.mlflow.log_metric(tracking_key("loss", name, stage), numeric_value, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> Path | None:
        copied_path = self.local.log_artifact(path, artifact_path=artifact_path)
        if copied_path is None:
            return None
        if copied_path.is_dir():
            self.mlflow.log_artifacts(str(copied_path), artifact_path=artifact_path)
        else:
            self.mlflow.log_artifact(str(copied_path), artifact_path=artifact_path)
        return copied_path

    def log_model(self, model: Any, *, name: str = "model") -> Path | None:
        model_path = self.local.log_model(model, name=name)
        if model_path is not None:
            self.mlflow.log_artifact(str(model_path), artifact_path="models")
        return model_path

    def close(self) -> None:
        if self.mlflow.active_run() is not None:
            if self.local.metrics_log.exists():
                self.mlflow.log_artifact(str(self.local.metrics_log), artifact_path="tracking")
            if self.local.losses_log.exists():
                self.mlflow.log_artifact(str(self.local.losses_log), artifact_path="tracking")
            self.mlflow.end_run()


"""
This is an edge case where we need to set an environment variable 
to **also** allow file-based tracking in MLflow.
"""
def _allow_file_store_if_needed(tracking_uri: Any) -> None:
    if tracking_uri is not None and str(tracking_uri).startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

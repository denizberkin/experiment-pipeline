from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval_pipeline.core.config import ExperimentConfig
from eval_pipeline.components.trackers.utils import LoggedRecord, is_relative_to
from eval_pipeline.interfaces.tracking import ExperimentTracker
from eval_pipeline.registry import register_component

if TYPE_CHECKING:
    from eval_pipeline.core.context import ExperimentPaths


@register_component("local", category="tracking")
class LocalExperimentTracker(ExperimentTracker):
    def start(self, config: ExperimentConfig, paths: ExperimentPaths) -> None:
        self.config = config
        self.paths = paths
        self.metrics_log = paths.logs_dir / "metrics.jsonl"
        self.losses_log = paths.logs_dir / "losses.jsonl"
        self.resolved_config_path = paths.experiment_dir / "config.resolved.json"
        self.source_config_path = paths.experiment_dir / f"config.source{config.config_path.suffix}"
        self._prepare_dirs()
        self.log_config(config)

    def log_config(self, config: ExperimentConfig) -> None:
        self.resolved_config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if config.config_path.exists():
            shutil.copy2(config.config_path, self.source_config_path)

    def log_metric(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        self._append_record(self.metrics_log, LoggedRecord("metric", name, value, step=step, stage=stage))

    def log_loss(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        self._append_record(self.losses_log, LoggedRecord("loss", name, value, step=step, stage=stage))

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> Path | None:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Artifact does not exist: {source}")

        target_dir = self.paths.artifacts_dir
        if artifact_path:
            target_dir = target_dir / artifact_path
        target_dir.mkdir(parents=True, exist_ok=True)

        target = target_dir / source.name
        if source == target or is_relative_to(source, target_dir):
            return source
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        return target

    def log_model(self, model: Any, *, name: str = "model") -> Path | None:
        model_path = Path(model).expanduser() if isinstance(model, str | Path) else None
        if model_path is not None and model_path.exists():
            return self.log_artifact(model_path, artifact_path="models")

        models_dir = self.paths.artifacts_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / f"{name}.pkl"
        with target.open("wb") as handle:
            pickle.dump(model, handle)
        return target

    def _prepare_dirs(self) -> None:
        self.paths.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    def _append_record(self, path: Path, record: LoggedRecord) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")

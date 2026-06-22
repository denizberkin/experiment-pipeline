from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval_pipeline.interfaces.pipeline import PipelineComponent

if TYPE_CHECKING:
    from eval_pipeline.core.config import ExperimentConfig
    from eval_pipeline.core.context import ExperimentPaths


class ExperimentTracker(PipelineComponent):
    def start(self, config: ExperimentConfig, paths: ExperimentPaths) -> None:
        self.log_config(config)

    def log_config(self, config: ExperimentConfig) -> None:
        raise NotImplementedError

    def log_metric(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        raise NotImplementedError

    def log_metrics(self, values: dict[str, Any], *, step: int | None = None, stage: str | None = None) -> None:
        for name, value in values.items():
            self.log_metric(name, value, step=step, stage=stage)

    def log_loss(self, name: str, value: Any, *, step: int | None = None, stage: str | None = None) -> None:
        raise NotImplementedError

    def log_losses(self, values: dict[str, Any], *, step: int | None = None, stage: str | None = None) -> None:
        for name, value in values.items():
            self.log_loss(name, value, step=step, stage=stage)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> Path | None:
        raise NotImplementedError

    def log_model(self, model: Any, *, name: str = "model") -> Path | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

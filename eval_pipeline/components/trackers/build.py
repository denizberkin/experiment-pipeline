from __future__ import annotations

from typing import TYPE_CHECKING

from eval_pipeline.core.config import ExperimentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.tracking import ExperimentTracker
if TYPE_CHECKING:
    from eval_pipeline.core.context import ExperimentPaths


def build_experiment_tracker(config: ExperimentConfig, paths: ExperimentPaths) -> ExperimentTracker:
    tracker = instantiate_component(config.tracking)
    tracker.start(config, paths)
    return tracker

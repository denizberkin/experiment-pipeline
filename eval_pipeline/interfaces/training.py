from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.context import TrainingContext
from eval_pipeline.interfaces.pipeline import PipelineComponent


class Trainer(PipelineComponent):
    """Owns model fitting logic for a framework or workflow."""

    @abstractmethod
    def train(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> Any:
        """Train a model and return serializable training results."""

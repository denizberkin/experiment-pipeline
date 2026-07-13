from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from eval_pipeline.interfaces.pipeline import PipelineComponent

if TYPE_CHECKING:
    from eval_pipeline.context import TrainingContext


class Trainer[Data, Model, Result](PipelineComponent):
    """Owns model fitting logic for a framework or workflow."""

    @abstractmethod
    def train(
        self,
        *,
        data: Data,
        model: Model,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> Result:
        """Train a model and return serializable training results."""

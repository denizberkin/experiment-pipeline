from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic

from eval_pipeline.context import TrainingContext
from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import DataT, ModelT, ResultT


class Trainer(PipelineComponent, Generic[DataT, ModelT, ResultT]):
    """Owns model fitting logic for a framework or workflow."""

    @abstractmethod
    def train(
        self,
        *,
        data: DataT,
        model: ModelT,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> ResultT:
        """Train a model and return serializable training results."""

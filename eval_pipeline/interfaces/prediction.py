from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from eval_pipeline.context import PredictionContext
from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import DataT, ModelT, ResultT


class Predictor(PipelineComponent, Generic[DataT, ModelT, ResultT]):
    """Owns inference-only prediction logic."""

    @abstractmethod
    def predict(self, *, data: DataT, model: ModelT, context: PredictionContext) -> ResultT:
        """Run inference and return serializable predictions or artifact references."""

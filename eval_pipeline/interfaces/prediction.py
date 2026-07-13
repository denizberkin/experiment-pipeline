from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from eval_pipeline.interfaces.pipeline import PipelineComponent

if TYPE_CHECKING:
    from eval_pipeline.context import PredictionContext


class Predictor[Data, Model, Result](PipelineComponent):
    """Owns inference-only prediction logic."""

    @abstractmethod
    def predict(self, *, data: Data, model: Model, context: PredictionContext) -> Result:
        """Run inference and return serializable predictions or artifact references."""

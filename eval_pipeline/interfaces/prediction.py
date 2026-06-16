from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.context import PredictionContext
from eval_pipeline.interfaces.pipeline import PipelineComponent


class Predictor(PipelineComponent):
    """Owns inference-only prediction logic."""

    @abstractmethod
    def predict(self, *, data: Any, model: Any, context: PredictionContext) -> Any:
        """Run inference and return serializable predictions or artifact references."""

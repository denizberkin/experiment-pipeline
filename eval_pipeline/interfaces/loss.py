from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import LossValueT, PredictionT, TargetT


class Loss(PipelineComponent, Generic[PredictionT, TargetT, LossValueT]):
    """Computes a training or evaluation loss."""

    @abstractmethod
    def __call__(self, prediction: PredictionT, target: TargetT) -> LossValueT:
        """Return a loss value."""

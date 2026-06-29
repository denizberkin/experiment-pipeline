from __future__ import annotations

from abc import abstractmethod

from eval_pipeline.interfaces.pipeline import PipelineComponent


class Loss[Prediction, Target, LossValue](PipelineComponent):
    """Computes a training or evaluation loss."""

    @abstractmethod
    def __call__(self, prediction: Prediction, target: Target) -> LossValue:
        """Return a loss value."""

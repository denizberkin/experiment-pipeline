from __future__ import annotations

from abc import abstractmethod

from eval_pipeline.interfaces.pipeline import PipelineComponent


class Metric[Prediction, Target, MetricValue](PipelineComponent):
    """Computes a metric from predictions and targets."""

    @abstractmethod
    def update(self, prediction: Prediction, target: Target) -> None:
        """Accumulate one batch or sample."""

    @abstractmethod
    def compute(self) -> MetricValue:
        """Return the final metric value."""

    def reset(self) -> None:
        """Reset internal state between evaluations."""

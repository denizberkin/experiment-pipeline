from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import MetricValueT, PredictionT, TargetT


class Metric(PipelineComponent, Generic[PredictionT, TargetT, MetricValueT]):
    """Computes a metric from predictions and targets."""

    @abstractmethod
    def update(self, prediction: PredictionT, target: TargetT) -> None:
        """Accumulate one batch or sample."""

    @abstractmethod
    def compute(self) -> MetricValueT:
        """Return the final metric value."""

    def reset(self) -> None:
        """Reset internal state between evaluations."""

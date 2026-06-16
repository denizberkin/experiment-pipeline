from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.interfaces.pipeline import PipelineComponent


class Metric(PipelineComponent):
    """Computes a metric from predictions and targets."""

    @abstractmethod
    def update(self, prediction: Any, target: Any) -> None:
        """Accumulate one batch or sample."""

    @abstractmethod
    def compute(self) -> Any:
        """Return the final metric value."""

    def reset(self) -> None:
        """Reset internal state between evaluations."""

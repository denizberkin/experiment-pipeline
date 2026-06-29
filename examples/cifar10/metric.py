from __future__ import annotations

from typing import Any

from eval_pipeline.components.metrics.base import Metric
from eval_pipeline.registry import register_component


@register_component("classification_accuracy", category="metric")
class ClassificationAccuracy(Metric[Any, Any, float]):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.correct = 0
        self.total = 0

    def update(self, prediction: Any, target: Any) -> None:
        predicted = prediction.argmax(dim=1)
        self.correct += int((predicted == target).sum().item())
        self.total += int(target.numel())

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

from __future__ import annotations

from typing import Any

from eval_pipeline.components.metrics.base import Metric
from eval_pipeline.registry import register_component


@register_component("mae", category="metric")
class ExampleMetric(Metric[Any, Any, float]):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(self, prediction: Any, target: Any) -> None:
        self.values.append(abs(float(prediction) - float(target)))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self) -> None:
        self.values.clear()

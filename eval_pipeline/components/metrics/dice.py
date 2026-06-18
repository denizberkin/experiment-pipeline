from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from eval_pipeline.components.metrics.base import Metric
from eval_pipeline.registry import register_component


@register_component("dice", category="metric")
class DiceScore(Metric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.threshold = float(params.get("threshold", 0.5))
        self.smooth = float(params.get("smooth", 1e-8))
        self._intersection = 0.0
        self._prediction_sum = 0.0
        self._target_sum = 0.0

    def update(self, prediction: Any, target: Any) -> None:
        predictions = [float(value >= self.threshold) for value in _flatten_numbers(prediction)]
        targets = [float(value >= self.threshold) for value in _flatten_numbers(target)]
        for pred, true in zip(predictions, targets, strict=True):
            self._intersection += pred * true
            self._prediction_sum += pred
            self._target_sum += true

    def compute(self) -> float:
        numerator = 2.0 * self._intersection + self.smooth
        denominator = self._prediction_sum + self._target_sum + self.smooth
        return numerator / denominator

    def reset(self) -> None:
        self._intersection = 0.0
        self._prediction_sum = 0.0
        self._target_sum = 0.0


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    raise TypeError(f"Expected numeric value or nested numeric iterable, got {type(value).__name__}.")

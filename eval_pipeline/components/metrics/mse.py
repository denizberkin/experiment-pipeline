from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from eval_pipeline.components.metrics.base import Metric
from eval_pipeline.registry import register_component


@register_component("mse", category="metric")
class MeanSquaredError(Metric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._squared_errors: list[float] = []

    def update(self, prediction: Any, target: Any) -> None:
        for pred, true in zip(_flatten_numbers(prediction), _flatten_numbers(target), strict=True):
            self._squared_errors.append((pred - true) ** 2)

    def compute(self) -> float:
        if not self._squared_errors:
            return 0.0
        return sum(self._squared_errors) / len(self._squared_errors)

    def reset(self) -> None:
        self._squared_errors.clear()


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    raise TypeError(f"Expected numeric value or nested numeric iterable, got {type(value).__name__}.")

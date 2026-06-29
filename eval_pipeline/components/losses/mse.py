from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from eval_pipeline.components.losses.base import Loss
from eval_pipeline.registry import register_component


@register_component("mse", category="loss")
class MeanSquaredErrorLoss(Loss[Any, Any, float]):
    def __call__(self, prediction: Any, target: Any) -> float:
        pairs = list(zip(_flatten_numbers(prediction), _flatten_numbers(target), strict=True))
        if not pairs:
            return 0.0
        return sum((pred - true) ** 2 for pred, true in pairs) / len(pairs)


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    raise TypeError(f"Expected numeric value or nested numeric iterable, got {type(value).__name__}.")

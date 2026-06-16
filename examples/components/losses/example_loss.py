from __future__ import annotations

from typing import Any

from eval_pipeline.components.losses.base import Loss
from eval_pipeline.registry import register_component


@register_component("mae_loss", category="loss")
class ExampleLoss(Loss):
    def __call__(self, prediction: Any, target: Any) -> float:
        return abs(float(prediction) - float(target))

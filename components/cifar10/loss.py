from __future__ import annotations

from typing import Any

from eval_pipeline.components.losses.base import Loss
from eval_pipeline.registry import register_component


@register_component("torch_ce", category="loss")
class TorchCrossEntropyLoss(Loss):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._loss = None

    def __call__(self, prediction: Any, target: Any) -> Any:
        if self._loss is None:
            try:
                import torch.nn as nn
            except ImportError as exc:
                raise ImportError("TorchCrossEntropyLoss requires torch.") from exc
            self._loss = nn.CrossEntropyLoss()
        return self._loss(prediction, target)

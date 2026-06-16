from __future__ import annotations

from typing import Any

from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.context import TrainingContext
from eval_pipeline.registry import register_component


@register_component("toy_trainer", category="training")
class ExampleTrainer(Trainer):
    def train(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> dict:
        return {
            "status": "training_not_implemented",
            "train_size": len(data.get("train", [])),
            "experiment": context.config.name,
        }

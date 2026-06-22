from __future__ import annotations

from typing import Any

from eval_pipeline.components.validators.base import Validator
from eval_pipeline.context import ValidationContext
from eval_pipeline.registry import register_component


@register_component("toy_validator", category="validation")
class ExampleValidator(Validator):
    def validate(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: ValidationContext,
    ) -> dict:
        validation_size = len(data.get("validation", []))
        context.tracker.log_metric("validation_size", validation_size, stage=context.stage)
        return {
            "status": "validation_not_implemented",
            "validation_size": validation_size,
            "experiment_dir": str(context.paths.experiment_dir),
        }

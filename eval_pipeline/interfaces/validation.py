from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.context import ValidationContext
from eval_pipeline.interfaces.pipeline import PipelineComponent


class Validator(PipelineComponent):
    """Owns validation logic during or after training."""

    @abstractmethod
    def validate(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: ValidationContext,
    ) -> Any:
        """Validate a model and return serializable validation results."""

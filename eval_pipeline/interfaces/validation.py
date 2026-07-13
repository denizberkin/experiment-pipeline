from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from eval_pipeline.interfaces.pipeline import PipelineComponent

if TYPE_CHECKING:
    from eval_pipeline.context import ValidationContext


class Validator[Data, Model, Result](PipelineComponent):
    """Owns validation logic during or after training."""

    @abstractmethod
    def validate(
        self,
        *,
        data: Data,
        model: Model,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: ValidationContext,
    ) -> Result:
        """Validate a model and return serializable validation results."""

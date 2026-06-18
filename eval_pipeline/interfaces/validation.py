from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic

from eval_pipeline.context import ValidationContext
from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import DataT, ModelT, ResultT


class Validator(PipelineComponent, Generic[DataT, ModelT, ResultT]):
    """Owns validation logic during or after training."""

    @abstractmethod
    def validate(
        self,
        *,
        data: DataT,
        model: ModelT,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: ValidationContext,
    ) -> ResultT:
        """Validate a model and return serializable validation results."""

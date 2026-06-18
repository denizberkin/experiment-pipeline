from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import ModelT


class ModelFactory(PipelineComponent, Generic[ModelT]):
    """Creates an ML or DL model instance."""

    @abstractmethod
    def build(self) -> ModelT:
        """Return a model instance."""

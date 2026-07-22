from __future__ import annotations

from abc import abstractmethod

from eval_pipeline.interfaces.pipeline import PipelineComponent


class ModelFactory[Model](PipelineComponent):
    """Creates an ML or DL model instance."""

    @abstractmethod
    def build(self) -> Model:
        """Return a model instance."""

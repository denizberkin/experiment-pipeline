from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.interfaces.pipeline import PipelineComponent


class ModelFactory(PipelineComponent):
    """Creates an ML or DL model instance."""

    @abstractmethod
    def build(self) -> Any:
        """Return a model instance."""

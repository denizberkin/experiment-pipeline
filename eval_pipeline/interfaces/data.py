from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.interfaces.pipeline import PipelineComponent


class DataModule(PipelineComponent):
    """Builds data splits or loaders for an experiment."""

    @abstractmethod
    def setup(self) -> Any:
        """Return data needed by training, validation, testing, or prediction."""

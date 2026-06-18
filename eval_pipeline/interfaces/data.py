from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import DataT


class DataModule(PipelineComponent, Generic[DataT]):
    """Builds data splits or loaders for an experiment."""

    @abstractmethod
    def setup(self) -> DataT:
        """Return data needed by training, validation, testing, or prediction."""

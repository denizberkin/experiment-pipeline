from __future__ import annotations

from abc import abstractmethod

from eval_pipeline.interfaces.pipeline import PipelineComponent


class DataModule[Data](PipelineComponent):
    """Builds data splits or loaders for an experiment."""

    @abstractmethod
    def setup(self) -> Data:
        """Return data needed by training, validation, testing, or prediction."""

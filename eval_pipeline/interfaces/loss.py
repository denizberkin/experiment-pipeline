from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.interfaces.pipeline import PipelineComponent


class Loss(PipelineComponent):
    """Computes a training or evaluation loss."""

    @abstractmethod
    def __call__(self, prediction: Any, target: Any) -> Any:
        """Return a loss value."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic

from eval_pipeline.context import TestContext
from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.interfaces.types import DataT, ModelT, ResultT


class Tester(PipelineComponent, Generic[DataT, ModelT, ResultT]):
    """Owns final held-out test evaluation logic."""

    @abstractmethod
    def test(
        self,
        *,
        data: DataT,
        model: ModelT,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TestContext,
    ) -> ResultT:
        """Test a model and return serializable test results."""

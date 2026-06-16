from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eval_pipeline.context import TestContext
from eval_pipeline.interfaces.pipeline import PipelineComponent


class Tester(PipelineComponent):
    """Owns final held-out test evaluation logic."""

    @abstractmethod
    def test(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TestContext,
    ) -> Any:
        """Test a model and return serializable test results."""

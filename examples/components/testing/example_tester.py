from __future__ import annotations

from typing import Any

from eval_pipeline.components.testers.base import Tester
from eval_pipeline.context import TestContext
from eval_pipeline.registry import register_component


@register_component("toy_tester", category="test")
class ExampleTester(Tester):
    def test(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TestContext,
    ) -> dict:
        return {
            "status": "test_not_implemented",
            "test_size": len(data.get("test", [])),
            "stage": context.stage,
        }

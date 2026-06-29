from __future__ import annotations

from typing import Any

from eval_pipeline.components.testers.base import Tester
from eval_pipeline.context import TestContext
from eval_pipeline.registry import register_component

type ToyData = dict[str, list[int]]
type ToyModel = dict[str, str]
type ToyResult = dict[str, object]


@register_component("toy_tester", category="test")
class ExampleTester(Tester[ToyData, ToyModel, ToyResult]):
    def test(
        self,
        *,
        data: ToyData,
        model: ToyModel,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TestContext,
    ) -> ToyResult:
        test_size = len(data.get("test", []))
        context.tracker.log_metric("test_size", test_size, stage=context.stage)
        return {
            "status": "test_not_implemented",
            "test_size": test_size,
            "stage": context.stage,
        }

from __future__ import annotations

from typing import Any

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


@register_component("toy_model", category="model")
class ExampleModelFactory(ModelFactory):
    def build(self) -> Any:
        return {"kind": "placeholder_model"}

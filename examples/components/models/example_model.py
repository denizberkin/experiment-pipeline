from __future__ import annotations

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component

type ToyModel = dict[str, str]


@register_component("toy_model", category="model")
class ExampleModelFactory(ModelFactory[ToyModel]):
    def build(self) -> ToyModel:
        return {"kind": "placeholder_model"}

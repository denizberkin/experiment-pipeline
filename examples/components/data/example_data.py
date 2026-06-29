from __future__ import annotations

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component

type ToyData = dict[str, list[int]]


@register_component("toy_data", category="data")
class ExampleDataModule(DataModule[ToyData]):
    def setup(self) -> ToyData:
        return {"train": [1, 2, 3], "validation": [4], "test": [5]}

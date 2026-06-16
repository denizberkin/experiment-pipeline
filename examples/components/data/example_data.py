from __future__ import annotations

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component


@register_component("toy_data", category="data")
class ExampleDataModule(DataModule):
    def setup(self) -> dict[str, list[int]]:
        return {"train": [1, 2, 3], "validation": [4], "test": [5]}

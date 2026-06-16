from __future__ import annotations

from typing import Any

from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.importing import import_class_from_path
from eval_pipeline.core.registry import resolve_component_class


def import_configured_class(config: ComponentConfig, *, expected_base_name: str | None = None) -> type[Any]:
    if expected_base_name is not None or config.path is None or config.class_name is None:
        return resolve_component_class(config)
    return import_class_from_path(config.path, config.class_name)


__all__ = ["import_class_from_path", "import_configured_class"]

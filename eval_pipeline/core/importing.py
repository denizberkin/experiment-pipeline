from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from eval_pipeline.core.config import ConfigError


def import_module_from_path(path: str | Path) -> ModuleType:
    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise ConfigError(f"Component file does not exist: {module_path}")
    if module_path.suffix != ".py":
        raise ConfigError(f"Component file must be a Python file: {module_path}")
    return _load_module(module_path)


def import_class_from_path(path: str | Path, class_name: str) -> type[Any]:
    module_path = Path(path).expanduser().resolve()
    module = import_module_from_path(module_path)
    try:
        component_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ConfigError(f"Class '{class_name}' was not found in {module_path}") from exc

    if not inspect.isclass(component_class):
        raise ConfigError(f"'{class_name}' in {module_path} is not a class.")
    return component_class


def _load_module(module_path: Path) -> ModuleType:
    module_name = f"_eval_pipeline_component_{abs(hash(module_path))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Could not create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module

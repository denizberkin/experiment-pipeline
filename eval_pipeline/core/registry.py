from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_pipeline.core.config import ComponentConfig, ConfigError
from eval_pipeline.core.importing import import_class_from_path, import_module_from_path
from eval_pipeline.interfaces import (
    DataModule,
    Loss,
    Metric,
    ModelFactory,
    Predictor,
    Tester,
    Trainer,
    ExperimentTracker,
    Validator,
)


@dataclass(frozen=True)
class ComponentSpec:
    category: str
    interface: type[Any]


@dataclass(frozen=True)
class RegisteredComponent:
    category: str
    name: str
    component_class: type[Any]
    source_path: Path | None = None


COMPONENT_SPECS: dict[str, ComponentSpec] = {
    "data": ComponentSpec(category="data", interface=DataModule),
    "model": ComponentSpec(category="model", interface=ModelFactory),
    "loss": ComponentSpec(category="loss", interface=Loss),
    "metric": ComponentSpec(category="metric", interface=Metric),
    "training": ComponentSpec(category="training", interface=Trainer),
    "validation": ComponentSpec(category="validation", interface=Validator),
    "test": ComponentSpec(category="test", interface=Tester),
    "prediction": ComponentSpec(category="prediction", interface=Predictor),
    "tracking": ComponentSpec(category="tracking", interface=ExperimentTracker),
}

_COMPONENT_REGISTRY: dict[tuple[str, str], RegisteredComponent] = {}

_COMPONENT_DIRS = {
    "data": "data",
    "model": "models",
    "loss": "losses",
    "metric": "metrics",
    "training": "trainers",
    "validation": "validators",
    "test": "testers",
    "prediction": "predictors",
    "tracking": "trackers",
}


def get_component_spec(category: str) -> ComponentSpec:
    try:
        return COMPONENT_SPECS[category]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPONENT_SPECS))
        raise ConfigError(f"Unsupported component category '{category}'. Supported: {supported}") from exc


def register_component(name: str, *, category: str):
    """Register a component class under a config-friendly alias."""

    if not isinstance(name, str) or not name:
        raise ConfigError("Registered component name must be a non-empty string.")

    spec = get_component_spec(category)

    def decorator(component_class: type[Any]) -> type[Any]:
        if not inspect.isclass(component_class) or not issubclass(component_class, spec.interface):
            raise ConfigError(
                f"Cannot register {component_class!r} as '{name}' for category '{category}': "
                f"it must subclass {spec.interface.__module__}.{spec.interface.__name__}."
            )

        key = (category, name)
        source_path = _component_source_path(component_class)
        existing = _COMPONENT_REGISTRY.get(key)
        if existing is not None and existing.component_class is not component_class:
            if existing.source_path is not None and existing.source_path == source_path:
                return component_class
            if _is_builtin_source_path(existing.source_path) and not _is_builtin_source_path(source_path):
                _COMPONENT_REGISTRY[key] = RegisteredComponent(
                    category=category,
                    name=name,
                    component_class=component_class,
                    source_path=source_path,
                )
                return component_class
            if not _is_builtin_source_path(existing.source_path) and _is_builtin_source_path(source_path):
                return component_class
            raise ConfigError(
                f"Component alias '{name}' for category '{category}' is already registered "
                f"by {existing.component_class.__module__}.{existing.component_class.__name__}."
            )

        _COMPONENT_REGISTRY[key] = RegisteredComponent(
            category=category,
            name=name,
            component_class=component_class,
            source_path=source_path,
        )
        return component_class

    return decorator


def get_registered_component(category: str, name: str) -> type[Any]:
    return _get_registered_component(category, name).component_class


def import_registry_paths(paths: list[Path]) -> None:
    for path in paths:
        import_module_from_path(path)


def resolve_component_class(config: ComponentConfig) -> type[Any]:
    spec = get_component_spec(config.category)
    registered: RegisteredComponent | None = None
    if config.class_name is not None:
        if config.path is None:
            raise ConfigError(f"Component '{config.label}' must define 'path' when using 'class_name'.")
        component_class = import_class_from_path(config.path, config.class_name)
    else:
        if config.path is not None:
            import_module_from_path(config.path)
        if config.name is None:
            raise ConfigError(f"Component in category '{config.category}' must define 'name' or 'class_name'.")
        registered = _find_registered_component(config.category, config.name)
        if config.path is not None and registered.source_path is not None and registered.source_path != config.path:
            raise ConfigError(
                f"Component alias '{config.name}' for category '{config.category}' was loaded from "
                f"{registered.source_path}, not configured path {config.path}."
            )
        component_class = registered.component_class

    if not inspect.isclass(component_class) or not issubclass(component_class, spec.interface):
        raise ConfigError(
            f"{config.label} from {config.path or (registered.source_path if registered else 'registry')} must subclass "
            f"{spec.interface.__module__}.{spec.interface.__name__}."
        )
    return component_class


def instantiate_component(config: ComponentConfig) -> Any:
    component_class = resolve_component_class(config)
    return component_class(**config.params)


def validate_component_imports(configs: list[ComponentConfig], *, registry_paths: list[Path] | None = None) -> None:
    import_registry_paths(registry_paths or [])
    for config in configs:
        resolve_component_class(config)


def _find_registered_component(category: str, name: str) -> RegisteredComponent:
    registered = _lookup_registered_component(category, name)
    if registered is not None:
        return registered

    _import_builtin_component_file(category, name)
    registered = _lookup_registered_component(category, name)
    if registered is not None:
        return registered

    builtin_hint = _builtin_component_hint(category, name)
    raise ConfigError(
        f"No registered component named '{name}' for category '{category}'. "
        f"For built-ins, add a registered component at {builtin_hint}. "
        "For external components, add the script to [registry].paths or set path on the component."
    )


def _get_registered_component(category: str, name: str) -> RegisteredComponent:
    registered = _lookup_registered_component(category, name)
    if registered is None:
        raise ConfigError(f"No registered component named '{name}' for category '{category}'.")
    return registered


def _lookup_registered_component(category: str, name: str) -> RegisteredComponent | None:
    return _COMPONENT_REGISTRY.get((category, name))


def _import_builtin_component_file(category: str, name: str) -> None:
    for path in _builtin_component_candidates(category, name):
        if path.exists():
            import_module_from_path(path)
            return


def _builtin_component_candidates(category: str, name: str) -> list[Path]:
    root = _builtin_category_dir(category)
    normalized_name = name.replace("-", "_")
    candidates = [root / f"{normalized_name}.py"]

    if "." in normalized_name:
        candidates.append(root.joinpath(*normalized_name.split(".")).with_suffix(".py"))

    return candidates


def _builtin_category_dir(category: str) -> Path:
    get_component_spec(category)
    return Path(__file__).resolve().parents[1] / "components" / _COMPONENT_DIRS[category]


def _builtin_component_hint(category: str, name: str) -> Path:
    return _builtin_component_candidates(category, name)[0]


def _component_source_path(component_class: type[Any]) -> Path | None:
    module = sys.modules.get(component_class.__module__)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).resolve()


def _is_builtin_source_path(path: Path | None) -> bool:
    if path is None:
        return False
    components_root = Path(__file__).resolve().parents[1] / "components"
    try:
        path.resolve().relative_to(components_root)
    except ValueError:
        return False
    return True

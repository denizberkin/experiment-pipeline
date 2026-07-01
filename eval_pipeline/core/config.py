from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


DeviceConfig = str | list[int | str]


@dataclass(frozen=True)
class ComponentConfig:
    category: str
    path: Path | None = None
    class_name: str | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        config_dir: Path,
        section: str,
        category: str,
    ) -> ComponentConfig:
        if not isinstance(raw, dict):
            raise ConfigError(f"[{section}] must be a table/object.")

        path = raw.get("path")
        class_name = raw.get("class_name")

        name = raw.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ConfigError(f"[{section}.name] must be a non-empty string when provided.")
        if class_name is not None and (not isinstance(class_name, str) or not class_name):
            raise ConfigError(f"[{section}.class_name] must be a non-empty string when provided.")
        if path is not None and (not isinstance(path, str) or not path):
            raise ConfigError(f"[{section}.path] must be a non-empty string when provided.")

        if name is None and class_name is None:
            raise ConfigError(f"[{section}] must define either 'name' or 'class_name'.")
        if class_name is not None and path is None:
            raise ConfigError(f"[{section}] must define 'path' when using 'class_name'.")

        params = raw.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError(f"[{section}.params] must be a table/object when provided.")

        resolved_path = None
        if path is not None:
            resolved_path = Path(path)
            if not resolved_path.is_absolute():
                resolved_path = config_dir / resolved_path
            resolved_path = resolved_path.resolve()

        metadata = {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in {"path", "class_name", "name", "params"}
        }

        return cls(
            category=category,
            path=resolved_path,
            class_name=class_name,
            name=name,
            params=copy.deepcopy(params),
            metadata=metadata,
        )

    @property
    def label(self) -> str:
        return self.name or self.class_name or self.category

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "path": str(self.path) if self.path else None,
            "class_name": self.class_name,
            "params": copy.deepcopy(self.params),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class LossConfig(ComponentConfig):
    weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, config_dir: Path, section: str) -> "LossConfig":
        component = ComponentConfig.from_mapping(raw, config_dir=config_dir, section=section, category="loss")
        weight = raw.get("weight", 1.0)
        if not isinstance(weight, int | float):
            raise ConfigError(f"[{section}.weight] must be a number when provided.")
        return cls(
            category=component.category,
            path=component.path,
            class_name=component.class_name,
            name=component.name,
            params=component.params,
            metadata={key: value for key, value in component.metadata.items() if key != "weight"},
            weight=float(weight),
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["weight"] = self.weight
        return data


@dataclass(frozen=True)
class MetricConfig(ComponentConfig):
    stages: tuple[str, ...] = ("validation", "test")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, config_dir: Path, section: str) -> "MetricConfig":
        component = ComponentConfig.from_mapping(raw, config_dir=config_dir, section=section, category="metric")
        raw_stages = raw.get("stages", raw.get("stage", ("validation", "test")))
        if isinstance(raw_stages, str):
            stages = (raw_stages,)
        elif isinstance(raw_stages, list) and all(isinstance(stage, str) and stage for stage in raw_stages):
            stages = tuple(raw_stages)
        else:
            raise ConfigError(f"[{section}.stages] must be a string or list of strings when provided.")

        return cls(
            category=component.category,
            path=component.path,
            class_name=component.class_name,
            name=component.name,
            params=component.params,
            metadata={key: value for key, value in component.metadata.items() if key not in {"stage", "stages"}},
            stages=stages,
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["stages"] = list(self.stages)
        return data


@dataclass(frozen=True)
class TrackingConfig(ComponentConfig):
    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, config_dir: Path, section: str) -> "TrackingConfig":
        component = ComponentConfig.from_mapping(raw, config_dir=config_dir, section=section, category="tracking")
        return cls(
            category=component.category,
            path=component.path,
            class_name=component.class_name,
            name=component.name,
            params=component.params,
            metadata=component.metadata,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    config_path: Path
    seed: int | None
    device: DeviceConfig | None
    output_dir: Path
    data: ComponentConfig
    model: ComponentConfig
    training: ComponentConfig | None = None
    validation: ComponentConfig | None = None
    test: ComponentConfig | None = None
    prediction: ComponentConfig | None = None
    losses: list[LossConfig] = field(default_factory=list)
    metrics: list[MetricConfig] = field(default_factory=list)
    tracking: TrackingConfig = field(default_factory=lambda: TrackingConfig(category="tracking", name="local"))
    registry_paths: list[Path] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def single_components(self) -> list[ComponentConfig]:
        return [
            component
            for component in [
                self.data,
                self.model,
                self.training,
                self.validation,
                self.test,
                self.prediction,
                self.tracking,
            ]
            if component is not None
        ]

    @property
    def all_components(self) -> list[ComponentConfig]:
        return [*self.single_components, *self.losses, *self.metrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": str(self.config_path),
            "seed": self.seed,
            "device": copy.deepcopy(self.device),
            "output_dir": str(self.output_dir),
            "registry_paths": [str(path) for path in self.registry_paths],
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict() if self.training else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "test": self.test.to_dict() if self.test else None,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "losses": [loss.to_dict() for loss in self.losses],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "tracking": self.tracking.to_dict(),
        }


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_raw_config(config_path)
    return parse_experiment_config(raw, config_path=config_path)


def parse_experiment_config(raw: dict[str, Any], *, config_path: Path) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a table/object.")

    config_dir = config_path.parent
    experiment = raw.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ConfigError("[experiment] must be a table/object when provided.")

    name = experiment.get("name", config_path.stem)
    if not isinstance(name, str) or not name:
        raise ConfigError("[experiment.name] must be a non-empty string.")

    seed = experiment.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ConfigError("[experiment.seed] must be an integer when provided.")
    device = _experiment_device(experiment)

    output_dir_raw = experiment.get("output_dir", "runs")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ConfigError("[experiment.output_dir] must be a non-empty string.")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    registry_paths = _registry_paths(raw, config_dir=config_dir)

    data = _required_component(raw, config_dir=config_dir, section="data", category="data")
    model = _with_default_device(
        _required_component(raw, config_dir=config_dir, section="model", category="model"), device
    )
    training = _with_default_device(
        _optional_component(raw, config_dir=config_dir, section="training", category="training"), device
    )
    validation = _with_default_device(
        _optional_component(raw, config_dir=config_dir, section="validation", category="validation"), device
    )
    test = _with_default_device(_optional_component(raw, config_dir=config_dir, section="test", category="test"), device)
    prediction = _with_default_device(
        _optional_component(raw, config_dir=config_dir, section="prediction", category="prediction"), device
    )
    losses = [_with_default_device(loss, device) for loss in _losses(raw, config_dir=config_dir)]
    metrics = [_with_default_device(metric, device) for metric in _metrics(raw, config_dir=config_dir)]

    return ExperimentConfig(
        name=name,
        config_path=config_path,
        seed=seed,
        device=device,
        output_dir=output_dir.resolve(),
        registry_paths=registry_paths,
        data=data,
        model=model,
        training=training,
        validation=validation,
        test=test,
        prediction=prediction,
        losses=losses,
        metrics=metrics,
        tracking=_tracking(raw, config_dir=config_dir),
        raw=copy.deepcopy(raw),
    )


def _experiment_device(experiment: dict[str, Any]) -> DeviceConfig | None:
    device = experiment.get("device")
    if device is None:
        return None
    if isinstance(device, str) and device:
        return device
    if isinstance(device, list) and device:
        if all(isinstance(item, int) or (isinstance(item, str) and item) for item in device):
            return copy.deepcopy(device)
    raise ConfigError("[experiment.device] must be a non-empty string or non-empty list of GPU ids/device strings.")


def _with_default_device[T: ComponentConfig](component: T | None, device: DeviceConfig | None) -> T | None:
    if component is None or device is None or "device" in component.params:
        return component
    params = copy.deepcopy(component.params)
    params["device"] = copy.deepcopy(device)
    return replace(component, params=params)


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    raise ConfigError(f"Unsupported config extension '{config_path.suffix}'. Use .toml or .json.")


def _registry_paths(raw: dict[str, Any], *, config_dir: Path) -> list[Path]:
    registry = raw.get("registry", {})
    if not isinstance(registry, dict):
        raise ConfigError("[registry] must be a table/object when provided.")

    raw_paths = registry.get("paths", [])
    if not isinstance(raw_paths, list) or not all(isinstance(path, str) and path for path in raw_paths):
        raise ConfigError("[registry.paths] must be a list of non-empty strings when provided.")

    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = config_dir / path
        paths.append(path.resolve())
    return paths


def _required_component(
    raw: dict[str, Any],
    *,
    config_dir: Path,
    section: str,
    category: str,
) -> ComponentConfig:
    if section not in raw:
        raise ConfigError(f"Config is missing required section: [{section}]")
    return ComponentConfig.from_mapping(raw[section], config_dir=config_dir, section=section, category=category)


def _optional_component(
    raw: dict[str, Any],
    *,
    config_dir: Path,
    section: str,
    category: str,
) -> ComponentConfig | None:
    if section not in raw:
        return None
    return ComponentConfig.from_mapping(raw[section], config_dir=config_dir, section=section, category=category)


def _losses(raw: dict[str, Any], *, config_dir: Path) -> list[LossConfig]:
    raw_losses = raw.get("losses", [])
    if not isinstance(raw_losses, list):
        raise ConfigError("[[losses]] must be an array of tables/objects.")

    return [
        LossConfig.from_mapping(loss, config_dir=config_dir, section=f"losses[{index}]")
        for index, loss in enumerate(raw_losses)
    ]


def _metrics(raw: dict[str, Any], *, config_dir: Path) -> list[MetricConfig]:
    raw_metrics = raw.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raise ConfigError("[[metrics]] must be an array of tables/objects.")

    return [
        MetricConfig.from_mapping(metric, config_dir=config_dir, section=f"metrics[{index}]")
        for index, metric in enumerate(raw_metrics)
    ]


def _tracking(raw: dict[str, Any], *, config_dir: Path) -> TrackingConfig:
    raw_tracking = raw.get("tracking")
    if raw_tracking is None:
        return TrackingConfig(category="tracking", name="local")
    if not isinstance(raw_tracking, dict):
        raise ConfigError("[tracking] must be a table/object when provided.")

    tracking = copy.deepcopy(raw_tracking)
    if "backend" in tracking:
        if "name" in tracking:
            raise ConfigError("[tracking] cannot define both 'name' and legacy 'backend'.")
        tracking["name"] = tracking.pop("backend")

    if "name" not in tracking and "class_name" not in tracking:
        tracking["name"] = "local"

    params = tracking.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("[tracking.params] must be a table/object when provided.")

    for key in ("experiment_name", "run_name", "tracking_uri"):
        if key in tracking:
            params[key] = tracking.pop(key)
    tracking["params"] = params

    return TrackingConfig.from_mapping(tracking, config_dir=config_dir, section="tracking")

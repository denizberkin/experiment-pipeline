from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from eval_pipeline.core.config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentPaths:
    """Stable output paths available to all experiment mechanisms."""

    output_dir: Path
    experiment_dir: Path
    artifacts_dir: Path
    logs_dir: Path


@dataclass
class ExperimentContext:
    """Base context shared by stage-specific contexts."""

    config: ExperimentConfig
    paths: ExperimentPaths
    seed: int | None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingContext(ExperimentContext):
    stage: Literal["training"] = "training"


@dataclass
class ValidationContext(ExperimentContext):
    stage: Literal["validation"] = "validation"


@dataclass
class TestContext(ExperimentContext):
    stage: Literal["test"] = "test"


@dataclass
class PredictionContext(ExperimentContext):
    stage: Literal["prediction"] = "prediction"


def build_experiment_paths(config: ExperimentConfig) -> ExperimentPaths:
    experiment_dir = config.output_dir / config.name
    return ExperimentPaths(
        output_dir=config.output_dir,
        experiment_dir=experiment_dir,
        artifacts_dir=experiment_dir / "artifacts",
        logs_dir=experiment_dir / "logs",
    )


def make_training_context(config: ExperimentConfig, *, state: dict[str, Any] | None = None) -> TrainingContext:
    return TrainingContext(
        config=config,
        paths=build_experiment_paths(config),
        seed=config.seed,
        state=state if state is not None else {},
    )


def make_validation_context(config: ExperimentConfig, *, state: dict[str, Any] | None = None) -> ValidationContext:
    return ValidationContext(
        config=config,
        paths=build_experiment_paths(config),
        seed=config.seed,
        state=state if state is not None else {},
    )


def make_test_context(config: ExperimentConfig, *, state: dict[str, Any] | None = None) -> TestContext:
    return TestContext(
        config=config,
        paths=build_experiment_paths(config),
        seed=config.seed,
        state=state if state is not None else {},
    )


def make_prediction_context(config: ExperimentConfig, *, state: dict[str, Any] | None = None) -> PredictionContext:
    return PredictionContext(
        config=config,
        paths=build_experiment_paths(config),
        seed=config.seed,
        state=state if state is not None else {},
    )

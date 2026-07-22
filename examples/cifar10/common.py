from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from eval_pipeline.components.models import load_torch_checkpoint


def get_device(preferred: Any = None):
    if not preferred:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(preferred, list):
        return torch.device(preferred)

    first = preferred[0]
    if not isinstance(first, int):
        return torch.device(first)
    return torch.device(f"cuda:{first}")


def load_model(model: Any, *, checkpoint: str | Path | None = None, device: Any = None) -> Any:
    load_device = get_device(device) if device is not None else None
    if checkpoint:
        load_torch_checkpoint(model, checkpoint, map_location=load_device or "cpu")
    if load_device is not None:
        model.to(load_device)
    return model


def primary_loss(losses: dict[str, Any]):
    if not losses:
        raise ValueError("CIFAR-10 training requires at least one configured loss.")
    first = next(iter(losses.values()))
    return first.get("loss_fn", first)


def metric_fns(metrics: dict[str, Any], stage: str) -> dict[str, Any]:
    selected = {}
    for name, value in metrics.items():
        stages = value.get("stages") if isinstance(value, dict) else None
        if stages is None or stage in stages:
            selected[name] = value.get("metric_fn", value) if isinstance(value, dict) else value
    return selected


def reset_metrics(metrics: dict[str, Any]) -> None:
    for metric in metrics.values():
        reset = getattr(metric, "reset", None)
        if reset is not None:
            reset()


def compute_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(metric.compute()) for name, metric in metrics.items()}

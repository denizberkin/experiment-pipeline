from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def get_device(preferred: str | None = None):

    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def save_checkpoint(model: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path

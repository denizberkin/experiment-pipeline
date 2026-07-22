from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def save_torch_checkpoint(model: Any, path: str | Path) -> Path:
    """Atomically save a model state dictionary."""
    torch = _require_torch()
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_torch_checkpoint(
    model: Any,
    path: str | Path,
    *,
    map_location: Any = "cpu",
    strict: bool = True,
) -> Any:
    """Load a model state dictionary into an existing model."""
    torch = _require_torch()
    state = torch.load(
        Path(path).expanduser(),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state, strict=strict)
    return model


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch checkpoint support requires `pip install -e ".[torch]"`.') from exc
    return torch

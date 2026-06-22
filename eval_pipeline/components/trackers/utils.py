from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoggedRecord:
    kind: str
    name: str
    value: Any
    step: int | None = None
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": datetime.now(UTC).isoformat(),
            "kind": self.kind,
            "name": self.name,
            "value": json_safe(self.value),
            "step": self.step,
            "stage": self.stage,
        }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return repr(value)


def as_number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if isinstance(item, int | float):
            return float(item)
    return None


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def tracking_key(kind: str, name: str, stage: str | None) -> str:
    prefix = f"{stage}." if stage else ""
    return f"{prefix}{kind}.{name}"

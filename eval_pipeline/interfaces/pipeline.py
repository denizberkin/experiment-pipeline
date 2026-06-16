from __future__ import annotations

from abc import ABC
from typing import Any


class PipelineComponent(ABC):
    """Base class for user-provided pipeline components."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    @classmethod
    def get_alias(cls) -> str:
        return cls.__name__

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from eval_pipeline.interfaces.pipeline import PipelineComponent
from eval_pipeline.utils.device import torch_load_device


class ModelFactory[Model](PipelineComponent):
    """Creates an ML or DL model instance."""

    @abstractmethod
    def build(self) -> Model:
        """Return a model instance."""

    def load(self, model: Model) -> Model:
        load_device = torch_load_device(self.params.get("device"))
        checkpoint = self.params.get("checkpoint")
        if checkpoint:
            import torch

            state = torch.load(Path(checkpoint).expanduser(), map_location=load_device)
            model.load_state_dict(state)

        move = getattr(model, "to", None)
        if load_device is not None and move is not None:
            move(load_device)
        return model

    def save(self, model: Model, path: str | Path) -> Path:
        import torch

        checkpoint = Path(path).expanduser()
        if not checkpoint.parent.exists():
            raise FileNotFoundError(checkpoint.parent)
        torch.save(model.state_dict(), checkpoint)
        return checkpoint

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


@register_component("cifar10", category="model")
class Cifar10SmallCnnFactory(ModelFactory[Any]):
    def build(self) -> Any:
        num_classes = int(self.params.get("num_classes", 10))
        dropout = float(self.params.get("dropout", 0.2))

        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        checkpoint = self.params.get("checkpoint")
        if checkpoint:
            model.load_state_dict(torch.load(Path(checkpoint).expanduser(), map_location="cpu"))
        return model

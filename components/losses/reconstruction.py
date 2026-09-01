from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from eval_pipeline.components.losses.base import Loss
from eval_pipeline.registry import register_component


@register_component("task3_l1", category="loss")
class L1Loss(Loss[torch.Tensor, torch.Tensor, torch.Tensor]):
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(prediction, target)


@register_component("task3_lpips", category="loss")
class LPIPSLoss(Loss[torch.Tensor, torch.Tensor, torch.Tensor]):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        from mrixfields.losses.perceptual import PerceptualLoss

        configured = params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
        if isinstance(configured, int):
            configured = f"cuda:{configured}"
        device = torch.device(configured or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.loss = PerceptualLoss(net=str(params.get("net", "alex"))).to(device).eval()

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS is a 2D network. For volumetric models score the axial slices
        # and average, the usual 2.5D perceptual loss; 4D input is unchanged.
        if prediction.ndim == 5:
            batch, channels, depth = prediction.shape[:3]
            shape = (batch * depth, channels, *prediction.shape[3:])
            prediction = prediction.permute(0, 2, 1, 3, 4).reshape(shape)
            target = target.permute(0, 2, 1, 3, 4).reshape(shape)
        return self.loss(prediction, target)

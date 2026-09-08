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


@register_component("task3_ssim", category="loss")
class SSIMLoss(Loss[torch.Tensor, torch.Tensor, torch.Tensor]):
    """1 - SSIM, computed the way the challenge's scorer computes it.

    SSIM is the ranked metric and nothing in the objective was optimising it: the loss was
    L1 + 0.05*LPIPS, which is why the submissions sit 3rd of 23 on LPIPS and 15th on nRMSE.

    This deliberately reproduces ``skimage.metrics.structural_similarity`` with its defaults
    rather than the Gaussian-windowed SSIM that most implementations ship, because the former
    is what ``scripts/eval_holdout.py`` scores with and what the leaderboard reports:

      * a 7x7 **uniform** window, not an 11x11 Gaussian one;
      * the unbiased covariance normalisation ``NP / (NP - 1)``;
      * the border dropped rather than padded --- skimage filters with reflection and then
        crops ``(win_size - 1) // 2``, which is exactly a 'valid' filter, so avg_pool2d with
        no padding gives the same numbers without the wasted work.

    Inputs arrive in the model's [-1, 1] convention and are mapped to [0, 1] here, so
    ``data_range`` is 1.0 and matches the scorer's call.
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.window = int(params.get("window", 7))
        self.k1 = float(params.get("k1", 0.01))
        self.k2 = float(params.get("k2", 0.03))

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1]. The prediction is not clamped: clamping is what the model does
        # at inference, and doing it here would zero the gradient wherever it saturates.
        x = prediction.add(1).div(2)
        y = target.add(1).div(2)

        window, points = self.window, self.window ** 2
        cov_norm = points / (points - 1)          # skimage's ddof=1
        mu_x = F.avg_pool2d(x, window, stride=1)
        mu_y = F.avg_pool2d(y, window, stride=1)
        vx = cov_norm * (F.avg_pool2d(x * x, window, stride=1) - mu_x * mu_x)
        vy = cov_norm * (F.avg_pool2d(y * y, window, stride=1) - mu_y * mu_y)
        vxy = cov_norm * (F.avg_pool2d(x * y, window, stride=1) - mu_x * mu_y)

        c1, c2 = self.k1 ** 2, self.k2 ** 2       # data_range = 1
        ssim = (((2 * mu_x * mu_y + c1) * (2 * vxy + c2))
                / ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2)))
        return 1 - ssim.mean()

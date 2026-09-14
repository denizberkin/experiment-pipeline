"""Conditional flow matching velocity loss (section 38).

The CFM objective is a plain squared error on the velocity,

    L = E_{t, x0, x1} || v_theta(x_t, t) - (x1 - x0) ||^2                    (their eq. 2)

so the loss component itself is just MSE; everything specific to flow matching -- drawing
t, interpolating x_t, forming the target velocity -- lives in the trainer, where it belongs,
because the loss interface only ever sees (prediction, target).

Registered separately from ``task3_l1`` rather than reusing it despite both being simple
regressions, because the quantities differ: L1 on an image and L2 on a velocity are not
interchangeable, and a config that names ``task3_cfm`` should fail loudly if it is ever
pointed at an image-space trainer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eval_pipeline.components.losses.base import Loss
from eval_pipeline.registry import register_component


@register_component("task3_cfm", category="loss")
class FlowMatchingLoss(Loss[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Squared error between the predicted and true conditional-flow velocity.

    Squared, not absolute: the CFM derivation regresses the conditional expectation of the
    velocity, and the mean is the minimiser of L2. An L1 objective would fit the median
    instead and the resulting field would not be the marginal velocity the ODE assumes.
    That the mean estimator blurs is the known cost, and is exactly what the paper's
    adversarial stage is for -- not something to fix by swapping the norm here.
    """

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)


@register_component("task3_cfm_masked", category="loss")
class MaskedFlowMatchingLoss(Loss[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Velocity MSE restricted to the contrasts a sample actually has.

    The retrospective cohort used for degradation-bridge pretraining is missing contrasts
    for some subjects, and a stacked three-channel input has to put *something* in those
    channels. Zero is a legitimate intensity here, so an absent contrast that is zero-filled
    and then scored would train the model to predict air where data merely does not exist.
    The trainer passes the per-channel mask through ``target`` as a second tensor.
    """

    def __call__(self, prediction: torch.Tensor, target: tuple[torch.Tensor, torch.Tensor]
                 ) -> torch.Tensor:
        velocity, mask = target
        if mask is None:
            return F.mse_loss(prediction, velocity)
        # mask is [B, C]; broadcast over the spatial dims and average over kept entries
        weights = mask.to(prediction.dtype).reshape(mask.shape[0], mask.shape[1], 1, 1)
        squared = (prediction - velocity).pow(2) * weights
        denominator = weights.expand_as(squared).sum().clamp(min=1.0)
        return squared.sum() / denominator

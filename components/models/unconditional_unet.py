from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component

# Reuse the conditional model's block verbatim rather than copying it, so the
# only difference between the two architectures is the bottleneck conditioning.
from components.models.conditional_unet import _ConvBlock


class UnconditionalUNet(nn.Module):
    """task3_conditional_unet with the domain conditioning removed, and nothing
    else changed. The controlled ablation for "does conditioning help?".

    Removed relative to task3_conditional_unet:
        - nn.Embedding(num_domains, bottleneck_channels) for source and target
        - the broadcast add of (E_src[d_s] + E_tgt[d_t]) onto the bottleneck

    Identical to it in every other respect: level count, channel schedule
    (min(base * 2**level, max_channels)), _ConvBlock internals
    (Conv3x3 -> InstanceNorm2d(affine=True) -> LeakyReLU(0.2), twice),
    MaxPool2d(2) downsampling, ConvTranspose2d(k=2, s=2) upsampling, concat
    skip fusion, and the Conv1x1 -> Tanh head. Parameter tensors match by name
    and shape; the only difference is the two absent embedding tables
    (2 x num_domains x bottleneck_channels).

    forward() still accepts the domain tensors so the shared
    task3_unet_trainer works unchanged, but ignores them - that is the ablation:
    the model cannot know which transfer it is being asked to perform.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        max_channels: int = 512,
        levels: int = 4,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        channels = [min(base_channels * 2**level, max_channels) for level in range(levels)]
        bottleneck_channels = min(channels[-1] * 2, max_channels)

        self.encoders = nn.ModuleList()
        previous = input_channels
        for channel in channels:
            self.encoders.append(_ConvBlock(previous, channel))
            previous = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for channel in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous, channel, 2, stride=2))
            self.decoders.append(_ConvBlock(channel * 2, channel))
            previous = channel
        self.output = nn.Sequential(nn.Conv2d(channels[0], output_channels, 1), nn.Tanh())

    def forward(self, image: torch.Tensor, *_: object) -> torch.Tensor:
        skips = []
        x = image
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)


@register_component("task3_unconditional_unet", category="model")
class UnconditionalUNetFactory(ModelFactory[UnconditionalUNet]):
    def build(self) -> UnconditionalUNet:
        model = UnconditionalUNet(
            input_channels=int(self.params.get("input_channels", 1)),
            output_channels=int(self.params.get("output_channels", 1)),
            base_channels=int(self.params.get("base_channels", 32)),
            max_channels=int(self.params.get("max_channels", 512)),
            levels=int(self.params.get("levels", 4)),
        )
        checkpoint = self.params.get("checkpoint")
        device = self._device()
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state.get("model", state))
        return model.to(device)

    def _device(self) -> torch.device:
        configured = self.params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
        if isinstance(configured, int):
            configured = f"cuda:{configured}"
        device = torch.device(configured or "cpu")
        return torch.device("cpu") if device.type == "cuda" and not torch.cuda.is_available() else device

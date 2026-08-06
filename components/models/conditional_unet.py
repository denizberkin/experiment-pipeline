from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_domains: int = 15,
        base_channels: int = 64,
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
        self.source_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.target_embedding = nn.Embedding(num_domains, bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for channel in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous, channel, 2, stride=2))
            self.decoders.append(_ConvBlock(channel * 2, channel))
            previous = channel
        self.output = nn.Sequential(nn.Conv2d(channels[0], output_channels, 1), nn.Tanh())

    def forward(
        self,
        image: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skips = []
        x = image
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        source_domain = target_domain if source_domain is None else source_domain
        conditioning = self.source_embedding(source_domain) + self.target_embedding(target_domain)
        x = x + conditioning.unsqueeze(-1).unsqueeze(-1)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)


@register_component("task3_conditional_unet", category="model")
class ConditionalUNetFactory(ModelFactory[ConditionalUNet]):
    def build(self) -> ConditionalUNet:
        model = ConditionalUNet(
            input_channels=int(self.params.get("input_channels", 1)),
            output_channels=int(self.params.get("output_channels", 1)),
            num_domains=int(self.params.get("num_domains", 15)),
            base_channels=int(self.params.get("base_channels", 64)),
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

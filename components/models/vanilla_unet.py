from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


class _DoubleConv(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, the standard U-Net block."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class VanillaUNet(nn.Module):
    """Standard U-Net (Ronneberger et al., 2015).

    Contracting path of `levels` _DoubleConv blocks with channels doubling from
    base_channels, MaxPool2d(2) between them; a bottleneck at 2x the last
    encoder width; an expanding path of ConvTranspose2d(k=2, s=2) upsampling
    with concatenated skip connections and a _DoubleConv after each; a 1x1
    convolution to output_channels.

    Padded 3x3 convolutions are used (rather than the paper's unpadded convs
    with cropped skips) so input and output share spatial dimensions.

    A Tanh output head is applied because the pipeline supplies and expects
    images in [-1, 1]; set `tanh_output = false` for a raw linear head.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        levels: int = 4,
        tanh_output: bool = True,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        channels = [base_channels * 2**level for level in range(levels)]

        self.encoders = nn.ModuleList()
        previous = input_channels
        for channel in channels:
            self.encoders.append(_DoubleConv(previous, channel))
            previous = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(channels[-1], channels[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = channels[-1] * 2
        for channel in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous, channel, 2, stride=2))
            self.decoders.append(_DoubleConv(channel * 2, channel))
            previous = channel

        head: list[nn.Module] = [nn.Conv2d(channels[0], output_channels, 1)]
        if tanh_output:
            head.append(nn.Tanh())
        self.output = nn.Sequential(*head)

    def forward(self, image: torch.Tensor, *_: object) -> torch.Tensor:
        # Extra positional arguments are accepted and discarded: the shared
        # task3_unet_trainer passes (source, target_domain, source_domain), and
        # a plain U-Net takes only the image.
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


@register_component("task3_vanilla_unet", category="model")
class VanillaUNetFactory(ModelFactory[VanillaUNet]):
    def build(self) -> VanillaUNet:
        model = VanillaUNet(
            input_channels=int(self.params.get("input_channels", 1)),
            output_channels=int(self.params.get("output_channels", 1)),
            base_channels=int(self.params.get("base_channels", 32)),
            levels=int(self.params.get("levels", 4)),
            tanh_output=bool(self.params.get("tanh_output", True)),
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

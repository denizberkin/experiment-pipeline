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


class _FiLM(nn.Module):
    """Scale and shift both normalisations of one decoder block.

    A conditioning vector *added* at the bottleneck is exactly the quantity
    InstanceNorm subtracts back out again, so little of it survives the trip to
    the output - the same leak that forced _ConditionalRefinement onto the Swin
    decoder. A scale and shift applied *after* a norm cannot be cancelled
    downstream, and the norm still bounds what the convolutions build up; the
    ViT's first run protected its embedding by dropping normalisation instead
    and its decoder features grew 12x.

    One projection covers both norms of the block - chunk(4) rather than two
    Linear layers, same parameter count - and is zero-initialised in weight
    *and* bias, so gamma = beta = 0 and the block computes exactly the plain
    _ConvBlock the existing checkpoints were trained as.
    """

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embed_dim, 4 * channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, block: _ConvBlock, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        # The block stays an nn.Sequential and is unpacked here rather than
        # rewritten as a FiLM-aware module: renaming its children would rename
        # every decoder tensor and break every existing checkpoint.
        conv1, norm1, activation1, conv2, norm2, activation2 = block
        gamma1, beta1, gamma2, beta2 = (
            part[:, :, None, None] for part in self.projection(conditioning).chunk(4, dim=1)
        )
        x = activation1(norm1(conv1(x)) * (1 + gamma1) + beta1)
        return activation2(norm2(conv2(x)) * (1 + gamma2) + beta2)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_domains: int = 15,
        base_channels: int = 64,
        max_channels: int = 512,
        levels: int = 4,
        residual_output: bool = False,
        film_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if residual_output and input_channels != output_channels and output_channels != 1:
            raise ValueError(
                "residual_output predicts a correction to the source image, so "
                f"input_channels ({input_channels}) must equal output_channels "
                f"({output_channels}), or output a single channel that the residual is "
                "added to (multi-contrast input)"
            )
        self.input_channels = int(input_channels)
        channels = [min(base_channels * 2**level, max_channels) for level in range(levels)]
        bottleneck_channels = min(channels[-1] * 2, max_channels)

        self.encoders = nn.ModuleList()
        previous = input_channels
        for channel in channels:
            self.encoders.append(_ConvBlock(previous, channel))
            previous = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(channels[-1], bottleneck_channels)
        # 5*m + c -> (m \in {t1, t2, t2*}) and (c \in {0.1T, 1.5T, 3T, 5T, 7T}) -> {0...14}
        self.source_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.target_embedding = nn.Embedding(num_domains, bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for channel in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous, channel, 2, stride=2))
            self.decoders.append(_ConvBlock(channel * 2, channel))
            previous = channel

        # Off by default: both flags add tensors or move them, and the
        # task3_unet_pro artifacts have to keep loading with strict=True.
        self.film_projections = (
            nn.ModuleList(_FiLM(bottleneck_channels, channel) for channel in reversed(channels))
            if film_conditioning
            else None
        )
        if residual_output:
            # Deliberately not named output.*: the head is what tells a loader
            # which of the two forwards a checkpoint was trained for, so the
            # submission script can read it off the tensors instead of being
            # handed a flag that may be wrong (same reason as derive_shape).
            self.residual_head = nn.Conv2d(channels[0], output_channels, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            self.output = None
        else:
            self.residual_head = None
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
        # The bottleneck add survives even with FiLM on. Removing it would
        # change the function a loaded checkpoint computes, and the whole point
        # of the zero-init FiLM is that step 0 reproduces that checkpoint.
        x = x + conditioning.unsqueeze(-1).unsqueeze(-1)
        films = (
            [None] * len(self.decoders) if self.film_projections is None else self.film_projections
        )
        for upconv, decoder, film, skip in zip(
            self.upconvs, self.decoders, films, reversed(skips), strict=True
        ):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            merged = torch.cat((x, skip), dim=1)
            x = decoder(merged) if film is None else film(decoder, merged, conditioning)

        if self.residual_head is None:
            return self.output(x)
        # The residual is added in the bounded range itself, not in tanh space
        # (the [-1, 1] analogue of the tubelet model's sigmoid(logit(x) + r)):
        # this model's I/O convention is [-1, 1] and ~85% of a slice is air at
        # -1, where atanh puts the base near -5 and the tanh derivative is
        # ~1e-4 - the saturation that froze the ViT's first run at a constant.
        # Zero-init head means step 0 is the exact identity, which already
        # scores SSIM 0.836 here. Clamping only outside training bounds the
        # prediction for inference without removing the gradient during it.
        # Multi-contrast input puts the contrast being predicted in channel 0 (see
        # CachedMultiContrastDataset), so the residual is added there. Single-channel input
        # takes the whole image, which is the same expression when there is one channel.
        base = image if image.shape[1] == self.residual_head.out_channels else image[:, :1]
        prediction = base + self.residual_head(x)
        return prediction if self.training else prediction.clamp(-1, 1)


def unet_from_state_dict(state: dict[str, "torch.Tensor"]) -> "ConditionalUNet":
    """Rebuild the exact architecture a checkpoint was trained with, then load it.

    Every consumer that hardcoded ``base_channels=32, max_channels=512, levels=4``
    was already one config edit away from a silent mismatch, and the residual head
    and per-scale FiLM add two more things to get wrong. All four are visible in
    the tensors, so read them rather than trust a caller: widths come from the
    first encoder conv and the bottleneck, and the two flags from the key names
    they move (``residual_head.*`` replaces ``output.0.*``; FiLM adds
    ``film_projections.*``). The load is strict, so a wrong guess still fails
    loudly instead of quietly computing something else.
    """
    base_channels = int(state["encoders.0.0.weight"].shape[0])
    input_channels = int(state["encoders.0.0.weight"].shape[1])
    levels = len({key.split(".")[1] for key in state if key.startswith("encoders.")})
    max_channels = int(state["bottleneck.0.weight"].shape[0])
    model = ConditionalUNet(
        input_channels=input_channels,
        base_channels=base_channels,
        max_channels=max_channels,
        levels=levels,
        num_domains=int(state["target_embedding.weight"].shape[0]),
        residual_output="residual_head.weight" in state,
        film_conditioning=any(key.startswith("film_projections.") for key in state),
    )
    model.load_state_dict(state, strict=True)
    return model


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
            residual_output=bool(self.params.get("residual_output", False)),
            film_conditioning=bool(self.params.get("film_conditioning", False)),
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

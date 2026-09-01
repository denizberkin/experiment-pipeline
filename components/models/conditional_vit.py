from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component

try:
    import timm
except ImportError as error:  # pragma: no cover - surfaced at config load
    raise ImportError(
        "task3_conditional_vit needs timm: pip install --no-deps timm huggingface_hub safetensors"
    ) from error


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm over at most 32 groups, whatever the channel count divides into."""
    return nn.GroupNorm(math.gcd(32, channels), channels)


class _ResidualConvUnit(nn.Module):
    """DPT refinement unit with adaptive normalisation.

    Conditioning and normalisation are not the either/or they look like. A
    domain embedding *added* to the features is exactly what a per-channel
    normalisation subtracts back out, so the first version of this block
    dropped normalisation to protect the embedding - and paid for it: with no
    brake on activation scale the decoder features grew 12x during training
    (std 5.9 -> 73.6), drove the output Tanh to a pre-activation of -109, and
    killed every gradient in the network.

    Normalising first and applying the embedding as a scale and shift
    *afterwards* keeps both properties. Nothing downstream of the modulation
    can cancel it, and the norm still bounds what the convolutions can build
    up. This is the adaptive-normalisation block used by conditional diffusion
    UNets and DiT. The modulation starts at zero, so the block begins as a
    plain normalised residual unit and learns how much conditioning it wants.
    """

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.modulation = nn.Linear(embed_dim, 2 * channels)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x)))
        scale, shift = self.modulation(conditioning).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.relu(h))
        return x + h


class _FusionBlock(nn.Module):
    """Merge the coarser decoder state into this scale's skip, then refine."""

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.refine_skip = _ResidualConvUnit(channels, embed_dim)
        self.refine_out = _ResidualConvUnit(channels, embed_dim)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(
        self, skip: torch.Tensor, coarser: torch.Tensor | None, conditioning: torch.Tensor
    ) -> torch.Tensor:
        x = self.refine_skip(skip, conditioning)
        if coarser is not None:
            x = x + F.interpolate(coarser, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(self.refine_out(x, conditioning))


class ConditionalViT(nn.Module):
    """Pretrained ViT encoder with a DPT-style decoder and domain conditioning.

    A source and a target embedding are summed into one vector, which every
    refinement unit turns into its own scale and shift, applied after that
    unit's normalisation. The UNet injects once at its bottleneck; the
    transformer has no single bottleneck, so the conditioning reaches all four
    scales. Conditioning only the deepest one would leave the shallower skips
    free to route unconditioned detail to the output.
    """

    def __init__(
        self,
        backbone: str = "vit_large_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        input_channels: int = 1,
        output_channels: int = 1,
        num_domains: int = 15,
        decoder_channels: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True
        )
        self.input_channels = input_channels
        embed_dim = self.encoder.embed_dim
        depth = len(self.encoder.blocks)
        # Evenly spaced taps, deepest last, matching DPT's [5, 11, 17, 23] on ViT-L/24.
        self.hook_indices = [round(depth * fraction) - 1 for fraction in (0.25, 0.5, 0.75, 1.0)]

        config = self.encoder.pretrained_cfg
        mean = torch.tensor(config.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std = torch.tensor(config.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        self.reassemble = nn.ModuleList(
            nn.Conv2d(embed_dim, decoder_channels, 1) for _ in self.hook_indices
        )
        # 5*m + c -> (m \in {t1, t2, t2*}) and (c \in {0.1T, 1.5T, 3T, 5T, 7T}) -> {0...14}
        self.source_embedding = nn.Embedding(num_domains, decoder_channels)
        self.target_embedding = nn.Embedding(num_domains, decoder_channels)

        self.fusions = nn.ModuleList(
            _FusionBlock(decoder_channels, decoder_channels) for _ in self.hook_indices
        )
        if input_channels != output_channels:
            raise ValueError(
                "the decoder predicts a correction to the source image, so "
                f"input_channels ({input_channels}) must equal output_channels ({output_channels})"
            )
        self.output = nn.Sequential(
            _group_norm(decoder_channels),
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, output_channels, 1),
        )
        # Zero-initialised, so the network starts as the exact identity and
        # learns a correction outward from there.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Map the [-1, 1] slice onto the encoder's pretraining statistics.

        Replicating to three channels and reusing the pretrained patch
        embedding unchanged is exact, unlike collapsing its weights to one
        channel, which cannot reproduce three different per-channel means.
        """
        x = image.mul(0.5).add(0.5)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.pixel_mean) / self.pixel_std

    def forward(
        self,
        image: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        features = self.encoder.forward_intermediates(
            self._prepare(image),
            indices=self.hook_indices,
            output_fmt="NCHW",
            intermediates_only=True,
            norm=False,
        )

        source_domain = target_domain if source_domain is None else source_domain
        conditioning = self.source_embedding(source_domain) + self.target_embedding(target_domain)

        # Shallow taps carry fine detail, deep taps carry semantics, so the
        # shallowest is reassembled at the finest stride.
        strides = (4, 8, 16, 32)
        scaled = []
        for tap, project, stride in zip(features, self.reassemble, strides, strict=True):
            size = (max(height // stride, 1), max(width // stride, 1))
            x = project(tap)
            if x.shape[-2:] != size:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            scaled.append(x)

        fused = None
        for skip, fusion in zip(reversed(scaled), reversed(self.fusions), strict=True):
            fused = fusion(skip, fused, conditioning)

        fused = F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=False)

        # Predicting a correction to the source rather than the image itself,
        # with no saturating output activation. A Tanh here is a trap, not a
        # bound: ~85% of every slice is air, so "-1 everywhere" is the
        # L1-optimal constant, and a saturated Tanh has no gradient left to
        # leave it with. The first attempt did exactly that - by epoch 2 the
        # pre-activation had reached -109, every pixel was pinned at -1, and the
        # loss sat frozen at 0.2536 while the optimizer went on taking steps.
        # The identity skip also stops the output ever becoming input-
        # independent, and clamping only outside training bounds the prediction
        # for inference without blocking a gradient during it.
        prediction = image + self.output(fused)
        return prediction if self.training else prediction.clamp(-1, 1)

    def optimizer_param_groups(self, learning_rate: float, encoder_scale: float) -> list[dict]:
        """Discriminative learning rates: a pretrained backbone needs a smaller
        step than a randomly initialised decoder, or the first epochs wash the
        pretrained features out."""
        decoder = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("encoder.") and parameter.requires_grad
        ]
        encoder = [
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith("encoder.") and parameter.requires_grad
        ]
        groups = [{"params": decoder, "lr": learning_rate}]
        if encoder:
            groups.append({"params": encoder, "lr": learning_rate * encoder_scale})
        return groups


@register_component("task3_conditional_vit", category="model")
class ConditionalViTFactory(ModelFactory[ConditionalViT]):
    def build(self) -> ConditionalViT:
        model = ConditionalViT(
            backbone=str(self.params.get("backbone", "vit_large_patch16_dinov3.lvd1689m")),
            pretrained=bool(self.params.get("pretrained", True)),
            input_channels=int(self.params.get("input_channels", 1)),
            output_channels=int(self.params.get("output_channels", 1)),
            num_domains=int(self.params.get("num_domains", 15)),
            decoder_channels=int(self.params.get("decoder_channels", 256)),
            freeze_encoder=bool(self.params.get("freeze_encoder", False)),
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

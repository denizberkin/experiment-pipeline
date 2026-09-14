"""Velocity network for conditional flow matching between field strengths (section 38).

Follows Imre et al., "Conditional Flow Matching for Cross-Field MRI Harmonisation"
(arXiv:2609.00960), whose reported 0.909 we are trying to reproduce and then pass. The
difference from ConditionalUNet is what the network *predicts*: a velocity field
v(x_t, t, s, tau), not an image or a residual on one. So there is no residual head here --
adding a velocity to an image would be a category error -- and no Tanh, because a velocity
is x1 - x0 and lives in [-2, 2] for inputs in [-1, 1].

Three deliberate carry-overs from ConditionalUNet, so the two stay comparable:

  * the same _ConvBlock trunk, channel schedule and skip layout, imported rather than
    copied, so a difference in results is the objective and not the architecture;
  * the same conditioning-at-the-bottleneck add;
  * the same zero-initialised FiLM after both decoder norms.

The one addition is the timestep. The paper sums all three signals into one vector,

    c = emb_t(t) + emb_s(s) + emb_tau(tau)                                    (their eq. 3)

which is what this does. A sinusoidal embedding is used for t rather than a learned table
because t is continuous and sampled fresh every step -- there is nothing to look up.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.models.conditional_unet import _ConvBlock, _FiLM
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


class _TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0, 1], then an MLP, as in the diffusion literature.

    Frequencies are geometric between 1 and ``max_period``; t is scaled by 1000 first so
    that the usable band matches the 0..1000 range those frequencies were chosen for. A
    learned table is not an option: t is continuous.
    """

    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_period = float(max_period)
        half = self.channels // 2
        frequencies = torch.exp(-math.log(self.max_period) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().reshape(-1, 1) * 1000.0 * self.frequencies.reshape(1, -1)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.channels:  # odd channel counts
            embedding = F.pad(embedding, (0, self.channels - embedding.shape[-1]))
        return self.mlp(embedding)


class ConditionalFlowUNet(nn.Module):
    """U-Net that regresses the conditional-flow velocity for every contrast at once.

    ``channels`` is 3 by default -- T1W, T2W, T2FLAIR stacked, in and out -- because the
    paper predicts a three-channel velocity in a single pass and the contrasts of one
    subject at one field are co-registered, so they share every spatial feature. That is
    also why there is no "primary" channel here as in CachedMultiContrastDataset: nothing
    is being singled out for prediction.

    The network never receives x0 separately, only the interpolant x_t. That is the
    paper's "leak-free" bridge and it is load-bearing rather than stylistic: given both x0
    and x_t a network can recover x1 by linear algebra and the objective collapses to
    plain regression, which their own Table 1 shows scoring lower (0.906 vs 0.909).
    """

    def __init__(
        self,
        channels: int = 3,
        num_domains: int = 15,
        base_channels: int = 64,
        max_channels: int = 512,
        levels: int = 4,
        film_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.channels = int(channels)
        widths = [min(base_channels * 2**level, max_channels) for level in range(levels)]
        bottleneck_channels = min(widths[-1] * 2, max_channels)

        self.encoders = nn.ModuleList()
        previous = self.channels
        for width in widths:
            self.encoders.append(_ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(widths[-1], bottleneck_channels)

        self.source_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.target_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.time_embedding = _TimestepEmbedding(bottleneck_channels)
        # The degradation bridge trains with no source field (the retrospective cohort is
        # unpaired, so there is nothing to name). An extra row stands for "unspecified"
        # rather than reusing a real field, which would teach the model a wrong pairing.
        self.null_source = nn.Parameter(torch.zeros(bottleneck_channels))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
            self.decoders.append(_ConvBlock(width * 2, width))
            previous = width

        self.film_projections = (
            nn.ModuleList(_FiLM(bottleneck_channels, width) for width in reversed(widths))
            if film_conditioning
            else None
        )
        # Zero-init: at step 0 the predicted velocity is exactly 0, so the ODE does not
        # move and the sampler returns the source unchanged -- the identity, which the
        # challenge already scores at 0.836. Same discipline as the residual head.
        self.velocity_head = nn.Conv2d(widths[0], self.channels, 1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @staticmethod
    def _embed(table: torch.nn.Embedding, domain: torch.Tensor) -> torch.Tensor:
        """Look up one or several domains per sample and sum them.

        ``domain`` is [B] for a single index per sample, or [B, K] for K of them -- which is
        how the joint (modality, field) conditioning arrives: one index per contrast channel,
        summed. Summing rather than concatenating keeps the conditioning vector one width
        regardless of K, so the FiLM projections and the bottleneck add are unchanged.
        """
        embedded = table(domain.clamp(min=0))
        return embedded.sum(dim=1) if embedded.dim() == 3 else embedded

    def conditioning(
        self, t: torch.Tensor, target_domain: torch.Tensor, source_domain: torch.Tensor | None
    ) -> torch.Tensor:
        """c = emb_t(t) + emb_s(s) + emb_tau(tau), their eq. 3.

        s and tau carry **modality and field strength**, not field alone: with
        ``num_domains = 15`` they are the joint indices ``MODALITIES.index(m) * 5 +
        FIELDS.index(f)``, passed as [B, 3] -- one per contrast channel -- and summed. The
        network predicts all three contrasts in one pass, so there is no single modality per
        sample; giving it all three lets the embedding tables hold per-(modality, field)
        structure instead of forcing every contrast at a field to share one row.

        A [B] tensor of plain field indices still works (``num_domains = 5``), which is what
        the paper does.
        """
        vector = self.time_embedding(t) + self._embed(self.target_embedding, target_domain)
        if source_domain is None:
            return vector + self.null_source.unsqueeze(0)
        # -1 marks "unspecified", so a degradation-bridge batch can mix specified and not.
        unspecified = (source_domain < 0).reshape(source_domain.shape[0], -1).all(dim=1)
        embedded = self._embed(self.source_embedding, source_domain)
        if bool(unspecified.any()):
            embedded = torch.where(
                unspecified.unsqueeze(-1), self.null_source.unsqueeze(0).expand_as(embedded),
                embedded,
            )
        return vector + embedded

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skips = []
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)
        conditioning = self.conditioning(t, target_domain, source_domain)
        h = h + conditioning.unsqueeze(-1).unsqueeze(-1)

        films = [None] * len(self.decoders) if self.film_projections is None else self.film_projections
        for upconv, decoder, film, skip in zip(
            self.upconvs, self.decoders, films, reversed(skips), strict=True
        ):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            merged = torch.cat((h, skip), dim=1)
            h = decoder(merged) if film is None else film(decoder, merged, conditioning)
        return self.velocity_head(h)


@torch.no_grad()
def heun_sample(
    model: ConditionalFlowUNet,
    x0: torch.Tensor,
    target_domain: torch.Tensor,
    source_domain: torch.Tensor | None = None,
    steps: int = 5,
) -> torch.Tensor:
    """Integrate the velocity from t=0 to t=1 with Heun's method (their eqs. 5-6).

        predictor  x~(t+h) = x(t) + h v(x(t), t)
        corrector  x(t+h)  = x(t) + (h/2) [ v(x(t), t) + v(x~(t+h), t+h) ]

    Two network calls per step, so ``steps=5`` is ten forward passes per slice. Five is
    where their ablation saturates (1 step 0.817, 5 steps 0.909, 10 steps 0.909); one step
    collapses because adversarial refinement calibrates the field for multi-step
    integration.

    Kept next to the model rather than in the trainer so validation and submission
    integrate the same way by construction -- section 20.1 cost 0.006 SSIM to a scoring
    path that had drifted from the training one.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    x = x0
    h = 1.0 / steps
    for index in range(steps):
        t = torch.full((x.shape[0],), index * h, device=x.device, dtype=torch.float32)
        t_next = torch.full_like(t, (index + 1) * h)
        v = model(x, t, target_domain, source_domain)
        x_predicted = x + h * v
        v_next = model(x_predicted, t_next, target_domain, source_domain)
        x = x + 0.5 * h * (v + v_next)
    return x


@register_component("task3_conditional_flow_unet", category="model")
class ConditionalFlowUNetFactory(ModelFactory[ConditionalFlowUNet]):
    def build(self) -> ConditionalFlowUNet:
        model = ConditionalFlowUNet(
            channels=int(self.params.get("channels", 3)),
            num_domains=int(self.params.get("num_domains", 15)),
            base_channels=int(self.params.get("base_channels", 64)),
            max_channels=int(self.params.get("max_channels", 512)),
            levels=int(self.params.get("levels", 4)),
            film_conditioning=bool(self.params.get("film_conditioning", True)),
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

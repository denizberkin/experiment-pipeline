"""Conditional multi-scale PatchGAN discriminator for flow-matching refinement (38.4).

After Imre et al. (arXiv:2609.00960) section 2.5, which follows Isola et al. for the patch
discriminator and Salimans et al. for the feature-matching term.

Two properties matter and both are deliberate:

*Patch, not global.* The output is a grid of judgements over local receptive fields rather
than one number per image. The refinement exists to restore high-frequency texture that a
squared-error velocity regression averages away; a global critic scores anatomy, which is
already correct, and would supply no gradient where the blur actually is.

*Conditional.* It sees the source x0 alongside the candidate and is told the two fields, so
it judges "is this a plausible 3T image **of this subject, from this 0.1T scan**" rather
than "is this a plausible brain". An unconditional critic is satisfied by any sharp brain,
which is precisely the failure mode -- hallucinated texture -- the faithfulness anchor and
this conditioning exist to prevent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _embed(table: nn.Embedding, domain: torch.Tensor) -> torch.Tensor:
    """Sum the embeddings of one or several domain indices per sample.

    Mirrors ConditionalFlowUNet._embed so the critic is conditioned on exactly what the
    generator is: [B] field indices, or [B, 3] joint (modality, field) indices.
    """
    embedded = table(domain.clamp(min=0))
    return embedded.sum(dim=1) if embedded.dim() == 3 else embedded


class PatchDiscriminator(nn.Module):
    """One scale of the PatchGAN. Returns (logits, intermediate features)."""

    def __init__(self, in_channels: int, base_channels: int = 64, layers: int = 3,
                 num_domains: int = 5) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        previous = in_channels
        width = base_channels
        # First block has no norm, as in the original PatchGAN: InstanceNorm on the raw
        # input would divide out the very intensity statistics being judged.
        self.blocks.append(nn.Sequential(
            nn.Conv2d(previous, width, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)))
        previous = width
        for _ in range(layers - 1):
            width = min(width * 2, 512)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(previous, width, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(width, affine=True),
                nn.LeakyReLU(0.2, inplace=True)))
            previous = width
        width = min(width * 2, 512)
        self.blocks.append(nn.Sequential(
            nn.Conv2d(previous, width, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.LeakyReLU(0.2, inplace=True)))
        self.logits = nn.Conv2d(width, 1, 4, stride=1, padding=1)

        # The field pair enters as a per-channel bias after the first block -- the same
        # additive conditioning the generator uses at its bottleneck, and enough for a
        # critic that only has to tell which of two field pairs it is looking at.
        self.source_embedding = nn.Embedding(num_domains, base_channels)
        self.target_embedding = nn.Embedding(num_domains, base_channels)

    def forward(self, x: torch.Tensor, source_domain: torch.Tensor,
                target_domain: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        h = x
        for index, block in enumerate(self.blocks):
            h = block(h)
            if index == 0:
                conditioning = (_embed(self.source_embedding, source_domain)
                                + _embed(self.target_embedding, target_domain))
                h = h + conditioning.unsqueeze(-1).unsqueeze(-1)
            features.append(h)
        return self.logits(h), features


class MultiScalePatchDiscriminator(nn.Module):
    """The same discriminator at several resolutions, on average-pooled copies.

    One scale sees only one band of frequencies. Two or three scales let the critic object
    to both fine texture and larger-scale structure, which is what keeps the refinement
    from trading one for the other.
    """

    def __init__(self, in_channels: int, scales: int = 2, base_channels: int = 64,
                 layers: int = 3, num_domains: int = 5) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PatchDiscriminator(in_channels, base_channels, layers, num_domains)
            for _ in range(scales))

    def forward(self, x: torch.Tensor, source_domain: torch.Tensor,
                target_domain: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        logits, features = [], []
        current = x
        for index, discriminator in enumerate(self.discriminators):
            if index:
                current = F.avg_pool2d(current, 2)
            scale_logits, scale_features = discriminator(current, source_domain, target_domain)
            logits.append(scale_logits)
            features.append(scale_features)
        return logits, features


def hinge_discriminator_loss(real_logits: list[torch.Tensor],
                             fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """E[relu(1 - D(real))] + E[relu(1 + D(fake))], summed over scales.

    Hinge rather than BCE: it stops rewarding the critic once a sample is on the correct
    side of the margin, so a discriminator that is winning stops sharpening and the
    generator keeps receiving usable gradient. With an adversarial weight of 1e-4 the
    generator cannot afford a critic that saturates.
    """
    total = real_logits[0].new_zeros(())
    for real, fake in zip(real_logits, fake_logits, strict=True):
        total = total + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return total / len(real_logits)


def hinge_generator_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """-E[D(fake)], summed over scales. No margin on the generator side, by convention."""
    total = fake_logits[0].new_zeros(())
    for fake in fake_logits:
        total = total - fake.mean()
    return total / len(fake_logits)


def feature_matching_loss(real_features: list[list[torch.Tensor]],
                          fake_features: list[list[torch.Tensor]]) -> torch.Tensor:
    """L1 between the critic's intermediate activations on real and generated samples.

    This is what actually stabilises the refinement. The adversarial term alone is a single
    scalar per patch and is happy with any sharp texture; matching the critic's own features
    asks for the *same* texture statistics, and it carries far more signal per step. Which
    is why the paper weights it 10 against the adversarial term's 1e-4 -- a ratio of 1e5.
    Real features are detached: they are a target, not something to optimise.
    """
    total = real_features[0][0].new_zeros(())
    count = 0
    for real_scale, fake_scale in zip(real_features, fake_features, strict=True):
        for real, fake in zip(real_scale, fake_scale, strict=True):
            total = total + F.l1_loss(fake, real.detach())
            count += 1
    return total / max(count, 1)

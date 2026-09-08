"""Conditional tubelet translator: LeJEPA-pretrained ViT-B encoder + conditioned UNETR decoder.

Wraps `ConditionalTubeletTranslator` from the lejepa_pretraining checkout so it runs under
this pipeline's trainer. The upstream model already carries the two pieces the 2D
conditional U-Net lacks -- per-decoder-stage zero-init FiLM, and a zero-init residual head
in logit space -- so nothing about the architecture is re-implemented here.

Three conversions are needed at the boundary:

* the trainer hands over ``(source, target_domain, source_domain)``; the translator wants
  ``(source, source_field, target_field, modality)``,
* domains arrive as joint indices ``5 * modality + field`` (mrixfields.data.utils), and
* ``task3_paired_volume`` emits [-1, 1] tensors while the translator asserts [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component

REPO_ROOT = Path(__file__).resolve().parents[3]  # analysis repo root
LEJEPA_ROOT = REPO_ROOT / "lejepa_pretraining" / "repo"
DEFAULT_ENCODER = REPO_ROOT / "lejepa_pretraining" / "artifacts" / "encoder_step_00040000.pt"

if str(LEJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(LEJEPA_ROOT))

from mrixfields_downstream.tubelet_model import (  # noqa: E402
    build_conditional_tubelet_translator,
)

NUM_FIELDS = 5


class ConditionalTubeletAdapter(nn.Module):
    """Give the tubelet translator this pipeline's ``(image, target, source)`` signature."""

    def __init__(self, translator: nn.Module) -> None:
        super().__init__()
        self.translator = translator

    @property
    def encoder(self) -> nn.Module:
        return self.translator.encoder

    def forward(
        self,
        image: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The LeJEPA encoder was pretrained on 16 consecutive AXIAL slices
        # (mrixfields_lejepa/tubelet_data.py read_slab -> [1,16,364,436]), i.e. axis 2 of
        # this dataset's (364,436,364) canonical volumes. task3_paired_volume crops in
        # array order, so an axial crop arrives as [B,1,H,W,16] and has to be rotated into
        # the encoder's [B,1,16,H,W]. Cropping 16 along axis 0 instead would feed sagittal
        # slabs to an encoder that never saw one.
        permuted = image.shape[2] != 16 and image.shape[-1] == 16
        if permuted:
            image = image.permute(0, 1, 4, 2, 3).contiguous()
        source_domain = target_domain if source_domain is None else source_domain
        source_modality, source_field = torch.div(
            source_domain, NUM_FIELDS, rounding_mode="floor"
        ), source_domain % NUM_FIELDS
        target_field = target_domain % NUM_FIELDS
        # Source and target always share a modality in Task 3, so either index will do.
        prediction = self.translator(
            image.add(1).div(2).clamp(0.0, 1.0),
            source_field.long(),
            target_field.long(),
            source_modality.long(),
        )
        prediction = prediction.mul(2).sub(1)
        return prediction.permute(0, 1, 3, 4, 2).contiguous() if permuted else prediction


@register_component("task3_conditional_tubelet", category="model")
class ConditionalTubeletFactory(ModelFactory[ConditionalTubeletAdapter]):
    def build(self) -> ConditionalTubeletAdapter:
        translator = build_conditional_tubelet_translator(
            gradient_checkpointing=bool(self.params.get("gradient_checkpointing", False))
        )
        checkpoint = self.params.get("encoder_checkpoint", str(DEFAULT_ENCODER))
        if checkpoint:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = REPO_ROOT / path
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("schema_version") != 2:
                raise ValueError(f"{path}: expected a schema-v2 tubelet encoder export")
            translator.encoder.load_state_dict(payload["state_dict"], strict=True)
            print(
                f"Loaded tubelet encoder from {path.name} "
                f"(global_step={payload.get('global_step')})",
                flush=True,
            )
        elif not bool(self.params.get("allow_random_encoder", False)):
            raise ValueError(
                "encoder_checkpoint is empty; set allow_random_encoder=true for the "
                "random-initialisation control arm"
            )
        else:
            print("Tubelet encoder left at random initialisation (control arm)", flush=True)
        return ConditionalTubeletAdapter(translator)

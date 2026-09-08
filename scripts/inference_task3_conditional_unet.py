#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from components.models.conditional_unet import (  # noqa: E402
    ConditionalUNet,
    unet_from_state_dict,
)
from mrixfields.data.transforms import CenterCropOrPad  # noqa: E402
from mrixfields.data.utils import get_joint_domain, load_nifti, save_nifti  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402


CROP_SIZE = (368, 448)


def predict_slab(
    model: ConditionalUNet,
    volume: np.ndarray,
    source_domain: int,
    target_domain: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    start, stop = Z_CLIP_RANGE
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(volume.shape[:2])
    output = np.zeros_like(volume, dtype=np.float32)

    for batch_start in range(start, stop, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, stop))
        slices = np.stack([crop(volume[:, :, index]) for index in indices])
        images = torch.from_numpy(slices).unsqueeze(1).to(device=device, dtype=torch.float32)
        images = images.mul(2).sub(1)
        source = torch.full((len(slices),), source_domain, device=device, dtype=torch.long)
        target = torch.full((len(slices),), target_domain, device=device, dtype=torch.long)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            predictions = model(images, target, source)
        predictions = predictions.float().cpu().numpy()[:, 0]
        for index, prediction in zip(indices, predictions, strict=True):
            output[:, :, index] = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    # Read the architecture off the checkpoint rather than hardcoding it: the
    # widths follow whatever config trained it, and the residual head and
    # per-scale FiLM arms move key names, so a fixed constructor silently
    # excludes those checkpoints.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = unet_from_state_dict(checkpoint.get("model", checkpoint))
    model.to(device).eval()
    source_domain = get_joint_domain(args.modality, args.source)
    target_domain = get_joint_domain(args.modality, args.target)
    inputs = sorted(args.input_dir.glob("*.nii.gz"))
    if not inputs:
        raise FileNotFoundError(f"No NIfTI inputs in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        original = nib.load(str(path))
        original_orientation = nib.io_orientation(original.affine)
        volume, canonical_affine = load_nifti(path)
        prediction = predict_slab(
            model, volume, source_domain, target_domain, device, args.batch_size
        )
        start, stop = Z_CLIP_RANGE
        prediction[:, :, start:stop] *= volume[:, :, start:stop] > 1e-6
        canonical_orientation = nib.io_orientation(canonical_affine)
        transform = nib.orientations.ornt_transform(canonical_orientation, original_orientation)
        prediction = nib.orientations.apply_orientation(prediction, transform)
        save_nifti(prediction, original.affine, args.output_dir / path.name, header=original.header)
        print(f"Saved {args.output_dir / path.name}", flush=True)


if __name__ == "__main__":
    main()

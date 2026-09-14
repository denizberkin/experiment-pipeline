"""Paired loader for conditional flow matching: both endpoints, all three contrasts.

Separate module from ``components/data/task3.py`` rather than a flag on it, for the same
reason Task3MultiContrastDataModule is separate: the sample shape differs (a three-channel
target instead of one), and threading that through the existing six-deep pair loop would
put every existing checkpoint's loader one edit away from changing.

The conditioning domain here is the **field strength**, not the joint (modality, field)
index the other modules emit. With all three contrasts predicted in a single pass there is
no modality left to condition on -- it is the channel axis -- so the embedding tables are
five rows, not fifteen. A config for this module must therefore set ``num_domains = 5``.

Geometric augmentation is applied jointly to source and target. That is not a style choice:
the pair is spatially registered and the whole task is the intensity mapping between two
aligned volumes, so an augmentation that moves one endpoint and not the other destroys the
supervision. Intensity augmentation is deliberately absent for the same reason -- see
section 38 -- and belongs only in the degradation bridge, where the two endpoints differ by
construction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component
from mrixfields.audit import reinit_for_worker
from mrixfields.data.cached_dataset import CachedFlowDataset
from mrixfields.data.utils import FIELD_STRENGTHS, MODALITIES, get_joint_domain
from mrixfields.env import get_preprocessed_dir


class _FlowPairDataset(Dataset):
    """Tags a directed pair with its field indices and applies joint augmentation."""

    def __init__(self, dataset: Dataset, source_domain, target_domain,
                 horizontal_flip: float = 0.0, vertical_flip: float = 0.0,
                 max_rotation: float = 0.0) -> None:
        self.dataset = dataset
        # Either a scalar field index or, with domain_mode="joint", the three joint
        # (modality, field) indices -- one per contrast channel, in MODALITIES order.
        # Stored as a tensor, not a list: default collate turns a list of ints into a list
        # of B-length tensors rather than a [B, 3] tensor, and the trainer would then be
        # handed something with no .to().
        self.source_domain = (torch.as_tensor(source_domain, dtype=torch.long)
                              if isinstance(source_domain, (list, tuple)) else int(source_domain))
        self.target_domain = (torch.as_tensor(target_domain, dtype=torch.long)
                              if isinstance(target_domain, (list, tuple)) else int(target_domain))
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.max_rotation = float(max_rotation)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        source, target = sample["source"], sample["target"]

        if self.horizontal_flip and torch.rand(()) < self.horizontal_flip:
            source, target = source.flip(-1), target.flip(-1)
        # Off by default and it should usually stay off: a left-right flip maps a brain onto
        # a plausible brain, an up-down flip does not, and the test set is not upside down.
        if self.vertical_flip and torch.rand(()) < self.vertical_flip:
            source, target = source.flip(-2), target.flip(-2)
        if self.max_rotation:
            angle = float(torch.empty(()).uniform_(-self.max_rotation, self.max_rotation))
            source, target = _rotate(source, angle), _rotate(target, angle)

        sample["source"], sample["target"] = source, target
        sample["source_domain"] = self.source_domain
        sample["target_domain"] = self.target_domain
        return sample


def _rotate(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate a [C, H, W] slice about its centre, filling with -1 (air in this convention).

    Zero-fill would be wrong: the network's input range is [-1, 1] and air is -1, so a
    zero border reads as mid-intensity tissue at the corners.
    """
    import torch.nn.functional as F

    radians = torch.tensor(degrees * torch.pi / 180.0)
    cos, sin = torch.cos(radians), torch.sin(radians)
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=image.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, [1, *image.shape], align_corners=False)
    shifted = F.grid_sample(image.unsqueeze(0) + 1.0, grid, align_corners=False,
                            padding_mode="zeros")
    return shifted.squeeze(0) - 1.0


@register_component("task3_flow", category="data")
class Task3FlowDataModule(DataModule[dict[str, DataLoader]]):
    def setup(self) -> dict[str, DataLoader]:
        modalities = tuple(self.params.get("modalities", MODALITIES))
        fields = list(self.params.get("field_strengths", FIELD_STRENGTHS))
        crop_size = tuple(self.params.get("crop_size", (368, 448)))
        batch_size = int(self.params.get("batch_size", 8))
        num_workers = int(self.params.get("num_workers", 4))
        seed = int(self.params.get("seed", 0))
        split = str(self.params.get("prospective_split", "pro_train"))
        preprocessed_dir = self.params.get("preprocessed_dir") or get_preprocessed_dir()
        if not preprocessed_dir:
            raise ValueError("task3_flow requires PREPROCESSED_DIR or preprocessed_dir")

        # "joint": s and tau carry modality *and* field -- 15 domains, one index per
        # contrast channel, summed in the model. "field": the paper's 5 field indices.
        domain_mode = str(self.params.get("domain_mode", "joint"))
        if domain_mode not in {"joint", "field"}:
            raise ValueError(f"domain_mode must be 'joint' or 'field', got {domain_mode!r}")

        horizontal_flip = float(self.params.get("horizontal_flip", 0.5))
        vertical_flip = float(self.params.get("vertical_flip", 0.0))
        max_rotation = float(self.params.get("max_rotation", 0.0))
        for name, value in (("horizontal_flip", horizontal_flip), ("vertical_flip", vertical_flip)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        datasets: list[Dataset] = []
        for source in fields:
            for target in fields:
                if source == target:
                    continue
                try:
                    pair = CachedFlowDataset(preprocessed_dir, split, source, target,
                                             modalities=modalities, crop_size=crop_size)
                except (FileNotFoundError, ValueError):
                    # A missing pair is skipped, not fatal: the retrospective split has one
                    # field per subject, so most pairs are legitimately empty there.
                    continue
                if domain_mode == "joint":
                    source_domain = [get_joint_domain(m, source) for m in modalities]
                    target_domain = [get_joint_domain(m, target) for m in modalities]
                else:
                    source_domain, target_domain = fields.index(source), fields.index(target)
                datasets.append(_FlowPairDataset(
                    pair, source_domain, target_domain,
                    horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                    max_rotation=max_rotation,
                ))
        if not datasets:
            raise FileNotFoundError("No flow-matching training pairs were found")

        print(f"task3_flow: domain_mode={domain_mode} "
              f"({'15 joint (modality, field)' if domain_mode == 'joint' else '5 field'} "
              f"domains), {len(datasets)} directed pairs, "
              f"{sum(len(d) for d in datasets)} slices, "
              f"flip={horizontal_flip} rotation=+-{max_rotation} deg", flush=True)
        return {"train": DataLoader(
            ConcatDataset(datasets), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0, worker_init_fn=reinit_for_worker,
            generator=torch.Generator().manual_seed(seed),
        )}

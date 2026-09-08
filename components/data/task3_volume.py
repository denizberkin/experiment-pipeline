from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component
from mrixfields.data.utils import (
    ABBR_TO_SPLIT,
    FIELD_STRENGTHS,
    MODALITIES,
    extract_subject_id,
    get_joint_domain,
    list_nifti_files,
    load_nifti,
)
from mrixfields.env import get_data_dir, get_preprocessed_dir


@dataclass(frozen=True)
class _Pair:
    source: Path
    target: Path
    source_domain: int
    target_domain: int


class _PairedVolumeDataset(Dataset):
    """Random 3D crops from paired volumes of the same subject and modality.

    Length is a configured sample count rather than a dataset size: crops are
    drawn at random, so an "epoch" is a choice, not a property of the data.
    """

    def __init__(
        self,
        pairs: list[_Pair],
        crop_size: tuple[int, int, int],
        samples_per_epoch: int,
        flip_probability: float,
        foreground_threshold: float,
        foreground_attempts: int,
        seed: int,
        axial_first: bool = False,
    ) -> None:
        self.pairs = pairs
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.axial_first = axial_first
        self.flip_probability = flip_probability
        self.foreground_threshold = foreground_threshold
        self.foreground_attempts = foreground_attempts
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _origin(self, shape: tuple[int, ...], volume: np.memmap, rng: random.Random) -> tuple[int, ...]:
        """Pick a crop origin, preferring crops that contain some anatomy.

        A 0.5 mm volume is mostly air, so uniform sampling wastes a large share
        of steps on empty cubes. Attempts are capped so a mostly-background
        volume cannot spin forever.
        """
        limits = [max(dimension - size, 0) for dimension, size in zip(shape, self.crop_size, strict=True)]
        origin = tuple(rng.randint(0, limit) for limit in limits)
        for _ in range(self.foreground_attempts):
            window = tuple(slice(start, start + size) for start, size in zip(origin, self.crop_size, strict=True))
            if float(volume[window].mean()) >= self.foreground_threshold:
                return origin
            origin = tuple(rng.randint(0, limit) for limit in limits)
        return origin

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random((self.seed, index, torch.initial_seed()).__hash__())
        pair = self.pairs[rng.randrange(len(self.pairs))]
        source = np.load(pair.source, mmap_mode="r")
        target = np.load(pair.target, mmap_mode="r")
        if self.axial_first:
            # Volumes are cached (x, y, z); axis 2 is the axial index. Models that expect
            # [B, C, D, H, W] -- and the LPIPS term, which folds axis 2 into the batch --
            # need the axial axis first, so a crop_size of (16, 224, 224) means 16 axial
            # slices rather than 16 sagittal ones. A transpose of a memmap is a view.
            source = source.transpose(2, 0, 1)
            target = target.transpose(2, 0, 1)

        origin = self._origin(source.shape, source, rng)
        window = tuple(slice(start, start + size) for start, size in zip(origin, self.crop_size, strict=True))
        source_crop = np.ascontiguousarray(source[window], dtype=np.float32)
        target_crop = np.ascontiguousarray(target[window], dtype=np.float32)

        # Pad if a volume is smaller than the crop on any axis.
        pad = [(0, size - actual) for size, actual in zip(self.crop_size, source_crop.shape, strict=True)]
        if any(after for _, after in pad):
            source_crop = np.pad(source_crop, pad)
            target_crop = np.pad(target_crop, pad)

        for axis in range(3):
            if rng.random() < self.flip_probability:
                source_crop = np.flip(source_crop, axis=axis)
                target_crop = np.flip(target_crop, axis=axis)
        source_crop = np.ascontiguousarray(source_crop)
        target_crop = np.ascontiguousarray(target_crop)

        # Volumes ship in [0, 1]; the models are Tanh-bounded, as in the 2D path.
        return {
            "source": torch.from_numpy(source_crop).unsqueeze(0).mul(2).sub(1),
            "target": torch.from_numpy(target_crop).unsqueeze(0).mul(2).sub(1),
            "source_domain": pair.source_domain,
            "target_domain": pair.target_domain,
        }


@register_component("task3_paired_volume", category="data")
class Task3VolumeDataModule(DataModule[dict[str, DataLoader]]):
    """Paired 3D volumes for the volumetric Task 3 models.

    NIfTI is gzipped, and nibabel cannot slice into a compressed file without
    inflating all of it, so every crop would cost a full 190 MB decompression.
    Volumes are therefore cached once as uncompressed .npy and read through a
    memmap, which the page cache shares across dataloader workers.
    """

    def setup(self) -> dict[str, DataLoader]:
        modalities = list(self.params.get("modalities", MODALITIES))
        fields = list(self.params.get("field_strengths", FIELD_STRENGTHS))
        crop_size = tuple(int(value) for value in self.params.get("crop_size", (96, 96, 96)))
        # Swin UNETR needs 32; the tubelet ViT needs depth 16 exactly and 16 in plane.
        # Keep 32 as the default so existing configs are unaffected.
        divisor = int(self.params.get("crop_divisor", 32))
        if any(size % divisor for size in crop_size):
            raise ValueError(f"crop_size {crop_size} must be divisible by {divisor}")

        data_dir = Path(self.params.get("data_dir") or get_data_dir())
        cache_dir = Path(self.params.get("cache_dir") or (Path(get_preprocessed_dir()) / "volumes_3d"))
        split = ABBR_TO_SPLIT[str(self.params.get("prospective_split", "pro_train"))]
        holdout = {str(subject) for subject in self.params.get("holdout_subjects", [])}
        axial_first = bool(self.params.get("axial_first", False))
        seed = int(self.params.get("seed", 0))

        cached = self._cache_volumes(data_dir, cache_dir, split, modalities, fields)
        pairs = self._pairs(cached, modalities, fields, holdout, exclude=True)
        if not pairs:
            raise FileNotFoundError(f"No paired volumes under {data_dir / split}")

        loaders = {
            "train": self._loader(
                _PairedVolumeDataset(
                    pairs,
                    crop_size,
                    int(self.params.get("samples_per_epoch", 400)),
                    float(self.params.get("flip_probability", 0.5)),
                    float(self.params.get("foreground_threshold", 0.02)),
                    int(self.params.get("foreground_attempts", 8)),
                    seed,
                    axial_first=axial_first,
                ),
                batch_size=int(self.params.get("batch_size", 2)),
                num_workers=int(self.params.get("num_workers", 4)),
                seed=seed,
            )
        }

        # A held-out subject is the only way to see overfitting locally: the
        # challenge validation split has no targets.
        if holdout:
            validation = self._pairs(cached, modalities, fields, holdout, exclude=False)
            if validation:
                loaders["validation"] = self._loader(
                    _PairedVolumeDataset(
                        validation,
                        crop_size,
                        int(self.params.get("validation_samples", 100)),
                        0.0,
                        float(self.params.get("foreground_threshold", 0.02)),
                        int(self.params.get("foreground_attempts", 8)),
                        seed + 1,
                        axial_first=axial_first,
                    ),
                    batch_size=int(self.params.get("batch_size", 2)),
                    num_workers=int(self.params.get("num_workers", 4)),
                    seed=seed + 1,
                )
        return loaders

    @staticmethod
    def _cache_volumes(
        data_dir: Path, cache_dir: Path, split: str, modalities: list[str], fields: list[str]
    ) -> dict[tuple[str, str, str], Path]:
        cached: dict[tuple[str, str, str], Path] = {}
        for modality in modalities:
            for field in fields:
                for path in list_nifti_files(data_dir, split, modality, field):
                    subject = extract_subject_id(path.name)
                    # removesuffix, not split("."): the field strength itself
                    # contains a dot, so splitting truncates 0.1T and 1.5T
                    # filenames to a stem every subject shares.
                    destination = (
                        cache_dir / split / modality / field / f"{path.name.removesuffix('.nii.gz')}.npy"
                    )
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        volume, _ = load_nifti(path)
                        # The temporary has to end in .npy too - np.save appends
                        # the extension itself when it is missing, and would
                        # write beside the path this then renames.
                        temporary = destination.with_name(f"{destination.stem}.tmp.npy")
                        np.save(temporary, volume.astype(np.float32))
                        temporary.replace(destination)
                        print(f"cached {destination.relative_to(cache_dir)}", flush=True)
                    cached[(modality, field, subject)] = destination
        return cached

    @staticmethod
    def _pairs(
        cached: dict[tuple[str, str, str], Path],
        modalities: list[str],
        fields: list[str],
        holdout: set[str],
        exclude: bool,
    ) -> list[_Pair]:
        subjects = sorted({subject for _, _, subject in cached})
        pairs: list[_Pair] = []
        for subject in subjects:
            held = any(subject.endswith(item) for item in holdout)
            if held == exclude:
                continue
            for modality in modalities:
                for source in fields:
                    for target in fields:
                        if source == target:
                            continue
                        source_path = cached.get((modality, source, subject))
                        target_path = cached.get((modality, target, subject))
                        if source_path is None or target_path is None:
                            continue
                        pairs.append(
                            _Pair(
                                source_path,
                                target_path,
                                get_joint_domain(modality, source),
                                get_joint_domain(modality, target),
                            )
                        )
        return pairs

    @staticmethod
    def _loader(dataset: Dataset, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # the dataset already samples pairs and crops at random
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component
from mrixfields.data.cached_dataset import CachedPairedDataset, CachedUnpairedDataset
from mrixfields.data.dataset import PairedMRIDataset, UnpairedMRIDataset
from mrixfields.data.utils import ABBR_TO_SPLIT, FIELD_STRENGTHS, MODALITIES, get_joint_domain
from mrixfields.env import get_data_dir, get_preprocessed_dir


class _DomainDataset(Dataset):
    def __init__(self, dataset: Dataset, domain: int) -> None:
        self.dataset = dataset
        self.domain = domain

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["domain"] = self.domain
        return sample


class _PairedDomainDataset(Dataset):
    def __init__(
        self, dataset: Dataset, source_domain: int, target_domain: int, horizontal_flip: float
    ) -> None:
        self.dataset = dataset
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        if torch.rand(()) < self.horizontal_flip:
            sample["source"] = sample["source"].flip(-1)
            sample["target"] = sample["target"].flip(-1)
        sample["source_domain"] = self.source_domain
        sample["target_domain"] = self.target_domain
        return sample


@register_component("task3_paired", category="data")
class Task3DataModule(DataModule[dict[str, DataLoader]]):
    def setup(self) -> dict[str, DataLoader]:
        modalities = list(self.params.get("modalities", MODALITIES))
        fields = list(self.params.get("field_strengths", FIELD_STRENGTHS))
        crop_size = tuple(self.params.get("crop_size", (512, 512)))
        batch_size = int(self.params.get("batch_size", 8))
        num_workers = int(self.params.get("num_workers", 4))
        seed = int(self.params.get("seed", 0))
        include_retro = bool(self.params.get("include_retro", False))
        use_preprocessed = bool(self.params.get("use_preprocessed", True))
        horizontal_flip = float(self.params.get("horizontal_flip", 0.0))
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be between 0 and 1")
        preprocessed_dir = (self.params.get("preprocessed_dir") or get_preprocessed_dir()) if use_preprocessed else None
        if use_preprocessed and not preprocessed_dir:
            raise ValueError("use_preprocessed=true requires PREPROCESSED_DIR or data.params.preprocessed_dir")
        if preprocessed_dir and not Path(preprocessed_dir).is_dir():
            raise FileNotFoundError(
                f"Preprocessed data not found: {preprocessed_dir}. "
                "Run scripts/preprocess.py extract-slices --splits retro_train pro_train"
            )
        data_dir = None if preprocessed_dir else (self.params.get("data_dir") or get_data_dir())

        paired = self._paired_datasets(
            modalities,
            fields,
            crop_size,
            data_dir,
            preprocessed_dir if use_preprocessed else None,
            horizontal_flip,
        )
        data = {
            "train": self._loader(
                ConcatDataset(paired),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                seed=seed,
            )
        }

        if include_retro:
            unpaired = self._unpaired_datasets(
                modalities, fields, crop_size, data_dir, preprocessed_dir if use_preprocessed else None
            )
            data["pretrain"] = self._loader(
                ConcatDataset(unpaired),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                seed=seed + 1,
            )
        return data

    def _paired_datasets(
        self,
        modalities: list[str],
        fields: list[str],
        crop_size: tuple[int, int],
        data_dir: str | None,
        preprocessed_dir: str | None,
        horizontal_flip: float,
    ) -> list[Dataset]:
        datasets: list[Dataset] = []
        found: set[tuple[str, str, str]] = set()
        split = str(self.params.get("prospective_split", "pro_train"))
        for modality in modalities:
            for source in fields:
                for target in fields:
                    if source == target:
                        continue
                    try:
                        dataset = (
                            CachedPairedDataset(
                                preprocessed_dir, split, modality, source, target, crop_size=crop_size
                            )
                            if preprocessed_dir
                            else PairedMRIDataset(
                                str(data_dir),
                                ABBR_TO_SPLIT.get(split, split),
                                modality,
                                source,
                                target,
                                crop_size=crop_size,
                            )
                        )
                    except (FileNotFoundError, ValueError):
                        continue
                    if len(dataset):
                        found.add((modality, source, target))
                        datasets.append(
                            _PairedDomainDataset(
                                dataset,
                                get_joint_domain(modality, source),
                                get_joint_domain(modality, target),
                                horizontal_flip,
                            )
                        )
        expected = {
            (modality, source, target)
            for modality in modalities
            for source in fields
            for target in fields
            if source != target
        }
        if self.params.get("require_complete", True) and found != expected:
            missing = [f"{modality}:{source}->{target}" for modality, source, target in sorted(expected - found)]
            raise FileNotFoundError(f"Missing prospective Task 3 pairs: {missing}")
        if not datasets:
            raise FileNotFoundError("No paired prospective Task 3 slices were found")
        return datasets

    def _unpaired_datasets(
        self,
        modalities: list[str],
        fields: list[str],
        crop_size: tuple[int, int],
        data_dir: str | None,
        preprocessed_dir: str | None,
    ) -> list[Dataset]:
        datasets: list[Dataset] = []
        found: set[tuple[str, str]] = set()
        split = str(self.params.get("retrospective_split", "retro_train"))
        for modality in modalities:
            for field in fields:
                try:
                    dataset = (
                        CachedUnpairedDataset(
                            preprocessed_dir, split, modality, field, crop_size=crop_size
                        )
                        if preprocessed_dir
                        else UnpairedMRIDataset(
                            str(data_dir),
                            ABBR_TO_SPLIT.get(split, split),
                            modality,
                            field,
                            crop_size=crop_size,
                        )
                    )
                except FileNotFoundError:
                    continue
                if len(dataset):
                    found.add((modality, field))
                    datasets.append(_DomainDataset(dataset, get_joint_domain(modality, field)))
        expected = {(modality, field) for modality in modalities for field in fields}
        if self.params.get("require_complete", True) and found != expected:
            missing = [f"{modality}:{field}" for modality, field in sorted(expected - found)]
            raise FileNotFoundError(f"Missing retrospective Task 3 domains: {missing}")
        if not datasets:
            raise FileNotFoundError("No unpaired retrospective Task 3 slices were found")
        return datasets

    @staticmethod
    def _loader(
        dataset: Dataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )

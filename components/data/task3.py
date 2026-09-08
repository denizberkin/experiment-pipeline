from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component
from mrixfields.data.cached_dataset import (
    CachedMultiContrastDataset,
    CachedPairedDataset,
    CachedUnpairedDataset,
)
from mrixfields.data.dataset import PairedMRIDataset, UnpairedMRIDataset
from mrixfields.data.utils import (
    ABBR_TO_SPLIT,
    FIELD_STRENGTHS,
    MODALITIES,
    extract_subject_id,
    get_joint_domain,
)
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
        holdout = {str(subject) for subject in self.params.get("holdout_subjects", [])}
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
            holdout,
            exclude=True,
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

        # A held-out subject is the only way to see overfitting locally: the
        # challenge validation split has no targets. Without it the trainer
        # falls back to early-stopping on the training loss.
        if holdout:
            validation = self._paired_datasets(
                modalities,
                fields,
                crop_size,
                data_dir,
                preprocessed_dir if use_preprocessed else None,
                0.0,  # no augmentation on the fold the checkpoint is chosen with
                holdout,
                exclude=False,
            )
            if validation:
                data["validation"] = self._loader(
                    self._subsample(
                        ConcatDataset(validation), int(self.params.get("validation_samples", 0))
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,  # a fixed order keeps the loss comparable across epochs
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
        holdout: set[str],
        exclude: bool,
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
                    if holdout:
                        dataset = self._subject_split(dataset, holdout, exclude)
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
        # Only the training split has to be complete; the held-out fold is
        # whatever the excluded subjects happen to cover, and an empty one just
        # means no validation loader.
        if exclude:
            if self.params.get("require_complete", True) and found != expected:
                missing = [
                    f"{modality}:{source}->{target}" for modality, source, target in sorted(expected - found)
                ]
                raise FileNotFoundError(f"Missing prospective Task 3 pairs: {missing}")
            if not datasets:
                raise FileNotFoundError("No paired prospective Task 3 slices were found")
        return datasets

    @staticmethod
    def _subject_split(dataset: Dataset, holdout: set[str], exclude: bool) -> Subset:
        """Keep the slices of the held-out subjects, or everything but them.

        The slice datasets index by file rather than by subject, so the split has
        to be applied after construction. Both layouts carry the subject in the
        source filename: cached slices as ``..._{subject}_s{NNN}.npz``, NIfTI
        volumes in the form ``extract_subject_id`` already parses.
        """
        entries = dataset.pairs if isinstance(dataset, CachedPairedDataset) else dataset.samples
        indices = []
        for index, entry in enumerate(entries):
            source = Path(entry[0])
            subject = (
                "_".join(source.stem.split("_")[-3:-1])
                if source.suffix == ".npz"
                else extract_subject_id(source.name)
            )
            # endswith, as in task3_volume: configs list bare ids such as "0009",
            # while the filenames carry them as "P_0009".
            if any(subject.endswith(item) for item in holdout) != exclude:
                indices.append(index)
        return Subset(dataset, indices)

    @staticmethod
    def _subsample(dataset: Dataset, samples: int) -> Dataset:
        """Evenly thin the held-out fold to at most ``samples`` slices.

        A whole subject is ~13k paired slices, half a training epoch, and it is
        re-scored every epoch. Striding keeps every transition represented; 0
        keeps the fold intact.
        """
        if samples <= 0 or len(dataset) <= samples:
            return dataset
        return Subset(dataset, range(0, len(dataset), -(-len(dataset) // samples)))

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


@register_component("task3_multicontrast", category="data")
class Task3MultiContrastDataModule(DataModule[dict[str, DataLoader]]):
    """Paired loader whose input is all three contrasts at the source field.

    Separate from Task3DataModule rather than a flag on it: that module is what every
    checkpoint in runs/ was trained through, and the six-deep nesting of its
    _paired_datasets loop is not worth threading a channel-count through. The domains it
    emits are identical -- source_domain = joint(target_modality, source_field),
    target_domain = joint(target_modality, target_field) -- so the embedding tables of a
    single-contrast checkpoint transfer unchanged.

    There is no pretrain loader here. The retrospective split has one field per subject and
    no guarantee the same subject appears in all three modalities, so a 3-channel unpaired
    loader is not well defined; a multi-contrast run instead adapts the first convolution of
    an already-pretrained single-channel checkpoint.
    """

    def setup(self) -> dict[str, DataLoader]:
        modalities = list(self.params.get("modalities", MODALITIES))
        fields = list(self.params.get("field_strengths", FIELD_STRENGTHS))
        crop_size = tuple(self.params.get("crop_size", (512, 512)))
        batch_size = int(self.params.get("batch_size", 8))
        num_workers = int(self.params.get("num_workers", 4))
        seed = int(self.params.get("seed", 0))
        horizontal_flip = float(self.params.get("horizontal_flip", 0.0))
        holdout = {str(subject) for subject in self.params.get("holdout_subjects", [])}
        split = str(self.params.get("prospective_split", "pro_train"))
        preprocessed_dir = self.params.get("preprocessed_dir") or get_preprocessed_dir()
        if not preprocessed_dir:
            raise ValueError("task3_multicontrast requires PREPROCESSED_DIR or preprocessed_dir")
        if not 0.0 <= horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be between 0 and 1")

        train = self._datasets(modalities, fields, crop_size, preprocessed_dir, split,
                               horizontal_flip, holdout, exclude=True)
        if not train:
            raise FileNotFoundError("No multi-contrast Task 3 training pairs were found")
        data = {
            "train": self._loader(ConcatDataset(train), batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True, seed=seed)
        }
        if holdout:
            validation = self._datasets(modalities, fields, crop_size, preprocessed_dir, split,
                                        0.0, holdout, exclude=False)
            if validation:
                data["validation"] = self._loader(
                    ConcatDataset(validation), batch_size=batch_size,
                    num_workers=num_workers, shuffle=False, seed=seed + 1,
                )
        return data

    def _datasets(self, modalities, fields, crop_size, preprocessed_dir, split,
                  horizontal_flip, holdout, exclude):
        datasets: list[Dataset] = []
        for modality in modalities:
            for source in fields:
                for target in fields:
                    if source == target:
                        continue
                    try:
                        dataset = CachedMultiContrastDataset(
                            preprocessed_dir, split, modality, source, target,
                            input_modalities=tuple(MODALITIES), crop_size=crop_size,
                        )
                    except (FileNotFoundError, ValueError):
                        continue
                    subset = self._filter_subjects(dataset, holdout, exclude)
                    if subset is None or not len(subset):
                        continue
                    datasets.append(
                        _PairedDomainDataset(
                            subset,
                            get_joint_domain(modality, source),
                            get_joint_domain(modality, target),
                            horizontal_flip,
                        )
                    )
        return datasets

    @staticmethod
    def _filter_subjects(dataset: "CachedMultiContrastDataset", holdout: set[str], exclude: bool):
        """Keep, or drop, the held-out subjects -- matched on the target file's name.

        Subset over filename-matched indices rather than a filtered dataset, for the same
        reason Task3DataModule does it: the sample list is built by file pairing, so an index
        is the only handle on a subject that does not require rebuilding the pairing.
        """
        if not holdout:
            return dataset if exclude else None
        keep = [
            i for i, (_, target_path) in enumerate(dataset.samples)
            if any(subject in target_path.name for subject in holdout) is not exclude
        ]
        return Subset(dataset, keep)

    @staticmethod
    def _loader(dataset: Dataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )

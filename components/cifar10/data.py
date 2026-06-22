from __future__ import annotations

from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
except ImportError as exc:
    raise ImportError("CIFAR-10 data requires torch and torchvision.") from exc

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.registry import register_component


@register_component("cifar10", category="data")
class Cifar10DataModule(DataModule):
    def setup(self) -> dict[str, Any]:
        data_dir = str(self.params.get("data_dir", "data"))
        batch_size = int(self.params.get("batch_size", 128))
        num_workers = int(self.params.get("num_workers", 2))
        validation_size = self.params.get("validation_size", 5000)
        download = bool(self.params.get("download", True))
        augment = bool(self.params.get("augment", True))
        seed = int(self.params.get("seed", 42))

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] if augment else []
        train_transform = transforms.Compose([*train_transforms, transforms.ToTensor(), normalize])
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_train = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_transform)
        full_validation = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
        test = datasets.CIFAR10(data_dir, train=False, download=download, transform=eval_transform)

        val_count = _validation_count(validation_size, len(full_train))
        train_count = len(full_train) - val_count
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=generator).tolist()
        train = Subset(full_train, indices[:train_count])
        validation = Subset(full_validation, indices[train_count:])

        return {
            "train": DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "validation": DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "test": DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "classes": full_train.classes,
        }


def _validation_count(value: Any, dataset_size: int) -> int:
    if isinstance(value, float):
        if not 0.0 < value < 1.0:
            raise ValueError("validation_size as a float must be between 0 and 1.")
        return max(1, int(dataset_size * value))
    count = int(value)
    if not 0 < count < dataset_size:
        raise ValueError("validation_size must be greater than 0 and smaller than the training dataset.")
    return count

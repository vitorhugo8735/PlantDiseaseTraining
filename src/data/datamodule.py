"""DataModule-like helpers for building dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.dataset_config import DatasetConfig

from .dataset import PlantDiseaseDataset
from .transforms import build_test_transforms, build_train_transforms, build_validation_transforms


Tensor = torch.Tensor


def _detection_collate_fn(batch: list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Collate detection samples into batched image tensors and target lists."""

    images = torch.stack([sample[0] for sample in batch], dim=0)
    boxes = [sample[1] for sample in batch]
    classes = [sample[2] for sample in batch]
    return images, boxes, classes


@dataclass(slots=True)
class PlantDiseaseDataModule:
    """Build train, validation and test data loaders from a dataset config."""

    config: DatasetConfig
    shuffle: bool = True
    dataset_path: Path = field(init=False)
    _train_dataset: PlantDiseaseDataset | None = field(init=False, default=None)
    _validation_dataset: PlantDiseaseDataset | None = field(init=False, default=None)
    _test_dataset: PlantDiseaseDataset | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.dataset_path = self.config.dataset_path

    def setup(self) -> None:
        """Instantiate split datasets lazily."""

        if self._train_dataset is None:
            self._train_dataset = PlantDiseaseDataset(
                dataset_path=self.dataset_path,
                split="train",
                transforms=build_train_transforms(self.config.image_size),
            )
        if self._validation_dataset is None:
            self._validation_dataset = PlantDiseaseDataset(
                dataset_path=self.dataset_path,
                split="valid",
                transforms=build_validation_transforms(self.config.image_size),
            )
        if self._test_dataset is None:
            self._test_dataset = PlantDiseaseDataset(
                dataset_path=self.dataset_path,
                split="test",
                transforms=build_test_transforms(self.config.image_size),
            )

    def train_dataloader(self) -> DataLoader:
        """Build the training data loader."""

        self.setup()
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=_detection_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation data loader."""

        self.setup()
        return DataLoader(
            dataset=self._validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=_detection_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Build the test data loader."""

        self.setup()
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=_detection_collate_fn,
        )
"""DataLoader orchestration and dataset reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.dataset_config import DatasetConfig
from data.dataset import PlantDiseaseDataset
from data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_validation_transforms,
)
from data.validation import validate_dataset_structure


@dataclass(slots=True, frozen=True)
class DatasetStatistics:
    """Aggregate dataset statistics for the PlantDisease dataset."""

    train_images: int
    train_labels: int
    valid_images: int
    valid_labels: int
    test_images: int
    test_labels: int
    num_classes: int


class PlantDiseaseDataModule:
    """Build train, validation, and test DataLoader instances."""

    def __init__(
        self,
        config: DatasetConfig | None = None,
        validate_splits: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.metadata = validate_dataset_structure(
            self.config, split_names=validate_splits
        )

    @property
    def train_loader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Return the training DataLoader."""

        return self._build_loader("train")

    @property
    def validation_loader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Return the validation DataLoader."""

        return self._build_loader("valid")

    @property
    def test_loader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Return the test DataLoader."""

        return self._build_loader("test")

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Compatibility alias for the train loader."""

        return self.train_loader

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Compatibility alias for the validation loader."""

        return self.validation_loader

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        """Compatibility alias for the test loader."""

        return self.test_loader

    def get_statistics(self) -> DatasetStatistics:
        """Return dataset statistics for all splits."""

        return DatasetStatistics(
            train_images=_count_images(
                self.config.train_path, self.config.image_extensions
            ),
            train_labels=_count_labels(
                self.config.train_path, self.config.label_extension
            ),
            valid_images=_count_images(
                self.config.valid_path, self.config.image_extensions
            ),
            valid_labels=_count_labels(
                self.config.valid_path, self.config.label_extension
            ),
            test_images=_count_images(
                self.config.test_path, self.config.image_extensions
            ),
            test_labels=_count_labels(
                self.config.test_path, self.config.label_extension
            ),
            num_classes=self.metadata.num_classes,
        )

    def build_report(self) -> str:
        """Build a formatted dataset summary without printing it."""

        statistics = self.get_statistics()
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"

        return "\n".join(
            [
                "==============================",
                "PlantDisease Dataset",
                "==============================",
                "",
                f"Train Images : {statistics.train_images}",
                f"Valid Images : {statistics.valid_images}",
                f"Test Images  : {statistics.test_images}",
                "",
                f"Classes      : {statistics.num_classes}",
                f"Batch Size   : {self.config.batch_size}",
                f"Workers      : {self.config.num_workers}",
                f"Device       : {device_name}",
            ]
        )

    def _build_loader(
        self, split_name: str
    ) -> DataLoader[tuple[torch.Tensor, dict[str, Any]]]:
        dataset = PlantDiseaseDataset(
            split_path=self.config.split_path(split_name),
            config=self.config,
            transform=self._resolve_transform(split_name),
            validate=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=split_name == self.config.train_split_name
            and self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
            and self.config.num_workers > 0,
            collate_fn=_collate_plant_disease_batch,
        )

    def _resolve_transform(self, split_name: str):
        if split_name == self.config.train_split_name:
            return get_train_transforms(self.config)
        if split_name == self.config.valid_split_name:
            return get_validation_transforms(self.config)
        return get_test_transforms(self.config)


def get_dataset_statistics(config: DatasetConfig | None = None) -> DatasetStatistics:
    """Convenience function to collect dataset statistics."""

    return PlantDiseaseDataModule(config=config).get_statistics()


def build_dataset_report(config: DatasetConfig | None = None) -> str:
    """Convenience function to build a dataset summary report."""

    return PlantDiseaseDataModule(config=config).build_report()


def _count_images(split_path: Path, image_extensions: tuple[str, ...]) -> int:
    images_dir = split_path / "images"
    if not images_dir.exists():
        return 0
    allowed_extensions = {extension.lower() for extension in image_extensions}
    return sum(
        1
        for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
    )


def _count_labels(split_path: Path, label_extension: str) -> int:
    labels_dir = split_path / "labels"
    if not labels_dir.exists():
        return 0
    return sum(
        1
        for file_path in labels_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == label_extension.lower()
    )


def _collate_plant_disease_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    return images, targets

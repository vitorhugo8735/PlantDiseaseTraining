"""Transform pipelines for the PlantDisease dataset."""

from __future__ import annotations

from torchvision import transforms as T

from configs.dataset_config import DatasetConfig


def get_train_transforms(config: DatasetConfig | None = None) -> T.Compose:
    """Return the train transform pipeline."""

    return _build_basic_transforms(config or DatasetConfig())


def get_validation_transforms(config: DatasetConfig | None = None) -> T.Compose:
    """Return the validation transform pipeline."""

    return _build_basic_transforms(config or DatasetConfig())


def get_test_transforms(config: DatasetConfig | None = None) -> T.Compose:
    """Return the test transform pipeline."""

    return _build_basic_transforms(config or DatasetConfig())


def _build_basic_transforms(config: DatasetConfig) -> T.Compose:
    return T.Compose(
        [
            T.Resize(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )

"""Configuration objects for the PlantDisease dataset layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    """Centralize dataset, loader, and normalization settings."""

    dataset_path: Path | str = Path("dataset")
    train_split_name: str = "train"
    valid_split_name: str = "valid"
    test_split_name: str = "test"
    images_dirname: str = "images"
    labels_dirname: str = "labels"
    data_yaml_filename: str = "data.yaml"
    batch_size: int = 32
    image_size: tuple[int, int] = (224, 224)
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle_train: bool = True
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    label_extension: str = ".txt"

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path).expanduser().resolve()

    @property
    def train_path(self) -> Path:
        """Return the train split path."""

        return self.dataset_path / self.train_split_name

    @property
    def valid_path(self) -> Path:
        """Return the validation split path."""

        return self.dataset_path / self.valid_split_name

    @property
    def test_path(self) -> Path:
        """Return the test split path."""

        return self.dataset_path / self.test_split_name

    @property
    def data_yaml_path(self) -> Path:
        """Return the dataset metadata file path."""

        return self.dataset_path / self.data_yaml_filename

    def split_path(self, split_name: str) -> Path:
        """Resolve a split name to its directory path."""

        normalized_split = split_name.strip().lower()
        if normalized_split == self.train_split_name:
            return self.train_path
        if normalized_split == self.valid_split_name:
            return self.valid_path
        if normalized_split == self.test_split_name:
            return self.test_path
        return self.dataset_path / normalized_split

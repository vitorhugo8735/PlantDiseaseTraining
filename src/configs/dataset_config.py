"""Dataset configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration values used by the data loading layer.

    Attributes:
        dataset_path: Root path that contains the dataset splits and `data.yaml`.
        image_size: Target square size used by the image transforms.
        batch_size: Batch size used by the data loaders.
        num_workers: Number of worker processes used by the data loaders.
        pin_memory: Whether data loaders should pin host memory.
    """

    dataset_path: Path
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
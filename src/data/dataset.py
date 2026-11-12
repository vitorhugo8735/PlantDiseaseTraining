"""Dataset implementation for PlantDiseaseTraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from configs.dataset_config import DatasetConfig
from data.validation import (
    DatasetMetadata,
    ParsedLabel,
    load_dataset_metadata,
    validate_split_structure,
    validate_label_file,
)


@dataclass(slots=True, frozen=True)
class PlantDiseaseSample:
    """Pair a single image with its label file."""

    image_path: Path
    label_path: Path


class PlantDiseaseDataset(Dataset[tuple[Tensor, dict[str, Any]]]):
    """Load PlantDisease images and their associated YOLO labels."""

    def __init__(
        self,
        split_path: str | Path,
        config: DatasetConfig | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
        validate: bool = True,
    ) -> None:
        self.config = config or DatasetConfig()
        self.split_path = Path(split_path).expanduser().resolve()
        self.transform = transform

        self.metadata = load_dataset_metadata(self.config)
        if validate:
            validate_split_structure(self.split_path, self.config, self.metadata.num_classes)
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self.transform(rgb_image) if self.transform is not None else _default_image_to_tensor(rgb_image)

        parsed_labels = validate_label_file(sample.label_path, self.metadata.num_classes)
        target = self._build_target(sample, parsed_labels)
        return image_tensor, target

    def _build_samples(self) -> list[PlantDiseaseSample]:
        images_dir = self.split_path / self.config.images_dirname
        labels_dir = self.split_path / self.config.labels_dirname

        image_files = sorted(
            file_path
            for file_path in images_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in {extension.lower() for extension in self.config.image_extensions}
        )

        samples: list[PlantDiseaseSample] = []
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}{self.config.label_extension}"
            samples.append(PlantDiseaseSample(image_path=image_path, label_path=label_path))

        return samples

    @staticmethod
    def _build_target(sample: PlantDiseaseSample, parsed_labels: list[ParsedLabel]) -> dict[str, Any]:
        class_ids = torch.tensor([parsed_label.class_id for parsed_label in parsed_labels], dtype=torch.long)
        primary_label = class_ids[0]

        boxes = [parsed_label.box for parsed_label in parsed_labels if parsed_label.box is not None]
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        return {
            "label": primary_label,
            "class_ids": class_ids,
            "boxes": boxes_tensor,
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
        }


def _default_image_to_tensor(image: Image.Image) -> Tensor:
    return to_tensor(image)

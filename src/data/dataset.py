"""Dataset implementation for YOLO-formatted plant disease samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .transforms import DetectionTransform, _yolo_to_xyxy
from .validation import ValidationIssue, list_image_files, load_dataset_metadata, validate_dataset_split


Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class DetectionSample:
    """Internal representation of a dataset sample."""

    image_path: Path
    label_path: Path


class PlantDiseaseDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset that loads images and YOLO annotations from a split directory."""

    def __init__(
        self,
        dataset_path: Path,
        split: str,
        transforms: DetectionTransform | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms

        split_root = self.dataset_path / split
        self.images_dir = split_root / "images"
        self.labels_dir = split_root / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        metadata = load_dataset_metadata(self.dataset_path)
        self.class_count = metadata.class_count
        self.class_names = metadata.class_names

        self.validation_issues = validate_dataset_split(
            self.images_dir,
            self.labels_dir,
            class_count=self.class_count,
        )
        if self.validation_issues:
            raise ValueError(self._format_validation_issues(self.validation_issues))

        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
            boxes, classes = self._load_annotations(sample.label_path)

            if self.transforms is not None:
                transformed_image, transformed_boxes, transformed_classes = self.transforms(
                    rgb_image,
                    boxes,
                    classes,
                )
                return transformed_image, transformed_boxes, transformed_classes

            image_tensor = T.ToTensor()(rgb_image)
            absolute_boxes = _yolo_to_xyxy(boxes, image_width=rgb_image.width, image_height=rgb_image.height)
            return image_tensor, absolute_boxes, classes

    def _build_samples(self) -> list[DetectionSample]:
        samples: list[DetectionSample] = []
        for image_path in list_image_files(self.images_dir):
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(DetectionSample(image_path=image_path, label_path=label_path))
        return samples

    def _load_annotations(self, label_path: Path) -> tuple[Tensor, Tensor]:
        boxes: list[list[float]] = []
        classes: list[int] = []

        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label format in {label_path} on line {line_number}.")

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as error:
                raise ValueError(f"Unable to parse YOLO annotation in {label_path} on line {line_number}.") from error

            boxes.append([x_center, y_center, width, height])
            classes.append(class_id)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            classes_tensor = torch.tensor(classes, dtype=torch.long)
            return boxes_tensor, classes_tensor

        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    @staticmethod
    def _format_validation_issues(issues: Sequence[ValidationIssue]) -> str:
        messages = ["Dataset validation failed:"]
        for issue in issues:
            messages.append(f"- {issue.path}: {issue.message}")
        return "\n".join(messages)
"""Validation helpers for YOLO-formatted dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Represents a validation problem discovered in the dataset."""

    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Metadata extracted from the dataset `data.yaml` file."""

    class_count: int | None
    class_names: tuple[str, ...]


def load_dataset_metadata(dataset_path: Path) -> DatasetMetadata:
    """Load lightweight metadata from the dataset `data.yaml` file.

    The parser intentionally supports only the fields needed by the data
    pipeline so the project does not depend on an additional YAML package.
    """

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return DatasetMetadata(class_count=None, class_names=())

    class_count: int | None = None
    class_names: tuple[str, ...] = ()

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nc:"):
            value = line.split(":", 1)[1].strip()
            try:
                class_count = int(value)
            except ValueError:
                class_count = None
            continue
        if line.startswith("names:"):
            names_value = line.split(":", 1)[1].strip()
            if names_value.startswith("[") and names_value.endswith("]"):
                raw_names = names_value[1:-1].split(",")
                parsed_names = [name.strip().strip("'\"") for name in raw_names if name.strip()]
                class_names = tuple(parsed_names)

    return DatasetMetadata(class_count=class_count, class_names=class_names)


def list_image_files(images_dir: Path) -> list[Path]:
    """Return all image files found in a split image directory."""

    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    """Build the expected label path for an image."""

    return labels_dir / f"{image_path.stem}.txt"


def validate_image_without_label(image_path: Path, labels_dir: Path) -> list[ValidationIssue]:
    """Validate that an image has a corresponding label file."""

    label_path = label_path_for_image(image_path, labels_dir)
    if not label_path.exists():
        return [ValidationIssue(path=image_path, message="Image without a matching label file.")]
    return []


def validate_label_without_image(label_path: Path, images_dir: Path) -> list[ValidationIssue]:
    """Validate that a label file has a corresponding image file."""

    stem = label_path.stem
    if any((images_dir / f"{stem}{suffix}").exists() for suffix in IMAGE_EXTENSIONS):
        return []
    return [ValidationIssue(path=label_path, message="Label without a matching image file.")]


def _read_label_lines(label_path: Path) -> list[str]:
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_empty_label(label_path: Path) -> list[ValidationIssue]:
    """Validate that a label file contains at least one annotation line."""

    try:
        lines = _read_label_lines(label_path)
    except OSError:
        return [ValidationIssue(path=label_path, message="Unable to read label file.")]

    if not lines:
        return [ValidationIssue(path=label_path, message="Label file is empty.")]
    return []


def validate_invalid_class(label_path: Path, class_count: int | None = None) -> list[ValidationIssue]:
    """Validate the class identifier in a YOLO label file."""

    issues: list[ValidationIssue] = []
    try:
        lines = _read_label_lines(label_path)
    except OSError:
        return [ValidationIssue(path=label_path, message="Unable to read label file.")]

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            issues.append(ValidationIssue(path=label_path, message=f"Invalid class id on line {line_number}."))
            continue
        if class_id < 0:
            issues.append(ValidationIssue(path=label_path, message=f"Negative class id on line {line_number}."))
            continue
        if class_count is not None and class_id >= class_count:
            issues.append(ValidationIssue(path=label_path, message=f"Class id out of range on line {line_number}."))
    return issues


def validate_invalid_yolo_coordinates(label_path: Path) -> list[ValidationIssue]:
    """Validate YOLO bounding box coordinates in a label file."""

    issues: list[ValidationIssue] = []
    try:
        lines = _read_label_lines(label_path)
    except OSError:
        return [ValidationIssue(path=label_path, message="Unable to read label file.")]

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(ValidationIssue(path=label_path, message=f"Invalid label format on line {line_number}."))
            continue

        try:
            _, x_center, y_center, width, height = (float(value) for value in parts)
        except ValueError:
            issues.append(ValidationIssue(path=label_path, message=f"Non-numeric YOLO coordinates on line {line_number}."))
            continue

        values = (x_center, y_center, width, height)
        if any(not math.isfinite(value) for value in values):
            issues.append(ValidationIssue(path=label_path, message=f"Non-finite YOLO coordinates on line {line_number}."))
            continue
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            issues.append(ValidationIssue(path=label_path, message=f"YOLO coordinates out of range on line {line_number}."))
            continue
        left = x_center - width / 2.0
        right = x_center + width / 2.0
        top = y_center - height / 2.0
        bottom = y_center + height / 2.0
        if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
            issues.append(ValidationIssue(path=label_path, message=f"YOLO box exceeds image bounds on line {line_number}."))

    return issues


def validate_corrupted_image(image_path: Path) -> list[ValidationIssue]:
    """Validate that an image file can be opened by Pillow."""

    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return [ValidationIssue(path=image_path, message="Corrupted or unreadable image file.")]
    return []


def validate_dataset_split(images_dir: Path, labels_dir: Path, class_count: int | None = None) -> list[ValidationIssue]:
    """Validate a dataset split and return all discovered issues."""

    issues: list[ValidationIssue] = []
    image_paths = list_image_files(images_dir)

    label_paths = sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt") if labels_dir.exists() else []

    for image_path in image_paths:
        issues.extend(validate_image_without_label(image_path, labels_dir))
        issues.extend(validate_corrupted_image(image_path))

    for label_path in label_paths:
        issues.extend(validate_label_without_image(label_path, images_dir))
        issues.extend(validate_empty_label(label_path))
        issues.extend(validate_invalid_class(label_path, class_count=class_count))
        issues.extend(validate_invalid_yolo_coordinates(label_path))

    return issues
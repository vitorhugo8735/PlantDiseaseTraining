"""Dataset validation utilities for the PlantDiseaseTraining project."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError

from configs.dataset_config import DatasetConfig


class DatasetValidationError(ValueError):
    """Base exception for all dataset validation failures."""


class DatasetPathNotFoundError(DatasetValidationError):
    """Raised when the dataset root or a split directory is missing."""


class MissingImagesDirectoryError(DatasetValidationError):
    """Raised when a split is missing its images directory."""


class MissingLabelsDirectoryError(DatasetValidationError):
    """Raised when a split is missing its labels directory."""


class EmptyDatasetError(DatasetValidationError):
    """Raised when a split or dataset has no usable files."""


class DataYamlNotFoundError(DatasetValidationError):
    """Raised when data.yaml is not available."""


class ClassDefinitionError(DatasetValidationError):
    """Raised when class definitions inside data.yaml are invalid."""


class MissingLabelError(DatasetValidationError):
    """Raised when an image does not have a matching label file."""


class MissingImageError(DatasetValidationError):
    """Raised when a label file does not have a matching image file."""


class CorruptedLabelError(DatasetValidationError):
    """Raised when a label file cannot be parsed as a valid YOLO annotation."""


class CorruptedImageError(DatasetValidationError):
    """Raised when an image file cannot be opened."""


@dataclass(slots=True, frozen=True)
class DatasetMetadata:
    """Metadata loaded from data.yaml."""

    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    data_yaml_path: Path
    dataset_path: Path

    @property
    def num_classes(self) -> int:
        """Return the total number of classes."""

        return len(self.class_names)


@dataclass(slots=True, frozen=True)
class ParsedLabel:
    """Single label entry parsed from a YOLO annotation line."""

    class_id: int
    box: tuple[float, float, float, float] | None


def load_dataset_metadata(config: DatasetConfig) -> DatasetMetadata:
    """Load class names and class indices from data.yaml."""

    if not config.dataset_path.exists():
        raise DatasetPathNotFoundError(
            f"Dataset path does not exist: {config.dataset_path}"
        )

    data_yaml_path = config.data_yaml_path
    if not data_yaml_path.exists():
        raise DataYamlNotFoundError(f"data.yaml not found: {data_yaml_path}")

    class_names = _parse_class_names(
        data_yaml_path.read_text(encoding="utf-8").splitlines()
    )
    if not class_names:
        raise ClassDefinitionError(f"No classes were defined in {data_yaml_path}")

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return DatasetMetadata(
        class_names=class_names,
        class_to_index=class_to_index,
        data_yaml_path=data_yaml_path,
        dataset_path=config.dataset_path,
    )


def validate_dataset_structure(
    config: DatasetConfig,
    metadata: DatasetMetadata | None = None,
    split_names: Sequence[str] | None = None,
) -> DatasetMetadata:
    """Validate the dataset root, split folders, and metadata file."""

    dataset_metadata = metadata or load_dataset_metadata(config)
    splits_to_validate = split_names or (
        config.train_split_name,
        config.valid_split_name,
        config.test_split_name,
    )
    for split_name in splits_to_validate:
        validate_split_structure(
            config.split_path(split_name), config, dataset_metadata.num_classes
        )
    return dataset_metadata


def validate_split_structure(
    split_path: Path, config: DatasetConfig, num_classes: int
) -> None:
    """Validate a single dataset split."""

    if not split_path.exists():
        raise DatasetPathNotFoundError(f"Split path does not exist: {split_path}")
    if not split_path.is_dir():
        raise DatasetPathNotFoundError(f"Split path is not a directory: {split_path}")

    images_dir = split_path / config.images_dirname
    labels_dir = split_path / config.labels_dirname

    if not images_dir.exists():
        raise MissingImagesDirectoryError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise MissingLabelsDirectoryError(f"Missing labels directory: {labels_dir}")

    image_files = _list_image_files(images_dir, config.image_extensions)
    label_files = _list_label_files(labels_dir, config.label_extension)

    if not image_files or not label_files:
        raise EmptyDatasetError(f"Split '{split_path.name}' is empty or incomplete")

    image_stems = {image_file.stem for image_file in image_files}
    label_stems = {label_file.stem for label_file in label_files}

    missing_label_stems = sorted(image_stems - label_stems)
    if missing_label_stems:
        raise MissingLabelError(
            f"Images without labels in '{split_path.name}': {', '.join(missing_label_stems)}"
        )

    missing_image_stems = sorted(label_stems - image_stems)
    if missing_image_stems:
        raise MissingImageError(
            f"Labels without images in '{split_path.name}': {', '.join(missing_image_stems)}"
        )

    for image_file in image_files:
        validate_image_file(image_file)

    for label_file in label_files:
        validate_label_file(label_file, num_classes)


def validate_image_file(image_path: Path) -> None:
    """Validate that an image file can be opened."""

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptedImageError(f"Unreadable image: {image_path}") from exc


def validate_label_file(label_path: Path, num_classes: int) -> list[ParsedLabel]:
    """Validate a YOLO label file and return its parsed annotations."""

    try:
        raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorruptedLabelError(f"Unable to read label file: {label_path}") from exc

    parsed_labels: list[ParsedLabel] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        tokens = stripped_line.split()
        if len(tokens) not in {1, 5}:
            raise CorruptedLabelError(
                f"Invalid annotation format in {label_path} at line {line_number}: expected 1 or 5 values"
            )

        class_id = _parse_class_id(tokens[0], label_path, line_number, num_classes)

        if len(tokens) == 1:
            parsed_labels.append(ParsedLabel(class_id=class_id, box=None))
            continue

        try:
            x_center, y_center, width, height = (float(token) for token in tokens[1:])
        except ValueError as exc:
            raise CorruptedLabelError(
                f"Invalid numeric value in {label_path} at line {line_number}"
            ) from exc

        if not all(
            0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
        ):
            raise CorruptedLabelError(
                f"YOLO coordinates out of range in {label_path} at line {line_number}"
            )
        if width <= 0.0 or height <= 0.0:
            raise CorruptedLabelError(
                f"YOLO box dimensions must be positive in {label_path} at line {line_number}"
            )

        parsed_labels.append(
            ParsedLabel(
                class_id=class_id,
                box=(x_center, y_center, width, height),
            )
        )

    if not parsed_labels:
        raise CorruptedLabelError(f"Label file is empty: {label_path}")

    return parsed_labels


def _parse_class_names(lines: Sequence[str]) -> tuple[str, ...]:
    names_started = False
    indexed_names: dict[int, str] = {}
    list_names: list[str] = []

    for raw_line in lines:
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        if stripped_line.startswith("names:"):
            names_started = True
            inline_value = stripped_line.removeprefix("names:").strip()
            if inline_value:
                parsed_inline = _parse_inline_names(inline_value)
                if parsed_inline is not None:
                    return parsed_inline
            continue

        if not names_started:
            continue

        if raw_line.startswith((" ", "\t")):
            if stripped_line.startswith("- "):
                list_names.append(
                    _clean_name_value(stripped_line.removeprefix("- ").strip())
                )
                continue

            if ":" not in stripped_line:
                continue

            index_text, class_name_text = stripped_line.split(":", 1)
            try:
                class_index = int(index_text.strip())
            except ValueError as exc:
                raise ClassDefinitionError(
                    f"Invalid class index in data.yaml: {index_text!r}"
                ) from exc

            class_name = _clean_name_value(class_name_text.strip())
            if not class_name:
                raise ClassDefinitionError(
                    f"Empty class name in data.yaml for index {class_index}"
                )
            indexed_names[class_index] = class_name
            continue

        break

    if indexed_names:
        expected_indexes = list(range(len(indexed_names)))
        actual_indexes = sorted(indexed_names)
        if actual_indexes != expected_indexes:
            raise ClassDefinitionError(
                "Class indexes in data.yaml must start at 0 and be contiguous"
            )
        return tuple(indexed_names[index] for index in expected_indexes)

    if list_names:
        return tuple(list_names)

    raise ClassDefinitionError("Unable to find a valid names section in data.yaml")


def _parse_inline_names(inline_value: str) -> tuple[str, ...] | None:
    try:
        parsed_value = literal_eval(inline_value)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed_value, dict):
        ordered_items = sorted(parsed_value.items(), key=lambda item: int(item[0]))
        return tuple(_clean_name_value(str(name)) for _, name in ordered_items)

    if isinstance(parsed_value, (list, tuple)):
        return tuple(_clean_name_value(str(name)) for name in parsed_value)

    return None


def _clean_name_value(raw_name: str) -> str:
    return raw_name.strip().strip('"').strip("'")


def _parse_class_id(
    token: str, label_path: Path, line_number: int, num_classes: int
) -> int:
    try:
        class_id = int(token)
    except ValueError as exc:
        raise CorruptedLabelError(
            f"Invalid class id in {label_path} at line {line_number}: {token!r}"
        ) from exc

    if class_id < 0 or class_id >= num_classes:
        raise ClassDefinitionError(
            f"Class id {class_id} in {label_path} at line {line_number} is outside the valid range"
        )
    return class_id


def _list_image_files(images_dir: Path, valid_extensions: Iterable[str]) -> list[Path]:
    allowed_extensions = {extension.lower() for extension in valid_extensions}
    return sorted(
        file_path
        for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
    )


def _list_label_files(labels_dir: Path, label_extension: str) -> list[Path]:
    return sorted(
        file_path
        for file_path in labels_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == label_extension.lower()
    )

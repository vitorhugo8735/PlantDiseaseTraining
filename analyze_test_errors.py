"""Analyze TEST-set prediction errors for PlantDiseaseTraining.

This script reuses the same inference path as the current evaluation entry
point, but adds per-image diagnostics, CSV exports, and error-folder copying.
It never retrains the model and never modifies the source dataset or checkpoint.
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs.dataset_config import DatasetConfig
from data.datamodule import PlantDiseaseDataModule
from data.validation import (
    ClassDefinitionError,
    DataYamlNotFoundError,
    DatasetPathNotFoundError,
    DatasetValidationError,
    EmptyDatasetError,
)
from models.model_config import ModelConfig
from models.model_factory import ModelFactory

DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset_redistributed"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
ERRORS_DIR = OUTPUT_DIR / "test_errors"
ERROR_SUMMARY_CSV = ERRORS_DIR / "error_summary.csv"
EXPECTED_CLASS_NAMES = ("mildew", "rose_P01", "rose_R02")
EXPECTED_TEST_IMAGES = 250
EXPECTED_NUM_CLASSES = 3
MODEL_NAME = "efficientnet_b0"


def main() -> None:
    dataset_config = DatasetConfig(
        dataset_path=DEFAULT_DATASET_PATH,
        batch_size=16,
        num_workers=0,
        pin_memory=_use_pin_memory(),
        persistent_workers=False,
    )

    _validate_environment(dataset_config)
    checkpoint_path = _validate_checkpoint()
    device = _resolve_device()

    datamodule = PlantDiseaseDataModule(dataset_config, validate_splits=("test",))
    _validate_test_metadata(datamodule)

    model = _load_model(datamodule, checkpoint_path, device)
    predictions = _collect_predictions(model, datamodule, device)

    _prepare_output_paths()
    _write_predictions_csv(predictions)
    _write_error_outputs(predictions)

    summary = _build_error_summary(predictions)
    _write_error_summary_csv(summary)

    _print_report(predictions, summary, device, checkpoint_path)


def _validate_environment(config: DatasetConfig) -> None:
    if not config.dataset_path.exists():
        raise SystemExit(f"TEST dataset not found: {config.dataset_path}")

    test_images_dir = config.test_path / config.images_dirname
    test_labels_dir = config.test_path / config.labels_dirname
    if not test_images_dir.exists():
        raise SystemExit(f"TEST images directory not found: {test_images_dir}")
    if not test_labels_dir.exists():
        raise SystemExit(f"TEST labels directory not found: {test_labels_dir}")

    image_count = sum(
        1
        for file_path in test_images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in config.image_extensions
    )
    label_count = sum(
        1
        for file_path in test_labels_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == config.label_extension
    )
    if image_count != EXPECTED_TEST_IMAGES or label_count != EXPECTED_TEST_IMAGES:
        raise SystemExit(
            "TEST split must contain exactly 250 images and 250 labels. "
            f"Found {image_count} images and {label_count} labels."
        )

    if not config.data_yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {config.data_yaml_path}")


def _validate_checkpoint() -> Path:
    checkpoint_path = DEFAULT_CHECKPOINT_PATH.expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_test_metadata(datamodule: PlantDiseaseDataModule) -> None:
    statistics = datamodule.get_statistics()
    if statistics.num_classes != EXPECTED_NUM_CLASSES:
        raise SystemExit(
            f"Expected {EXPECTED_NUM_CLASSES} classes, found {statistics.num_classes}."
        )
    if tuple(datamodule.metadata.class_names) != EXPECTED_CLASS_NAMES:
        raise SystemExit(
            "TEST dataset class order does not match the expected mapping: "
            f"expected {EXPECTED_CLASS_NAMES}, found {tuple(datamodule.metadata.class_names)}."
        )
    if (
        statistics.test_images != EXPECTED_TEST_IMAGES
        or statistics.test_labels != EXPECTED_TEST_IMAGES
    ):
        raise SystemExit(
            "TEST split must contain exactly 250 images and 250 labels. "
            f"Found {statistics.test_images} images and {statistics.test_labels} labels."
        )


def _load_model(
    datamodule: PlantDiseaseDataModule,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_config = ModelConfig(
        model_name=MODEL_NAME,
        num_classes=datamodule.metadata.num_classes,
        pretrained=False,
        device=device.type,
    )
    model = ModelFactory.create(model_config)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(state_dict, dict):
            raise SystemExit(
                "Checkpoint format is invalid: expected a state_dict saved by training."
            )
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(f"Checkpoint incompatible with {MODEL_NAME}: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Unable to load checkpoint: {exc}") from exc
    return model


def _collect_predictions(
    model: torch.nn.Module,
    datamodule: PlantDiseaseDataModule,
    device: torch.device,
) -> list[dict[str, Any]]:
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in datamodule.test_loader:
            images = images.to(device)
            labels = _extract_labels(targets).to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels)
            predictions = torch.argmax(probabilities, dim=1)

            _ = loss  # Kept to ensure the same inference path is exercised.

            for index in range(images.size(0)):
                image_path = Path(str(targets[index]["image_path"]))
                label_value = int(labels[index].item())
                prediction_value = int(predictions[index].item())
                probability_values = probabilities[index].detach().cpu().tolist()
                confidence = float(probability_values[prediction_value])
                correct = prediction_value == label_value

                rows.append(
                    {
                        "image": image_path.name,
                        "image_path": image_path,
                        "true_class": label_value,
                        "true_class_name": EXPECTED_CLASS_NAMES[label_value],
                        "predicted_class": prediction_value,
                        "predicted_class_name": EXPECTED_CLASS_NAMES[prediction_value],
                        "confidence": confidence,
                        "prob_class_0": float(probability_values[0]),
                        "prob_class_1": float(probability_values[1]),
                        "prob_class_2": float(probability_values[2]),
                        "correct": correct,
                    }
                )

    return rows


def _extract_labels(targets: list[dict[str, Any]]) -> torch.Tensor:
    labels: list[torch.Tensor] = []
    for target in targets:
        label_value = target["label"]
        if isinstance(label_value, torch.Tensor):
            labels.append(label_value.reshape(1).to(dtype=torch.long))
        else:
            labels.append(torch.tensor([int(label_value)], dtype=torch.long))
    return torch.cat(labels, dim=0)


def _prepare_output_paths() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if ERRORS_DIR.exists():
        shutil.rmtree(ERRORS_DIR)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_CSV.exists():
        PREDICTIONS_CSV.unlink()
    if ERROR_SUMMARY_CSV.exists():
        ERROR_SUMMARY_CSV.unlink()


def _write_predictions_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "true_class",
        "true_class_name",
        "predicted_class",
        "predicted_class_name",
        "confidence",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
        "correct",
    ]
    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "true_class": row["true_class"],
                    "true_class_name": row["true_class_name"],
                    "predicted_class": row["predicted_class"],
                    "predicted_class_name": row["predicted_class_name"],
                    "confidence": f"{row['confidence']:.10f}",
                    "prob_class_0": f"{row['prob_class_0']:.10f}",
                    "prob_class_1": f"{row['prob_class_1']:.10f}",
                    "prob_class_2": f"{row['prob_class_2']:.10f}",
                    "correct": str(bool(row["correct"])),
                }
            )


def _write_error_outputs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["correct"]:
            continue
        confusion_folder = f"{row['true_class_name']}_to_{row['predicted_class_name']}"
        destination_dir = ERRORS_DIR / confusion_folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        source_image = Path(row["image_path"])
        shutil.copy2(source_image, destination_dir / source_image.name)


def _build_error_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["correct"]:
            continue
        grouped[(row["true_class"], row["predicted_class"])].append(
            float(row["confidence"])
        )

    summary_rows: list[dict[str, Any]] = []
    for (true_class, predicted_class), confidences in sorted(grouped.items()):
        summary_rows.append(
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": len(confidences),
                "mean_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
            }
        )
    return summary_rows


def _write_error_summary_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "true_class",
        "predicted_class",
        "count",
        "mean_confidence",
        "min_confidence",
        "max_confidence",
    ]
    with ERROR_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "true_class": row["true_class"],
                    "predicted_class": row["predicted_class"],
                    "count": row["count"],
                    "mean_confidence": f"{row['mean_confidence']:.10f}",
                    "min_confidence": f"{row['min_confidence']:.10f}",
                    "max_confidence": f"{row['max_confidence']:.10f}",
                }
            )


def _print_report(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    total_images = len(rows)
    correct_rows = [row for row in rows if row["correct"]]
    error_rows = [row for row in rows if not row["correct"]]
    correct_count = len(correct_rows)
    error_count = len(error_rows)
    accuracy = (correct_count / total_images * 100.0) if total_images else 0.0
    confusion_counts = Counter(
        (row["true_class_name"], row["predicted_class_name"]) for row in error_rows
    )

    print("==============================")
    print("PlantDiseaseTraining - ERROR ANALYSIS")
    print("==============================")
    print(f"Device       : {'CUDA' if device.type == 'cuda' else 'CPU'}")
    print(f"Test Images  : {total_images}")
    print(f"Correct      : {correct_count}")
    print(f"Errors       : {error_count}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print()
    print("Confusion breakdown:")
    for true_name, predicted_name in _confusion_pairs():
        count = confusion_counts.get((true_name, predicted_name), 0)
        print(f"{true_name} -> {predicted_name:<12}: {count}")
    print()
    print("Top errors by confidence:")
    print()
    print("IMAGE | REAL | PREDICTED | CONFIDENCE")
    for row in sorted(error_rows, key=lambda item: item["confidence"], reverse=True):
        print(
            f"{row['image']} | {row['true_class_name']} | {row['predicted_class_name']} | {row['confidence'] * 100:.2f}%"
        )
    print()
    print("Files created:")
    print("- outputs/test_predictions.csv")
    print("- outputs/test_errors/error_summary.csv")
    print("- outputs/test_errors/")


def _confusion_pairs() -> list[tuple[str, str]]:
    return [
        ("mildew", "rose_P01"),
        ("mildew", "rose_R02"),
        ("rose_P01", "mildew"),
        ("rose_P01", "rose_R02"),
        ("rose_R02", "mildew"),
        ("rose_R02", "rose_P01"),
    ]


def _use_pin_memory() -> bool:
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()

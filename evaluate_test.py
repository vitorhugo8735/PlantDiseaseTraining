"""Standalone TEST evaluation entry point for PlantDiseaseTraining.

This script evaluates only the redistributed TEST split and never retrains or
rewrites the model checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from evaluation.confusion_matrix import ConfusionMatrixGenerator
from evaluation.metrics import MetricsCalculator
from models.model_config import ModelConfig
from models.model_factory import ModelFactory

DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset_redistributed"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
EXPECTED_TEST_IMAGES = 250
EXPECTED_TEST_LABELS = 250
EXPECTED_NUM_CLASSES = 3
MODEL_NAME = "efficientnet_b0"
CLASS_NAMES = ("mildew", "rose_P01", "rose_R02")


def main() -> None:
    dataset_config = DatasetConfig(
        dataset_path=DEFAULT_DATASET_PATH,
        batch_size=16,
        num_workers=0,
        pin_memory=_use_pin_memory(),
        persistent_workers=False,
    )

    try:
        datamodule = PlantDiseaseDataModule(dataset_config, validate_splits=("test",))
    except (
        DatasetValidationError,
        DatasetPathNotFoundError,
        DataYamlNotFoundError,
        EmptyDatasetError,
        ClassDefinitionError,
    ) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc

    statistics = datamodule.get_statistics()
    if statistics.num_classes != EXPECTED_NUM_CLASSES:
        raise SystemExit(
            f"Expected {EXPECTED_NUM_CLASSES} classes in TEST dataset, found {statistics.num_classes}."
        )
    if tuple(datamodule.metadata.class_names) != CLASS_NAMES:
        raise SystemExit(
            "TEST dataset class order does not match the expected mapping: "
            f"expected {CLASS_NAMES}, found {tuple(datamodule.metadata.class_names)}."
        )
    if (
        statistics.test_images != EXPECTED_TEST_IMAGES
        or statistics.test_labels != EXPECTED_TEST_LABELS
    ):
        raise SystemExit(
            "TEST split must contain exactly 250 images and 250 labels. "
            f"Found {statistics.test_images} images and {statistics.test_labels} labels."
        )

    checkpoint_path = DEFAULT_CHECKPOINT_PATH.expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    display_device = "CUDA" if device.type == "cuda" else "CPU"

    model_config = ModelConfig(
        model_name=MODEL_NAME,
        num_classes=statistics.num_classes,
        pretrained=False,
        device=device.type,
    )

    model = ModelFactory.create(model_config)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not isinstance(state_dict, dict):
            raise SystemExit(
                "Checkpoint format is invalid: expected a model state_dict saved by training."
            )
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(f"Checkpoint incompatible with {MODEL_NAME}: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Unable to load checkpoint: {exc}") from exc

    result = _evaluate_test_split(model, datamodule, device)

    print("==============================")
    print("PlantDiseaseTraining - TEST")
    print("==============================")
    print(f"Device       : {display_device}")
    print(f"Classes      : {statistics.num_classes}")
    print(f"Test Images  : {statistics.test_images}")
    print(f"Test Labels  : {statistics.test_labels}")
    print()
    print("Loading checkpoint:")
    print(checkpoint_path)
    print()
    print("Evaluation result:")
    print(f"Loss       : {result['loss']:.6f}")
    print(f"Accuracy   : {result['accuracy'] * 100:.2f}%")
    print(f"Precision  : {result['precision'] * 100:.2f}%")
    print(f"Recall     : {result['recall'] * 100:.2f}%")
    print(f"F1 Score   : {result['f1_score'] * 100:.2f}%")
    print()
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print()
    print("Per-class results:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        correct = int(result["confusion_matrix"][class_index, class_index])
        total = int(result["confusion_matrix"][class_index].sum())
        errors = total - correct
        print(f"{class_name:<10}: {correct}/{total} correct | {errors} errors")
    print()
    print("TEST evaluation completed.")


def _evaluate_test_split(
    model: torch.nn.Module,
    datamodule: PlantDiseaseDataModule,
    device: torch.device,
) -> dict[str, object]:
    criterion = torch.nn.CrossEntropyLoss()
    metrics = MetricsCalculator()
    confusion_matrix_generator = ConfusionMatrixGenerator()

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for images, targets in datamodule.test_loader:
            images = images.to(device)
            labels = _extract_labels(targets).to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            true_labels.extend(labels.detach().cpu().tolist())
            predicted_labels.extend(predictions.detach().cpu().tolist())

    average_loss = _safe_average(total_loss, total_examples)
    confusion_matrix = confusion_matrix_generator.generate(
        true_labels, predicted_labels, labels=[0, 1, 2]
    )

    return {
        "loss": average_loss,
        "accuracy": metrics.accuracy(true_labels, predicted_labels),
        "precision": metrics.precision(true_labels, predicted_labels),
        "recall": metrics.recall(true_labels, predicted_labels),
        "f1_score": metrics.f1_score(true_labels, predicted_labels),
        "confusion_matrix": confusion_matrix,
    }


def _extract_labels(targets: list[dict[str, object]]) -> torch.Tensor:
    labels: list[torch.Tensor] = []
    for target in targets:
        label_value = target["label"]
        if isinstance(label_value, torch.Tensor):
            labels.append(label_value.reshape(1).to(dtype=torch.long))
        else:
            labels.append(torch.tensor([int(label_value)], dtype=torch.long))
    return torch.cat(labels, dim=0)


def _safe_average(total_loss: float, total_examples: int) -> float:
    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def _use_pin_memory() -> bool:
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()

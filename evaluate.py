"""Standalone evaluation entry point for PlantDiseaseTraining."""

from __future__ import annotations

import argparse
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
from evaluation.evaluator import Evaluator
from models.model_config import ModelConfig
from models.model_factory import ModelFactory

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PlantDisease model"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_default_dataset_path(),
        help="Path to the dataset root. Defaults to dataset_redistributed when available.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=("train", "valid", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=_default_device_name())
    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_device = _resolve_device_name(args.device)
    pin_memory = selected_device.startswith("cuda")

    dataset_config = DatasetConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    try:
        datamodule = PlantDiseaseDataModule(
            dataset_config,
            validate_splits=(args.split,),
        )
    except (
        DatasetValidationError,
        DatasetPathNotFoundError,
        DataYamlNotFoundError,
        EmptyDatasetError,
        ClassDefinitionError,
    ) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc

    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    num_classes = datamodule.metadata.num_classes
    model_config = ModelConfig(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,
        device=selected_device,
    )

    model = ModelFactory.create(model_config)
    state_dict = torch.load(
        checkpoint_path, map_location=selected_device, weights_only=True
    )
    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()
    evaluator = Evaluator(
        model=model,
        dataloader=_resolve_dataloader(datamodule, args.split),
        criterion=criterion,
        device=selected_device,
    )
    result = evaluator.evaluate()

    _print_header(selected_device, datamodule, args.split)
    _print_result(result)
    print(f"Checkpoint used: {checkpoint_path}")


def _resolve_dataloader(datamodule: PlantDiseaseDataModule, split: str):
    if split == "train":
        return datamodule.train_loader
    if split == "valid":
        return datamodule.validation_loader
    return datamodule.test_loader


def _print_header(
    device_name: str, datamodule: PlantDiseaseDataModule, split: str
) -> None:
    statistics = datamodule.get_statistics()
    split_images = {
        "train": statistics.train_images,
        "valid": statistics.valid_images,
        "test": statistics.test_images,
    }
    print("==============================")
    print("PlantDisease Evaluation")
    print("==============================")
    print(f"Device       : {device_name.upper()}")
    print(f"Dataset path  : {datamodule.config.dataset_path}")
    print(f"Split        : {split}")
    print(f"Images       : {split_images[split]}")
    print(f"Classes      : {statistics.num_classes}")


def _print_result(result) -> None:
    print("Evaluation result:")
    print(f"Loss       : {result.loss:.6f}")
    print(f"Accuracy   : {result.accuracy:.4f}")
    print(f"Precision  : {result.precision:.4f}")
    print(f"Recall     : {result.recall:.4f}")
    print(f"F1 Score   : {result.f1_score:.4f}")
    print(f"Confusion Matrix:\n{result.confusion_matrix}")


def _default_device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_dataset_path() -> Path:
    redistributed_path = PROJECT_ROOT / "dataset_redistributed"
    if redistributed_path.exists():
        return redistributed_path
    return PROJECT_ROOT / "dataset"


def _resolve_device_name(requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return normalized


if __name__ == "__main__":
    main()

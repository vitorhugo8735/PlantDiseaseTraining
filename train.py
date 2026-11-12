"""Entry point for PlantDiseaseTraining model training.

This script wires together the existing dataset, model, training, and
evaluation layers without re-implementing their responsibilities. Keeping the
orchestration here makes the project easy to run from the command line while
preserving the modular architecture used throughout the codebase.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from training.losses import LossFactory
from training.optimizer import OptimizerFactory
from training.scheduler import SchedulerFactory
from training.trainer import Trainer
from training.training_config import TrainingConfig

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "outputs" / "training_history.json"


def parse_args() -> argparse.Namespace:
    """Parse a small set of CLI arguments for quick local runs."""

    parser = argparse.ArgumentParser(description="Train the PlantDisease model")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_default_dataset_path(),
        help="Path to the dataset root. Defaults to dataset_redistributed when available.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="steplr")
    parser.add_argument("--device", type=str, default=_default_device_name())
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the training and evaluation workflow."""

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
            validate_splits=("train", "valid"),
        )
    except (
        DatasetValidationError,
        DatasetPathNotFoundError,
        DataYamlNotFoundError,
        EmptyDatasetError,
        ClassDefinitionError,
    ) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc

    num_classes = datamodule.metadata.num_classes
    if num_classes <= 0:
        raise SystemExit(
            "Dataset validation failed: number of classes must be greater than zero."
        )

    model_config = ModelConfig(
        model_name="efficientnet_b0",
        num_classes=num_classes,
        pretrained=False,
        device=selected_device,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=selected_device,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        random_seed=args.random_seed,
        save_best_only=True,
        checkpoint_path=str(args.checkpoint_path),
    )

    _print_header(selected_device, datamodule)

    model = ModelFactory.create(model_config)
    _validate_torchscript_compatibility(model)

    criterion = LossFactory.create(training_config)
    optimizer = OptimizerFactory.create(model, training_config)
    scheduler = SchedulerFactory.create(optimizer, training_config)

    trainer = Trainer(
        model,
        datamodule.train_loader,
        datamodule.validation_loader,
        optimizer,
        scheduler,
        criterion,
        training_config,
    )

    history = trainer.fit()
    _save_training_history(history, args.history_path)

    _print_training_progress(trainer, history)
    _validate_torchscript_compatibility(trainer.model)

    evaluator = Evaluator(
        trainer.model,
        datamodule.validation_loader,
        criterion,
        selected_device,
    )
    result = evaluator.evaluate()

    _print_evaluation_result(result)
    print(f"Checkpoint saved to: {args.checkpoint_path}")
    print(f"Training history saved to: {args.history_path}")


def _print_header(device_name: str, datamodule: PlantDiseaseDataModule) -> None:
    statistics = datamodule.get_statistics()
    print("==============================")
    print("PlantDiseaseTraining")
    print("==============================")
    print(f"Device       : {device_name.upper()}")
    print(f"Classes      : {statistics.num_classes}")
    print(f"Train Images : {statistics.train_images}")
    print(f"Valid Images : {statistics.valid_images}")
    print(f"Test Images  : {statistics.test_images}")


def _print_training_progress(trainer: Trainer, history) -> None:
    total_epochs = len(history.train_loss)
    if total_epochs == 0:
        print("No training epochs were executed.")
        return

    for epoch_index, (
        train_loss,
        validation_loss,
        train_accuracy,
        validation_accuracy,
    ) in enumerate(
        zip(
            history.train_loss,
            history.validation_loss,
            history.train_accuracy,
            history.validation_accuracy,
            strict=True,
        ),
        start=1,
    ):
        print(
            f"Epoch {epoch_index}/{total_epochs} - "
            f"train_loss: {train_loss:.6f} - "
            f"validation_loss: {validation_loss:.6f} - "
            f"train_accuracy: {train_accuracy:.4f} - "
            f"validation_accuracy: {validation_accuracy:.4f}"
        )

    print(f"Best validation loss: {trainer.best_validation_loss:.6f}")


def _print_evaluation_result(result) -> None:
    print("Evaluation result:")
    print(f"Loss       : {result.loss:.6f}")
    print(f"Accuracy   : {result.accuracy:.4f}")
    print(f"Precision  : {result.precision:.4f}")
    print(f"Recall     : {result.recall:.4f}")
    print(f"F1 Score   : {result.f1_score:.4f}")
    print(f"Confusion Matrix:\n{result.confusion_matrix}")


def _save_training_history(history, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file_handle:
        json.dump(asdict(history), file_handle, indent=2)


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


def _validate_torchscript_compatibility(model: torch.nn.Module) -> None:
    try:
        torch.jit.script(model)
    except Exception as exc:  # pragma: no cover - surfaced directly to the user
        raise RuntimeError(f"TorchScript validation failed: {exc}") from exc


if __name__ == "__main__":
    main()

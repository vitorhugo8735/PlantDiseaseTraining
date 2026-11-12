"""Batch inference for PlantDiseaseTraining.

This script reuses the same model architecture, checkpoint loading, and
preprocessing used by predict.py, but processes an input folder in batches and
writes a CSV with per-image predictions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from inference.predictor import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_NAME,
    DEFAULT_SUPPORTED_IMAGE_EXTENSIONS,
    PlantDiseasePredictor,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "batch_predictions.csv"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class FailedImage:
    image: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict PlantDisease images")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing images to predict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path. Defaults to outputs/batch_predictions.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device_name(),
        help="Execution device. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("Batch size must be greater than zero.")

    input_folder = _validate_input_folder(args.input)
    image_paths = _collect_image_paths(input_folder)

    try:
        predictor = PlantDiseasePredictor(
            checkpoint_path=args.checkpoint_path,
            class_names=DEFAULT_CLASS_NAMES,
            device=args.device,
        )
        predictions = predictor.predict_batch(image_paths, batch_size=args.batch_size)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    failures = [
        FailedImage(image=failure.image_path.name, reason=failure.reason)
        for failure in predictor.last_failures
    ]

    output_path = _write_predictions_csv(args.output, predictions)
    _print_summary(
        input_folder, predictor.device_name, predictions, failures, output_path
    )


def _validate_input_folder(input_folder: Path) -> Path:
    resolved_folder = input_folder.expanduser().resolve()
    if not resolved_folder.exists():
        raise SystemExit(f"Input folder not found: {resolved_folder}")
    if not resolved_folder.is_dir():
        raise SystemExit(f"Input path is not a folder: {resolved_folder}")
    return resolved_folder


def _collect_image_paths(input_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _write_predictions_csv(output_path: Path, predictions) -> Path:
    resolved_output_path = output_path.expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "predicted_class",
        "confidence",
        "mildew_probability",
        "rose_P01_probability",
        "rose_R02_probability",
    ]
    with resolved_output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "image": prediction.image_path.name,
                    "predicted_class": prediction.predicted_class,
                    "confidence": f"{prediction.confidence:.4f}",
                    "mildew_probability": f"{prediction.probabilities[DEFAULT_CLASS_NAMES[0]]:.4f}",
                    "rose_P01_probability": f"{prediction.probabilities[DEFAULT_CLASS_NAMES[1]]:.4f}",
                    "rose_R02_probability": f"{prediction.probabilities[DEFAULT_CLASS_NAMES[2]]:.4f}",
                }
            )
    return resolved_output_path


def _print_summary(
    input_folder: Path,
    device_name: str,
    predictions,
    failures: list[FailedImage],
    output_path: Path,
) -> None:
    total_images = len(predictions) + len(failures)
    distribution = {class_name: 0 for class_name in DEFAULT_CLASS_NAMES}
    for prediction in predictions:
        distribution[prediction.predicted_class] += 1

    average_confidence = (
        sum(prediction.confidence for prediction in predictions) / len(predictions)
        if predictions
        else 0.0
    )

    print("==================================")
    print("PlantDiseaseTraining - BATCH")
    print("==================================")
    print(f"Device       : {device_name}")
    print(f"Input folder : {input_folder}")
    print(f"Images       : {total_images}")
    print(f"Processed    : {len(predictions)}")
    print(f"Errors       : {len(failures)}")
    print()
    print("Prediction distribution:")
    for class_name in DEFAULT_CLASS_NAMES:
        print(f"{class_name:<12}: {distribution[class_name]}")
    print()
    print("Average confidence:")
    print(f"{average_confidence * 100:.2f}%")
    print()
    print("Output:")
    print(output_path)

    if failures:
        print()
        print("Failed images:")
        for failure in failures:
            print(f"- {failure.image}: {failure.reason}")


def _default_device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()

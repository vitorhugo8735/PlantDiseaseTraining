"""Predict the class of a single PlantDisease image using the trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.predictor import (
    DEFAULT_CLASS_NAMES,
    PlantDiseasePredictor,
    PredictionError,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single PlantDisease image")
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to the input image"
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
    try:
        predictor = PlantDiseasePredictor(
            checkpoint_path=args.checkpoint_path,
            class_names=DEFAULT_CLASS_NAMES,
            device=args.device,
        )
        result = predictor.predict(args.image)
    except (FileNotFoundError, PredictionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    _print_result(result, predictor.device_name)


def _default_device_name() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _print_result(result, device_name: str) -> None:
    print("==================================")
    print("PlantDiseaseTraining - PREDICTION")
    print("==================================")
    print(f"Device       : {device_name}")
    print(f"Image        : {result.image_path}")
    print()
    print(f"Prediction   : {result.predicted_class}")
    print(f"Confidence   : {result.confidence * 100:.2f}%")
    print()
    print("Probabilities:")
    for class_name in DEFAULT_CLASS_NAMES:
        print(f"{class_name:<12}: {result.probabilities[class_name] * 100:.2f}%")


if __name__ == "__main__":
    main()

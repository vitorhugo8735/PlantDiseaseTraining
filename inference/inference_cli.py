"""CLI entrypoint used by the C# backend to run one PlantDisease prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference.predictor import (  # noqa: E402
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_NAME,
    PredictionError,
    PlantDiseasePredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PlantDisease single-image inference")
    parser.add_argument("--image", type=Path, required=True, help="Image to predict")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name used by the predictor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional execution device. When omitted, the predictor selects automatically.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        predictor = PlantDiseasePredictor(
            checkpoint_path=args.checkpoint_path,
            class_names=DEFAULT_CLASS_NAMES,
            model_name=args.model_name,
            device=args.device,
        )
        result = predictor.predict(args.image)
    except (FileNotFoundError, PredictionError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "predictedClassId": result.predicted_class_id,
        "predictedClass": result.predicted_class,
        "confidence": round(result.confidence, 4),
        "probabilities": {
            class_name: round(result.probabilities[class_name], 4)
            for class_name in DEFAULT_CLASS_NAMES
        },
    }

    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

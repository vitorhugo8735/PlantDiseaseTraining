"""Reusable inference core for PlantDiseaseTraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, UnidentifiedImageError

from configs.dataset_config import DatasetConfig
from data.transforms import get_validation_transforms
from models.model_config import ModelConfig
from models.model_factory import ModelFactory

DEFAULT_CLASS_NAMES = ("mildew", "rose_P01", "rose_R02")
DEFAULT_MODEL_NAME = "efficientnet_b0"
DEFAULT_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "best_model.pt"


class PredictionError(RuntimeError):
    """Raised when inference cannot proceed for a valid reason."""


@dataclass(slots=True, frozen=True)
class PredictionResult:
    """Structured single-image prediction output."""

    image_path: Path
    predicted_class_id: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(slots=True, frozen=True)
class PredictionFailure:
    """Record a failed image and why it could not be processed."""

    image_path: Path
    reason: str


@dataclass(slots=True)
class _PreparedImage:
    image_path: Path
    tensor: torch.Tensor


class PlantDiseasePredictor:
    """Load the trained PlantDisease model once and reuse it for inference."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | torch.device | None = None,
        dataset_config: DatasetConfig | None = None,
        supported_image_extensions: Sequence[str] = DEFAULT_SUPPORTED_IMAGE_EXTENSIONS,
    ) -> None:
        self.class_names = tuple(class_names)
        if not self.class_names:
            raise ValueError("class_names must contain at least one class")

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.checkpoint_path = self._validate_checkpoint_path(checkpoint_path)
        self.supported_image_extensions = {
            extension.lower() for extension in supported_image_extensions
        }
        self.dataset_config = dataset_config or DatasetConfig()
        self.transform = get_validation_transforms(self.dataset_config)
        self.last_failures: list[PredictionFailure] = []

        self.model = self._build_model()
        self._load_checkpoint()

    @property
    def device_name(self) -> str:
        return "CUDA" if self.device.type == "cuda" else "CPU"

    def predict(self, image_path: Path | str) -> PredictionResult:
        """Predict a single image and return the structured result."""

        prepared = self._prepare_image(image_path)
        batch_results = self._predict_prepared_batch([prepared])
        return batch_results[0]

    def predict_batch(
        self,
        image_paths: Iterable[Path | str],
        batch_size: int = 32,
    ) -> list[PredictionResult]:
        """Predict a batch of images while preserving the input order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.last_failures = []
        results: list[PredictionResult] = []
        pending: list[_PreparedImage] = []

        for image_path in image_paths:
            try:
                prepared = self._prepare_image(image_path)
            except PredictionError as exc:
                resolved_path = self._resolve_path(image_path)
                self.last_failures.append(
                    PredictionFailure(image_path=resolved_path, reason=str(exc))
                )
                continue

            pending.append(prepared)
            if len(pending) == batch_size:
                results.extend(self._predict_prepared_batch(pending))
                pending = []

        if pending:
            results.extend(self._predict_prepared_batch(pending))

        return results

    def _build_model(self) -> torch.nn.Module:
        model_config = ModelConfig(
            model_name=self.model_name,
            num_classes=len(self.class_names),
            pretrained=False,
            device=self.device.type,
        )
        return ModelFactory.create(model_config)

    def _load_checkpoint(self) -> None:
        try:
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
        except Exception as exc:
            raise PredictionError(
                f"Unable to load checkpoint: {self.checkpoint_path} ({exc})"
            ) from exc

        state_dict = self._extract_state_dict(payload)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise PredictionError(
                f"Checkpoint incompatible with {self.model_name}: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image_path: Path | str) -> _PreparedImage:
        resolved_path = self._validate_image_path(image_path)
        try:
            with Image.open(resolved_path) as image:
                rgb_image = image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise PredictionError(
                f"Unable to load image: {resolved_path} ({exc})"
            ) from exc
        except OSError as exc:
            raise PredictionError(
                f"Unable to load image: {resolved_path} ({exc})"
            ) from exc

        tensor = self.transform(rgb_image)
        return _PreparedImage(image_path=resolved_path, tensor=tensor)

    def _predict_prepared_batch(
        self, prepared_images: list[_PreparedImage]
    ) -> list[PredictionResult]:
        if not prepared_images:
            return []

        batch_tensor = torch.stack([item.tensor for item in prepared_images], dim=0).to(
            self.device
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()

        results: list[PredictionResult] = []
        for index, prepared_image in enumerate(prepared_images):
            row_probabilities = probabilities[index].tolist()
            predicted_class_id = int(
                max(range(len(row_probabilities)), key=row_probabilities.__getitem__)
            )
            probabilities_map = {
                class_name: float(row_probabilities[class_index])
                for class_index, class_name in enumerate(self.class_names)
            }
            results.append(
                PredictionResult(
                    image_path=prepared_image.image_path,
                    predicted_class_id=predicted_class_id,
                    predicted_class=self.class_names[predicted_class_id],
                    confidence=float(row_probabilities[predicted_class_id]),
                    probabilities=probabilities_map,
                )
            )

        return results

    def _validate_image_path(self, image_path: Path | str) -> Path:
        resolved_path = self._resolve_path(image_path)
        if not resolved_path.exists():
            raise PredictionError(f"Image not found: {resolved_path}")
        if not resolved_path.is_file():
            raise PredictionError(f"Image path is not a file: {resolved_path}")
        if resolved_path.suffix.lower() not in self.supported_image_extensions:
            raise PredictionError(
                f"Unsupported image extension: {resolved_path.suffix}. "
                f"Supported extensions: {sorted(self.supported_image_extensions)}"
            )
        return resolved_path

    def _validate_checkpoint_path(self, checkpoint_path: Path | str) -> Path:
        resolved_path = self._resolve_path(checkpoint_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint path is not a file: {resolved_path}")
        return resolved_path

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            if PlantDiseasePredictor._looks_like_state_dict(payload):
                return payload
            for key in ("model_state_dict", "state_dict", "model"):
                value = payload.get(key)
                if isinstance(
                    value, dict
                ) and PlantDiseasePredictor._looks_like_state_dict(value):
                    return value
        raise PredictionError(
            "Checkpoint format is invalid: expected a state_dict or a checkpoint containing model_state_dict/state_dict."
        )

    @staticmethod
    def _looks_like_state_dict(candidate: dict[str, Any]) -> bool:
        if not candidate:
            return False
        return all(isinstance(value, torch.Tensor) for value in candidate.values())

    def _resolve_device(self, device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            normalized_device = device
        else:
            normalized = device.strip().lower()
            normalized_device = torch.device(normalized)

        if normalized_device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return normalized_device

    @staticmethod
    def _resolve_path(path: Path | str) -> Path:
        return Path(path).expanduser().resolve()

"""Classifier wrapper for plant disease vision models."""

from __future__ import annotations

from torch import Tensor, nn

from models.model_config import ModelConfig


class PlantDiseaseClassifier(nn.Module):
    """Wrap a backbone model so the classifier interface stays stable.

    The wrapper exists so the rest of the project can depend on a single
    forward(images) contract, while the underlying backbone can change without
    affecting the rest of the training or TorchScript export pipeline.
    """

    def __init__(self, backbone: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = backbone
        _ = config

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits for a batch of images."""

        return self.backbone(images)

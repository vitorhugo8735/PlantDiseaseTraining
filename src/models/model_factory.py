"""Model factory for PlantDiseaseTraining."""

from __future__ import annotations

from torch import nn

from models.classifier import PlantDiseaseClassifier
from models.efficientnet import build_efficientnet_b0
from models.model_config import ModelConfig


class ModelFactory:
    """Create model instances behind a stable interface.

    A factory keeps the model selection logic isolated from the rest of the
    project, which makes it easier to add new backbones later and keeps the
    export path TorchScript-friendly by avoiding dynamic construction patterns.
    """

    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Build the requested model using the supplied configuration."""

        model_name = config.model_name.strip().lower()
        if model_name == "efficientnet_b0":
            backbone = build_efficientnet_b0(config)
            return PlantDiseaseClassifier(backbone=backbone, config=config)
        if model_name == "resnet50":
            raise NotImplementedError("resnet50 is reserved for a future model implementation")
        if model_name == "mobilenet_v3":
            raise NotImplementedError("mobilenet_v3 is reserved for a future model implementation")
        if model_name == "convnext":
            raise NotImplementedError("convnext is reserved for a future model implementation")
        if model_name == "vit":
            raise NotImplementedError("vit is reserved for a future model implementation")
        raise ValueError(f"Unsupported model_name: {config.model_name}")

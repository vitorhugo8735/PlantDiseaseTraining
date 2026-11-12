"""EfficientNet backbone construction utilities."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from models.model_config import ModelConfig


def build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    """Build an EfficientNet-B0 backbone adapted to the target classes."""

    weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config.num_classes)
    return model

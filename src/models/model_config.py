"""Configuration objects for model construction and export preparation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Centralize model construction parameters in one place.

    Keeping the model settings in a dedicated configuration object makes the
    model layer easy to test, swap, and later export to TorchScript without
    hardcoding architecture details in the wrapper classes.
    """

    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = False
    dropout: float = 0.2
    image_size: tuple[int, int] = (224, 224)
    device: str = "cpu"

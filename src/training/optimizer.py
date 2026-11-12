"""Optimizer factory for the PlantDisease training engine."""

from __future__ import annotations

from torch import nn
from torch.optim import Adam, AdamW, SGD, Optimizer

from training.training_config import TrainingConfig


class OptimizerFactory:
    """Create optimizers without embedding optimizer policy in the trainer.

    The factory pattern keeps the trainer focused on executing the learning
    loop. When new optimizers are added later, the trainer will not need to be
    rewritten, which keeps the architecture simpler to test and evolve.
    """

    @staticmethod
    def create(model: nn.Module, config: TrainingConfig) -> Optimizer:
        """Build the optimizer requested in the configuration."""

        optimizer_name = config.optimizer.strip().lower()
        parameters = model.parameters()

        if optimizer_name == "adam":
            return Adam(
                parameters, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        if optimizer_name == "adamw":
            return AdamW(
                parameters, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        if optimizer_name == "sgd":
            return SGD(
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

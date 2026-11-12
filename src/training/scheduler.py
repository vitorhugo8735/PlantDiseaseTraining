"""Scheduler factory for the PlantDisease training engine."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from training.training_config import TrainingConfig

SchedulerType = StepLR | CosineAnnealingLR | ReduceLROnPlateau


class SchedulerFactory:
    """Create learning-rate schedulers behind a stable interface.

    Scheduler selection lives here instead of inside the trainer so the loop
    only decides when to step a scheduler. That keeps the code ready for future
    changes like OneCycleLR or warmup policies without entangling the core loop.
    """

    @staticmethod
    def create(optimizer: Optimizer, config: TrainingConfig) -> SchedulerType | None:
        """Build the configured scheduler or return None when not requested."""

        scheduler_name = config.scheduler.strip().lower()
        if scheduler_name in {"", "none"}:
            return None
        if scheduler_name == "steplr":
            return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        if scheduler_name == "cosineannealinglr":
            return CosineAnnealingLR(
                optimizer, T_max=config.t_max, eta_min=config.eta_min
            )
        if scheduler_name == "reducelronplateau":
            return ReduceLROnPlateau(
                optimizer,
                factor=config.factor,
                patience=config.patience,
                threshold=config.threshold,
                min_lr=config.min_lr,
            )
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

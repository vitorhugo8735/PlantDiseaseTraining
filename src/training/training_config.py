"""Training configuration for the PlantDisease training engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    """Centralize training settings so the trainer stays free of hardcoded values.

    This separation keeps the training loop focused on execution while the
    configuration object owns tunable values. That makes the code easier to
    test and leaves room for future additions like AMP, checkpoints, and
    TensorBoard without changing the trainer contract.
    """

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "none"
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    random_seed: int = 42
    save_best_only: bool = False
    checkpoint_path: str = "checkpoints/best_model.pt"
    loss_name: str = "cross_entropy"
    momentum: float = 0.9
    nesterov: bool = False
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 10
    eta_min: float = 0.0
    factor: float = 0.1
    patience: int = 3
    threshold: float = 1e-4
    min_lr: float = 0.0

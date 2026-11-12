"""Training loop orchestration for the PlantDisease training engine."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from training.history import TrainingHistory
from training.training_config import TrainingConfig

SchedulerType = StepLR | CosineAnnealingLR | ReduceLROnPlateau


class Trainer:
    """Execute the training and validation loop.

    The trainer only knows how to run epochs. Factories provide the criterion,
    optimizer, and scheduler so this class stays narrow, predictable, and easy
    to test. That separation also makes future additions like AMP,
    checkpoints, and TensorBoard a matter of layering new concerns around this
    loop instead of rewriting it.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[Tensor, Any]],
        validation_loader: DataLoader[tuple[Tensor, Any]],
        optimizer: Optimizer,
        scheduler: SchedulerType | None,
        criterion: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.history = TrainingHistory()
        self.best_validation_loss = float("inf")
        self.best_epoch_index = -1
        self._best_state_dict: dict[str, Any] | None = None

        self._set_random_seed(config.random_seed)

    def train_epoch(self) -> tuple[float, float]:
        """Run one training epoch and return loss and accuracy."""

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            labels = self._extract_labels(targets).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

        return _safe_epoch_result(total_loss, total_correct, total_examples)

    def validate_epoch(self) -> tuple[float, float]:
        """Run one validation epoch without updating model weights."""

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for images, targets in self.validation_loader:
                images = images.to(self.device)
                labels = self._extract_labels(targets).to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += batch_size

        return _safe_epoch_result(total_loss, total_correct, total_examples)

    def fit(self) -> TrainingHistory:
        """Run the full training loop and return the accumulated history."""

        for _ in range(self.config.epochs):
            train_loss, train_accuracy = self.train_epoch()
            validation_loss, validation_accuracy = self.validate_epoch()

            self.history.record_epoch(
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_accuracy=train_accuracy,
                validation_accuracy=validation_accuracy,
            )

            current_epoch_index = len(self.history.train_loss) - 1
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.best_epoch_index = current_epoch_index
                self._best_state_dict = deepcopy(self.model.state_dict())
                if self.config.save_best_only:
                    self._save_checkpoint(self.config.checkpoint_path)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
            if self.config.save_best_only:
                self._save_checkpoint(self.config.checkpoint_path)

        return self.history

    def _extract_labels(self, targets: list[dict[str, Any]]) -> Tensor:
        labels: list[Tensor] = []
        for target in targets:
            label_value = target["label"]
            if isinstance(label_value, Tensor):
                labels.append(label_value.reshape(1).to(dtype=torch.long))
            else:
                labels.append(torch.tensor([int(label_value)], dtype=torch.long))
        return torch.cat(labels, dim=0)

    @staticmethod
    def _set_random_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_file)


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _safe_epoch_result(
    total_loss: float, total_correct: int, total_examples: int
) -> tuple[float, float]:
    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples

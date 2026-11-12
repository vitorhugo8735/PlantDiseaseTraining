"""Training history storage for the PlantDisease training engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingHistory:
    """Store epoch metrics without mixing persistence or visualization logic.

    Keeping history as a dedicated object makes the trainer easier to unit test
    and leaves room for future graphing, checkpointing, or reporting layers to
    consume the same data structure.
    """

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)

    def record_epoch(
        self,
        train_loss: float,
        validation_loss: float,
        train_accuracy: float,
        validation_accuracy: float,
    ) -> None:
        """Append one epoch of metrics to the history."""

        self.train_loss.append(train_loss)
        self.validation_loss.append(validation_loss)
        self.train_accuracy.append(train_accuracy)
        self.validation_accuracy.append(validation_accuracy)

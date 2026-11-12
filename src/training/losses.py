"""Loss factory for the PlantDisease training engine."""

from __future__ import annotations

from torch import nn

from training.training_config import TrainingConfig


class LossFactory:
    """Create the configured loss function behind a stable interface.

    A factory keeps loss selection out of the trainer so the loop stays focused
    on optimization mechanics. That separation also makes future additions such
    as focal loss or label smoothing straightforward without changing the
    trainer contract.
    """

    @staticmethod
    def create(config: TrainingConfig | None = None) -> nn.Module:
        """Build the configured criterion.

        Only CrossEntropyLoss is enabled in this stage, but the switch-based
        structure makes future expansion explicit and easy to test.
        """

        loss_name = (
            "cross_entropy" if config is None else config.loss_name.strip().lower()
        )
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        if loss_name == "focal_loss":
            raise NotImplementedError("focal_loss will be added in a later stage")
        if loss_name == "label_smoothing":
            raise NotImplementedError("label_smoothing will be added in a later stage")
        if loss_name == "weighted_cross_entropy":
            raise NotImplementedError(
                "weighted_cross_entropy will be added in a later stage"
            )
        raise ValueError(f"Unsupported loss_name: {loss_name}")

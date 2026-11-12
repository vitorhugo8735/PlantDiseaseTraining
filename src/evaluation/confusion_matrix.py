"""Confusion matrix generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(slots=True)
class ConfusionMatrixGenerator:
    """Generate a confusion matrix without producing any plots or images.

    The generator is isolated so the evaluator can stay focused on orchestration
    and future presentation layers can reuse the same matrix output for charts
    or reports.
    """

    def generate(
        self,
        y_true: list[int],
        y_pred: list[int],
        labels: list[int] | None = None,
    ) -> np.ndarray:
        """Return the raw confusion matrix as a NumPy array."""

        return confusion_matrix(y_true, y_pred, labels=labels)

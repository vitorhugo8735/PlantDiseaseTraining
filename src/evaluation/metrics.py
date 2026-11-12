"""Metric calculation helpers for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

AveragingStrategy = Literal["weighted", "macro", "micro", "samples"]


@dataclass(slots=True)
class MetricsCalculator:
    """Calculate classification metrics without embedding that logic in the evaluator.

    Keeping metrics separate ensures the evaluator only coordinates the
    workflow, while this class owns the scoring policy. That makes the design
    easier to test and allows future changes such as macro or micro averaging
    without rewriting the evaluation loop.
    """

    average: AveragingStrategy = "weighted"

    def accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
        """Return the accuracy score for the given labels."""

        return float(accuracy_score(y_true, y_pred))

    def precision(self, y_true: list[int], y_pred: list[int]) -> float:
        """Return the precision score using the configured averaging strategy."""

        return float(
            precision_score(y_true, y_pred, average=self.average, zero_division=0)
        )

    def recall(self, y_true: list[int], y_pred: list[int]) -> float:
        """Return the recall score using the configured averaging strategy."""

        return float(
            recall_score(y_true, y_pred, average=self.average, zero_division=0)
        )

    def f1_score(self, y_true: list[int], y_pred: list[int]) -> float:
        """Return the F1 score using the configured averaging strategy."""

        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))

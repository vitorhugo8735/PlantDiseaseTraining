"""Classification report generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import classification_report


@dataclass(slots=True)
class ClassificationReportGenerator:
    """Generate the classification report as structured data.

    Returning a dictionary keeps the result machine-friendly for future export
    layers, dashboards, and comparisons without forcing any presentation logic
    into the evaluator.
    """

    def generate(
        self,
        y_true: list[int],
        y_pred: list[int],
        labels: list[int] | None = None,
        target_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return sklearn's classification report in dictionary form."""

        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

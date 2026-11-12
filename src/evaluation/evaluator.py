"""Evaluation loop for trained PlantDisease models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from evaluation.classification_report import ClassificationReportGenerator
from evaluation.confusion_matrix import ConfusionMatrixGenerator
from evaluation.evaluation_result import EvaluationResult
from evaluation.metrics import MetricsCalculator


class Evaluator:
    """Run model evaluation and assemble the final structured result.

    The evaluator only coordinates the flow: run the model, collect labels, and
    ask the specialized helpers for metrics and reports. This keeps the design
    modular and makes it straightforward to add TensorBoard, graphs, or export
    formats later without rewriting the evaluation logic.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader[tuple[Tensor, Any]],
        criterion: nn.Module,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = torch.device(device)
        self.metrics_calculator = MetricsCalculator()
        self.confusion_matrix_generator = ConfusionMatrixGenerator()
        self.classification_report_generator = ClassificationReportGenerator()

        self.model.to(self.device)

    def evaluate(self) -> EvaluationResult:
        """Evaluate the model and return all aggregate metrics."""

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        true_labels: list[int] = []
        predicted_labels: list[int] = []

        with torch.no_grad():
            for images, targets in self.dataloader:
                images = images.to(self.device)
                labels = self._extract_labels(targets).to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                true_labels.extend(labels.detach().cpu().tolist())
                predicted_labels.extend(predictions.detach().cpu().tolist())

        loss = self._safe_average(total_loss, total_examples)
        accuracy = self.metrics_calculator.accuracy(true_labels, predicted_labels)
        precision = self.metrics_calculator.precision(true_labels, predicted_labels)
        recall = self.metrics_calculator.recall(true_labels, predicted_labels)
        f1_score = self.metrics_calculator.f1_score(true_labels, predicted_labels)
        confusion_matrix = self.confusion_matrix_generator.generate(
            true_labels, predicted_labels
        )
        classification_report = self.classification_report_generator.generate(
            true_labels, predicted_labels
        )

        return EvaluationResult(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_matrix,
            classification_report=classification_report,
        )

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
    def _safe_average(total_loss: float, total_examples: int) -> float:
        if total_examples == 0:
            return 0.0
        return total_loss / total_examples

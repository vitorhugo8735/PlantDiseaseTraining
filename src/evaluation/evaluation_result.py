"""Structured evaluation output for the PlantDisease project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class EvaluationResult:
    """Store the outcome of one evaluation pass.

    This object exists so the evaluator returns a single structured payload
    instead of scattering metrics, matrices, and reports across separate
    variables. That keeps the evaluation layer easy to test and makes future
    integrations with TensorBoard, plots, and exports simpler.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: dict[str, Any]

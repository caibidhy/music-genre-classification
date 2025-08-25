from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    accuracy: float
    report: Dict[str, Dict[str, float]]


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> EvaluationResult:
    """Run ``model`` on ``loader`` and compute basic metrics.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    loader:
        DataLoader providing ``(inputs, labels)`` batches.
    device:
        Device on which inference should run.
    """

    model.eval()
    model.to(device)

    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)
    return EvaluationResult(accuracy=accuracy, report=report)


__all__ = ["evaluate", "EvaluationResult"]
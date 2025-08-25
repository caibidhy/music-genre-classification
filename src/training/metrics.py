from __future__ import annotations

import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the average classification accuracy."""
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).float().mean().item()
    return float(correct)


__all__ = ["accuracy"]
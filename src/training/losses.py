from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the standard cross entropy loss."""
    return F.cross_entropy(outputs, targets)


__all__ = ["cross_entropy"]
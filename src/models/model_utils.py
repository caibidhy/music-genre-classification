from __future__ import annotations

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["count_parameters"]
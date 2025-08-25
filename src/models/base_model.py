from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models used in the project."""

    def __init__(self) -> None:  # pragma: no cover - simple pass through
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        raise NotImplementedError


__all__ = ["BaseModel"]
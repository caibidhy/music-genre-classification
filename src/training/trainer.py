from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .metrics import accuracy


@dataclass
class Trainer:
    """Utility class that encapsulates a basic training and evaluation loop."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: str | torch.device = "cpu"
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = accuracy

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def _move_batch(self, batch: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(b.to(self.device) for b in batch)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = self._move_batch((inputs, targets))
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(loader.dataset)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        for inputs, targets in loader:
            inputs, targets = self._move_batch((inputs, targets))
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            if self.metric_fn is not None:
                total_metric += self.metric_fn(outputs, targets) * inputs.size(0)
        result = {"loss": total_loss / len(loader.dataset)}
        if self.metric_fn is not None:
            result["metric"] = total_metric / len(loader.dataset)
        return result


__all__ = ["Trainer"]
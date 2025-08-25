from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Simple early stopping mechanism based on a monitored metric."""

    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> None:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


__all__ = ["EarlyStopping"]
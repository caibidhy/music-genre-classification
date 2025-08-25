from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


class CRNNClassifier(BaseModel):
    """CRNN that first extracts features with CNN layers and then processes
    the time dimension with a GRU layer."""

    def __init__(
        self, num_classes: int, n_mels: int = 128, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.rnn = nn.GRU((n_mels // 2) * 32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.rnn(x)
        out = out[:, -1]
        return self.fc(out)


__all__ = ["CRNNClassifier"]
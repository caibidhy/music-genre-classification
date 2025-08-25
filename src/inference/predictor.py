from __future__ import annotations

from typing import List, Sequence

import torch

from .audio_processor import AudioProcessor


class Predictor:
    """Load a trained model and perform inference on audio files."""

    def __init__(
        self,
        model: torch.nn.Module,
        labels: Sequence[str],
        processor: AudioProcessor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.labels = list(labels)
        self.processor = processor if processor is not None else AudioProcessor()
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(self, path: str) -> str:
        """Predict the genre label for the given audio file path."""
        tensor = self.processor(path).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        idx = int(torch.argmax(probs, dim=1))
        return self.labels[idx]


__all__ = ["Predictor"]
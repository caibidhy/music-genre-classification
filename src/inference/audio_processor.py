from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..data.preprocessing import load_audio, normalize_audio
from ..data.spectrogram_utils import melspectrogram


@dataclass
class AudioProcessor:
    """Callable object that converts an audio file path to a model-ready tensor."""

    sample_rate: int = 22050
    n_mels: int = 128
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, path: str) -> torch.Tensor:
        signal, sr = load_audio(path, sample_rate=self.sample_rate)
        signal = normalize_audio(signal)
        spec = melspectrogram(signal, sr, n_mels=self.n_mels)
        tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, time)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


__all__ = ["AudioProcessor"]
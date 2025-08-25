from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_audio, normalize_audio
from .spectrogram_utils import melspectrogram

Item = Tuple[str, int]


def _ensure_list(items: Iterable[Item]) -> List[Item]:
    if isinstance(items, list):
        return items
    return list(items)


class AudioDataset(Dataset):
    """Simple dataset that loads audio files and returns mel spectrograms."""

    def __init__(
        self,
        items: Iterable[Item],
        sample_rate: int = 22050,
        n_mels: int = 128,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.items: List[Item] = _ensure_list(items)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[index]
        signal, sr = load_audio(path, self.sample_rate)
        signal = normalize_audio(signal)
        spec = melspectrogram(signal, sr, n_mels=self.n_mels)
        if self.transform is not None:
            spec = self.transform(spec)
        spec = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, time)
        return spec, label


__all__ = ["AudioDataset"]
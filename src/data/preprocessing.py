from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def load_audio(
    path: str,
    sample_rate: int = 22050,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file using :func:`librosa.load`.

    The function is a thin wrapper that exposes the most commonly used options
    in this project. It returns the audio signal as a NumPy array together with
    the sampling rate used for loading.
    """

    signal, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return signal, sr


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    """Normalise an audio signal to the range ``[-1, 1]``.

    Parameters
    ----------
    signal:
        Input audio array.
    """

    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


__all__ = ["load_audio", "normalize_audio"]
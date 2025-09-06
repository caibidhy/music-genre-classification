from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import torch
import torchaudio


def melspectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a log-scaled mel spectrogram using ``librosa``.

    Parameters
    ----------
    signal:
        Audio signal as a 1‑D NumPy array.
    sample_rate:
        Sampling rate of the ``signal``.
    n_mels, hop_length, n_fft:
        Parameters forwarded to :func:`librosa.feature.melspectrogram`.
    **kwargs:
        Additional keyword arguments passed to the underlying librosa function.
    """

    try:
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            **kwargs,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel.astype(np.float32)
    except Exception:
        # Fall back to torchaudio implementation on failure. Avoid relying on
        # librosa for any subsequent processing in this branch to prevent
        # potential crashes from propagating.
        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        mel = transform(waveform).squeeze(0)
        mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        return mel.numpy().astype(np.float32)


__all__ = ["melspectrogram"]

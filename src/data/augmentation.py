from __future__ import annotations

import numpy as np


def add_noise(signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add white noise to an audio signal."""
    noise = np.random.randn(len(signal)) * noise_factor
    return signal + noise


def time_shift(signal: np.ndarray, shift: int) -> np.ndarray:
    """Roll the signal by ``shift`` samples."""
    return np.roll(signal, shift)


__all__ = ["add_noise", "time_shift"]
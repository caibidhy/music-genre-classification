import numpy as np
import pytest
import soundfile as sf

from src.data.preprocessing import load_audio, normalize_audio
from src.data import spectrogram_utils


def test_load_audio_reads_file(tmp_path):
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    file_path = tmp_path / 'test.wav'
    sf.write(file_path, signal, sr, subtype="FLOAT")

    loaded_signal, loaded_sr = load_audio(str(file_path), sample_rate=sr)

    assert loaded_sr == sr
    assert np.allclose(loaded_signal, signal)


def test_normalize_audio_scales_to_unit_range():
    signal = np.array([0.5, -0.25, 0.25], dtype=np.float32)
    normalized = normalize_audio(signal)
    assert np.max(np.abs(normalized)) == pytest.approx(1.0)
    assert np.allclose(normalized, signal / 0.5)


def test_melspectrogram_fallback(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(spectrogram_utils.librosa.feature, "melspectrogram", _raise)

    signal = np.random.randn(22050).astype(np.float32)
    mel = spectrogram_utils.melspectrogram(signal, sample_rate=22050, n_mels=64)

    assert mel.shape[0] == 64

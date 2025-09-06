import numpy as np
import pytest
import soundfile as sf

from src.data.preprocessing import load_audio, normalize_audio


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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import AudioDataset
from src.models.cnn_model import CNNClassifier
from src.training.trainer import Trainer


def _create_tiny_dataset(tmpdir: Path) -> list[tuple[str, int]]:
    sr = 22_050
    items: list[tuple[str, int]] = []
    for label in range(2):
        signal = np.random.randn(sr).astype(np.float32)
        file_path = tmpdir / f"sample_{label}.wav"
        sf.write(file_path, signal, sr)
        items.append((str(file_path), label))
    return items


def test_trainer_smoke_run_one_epoch(tmp_path: Path) -> None:
    items = _create_tiny_dataset(tmp_path)
    dataset = AudioDataset(items)
    loader = DataLoader(dataset, batch_size=2)

    model = CNNClassifier(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)

    metrics = trainer.evaluate(loader)
    assert set(metrics.keys()) == {"loss", "metric"}
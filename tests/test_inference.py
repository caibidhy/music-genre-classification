import torch

from src.inference.predictor import Predictor


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.1, 0.9]])


class DummyProcessor:
    def __call__(self, path: str):
        return torch.zeros(1, 128, 44)


def test_predictor_returns_label():
    predictor = Predictor(DummyModel(), ["class0", "class1"], processor=DummyProcessor())
    label = predictor.predict("dummy.wav")
    assert label == "class1"
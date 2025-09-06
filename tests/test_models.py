import torch

from src.models.cnn_model import CNNClassifier
from src.models.crnn_model import CRNNClassifier


def test_cnn_classifier_output_shape():
    model = CNNClassifier(num_classes=10)
    x = torch.randn(4, 1, 64, 64)
    out = model(x)
    assert out.shape == (4, 10)


def test_crnn_classifier_output_shape():
    model = CRNNClassifier(num_classes=5, n_mels=128)
    x = torch.randn(2, 1, 128, 32)
    out = model(x)
    assert out.shape == (2, 5)
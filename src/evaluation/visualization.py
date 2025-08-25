from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    targets: List[int], preds: List[int], class_names: List[str] | None = None
) -> plt.Axes:
    """Plot a confusion matrix using ``seaborn``."""

    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=0)
    return ax


def plot_spectrogram(spec: np.ndarray) -> plt.Axes:
    """Display a mel spectrogram."""

    fig, ax = plt.subplots()
    img = ax.imshow(spec, origin="lower", aspect="auto")
    fig.colorbar(img, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bins")
    return ax


__all__ = ["plot_confusion_matrix", "plot_spectrogram"]
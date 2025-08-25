from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GradCAM:
    """Compute Grad-CAM heatmaps for a given convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _: nn.Module, __, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        if isinstance(class_idx, torch.Tensor):
            loss = output[torch.arange(output.size(0)), class_idx]
            loss = loss.sum()
        else:  # single int
            loss = output[:, class_idx].sum()
        loss.backward()

        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        # Normalise each CAM to [0, 1]
        cam -= cam.amin(dim=(1, 2), keepdim=True)
        cam /= cam.amax(dim=(1, 2), keepdim=True) + 1e-6
        return cam

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


__all__ = ["GradCAM"]
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across ``random``, ``numpy`` and
    :mod:`torch`.

    Parameters
    ----------
    seed:
        The seed value to set for all relevant libraries.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """Save training state to disk.

    Parameters
    ----------
    state:
        Dictionary containing model state and meta information.
    path:
        Destination file path. Parent directories will be created
        automatically.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


__all__ = ["set_seed", "save_checkpoint"]
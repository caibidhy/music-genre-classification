from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file. The path can be given as a string or
        :class:`~pathlib.Path` instance.

    Returns
    -------
    dict
        Parsed configuration represented as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the provided path does not point to an existing file.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise yaml.YAMLError("Configuration root element must be a mapping")

    return config


__all__ = ["load_config"]
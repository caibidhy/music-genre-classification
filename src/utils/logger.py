from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    Parameters
    ----------
    name:
        Name of the logger. If ``None`` a logger with the module's ``__name__``
        will be created.
    level:
        Logging level; defaults to :data:`logging.INFO`.
    log_file:
        Optional path to a log file. When provided, a file handler is added
        alongside the standard stream handler.
    """

    logger = logging.getLogger(name if name is not None else __name__)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
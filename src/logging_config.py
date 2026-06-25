"""Logging configuration for AlphaLens.

Call setup_logging() once at application startup (in main.py). All modules then
obtain a logger via logging.getLogger(__name__) and their output is formatted
consistently. This replaces ad-hoc print() calls with leveled, timestamped logs
that can be filtered or silenced centrally.
"""
from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging format and level for the application.

    Args:
        level: The minimum logging level to emit (e.g. logging.INFO,
            logging.DEBUG). Defaults to INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
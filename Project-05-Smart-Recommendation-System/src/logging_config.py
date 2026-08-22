"""Application-wide logging configuration.

Logs are written both to the console and to ``logs/pipeline.log``.
The setup is idempotent: calling :func:`get_logger` multiple times
never duplicates handlers.
"""

import logging
from logging.handlers import RotatingFileHandler

from src import config

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for console + file output."""
    logger = logging.getLogger(name)
    if getattr(logger, "_codevedx_configured", False):
        return logger

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    file_handler = RotatingFileHandler(
        config.LOGS_DIR / config.LOG_FILE_NAME,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._codevedx_configured = True  # type: ignore[attr-defined]
    return logger

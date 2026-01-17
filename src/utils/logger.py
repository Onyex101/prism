"""
Logger Setup Module
===================

This module configures logging for the PRISM application.

It uses loguru for enhanced logging with color support and
file rotation capabilities.

Example:
    >>> from src.utils.logger import setup_logger
    >>> setup_logger(log_level="DEBUG", log_file="logs/app.log")
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure the application logger.

    Sets up console logging with color output and optionally file
    logging with rotation.

    :param log_level: Minimum log level. Options: "DEBUG", "INFO",
        "WARNING", "ERROR", "CRITICAL".
    :type log_level: str
    :param log_file: Path to log file (optional). If provided, logs
        will be written to this file with rotation.
    :type log_file: Optional[str]
    :param rotation: When to rotate log files (e.g., "10 MB", "1 day").
    :type rotation: str
    :param retention: How long to keep log files (e.g., "30 days", "1 week").
    :type retention: str

    Example:
        >>> setup_logger(log_level="INFO")
        >>> setup_logger(
        ...     log_level="DEBUG",
        ...     log_file="logs/prism.log",
        ...     rotation="5 MB",
        ...     retention="7 days"
        ... )
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    _add_console_handler(log_level)

    # File handler (if specified)
    if log_file:
        _add_file_handler(log_file, log_level, rotation, retention)

    logger.info(f"Logger configured with level: {log_level}")


def _add_console_handler(log_level: str) -> None:
    """
    Add console handler with color formatting.

    :param log_level: Minimum log level.
    :type log_level: str
    """
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def _add_file_handler(
    log_file: str,
    log_level: str,
    rotation: str,
    retention: str,
) -> None:
    """
    Add file handler with rotation.

    :param log_file: Path to log file.
    :type log_file: str
    :param log_level: Minimum log level.
    :type log_level: str
    :param rotation: Rotation setting.
    :type rotation: str
    :param retention: Retention setting.
    :type retention: str
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | " "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

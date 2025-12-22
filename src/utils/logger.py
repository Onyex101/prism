"""
Logger Setup Module

Configures logging for the PRISM application.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure the application logger.

    Args:
        log_level: Minimum log level
        log_file: Path to log file (optional)
        rotation: When to rotate log files
        retention: How long to keep log files
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
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

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logger configured with level: {log_level}")

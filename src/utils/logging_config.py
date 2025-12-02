"""
Centralized logging configuration for Water Quality Index application.

This module provides a single place to configure logging for the application.
It should only be called from entry points (main scripts, Streamlit app).
Library modules should use `logger = logging.getLogger(__name__)` only.
"""

import logging
import os


def configure_logging(level: str = None) -> None:
    """
    Configure logging for the application.

    Call this ONCE at application startup from entry points only:
    - streamlit_app/app.py
    - train_models.py
    - scripts that run as __main__

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to
               LOG_LEVEL environment variable or INFO.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

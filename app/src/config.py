"""
Configuration module for logging setup.

This module provides logging configuration functions for the application,
setting up handlers for both console and file output.
"""

import logging
import sys
from typing import Any


def logger() -> Any:
    """
    Configure and return a logger instance with console and file handlers.

    Sets up logging configuration for the application, including handlers
    for both console output and file logging. Also configures log levels
    for external libraries to reduce noise.

    Returns:
    --------
    Any
        Configured logger instance.
    """
    # Set higher level for external libs logging
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("boto3").setLevel(logging.ERROR)
    logging.getLogger("urllib").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    level = "INFO"

    log_format = "%(asctime)s - %(levelname)s - %(filename)s[%(lineno)s] %(message)s"
    app_logger = logging.getLogger()
    for handler in app_logger.handlers:
        app_logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    app_logger.addHandler(console_handler)
    app_logger.setLevel(level)

    # save log to file
    file_handler = logging.FileHandler("log.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    app_logger.addHandler(file_handler)

    return app_logger

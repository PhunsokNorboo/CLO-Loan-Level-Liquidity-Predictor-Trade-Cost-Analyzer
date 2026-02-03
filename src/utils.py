"""
Common utility functions for the loan liquidity predictor.

Provides:
- Logging configuration
- Project path management
- Configuration loading
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root directory.
    """
    return Path(__file__).parent.parent


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the data directory path.

    Args:
        subdir: Optional subdirectory (e.g., 'raw', 'processed', 'external')

    Returns:
        Path to the data directory or subdirectory.
    """
    data_dir = get_project_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """
    Get the models directory path.

    Returns:
        Path to the models directory.
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_reports_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the reports directory path.

    Args:
        subdir: Optional subdirectory (e.g., 'figures')

    Returns:
        Path to the reports directory or subdirectory.
    """
    reports_dir = get_project_root() / "reports"
    if subdir:
        reports_dir = reports_dir / subdir
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("loan_liquidity_predictor")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = get_project_root() / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def load_environment() -> None:
    """
    Load environment variables from .env file.

    Searches for .env file in the project root directory.
    """
    env_path = get_project_root() / ".env"
    load_dotenv(env_path)


def get_api_key(key_name: str) -> Optional[str]:
    """
    Get an API key from environment variables.

    Args:
        key_name: Name of the environment variable

    Returns:
        API key value or None if not found.

    Raises:
        ValueError: If the key is not set and is required.
    """
    load_environment()
    value = os.getenv(key_name)
    if value is None or value.startswith("your_"):
        return None
    return value


# Initialize logging when module is imported
logger = setup_logging()

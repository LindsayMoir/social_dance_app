"""
Centralized logging configuration for multi-environment support.

This module provides a unified way to configure logging that adapts based on
the execution environment (local development vs Render production).

Environment Behavior:
    - Local Development: Logs written to files in logs/ directory
    - Render Production: Logs written to stdout for visibility in Render console

Usage:
    from logging_config import setup_logging

    # In your script
    setup_logging('script_name')
    logging.info("This will go to the right place automatically")

Features:
    - Automatic environment detection (RENDER environment variable)
    - Consistent log format across all scripts
    - Creates log directory if needed (local only)
    - Thread-safe and works with multiprocessing
"""

import logging
import os
import sys


def setup_logging(script_name: str, level=logging.INFO):
    """
    Configure logging based on execution environment.

    Args:
        script_name (str): Name of the script (used for log filename in local mode)
        level (int): Logging level (default: logging.INFO)

    Environment Detection:
        - RENDER='true': Logs to stdout (for Render console)
        - Otherwise: Logs to logs/{script_name}_log.txt

    Example:
        setup_logging('emails')
        logging.info("Processing emails...")  # Goes to right destination automatically
    """
    # Check if running on Render
    is_render = os.getenv('RENDER') == 'true'

    # Common format for all logs with pipeline run correlation context.
    run_id = os.getenv("DS_RUN_ID", "na")
    step_name = os.getenv("DS_STEP_NAME", script_name)
    log_format = (
        f"%(asctime)s - %(levelname)s - [run_id={run_id}] "
        f"[step={step_name}] - %(message)s"
    )
    date_format = '%Y-%m-%d %H:%M:%S'

    if is_render:
        # RENDER MODE: Log to stdout so it appears in Render console
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True  # Override any existing configuration
        )
        logging.info(f"Logging configured for Render (stdout) - {script_name}")
    else:
        # LOCAL MODE: Log to file
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/{script_name}_log.txt"

        logging.basicConfig(
            filename=log_file,
            filemode='a',
            level=level,
            format=log_format,
            datefmt=date_format,
            force=True  # Override any existing configuration
        )
        logging.info(f"Logging configured for local development (file: {log_file})")

    # Suppress verbose HTTP logging from third-party libraries
    # This prevents DEBUG logs from httpx, httpcore, and openai from cluttering logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    This is useful when you want module-specific loggers that still
    respect the centralized configuration.

    Args:
        name (str): Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Module-specific log message")
    """
    return logging.getLogger(name)

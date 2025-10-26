"""
Database configuration utility for multi-environment support.

This module centralizes all database connection logic and provides functions
to get the appropriate connection string based on execution environment.

Environment Variables:
    DATABASE_TARGET: Explicitly set target database
        - 'local': Local PostgreSQL database
        - 'render_dev': Render development database
        - 'render_prod': Render production database

    RENDER: Set to 'true' when running on Render platform
        - Used to choose INTERNAL vs EXTERNAL connection URLs
        - INTERNAL URLs are faster within Render's network

Execution Modes:
    1. Local Development: DATABASE_TARGET=local, RENDER=false
       → Uses DATABASE_CONNECTION_STRING

    2. Local → Render Dev: DATABASE_TARGET=render_dev, RENDER=false
       → Uses RENDER_DEV_EXTERNAL_DB_URL

    3. Render CRON → Render Dev: DATABASE_TARGET=render_dev, RENDER=true
       → Uses RENDER_DEV_INTERNAL_DB_URL

    4. Production Web Service: DATABASE_TARGET=render_prod, RENDER=true
       → Uses RENDER_INTERNAL_DB_URL

Usage:
    from db_config import get_database_config

    connection_string, env_name = get_database_config()
    engine = create_engine(connection_string)
"""

import os
import logging
from typing import Tuple
try:
    from environment import IS_RENDER
except ImportError:
    # Fallback for import paths that don't have src as root
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import IS_RENDER


def get_database_config() -> Tuple[str, str]:
    """
    Determine which database to connect to based on environment variables.

    Returns:
        Tuple[str, str]: (connection_string, environment_name)

    Raises:
        ValueError: If DATABASE_TARGET is invalid or connection string is missing

    Environment Variables Used:
        - DATABASE_TARGET: 'local', 'render_dev', or 'render_prod'
        - RENDER: 'true' if running on Render, otherwise unset/false

    Logic:
        1. If DATABASE_TARGET is explicitly set, use that
        2. Otherwise, infer from RENDER environment variable:
           - RENDER=true → render_prod (safe default for production)
           - RENDER=false/unset → local
        3. Choose INTERNAL vs EXTERNAL URL based on RENDER variable
    """
    # Auto-detect if running on Render platform
    # Render automatically sets these environment variables
    is_render = (
        IS_RENDER or
        os.getenv('RENDER_SERVICE_NAME') is not None or
        os.getenv('RENDER_INSTANCE_ID') is not None or
        'render.com' in os.getenv('HOSTNAME', '')
    )

    # Get explicit target or auto-detect
    target = os.getenv('DATABASE_TARGET', '').lower().strip()

    if not target:
        # Auto-detect based on platform
        if is_render:
            # Running on Render - default to render_dev (safe for CRON jobs)
            # Production web services should set DATABASE_TARGET=render_prod explicitly
            target = 'render_dev'
            logging.info(f"Auto-detected: Running on Render → DATABASE_TARGET='render_dev'")
        else:
            # Running locally (your machine) - use local database
            target = 'local'
            logging.info(f"Auto-detected: Running locally → DATABASE_TARGET='local'")
    else:
        # If DATABASE_TARGET is explicitly set but empty/whitespace, still auto-detect
        if target == '':
            if is_render:
                target = 'render_dev'
                logging.info(f"DATABASE_TARGET empty, auto-detected: Running on Render → 'render_dev'")
            else:
                target = 'local'
                logging.info(f"DATABASE_TARGET empty, auto-detected: Running locally → 'local'")
        else:
            logging.info(f"DATABASE_TARGET explicitly set: {target}")

    # Map target to connection string
    # Use INTERNAL URLs when running on Render (faster), EXTERNAL from local machine
    connection_map = {
        'local': (
            os.getenv('DATABASE_CONNECTION_STRING'),
            'Local PostgreSQL (localhost)'
        ),
        'render_dev': (
            os.getenv('RENDER_DEV_INTERNAL_DB_URL') if is_render
                else os.getenv('RENDER_DEV_EXTERNAL_DB_URL'),
            'Render Development Database'
        ),
        'render_prod': (
            os.getenv('RENDER_INTERNAL_DB_URL') if is_render
                else os.getenv('RENDER_EXTERNAL_DB_URL'),
            'Render Production Database'
        )
    }

    # Validate target
    if target not in connection_map:
        raise ValueError(
            f"Invalid DATABASE_TARGET: '{target}'. "
            f"Must be one of: local, render_dev, render_prod"
        )

    connection_string, env_name = connection_map[target]

    # Validate connection string exists
    if not connection_string:
        raise ValueError(
            f"Database connection string not found for target '{target}'. "
            f"Check your environment variables. "
            f"Running on Render: {is_render}"
        )

    # Log configuration for debugging
    logging.info(f"=== Database Configuration ===")
    logging.info(f"  Target: {target}")
    logging.info(f"  Environment: {env_name}")
    logging.info(f"  Running on: {'Render' if is_render else 'Local Machine'}")
    logging.info(f"  URL Type: {'INTERNAL' if is_render else 'EXTERNAL'}")
    logging.info(f"==============================")

    return connection_string, env_name


def get_production_database_url() -> str:
    """
    Get the production database URL for copying data to production.

    This is always the production database, regardless of where code is running
    or what DATABASE_TARGET is set to. Used by the final step of pipeline.py
    to copy data from dev/local to production.

    Returns:
        str: Production database connection string

    Raises:
        ValueError: If production database URL is not configured

    Note:
        - Uses INTERNAL URL when running on Render (faster)
        - Uses EXTERNAL URL when running locally
    """
    is_render = os.getenv('RENDER', '').lower() == 'true'

    if is_render:
        prod_url = os.getenv('RENDER_INTERNAL_DB_URL')
    else:
        prod_url = os.getenv('RENDER_EXTERNAL_DB_URL')

    if not prod_url:
        raise ValueError(
            "Production database URL not configured. "
            "Check RENDER_INTERNAL_DB_URL or RENDER_EXTERNAL_DB_URL"
        )

    logging.info(f"Production database URL obtained for data copy operation")
    return prod_url


def is_local_environment() -> bool:
    """
    Check if code is running in local development environment.

    Returns:
        bool: True if running locally, False if on Render
    """
    return os.getenv('RENDER', '').lower() != 'true'


def is_production_target() -> bool:
    """
    Check if the target database is production.

    Returns:
        bool: True if targeting production database

    Warning:
        Use this to add extra safety checks before writing to production.
    """
    target = os.getenv('DATABASE_TARGET', '').lower().strip()
    is_render = os.getenv('RENDER', '').lower() == 'true'

    # If no explicit target, infer
    if not target:
        target = 'render_prod' if is_render else 'local'

    return target == 'render_prod'


def get_environment_summary() -> dict:
    """
    Get a summary of the current environment configuration.

    Returns:
        dict: Environment details including target, location, and safety info

    Useful for:
        - Debugging configuration issues
        - Logging at application startup
        - Pre-flight checks before dangerous operations
    """
    target = os.getenv('DATABASE_TARGET', '').lower().strip()
    is_render = os.getenv('RENDER', '').lower() == 'true'

    if not target:
        target = 'render_prod' if is_render else 'local'
        inferred = True
    else:
        inferred = False

    return {
        'database_target': target,
        'target_inferred': inferred,
        'running_on_render': is_render,
        'is_production': target == 'render_prod',
        'is_local': target == 'local',
        'is_dev': target == 'render_dev',
    }

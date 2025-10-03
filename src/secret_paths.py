"""
secret_paths.py

Utility module for resolving secret file paths in both local and Render environments.

Priority order for finding auth credentials:
1. Database (auth_storage table) - syncs across all environments
2. Render Secret Files (/etc/secrets/) - for small files on Render
3. Local filesystem - for development

In local development:
- Checks database first, then falls back to local files
- Writes auth files to database when they're updated

In Render (production):
- Checks database first (always in sync via db copy)
- Falls back to /etc/secrets/ for small files
- No filesystem access needed

Usage:
    from secret_paths import get_auth_file

    auth_file = get_auth_file('facebook')
    # Returns temp file path with contents from database or filesystem
"""

import os
import json
import tempfile
import logging
from typing import Optional

def get_secret_path(filename: str, local_path: str = None) -> str:
    """
    Get the path to a secret file, checking Render's secret directory first.

    Args:
        filename: Name of the secret file (e.g., 'facebook_auth.json')
        local_path: Optional local path override (from env var). If not provided,
                   defaults to current directory

    Returns:
        Full path to the secret file

    Examples:
        # Simple usage - checks /etc/secrets/ on Render, ./ locally
        get_secret_path('facebook_auth.json')

        # With custom local path from environment variable
        local = os.getenv("GMAIL_CLIENT_SECRET_PATH")
        get_secret_path('desktop_client_secret.json', local)
    """
    # Render mounts Secret Files to /etc/secrets/
    render_path = f"/etc/secrets/{filename}"

    # Check if we're on Render (secret file exists in /etc/secrets/)
    if os.path.exists(render_path):
        logging.info(f"Using Render secret file: {render_path}")
        return render_path

    # Use provided local path or default to current directory
    fallback_path = local_path if local_path else filename

    if os.path.exists(fallback_path):
        logging.info(f"Using local secret file: {fallback_path}")
    else:
        logging.warning(f"Secret file not found at {render_path} or {fallback_path}")

    return fallback_path


def _get_db_connection():
    """Get database connection using centralized db_config."""
    try:
        from db_config import get_database_config
        from sqlalchemy import create_engine
        connection_string, _ = get_database_config()
        return create_engine(connection_string)
    except Exception as e:
        logging.warning(f"Could not connect to database for auth storage: {e}")
        return None


def _get_auth_from_db(service: str) -> Optional[dict]:
    """
    Retrieve auth data from database auth_storage table.

    Args:
        service: Service name (e.g., 'facebook', 'eventbrite')

    Returns:
        Auth JSON data as dict, or None if not found
    """
    engine = _get_db_connection()
    if not engine:
        return None

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT auth_data FROM auth_storage WHERE service_name = :service"),
                {"service": service}
            ).fetchone()
            if result:
                logging.info(f"Retrieved auth for '{service}' from database")
                return result[0]  # JSONB returns as dict
    except Exception as e:
        logging.warning(f"Error retrieving auth from database for '{service}': {e}")

    return None


def _write_temp_auth_file(auth_data: dict, service: str) -> str:
    """
    Write auth data to a temporary file.

    Args:
        auth_data: Auth JSON data as dict
        service: Service name for temp file naming

    Returns:
        Path to temporary file
    """
    # Create temp file in system temp directory
    fd, temp_path = tempfile.mkstemp(suffix=f'_{service}_auth.json', text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(auth_data, f, indent=2)
        logging.info(f"Wrote auth data for '{service}' to temp file: {temp_path}")
        return temp_path
    except Exception as e:
        logging.error(f"Error writing temp auth file for '{service}': {e}")
        os.close(fd)
        raise


def get_auth_file(service: str) -> str:
    """
    Get authentication file path for a specific service.

    Priority order:
    1. Database (auth_storage table)
    2. /etc/secrets/ (Render)
    3. Local filesystem

    Args:
        service: Service name (e.g., 'facebook', 'eventbrite', 'google')

    Returns:
        Full path to the auth file (may be temporary file if from database)

    Examples:
        get_auth_file('facebook')  # Returns path to facebook_auth.json
        get_auth_file('eventbrite')  # Returns path to eventbrite_auth.json
    """
    service = service.lower()

    # 1. Try database first
    auth_data = _get_auth_from_db(service)
    if auth_data:
        return _write_temp_auth_file(auth_data, service)

    # 2. Fall back to filesystem
    filename = f"{service}_auth.json"
    return get_secret_path(filename)


def is_render_environment() -> bool:
    """
    Check if code is running in Render environment.

    Returns:
        True if running on Render, False otherwise
    """
    # Render sets RENDER=true environment variable
    return os.getenv('RENDER') == 'true'

"""
secret_paths.py

Utility module for resolving secret file paths in both local and Render environments.

In local development:
- Reads from current directory (e.g., './facebook_auth.json')
- Uses paths from .env file for Google credentials

In Render (production):
- Reads from /etc/secrets/ directory where Render mounts Secret Files
- Falls back to local paths if Render paths don't exist

Usage:
    from secret_paths import get_secret_path

    auth_file = get_secret_path('facebook_auth.json')
    # Returns '/etc/secrets/facebook_auth.json' on Render
    # Returns './facebook_auth.json' locally
"""

import os
import logging

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


def get_auth_file(service: str) -> str:
    """
    Get authentication file path for a specific service.

    Args:
        service: Service name (e.g., 'facebook', 'eventbrite', 'google')

    Returns:
        Full path to the auth file

    Examples:
        get_auth_file('facebook')  # Returns path to facebook_auth.json
        get_auth_file('eventbrite')  # Returns path to eventbrite_auth.json
    """
    filename = f"{service.lower()}_auth.json"
    return get_secret_path(filename)


def is_render_environment() -> bool:
    """
    Check if code is running in Render environment.

    Returns:
        True if running on Render, False otherwise
    """
    # Render sets RENDER=true environment variable
    return os.getenv('RENDER') == 'true'

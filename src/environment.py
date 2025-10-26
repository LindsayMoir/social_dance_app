"""
Environment configuration constants.

Centralizes environment detection to avoid repeated os.getenv() calls throughout
the codebase. This module is loaded once at startup, improving performance and
ensuring consistent environment detection across all modules.

Usage:
    from environment import IS_RENDER, IS_LOCAL

    if IS_RENDER:
        # Render-specific code
        pass
    elif IS_LOCAL:
        # Local development code
        pass
"""

import os

# Detect environment once at module load time
IS_RENDER = os.getenv('RENDER') == 'true'
IS_LOCAL = not IS_RENDER

# Environment name for logging/debugging
ENVIRONMENT = 'Render' if IS_RENDER else 'Local'

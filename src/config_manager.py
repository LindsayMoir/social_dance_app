"""
Centralized configuration management.

Provides singleton access to application configuration with:
- Single load point (config.yaml loaded once)
- Validation of required keys
- Database configuration management
- Backward compatible access patterns
"""
import os
import yaml
import logging
from typing import Any, Optional, Dict, Tuple


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigManager:
    """
    Singleton configuration manager.

    Ensures configuration is loaded once and provides consistent access
    across the application.

    Usage:
        # Get instance
        config = ConfigManager.get_instance().config

        # Get specific key
        keywords = ConfigManager.get('keywords', default=[])

        # Access nested config
        llm_config = ConfigManager.get('llm')
    """

    _instance = None
    _config = None

    @staticmethod
    def get_instance() -> 'ConfigManager':
        """
        Get or create singleton instance.
        Config is loaded only once on first access.

        Returns:
            ConfigManager: Singleton instance
        """
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance

    def __init__(self):
        """Initialize and load configuration."""
        if ConfigManager._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Looks for config.yaml in the config/ directory relative to project root.

        Raises:
            ConfigError: If config file not found or YAML parsing fails
        """
        try:
            # Compute path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'config.yaml')

            if not os.path.exists(config_path):
                raise ConfigError(f"Config file not found: {config_path}")

            logging.info(f"Loading configuration from: {config_path}")

            with open(config_path, 'r') as f:
                ConfigManager._config = yaml.safe_load(f)

            if not ConfigManager._config:
                raise ConfigError("Config file is empty or invalid YAML")

            logging.info("Configuration loaded successfully")

        except FileNotFoundError as e:
            raise ConfigError(f"Config file not found: {e}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.

        Returns:
            dict: Configuration loaded from config.yaml
        """
        if ConfigManager._config is None:
            self._load_config()
        return ConfigManager._config

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Supports nested keys with dot notation:
            ConfigManager.get('llm.provider') → returns config['llm']['provider']

        Args:
            key: Configuration key (supports dots for nesting)
            default: Default value if key not found

        Returns:
            Configuration value or default if not found

        Example:
            >>> ConfigManager.get('keywords', default=[])
            ['dance', 'music', ...]
        """
        instance = ConfigManager.get_instance()
        config = instance.config

        # Support dot notation for nested access
        if '.' in key:
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value if value is not None else default

        return config.get(key, default)

    @staticmethod
    def reload() -> None:
        """
        Force reload of configuration.

        Useful for testing or if config changes at runtime.
        """
        ConfigManager._config = None
        ConfigManager.get_instance()
        logging.info("Configuration reloaded")

    @staticmethod
    def validate_required(required_keys: list) -> bool:
        """
        Validate that required configuration keys exist.

        Args:
            required_keys: List of keys that must exist

        Returns:
            bool: True if all keys exist

        Raises:
            ConfigError: If any required key is missing
        """
        instance = ConfigManager.get_instance()
        config = instance.config

        missing = []
        for key in required_keys:
            if key not in config:
                missing.append(key)

        if missing:
            raise ConfigError(f"Missing required config keys: {missing}")

        return True

    @staticmethod
    def _is_render_platform() -> bool:
        """
        Detect if code is running on Render platform.

        Checks both explicit RENDER env var and auto-detection via Render-specific vars.

        Returns:
            bool: True if running on Render, False if local
        """
        try:
            from environment import IS_RENDER
            is_render = IS_RENDER
        except ImportError:
            is_render = False

        return (
            is_render or
            os.getenv('RENDER_SERVICE_NAME') is not None or
            os.getenv('RENDER_INSTANCE_ID') is not None or
            'render.com' in os.getenv('HOSTNAME', '')
        )

    @staticmethod
    def get_database_config() -> Tuple[str, str]:
        """
        Determine which database to connect to based on environment variables.

        Consolidated from db_config.py for centralized configuration management.

        Returns:
            Tuple[str, str]: (connection_string, environment_name)

        Raises:
            ConfigError: If DATABASE_TARGET is invalid or connection string is missing

        Environment Variables Used:
            - DATABASE_TARGET: 'local', 'render_dev', or 'render_prod'
            - RENDER: 'true' if running on Render, otherwise unset/false

        Logic:
            1. If DATABASE_TARGET is explicitly set, use that
            2. Otherwise, infer from RENDER environment variable
            3. Choose INTERNAL vs EXTERNAL URL based on RENDER variable

        Example:
            >>> conn_str, env_name = ConfigManager.get_database_config()
            >>> print(env_name)
            'Local PostgreSQL (localhost)'
        """
        is_render = ConfigManager._is_render_platform()

        # Get explicit target or auto-detect
        target = os.getenv('DATABASE_TARGET', '').lower().strip()

        if not target:
            # Auto-detect based on platform
            if is_render:
                target = 'render_dev'
                logging.info("Auto-detected: Running on Render → DATABASE_TARGET='render_dev'")
            else:
                target = 'local'
                logging.info("Auto-detected: Running locally → DATABASE_TARGET='local'")
        else:
            logging.info(f"DATABASE_TARGET explicitly set: {target}")

        # Map target to connection string
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
            raise ConfigError(
                f"Invalid DATABASE_TARGET: '{target}'. "
                f"Must be one of: local, render_dev, render_prod"
            )

        connection_string, env_name = connection_map[target]

        # Validate connection string exists
        if not connection_string:
            raise ConfigError(
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

    @staticmethod
    def get_production_database_url() -> str:
        """
        Get the production database URL for copying data to production.

        This is always the production database, regardless of where code is running
        or what DATABASE_TARGET is set to. Used by the final step of pipeline.py
        to copy data from dev/local to production.

        Returns:
            str: Production database connection string

        Raises:
            ConfigError: If production database URL is not configured

        Note:
            - Uses INTERNAL URL when running on Render (faster)
            - Uses EXTERNAL URL when running locally
        """
        is_render = ConfigManager._is_render_platform()

        if is_render:
            prod_url = os.getenv('RENDER_INTERNAL_DB_URL')
        else:
            prod_url = os.getenv('RENDER_EXTERNAL_DB_URL')

        if not prod_url:
            raise ConfigError(
                "Production database URL not configured. "
                "Check RENDER_INTERNAL_DB_URL or RENDER_EXTERNAL_DB_URL"
            )

        logging.info(f"Production database URL obtained for data copy operation")
        return prod_url

    @staticmethod
    def is_local_environment() -> bool:
        """
        Check if code is running in local development environment.

        Returns:
            bool: True if running locally, False if on Render
        """
        return not ConfigManager._is_render_platform()

    @staticmethod
    def is_production_target() -> bool:
        """
        Check if the target database is production.

        Returns:
            bool: True if targeting production database

        Warning:
            Use this to add extra safety checks before writing to production.
        """
        target = os.getenv('DATABASE_TARGET', '').lower().strip()
        is_render = ConfigManager._is_render_platform()

        # If no explicit target, infer
        if not target:
            target = 'render_prod' if is_render else 'local'

        return target == 'render_prod'

    @staticmethod
    def get_environment_summary() -> dict:
        """
        Get a summary of the current environment configuration.

        Returns:
            dict: Environment details including target, location, and safety info

        Useful for:
            - Debugging configuration issues
            - Logging at application startup
            - Pre-flight checks before dangerous operations

        Example:
            >>> summary = ConfigManager.get_environment_summary()
            >>> print(f"Is production: {summary['is_production']}")
        """
        target = os.getenv('DATABASE_TARGET', '').lower().strip()
        is_render = ConfigManager._is_render_platform()

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

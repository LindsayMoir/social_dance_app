"""
Centralized configuration management.

Provides singleton access to application configuration with:
- Single load point (config.yaml loaded once)
- Validation of required keys
- Backward compatible access patterns
"""
import os
import yaml
import logging
from typing import Any, Optional, Dict


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

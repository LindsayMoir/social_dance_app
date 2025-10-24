#!/usr/bin/env python3
"""
deployment_config.py - Deployment Configuration Management

Provides environment-aware configuration loading and validation for different
deployment scenarios:
- Local development
- Staging environment
- Production environment
- Docker containers

Features:
- Environment-based config loading
- Configuration validation
- Default values with overrides
- Sensitive data handling (database credentials, API keys)
- Deployment-specific optimizations
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class DeploymentEnvironment(Enum):
    """Supported deployment environments."""
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    DOCKER = "docker"


class DeploymentConfig:
    """
    Centralized deployment configuration management.

    Handles loading and validating configuration for different environments
    with support for environment variables, config files, and defaults.
    """

    # Default configuration values
    DEFAULTS = {
        'environment': 'local',
        'debug': True,
        'log_level': 'DEBUG',
        'log_format': 'detailed',
        'headless': False,
        'crawling': {
            'headless': False,
            'scroll_depth': 5,
            'urls_run_limit': 100,
            'prompt_max_length': 8000,
            'timeout_seconds': 30
        },
        'database': {
            'type': 'sqlite',
            'path': 'data/social_dance.db'
        },
        'monitoring': {
            'enabled': False,
            'metrics_port': 8000,
            'health_check_interval': 60
        },
        'performance': {
            'max_workers': 4,
            'batch_size': 100,
            'cache_size_mb': 100,
            'timeout_seconds': 300
        }
    }

    def __init__(self, environment: Optional[str] = None):
        """
        Initialize deployment configuration.

        Args:
            environment: Deployment environment name
                        (local, staging, production, docker)
                        If None, loads from DEPLOYMENT_ENV environment variable
        """
        self.logger = logging.getLogger(__name__)
        self.environment = environment or os.getenv('DEPLOYMENT_ENV', 'local')
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration for the current environment.

        Priority (highest to lowest):
        1. Environment variables (prefixed with SCRAPER_)
        2. Environment-specific config file
        3. Config/config.yaml (existing)
        4. Defaults defined in DEFAULTS

        Returns:
            Configuration dictionary
        """
        config = self.DEFAULTS.copy()

        # Load from config files
        config_dir = Path('config')
        base_config_file = config_dir / 'config.yaml'
        env_config_file = config_dir / f'config.{self.environment}.yaml'

        if base_config_file.exists():
            config.update(self._load_yaml(base_config_file))
            self.logger.info(f"Loaded base configuration from {base_config_file}")

        if env_config_file.exists():
            config.update(self._load_yaml(env_config_file))
            self.logger.info(f"Loaded {self.environment} configuration from {env_config_file}")

        # Override with environment variables (SCRAPER_ prefix)
        config.update(self._load_from_env())

        return config

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except ImportError:
            self.logger.warning("PyYAML not available, skipping YAML config")
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load {filepath}: {e}")
            return {}

    def _load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Expected format: SCRAPER_KEY=value
        For nested keys: SCRAPER_SECTION__SUBSECTION=value
        """
        config = {}
        prefix = 'SCRAPER_'

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()

                # Handle nested keys (SCRAPER_DB__HOST -> config['db']['host'])
                if '__' in config_key:
                    parts = config_key.split('__')
                    current = config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_value(value)
                else:
                    config[config_key] = self._parse_value(value)

        return config

    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Converts:
        - 'true'/'false' to boolean
        - numeric strings to int/float
        - JSON objects/arrays to dict/list
        - other strings remain as-is
        """
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        if value.lower() in ('false', 'no', 'off', '0'):
            return False

        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Try to parse as JSON
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value

    def _validate_config(self):
        """Validate configuration for the current environment."""
        self.logger.info(f"Deployment environment: {self.environment}")

        # Environment-specific validation
        if self.environment == 'production':
            self._validate_production()
        elif self.environment == 'staging':
            self._validate_staging()
        elif self.environment == 'docker':
            self._validate_docker()

    def _validate_production(self):
        """Validate production-specific configuration."""
        if self.config.get('debug'):
            self.logger.warning("⚠️  DEBUG mode enabled in production")

        if self.config.get('log_level') == 'DEBUG':
            self.logger.warning("⚠️  Log level is DEBUG in production")

        if self.config.get('headless') is False:
            self.logger.warning("⚠️  Headless mode disabled in production")

        # Check for required production settings
        if not self.config.get('database', {}).get('password'):
            self.logger.warning("⚠️  Database password not set")

    def _validate_staging(self):
        """Validate staging-specific configuration."""
        self.logger.info("✓ Staging environment validated")

    def _validate_docker(self):
        """Validate Docker-specific configuration."""
        # Docker should use environment variables for secrets
        if not os.getenv('DATABASE_PASSWORD'):
            self.logger.warning("⚠️  DATABASE_PASSWORD not set in Docker")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports dot notation for nested access: 'database.host'

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self.config.get(key, default)

        # Handle nested keys
        parts = key.split('.')
        current = self.config
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default

        return current

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration for this environment.

        Returns:
            Dictionary with logging setup parameters
        """
        return {
            'level': self.config.get('log_level', 'INFO'),
            'format': self.config.get('log_format', 'detailed'),
            'debug': self.config.get('debug', False),
            'environment': self.environment
        }

    def get_scraper_config(self) -> Dict[str, Any]:
        """
        Get scraper-specific configuration.

        Returns:
            Dictionary with scraper parameters
        """
        return {
            'headless': self.config.get('crawling', {}).get('headless', True),
            'scroll_depth': self.config.get('crawling', {}).get('scroll_depth', 5),
            'urls_run_limit': self.config.get('crawling', {}).get('urls_run_limit', 100),
            'timeout_seconds': self.config.get('crawling', {}).get('timeout_seconds', 30)
        }

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database-specific configuration.

        Returns:
            Dictionary with database connection parameters
        """
        db_config = self.config.get('database', {})
        return {
            'type': db_config.get('type', 'sqlite'),
            'host': db_config.get('host', 'localhost'),
            'port': db_config.get('port', 5432),
            'name': db_config.get('name', 'social_dance'),
            'user': db_config.get('user', ''),
            'password': db_config.get('password', ''),
            'path': db_config.get('path', 'data/social_dance.db')
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring and health check configuration.

        Returns:
            Dictionary with monitoring parameters
        """
        return {
            'enabled': self.config.get('monitoring', {}).get('enabled', False),
            'metrics_port': self.config.get('monitoring', {}).get('metrics_port', 8000),
            'health_check_interval': self.config.get('monitoring', {}).get('health_check_interval', 60)
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance tuning configuration.

        Returns:
            Dictionary with performance parameters
        """
        return {
            'max_workers': self.config.get('performance', {}).get('max_workers', 4),
            'batch_size': self.config.get('performance', {}).get('batch_size', 100),
            'cache_size_mb': self.config.get('performance', {}).get('cache_size_mb', 100),
            'timeout_seconds': self.config.get('performance', {}).get('timeout_seconds', 300)
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.

        Note: Excludes sensitive data (passwords, API keys)

        Returns:
            Configuration dictionary (sanitized)
        """
        config = json.loads(json.dumps(self.config))  # Deep copy

        # Sanitize sensitive fields
        sensitive_keys = ['password', 'token', 'api_key', 'secret', 'credential']
        self._sanitize_dict(config, sensitive_keys)

        return config

    @staticmethod
    def _sanitize_dict(data: Dict[str, Any], sensitive_keys: list):
        """Recursively sanitize sensitive keys in configuration dictionary."""
        for key, value in list(data.items()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                data[key] = '***REDACTED***'
            elif isinstance(value, dict):
                DeploymentConfig._sanitize_dict(value, sensitive_keys)

    def __repr__(self) -> str:
        """String representation of deployment configuration."""
        return f"DeploymentConfig(environment={self.environment})"


# Global config instance
_config_instance: Optional[DeploymentConfig] = None


def get_config(environment: Optional[str] = None) -> DeploymentConfig:
    """
    Get or create global deployment configuration instance.

    Args:
        environment: Deployment environment name (for initial creation)

    Returns:
        DeploymentConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DeploymentConfig(environment)
    return _config_instance


def set_config(config: DeploymentConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Config: {json.dumps(config.to_dict(), indent=2)}")

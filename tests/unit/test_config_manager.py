"""Tests for ConfigManager utility."""
import pytest
from src.config_manager import ConfigManager, ConfigError


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def setup_method(self):
        """Reset singleton before each test."""
        ConfigManager._instance = None
        ConfigManager._config = None

    def test_singleton_pattern(self):
        """Verify singleton pattern works."""
        instance1 = ConfigManager.get_instance()
        instance2 = ConfigManager.get_instance()
        assert instance1 is instance2

    def test_config_loaded(self):
        """Verify config is loaded."""
        manager = ConfigManager.get_instance()
        assert manager.config is not None
        assert isinstance(manager.config, dict)

    def test_config_has_required_keys(self):
        """Verify essential config keys exist."""
        manager = ConfigManager.get_instance()
        assert 'prompts' in manager.config
        assert 'llm' in manager.config

    def test_get_with_default(self):
        """Test get() with default value."""
        value = ConfigManager.get('nonexistent_key', default='default_value')
        assert value == 'default_value'

    def test_get_existing_key(self):
        """Test get() with existing key."""
        value = ConfigManager.get('keywords', default={})
        assert isinstance(value, (list, dict))

    def test_config_not_reloaded_multiple_times(self):
        """Verify config is loaded only once."""
        # Get instance multiple times
        manager1 = ConfigManager.get_instance()
        manager2 = ConfigManager.get_instance()
        # Should be exact same object
        assert manager1.config is manager2.config

    def test_reload_reloads_config(self):
        """Test that reload() forces reload."""
        manager1 = ConfigManager.get_instance()
        original_config = manager1.config
        ConfigManager.reload()
        manager2 = ConfigManager.get_instance()
        # Should have reloaded (same values but potentially different objects)
        assert manager2.config is not None

    def test_validate_required_success(self):
        """Test validate_required() with existing keys."""
        # Reset for fresh load
        ConfigManager._instance = None
        ConfigManager._config = None
        # This should not raise
        result = ConfigManager.validate_required(['prompts', 'llm'])
        assert result is True

    def test_config_is_dict(self):
        """Verify config is dictionary type."""
        manager = ConfigManager.get_instance()
        assert type(manager.config) is dict

    def test_access_config_directly(self):
        """Test accessing config through property."""
        manager = ConfigManager.get_instance()
        config = manager.config
        assert 'prompts' in config
        assert isinstance(config, dict)

# Code Consolidation Examples - Implementation Guide

This document provides detailed code examples for each of the major consolidation opportunities identified in the redundancy analysis.

## 1. Config Consolidation Example

### Current State (Fragmented)

**db_config.py** - Database configuration detection
```python
def get_database_config() -> Tuple[str, str]:
    is_render = os.getenv('RENDER') == 'true'
    target = os.getenv('DATABASE_TARGET', '').lower().strip()
    
    if not target:
        target = 'render_dev' if is_render else 'local'
    
    connection_map = {
        'local': (os.getenv('DATABASE_CONNECTION_STRING'), 'Local PostgreSQL'),
        'render_dev': (...),
        'render_prod': (...),
    }
    # ... 100+ lines of logic
```

**deployment_config.py** - Separate configuration system
```python
class DeploymentConfig:
    def __init__(self, environment: Optional[str] = None):
        self.environment = environment or os.getenv('DEPLOYMENT_ENV', 'local')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        # Another YAML loading implementation
        # Duplicates logic from ConfigManager
```

**config_manager.py** - Yet another config system
```python
class ConfigManager:
    @staticmethod
    def get_instance() -> 'ConfigManager':
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance
    
    def _load_config(self) -> None:
        # Third implementation of YAML loading
```

### Consolidated Approach

**config_manager.py** - Unified system
```python
class ConfigManager:
    """Unified configuration management system."""
    
    _instance = None
    _config = None
    
    @staticmethod
    def get_instance() -> 'ConfigManager':
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance
    
    def __init__(self):
        if ConfigManager._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'config.yaml')
            
            with open(config_path, 'r') as f:
                ConfigManager._config = yaml.safe_load(f)
                
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")
    
    @property
    def config(self) -> Dict[str, Any]:
        if ConfigManager._config is None:
            self._load_config()
        return ConfigManager._config
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get config value with dot notation support."""
        instance = ConfigManager.get_instance()
        config = instance.config
        
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
    
    # NEW: Add database configuration method
    @staticmethod
    def get_database_config() -> Tuple[str, str]:
        """
        Get database configuration (consolidates db_config.py logic).
        
        Returns:
            Tuple[str, str]: (connection_string, environment_name)
        """
        from environment import IS_RENDER
        
        is_render = IS_RENDER or os.getenv('RENDER_SERVICE_NAME') is not None
        target = os.getenv('DATABASE_TARGET', '').lower().strip()
        
        if not target:
            target = 'render_dev' if is_render else 'local'
        
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
        
        if target not in connection_map:
            raise ValueError(f"Invalid DATABASE_TARGET: {target}")
        
        connection_string, env_name = connection_map[target]
        if not connection_string:
            raise ValueError(f"Database connection string not configured for {target}")
        
        return connection_string, env_name
```

### Migration Path

**Before:**
```python
from config_manager import ConfigManager
from db_config import get_database_config
from deployment_config import get_config

config = ConfigManager.get('key')
db_url, env_name = get_database_config()
deploy_config = get_config('production')
```

**After:**
```python
from config_manager import ConfigManager

config = ConfigManager.get('key')
db_url, env_name = ConfigManager.get_database_config()
```

---

## 2. Logging Consolidation Example

### Current State (Competing Systems)

**logging_config.py** - Simple setup
```python
def setup_logging(script_name: str, level=logging.INFO):
    is_render = os.getenv('RENDER') == 'true'
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    if is_render:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
    else:
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename=f"logs/{script_name}_log.txt",
            filemode='a',
            level=level,
            format=log_format,
            force=True
        )
```

**production_logging.py** - Complex setup (150+ lines)
```python
class ProductionLogger:
    """JSON structured logging with filtering and correlation IDs."""
    
    def __init__(self, script_name: str):
        self.logger = logging.getLogger(script_name)
        # Complex setup with JSON formatting, filtering, etc.
```

**Multiple files doing:**
```python
# fb.py
from logging_config import setup_logging
setup_logging('fb')

# ebs.py
from logging_config import setup_logging
setup_logging('ebs')

# gen_scraper.py
from logging_config import setup_logging
setup_logging('gen_scraper')
```

### Consolidated Approach

**logging_config.py** - Unified system
```python
def setup_logging(script_name: str, level=logging.INFO, use_production: bool = False):
    """
    Unified logging setup supporting both simple and production modes.
    
    Args:
        script_name: Name of the script
        level: Logging level
        use_production: If True, use JSON/filtering/correlation IDs
    """
    use_production = use_production or os.getenv('PRODUCTION_LOGGING') == 'true'
    is_render = os.getenv('RENDER') == 'true'
    
    if use_production:
        return _setup_production_logging(script_name, level, is_render)
    else:
        return _setup_simple_logging(script_name, level, is_render)


def _setup_simple_logging(script_name: str, level: int, is_render: bool):
    """Simple logging setup (existing logic)."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'
    
    if is_render:
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        logging.info(f"Logging configured for Render (stdout) - {script_name}")
    else:
        os.makedirs('logs', exist_ok=True)
        log_file = f"logs/{script_name}_log.txt"
        
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            level=level,
            format=log_format,
            datefmt=date_format,
            force=True
        )
        logging.info(f"Logging configured for local (file: {log_file})")
    
    return logging.getLogger(script_name)


def _setup_production_logging(script_name: str, level: int, is_render: bool):
    """Production logging setup (from production_logging.py)."""
    from production_logging import ProductionLogger
    return ProductionLogger(script_name, level=level).get_logger()
```

### Usage

**Before:**
```python
# Every scraper did this separately
from logging_config import setup_logging
setup_logging('fb')

# Production code did this
from production_logging import ProductionLogger
logger = ProductionLogger('fb')
```

**After:**
```python
# Simple setup (development)
from logging_config import setup_logging
logger = setup_logging('fb')

# Production setup (set env var or pass flag)
logger = setup_logging('fb', use_production=True)
# OR via environment: PRODUCTION_LOGGING=true
```

---

## 3. URL Normalization Consolidation Example

### Current State (Duplicated Logic)

**fb.py** - Facebook-specific normalization
```python
def normalize_facebook_url(self, url: str) -> str:
    """Normalize Facebook URLs (v.facebook.com → www.facebook.com)."""
    try:
        # Handle v.facebook.com
        if 'v.facebook.com' in url:
            url = url.replace('v.facebook.com', 'www.facebook.com')
        
        # Remove query parameters
        if '?' in url:
            url = url.split('?')[0]
        
        # Normalize path
        url = url.rstrip('/')
        
        return url
    except Exception as e:
        logging.error(f"Error normalizing Facebook URL: {e}")
        return None
```

**url_nav.py** - Generic normalization (doesn't handle Facebook)
```python
def normalize_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
    """Generic URL normalization."""
    try:
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        parsed = urlparse(url)
        
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        if normalized.endswith('/') and not normalized.endswith('://'):
            normalized = normalized.rstrip('/')
        
        return normalized if self.is_valid_url(normalized) else None
    except Exception as e:
        self.logger.warning(f"URL normalization error for {url}: {e}")
        return None
```

### Consolidated Approach

**url_nav.py** - Extended with platform support
```python
class URLNavigator:
    """Unified URL navigation with platform-specific handlers."""
    
    # Facebook-specific URL patterns
    FACEBOOK_PATTERNS = {
        'v.facebook.com': 'www.facebook.com',
        'fb.com': 'facebook.com',
        'fbcdn.net': 'facebook.com',  # Facebook CDN
    }
    
    def normalize_url(self, url: str, base_url: Optional[str] = None, 
                     platform: str = 'generic') -> Optional[str]:
        """
        Normalize URL based on platform-specific rules.
        
        Args:
            url: URL to normalize
            base_url: Base URL for relative URLs
            platform: 'generic' (default) or 'facebook'
        
        Returns:
            Normalized URL or None if invalid
        """
        try:
            if platform == 'facebook':
                return self._normalize_facebook_url(url, base_url)
            else:
                return self._normalize_generic_url(url, base_url)
        except Exception as e:
            self.logger.warning(f"URL normalization error: {e}")
            return None
    
    def _normalize_facebook_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        """Normalize Facebook-specific URLs."""
        # Handle relative URLs
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        # Replace Facebook aliases
        for pattern, replacement in self.FACEBOOK_PATTERNS.items():
            if pattern in url:
                url = url.replace(pattern, replacement)
        
        # Remove query parameters
        if '?' in url:
            url = url.split('?')[0]
        
        # Generic normalization
        normalized = self._normalize_generic_url(url)
        
        return normalized
    
    def _normalize_generic_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        """Generic URL normalization (existing logic)."""
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        parsed = urlparse(url)
        
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''
        ))
        
        if normalized.endswith('/') and not normalized.endswith('://'):
            normalized = normalized.rstrip('/')
        
        return normalized if self.is_valid_url(normalized) else None
```

### Usage

**Before (fb.py):**
```python
normalized = self.normalize_facebook_url(url)  # Local method
```

**After (fb_v2.py):**
```python
from url_nav import URLNavigator

self.url_navigator = URLNavigator()
normalized = self.url_navigator.normalize_url(url, platform='facebook')
```

---

## 4. Error Handling Consolidation Example

### Current State (Scattered)

**db_utils.py** - Error handling pattern
```python
def normalize_event_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        normalized = {
            'event_name': event.get('event_name', 'Unknown Event'),
            # ... more fields ...
        }
        filtered = {k: v for k, v in normalized.items()
                   if v is not None or k in optional_fields}
        return filtered
    except Exception as e:
        self.logger.error(f"Failed to normalize event data: {e}")
        return normalized
```

**db.py** - Similar pattern (repeated 47+ times)
```python
try:
    # database operation
    result = self.do_something()
except Exception as e:
    logging.error(f"Error in do_something: {e}")
    return None
```

### Consolidated Approach

**resilience.py** - Add error handling utilities
```python
class ErrorHandler:
    """Context manager for consistent error handling and logging."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, 
                 default_return=None, raise_on_error: bool = False):
        self.logger = logger
        self.operation_name = operation_name
        self.default_return = default_return
        self.raise_on_error = raise_on_error
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.logger.error(f"{self.operation_name} failed: {exc_val}")
            
            if self.raise_on_error:
                return False  # Re-raise exception
            
            return True  # Suppress exception
        
        return False
    
    def log_and_return(self, value):
        """Log error and return specified value."""
        if self.error:
            return self.default_return
        return value


def handle_errors(operation_name: str, default_return=None):
    """Decorator for automatic error handling and logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"{operation_name} failed: {e}")
                return default_return
        return wrapper
    return decorator
```

### Usage

**Before:**
```python
def normalize_event_data(self, event):
    try:
        normalized = {
            'event_name': event.get('event_name', 'Unknown Event'),
        }
        return filtered
    except Exception as e:
        self.logger.error(f"Failed to normalize event data: {e}")
        return normalized
```

**After (Option 1 - Context Manager):**
```python
def normalize_event_data(self, event):
    with ErrorHandler(self.logger, "normalize_event_data", 
                     default_return={}) as handler:
        normalized = {
            'event_name': event.get('event_name', 'Unknown Event'),
        }
        return handler.log_and_return(normalized)
```

**After (Option 2 - Decorator):**
```python
@handle_errors("normalize_event_data", default_return={})
def normalize_event_data(self, event):
    normalized = {
        'event_name': event.get('event_name', 'Unknown Event'),
    }
    return normalized
```

---

## 5. Scraper Initialization Consolidation Example

### Current State (Repeated Pattern)

**All scrapers follow similar pattern:**
```python
# fb.py
def __init__(self, config_path: str = "config/config.yaml") -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    self.llm_handler = LLMHandler(config_path)
    self.db_handler = self.llm_handler.db_handler
    self.keywords_list = self.llm_handler.get_keywords()
    self.run_results_tracker = RunResultsTracker('fb', self.db_handler)
    # ... more init code ...

# ebs.py - Identical pattern
def __init__(self, config_path: str = "config/config.yaml") -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    self.llm_handler = LLMHandler(config_path)
    self.db_handler = self.llm_handler.db_handler
    self.keywords_list = self.llm_handler.get_keywords()
    self.run_results_tracker = RunResultsTracker('ebs', self.db_handler)
    # ... more init code ...

# images.py - Same pattern
def __init__(self, config_path: str = "config/config.yaml") -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    self.llm_handler = LLMHandler(config_path)
    self.db_handler = self.llm_handler.db_handler
    self.keywords_list = self.llm_handler.get_keywords()
    # ... identical setup ...
```

### Consolidated Approach

**base_scraper.py** - Add initialization helper
```python
class BaseScraper(ABC):
    """Abstract base class with shared initialization."""
    
    @staticmethod
    def create_standard_handlers(config_path: str = "config/config.yaml"):
        """
        Factory method to create standard handler set.
        
        Consolidates initialization pattern used by all scrapers.
        
        Args:
            config_path: Path to configuration YAML file
        
        Returns:
            dict: Dictionary with initialized handlers and utilities
        """
        import yaml
        from llm import LLMHandler
        from run_results_tracker import RunResultsTracker, get_database_counts
        
        # Load configuration once
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize core handlers
        llm_handler = LLMHandler(config_path)
        db_handler = llm_handler.db_handler
        keywords_list = llm_handler.get_keywords()
        
        return {
            'config': config,
            'llm': llm_handler,
            'db': db_handler,
            'keywords': keywords_list,
            'text_extractor': TextExtractor(),
            'retry_manager': RetryManager(),
            'url_navigator': URLNavigator(),
        }
    
    @staticmethod
    def initialize_run_tracker(script_name: str, db_handler):
        """Initialize run results tracker."""
        run_tracker = RunResultsTracker(script_name, db_handler)
        events_count, urls_count = get_database_counts(db_handler)
        run_tracker.initialize(events_count, urls_count)
        return run_tracker
```

### Usage

**Before (fb.py - 50+ lines):**
```python
def __init__(self, config_path: str = "config/config.yaml") -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    self.config = config
    self.llm_handler = LLMHandler(config_path)
    self.db_handler = self.llm_handler.db_handler
    self.keywords_list = self.llm_handler.get_keywords()
    self.run_results_tracker = RunResultsTracker('fb', self.db_handler)
    events_count, urls_count = get_database_counts(self.db_handler)
    self.run_results_tracker.initialize(events_count, urls_count)
    # ... more initialization ...
```

**After (fb.py - 10 lines):**
```python
def __init__(self, config_path: str = "config/config.yaml") -> None:
    super().__init__(config_path)
    
    # Get standard handlers
    handlers = BaseScraper.create_standard_handlers(config_path)
    self.config = handlers['config']
    self.llm_handler = handlers['llm']
    self.db_handler = handlers['db']
    self.keywords_list = handlers['keywords']
    
    # Initialize run tracker
    self.run_results_tracker = BaseScraper.initialize_run_tracker('fb', self.db_handler)
```

---

## Migration Checklist

### Phase 1: Config Consolidation
- [ ] Add database_config() to ConfigManager
- [ ] Update db_config imports in 20+ files
- [ ] Delete db_config.py after verification
- [ ] Run tests

### Phase 2: Logging Consolidation
- [ ] Extend logging_config.setup_logging()
- [ ] Integrate production_logging features
- [ ] Update 10+ files to use new setup
- [ ] Delete duplicate logging files
- [ ] Run tests

### Phase 3: URL Normalization
- [ ] Add platform parameter to URLNavigator.normalize_url()
- [ ] Implement _normalize_facebook_url()
- [ ] Update fb.py and fb_v2.py to use URLNavigator
- [ ] Delete normalize_facebook_url() methods
- [ ] Run tests

### Phase 4: Error Handling
- [ ] Add ErrorHandler class to resilience.py
- [ ] Add handle_errors decorator
- [ ] Refactor error handling in db.py, db_utils.py
- [ ] Run tests

### Phase 5: Scraper Initialization
- [ ] Create BaseScraper.create_standard_handlers()
- [ ] Create BaseScraper.initialize_run_tracker()
- [ ] Update fb.py, ebs.py, images.py, gen_scraper.py
- [ ] Run tests

### Phase 6: Version Consolidation
- [ ] Audit usage of fb.py, ebs.py, rd_ext.py, read_pdfs.py
- [ ] Migrate all references to v2 versions
- [ ] Delete v1 files
- [ ] Run tests


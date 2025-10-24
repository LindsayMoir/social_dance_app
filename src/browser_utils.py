"""
Browser utilities for unified Playwright management across scrapers.

This module consolidates browser initialization, context management, and
navigation patterns used across multiple scrapers (fb.py, rd_ext.py, images.py, ebs.py).

Classes:
    PlaywrightManager: Centralized Playwright browser and context management

Key responsibilities:
    - Centralized browser launch configuration
    - Context and page creation with consistent settings
    - Header/User-Agent standardization
    - Viewport and timeout management
    - Anti-bot delay generation
    - Safe navigation with error handling
"""

import logging
import random
import time
from typing import Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright, AsyncBrowser, AsyncBrowserContext, AsyncPage


class PlaywrightManager:
    """
    Centralized manager for Playwright browser initialization and operations.

    Consolidates browser launch, context creation, navigation, and cleanup
    patterns that are repeated across all scrapers.
    """

    # Standard headers used across all scrapers
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/100.0.0.0 Safari/537.36"
    )

    # Standard viewport settings
    DEFAULT_VIEWPORT = {"width": 1920, "height": 1080}

    # Standard timeouts (in ms)
    DEFAULT_NAVIGATION_TIMEOUT = 20000  # 20 seconds
    DEFAULT_WAIT_TIMEOUT = 10000  # 10 seconds

    # Anti-bot delays
    MIN_DELAY = 2
    MAX_DELAY = 5

    def __init__(self, config: dict, headless: bool = True):
        """
        Initialize PlaywrightManager.

        Args:
            config (dict): Configuration dictionary containing crawling settings
            headless (bool): Whether to run browser in headless mode. Defaults to True.
        """
        self.config = config
        self.headless = config.get('crawling', {}).get('headless', headless)
        self.logger = logging.getLogger(__name__)

    def get_headers(self) -> dict:
        """
        Get standardized headers for HTTP requests.

        Returns:
            dict: Headers with standard User-Agent and common headers
        """
        return {
            'User-Agent': self.DEFAULT_USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def get_viewport_size(self) -> dict:
        """
        Get standardized viewport size for browser pages.

        Returns:
            dict: Viewport configuration with width and height
        """
        return self.DEFAULT_VIEWPORT.copy()

    def get_timeout(self, timeout_type: str = 'navigation') -> int:
        """
        Get standardized timeout value based on operation type.

        Args:
            timeout_type (str): Type of timeout ('navigation' or 'wait'). Defaults to 'navigation'.

        Returns:
            int: Timeout in milliseconds with random variation to avoid detection
        """
        base_timeout = self.DEFAULT_NAVIGATION_TIMEOUT if timeout_type == 'navigation' else self.DEFAULT_WAIT_TIMEOUT
        # Add ±50% variation to avoid consistent timing detection
        variation = base_timeout * 0.5
        return int(base_timeout + random.uniform(-variation, variation))

    def get_random_delay(self) -> float:
        """
        Get random delay for anti-bot behavior.

        Returns:
            float: Delay in seconds
        """
        return random.uniform(self.MIN_DELAY, self.MAX_DELAY)

    def apply_delay(self, min_delay: Optional[float] = None, max_delay: Optional[float] = None) -> None:
        """
        Apply random delay with optional custom range.

        Args:
            min_delay (float, optional): Minimum delay in seconds
            max_delay (float, optional): Maximum delay in seconds
        """
        min_delay = min_delay or self.MIN_DELAY
        max_delay = max_delay or self.MAX_DELAY
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)

    def create_browser_context(self, browser: Browser, storage_state: Optional[str] = None,
                               extra_http_headers: Optional[dict] = None) -> Tuple[BrowserContext, Page]:
        """
        Create a browser context with standard settings.

        Args:
            browser (Browser): Playwright Browser instance
            storage_state (str, optional): Path to storage state file (for cookies/auth)
            extra_http_headers (dict, optional): Additional HTTP headers

        Returns:
            Tuple[BrowserContext, Page]: Created context and page
        """
        context_kwargs = {
            'viewport': self.get_viewport_size(),
            'user_agent': self.DEFAULT_USER_AGENT,
        }

        if storage_state:
            context_kwargs['storage_state'] = storage_state

        if extra_http_headers:
            context_kwargs['extra_http_headers'] = extra_http_headers
        else:
            context_kwargs['extra_http_headers'] = self.get_headers()

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        # Set default navigation and wait timeouts
        page.set_default_navigation_timeout(self.get_timeout('navigation'))
        page.set_default_timeout(self.get_timeout('wait'))

        return context, page

    def launch_browser_sync(self) -> Browser:
        """
        Launch a Chromium browser synchronously.

        Returns:
            Browser: Playwright Browser instance
        """
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=self.headless)
        self.logger.info(f"Browser launched (headless={self.headless})")
        return browser

    async def launch_browser_async(self) -> AsyncBrowser:
        """
        Launch a Chromium browser asynchronously.

        Returns:
            AsyncBrowser: Async Playwright Browser instance
        """
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self.headless)
        self.logger.info(f"Async browser launched (headless={self.headless})")
        return browser

    def navigate_safe(self, page: Page, url: str, max_retries: int = 3) -> bool:
        """
        Navigate to URL with error handling and retries.

        Args:
            page (Page): Playwright page
            url (str): URL to navigate to
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if navigation successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                page.goto(url, wait_until='networkidle', timeout=self.get_timeout('navigation'))
                self.logger.info(f"Successfully navigated to {url}")
                return True
            except PlaywrightTimeoutError:
                self.logger.warning(f"Navigation timeout for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    self.apply_delay()
            except Exception as e:
                self.logger.error(f"Navigation error for {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    self.apply_delay()

        self.logger.error(f"Failed to navigate to {url} after {max_retries} attempts")
        return False

    async def navigate_safe_async(self, page: AsyncPage, url: str, max_retries: int = 3) -> bool:
        """
        Navigate to URL asynchronously with error handling and retries.

        Args:
            page (AsyncPage): Async Playwright page
            url (str): URL to navigate to
            max_retries (int): Maximum number of retry attempts

        Returns:
            bool: True if navigation successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                await page.goto(url, wait_until='networkidle', timeout=self.get_timeout('navigation'))
                self.logger.info(f"Successfully navigated to {url}")
                return True
            except PlaywrightTimeoutError:
                self.logger.warning(f"Navigation timeout for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.get_random_delay())
            except Exception as e:
                self.logger.error(f"Navigation error for {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.get_random_delay())

        self.logger.error(f"Failed to navigate to {url} after {max_retries} attempts")
        return False

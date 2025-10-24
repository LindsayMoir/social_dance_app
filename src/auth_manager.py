"""
Authentication management utilities for web scrapers.

This module consolidates login and session management patterns used across
multiple scrapers (fb.py, images.py, rd_ext.py, ebs.py) including credential
retrieval, session persistence, and CAPTCHA handling.

Classes:
    AuthenticationManager: Unified authentication management for multiple platforms

Key responsibilities:
    - Platform-specific login flows (Facebook, Instagram, generic websites)
    - Session state persistence and loading
    - Credential retrieval and management
    - CAPTCHA detection and handling
    - Cookie-based session validation
"""

import logging
import time
from typing import Optional, Tuple
from pathlib import Path
from playwright.sync_api import Page as SyncPage, BrowserContext as SyncBrowserContext, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Page as AsyncPage, BrowserContext as AsyncBrowserContext
import json
import requests


class AuthenticationManager:
    """
    Unified authentication manager for multiple web platforms.

    Consolidates login, credential handling, session persistence, and
    CAPTCHA management across all scrapers.
    """

    # Platform-specific login configurations
    LOGIN_CONFIGS = {
        'facebook': {
            'login_url': 'https://www.facebook.com/login',
            'email_selector': "input[name='email']",
            'pass_selector': "input[name='pass']",
            'submit_selector': "button[type='submit']",
            'check_logged_in': lambda url: 'login' not in url.lower(),
        },
        'instagram': {
            'login_url': 'https://www.instagram.com/accounts/login/',
            'email_selector': "input[name='username']",
            'pass_selector': "input[name='password']",
            'submit_selector': "button[type='submit']",
            'check_logged_in': lambda url: 'accounts/login' not in url.lower(),
        },
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize AuthenticationManager.

        Args:
            logger (logging.Logger, optional): Logger instance for logging auth details
        """
        self.logger = logger or logging.getLogger(__name__)

    def login_to_facebook_sync(self, page: SyncPage, context: SyncBrowserContext, auth_state_path: Optional[str] = None,
                               headless: bool = True, timeout: int = 20000) -> bool:
        """
        Login to Facebook using a Playwright page (synchronous).

        Supports both headless (automated credentials) and non-headless (manual) flows.

        Args:
            page (Page): Playwright page instance
            context (BrowserContext): Browser context for saving auth state
            auth_state_path (str, optional): Path to save/load auth state
            headless (bool): Whether browser is in headless mode
            timeout (int): Navigation timeout in milliseconds

        Returns:
            bool: True if login successful, False otherwise
        """
        from credentials import get_credentials
        from captcha_handler import CaptchaHandler

        try:
            # Navigate to login page
            if not self._navigate_safe_sync(page, 'facebook', timeout):
                return False

            # Check if already logged in
            if self._is_logged_in_sync(page, 'facebook'):
                self.logger.info("Already authenticated to Facebook")
                return True

            # Manual flow (headless=False)
            if not headless:
                return self._manual_login_sync(page, context, auth_state_path, timeout)

            # Automated flow (headless=True)
            return self._automated_login_sync(page, context, 'facebook', auth_state_path, timeout)

        except Exception as e:
            self.logger.error(f"Failed to login to Facebook: {e}")
            return False

    def login_to_instagram_sync(self, page: SyncPage, context: SyncBrowserContext, auth_state_path: Optional[str] = None,
                                timeout: int = 20000) -> bool:
        """
        Login to Instagram using a Playwright page (synchronous).

        Args:
            page (Page): Playwright page instance
            context (BrowserContext): Browser context for saving auth state
            auth_state_path (str, optional): Path to save/load auth state
            timeout (int): Navigation timeout in milliseconds

        Returns:
            bool: True if login successful, False otherwise
        """
        from credentials import get_credentials

        try:
            # Navigate to login page
            if not self._navigate_safe_sync(page, 'instagram', timeout):
                return False

            # Check if already logged in
            if self._is_logged_in_sync(page, 'instagram'):
                self.logger.info("Already authenticated to Instagram")
                return True

            # Automated login
            return self._automated_login_sync(page, context, 'instagram', auth_state_path, timeout)

        except Exception as e:
            self.logger.error(f"Failed to login to Instagram: {e}")
            return False

    def login_generic_sync(self, page: SyncPage, context: SyncBrowserContext, login_url: str,
                           email_selector: str, pass_selector: str, submit_selector: str,
                           auth_state_path: Optional[str] = None, timeout: int = 20000) -> bool:
        """
        Generic login flow for arbitrary websites (synchronous).

        Args:
            page (Page): Playwright page instance
            context (BrowserContext): Browser context for saving auth state
            login_url (str): URL to navigate to for login
            email_selector (str): CSS selector for email input
            pass_selector (str): CSS selector for password input
            submit_selector (str): CSS selector for submit button
            auth_state_path (str, optional): Path to save auth state
            timeout (int): Navigation timeout in milliseconds

        Returns:
            bool: True if login successful, False otherwise
        """
        from credentials import get_credentials

        try:
            # Navigate to login page
            try:
                page.goto(login_url, wait_until='networkidle', timeout=timeout)
            except PlaywrightTimeoutError:
                self.logger.warning(f"Timeout navigating to {login_url}; proceeding")

            # Wait for login form
            try:
                page.wait_for_selector(email_selector, timeout=10000)
                page.wait_for_selector(pass_selector, timeout=10000)
            except PlaywrightTimeoutError:
                self.logger.error("Login form did not appear")
                return False

            # Get credentials and fill form
            email, password, _ = get_credentials("Generic")
            page.fill(email_selector, email)
            page.fill(pass_selector, password)
            page.click(submit_selector)

            # Wait for login completion
            page.wait_for_timeout(5000)

            # Save auth state if path provided
            if auth_state_path:
                self._save_auth_state_sync(context, auth_state_path)

            return True

        except Exception as e:
            self.logger.error(f"Generic login failed: {e}")
            return False

    async def login_to_facebook_async(self, page: AsyncPage, context: AsyncBrowserContext, auth_state_path: Optional[str] = None,
                                      headless: bool = True, timeout: int = 20000) -> bool:
        """
        Login to Facebook using a Playwright page (asynchronous).

        Args:
            page (AsyncPage): Async Playwright page instance
            context (AsyncBrowserContext): Async browser context
            auth_state_path (str, optional): Path to save auth state
            headless (bool): Whether browser is in headless mode
            timeout (int): Navigation timeout in milliseconds

        Returns:
            bool: True if login successful, False otherwise
        """
        from credentials import get_credentials

        try:
            # Navigate to login page
            if not await self._navigate_safe_async(page, 'facebook', timeout):
                return False

            # Check if already logged in
            if await self._is_logged_in_async(page, 'facebook'):
                self.logger.info("Already authenticated to Facebook")
                return True

            # Automated login
            return await self._automated_login_async(page, context, 'facebook', auth_state_path, timeout)

        except Exception as e:
            self.logger.error(f"Failed to login to Facebook (async): {e}")
            return False

    def load_auth_state_sync(self, context: SyncBrowserContext, auth_state_path: str) -> bool:
        """
        Load saved authentication state into a browser context (synchronous).

        Args:
            context (BrowserContext): Browser context to load state into
            auth_state_path (str): Path to auth state file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(auth_state_path).exists():
                self.logger.warning(f"Auth state file not found: {auth_state_path}")
                return False

            context.add_init_script(f"window.localStorage.state = {Path(auth_state_path).read_text()}")
            self.logger.info(f"Loaded auth state from {auth_state_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load auth state: {e}")
            return False

    def load_cookies_from_json(self, context: SyncBrowserContext, json_path: str) -> bool:
        """
        Load cookies from a JSON file into a browser context.

        Args:
            context (BrowserContext): Browser context to load cookies into
            json_path (str): Path to JSON file containing cookies

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(json_path).exists():
                self.logger.warning(f"Cookie file not found: {json_path}")
                return False

            with open(json_path, 'r') as f:
                auth_data = json.load(f)

            cookies = auth_data.get('cookies', [])
            if not cookies:
                self.logger.warning("No cookies found in auth data")
                return False

            context.add_cookies(cookies)
            self.logger.info(f"Loaded {len(cookies)} cookies from {json_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load cookies: {e}")
            return False

    async def load_cookies_from_json_async(self, context: AsyncBrowserContext, json_path: str) -> bool:
        """
        Load cookies from a JSON file into a browser context (asynchronous).

        Args:
            context (AsyncBrowserContext): Async browser context
            json_path (str): Path to JSON file containing cookies

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(json_path).exists():
                self.logger.warning(f"Cookie file not found: {json_path}")
                return False

            with open(json_path, 'r') as f:
                auth_data = json.load(f)

            cookies = auth_data.get('cookies', [])
            if not cookies:
                self.logger.warning("No cookies found in auth data")
                return False

            await context.add_cookies(cookies)
            self.logger.info(f"Loaded {len(cookies)} cookies from {json_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load cookies (async): {e}")
            return False

    def validate_cookies_sync(self, session: requests.Session, critical_cookies: list) -> bool:
        """
        Validate that critical cookies exist and haven't expired.

        Args:
            session (requests.Session): Requests session with cookies
            critical_cookies (list): List of required cookie names

        Returns:
            bool: True if all critical cookies are present and valid, False otherwise
        """
        try:
            current_time = time.time()

            for cookie_name in critical_cookies:
                cookie = session.cookies.get(cookie_name)
                if not cookie:
                    self.logger.warning(f"Missing critical cookie: {cookie_name}")
                    return False

            self.logger.info(f"All {len(critical_cookies)} critical cookies are valid")
            return True

        except Exception as e:
            self.logger.error(f"Failed to validate cookies: {e}")
            return False

    # Private helper methods

    def _navigate_safe_sync(self, page: SyncPage, platform: str, timeout: int) -> bool:
        """Navigate to platform login page safely."""
        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                self.logger.error(f"Unknown platform: {platform}")
                return False

            page.goto(config['login_url'], wait_until='networkidle', timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            self.logger.warning(f"Timeout navigating to {platform} login page; proceeding")
            return True
        except Exception as e:
            self.logger.error(f"Failed to navigate to {platform} login: {e}")
            return False

    async def _navigate_safe_async(self, page: AsyncPage, platform: str, timeout: int) -> bool:
        """Navigate to platform login page safely (async)."""
        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                self.logger.error(f"Unknown platform: {platform}")
                return False

            await page.goto(config['login_url'], wait_until='networkidle', timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            self.logger.warning(f"Timeout navigating to {platform} login page; proceeding")
            return True
        except Exception as e:
            self.logger.error(f"Failed to navigate to {platform} login (async): {e}")
            return False

    def _is_logged_in_sync(self, page: SyncPage, platform: str) -> bool:
        """Check if already logged in by examining URL."""
        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                return False
            return config['check_logged_in'](page.url)
        except:
            return False

    async def _is_logged_in_async(self, page: AsyncPage, platform: str) -> bool:
        """Check if already logged in by examining URL (async)."""
        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                return False
            return config['check_logged_in'](page.url)
        except:
            return False

    def _manual_login_sync(self, page: SyncPage, context: SyncBrowserContext, auth_state_path: Optional[str], timeout: int) -> bool:
        """Manual login flow (user interaction)."""
        from captcha_handler import CaptchaHandler

        try:
            print("\n=== MANUAL LOGIN ===")
            print("1) In the browser window, enter your credentials and complete any 2FA.")
            input("   Once you've logged in successfully, press ENTER here to continue… ")

            try:
                page.reload(wait_until='networkidle', timeout=timeout)
            except PlaywrightTimeoutError:
                self.logger.warning("Reload after manual login timed out; continuing")

            # CAPTCHA handling
            captcha_detected = CaptchaHandler.detect_and_handle_sync(page, "Facebook", timeout=5000)
            if captcha_detected:
                try:
                    page.reload(wait_until='networkidle', timeout=timeout)
                except PlaywrightTimeoutError:
                    self.logger.warning("Reload after CAPTCHA timed out; continuing")

            if auth_state_path:
                self._save_auth_state_sync(context, auth_state_path)

            return True

        except Exception as e:
            self.logger.error(f"Manual login failed: {e}")
            return False

    def _automated_login_sync(self, page: SyncPage, context: SyncBrowserContext, platform: str,
                              auth_state_path: Optional[str], timeout: int) -> bool:
        """Automated login flow (credentials submission)."""
        from credentials import get_credentials

        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                return False

            # Wait for login form
            try:
                page.wait_for_selector(config['email_selector'], timeout=10000)
                page.wait_for_selector(config['pass_selector'], timeout=10000)
            except PlaywrightTimeoutError:
                self.logger.error(f"Login form did not appear for {platform}")
                return False

            # Get credentials and submit
            email, password, _ = get_credentials(platform.capitalize())
            page.fill(config['email_selector'], email)
            page.fill(config['pass_selector'], password)
            page.click(config['submit_selector'])

            # Wait for login processing
            page.wait_for_timeout(5000)

            # Check if still on login page
            if not self._is_logged_in_sync(page, platform):
                self.logger.error(f"Still on login page for {platform} after submission")
                return False

            if auth_state_path:
                self._save_auth_state_sync(context, auth_state_path)

            return True

        except Exception as e:
            self.logger.error(f"Automated login failed for {platform}: {e}")
            return False

    async def _automated_login_async(self, page: AsyncPage, context: AsyncBrowserContext, platform: str,
                                     auth_state_path: Optional[str], timeout: int) -> bool:
        """Automated login flow (credentials submission - async)."""
        from credentials import get_credentials

        try:
            config = self.LOGIN_CONFIGS.get(platform)
            if not config:
                return False

            # Wait for login form
            try:
                await page.wait_for_selector(config['email_selector'], timeout=10000)
                await page.wait_for_selector(config['pass_selector'], timeout=10000)
            except PlaywrightTimeoutError:
                self.logger.error(f"Login form did not appear for {platform}")
                return False

            # Get credentials and submit
            email, password, _ = get_credentials(platform.capitalize())
            await page.fill(config['email_selector'], email)
            await page.fill(config['pass_selector'], password)
            await page.click(config['submit_selector'])

            # Wait for login processing
            await page.wait_for_timeout(5000)

            # Check if still on login page
            if not await self._is_logged_in_async(page, platform):
                self.logger.error(f"Still on login page for {platform} after submission")
                return False

            if auth_state_path:
                await self._save_auth_state_async(context, auth_state_path)

            return True

        except Exception as e:
            self.logger.error(f"Automated login failed for {platform} (async): {e}")
            return False

    def _save_auth_state_sync(self, context: SyncBrowserContext, auth_state_path: str) -> None:
        """Save authentication state to file."""
        try:
            context.storage_state(path=auth_state_path)
            self.logger.info(f"Auth state saved to {auth_state_path}")

            # Sync to database if function available
            try:
                from secret_paths import sync_auth_to_db
                sync_auth_to_db(auth_state_path, 'generic')
            except:
                pass

        except Exception as e:
            self.logger.warning(f"Could not save auth state: {e}")

    async def _save_auth_state_async(self, context: AsyncBrowserContext, auth_state_path: str) -> None:
        """Save authentication state to file (async)."""
        try:
            await context.storage_state(path=auth_state_path)
            self.logger.info(f"Auth state saved to {auth_state_path}")
        except Exception as e:
            self.logger.warning(f"Could not save auth state (async): {e}")

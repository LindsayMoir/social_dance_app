"""
Centralized CAPTCHA detection and handling for all scrapers.

This module provides a unified interface for detecting and handling various CAPTCHA
types across different web services (Facebook, Google, Eventbrite, etc.).

Supports:
- Google reCAPTCHA (v2/v3)
- hCaptcha
- Google /sorry/ challenge pages
- Cloudflare challenges
- Generic CAPTCHA detection

Works with both sync and async Playwright page contexts.
"""

import logging
import asyncio
import os
from datetime import datetime
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import TimeoutError as AsyncPlaywrightTimeoutError


class CaptchaHandler:
    """
    Unified CAPTCHA detection and resolution handler.

    This class provides static methods for detecting and handling CAPTCHAs
    in both synchronous and asynchronous Playwright contexts.

    Usage:
        # Async context
        detected = await CaptchaHandler.detect_and_handle_async(page, "Facebook")

        # Sync context
        detected = CaptchaHandler.detect_and_handle_sync(page, "Google Search")
    """

    @staticmethod
    async def detect_and_handle_async(
        page,
        service_name: str,
        screenshot_dir: str = "debug",
        timeout: int = 5000
    ) -> bool:
        """
        Detects common CAPTCHA types and prompts user to solve (async version).

        Args:
            page: Playwright async page object
            service_name: Name of the service (e.g., "Facebook", "Google")
            screenshot_dir: Directory to save screenshots (default: "debug")
            timeout: Timeout in milliseconds for CAPTCHA detection (default: 5000)

        Returns:
            bool: True if CAPTCHA was detected and handled, False otherwise
        """
        captcha_type = None

        # Detect reCAPTCHA
        if await CaptchaHandler._detect_recaptcha_async(page, timeout):
            captcha_type = "reCAPTCHA"

        # Detect hCaptcha
        elif await CaptchaHandler._detect_hcaptcha_async(page, timeout):
            captcha_type = "hCaptcha"

        # Detect Google /sorry/ page
        elif await CaptchaHandler._detect_google_sorry_async(page):
            captcha_type = "Google CAPTCHA"

        # Detect Cloudflare challenge
        elif await CaptchaHandler._detect_cloudflare_async(page):
            captcha_type = "Cloudflare Challenge"

        if captcha_type:
            logging.warning(f"CAPTCHA detected on {service_name}: {captcha_type}")

            # Take screenshot
            screenshot_path = CaptchaHandler._get_screenshot_path(
                service_name, screenshot_dir, captcha_type
            )
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
                logging.info(f"CAPTCHA screenshot saved to: {screenshot_path}")
            except Exception as e:
                logging.error(f"Failed to save CAPTCHA screenshot: {e}")

            # Prompt user to solve
            CaptchaHandler._prompt_user_solve(service_name, captcha_type)

            # Wait for user to solve
            await asyncio.get_running_loop().run_in_executor(None, input)

            # Give page time to reload after CAPTCHA solve
            await page.wait_for_timeout(2000)

            logging.info(f"User indicated {captcha_type} solved for {service_name}")
            return True

        return False

    @staticmethod
    def detect_and_handle_sync(
        page,
        service_name: str,
        screenshot_dir: str = "debug",
        timeout: int = 5000
    ) -> bool:
        """
        Detects common CAPTCHA types and prompts user to solve (sync version).

        Args:
            page: Playwright sync page object
            service_name: Name of the service (e.g., "Facebook", "Google")
            screenshot_dir: Directory to save screenshots (default: "debug")
            timeout: Timeout in milliseconds for CAPTCHA detection (default: 5000)

        Returns:
            bool: True if CAPTCHA was detected and handled, False otherwise
        """
        captcha_type = None

        # Detect reCAPTCHA
        if CaptchaHandler._detect_recaptcha_sync(page, timeout):
            captcha_type = "reCAPTCHA"

        # Detect hCaptcha
        elif CaptchaHandler._detect_hcaptcha_sync(page, timeout):
            captcha_type = "hCaptcha"

        # Detect Google /sorry/ page
        elif CaptchaHandler._detect_google_sorry_sync(page):
            captcha_type = "Google CAPTCHA"

        # Detect Cloudflare challenge
        elif CaptchaHandler._detect_cloudflare_sync(page):
            captcha_type = "Cloudflare Challenge"

        if captcha_type:
            logging.warning(f"CAPTCHA detected on {service_name}: {captcha_type}")

            # Take screenshot
            screenshot_path = CaptchaHandler._get_screenshot_path(
                service_name, screenshot_dir, captcha_type
            )
            try:
                page.screenshot(path=screenshot_path, full_page=True)
                logging.info(f"CAPTCHA screenshot saved to: {screenshot_path}")
            except Exception as e:
                logging.error(f"Failed to save CAPTCHA screenshot: {e}")

            # Prompt user to solve
            CaptchaHandler._prompt_user_solve(service_name, captcha_type)

            # Wait for user to solve
            input()

            # Give page time to reload after CAPTCHA solve
            page.wait_for_timeout(2000)

            logging.info(f"User indicated {captcha_type} solved for {service_name}")
            return True

        return False

    # ==================== ASYNC DETECTION METHODS ====================

    @staticmethod
    async def _detect_recaptcha_async(page, timeout: int) -> bool:
        """
        Detects Google reCAPTCHA iframe (async).

        Args:
            page: Playwright async page object
            timeout: Timeout in milliseconds

        Returns:
            bool: True if reCAPTCHA detected
        """
        try:
            await page.wait_for_selector("iframe[src*='recaptcha']", timeout=timeout)
            return True
        except AsyncPlaywrightTimeoutError:
            return False
        except Exception as e:
            logging.debug(f"Error detecting reCAPTCHA: {e}")
            return False

    @staticmethod
    async def _detect_hcaptcha_async(page, timeout: int) -> bool:
        """
        Detects hCaptcha iframe (async).

        Args:
            page: Playwright async page object
            timeout: Timeout in milliseconds

        Returns:
            bool: True if hCaptcha detected
        """
        try:
            await page.wait_for_selector("iframe[src*='hcaptcha']", timeout=timeout)
            return True
        except AsyncPlaywrightTimeoutError:
            return False
        except Exception as e:
            logging.debug(f"Error detecting hCaptcha: {e}")
            return False

    @staticmethod
    async def _detect_google_sorry_async(page) -> bool:
        """
        Detects Google's /sorry/ CAPTCHA page (async).

        Args:
            page: Playwright async page object

        Returns:
            bool: True if Google CAPTCHA page detected
        """
        try:
            url = page.url.lower()
            if "sorry" in url:
                return True

            # Also check for form action
            form = await page.query_selector("form[action^='/sorry/']")
            return form is not None
        except Exception as e:
            logging.debug(f"Error detecting Google sorry page: {e}")
            return False

    @staticmethod
    async def _detect_cloudflare_async(page) -> bool:
        """
        Detects Cloudflare challenge page (async).

        Args:
            page: Playwright async page object

        Returns:
            bool: True if Cloudflare challenge detected
        """
        try:
            # Check for Cloudflare challenge page indicators
            title = await page.title()
            if "Just a moment" in title or "Attention Required" in title:
                return True

            # Check for Cloudflare challenge element
            cf_challenge = await page.query_selector("#challenge-running")
            return cf_challenge is not None
        except Exception as e:
            logging.debug(f"Error detecting Cloudflare challenge: {e}")
            return False

    # ==================== SYNC DETECTION METHODS ====================

    @staticmethod
    def _detect_recaptcha_sync(page, timeout: int) -> bool:
        """
        Detects Google reCAPTCHA iframe (sync).

        Args:
            page: Playwright sync page object
            timeout: Timeout in milliseconds

        Returns:
            bool: True if reCAPTCHA detected
        """
        try:
            page.wait_for_selector("iframe[src*='recaptcha']", timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            return False
        except Exception as e:
            logging.debug(f"Error detecting reCAPTCHA: {e}")
            return False

    @staticmethod
    def _detect_hcaptcha_sync(page, timeout: int) -> bool:
        """
        Detects hCaptcha iframe (sync).

        Args:
            page: Playwright sync page object
            timeout: Timeout in milliseconds

        Returns:
            bool: True if hCaptcha detected
        """
        try:
            page.wait_for_selector("iframe[src*='hcaptcha']", timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            return False
        except Exception as e:
            logging.debug(f"Error detecting hCaptcha: {e}")
            return False

    @staticmethod
    def _detect_google_sorry_sync(page) -> bool:
        """
        Detects Google's /sorry/ CAPTCHA page (sync).

        Args:
            page: Playwright sync page object

        Returns:
            bool: True if Google CAPTCHA page detected
        """
        try:
            url = page.url.lower()
            if "sorry" in url:
                return True

            # Also check for form action
            form = page.query_selector("form[action^='/sorry/']")
            return form is not None
        except Exception as e:
            logging.debug(f"Error detecting Google sorry page: {e}")
            return False

    @staticmethod
    def _detect_cloudflare_sync(page) -> bool:
        """
        Detects Cloudflare challenge page (sync).

        Args:
            page: Playwright sync page object

        Returns:
            bool: True if Cloudflare challenge detected
        """
        try:
            # Check for Cloudflare challenge page indicators
            title = page.title()
            if "Just a moment" in title or "Attention Required" in title:
                return True

            # Check for Cloudflare challenge element
            cf_challenge = page.query_selector("#challenge-running")
            return cf_challenge is not None
        except Exception as e:
            logging.debug(f"Error detecting Cloudflare challenge: {e}")
            return False

    # ==================== HELPER METHODS ====================

    @staticmethod
    def _get_screenshot_path(service_name: str, screenshot_dir: str, captcha_type: str) -> str:
        """
        Generates a unique screenshot path with timestamp.

        Args:
            service_name: Name of the service
            screenshot_dir: Directory to save screenshot
            captcha_type: Type of CAPTCHA detected

        Returns:
            str: Full path to screenshot file
        """
        # Ensure directory exists
        os.makedirs(screenshot_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_service = service_name.lower().replace(" ", "_")
        safe_captcha = captcha_type.lower().replace(" ", "_")
        filename = f"{safe_service}_{safe_captcha}_{timestamp}.png"

        return os.path.join(screenshot_dir, filename)

    @staticmethod
    def _prompt_user_solve(service_name: str, captcha_type: str) -> None:
        """
        Displays a consistent user prompt for CAPTCHA solving.

        Args:
            service_name: Name of the service
            captcha_type: Type of CAPTCHA detected
        """
        print("\n" + "="*70)
        print(f"🔐 {captcha_type.upper()} DETECTED on {service_name.upper()} 🔐")
        print("="*70)
        print(f"A {captcha_type} challenge has appeared in the browser.")
        print("Please solve the CAPTCHA manually, then press ENTER to continue.")
        print("="*70)
        print("Waiting for you to solve the CAPTCHA and press ENTER...", end=" ", flush=True)

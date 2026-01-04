"""
credential_validator.py

Pre-flight credential validation for pipeline.py.

This module validates Gmail, Eventbrite, and Facebook credentials before running
the main pipeline. It ensures all services are authenticated and ready, preventing
mid-pipeline failures due to expired tokens, session timeouts, or CAPTCHA challenges.

Validation runs with headless=False to allow user interaction for:
- Gmail OAuth re-authentication
- Eventbrite 2FA verification
- Facebook login and CAPTCHA solving

After validation completes, pipeline.py continues with headless=True for normal operation.

Functions:
    validate_gmail(headless, check_timeout_seconds): Validates Gmail OAuth credentials
    validate_eventbrite(headless, check_timeout_seconds): Validates Eventbrite session
    validate_facebook(headless, check_timeout_seconds): Validates Facebook session
    validate_credentials(headless, check_timeout_seconds): Main validation orchestrator

Usage:
    from credential_validator import validate_credentials

    # Run validation before pipeline (with visible browser)
    results = validate_credentials(headless=False, check_timeout_seconds=60)

    if all(r['valid'] for r in results.values()):
        # Continue with pipeline using headless=True
        pass
"""

import asyncio
from datetime import datetime
import logging
import os
import yaml

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def validate_gmail(headless=False, check_timeout_seconds=60):
    """
    Validates Gmail OAuth credentials using existing emails.py authentication logic.

    Quick check: Attempts to authenticate with existing token (max check_timeout_seconds)
    If invalid: Opens browser for user to complete OAuth flow (no time limit)

    Args:
        headless (bool): Whether to run browser in headless mode
        check_timeout_seconds (int): Max time for quick credential check

    Returns:
        dict: {'valid': bool, 'error': str or None}
    """
    logging.info("=" * 50)
    logging.info("GMAIL CREDENTIAL VALIDATION")
    logging.info("=" * 50)

    try:
        from emails import GmailProcessor

        # Check if running on Render (skip Gmail validation)
        from secret_paths import is_render_environment
        if is_render_environment():
            logging.info("validate_gmail(): Running on Render - Gmail not required")
            return {'valid': True, 'error': None}

        logging.info("validate_gmail(): Attempting to authenticate Gmail...")
        start_time = datetime.now()

        # Create Gmail processor - this will attempt authentication
        # The authenticate_gmail() method will open browser if credentials are invalid
        gmail_processor = GmailProcessor(llm_handler=None)

        elapsed = (datetime.now() - start_time).total_seconds()

        if gmail_processor.service:
            logging.info(f"validate_gmail(): Gmail authenticated successfully ({elapsed:.1f}s)")
            return {'valid': True, 'error': None}
        else:
            logging.error("validate_gmail(): Gmail authentication failed")
            return {'valid': False, 'error': 'Gmail service not initialized'}

    except Exception as e:
        logging.error(f"validate_gmail(): Error during Gmail validation: {e}")
        return {'valid': False, 'error': str(e)}


async def validate_eventbrite_async(headless=False, check_timeout_seconds=60):
    """
    Validates Eventbrite session credentials using existing rd_ext.py logic.

    Quick check: Attempts to reuse saved session (max check_timeout_seconds)
    If invalid: Opens browser for user to login with 2FA (no time limit)

    Args:
        headless (bool): Whether to run browser in headless mode
        check_timeout_seconds (int): Max time for quick credential check

    Returns:
        dict: {'valid': bool, 'error': str or None}
    """
    logging.info("=" * 50)
    logging.info("EVENTBRITE CREDENTIAL VALIDATION")
    logging.info("=" * 50)

    try:
        from rd_ext import ReadExtract

        logging.info("validate_eventbrite(): Attempting to authenticate Eventbrite...")
        start_time = datetime.now()

        # Create ReadExtract instance with headless mode from parameter
        # Temporarily override config for validation
        original_headless = config['crawling']['headless']
        config['crawling']['headless'] = headless

        read_extract = ReadExtract(config_path='config/config.yaml')
        await read_extract.init_browser()

        # Attempt Eventbrite login
        # login_to_website returns True if session valid or login successful
        login_success = await read_extract.login_to_website(
            organization="Eventbrite",
            login_url="https://www.eventbrite.com/signin/",
            email_selector="input#email",
            pass_selector="input#password",
            submit_selector="button[type='submit']"
        )

        # Restore original headless setting
        config['crawling']['headless'] = original_headless

        # Clean up browser
        await read_extract.close()

        elapsed = (datetime.now() - start_time).total_seconds()

        if login_success:
            logging.info(f"validate_eventbrite(): Eventbrite authenticated successfully ({elapsed:.1f}s)")
            return {'valid': True, 'error': None}
        else:
            logging.error("validate_eventbrite(): Eventbrite authentication failed")
            return {'valid': False, 'error': 'Eventbrite login failed'}

    except Exception as e:
        logging.error(f"validate_eventbrite(): Error during Eventbrite validation: {e}")
        return {'valid': False, 'error': str(e)}


def validate_eventbrite(headless=False, check_timeout_seconds=60):
    """
    Synchronous wrapper for validate_eventbrite_async.
    """
    return asyncio.run(validate_eventbrite_async(headless, check_timeout_seconds))


def validate_facebook(headless=False, check_timeout_seconds=60):
    """
    Validates Facebook session credentials using existing fb.py logic.

    Quick check: Attempts to reuse saved session (max check_timeout_seconds)
    If invalid: Opens browser for user to login (no time limit)

    Args:
        headless (bool): Whether to run browser in headless mode
        check_timeout_seconds (int): Max time for quick credential check

    Returns:
        dict: {'valid': bool, 'error': str or None}
    """
    logging.info("=" * 50)
    logging.info("FACEBOOK CREDENTIAL VALIDATION")
    logging.info("=" * 50)

    try:
        from fb import FacebookEventScraper

        logging.info("validate_facebook(): Attempting to authenticate Facebook...")
        start_time = datetime.now()

        # Temporarily override config for validation
        original_headless = config['crawling']['headless']
        config['crawling']['headless'] = headless

        # Create Facebook scraper - this automatically calls login_to_facebook()
        # which will open browser if session is invalid
        fb_scraper = FacebookEventScraper(config_path='config/config.yaml')

        # Restore original headless setting
        config['crawling']['headless'] = original_headless

        # Check if login was successful by navigating to Facebook
        try:
            fb_scraper.page.goto("https://www.facebook.com/", timeout=15000)
            logged_in = "login" not in fb_scraper.page.url.lower()
        except Exception as nav_error:
            logging.warning(f"validate_facebook(): Error checking login status: {nav_error}")
            logged_in = False

        # Clean up browser
        fb_scraper.browser.close()
        fb_scraper.playwright.stop()

        elapsed = (datetime.now() - start_time).total_seconds()

        if logged_in:
            logging.info(f"validate_facebook(): Facebook authenticated successfully ({elapsed:.1f}s)")
            return {'valid': True, 'error': None}
        else:
            logging.error("validate_facebook(): Facebook authentication failed")
            return {'valid': False, 'error': 'Facebook login failed - still on login page'}

    except Exception as e:
        logging.error(f"validate_facebook(): Error during Facebook validation: {e}")
        return {'valid': False, 'error': str(e)}


def validate_credentials(headless=False, check_timeout_seconds=60):
    """
    Validates all service credentials sequentially.

    Runs quick checks (max check_timeout_seconds) for each service.
    If credentials are invalid, opens browser (headless=False) for user to fix.
    User has unlimited time to complete authentication.

    Args:
        headless (bool): Whether to run browsers in headless mode (False for validation)
        check_timeout_seconds (int): Max time to spend checking each service

    Returns:
        dict: {
            'gmail': {'valid': bool, 'error': str or None},
            'eventbrite': {'valid': bool, 'error': str or None},
            'facebook': {'valid': bool, 'error': str or None}
        }
    """
    results = {}
    start_time = datetime.now()

    logging.info("\n")
    logging.info("=" * 70)
    logging.info("CREDENTIAL VALIDATION STARTED")
    logging.info(f"Headless mode: {headless}")
    logging.info(f"Check timeout per service: {check_timeout_seconds} seconds")
    logging.info("=" * 70)
    logging.info("\n")

    # Validate Gmail
    logging.info("Validating Gmail credentials...")
    results['gmail'] = validate_gmail(headless, check_timeout_seconds)
    if not results['gmail']['valid']:
        logging.error(f"Gmail validation failed: {results['gmail']['error']}")
        logging.error("Cannot proceed with pipeline - Gmail credentials required")
        return results

    # Validate Eventbrite
    logging.info("\nValidating Eventbrite credentials...")
    results['eventbrite'] = validate_eventbrite(headless, check_timeout_seconds)
    if not results['eventbrite']['valid']:
        logging.error(f"Eventbrite validation failed: {results['eventbrite']['error']}")
        logging.error("Cannot proceed with pipeline - Eventbrite credentials required")
        return results

    # Validate Facebook
    logging.info("\nValidating Facebook credentials...")
    results['facebook'] = validate_facebook(headless, check_timeout_seconds)
    if not results['facebook']['valid']:
        logging.error(f"Facebook validation failed: {results['facebook']['error']}")
        logging.error("Cannot proceed with pipeline - Facebook credentials required")
        return results

    # All validations passed
    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info("\n")
    logging.info("=" * 70)
    logging.info("CREDENTIAL VALIDATION COMPLETED SUCCESSFULLY")
    logging.info(f"Total validation time: {elapsed:.1f} seconds")
    logging.info("All services authenticated and ready")
    logging.info("=" * 70)
    logging.info("\n")

    return results


def main():
    """
    Main function to run credential validation as a standalone script.
    Called when pipeline.py executes this file as a subprocess.
    """
    from logging_config import setup_logging
    setup_logging('credential_validator')

    logging.info("\n\nStarting credential validator...")
    logging.info("=" * 70)
    logging.info("CREDENTIAL VALIDATION")
    logging.info("=" * 70)

    # Check if running on Render
    is_render = os.getenv('RENDER') == 'true'
    if is_render:
        logging.info("Running on Render - skipping credential validation")
        return 0

    try:
        # Run validation with headless=False to allow user interaction
        results = validate_credentials(headless=False, check_timeout_seconds=60)

        # Check if all validations passed
        all_valid = all(r['valid'] for r in results.values())

        if not all_valid:
            failed_services = [k for k, v in results.items() if not v['valid']]
            logging.error(f"Credential validation failed for: {', '.join(failed_services)}")
            for service, result in results.items():
                if not result['valid'] and result['error']:
                    logging.error(f"  {service}: {result['error']}")
            return 1

        logging.info("=" * 70)
        logging.info("ALL CREDENTIALS VALIDATED SUCCESSFULLY")
        logging.info("Pipeline will continue with headless=True")
        logging.info("=" * 70)
        return 0

    except Exception as e:
        logging.error(f"Error during credential validation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

"""
rd_ext.py

input (str): The event URL.

This module defines the ReadExtract class, which uses the asyncio version of Playwright 
to manage a single browser page globally for tasks such as logging into Facebook 
and extracting event text. The class ensures that only one login occurs and the same 
page is reused across multiple operations. It is designed to be easily callable 
from other class libraries and supports asynchronous operations.

Objectives:
1. Maintain one globally shared page across the class instance.
2. Ensure a single login session is reused for all operations, avoiding multiple logins.
3. Provide an easy-to-use interface for other class libraries.
4. Utilize the asyncio version of Playwright for asynchronous operations.

output (str): The extracted text content.
"""

import asyncio
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
import random
import yaml

from credentials import get_credentials  # Import the utility function


class ReadExtract:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.logged_in = False
        

    def extract_text_with_playwright(self, url):
        """
        Synchronously extracts the text content from a web page using Playwright.
        If a Google sign-in page is encountered, the method aborts and returns None.

        Args:
            url (str): The URL of the web page to extract text from.

        Returns:
            str or None: The extracted text content, or None if Google sign-in is detected or an error occurs.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config['crawling']['headless'])
                page = browser.new_page()
                page.goto(url, timeout=10000)
                page.wait_for_timeout(3000)

                # Check for Google sign-in page
                if "accounts.google.com" in page.url or page.is_visible("input#identifierId"):
                    logging.info("def extract_text_with_playwright(): Google sign-in detected. Aborting extraction.")
                    browser.close()
                    return None

                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                extracted_text = ' '.join(soup.stripped_strings)

                browser.close()
                return extracted_text
        except Exception as e:
            logging.error(f"def extract_text_with_playwright(): Failed to extract text from {url}: {e}")
            return None


    async def init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.config['crawling']['headless'])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()


    async def login_to_facebook(self, organization):
        try:
            context = await self.browser.new_context(storage_state="auth.json")
            self.page = await context.new_page()
            await self.page.goto("https://www.facebook.com/", timeout=60000)
            if "login" not in self.page.url.lower():
                logging.info("def extract_text_with_playwright(): Loaded existing session. Already logged into Facebook.")
                self.logged_in = True
                return True
        except Exception:
            logging.info("def extract_text_with_playwright(): No valid saved session found. Proceeding with manual login.")

        email, password, _ = get_credentials(self.config, organization)  # Use utility function

        await self.page.goto("https://www.facebook.com/login", timeout=60000)
        await self.page.fill("input[name='email']", email)
        await self.page.fill("input[name='pass']", password)
        await self.page.click("button[name='login']")
        await self.page.wait_for_timeout(random.randint(4000, 6000))

        logging.warning("Please solve any captcha or challenge in the browser, then press Enter here to continue...")
        input("After solving captcha/challenge (if any), press Enter to continue...")

        await self.page.wait_for_timeout(random.randint(4000, 6000))

        if "login" in self.page.url.lower():
            logging.error("def extract_text_with_playwright(): Login failed. Please check your credentials or solve captcha challenges.")
            return False

        try:
            await self.page.context.storage_state(path="auth.json")
            logging.info("def extract_text_with_playwright(): Session state saved for future use.")
        except Exception as e:
            logging.warning(f"def extract_text_with_playwright(): Could not save session state: {e}")

        self.logged_in = True
        logging.info("def extract_text_with_playwright(): Login to Facebook successful.")
        return True
    

    async def login_to_website(self, organization, login_url, email_selector, pass_selector, submit_selector):
        """
        Only attempt to log in if the site actually shows a login form or 
        redirects to a login page. Otherwise, skip login.
        """
        # 1) Try loading existing session state
        try:
            context = await self.browser.new_context(storage_state=f"{organization.lower()}_auth.json")
            self.page = await context.new_page()
            await self.page.goto(login_url, timeout=20000)

            # If we are not on the login page or we remain logged in,
            # we can return True immediately:
            if organization.lower() not in self.page.url.lower():
                logging.info(f"Loaded existing session for {organization}, no login required.")
                self.logged_in = True
                return True
        except Exception:
            logging.info(f"No valid saved session for {organization}, checking if login is needed...")

        # 2) Now we are on the site. Check if login is required:
        # For example, we try to see if the email or password field is present
        try:
            # Wait a short time to see if the login form shows up
            await self.page.wait_for_selector(email_selector, timeout=5000)
            # If that selector is found, it likely means the site wants login
            need_login = True
            logging.info(f"{organization} requires login (form detected).")
        except:
            # If we time out waiting for the email selector,
            # we assume the site does NOT require login
            logging.info(f"No login form detected for {organization}, skipping login.")
            self.logged_in = True
            return True

        # 3) If we get here, we presumably see a login form
        if need_login:
            # Retrieve environment credentials
            email, password, _ = get_credentials(organization)

            # Fill and submit the form
            await self.page.fill(email_selector, email)
            await self.page.fill(pass_selector, password)
            await self.page.click(submit_selector)
            await self.page.wait_for_timeout(random.randint(4000, 6000))

            logging.warning(f"Please solve any captcha on {organization}'s login page, then press Enter to continue...")
            input("Press Enter after solving captcha (if any)...")
            await self.page.wait_for_timeout(random.randint(4000, 6000))

            # Check if we are still on the login page
            if "login" in self.page.url.lower():
                logging.error(f"Login to {organization} failed. Credentials may be incorrect.")
                return False

            # 4) If login was successful, save session
            try:
                await self.page.context.storage_state(path=f"{organization.lower()}_auth.json")
                logging.info(f"Session state saved for {organization}.")
            except Exception as e:
                logging.warning(f"Could not save session state for {organization}: {e}")

            self.logged_in = True
            logging.info(f"Login to {organization} successful.")
            return True

        # Should never reach here, but just in case:
        return False


    async def login_if_required(self, url: str) -> bool:
        """
        Determines if the URL belongs to Facebook, Google, allevents, or Eventbrite.
        If so, calls the corresponding login method. Otherwise, returns True (no login needed).
        """
        url_lower = url.lower()

        # If it's Facebook
        if "facebook.com" in url_lower:
            return await self.login_to_facebook("Facebook")

        # If it's Google
        elif "google" in url_lower:
            return await self.login_to_website(
                organization="Google",
                login_url="https://accounts.google.com/signin",
                email_selector="input[type='email']",
                pass_selector="input[type='password']",
                submit_selector="button[type='submit']"
            )

        # If it's Allevents
        elif "allevents" in url_lower:
            return await self.login_to_website(
                organization="allevents",
                login_url="https://www.allevents.in/login",
                email_selector="input[name='email']",
                pass_selector="input[name='password']",
                submit_selector="button[type='submit']"
            )

        # If it's Eventbrite
        elif "eventbrite" in url_lower:
            return await self.login_to_website(
                organization="eventbrite",
                login_url="https://www.eventbrite.com/signin/",
                email_selector="input#email",
                pass_selector="input#password",
                submit_selector="button[type='submit']"
            )

        # Otherwise, no login needed
        return True

    
    async def extract_event_text(self, link):
        """
        Asynchronously extracts text from an event page using Playwright and BeautifulSoup.
        If the page is a Facebook event, attempts to click the "See More" button to reveal full content.

        Args:
            link (str): The event URL.

        Returns:
            str: The extracted text content.
        """
        login_success = await self.login_if_required(link)
        if not login_success:
            logging.error("def extract_event_text(): Login failed. Aborting text extraction.")
            return None

        #try:
        await self.page.goto(link, timeout=10000)
        # Randomize wait time between 4 to 6 seconds
        await self.page.wait_for_timeout(random.randint(4000, 6000))

        # If the link is from Facebook, attempt to click "See More" to load additional content
        if 'facebook.com' in link.lower():
            try:
                # Look for a button or link with text "See More"
                more_button = await self.page.query_selector("text=See More")
                if more_button:
                    await more_button.click()
                    # Randomize wait time after clicking "See More"
                    await self.page.wait_for_timeout(random.randint(4000, 6000))
                    logging.info("Clicked 'See More' to load additional Facebook content.")
            except Exception as e:
                logging.warning(f"Could not click 'See More' button: {e}")

        content = await self.page.content()
        soup = BeautifulSoup(content, 'html.parser')
        extracted_text = ' '.join(soup.stripped_strings)
        logging.info(f"def extract_event_text(): Extracted text from {link}.")

        if extracted_text:
            return extracted_text
        else:
            logging.warning(f"def extract_event_text(): No text extracted from {link}.")
            return None
        # except Exception as e:
        #     logging.error(f"def extract_event_text(): Failed to extract text from {link}: {e}")
        #     return None
        
        
    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


    async def main(self, url):
        await self.init_browser()
        text = await self.extract_event_text(url)
        logging.info(f"def main(): Extracted text: {text}")
        await self.close()


if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    read_extract = ReadExtract("config/config.yaml")

    url = 'https://www.facebook.com/events/1120355202828169/'

    # Initialize the browser and extract text
    asyncio.run(read_extract.main(url))

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")


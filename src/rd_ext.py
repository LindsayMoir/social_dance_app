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
5. Provides a place to deal with odd edge cases that may arise if you are using the logic in __main__
    a. This will result in database updates to the events and urls tables.

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
import re
import yaml

from llm import LLMHandler
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
                logging.info("def login_to_facebook(): Loaded existing session. Already logged into Facebook.")
                self.logged_in = True
                return True
        except Exception:
            logging.info("def login_to_facebook(): No valid saved session found. Proceeding with manual login.")

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
            logging.error("def login_to_facebook(): Login failed. Please check your credentials or solve captcha challenges.")
            return False

        try:
            await self.page.context.storage_state(path="auth.json")
            logging.info("def login_to_facebook(): Session state saved for future use.")
        except Exception as e:
            logging.warning(f"def login_to_facebook(): Could not save session state: {e}")

        self.logged_in = True
        logging.info("def login_to_facebook(): Login to Facebook successful.")
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


    async def extract_event_text(self, link, max_retries=3):
        """
        Extracts text from an event page with retries if extraction fails.

        Args:
            link (str): The event URL.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            str: Extracted text content or None if extraction fails.
        """
        login_success = await self.login_if_required(link)
        if not login_success:
            logging.error(f"def extract_event_text(): Login failed. Aborting extraction for {link}.")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                await self.page.goto(link, timeout=15000)
                await self.page.wait_for_load_state("domcontentloaded")  # Ensure the DOM is fully loaded
                await asyncio.sleep(random.uniform(3, 6))  # Randomized delay for stability

                # Extract page content
                content = await self.page.content()
                soup = BeautifulSoup(content, 'html.parser')
                extracted_text = ' '.join(soup.stripped_strings)

                if extracted_text.strip():  # Ensure non-empty text
                    logging.info(f"def extract_event_text(): Successfully extracted text from {link} on attempt {attempt}.")
                    return extracted_text
                else:
                    logging.warning(f"def extract_event_text(): Attempt {attempt} - No text found for {link}. Retrying...")

            except Exception as e:
                logging.error(f"def extract_event_text(): Attempt {attempt} failed for {link}. Error: {e}")

            # Wait before retrying (exponential backoff)
            await asyncio.sleep(attempt * 2)

        logging.error(f"def extract_event_text(): Extraction failed after {max_retries} attempts for {link}.")
        return None

        
    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


    async def main(self, url):
        await self.init_browser()
        logging.info(f"def main(): Initialized browser for url: {url}")
        text = await self.extract_event_text(url)
        logging.info(f"def main(): Extracted text: {text}")
        await self.close()
        return text


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

    # Instantiate the classes
    read_extract = ReadExtract("config/config.yaml")
    llm_handler = LLMHandler("config/config.yaml")

    # Read .csv file to deal with oddities
    df = pd.read_csv(config['input']['edge_cases'])

    # Iterate over the rows of the dataframe
    for row in df.itertuples(index=True, name=None):

        # Get the url, organization name, keywords
        idx, source, keywords, url  = row

        logging.info(f"(__main__ in rd_ext.py: idx: {idx}, url: {url}, source: {source}, keywords: {keywords})")

        logging.info(f"__main__: Extracting text from {url}...")
        # Initialize the browser and extract text
        extracted_text = asyncio.run(read_extract.main(url))

        # Process the extracted text with LLM. The url is the key into config to get the right prompt
        llm_status = llm_handler.process_llm_response(url, extracted_text, source, keywords, prompt=url)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")


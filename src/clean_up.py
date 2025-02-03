"""
clean_up.py

This module defines the CleanUp class, which performs data cleanup on event records 
that do not have URLs. The process:

1. Retrieves events without URLs from the database.
2. For each such event, performs a (blocking) Google search using the event name to find 
   a relevant URL (fuzzy matching + ranking).
3. Uses Playwright's async API to extract text from the found URL.
4. Checks if the extracted text is relevant (LLMHandler).
5. If relevant, queries the LLM for structured event info.
6. Merges with the existing DB record.
7. Updates the DB and writes the found URL to the URLs table.

Dependencies:
    - pandas, fuzzywuzzy, logging, googleapiclient, etc.
    - db.DatabaseHandler, fb.FacebookEventScraper, gs.GoogleSearch, 
      llm.LLMHandler, credentials.get_credentials
"""

import asyncio
import logging
import random
import pandas as pd
from fuzzywuzzy import fuzz
import yaml
from googleapiclient.discovery import build
from datetime import datetime
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from db import DatabaseHandler
from llm import LLMHandler
from credentials import get_credentials


class CleanUp:
    def __init__(self, config):
        self.config = config
        self.db_handler = DatabaseHandler(config)
        self.llm_handler = LLMHandler(config_path="config/config.yaml")

        # Establish database connection
        self.conn = self.db_handler.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        logging.info("def __init__(): Database connection established.")

        # Retrieve Google API credentials using credentials.py
        _, self.api_key, self.cse_id = get_credentials('Google')

        # We'll store references to the async browser/page once we init them
        self.browser = None
        self.context = None
        self.logged_in_page = None


    async def login_to_facebook(self, page, browser) -> bool:
            """
            Asynchronously logs into Facebook using credentials, handles login prompts,
            and saves session state for future reuse. Manual steps may be required for captcha.

            Args:
                page (playwright.async_api.Page): The Playwright page instance.
                browser (playwright.async_api.Browser): The Playwright browser instance.

            Returns:
                bool: True if login is successful, False otherwise.
            """
            try:
                context = await browser.new_context(storage_state="auth.json")
                page = await context.new_page()
                await page.goto("https://www.facebook.com/", timeout=60000)

                # If "login" not in the URL, we assume we're already logged in from storage_state
                if "login" not in page.url.lower():
                    logging.info("Loaded existing session (async). Already logged into Facebook.")
                    self.logged_in_page = page
                    return True
            except Exception:
                logging.info("No valid saved session found. Proceeding with manual async login...")

            # Fallback: manual login
            email, password, _ = get_credentials('Facebook')  # or however you retrieve credentials

            await page.goto("https://www.facebook.com/login", timeout=60000)
            await page.fill("input[name='email']", email)
            await page.fill("input[name='pass']", password)
            await page.click("button[name='login']")
            await page.wait_for_timeout(5000)

            async def check_additional_login(p):
                email_visible = await p.is_visible("input[name='email']")
                pass_visible = await p.is_visible("input[name='pass']")
                if email_visible and pass_visible:
                    logging.info("Additional login prompt detected. Re-filling credentials (async).")
                    await p.fill("input[name='email']", email)
                    await p.fill("input[name='pass']", password)
                    await p.click("button[name='login']")
                    await p.wait_for_timeout(5000)
                    return True
                return False

            # Attempt to handle subsequent prompts
            attempts = 0
            while attempts < 3 and (await check_additional_login(page)):
                attempts += 1

            # Pause for manual captcha/challenge
            logging.warning("Please solve any captcha/challenge in the browser. Pausing...")
            print("Press Enter here once done.")
            await asyncio.get_running_loop().run_in_executor(None, input)  # non-blocking approach

            # Check if we ended up logged in
            await page.wait_for_timeout(5000)
            if "login" in page.url.lower():
                logging.error("Async login failed. Check credentials or captcha challenges.")
                return False

            # Save session
            try:
                context = page.context
                await context.storage_state(path="auth.json")
                logging.info("Session state saved (async).")
            except Exception as e:
                logging.warning(f"Could not save session state (async): {e}")

            self.logged_in_page = page
            logging.info("Async login to Facebook successful.")
            return True
    

    async def extract_text_from_fb_url(self, url: str) -> str | None:
        """
        Asynchronously extracts text content from a Facebook event URL using
        Playwright and BeautifulSoup.

        Args:
            url (str): The Facebook event URL.

        Returns:
            str or None: Extracted text content.
        """
        if not self.browser or not self.logged_in_page:
            logging.error("Browser or logged_in_page not initialized in CleanUp.")
            return None

        fb_status = await self.login_to_facebook(self.logged_in_page, self.browser)
        if not fb_status:
            logging.warning("extract_text_from_fb_url: Could not log in to Facebook (async).")
            return None

        logging.info("extract_text_from_fb_url (async): Logged into Facebook.")
        extracted_text = await self.extract_event_text(url)
        return extracted_text
    

    async def extract_event_text(self, link: str) -> str | None:
        """
        Asynchronously extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            link (str): The event link.

        Returns:
            str or None: The extracted relevant text content, or None if not found.
        """
        if not self.logged_in_page:
            logging.error("No logged_in_page available. Did you call login_to_facebook (async)?")
            return None

        #try:
        page = self.logged_in_page  # or create a fresh page if you prefer
        await page.goto(link, timeout=10000)

        # Look for "See more" buttons
        more_buttons = await page.query_selector_all("text=/See more/i")
        for mb in more_buttons:
            try:
                await mb.wait_for_element_state("stable", timeout=3000)
                await mb.click()
                await page.wait_for_timeout(random.randint(3000, 5000))
                logging.info(f"Clicked 'See more' button in URL: {link}")
            except Exception as e:
                logging.warning(f"Could not click 'See more' in {link}: {e}")

        if not more_buttons:
            logging.debug(f"No 'See more' buttons found in URL: {link}")

        await page.wait_for_timeout(5000)
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        extracted_text = ' '.join(soup.stripped_strings)
        logging.info(f"(async) raw text from {link}: {extracted_text}")

        if extracted_text:
            # If you have a separate method to refine the text:
            extracted_text = self.extract_relevant_text(extracted_text, link)
            if extracted_text:
                logging.info(f"(async) def extract_event_text(): relevant text from {link}: {extracted_text}")
                return extracted_text
            else:
                logging.info(f"(async) def extract_event_text(): No relevant text found in {link}.")
                return None
        else:
            logging.info(f"(async) def extract_event_text(): No text extracted from {link}.")
            return None

        # except Exception as e:
        #     logging.error(f"(async) def extract_event_text(): Failed to extract text from {link}: {e}")
        #     return None
    

    def extract_relevant_text(self, content, link):
        """
        Extracts a relevant portion of text from the given content based on specific patterns.

        This function searches for the first occurrence of "More About Discussion" in the content,
        then finds the last occurrence of a day of the week before this phrase, and finally extracts
        the text from this day up to the phrase "Guests See All".

        Args:
            content (str): The text content to be processed.

        Returns:
            str: The extracted relevant text if all patterns are found, otherwise None.

        Logs:
            Logs warnings if any of the required patterns ("More About Discussion", a day of the week,
            or "Guests See All") are not found in the expected positions within the content.
        """

        # Step 1: Find the first occurrence of "More About Discussion" (case-insensitive)
        mad_pattern = re.compile(r"More About Discussion", re.IGNORECASE)
        mad_match = mad_pattern.search(content)
        if not mad_match:
            logging.warning(f"'More About Discussion' not found in {link}.")
            return None

        mad_start = mad_match.start()

        # Step 2: In the text before "More About Discussion", find the last occurrence of a day of the week
        days_pattern = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)
        days_matches = list(days_pattern.finditer(content, 0, mad_start))
        if not days_matches:
            logging.warning(f"No day of the week found before 'More About Discussion' in {link}.")
            return None

        last_day_match = days_matches[-1]
        day_start = last_day_match.start()

        # Step 3: From the last day match, extract up to "Guests See All"
        gsa_pattern = re.compile(r"Guests See All", re.IGNORECASE)
        gsa_match = gsa_pattern.search(content, last_day_match.end())
        if not gsa_match:
            logging.warning(f"'Guests See All' not found after last day of the week in {link}.")
            return None

        gsa_end = gsa_match.end()

        # Extract the desired text
        extracted_text = content[day_start:gsa_end]

        return extracted_text
    

    async def process_events_without_url(self):
        """
        Asynchronously processes events without URLs by:
          1) Querying the DB for events with no URL.
          2) Doing a Google search to find candidate URLs.
          3) Extracting text from the best candidate URL using async Playwright.
          4) Checking relevance (LLMHandler).
          5) If relevant, fetching structured info from LLM.
          6) Merging/Updating DB records.
        """
        query = "SELECT * FROM events WHERE url = '' OR url IS NULL"
        no_urls_df = pd.read_sql(query, self.db_handler.conn)
        logging.info(
            f"def process_events_without_url(): Found {no_urls_df.shape[0]} events without URLs."
        )

        if no_urls_df.empty:
            logging.info("def process_events_without_url(): All events have a URL.")
            return

        # Optionally reduce dataframe size for testing
        if self.config['testing']['status']:
            no_urls_df = no_urls_df.head(self.config['testing']['nof_no_url_events'])

        # Process each event
        for event_row in no_urls_df.itertuples(index=False):
            event_name = event_row.event_name

            # 1) Google search (synchronous, but that's usually fine)
            results_df = self.google_search(event_name)

            # 2) Find best URL
            best_url = self.find_best_url_for_event(event_name, results_df)
            if not best_url:
                logging.info(f"def process_events_without_url(): No URL found for event: {event_name}")
                continue

            # 3) Asynchronously extract text from the URL
            extracted_text = await self.extract_text_with_playwright_async(best_url)
            logging.info(f"def process_events_without_url(): Extracted text from URL: {best_url}")

            # 4) Check relevance
            org_name = event_row.org_name
            keywords_list = event_row.dance_style.split(",") if event_row.dance_style else []
            relevant = self.llm_handler.check_keywords_in_text(
                best_url, extracted_text, org_name, keywords_list
            )

            if relevant:
                # 5) If relevant, determine prompt type and possibly refine text (if facebook)
                prompt_type = "fb" if "facebook" in best_url.lower() or "instagram" in best_url.lower() else "default"
                if prompt_type == "fb":
                    # Refine extracted text
                    extracted_text = self.extract_text_from_fb_url(best_url)

                prompt = self.llm_handler.generate_prompt(best_url, extracted_text, prompt_type)
                llm_response = self.llm_handler.query_llm(prompt)

                if llm_response:
                    parsed_result = self.llm_handler.extract_and_parse_json(llm_response, best_url)
                    if parsed_result:
                        events_df = pd.DataFrame(parsed_result)
                        events_df["url"] = best_url
                        events_df.to_csv("debug/events_df.csv", index=False)

                        # 6) Merge data if new row is a good match
                        new_row = self.select_best_match(event_name, events_df)
                        if new_row is not None:
                            merged_row = self.merge_rows(event_row, new_row)
                            self.update_event_row(merged_row)

                            org_names = merged_row["org_name"]
                            keywords = merged_row["dance_style"]
                            # Write URL to db
                            self.db_handler.write_url_to_db(org_names, keywords, best_url, "", True, 1)

            else:
                logging.info(f"def process_events_without_url(): Event {event_name} and {event_row.event_id} is not relevant.")
                self.db_handler.delete_event_with_event_id(event_row.event_id)

        logging.info("def process_events_without_url(): Finished processing events without URLs.")


    def google_search(self, event_name):
        """
        Finds URLs for an event using Google Search. (Synchronous call)
        """
        location = self.config['location']['epicentre']
        query = f"{event_name} {location}"

        logging.info(f"def google_search(): Performing Google search for query: {query}")
        service = build("customsearch", "v1", developerKey=self.api_key)
        response = (
            service.cse()
            .list(
                q=query,
                cx=self.cse_id,
                num=self.config['search']['gs_num_results'],
            )
            .execute()
        )

        results = []
        if "items" in response:
            for item in response["items"]:
                title = item.get("title")
                url = item.get("link")
                results.append({"event_name": title, "url": url})
            logging.info(f"def google_search(): Found {len(results)} results for query: {query}")
        else:
            logging.info(f"def google_search(): No results found for query: {query}")

        return pd.DataFrame(results)


    def find_best_url_for_event(self, original_event_name, results_df):
        """
        Finds the best URL for an event based on the original event name and a DataFrame of results.

        The function uses fuzzy string matching to compare the original event name with event names in the results DataFrame.
        It ranks URLs based on their source, giving preference to 'allevents' URLs over normal URLs, and normal URLs over 'facebook' or 'instagram' URLs.

        Parameters:
        original_event_name (str): The name of the original event to match against.
        results_df (pandas.DataFrame): A DataFrame containing event names and URLs to search through.

        Returns:
        str: The best matching URL if a match is found with a score of 80 or higher, otherwise an empty string.
        """

        logging.info(f"def find_best_url_for_event(): original_event_name: {original_event_name}")
        best_match = None
        best_score = 0
        best_rank = 0

        for row in results_df.itertuples(index=False):
            logging.info(F"def find_best_url_for_event(): row: {row}")
            score = fuzz.ratio(original_event_name, row.event_name)
            logging.info(F"def find_best_url_for_event(): score: {score}")
            if score < self.config['fuzzywuzzy']['hurdle']:
                continue

            url = str(row.url).lower()
            # Ranking logic: 'allevents' > normal > 'facebook'/'instagram'
            if "allevents" in url:
                rank = 3
            elif "facebook" in url or "instagram" in url:
                rank = 1
            else:
                rank = 2

            if (score > best_score) or (score == best_score and rank > best_rank):
                best_score = score
                best_rank = rank
                best_match = row

        return best_match.url if best_match is not None else ""
    

    async def extract_text_with_playwright_async(self, url: str) -> str:
        """
        Asynchronously extracts text from a web page using Playwright's async API.
        Returns an empty string if any error or if sign-in pages are detected.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.config['crawling']['headless'])
                page = await browser.new_page()

                # Navigate to the URL with a 10-second timeout
                await page.goto(url, timeout=10000)

                # Wait an extra 3 seconds for the page to render
                await page.wait_for_timeout(3000)

                # Detect Google sign-in
                if "accounts.google.com" in page.url.lower():
                    logging.info("Google sign-in detected. Aborting extraction.")
                    await browser.close()
                    return ""

                # Check for sign-in input
                sign_in_input = await page.query_selector("input#identifierId")
                if sign_in_input:
                    logging.info("Google sign-in input detected. Aborting extraction.")
                    await browser.close()
                    return ""

                # Get the page's HTML content
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                extracted_text = " ".join(soup.stripped_strings)
                logging.info(f"def extract_text_with_playwright_async(): Extracted text from {url}: {extracted_text}")

                await browser.close()
                return extracted_text
            

        except PlaywrightTimeoutError:
            logging.error(f"Timeout while extracting text from {url}.")
            return ""
        except Exception as e:
            logging.error(f"Failed to extract text from {url}: {e}")
            return ""

    def select_best_match(self, original_event_name, events_df):
        best_match = None
        best_score = 0
        for row in events_df.itertuples(index=False):
            score = fuzz.ratio(original_event_name, row.event_name)
            if score >= 80 and score > best_score:
                best_score = score
                best_match = row
        return best_match

    def merge_rows(self, original_row, new_row):
        """
        Merges data from new_row into original_row if new fields are longer or original fields are empty.
        """
        merged = original_row._asdict()  # Convert namedtuple to dict
        for col in original_row._fields:
            orig_val = getattr(original_row, col)
            new_val = getattr(new_row, col, None) if hasattr(new_row, col) else None
            if pd.isna(orig_val) or orig_val == "":
                merged[col] = new_val
            elif pd.notna(new_val) and new_val != "":
                if len(str(new_val)) > len(str(orig_val)):
                    merged[col] = new_val
        return merged

    def update_event_row(self, merged_row):
        event_id = merged_row["event_id"]
        update_columns = [col for col in merged_row if col != "event_id"]
        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        update_params = {col: merged_row[col] for col in update_columns}
        update_params["event_id"] = event_id
        self.db_handler.execute_query(update_query, update_params)


# ------------------------------------------------------------------------
# Main entry point - run asynchronously
# ------------------------------------------------------------------------
async def main():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    logging.basicConfig(
        filename=config["logging"]["log_file"],
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    clean_up_instance = CleanUp(config)
    await clean_up_instance.process_events_without_url()

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")


if __name__ == "__main__":
    # Standard approach for async Python scripts
    asyncio.run(main())

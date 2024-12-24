import logging
import os
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import yaml

# Import DatabaseHandler class
from db import DatabaseHandler

class FacebookEventScraper:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration from a YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set up logging
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info("FacebookEventScraper initialized.")

    def get_credentials(self, organization):
        """
        Retrieves credentials for a given organization from the keys file.

        Args:
            organization (str): The organization for which to retrieve credentials.

        Returns:
            tuple: appid_uid, key_pw, access_token for the organization.
        """
        keys_df = pd.read_csv(self.config['input']['keys'])
        keys_df = keys_df[keys_df['organization'] == organization]
        appid_uid, key_pw, access_token = keys_df.iloc[0][['appid_uid', 'key_pw', 'access_token']]
        logging.info(f"def get_credentials(): Retrieved credentials for {organization}.")
        return appid_uid, key_pw, access_token

    def login_to_facebook(self, page):
        """
        Logs into Facebook using credentials from the configuration.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        email, password, _ = self.get_credentials('Facebook')

        page.goto("https://www.facebook.com/login", timeout=60000)
        page.fill("input[name='email']", email)
        page.fill("input[name='pass']", password)
        page.click("button[name='login']")

        page.wait_for_timeout(5000)
        if "login" in page.url:
            logging.error("Login failed. Please check your credentials.")
            return False

        logging.info("def login_to_facebook(): Login successful.")
        return True

    def extract_event_links(self, page, search_url):
        """
        Extracts event links from a search page using Playwright and regex.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.
            search_url (str): The Facebook events search URL.

        Returns:
            set: A set of extracted event links.
        """
        page.goto(search_url, timeout=60000)
        page.wait_for_timeout(5000)

        for _ in range(self.config['crawling']['scroll_depth']):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

        content = page.content()
        links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))
        logging.info(f"def extract_event_links(): Extracted {len(links)} event links from {search_url}.")

        return links

    def extract_event_text(self, page, link):
        """
        Extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.
            link (str): The event link.

        Returns:
            str: The extracted text content.
        """
        page.goto(link, timeout=60000)
        page.wait_for_timeout(5000)
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        extracted_text = ' '.join(soup.stripped_strings)
        logging.info(f"def extract_event_text(): Extracted text from {link}.")

        return extracted_text

    def scrape_events(self, keywords):
        """
        Logs into Facebook, performs searches for keywords, and extracts event links and text.

        Args:
            keywords (list): List of keywords to search for.

        Returns:
            list: A list of tuples containing event links and extracted text.
        """
        base_url = self.config['constants']['fb_base_url']
        location = self.config['constants']['location']
        extracted_text_list = []
        urls_visited = set()

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config['crawling']['headless'])
                context = browser.new_context()
                page = context.new_page()

                if not self.login_to_facebook(page):
                    return []

                for keyword in keywords:
                    search_url = f"{base_url} {location} {keyword}"
                    event_links = self.extract_event_links(page, search_url)
                    logging.info(f"def scrape_events: Used {search_url} to get events")

                    for link in event_links:
                        if link in urls_visited:
                            continue

                        extracted_text = self.extract_event_text(page, link)
                        extracted_text_list.append((link, extracted_text))
                        urls_visited.add(link)

                        # Get second-level links
                        second_level_links = self.extract_event_links(page, link)
                        for second_level_link in second_level_links:
                            if second_level_link in urls_visited:
                                continue

                            second_level_text = self.extract_event_text(page, second_level_link)
                            extracted_text_list.append((second_level_link, second_level_text))
                            urls_visited.add(second_level_link)

                browser.close()

        except Exception as e:
            logging.error(f"Failed to scrape events: {e}")

        logging.info(f"def scrape_events(): Extracted text from {len(extracted_text_list)} events.")

        return extracted_text_list
    

    def save_to_csv(self, extracted_text_list, output_path):
        """
        Saves extracted event links and text to a CSV file.

        Args:
            extracted_text_list (list): A list of tuples containing event links and extracted text.
            output_path (str): The file path to save the CSV.
        """
        df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])
        df.to_csv(output_path, index=False)
        logging.info(f"def save_to_csv(): Extracted text data written to {output_path}")


# Example Usage
if __name__ == "__main__":

    # Initialize scraper and database handler
    scraper = FacebookEventScraper()
    db_handler = DatabaseHandler(scraper.config)
    
    # Scrape events and save to CSV
    keywords = ['bachata']
    extracted_text_list = scraper.scrape_events(keywords)

    output_csv_path = 'output/extracted_text.csv'
    scraper.save_to_csv(extracted_text_list, output_csv_path)

    # Run deduplication and set calendar URLs
    db_handler.dedup()
    db_handler.set_calendar_urls()

    print(f"Extracted {len(extracted_text_list)} events. Data saved to {output_csv_path}.")

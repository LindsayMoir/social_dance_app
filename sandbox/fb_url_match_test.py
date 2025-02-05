from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright
import re
import requests
import yaml

# Import other classes
from fb import FacebookEventScraper
from llm import LLMHandler

# Initialize shared dependencies
config_path = "config/config.yaml"

# Load configuration from a YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Set up logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Instantiate the class libraries
fb_scraper = FacebookEventScraper(config_path=config_path)
llm_handler = LLMHandler(config_path=config_path)


def get_event_links_text():
    """
    Logs into Facebook, performs searches for keywords, and extracts event links and text.

    Args:
        keywords (list): List of keywords to search for.

    Returns:
        list: A list of tuples containing event links and extracted text.
    """
    url = 'https://www.facebook.com/events/1242977450115149/'
    extracted_text_list = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config['crawling']['headless'])
            context = browser.new_context()
            page = context.new_page()

            if not fb_scraper.login_to_facebook(page):
                return []

            event_links = fb_scraper.extract_event_links(page, url)
            logging.info(f"def scrape_events(): Used {url} to get {len(event_links)} event_links\n")
            logging.info(f"def get_event_links_text: events_links are: \n{event_links}")

            extracted_text = fb_scraper.extract_event_text(page, url)
            extracted_text_list.append((url, extracted_text))

            browser.close()

    except Exception as e:
        logging.error(f"def get_event_links(): Failed to scrape events: {e}")

    logging.info(f"def get_event_links(): Extracted text from {len(extracted_text_list)} events.")

    # Call the llm to extract the events
    process_llm_response(url, extracted_text)


def process_llm_response(url, extracted_text):
        """
        Generate a prompt, query a Language Learning Model (LLM), and process the response.

        This method generates a prompt based on the provided URL and extracted text, queries the LLM with the prompt,
        and processes the LLM's response. If the response is successfully parsed, it converts the parsed result into
        a DataFrame, writes the events to the database, and logs the relevant information.

        Args:
            url (str): The URL of the webpage being processed.
            extracted_text (str): The text extracted from the webpage.

        Returns:
            None
        """
        # Generate prompt, query LLM, and process the response.
        prompt = llm_handler.generate_prompt(url, extracted_text, 'fb')
        llm_response = llm_handler.query_llm(prompt, url)

        if llm_response:
            parsed_result = llm_handler.extract_and_parse_json(llm_response, url)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                events_df.to_csv('output/events_df.csv', index=False)
                logging.info(f"def process_llm_response: events_df written to output/events_df.csv.")
        
        else:
            logging.error(f"def process_llm_response: Failed to process LLM response for URL: {url}")


# Example Usage
if __name__ == "__main__":
    
    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")
    get_event_links_text()
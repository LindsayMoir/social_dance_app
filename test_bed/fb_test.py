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
from db import DatabaseHandler
from fb import FacebookEventScraper
from llm import LLMHandler
from scraper import EventSpider

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


def get_event_links_text(keywords):
    """
    Logs into Facebook, performs searches for keywords, and extracts event links and text.

    Args:
        keywords (list): List of keywords to search for.

    Returns:
        list: A list of tuples containing event links and extracted text.
    """
    base_url = config['constants']['fb_base_url']
    location = config['constants']['location']
    extracted_text_list = []

    visited_links = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config['crawling']['headless'])
            context = browser.new_context()
            page = context.new_page()

            if not fb_scraper.login_to_facebook(page):
                return []

            for keyword in keywords:
                search_url = f"{base_url}{location}{keyword}"
                logging.info(f"********search_url********\n{search_url}")

                print("********search_url********\n", search_url)

                event_links = fb_scraper.extract_event_links(page, search_url)
                logging.info(f"def scrape_events(): Used {search_url} to get {len(event_links)} event_links\n")

                for link in event_links:
                    if link not in visited_links:
                        extracted_text = fb_scraper.extract_event_text(page, link)
                        extracted_text_list.append((link, extracted_text))
                        visited_links.add(link)
                        if len(visited_links) > 5:
                            break

                        # Get second-level links
                        second_level_links = fb_scraper.extract_event_links(page, link)

                        # Reverse the order that the links are loooked at. The last ones are the friend ones. 
                        # Those should be the first ones to be looked at.
                        second_level_links = list(second_level_links)
                        second_level_links.reverse()

                        for second_level_link in second_level_links:
                            if second_level_link not in visited_links:
                                second_level_text = fb_scraper.extract_event_text(page, second_level_link)
                                extracted_text_list.append((second_level_link, second_level_text))
                                visited_links.add(second_level_link)
                                if len(visited_links) > 5:
                                    break

            browser.close()

    except Exception as e:
        logging.error(f"def get_event_links(): Failed to scrape events: {e}")

    logging.info(f"def get_event_links(): Extracted text from {len(extracted_text_list)} events.")

    df = pd.DataFrame(extracted_text_list, columns=['link', 'text'])
    df.to_csv('output/links_text_fb.csv', index=False)


# Example Usage
if __name__ == "__main__":
    
    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")
    get_event_links_text(['dance'])
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright
import os
import re
import yaml


# Load configuration from a YAML file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("global: Working directory is: %s", os.getcwd())


def build_search_term(keywords):
    """
    Builds a search term for Facebook events search using a list of keywords.

    Args:
        keywords (list): A list of keywords to search for.

    Returns:
        str: A search term for Facebook events search.
    """
    # Build the search term

    # base url
    base_url = "https://www.facebook.com/search/events?q=events"

    # Location
    location = config['constants']['location']

    # Create search urls for each keyword and submit that to fb_extract_event_links
    event_urls_list = []
    for keyword in keywords:
        search_url = base_url + location + keyword
        logging.info('\n******search_url******\n', search_url)
        event_urls = fb_extract_event_links(search_url)
        event_urls_list.extend(event_urls)

    return set(event_urls_list)


def get_credentials(organization):
    """
    Retrieves credentials for a given organization from the keys file.

    Args:
        organization (str): The organization for which to retrieve credentials.
        key (str): The key for the organization's credentials.
        app_id (str): The app ID for the organization's credentials.
        access_token (str): The access token for the organization's credentials.

    Returns:
        str: The retrieved credentials for the organization.
    """
    # Load the keys file
    keys_df = pd.read_csv(config['input']['keys'])
    appid_uid= keys_df.loc[keys_df['organization'] == organization, 'appid_uid'].values[0]
    key_pw = keys_df.loc[keys_df['organization'] == organization, 'key_pw'].values[0]
    access_token = keys_df.loc[keys_df['organization'] == organization, 'access_token'].values[0]
    
    return appid_uid, key_pw, access_token


def fb_extract_event_links(url):
    """
    Extracts event links from a Facebook events search page using Playwright and regex.
    Includes login functionality if required.

    Args:
        url (str): The URL of the Facebook events search page.
        email (str): Facebook login email.
        password (str): Facebook login password.

    Returns:
        list: A list of extracted event links matching the desired pattern.
    """
    # Load Facebook login credentials from the keys file
    email, password, _ = get_credentials('Facebook')
    event_links = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config['crawling']['headless'])  # Set headless=True to run without GUI
            context = browser.new_context()
            page = context.new_page()

            # Navigate to the Facebook login page
            page.goto("https://www.facebook.com/login", timeout=60000)

            # Perform login
            page.fill("input[name='email']", email)
            page.fill("input[name='pass']", password)
            page.click("button[name='login']")

            # Wait for the login to complete and check if successful
            page.wait_for_timeout(5000)  # Allow time for login to process
            if "login" in page.url:
                logging.error("Login failed. Please check your credentials.")
                return []

            logging.info("Login successful. Navigating to the search URL.")

            # Navigate to the events search URL
            page.goto(url, timeout=60000)
            page.wait_for_timeout(5000)  # Allow additional time for dynamic content to load

            # Scroll to load more content if needed
            for _ in range(config['crawling']['scroll_depth']):  # Adjust the range for more or fewer scrolls
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)

            # Get the final HTML content of the page
            content = page.content()

            # Use regex to extract event links
            event_links = re.findall(r'https://www\.facebook\.com/events/\d+/', content)
            event_links = list(set(event_links))  # Remove duplicates

            browser.close()

    except Exception as e:
        logging.error(f"Failed to extract event links from {url}: {e}")
    
    return event_links


# Example Usage
if __name__ == "__main__":

    # Extract event links
    event_urls_set = build_search_term(keywords=['swing', 'west coast swing'])
    print("Extracted Event Links:")
    print(event_urls_set)

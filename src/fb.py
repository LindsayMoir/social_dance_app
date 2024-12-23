from bs4 import BeautifulSoup
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright
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

print(os.getcwd())

def get_credentials(organization):
    """
    Retrieves credentials for a given organization from the keys file.

    Args:
        organization (str): The organization for which to retrieve credentials.

    Returns:
        tuple: appid_uid, key_pw, access_token for the organization.
    """
    keys_df = pd.read_csv(config['input']['keys'])
    keys_df = keys_df[keys_df['organization'] == organization]
    appid_uid, key_pw, access_token = keys_df.iloc[0][['appid_uid', 'key_pw', 'access_token']]

    return appid_uid, key_pw, access_token


def login_to_facebook(page, email, password):
    """
    Logs into Facebook using the provided credentials.

    Args:
        page: Playwright page instance.
        email (str): Facebook login email.
        password (str): Facebook login password.

    Returns:
        bool: True if login is successful, False otherwise.
    """
    page.goto("https://www.facebook.com/login", timeout=60000)
    page.fill("input[name='email']", email)
    page.fill("input[name='pass']", password)
    page.click("button[name='login']")
    page.wait_for_timeout(5000)
    if "login" in page.url:
        logging.error("Login failed. Please check your credentials.")
        return False
    logging.info("Login successful.")

    return True


def extract_links_and_text(page, url):
    """
    Extracts event links and visible text from a given page.

    Args:
        page: Playwright page instance.
        url (str): URL to extract data from.

    Returns:
        tuple: A set of extracted event links and a list of tuples containing URLs and extracted text.
    """
    event_links = set()
    extracted_text_list = []

    page.goto(url, timeout=60000)
    page.wait_for_timeout(5000)

    # Scroll to load more content if needed
    for _ in range(config['crawling']['scroll_depth']):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)

    # Extract event links using regex
    content = page.content()
    links = re.findall(r'https://www\.facebook\.com/events/\d+/', content)
    event_links.update(links)

    # Extract visible text
    soup = BeautifulSoup(content, 'html.parser')
    extracted_text = ' '.join(soup.stripped_strings)
    extracted_text_list.append((url, extracted_text))

    return event_links, extracted_text_list


def fb_login_and_extract_links(keywords):
    """
    Logs into Facebook once and extracts event links and text for all provided keywords.

    Args:
        keywords (list): List of keywords to search for.

    Returns:
        list: A list of tuples containing URLs and extracted text.
    """
    email, password, _ = get_credentials('Facebook')
    base_url = config['constants']['fb_base_url']
    location = config['constants']['location']

    event_links = set()
    extracted_text_list = []
    urls_visited = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config['crawling']['headless'])
            context = browser.new_context()
            page = context.new_page()

            if not login_to_facebook(page, email, password):
                return []

            for keyword in keywords:
                search_url = f"{base_url} {location} {keyword}"
                logging.info(f"Navigating to search URL: {search_url}")

                links, text_list = extract_links_and_text(page, search_url)
                event_links.update(links)
                extracted_text_list.extend(text_list)

                for link in links:
                    if link not in urls_visited:
                        second_level_links, second_level_text = extract_links_and_text(page, link)
                        event_links.update(second_level_links)
                        extracted_text_list.extend(second_level_text)
                        urls_visited.add(link)

            logging.info(f"Extracted {len(extracted_text_list)} unique event links and texts.")
            browser.close()

    except Exception as e:
        logging.error(f"Failed to extract event links: {e}")

    return extracted_text_list


# Example Usage
if __name__ == "__main__":

    keywords = ['swing', 'west coast swing']
    extracted_text_list = fb_login_and_extract_links(keywords)

    # Create a DataFrame from the extracted text list
    ex_text_df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])

    # Write the DataFrame to a CSV file
    output_csv_path = 'data/extracted_text.csv'
    ex_text_df.to_csv(output_csv_path, index=False)

    logging.info(f"Extracted text data written to {output_csv_path}")

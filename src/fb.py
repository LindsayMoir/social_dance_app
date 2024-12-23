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


def fb_login_and_extract_links(keywords):
    """
    Logs into Facebook once and extracts event links for all provided keywords.

    Args:
        keywords (list): List of keywords to search for.

    Returns:
        set: A set of unique event links.
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

            # Log in to Facebook
            page.goto("https://www.facebook.com/login", timeout=60000)
            page.fill("input[name='email']", email)
            page.fill("input[name='pass']", password)
            page.click("button[name='login']")

            # Wait for login to complete
            page.wait_for_timeout(5000)
            if "login" in page.url:
                logging.error("Login failed. Please check your credentials.")
                return set()

            logging.info("Login successful. Starting event link extraction.")

            # Iterate through keywords and extract event links
            for keyword in keywords:
                search_url = f"{base_url} {location} {keyword}"
                logging.info(f"Navigating to search URL: {search_url}")
                page.goto(search_url, timeout=60000)
                page.wait_for_timeout(5000)

                # Scroll to load more content if needed
                for _ in range(config['crawling']['scroll_depth']):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)

                urls_visited.update(search_url)

                # Extract event links using regex
                content = page.content()
                links = re.findall(r'https://www\.facebook\.com/events/\d+/', content)
                event_links.update(links)

                # Extract the visible text
                for link in event_links:

                    if link in urls_visited:
                        pass
                    else:
                        # Got to the event page
                        page.goto(search_url, timeout=60000)
                        page.wait_for_timeout(5000)

                        # Extract the visible text
                        content = page.content()  # Get the full HTML content
                        soup = BeautifulSoup(content, 'html.parser')
                        extracted_text = ' '.join(soup.stripped_strings)
                        extracted_text_list.append((link, extracted_text))
                        urls_visited.add(link)

                        # Get the second level of links and extract the visible text
                        second_level_links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))
                        for second_level_link in second_level_links:
                            if second_level_link in urls_visited:
                                pass
                            else:
                                page.goto(second_level_link, timeout=60000)
                                page.wait_for_timeout(5000)
                                content = page.content()
                                soup = BeautifulSoup(content, 'html.parser')
                                extracted_text = ' '.join(soup.stripped_strings)
                                extracted_text_list.append((second_level_link, extracted_text))
                                urls_visited.add(second_level_link)

            logging.info(f"Extracted {len(extracted_text_list)} unique event links.")
            browser.close()

    except Exception as e:
        logging.error(f"Failed to extract event links: {e}")

    return extracted_text_list


# Example Usage
if __name__ == "__main__":
    keywords = ['swing', 'west coast swing']
    extracted_text_list = fb_login_and_extract_links(keywords)
    print("Extracted text with url text:")
    print(extracted_text_list)

    # Create a DataFrame from the extracted text list
    ex_text_df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])

    # Write the DataFrame to a CSV file
    output_csv_path = 'data/extracted_text.csv'
    ex_text_df.to_csv(output_csv_path, index=False)

    logging.info(f"Extracted text data written to {output_csv_path}")

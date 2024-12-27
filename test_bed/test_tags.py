from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import yaml
import re

def get_credentials(organization, config):
    """
    Retrieves credentials for a given organization from the keys file.

    Args:
        organization (str): The organization for which to retrieve credentials.
        config (dict): The configuration dictionary containing the path to the keys file.

    Returns:
        tuple: appid_uid, key_pw, access_token for the organization.

    Raises:
        ValueError: If the organization is not found in the keys file.
    """

    keys_df = pd.read_csv(config['input']['keys'])
    try:
        keys_df = keys_df[keys_df['organization'] == organization]
        appid_uid, key_pw, access_token = keys_df.iloc[0][['appid_uid', 'key_pw', 'access_token']]
        return appid_uid, key_pw, access_token
    except IndexError:
        raise ValueError(f"Credentials for organization '{organization}' not found in keys file.")

def analyze_page_text(page):
    """
    Analyzes the page content and returns a dictionary containing tag names 
    as keys and a tuple containing character count and number of links as values.

    Args:
        page: A Playwright page object.

    Returns:
        A dictionary where keys are tag names and values are tuples of 
        (character count, number of links).
    """

    html_content = page.content()
    soup = BeautifulSoup(html_content, 'html.parser')

    tag_data = {}
    link_pattern = re.compile(r'https://www\.facebook\.com/events/\d+/')
    for tag in soup.find_all():
        tag_name = tag.name
        text = tag.get_text(separator=' ', strip=True)
        text_length = len(text)
        links = len(link_pattern.findall(text))
        tag_data[tag_name] = (text_length, links)

    return tag_data

def facebook_login(page, username, password):
    """
    Logs into a Facebook page using Playwright.

    Args:
        page: A Playwright page object.
        username: The username or email address for the Facebook account.
        password: The password for the Facebook account.

    Returns:
        True if login is successful, False otherwise.
    """

    try:
        page.goto("https://www.facebook.com/")

        # Locate and fill the username/email field
        username_field = page.locator('input[name="email"]')
        username_field.fill(username)

        # Locate and fill the password field
        password_field = page.locator('input[name="pass"]')
        password_field.fill(password)

        # Locate and click the login button
        login_button = page.locator('button[type="submit"]')
        login_button.click()

        # Wait for the user's profile to load (adjust this selector as needed)
        page.wait_for_selector('//span[text()="Home"]')

        return True

    except Exception as e:
        print(f"Login failed: {e}")
        return False

# Example usage
with sync_playwright() as p:
    # Load configuration from a YAML file
    with open('config/config.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Set headless mode based on configuration
    headless = config['crawling']['headless']
    browser = p.chromium.launch(headless=headless)
    page = browser.new_page()

    # Target the specific Facebook event using the provided URL
    url = "https://www.facebook.com/events/1084627572499830/"

    try:
        page.goto(url)

        # Check if login is required based on page content or response status (adjust as needed)
        if page.url.endswith("/login.php") or "checkpoint" in page.url:
            print("Login required.")

            # Retrieve credentials from the keys file based on organization (replace with your actual organization)
            organization = "Facebook"  # Replace with your organization name
            appid_uid, key_pw, access_token = get_credentials(organization, config)

            # Attempt to login with retrieved credentials
            if not facebook_login(page, appid_uid, key_pw):
                print("Login failed. Please check your credentials or Facebook login requirements.")
                browser.close()
                exit()

        # Analyze and print tag text lengths
        tag_lengths = analyze_page_text(page)
        for tag, (length, link_count) in tag_lengths.items():
            print(f"{tag}: {length} characters, {link_count} links")

    except Exception as e:
        print(f"An error occurred: {e}")

    browser.close()
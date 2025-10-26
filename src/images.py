# images.py

from pathlib import Path
import asyncio
import json
import logging
import os
import sys
import time
import random
from datetime import datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from io import BytesIO
import pandas as pd
import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from scrapy import Selector
from sqlalchemy import text
import yaml

from db import DatabaseHandler
from llm import LLMHandler
from rd_ext import ReadExtract
from secret_paths import get_auth_file
from scraper_utils import check_keywords

# Constants
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".tiff", ".tif", ".svg", ".webp", ".ico",
    ".avif", ".jfif"
}
# Default headers
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/100.0.0.0 Safari/537.36"
)

# 0. Configure Tesseract binary path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 1. Load environment and configuration
load_dotenv()

# Setup centralized logging
from logging_config import setup_logging
setup_logging('images')

config_path = Path("config/config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logger.info("\n\nStarting images.py ...")

class ImageScraper:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImageScraper")
        self.urls_visited = set()

        # Handlers
        self.llm_handler = LLMHandler(config)
        self.db_handler = self.llm_handler.db_handler  # Use the DatabaseHandler from LLMHandler
        self.keywords_list = self.llm_handler.get_keywords()

        # Directories
        self.download_dir = Path(config.get("image_download_dir", "images/"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Async event loop for Playwright
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize ReadExtract and login
        self.read_extract = ReadExtract(config_path=str(config_path))
        self.loop.run_until_complete(self.read_extract.init_browser())
        if not self.loop.run_until_complete(self._login_to_instagram()):
            self.logger.error("Instagram login failed. Exiting.")
            sys.exit(1)


    def _load_instagram_cookies(self) -> list | None:
        """
        Loads pre-saved Instagram authentication cookies from database or filesystem.

        Priority order:
        1. Database (auth_storage table) - synced across all environments
        2. Render Secret Files (/etc/secrets/) - for Render environment
        3. Local filesystem - for development

        Validates that critical cookies (sessionid, csrftoken) are still valid and not expired.

        Returns:
            list | None: List of cookie dictionaries if valid, None if invalid, missing, or expired.
        """
        try:
            # Use secret_paths to get auth file from database, Render secrets, or filesystem
            auth_file_path = get_auth_file('instagram')
            self.logger.info(f"_load_instagram_cookies(): Reading from {auth_file_path}")

            if not Path(auth_file_path).exists():
                self.logger.warning(f"_load_instagram_cookies(): Auth file not found at {auth_file_path}")
                return None

            with open(auth_file_path, 'r') as f:
                auth_data = json.load(f)

            cookies = auth_data.get('cookies', [])
            if not cookies:
                self.logger.warning("_load_instagram_cookies(): No cookies found in auth data")
                return None

            # Check if critical cookies are present and valid
            cookie_dict = {c['name']: c for c in cookies}
            critical_cookies = ['sessionid', 'csrftoken']

            for cookie_name in critical_cookies:
                if cookie_name not in cookie_dict:
                    self.logger.warning(f"_load_instagram_cookies(): Missing critical cookie '{cookie_name}'")
                    return None

            # Verify expiration times
            current_time = time.time()
            for cookie in cookies:
                expires = cookie.get('expires', -1)
                if expires > 0 and expires < current_time:
                    self.logger.warning(f"_load_instagram_cookies(): Cookie '{cookie['name']}' has expired (expires: {expires}, current: {current_time})")
                    return None

            self.logger.info("_load_instagram_cookies(): Successfully loaded and validated Instagram cookies from database/filesystem")
            return cookies

        except Exception as e:
            self.logger.warning(f"_load_instagram_cookies(): Failed to load cookies: {e}")
            return None

    async def _verify_instagram_session(self) -> bool:
        """
        Verifies that the current Instagram session is valid by making a test request.

        Returns:
            bool: True if session is valid, False otherwise.
        """
        try:
            # Try to fetch Instagram's main page as a validation test
            response = await self.read_extract.page.request.get("https://www.instagram.com/")

            if response.status == 200:
                self.logger.info("_verify_instagram_session(): Instagram session verified successfully")
                return True
            elif response.status == 403 or response.status == 401:
                self.logger.warning(f"_verify_instagram_session(): Session verification failed with status {response.status}")
                return False
            else:
                self.logger.info(f"_verify_instagram_session(): Unexpected status {response.status}, continuing with session")
                return True

        except Exception as e:
            self.logger.warning(f"_verify_instagram_session(): Failed to verify session: {e}")
            return False

    async def _login_to_instagram(self) -> bool:
        if hasattr(self, 'ig_session'):
            self.logger.info("_login_to_instagram(): Reusing existing Instagram session")
            return True

        # First attempt: Try to load and use pre-saved cookies
        self.logger.info("_login_to_instagram(): Attempting to load pre-saved Instagram cookies")
        saved_cookies = self._load_instagram_cookies()

        if saved_cookies:
            try:
                # Add pre-saved cookies to Playwright context
                self.logger.info("_login_to_instagram(): Adding pre-saved cookies to Playwright context")
                await self.read_extract.context.add_cookies(saved_cookies)

                # Verify the session is valid
                if await self._verify_instagram_session():
                    self.logger.info("_login_to_instagram(): Pre-saved cookies verified successfully, using cached session")
                    # Extract cookies for requests.Session
                    cookies = await self.read_extract.context.cookies()
                    session = requests.Session()
                    for ck in cookies:
                        session.cookies.set(ck['name'], ck['value'], domain=ck['domain'])

                    ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
                    ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
                    session.headers.update({
                        "User-Agent": ua,
                        "Referer": "https://www.instagram.com/",
                        "x-csrftoken": session.cookies.get('csrftoken', ''),
                        "x-ig-app-id": ig_app_id,
                        "x-ig-www-claim": "0",
                        "x-requested-with": "XMLHttpRequest",
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "same-origin",
                    })
                    self.ig_session = session
                    return True
                else:
                    self.logger.warning("_login_to_instagram(): Pre-saved cookies failed verification, falling back to fresh login")

            except Exception as e:
                self.logger.warning(f"_login_to_instagram(): Failed to use pre-saved cookies: {e}. Attempting fresh login.")

        # Fallback: Perform fresh login via credentials
        self.logger.info("_login_to_instagram(): Performing fresh Instagram login")
        success = await self.read_extract.login_to_website(
            organization="instagram",
            login_url="https://www.instagram.com/accounts/login/",
            email_selector="input[name='username']",
            pass_selector="input[name='password']",
            submit_selector="button[type='submit']"
        )
        if not success:
            self.logger.error("_login_to_instagram(): Fresh login attempt failed")
            return False

        self.logger.info("_login_to_instagram(): Fresh login successful, setting up session")
        cookies = await self.read_extract.context.cookies()
        session = requests.Session()
        for ck in cookies:
            session.cookies.set(ck['name'], ck['value'], domain=ck['domain'])
        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
        # Use Instagram App ID from environment or fallback to a common one
        ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
        session.headers.update({
            "User-Agent": ua,
            "Referer": "https://www.instagram.com/",
            "x-csrftoken": session.cookies.get('csrftoken', ''),
            "x-ig-app-id": ig_app_id,
            "x-ig-www-claim": "0",
            "x-requested-with": "XMLHttpRequest",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        })
        self.ig_session = session
        return True
    

    def is_image_url(self, url: str) -> bool:
        """
        Determines whether the given URL points to an image file based on its extension.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL has an image file extension, False otherwise.
        """
        path = urlparse(url).path
        ext = Path(path).suffix.lower()
        return ext in IMAGE_EXTENSIONS

    async def _download_image_playwright(self, image_url: str, max_retries: int = 1) -> Path | None:
        """
        Downloads an image using Playwright to maintain Instagram authentication context.
        This method preserves the logged-in session state that requests.Session loses.

        Args:
            image_url (str): The URL of the image to download.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            Path | None: The local file path to the downloaded image if successful, or None if the download failed.
        """
        filename = Path(urlparse(image_url).path).name
        local_path = self.download_dir / filename
        if local_path.exists():
            return local_path

        # Retry mechanism with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying Playwright download after {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)

                # Use Playwright's authenticated context to fetch the image
                self.logger.debug(f"Using Playwright to download Instagram image: {image_url}")
                response = await self.read_extract.page.request.get(image_url)

                if response.status == 200:
                    image_data = await response.body()
                    with local_path.open('wb') as f:
                        f.write(image_data)
                    self.logger.info(f"Downloaded Instagram image via Playwright to {local_path}")
                    return local_path
                elif response.status == 403:
                    self.logger.warning(f"403 Forbidden via Playwright for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All Playwright retry attempts failed for {image_url} - 403 Forbidden")
                        return None
                elif response.status == 429:
                    self.logger.warning(f"Rate limited via Playwright for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All Playwright retry attempts failed for {image_url} - Rate limited")
                        return None
                else:
                    self.logger.error(f"Playwright HTTP error {response.status} for {image_url}")
                    return None

            except Exception as e:
                self.logger.warning(f"Playwright download attempt {attempt + 1} failed for {image_url}: {e}")
                if attempt == max_retries:
                    self.logger.exception(f"All Playwright retry attempts failed for {image_url}")
                    return None

        return None

    def download_image(self, image_url: str, max_retries: int = 1) -> Path | None:
        """
        Downloads an image from the specified URL and saves it to the local download directory.
        Uses Playwright for Instagram images to maintain authentication context.

        Args:
            image_url (str): The URL of the image to download.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            Path | None: The local file path to the downloaded image if successful, or None if the download failed.

        Notes:
            - If the image already exists in the download directory, the existing file path is returned.
            - Uses Playwright for Instagram/Facebook CDN images to maintain authentication state.
            - Falls back to requests session for non-Instagram images.
            - Implements exponential backoff retry mechanism.
        """
        filename = Path(urlparse(image_url).path).name
        local_path = self.download_dir / filename
        if local_path.exists():
            return local_path

        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT

        # Enhanced headers for Instagram images
        headers = {
            "User-Agent": ua,
            "Accept": "image/webp,image/apng,image/*;q=0.8,*/*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.instagram.com/",
            "Origin": "https://www.instagram.com",
            "sec-ch-ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "image",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "cross-site",
        }

        # Add Instagram-specific headers if we have session cookies
        if hasattr(self, 'ig_session') and self.ig_session.cookies:
            csrf_token = self.ig_session.cookies.get('csrftoken', '')
            if csrf_token:
                headers["x-csrftoken"] = csrf_token

            # Use Instagram App ID from environment or fallback to a common one
            ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
            headers["x-ig-app-id"] = ig_app_id

        # Use Playwright for Instagram images to maintain authentication context
        if 'instagram.com' in image_url or 'fbcdn.net' in image_url:
            return self.loop.run_until_complete(self._download_image_playwright(image_url, max_retries))

        # Use requests session for non-Instagram images
        for attempt in range(max_retries + 1):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying download after {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)

                resp = self.ig_session.get(image_url, headers=headers, stream=True, timeout=15)
                resp.raise_for_status()

                with local_path.open('wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

                self.logger.info(f"Downloaded image to {local_path}")
                return local_path

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    self.logger.warning(f"403 Forbidden for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All retry attempts failed for {image_url} - 403 Forbidden")
                        return None
                elif e.response.status_code == 429:
                    self.logger.warning(f"Rate limited for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All retry attempts failed for {image_url} - Rate limited")
                        return None
                else:
                    self.logger.error(f"HTTP error {e.response.status_code} for {image_url}")
                    return None
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {image_url}: {e}")
                if attempt == max_retries:
                    self.logger.exception(f"All retry attempts failed for {image_url}")
                    return None

        return None
        

    def ocr_image_to_text(self, local_path: Path) -> str:
        """
        Performs OCR (Optical Character Recognition) on a given image file and returns the extracted text.

        The method attempts to open the image at the specified local path, resizes it if its largest dimension is less than 800 pixels,
        converts it to grayscale, enhances its contrast, and binarizes it before passing it to Tesseract OCR for text extraction.

        Args:
            local_path (Path): The file path to the local image to be processed.

        Returns:
            str: The text extracted from the image. Returns an empty string if the image cannot be opened or OCR fails.

        Logs:
            Exceptions encountered during image opening or OCR processing are logged with the associated file path.
        """
        try:
            img = Image.open(local_path).convert('RGB')
        except Exception:
            self.logger.exception(f"Cannot open image {local_path}")
            return ""
        max_dim = max(img.size)
        if max_dim < 800:
            scale = 800/max_dim
            img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        gray = img.convert('L')
        bw = ImageEnhance.Contrast(gray).enhance(1.5).point(lambda x:0 if x<128 else 255,'1')
        try:
            return pytesseract.image_to_string(bw, lang='eng', config='--oem 3 --psm 6')
        except Exception:
            self.logger.exception(f"OCR failed on {local_path}")
            return ""


    def get_image_links(self) -> pd.DataFrame:
        """
        Retrieves and combines image links from a CSV file and Instagram links from the database.

        This method reads image link data from a CSV file specified in the configuration and queries the database
        for Instagram-related links using a parameterized SQL query. The results from both sources are concatenated
        into a single DataFrame, and duplicate entries (based on 'link' and 'parent_url') are removed.

        Instagram CDN URLs (fbcdn.net) contain time-limited access tokens that expire after 24-48 hours.
        Old Instagram URLs are filtered out to avoid unnecessary 403/404 errors.

        Returns:
            pd.DataFrame: A DataFrame containing unique image links and their associated metadata from both the CSV and database.
            If the CSV file does not exist, returns an empty DataFrame.

        Logs:
            - An error if the CSV file is not found.
            - An info message with the count of Instagram URLs retrieved from the database.
        """
        csv_path = Path(self.config['input']['images'])
        if not csv_path.exists():
            self.logger.error(f"CSV not found: {csv_path}")
            return pd.DataFrame()
        df_csv = pd.read_csv(csv_path)

        # Parameterized query using sqlalchemy.text for safe ILIKE
        # Only get Instagram URLs from the last 24 hours to avoid stale CDN tokens
        query = text("""
            SELECT link, parent_url, source, keywords, relevant, crawl_try, time_stamp
            FROM urls
            WHERE link ILIKE :link_pattern
              AND time_stamp >= (CURRENT_TIMESTAMP - INTERVAL '24 hours')
        """ )
        params = {'link_pattern': '%instagram%'}
        df_db = pd.read_sql(query, self.db_handler.conn, params=params)
        self.logger.info(f"get_image_links(): Retrieved {df_db.shape[0]} Instagram URLs from the database (filtered to last 24 hours).")

        # Combine CSV and DB results
        df = pd.concat([df_csv, df_db], ignore_index=True)

        # Remove duplicates
        df = df.drop_duplicates(['link', 'parent_url']).reset_index(drop=True)

        # Filter out old fbcdn URLs from CSV as well (they also expire)
        if 'time_stamp' in df.columns:
            df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')
            before_filter = len(df)
            # For fbcdn URLs, only keep recent ones (24 hours)
            df = df[
                ~(df['link'].str.contains('fbcdn.net', case=False, na=False) &
                  (df['time_stamp'].isna() |
                   (pd.Timestamp.now() - df['time_stamp'] > pd.Timedelta(hours=24))))
            ]
            after_filter = len(df)
            if before_filter > after_filter:
                self.logger.info(f"get_image_links(): Filtered out {before_filter - after_filter} stale fbcdn URLs (older than 24 hours).")

        # Restrict to configured maximum
        limit = self.config['crawling']['urls_run_limit']
        if isinstance(limit, int) and limit > 0:
            df = df.iloc[:limit]

        return df


    def process_webpage_url(self, page_url: str, parent_url: str, source: str, keywords: str) -> None:
        """
        Processes a webpage URL by fetching its content, extracting visible text,
        searching for specified keywords, and handling embedded images.

        Args:
            page_url (str): The URL of the webpage to process.
            parent_url (str): The parent page URL for context.
            source (str): The source identifier for categorization.
            keywords(str): The keywords from the original source (.csv or db).
        Returns:
            None
        """
        if page_url in self.urls_visited:
            self.logger.info(f"process_webpage_url(): Already visited {page_url}, skipping.")
            return
        self.urls_visited.add(page_url)

        # Instantiate url_row
        url_row = (page_url, parent_url, source, '', False, 1, datetime.now())

        try:
            resp = requests.get(page_url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            self.logger.error(f"process_webpage_url(): Failed to fetch {page_url}: {e}")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return

        # strip scripts/styles & extract visible text
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        visible_text = soup.get_text(separator=' ')
        text = ' '.join(visible_text.split())
        self.logger.info(f"process_webpage_url(): Extracted text length {len(text)}")
        self.logger.info(f"process_webpage_url(): Extracted text is:\n{text}")

        # Fallback to Playwright for JS‑heavy or too‑short scrapes
        if not text or len(text) < 200 or "instagram.com" in page_url:
            self.logger.info(f"process_webpage_url(): Falling back to Playwright for {page_url}")
            pw_text = self.loop.run_until_complete(
                self.read_extract.extract_event_text(page_url)
            )
            if not pw_text:
                self.logger.info(f"process_webpage_url(): Playwright extraction failed for {page_url}")
                self.db_handler.url_repo.write_url_to_db(url_row)
                return
            text = pw_text
            self.logger.info(f"process_webpage_url(): Playwright-extracted text length {len(text)}")
            self.logger.info(f"process_webpage_url(): Playwright text is:\n{text}")

        if not text:
            self.logger.info(f"process_webpage_url(): No visible text in {page_url}")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return

        # keyword filtering
        found = check_keywords(text, self.keywords_list)
        if not found:
            self.logger.info(f"process_webpage_url(): No keywords found in {page_url}")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return
        self.logger.info(f"process_webpage_url(): Keywords {found} found in {page_url}")

        # LLM processing
        prompt_text, schema_type = self.llm_handler.generate_prompt(page_url, text, 'default')
        status = self.llm_handler.process_llm_response(
            page_url, parent_url, text, source, found, 'default'
        )
        if status:
            self.logger.info(f"process_webpage_url(): LLM succeeded for {page_url}")
        else:
            self.logger.warning(f"process_webpage_url(): LLM produced no events for {page_url}")

        # extract and process images
        # pull down the browser’s rendered DOM
        rendered_html = self.loop.run_until_complete(
            self.read_extract.page.content()
        )
        imgs = Selector(text=rendered_html).xpath('//img/@src').getall()
        img_urls = [urljoin(page_url, u) for u in imgs if self.is_image_url(u)]

        # Establish a minimum size for the img_urls.
        MIN_W, MIN_H = 500, 500

        valid_imgs = []
        for url in img_urls:
            try:
                # Use Playwright for Instagram images, requests for others
                if 'instagram.com' in url or 'fbcdn.net' in url:
                    # Use Playwright's authenticated context for Instagram images
                    async def fetch_instagram_image():
                        response = await self.read_extract.page.request.get(url)
                        if response.status == 200:
                            return await response.body()
                        else:
                            self.logger.info(f"Skipping {url}: Playwright returned status {response.status}")
                            return None

                    image_data = self.loop.run_until_complete(fetch_instagram_image())
                    if image_data:
                        img = Image.open(BytesIO(image_data))
                    else:
                        continue
                else:
                    # Use requests session for non-Instagram images
                    resp = self.ig_session.get(url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content))
            except Exception as e:
                self.logger.info(f"Skipping {url}: failed to download/open - {e}")
                continue

            w, h = img.size
            if w >= MIN_W and h >= MIN_H:
                valid_imgs.append(url)
            else:
                self.logger.info(f"Skipping {url}: too small ({w}x{h})")

        # now process only the valid ones
        for src in valid_imgs[:config['crawling']['max_website_urls']]:
            self.process_image_url(src, page_url, source, keywords)


    def process_image_url(self, image_url:str, parent_url:str, source:str, keywords: str) -> None:
        """
        Processes an image URL by downloading the image, extracting text using OCR, 
        searching for specified keywords, and handling the result with an LLM handler.
        Args:
            image_url (str): The URL of the image to process.
            parent_url (str): The URL of the parent page where the image was found.
            source (str): The source identifier for the image.
            keywords(str): The keywords that were taken from the original input.
        Returns:
            None
        Workflow:
            - Skips processing if the image URL has already been visited or if related events exist.
            - Downloads the image from the provided URL.
            - Extracts text from the image using OCR.
            - Searches for keywords in the extracted text.
            - If keywords are found, generates a prompt and processes the response using the LLM handler.
        """
        self.logger.info(f"process_image_url(): Starting processing for {image_url}")

        # Instantiate url_row
        url_row = (image_url, parent_url, source, keywords, False, 1, datetime.now())

        # Skip if already visited
        if image_url in self.urls_visited:
            self.logger.info(f"process_image_url(): Already visited {image_url}, skipping.")
            return
        self.urls_visited.add(image_url)
        self.logger.info(f"process_image_url(): Marked {image_url} as visited.")

        # Skip if events already exist
        if self.db_handler.check_image_events_exist(image_url):
            self.logger.info(f"process_image_url(): Events already exist for {image_url}, skipping OCR.")
            url_row = (image_url, parent_url, source, keywords, True, 1, datetime.now())
            self.db_handler.url_repo.write_url_to_db(url_row)
            return
        
        # Check and see if we should process this url
        if not self.db_handler.should_process_url(image_url):
            self.logger.info(f"process_image_url(): should_process_url for {image_url}, returned False.")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return

        # Download the image
        self.logger.info(f"process_image_url(): Downloading image from {image_url}")
        path = self.download_image(image_url)
        if not path:
            self.logger.error(f"process_image_url(): download_image() failed for {image_url}")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return
        self.logger.info(f"process_image_url(): Image saved to {path}")

        # Run OCR
        self.logger.info(f"process_image_url(): Running OCR on {path}")
        text = self.ocr_image_to_text(path)
        if not text:
            self.logger.info(f"process_image_url(): No text extracted from {path}, skipping.")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return
        self.logger.info(f"process_image_url(): Extracted text length {len(text)} characters")
        self.logger.info(f"process_image_url(): Extracted text: \n{text}")

        # Keyword filtering
        found = check_keywords(text, self.keywords_list)
        if not found:
            self.logger.info(f"process_image_url(): No relevant keywords in OCR text for {image_url}")
            self.db_handler.url_repo.write_url_to_db(url_row)
            return
        self.logger.info(f"process_image_url(): Found keywords {found} in image {image_url}")

        # LLM prompt & response
        prompt_text, schema_type = self.llm_handler.generate_prompt(image_url, text, 'default')
        if len(prompt_text) > config['crawling']['prompt_max_length']:
            logging.warning(f"def process_image_url(): Prompt for URL {url} exceeds maximum length. Skipping LLM query.")
            return 
        
        self.logger.info(f"process_image_url(): Generated default prompt for image_url: {image_url}")
        status = self.llm_handler.process_llm_response(
            image_url, parent_url, text, source, found, 'default'
        )
        if status:
            self.logger.info(f"process_image_url(): LLM processing succeeded for {image_url}")
        else:
            self.logger.warning(f"process_image_url(): LLM processing did not produce any events for {image_url}")


    def process_images(self) -> None:
        """
        Fetches all image and webpage links, then processes each one.

        Workflow:
        1. Retrieve links from CSV and database via `get_image_links()`.
        2. Log the total number of links to process.
        3. Iterate through each entry, logging its index and details.
        4. Dispatch to `process_image_url()` for direct image URLs,
           or to `process_webpage_url()` for webpage URLs.
        5. Log completion when done.
        """
        # Get the file name of the code that is running
        file_name = os.path.basename(__file__)

        # Count events and urls at start
        start_df = self.db_handler.count_events_urls_start(file_name)

        self.logger.info("process_images(): Starting batch processing of image/webpage links.")
        df = self.get_image_links()
        total = len(df)
        self.logger.info(f"process_images(): Retrieved {total} links to process.")

        for idx, row in enumerate(df.itertuples(index=False), start=1):
            url, parent, source, keywords, *_ = row
            self.logger.info(f"process_images(): [{idx}/{total}] url={url}, parent={parent}, source={source}")

            # Check and see if we should process this url
            if not self.db_handler.should_process_url(url):
                self.logger.info(f"process_images(): should_process_url returned False for url: {url}")
                continue

            if self.is_image_url(url):
                self.logger.info(f"process_images(): Detected direct image URL ({url}), invoking OCR pipeline.")
                self.process_image_url(url, parent, source, keywords)
            else:
                self.logger.info(f"process_images(): Detected webpage URL ({url}), extracting embedded images.")
                self.process_webpage_url(url, parent, source, keywords)

        self.logger.info("process_images(): Completed processing all links.")

        self.db_handler.count_events_urls_end(start_df, __file__)
        logging.info(f"Wrote events and urls statistics to: {file_name}")


if __name__ == '__main__':
    start = datetime.now()

    scraper = ImageScraper(config)
    scraper.process_images()

    scraper.loop.run_until_complete(scraper.read_extract.close())
    scraper.loop.close()

    logger.info(f"Finished in {datetime.now()-start}\n")

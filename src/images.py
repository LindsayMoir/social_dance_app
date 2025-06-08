# images.py

import asyncio
from datetime import datetime
from dateutil import parser as dateparser
from dotenv import load_dotenv
import logging
import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import random
import re
import requests
from scrapy import Selector
import sys
import pytesseract
from urllib.parse import urljoin, urlparse
import yaml

# Import other classes
from db import DatabaseHandler
from llm import LLMHandler
from rd_ext import ReadExtract

# ─── 0. Tell PyTesseract where the Tesseract executable lives ───────────────
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# ─── 1. Load environment and configuration ───────────────────────────────────────
load_dotenv()

with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# ─── 2. Configure logging ────────────────────────────────────────────────────────
log_path = config['logging']['log_file']
log_dir = os.path.dirname(log_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Keep a separate name for the module‐level logger, so we do NOT do shadow `logging`
logger = logging.getLogger(__name__)
logger.info("Starting images.py …")

# ─── 3. Instantiate shared handlers ──────────────────────────────────────────────
db_handler = DatabaseHandler(config)
llm_handler = LLMHandler(config)
read_extract = ReadExtract("config/config.yaml")

# ─── 4. ImageScraper skeleton ────────────────────────────────────────────────────
class ImageScraper:
    def __init__(self, config: dict):
        self.config = config
        # Give each class its own named logger
        self.logger = logging.getLogger(f"{__name__}.ImageScraper")

        # URL tracking
        self.urls_visited = set()

        # LLM keywords for relevance checking
        self.keywords_list = llm_handler.get_keywords()

        # Directory for downloaded images
        self.download_dir = self.config.get("image_download_dir", "images/")
        os.makedirs(self.download_dir, exist_ok=True)

        # login to instagram
        ig_login_status = self.login_to_instagram(page=None)
        if not ig_login_status:
            self.logger.error("Instagram login failed. Cannot proceed with scraping.")
            sys.exit(1)


    def login_to_instagram(self, page):
        """
        Ensure an authenticated Instagram session for downstream requests.

        This method will, on first invocation:
        1. Initialize the Playwright browser context via `self.read_extract.init_browser()`.
        2. Perform a login flow against https://www.instagram.com/accounts/login/ using
            the provided selectors and credentials stored in your config.
        3. If login succeeds, extract the resulting cookies from the Playwright context
            and build a `requests.Session` (`self.ig_session`) seeded with those cookies
            and the necessary headers (`User-Agent`, `x-csrftoken`, `x-ig-app-id`).

        On subsequent calls (when `self.ig_session` already exists), it simply logs that
        the session is being reused.

        Args:
            page (playwright.sync_api.Page): A Playwright Page instance (currently unused,
                but reserved for any direct page-based login operations).

        Returns:
            bool:
                True if an Instagram session is available (either freshly logged in
                or reused), or False if the login attempt failed and scraping cannot proceed.
        """
        # 1) Login & grab session once
        if not hasattr(self, "ig_session"):
            asyncio.run(read_extract.init_browser())
            logged_in = asyncio.run(self.read_extract.login_to_website(
                "instagram",
                login_url="https://www.instagram.com/accounts/login/",
                email_selector="input[name='username']",
                pass_selector="input[name='password']",
                submit_selector="button[type='submit']"
            ))

            if not logged_in:
                self.logger.error("Cannot scrape Instagram without login.")
                return False
            
            # build a requests.Session
            raw_cookies = asyncio.run(self.read_ext.context.cookies())
            self.ig_session = requests.Session()
            for ck in raw_cookies:
                self.ig_session.cookies.set(ck["name"], ck["value"], domain=ck["domain"])
            self.ig_session.headers.update({
                "User-Agent": self.config["crawling"]["user_agent"],
                "x-csrftoken": self.ig_session.cookies.get("csrftoken"),
                "x-ig-app-id": "936619743392459",
            })

            return True
        else:
            self.logger.info("Already logged in to Instagram, reusing session.")
            return True


    def detect_image_only_event(self, response):
        """
        Return a list of image URLs that look like event flyers
        if no textual event cues are found.
        """
        sel = Selector(text=response.text)
        visible_text = [t.strip() for t in sel.xpath('//body//text()').getall() if t.strip()]
        keywords = ['am', 'pm', 'Saturday', 'Sunday', '2025', 'June', 'July']
        has_textual = any(any(kw.lower() in t.lower() for kw in keywords) for t in visible_text)
        if has_textual:
            return []

        image_urls = []
        for img in sel.xpath('//img'):
            src = img.attrib.get('src') or img.attrib.get('data-src')
            if not src:
                continue
            abs_url = urljoin(response.url, src)
            if any(k in abs_url.lower() for k in ['event', 'flyer', 'poster']):
                image_urls.append(abs_url)
                continue
            width = img.attrib.get('width')
            height = img.attrib.get('height')
            try:
                if width and height and (int(width) >= 600 or int(height) >= 400):
                    image_urls.append(abs_url)
            except ValueError:
                pass

        return list(set(image_urls))

    def download_image(self, image_url: str) -> str:
        """
        Download image_url into self.download_dir. Return local filepath or None if it fails.
        """
        filename = os.path.basename(image_url).split('?')[0]
        local_path = os.path.join(self.download_dir, filename)
        if os.path.exists(local_path):
            return local_path

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*;q=0.8"
        }

        try:
            resp = requests.get(image_url, headers=headers, stream=True, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to download {image_url}: {e}")
            return None

        try:
            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded image: {image_url} to {local_path}")
        except Exception as e:
            self.logger.error(f"Error writing image to disk {local_path}: {e}")
            return None

        return local_path
    

    def ocr_image_to_text(self, local_path: str) -> str:
        """
        Run pre-processing and pytesseract OCR. Return raw text (may be empty).
        """
        try:
            img = Image.open(local_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Cannot open image {local_path}: {e}")
            return ""

        # Upscale if very small, convert to grayscale, boost contrast
        max_dim = max(img.size)
        if max_dim < 800:
            scale = 800 / max_dim
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

        gray = img.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)
        bw = enhanced.point(lambda x: 0 if x < 128 else 255, '1')

        try:
            text = pytesseract.image_to_string(bw, lang='eng', config='--oem 3 --psm 6')
        except pytesseract.TesseractError as e:
            self.logger.error(f"OCR failed on {local_path}: {e}")
            return ""

        return text
    

    def process_webpage_url(self, image_url, parent_url, source: str):
        """
        Process a webpage URL:
        1. Download the page.
        2. Extract text and check for keywords.
        3. If no keywords found, skip.
        4. Otherwise, extract image URLs and process each with OCR+LLM.
        """
        self.logger.info(f"Processing webpage URL: {image_url}")

        # Skip if already visited
        if image_url in self.urls_visited:
            self.logger.info(f"Already visited {image_url}, skipping.")
            return

        self.urls_visited.add(image_url)

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {image_url}: {e}")
            return

        # Extract text from the page
        sel = Selector(text=response.text)
        page_text = sel.xpath('//body//text()').getall()
        page_text = ' '.join([t.strip() for t in page_text if t.strip()])

        # Check for keywords
        found_keywords = [
            kw for kw in self.keywords_list
            if kw.lower() in page_text.lower()
        ]
        if not found_keywords:
            self.logger.info(f"No keywords found in webpage {image_url}, skipping.")
            return

        self.logger.info(f"Found keywords {found_keywords} in webpage {image_url}")

        # Generate LLM prompt and process response
        prompt_type = 'default'
        prompt = llm_handler.generate_prompt(image_url, page_text, prompt_type)

        llm_status = llm_handler.process_llm_response(
            image_url,
            parent_url,
            page_text,
            source,
            found_keywords,
            prompt
        )
        
        if llm_status:
            self.logger.info(f"LLM processing succeeded for {image_url}")
        else:
            self.logger.info(f"No events written to DB for {image_url}")

        # Extract image URLs from the page
        image_urls = sel.xpath('//img/@src').getall()
        image_urls = [urljoin(image_url, img) for img in image_urls if self.is_image_url(img)]
        if not image_urls:
            self.logger.info(f"No image URLs found in webpage {image_url}, skipping.")
            return
        self.logger.info(f"Found {len(image_urls)} image URLs in webpage {image_url}")
        # Process each image URL
        for img_url in image_urls:
            # Skip if already visited
            if img_url in self.urls_visited:
                self.logger.info(f"Already visited {img_url}, skipping.")
                continue

            # Process the image URL
            self.process_image_url(image_url, parent_url, source)


    def check_image_events_exist(self, image_url: str) -> bool:
        """
        Check if there are any events associated with the given image URL in the database.
        Checks in events_history and events tables.
        If they exist, in events_history and NOT in events, then copies them to events, 
        provided they are newer than config['clean_up']['old_events']

        Returns
        :param image_url: The image URL to check.
        :return: True if events exist for this image URL, False otherwise.
        """
        sql = """
        SELECT COUNT(*) FROM events
        WHERE image_url = %s
        """
        count = db_handler.execute_query(sql, (image_url,))
        if count and count[0][0] > 0:
            self.logger.info(f"Events already exist for image URL: {image_url}")
            return True
        else:
            self.logger.info(f"No events found for image URL: {image_url} in events table, checking events_history.")
        sql_history = """
        SELECT COUNT(*) FROM events_history
        WHERE image_url = %s
        """
        count_history = db_handler.execute_query(sql_history, (image_url,))
        if count_history and count_history[0][0] > 0:
            self.logger.info(f"Events found for image URL: {image_url} in events_history table, copying to events.")
            # Copy events from history to events if they are newer than the configured threshold
            sql_copy = """
            INSERT INTO events (image_url, event_data, created_at)
            SELECT image_url, event_data, created_at
            FROM events_history
            WHERE image_url = %s AND created_at > NOW() - INTERVAL %s
            ON CONFLICT (image_url) DO NOTHING
            """
            db_handler.execute_query(sql_copy, (image_url, self.config['clean_up']['old_events']))
            return True
        else:
            self.logger.info(f"No events found for image URL: {image_url} in events_history table.")
            return False


    def process_image_url(self, url, parent_url, source):
        """
        Process a single image URL:
        1. Download the image.
        2. Run OCR to extract text.
        3. Check for keywords in the extracted text.
        4. Generate LLM prompt and process response.
        """
        self.logger.info(f"def process_image_url(): Processing image URL: {url}")

        # Skip if already visited
        if url in self.urls_visited:
            self.logger.info(f"def process_image_url(): Already visited {url}, skipping.")
            return
        self.urls_visited.add(url)

        # See if there are events associated with this actual image URL in the database alreaady
        # We can avoid scraping, ocring etc. if we already have events for this image
        events_exist = self.check_image_events_exist(url)
        if events_exist:
            self.logger.info(f"def process_image_url(): Events already exist for {url}, skipping further processing.")
            return

        # Download the image
        local_path = self.download_image(url)
        if not local_path:
            self.logger.error(f"def process_image_url(): Failed to download image from {url}")
            return

        # Run OCR on the downloaded image
        extracted_text = self.ocr_image_to_text(local_path)
        if not extracted_text:
            self.logger.info(f"def process_image_url(): No text extracted from {url}, skipping.")
            return

        # Check for keywords
        found_keywords = [
            kw for kw in self.keywords_list
            if kw.lower() in extracted_text.lower()
        ]
        if not found_keywords:
            self.logger.info(f"def process_image_url(): No keywords found in image {url}, skipping.")
            return

        self.logger.info(f"def process_image_url(): Found keywords {found_keywords} in image {url}")

        # Generate LLM prompt and process response
        prompt_type = 'default'
        prompt = llm_handler.generate_prompt(url, extracted_text, prompt_type)
        llm_status = llm_handler.process_llm_response(
            url,
            parent_url,  # No parent URL for standalone images
            extracted_text,
            source,
            found_keywords,
            prompt
        )
        
        if llm_status:
            self.logger.info(f"def process_image_url(): LLM processing succeeded for {url}")
        else:
            self.logger.info(f"def process_image_url(): No events written to DB for {url}")


    def is_image_url(self, url: str) -> bool:
        """
        Return True if `url` ends in a known image file extension.

        Checks only the path portion (ignores query strings or fragments),
        against a set of common image extensions.

        Examples:
            >>> is_image_url("https://example.com/foo.jpg")
            True
            >>> is_image_url("https://example.com/foo.jpg?size=large")
            True
            >>> is_image_url("https://example.com/page.html")
            False
            >>> is_image_url("https://example.com/image.SVG#frag")
            True
        """
        logging.info(f"is_image_url(): Checking URL for image type if any: {url}")

        # Parse out only the path (so we ignore any query params or fragments)
        path = urlparse(url).path
        # Extract the extension (including the leading dot), in lowercase
        ext = os.path.splitext(path)[1].lower()
        # A set of all extensions you consider to be “images”
        IMAGE_EXTENSIONS = {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".tiff", ".tif", ".svg", ".webp", ".ico",
            ".avif", ".jfif"
        }
        return ext in IMAGE_EXTENSIONS


    def get_image_links(self):
        """
        Read in the images.csv
        Get the image links from the db.
        This method is a placeholder for any future logic to scrape image links.
        """
        # images.csv is populated with image URLs or webpagte links that likely has images
        if not os.path.exists(self.config['input']['images']):
            self.logger.error(f"Images CSV file not found: {self.config['input']['images']}")
            return

        image_links_csv_df = pd.read_csv(self.config['input']['images'])
        if image_links_csv_df.empty:
            self.logger.info("No images found in CSV.")
            return

        self.logger.info(f"Found {len(image_links_csv_df)} images in CSV to process.")

        image_links_db_df

        # Pull all instagram URLs from your urls table
        sql = """
        SELECT *
        FROM urls
        WHERE link ILIKE '%instagram%'
        """
        image_links_db_df = pd.read_sql(sql, db_handler.conn)

        if image_links_db_df.empty:
            self.logger.info("driver_img_urls(): No Instagram URLs found in DB.")
            return

        self.logger.info(f"driver_img_urls(): Found {len(image_links_db_df)} Instagram URLs to process.")

        # Append the two DataFrames
        images_df = pd.concat([image_links_csv_df, image_links_db_df], ignore_index=True)
        images_df = images_df.drop_duplicates(subset=['link', 'parent_url'], keep='last').reset_index(drop=True)

        return images_df


    def process_images(self):
        """
        Processes image links from a CSV file by scraping and handling each image or webpage.

        This method reads image data from a CSV file using `get_image_links()`, then iterates over each entry.
        For each image link, it checks if the URL has already been visited. If not, it determines whether the
        link is a direct image URL or a webpage. Direct image URLs are processed with `process_image_url()`,
        while webpage links are handled with `process_webpage_url()`.

        Returns:
            None
        """
        # Get images
        images_df = self.get_image_links()

        # Process each image link
        for idx, image_url, parent_url, source, keywords, relevant, crawl_try in images_df.itertuples():

            # Skip if already visited
            if image_url in self.urls_visited:
                self.logger.info(f"process_images(): Already visited {image_url}, skipping.")
                continue

            # See if this is an image URL or a webpage link
            if self.is_image_url(image_url):
                self.process_image_url(image_url, parent_url, source)
            else:
                # handle it as a webpage…
                self.process_webpage_url(image_url, parent_url, source)


# ─── 5. Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting images.py ...")
    start_time = datetime.now()
    file_name = os.path.basename(__file__)

    # Track row counts in DB before we run
    start_df = db_handler.count_events_urls_start(file_name)

    # Initialize and run
    img_handler = ImageScraper(config=config)
    img_handler.process_images()

    # Track row counts after
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logger.info(f"__main__: Finished images.py at {end_time}")
    elapsed = end_time - start_time
    logger.info(f"__main__: Elapsed time: {elapsed}")

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
from urllib.parse import urljoin
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


    def images_from_csv(self):
        """
        Read in the images.csv and scrape the images.
        """
        df = pd.read_csv(self.config['input']['images'])
        for idx, source, keywords, image_url, parent_url in df.itertuples():
            # 1. Download the image
            local_path = self.download_image(image_url)
            logger.info(f"images_from_csv(): Downloaded {image_url} to {local_path}")
            
            if local_path is None:
                # Download failed—skip OCR and move to next row
                self.logger.error(
                    f"images_from_csv(): Skipping OCR because download_image returned None for {image_url}"
                )
                continue

            # 2. Run OCR only if we actually have a file path
            extracted_text = self.ocr_image_to_text(local_path)
            self.logger.info(f"images_from_csv(): extracted text is:\n{extracted_text}")

            # 3. Check for any of the keywords in OCR’d text
            found_keywords = [
                kw for kw in self.keywords_list
                if kw.lower() in extracted_text.lower()
            ]
            if not found_keywords:
                self.logger.info(
                    f"images_from_csv(): No keywords found in image_url: {image_url} (parent: {parent_url})"
                )
                continue
            
            self.logger.info(
                f"images_from_csv(): Found keywords {found_keywords} in image {image_url} (parent: {parent_url})"
            )

            # 4. Generate LLM prompt and process
            prompt_type = 'default'
            prompt = llm_handler.generate_prompt(image_url, extracted_text, prompt_type)
            llm_status = llm_handler.process_llm_response(
                image_url,
                parent_url,
                extracted_text,
                source,
                found_keywords,
                prompt
            )
            if llm_status:
                self.logger.info(f"images_from_csv(): LLM succeeded for {image_url}")
            else:
                self.logger.info(
                    f"images_from_csv(): No events written to DB for {image_url} (parent: {parent_url})"
                )


    def driver_img_urls(self):
        """
        From Instagram URLs in the DB, do the following for each link:
        1. Skip if already visited.
        2. Extract the page text (via ReadExtract.extract_event_text) and check for any of self.keywords_list.
        3. If no keywords are found, skip.
        4. Otherwise, fetch the profile’s JSON feed (?__a=1&__d=dis) using the already-initialized
            self.ig_session and extract each image URL.
        5. Pass each image URL to self.process_image_url for OCR+LLM processing.
        """
        # 1) Pull all instagram URLs from your urls table
        sql = """
        SELECT *
        FROM urls
        WHERE link ILIKE '%instagram%'
        """
        ig_df = pd.read_sql(sql, db_handler.conn)

        if ig_df.empty:
            self.logger.info("driver_img_urls(): No Instagram URLs found in DB.")
            return

        self.logger.info(f"driver_img_urls(): Found {len(ig_df)} Instagram URLs to process.")

        # 2) Process each URL
        for _, row in ig_df.iterrows():
            url = row["link"]

            # 2a) Skip duplicates
            if url in self.urls_visited:
                self.logger.info(f"driver_img_urls(): Already visited {url}, skipping.")
                continue
            self.urls_visited.add(url)
            self.logger.info(f"driver_img_urls(): Processing Instagram URL: {url}")

            # 2b) Extract page text and look for keywords
            page_text = asyncio.run(read_extract.extract_event_text(url))
            if not page_text:
                self.logger.info(f"driver_img_urls(): No text extracted from {url}, skipping.")
                continue

            found_keywords = [
                kw for kw in self.keywords_list
                if kw.lower() in page_text.lower()
            ]
            if not found_keywords:
                self.logger.info(f"driver_img_urls(): No keywords {self.keywords_list} found on {url}, skipping.")
                continue

            self.logger.info(f"driver_img_urls(): Keywords {found_keywords} found on {url}; fetching images.")

            # 2c) Fetch the JSON feed for that profile or post
            base = url.split('?')[0].rstrip('/')
            json_url = f"{base}/?__a=1&__d=dis"
            try:
                resp = self.ig_session.get(json_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                edges = data["graphql"]["user"]["edge_owner_to_timeline_media"]["edges"]
                image_urls = [edge["node"]["display_url"] for edge in edges]
            except Exception as e:
                self.logger.error(f"driver_img_urls(): Failed to fetch JSON for {url}: {e}")
                continue

            # 2d) Hand off each image URL to your OCR+LLM pipeline
            for img_url in image_urls:
                self.process_image_url(img_url)


# ─── 5. Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting images.py ...")
    start_time = datetime.now()
    file_name = os.path.basename(__file__)

    # Track row counts in DB before we run
    start_df = db_handler.count_events_urls_start(file_name)

    # Initialize and run
    img_handler = ImageScraper(config=config)
    img_handler.images_from_csv()
    img_handler.driver_img_urls()

    # Track row counts after
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logger.info(f"__main__: Finished images.py at {end_time}")
    elapsed = end_time - start_time
    logger.info(f"__main__: Elapsed time: {elapsed}")

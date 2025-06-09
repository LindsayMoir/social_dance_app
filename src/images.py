# images.py

from pathlib import Path
import asyncio
import logging
import sys
from datetime import datetime
from urllib.parse import urljoin, urlparse

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
config_path = Path("config/config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)

# 2. Configure root logger
log_path = Path(config['logging']['log_file'])
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_path), filemode='a', level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("\n\nStarting images.py ...")

class ImageScraper:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImageScraper")
        self.urls_visited = set()

        # Handlers
        self.db_handler = DatabaseHandler(config)
        self.llm_handler = LLMHandler(config)
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

    async def _login_to_instagram(self) -> bool:
        if hasattr(self, 'ig_session'):
            self.logger.debug("Reusing Instagram session")
            return True
        success = await self.read_extract.login_to_website(
            organization="instagram",
            login_url="https://www.instagram.com/accounts/login/",
            email_selector="input[name='username']",
            pass_selector="input[name='password']",
            submit_selector="button[type='submit']"
        )
        if not success:
            return False
        cookies = await self.read_extract.context.cookies()
        session = requests.Session()
        for ck in cookies:
            session.cookies.set(ck['name'], ck['value'], domain=ck['domain'])
        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
        session.headers.update({
            "User-Agent": ua,
            "x-csrftoken": session.cookies.get('csrftoken'),
            "x-ig-app-id": self.config.get('crawling', {}).get('ig_app_id', ''),
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

    def download_image(self, image_url: str) -> Path | None:
        """
        Downloads an image from the specified URL and saves it to the local download directory.

        Args:
            image_url (str): The URL of the image to download.

        Returns:
            Path | None: The local file path to the downloaded image if successful, or None if the download failed.

        Notes:
            - If the image already exists in the download directory, the existing file path is returned.
            - Uses a custom User-Agent from configuration if available, otherwise uses a default.
            - Logs the outcome of the download attempt.
        """
        filename = Path(urlparse(image_url).path).name
        local_path = self.download_dir / filename
        if local_path.exists():
            return local_path
        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
        headers = {"User-Agent": ua, "Accept": "image/webp,image/apng,image/*;q=0.8"}
        try:
            resp = self.ig_session.get(image_url, headers=headers, stream=True, timeout=15)
            resp.raise_for_status()
            with local_path.open('wb') as f:
                for chunk in resp.iter_content(8192): f.write(chunk)
            self.logger.info(f"Downloaded image to {local_path}")
            return local_path
        except Exception:
            self.logger.exception(f"Failed to download image {image_url}")
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
        
        
    def check_image_events_exist(self, image_url: str) -> bool:
        """
        Determines whether there are any events associated with the specified image URL.

        The method performs the following steps:
        1. Checks if any events with the given image URL exist in the `events` table.
           - If found, returns True.
        2. If not found, checks the `events_history` table for events with the given image URL.
           - If not found in history, returns False.
        3. If found in history, copies recent events (within a configurable number of days)
           from `events_history` to `events`, then returns True.

        Args:
            image_url (str): The URL of the image to check for associated events.

        Returns:
            bool: True if events exist or were copied from history; False otherwise.

        Side Effects:
            - May insert records into the `events` table if recent history events are found.
            - Logs actions and outcomes at each step.
        """
        # 1) Check live events table
        sql_live = """
        SELECT COUNT(*)
          FROM events
         WHERE url = :url
        """
        params = {'url': image_url}
        live = self.db_handler.execute_query(sql_live, params)
        if live and live[0][0] > 0:
            self.logger.info(f"Events already exist for URL: {image_url}")
            return True

        # 2) Check history table
        sql_hist = """
        SELECT COUNT(*)
          FROM events_history
         WHERE url = :url
        """
        hist = self.db_handler.execute_query(sql_hist, params)
        if not (hist and hist[0][0] > 0):
            self.logger.info(f"No history events for URL: {image_url}")
            return False

        # 3) Copy recent history into events
        sql_copy = """
        INSERT INTO events (
            event_name, dance_style, description, day_of_week,
            start_date, end_date, start_time, end_time,
            source, location, price, url,
            event_type, address_id, time_stamp
        )
        SELECT
            event_name, dance_style, description, day_of_week,
            start_date, end_date, start_time, end_time,
            source, location, price, url,
            event_type, address_id, time_stamp
          FROM events_history
         WHERE url = :url
           AND time_stamp > NOW() - (:days * INTERVAL '1 day')
        """
        params_copy = {
            'url':  image_url,
            'days': self.config['clean_up']['old_events']
        }
        self.db_handler.execute_query(sql_copy, params_copy)
        self.logger.info(f"Copied history events into events for URL: {image_url}")
        return True


    def get_image_links(self) -> pd.DataFrame:
        """
        Retrieves and combines image links from a CSV file and Instagram links from the database.

        This method reads image link data from a CSV file specified in the configuration and queries the database
        for Instagram-related links using a parameterized SQL query. The results from both sources are concatenated
        into a single DataFrame, and duplicate entries (based on 'link' and 'parent_url') are removed.

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
        query = text("""
            SELECT link, parent_url, source, keywords
            FROM urls
            WHERE link ILIKE :link_pattern
        """ )
        params = {'link_pattern': '%instagram%'}
        df_db = pd.read_sql(query, self.db_handler.conn, params=params)
        self.logger.info(f"get_image_links(): Retrieved {df_db.shape[0]} Instagram URLs from the database.")

        # Combine CSV and DB results
        df = pd.concat([df_csv, df_db], ignore_index=True)

        # Remove duplicates
        df = df.drop_duplicates(['link', 'parent_url']).reset_index(drop=True)

        # Restrict to configured maximum
        limit = self.config['crawling']['urls_run_limit']
        if isinstance(limit, int) and limit > 0:
            df = df.iloc[:limit]
            
        return df


    def process_webpage_url(self, page_url:str, parent_url:str, source:str) -> None:
        """
        Processes a webpage URL by fetching its content, extracting text, searching for specified keywords,
        and handling images found on the page.
        Args:
            page_url (str): The URL of the webpage to process.
            parent_url (str): The URL of the parent page from which this page was discovered.
            source (str): The source identifier or label for tracking the origin of the request.
        Returns:
            None
        Workflow:
            - Skips processing if the URL has already been visited.
            - Fetches the webpage content using an HTTP GET request.
            - Extracts and concatenates all text from the page body.
            - Searches for any keywords from the predefined list within the page text.
            - If keywords are found, generates a prompt and processes the LLM response.
            - Extracts all image URLs from the page and processes each valid image URL.
        Exceptions:
            - Logs and skips the page if fetching the content fails.
        """
        if page_url in self.urls_visited: return
        self.urls_visited.add(page_url)

        try:
            resp = requests.get(page_url, timeout=10); resp.raise_for_status()
        except Exception:
            self.logger.exception(f"Failed to fetch page {page_url}"); return
        
        text = ' '.join(Selector(text=resp.text).xpath('//body//text()').getall()).strip()
        if not text: 
            logger.info(f"process_webpage_url(): No text extracted from url: {page_url}")
            return

        found = [kw for kw in self.keywords_list if kw.lower() in text.lower()]
        if not found: 
            logger.info(f"process_webpage_url(): No keywords found in url: {page_url}")
            return

        prompt = self.llm_handler.generate_prompt(page_url, text, 'default')
        self.llm_handler.process_llm_response(page_url, parent_url, text, source, found, prompt)

        imgs = Selector(text=resp.text).xpath('//img/@src').getall()
        # build full URLs and filter by extension
        img_urls = [urljoin(page_url, i) for i in imgs if self.is_image_url(i)]
        logger.info(f"process_webpage_url(): Extracted {len(img_urls)} image urls, from url: {page_url}")

        # limit to configured maximum
        max_imgs = self.config.get('crawling', {}).get('max_website_urls', len(img_urls))
        logger.info(f"process_webpage_url(): Only processing {max_imgs} due to config constraints.")
        for src in img_urls[:max_imgs]:
            logger.info(f"process_webpage_url(): Now processing image {src}")
            self.process_image_url(src, page_url, source)


    def process_image_url(self, image_url:str, parent_url:str, source:str) -> None:
        """
        Processes an image URL by downloading the image, extracting text using OCR, 
        searching for specified keywords, and handling the result with an LLM handler.
        Args:
            image_url (str): The URL of the image to process.
            parent_url (str): The URL of the parent page where the image was found.
            source (str): The source identifier for the image.
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

        # Skip if already visited
        if image_url in self.urls_visited:
            self.logger.debug(f"process_image_url(): Already visited {image_url}, skipping.")
            return
        self.urls_visited.add(image_url)
        self.logger.debug(f"process_image_url(): Marked {image_url} as visited.")

        # Skip if events already exist
        if self.check_image_events_exist(image_url):
            self.logger.info(f"process_image_url(): Events already exist for {image_url}, skipping OCR.")
            return

        # Download the image
        self.logger.info(f"process_image_url(): Downloading image from {image_url}")
        path = self.download_image(image_url)
        if not path:
            self.logger.error(f"process_image_url(): download_image() failed for {image_url}")
            return
        self.logger.debug(f"process_image_url(): Image saved to {path}")

        # Run OCR
        self.logger.info(f"process_image_url(): Running OCR on {path}")
        text = self.ocr_image_to_text(path)
        if not text:
            self.logger.info(f"process_image_url(): No text extracted from {path}, skipping.")
            return
        self.logger.debug(f"process_image_url(): Extracted text length {len(text)} characters")

        # Keyword filtering
        found = [kw for kw in self.keywords_list if kw.lower() in text.lower()]
        if not found:
            self.logger.info(f"process_image_url(): No relevant keywords in OCR text for {image_url}")
            return
        self.logger.info(f"process_image_url(): Found keywords {found} in image {image_url}")

        # LLM prompt & response
        prompt = self.llm_handler.generate_prompt(image_url, text, 'default')
        self.logger.debug(f"process_image_url(): Generated prompt for LLM: {prompt!r}")
        status = self.llm_handler.process_llm_response(
            image_url, parent_url, text, source, found, prompt
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
        self.logger.info("process_images(): Starting batch processing of image/webpage links.")
        df = self.get_image_links()
        total = len(df)
        self.logger.info(f"process_images(): Retrieved {total} links to process.")

        for idx, row in enumerate(df.itertuples(index=False), start=1):
            url, parent, source, *_ = row
            self.logger.debug(f"process_images(): [{idx}/{total}] url={url}, parent={parent}, source={source}")

            if self.is_image_url(url):
                self.logger.info(f"process_images(): Detected direct image URL ({url}), invoking OCR pipeline.")
                self.process_image_url(url, parent, source)
            else:
                self.logger.info(f"process_images(): Detected webpage URL ({url}), extracting embedded images.")
                self.process_webpage_url(url, parent, source)

        self.logger.info("process_images(): Completed processing all links.")


if __name__ == '__main__':
    start = datetime.now()
    scraper = ImageScraper(config)
    scraper.process_images()
    scraper.db_handler.count_events_urls_end({}, __file__)
    logger.info(f"Finished in {datetime.now()-start}\n")

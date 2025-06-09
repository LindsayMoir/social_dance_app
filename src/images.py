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
        path = urlparse(url).path
        ext = Path(path).suffix.lower()
        return ext in IMAGE_EXTENSIONS

    def download_image(self, image_url: str) -> Path | None:
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
        Check if there are any events associated with the given image URL in the database.
        If found in `events`, return True. Otherwise, if found in `events_history`,
        copy recent ones into `events` and return True. Else return False.
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
        Combine CSV-sourced links with DB Instagram links into a single DataFrame.
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

        # Combine CSV and DB results, then dedup
        df = pd.concat([df_csv, df_db], ignore_index=True)
        return df.drop_duplicates(['link','parent_url'])
    

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
        if not text: return

        found = [kw for kw in self.keywords_list if kw.lower() in text.lower()]
        if not found: return

        prompt = self.llm_handler.generate_prompt(page_url, text, 'default')
        self.llm_handler.process_llm_response(page_url, parent_url, text, source, found, prompt)

        imgs = Selector(text=resp.text).xpath('//img/@src').getall()
        # build full URLs and filter by extension
        img_urls = [urljoin(page_url, i) for i in imgs if self.is_image_url(i)]
        
        # limit to configured maximum
        max_imgs = self.config.get('crawling', {}).get('max_website_urls', len(img_urls))
        for src in img_urls[:max_imgs]:
            self.process_image_url(src, page_url, source)


    def process_image_url(self, image_url:str, parent_url:str, source:str) -> None:
        if image_url in self.urls_visited: return
        self.urls_visited.add(image_url)
        if self.check_image_events_exist(image_url): return
        path = self.download_image(image_url);  # download
        if not path: return
        text = self.ocr_image_to_text(path)
        if not text: return
        found = [kw for kw in self.keywords_list if kw.lower() in text.lower()]
        if not found: return
        prompt = self.llm_handler.generate_prompt(image_url, text, 'default')
        self.llm_handler.process_llm_response(image_url, parent_url, text, source, found, prompt)

    def process_images(self) -> None:
        df = self.get_image_links()
        for row in df.itertuples(index=False):
            url, parent, source, *_ = row
            if self.is_image_url(url):
                self.process_image_url(url, parent, source)
            else:
                self.process_webpage_url(url, parent, source)

if __name__ == '__main__':
    start = datetime.now()
    scraper = ImageScraper(config)
    scraper.process_images()
    scraper.db_handler.count_events_urls_end({}, __file__)
    logger.info(f"Finished in {datetime.now()-start}\n")

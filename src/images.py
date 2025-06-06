# images.py

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
import pytesseract
from urllib.parse import urljoin
import yaml

# Import other classes
from db import DatabaseHandler
from llm import LLMHandler

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

# Keep a separate name for the module‐level logger, so we do NOT shadow `logging`
logger = logging.getLogger(__name__)
logger.info("Starting images.py …")

# ─── 3. Instantiate shared handlers ──────────────────────────────────────────────
db_handler = DatabaseHandler(config)
llm_handler = LLMHandler(config)

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
        self.download_dir = self.config.get("image_download_dir", "downloads/")
        os.makedirs(self.download_dir, exist_ok=True)


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
        From each stored URL or list, call detect_image_only_event → download → ocr → parse → store.
        """
        self.logger.info("driver_img_urls() not implemented yet.")
        # e.g.:
        # for response in self.get_all_responses():
        #     image_urls = self.detect_image_only_event(response)
        #     for img_url in image_urls:
        #         local = self.download_image(img_url)
        #         if not local: continue
        #         raw = self.ocr_image_to_text(local)
        #         data = self.parse_event_text(raw)
        #         self.store_event(data)

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

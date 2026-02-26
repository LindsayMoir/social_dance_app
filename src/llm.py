"""
llm.py

This module defines the LLMHandler class, which facilitates interactions with a 
Language Learning Model (LLM) for processing event-related data. It integrates 
with a PostgreSQL database to store and update event information, uses OpenAI 
for language model queries, and leverages configuration settings loaded from 
a YAML file.

PROMPT MAPPING SYSTEM:
    This module implements a flexible prompt mapping system that supports both
    simple key-based prompts and URL-based prompts for site-specific customization:
    
    Simple Key Prompts:
        - 'fb' → prompts/fb_prompt.txt
        - 'default' → prompts/default.txt  
        - 'images' → prompts/images_prompt.txt
        - 'dedup' → prompts/dedup_prompt.txt
        
    URL-Based Prompts (for site-specific customization):
        - 'https://gotothecoda.com/calendar' → prompts/the_coda_prompt.txt
        - 'https://www.bardandbanker.com/live-music' → prompts/bard_and_banker_prompt.txt
        - 'https://www.debrhymerband.com/shows' → prompts/deb_rhymer_prompt.txt
        - 'https://vbds.org/other-dancing-opportunities/' → prompts/default.txt
        
    The mapping is configured in config/config.yaml under the 'prompts' section.
    If a prompt_type is not found, it automatically falls back to the 'default' prompt.

Classes:
    LLMHandler:
        - Initializes with configuration and ensures a DatabaseHandler instance is available.
        - Provides methods to drive relevance determination, 
          check keywords in text, and process responses from the LLM.
        - Generates prompts for the LLM based on extracted event text and context.
        - Queries the LLM using OpenAI's API and processes the model's responses.
        - Extracts and parses JSON data from LLM responses to obtain structured event details.
        - Contains utility methods for validating and processing LLM output.
        - The `driver` method coordinates end-to-end processing for a given URL,
          including querying the LLM, evaluating relevance, and updating the database.

Usage Example:
    if __name__ == "__main__":
        # Setup centralized logging
        from logging_config import setup_logging
        setup_logging('llm')

        # Load configuration
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Instantiate the LLM handler
        llm_handler = LLMHandler(config_path="config/config.yaml")

        # Example of using the driver method with test data
        test_url = "https://example.com/event"
        search_term = "sample search term"
        extracted_text = "Sample extracted event text..."
        source = "Sample Organization"
        keywords = ["dance", "event"]

        llm_handler.driver(test_url, search_term, extracted_text, source, keywords)

Dependencies:
    - openai: For interacting with the OpenAI API to query the LLM.
    - pandas: For handling tabular data and reading/writing CSV files.
    - yaml: For loading configuration from YAML files.
    - logging: For logging debug and error messages.
    - json, re, os, datetime: Standard libraries for JSON parsing, regular expressions,
      operating system interactions, and time handling.
    - DatabaseHandler from db module: For database interactions.

Note:
    - The module reads configuration from 'config/config.yaml' by default.
    - LLMHandler expects valid API keys and database credentials specified in the 
      configuration and keys files.
    - Logging should be configured in the main execution context to capture log messages.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()
import json
import logging
from mistralai import Mistral
import numpy as np
from openai import OpenAI
import os
import openai
import pandas as pd
import re
import requests
from sqlalchemy.exc import SQLAlchemyError
import yaml

from db import DatabaseHandler


class LLMHandler:
    def __init__(self, config_path=None):
        # Calculate the path to config.yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'config.yaml')

        # Get config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Instantiate DatabaseHandler
        self.db_handler = DatabaseHandler(self.config)

        # Inject LLMHandler back into DatabaseHandler to break circular import
        self.db_handler.set_llm_handler(self)

        # Set up OpenAI client with timeout
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        # Set reasonable timeout for OpenAI API calls (60 seconds)
        self.openai_client = OpenAI(timeout=60.0)

        # Set up Mistral client with timeout
        mistral_api_key = os.environ["MISTRAL_API_KEY"]
        # Set reasonable timeout for Mistral API calls (60 seconds = 60000 milliseconds)
        # Note: Mistral uses timeout_ms (milliseconds) instead of timeout (seconds)
        self.mistral_client = Mistral(api_key=mistral_api_key, timeout_ms=60000)
        self.mistral_rate_limit_strikes = 0
        llm_config = self.config.get('llm', {})
        self.mistral_rate_limit_max_strikes = int(llm_config.get('mistral_rate_limit_max_strikes', 2) or 2)
        self.mistral_cooldown_seconds = int(llm_config.get('mistral_cooldown_seconds', 300) or 300)
        self.mistral_cooldown_until = None
        self.gemini_token = os.getenv("GEMINI_API" + "_KEY")
        self.gemini_timeout_seconds = int(llm_config.get("gemini_timeout_seconds", 60) or 60)
        self.provider_rotation_enabled = bool(llm_config.get("provider_rotation_enabled", False))
        self.provider_rotation_index = 0
        self._resolved_provider_models: dict[str, str] = {}

        # Get the keywords      
        self.keywords_list = self.get_keywords()

        # The exact fields every event must have
        self.EVENT_KEYS = {
            "source", "dance_style", "url", "event_type", "event_name",
            "day_of_week", "start_date", "end_date", "start_time",
            "end_time", "price", "location", "description"
        }

        self.ADDRESS_KEYS = {
            "address_id", "full_address", "building_name", "street_number",
            "street_name", "street_type", "direction", "city", "met_area",
            "province_or_state", "postal_code", "country_id"
        }

        # Keys for address deduplication responses
        self.ADDRESS_DEDUP_KEYS = {
            "address_id", "Label"
        }
        self.FIELD_RE = re.compile(r'^\s*"(?P<key>[^"]+)"\s*:\s*(?P<raw>.*)$')


    def driver(self, url, search_term, extracted_text, source, keywords_list):
        """
        Determine the relevance of a given URL based on its content, keywords, or organization name.

        Parameters:
        url (str): The URL to be evaluated.
        keywords (list of str): A list of keywords to check within the URL content.

        Returns:
        bool: True if the URL is relevant, False otherwise.
        """
        # Set default value of prompt type
        prompt_type = 'default'

        # Check keywords in the extracted text
        if 'facebook' in url:
            logging.info(f"def driver(): URL {url} 'facebook' is in the URL.")
            fb_status = True
            if fb_status:
                prompt_type = 'fb'

        # Check keywords in the extracted text
        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]

        # Initialize url_row with default values
        relevant, increment_crawl_try, time_stamp = False, 1, datetime.now()
        url_row = [url, search_term, source, found_keywords, relevant, increment_crawl_try, time_stamp]
    
        if found_keywords:
            logging.info(f"def driver(): Found keywords in text for URL {url}: {found_keywords}")
        
            if fb_status == True:
                # Call the llm to process the extracted text
                parent_url = search_term
                llm_status = self.process_llm_response(url, parent_url, extracted_text, source, found_keywords, prompt_type)

                if llm_status:
                    # Mark the event link as relevant
                    relevant = True
                    self.db_handler.write_url_to_db(url_row)
                    return True
                else:
                    # Mark the event link as irrelevant
                    relevant = False
                    self.db_handler.write_url_to_db(url_row)
                    return False
            else:
                # Mark the event link as irrelevant
                relevant = False
                self.db_handler.write_url_to_db(url_row)
                return False
        else:
            logging.info(f"def driver(): No keywords found in text for URL {url}\n search_term {search_term}.")
            # Mark the event link as irrelevant
            relevant = False
            self.db_handler.write_url_to_db(url_row)
            return False
    

    def get_keywords(self) -> list:
        """
        Reads the 'keywords.csv' file and returns a list of keywords.

        Returns:
            list: A list of keywords.
        """
        keywords_df = pd.read_csv(self.config['input']['data_keywords'])

        # Convert to a list, strip spaces, split on commas, and remove duplicates
        keywords_list = sorted(set(
            keyword.strip()
            for keywords in keywords_df["keywords"]
            for keyword in str(keywords).split(',')
        ))
        
        return keywords_list
    

    def process_llm_response(self, url, parent_url, extracted_text, source, keywords_list, prompt_type):
        """
        Generate a prompt, query a Language Learning Model (LLM), and process the response for EVENT EXTRACTION.

        This method is designed specifically for extracting structured event data from text. It generates a prompt,
        queries the LLM, parses the JSON response, and writes events to the database.
        
        IMPORTANT: This method should NOT be used for simple relevance checking (True/False responses).
        For relevance checking, use query_llm() directly.

        Args:
            url (str): The URL of the webpage being processed.
            parent_url (str): The parent URL context for the webpage.
            extracted_text (str): The text extracted from the webpage.
            source (str): The source organization name.
            keywords_list (list): A list of keywords relevant to the events.
            prompt_type (str): Specifies which prompt to use for EVENT EXTRACTION. Accepts:
                - Simple keys: 'fb', 'default', 'images', etc.
                - Full URLs: 'https://gotothecoda.com/calendar' for site-specific prompts
                - Falls back to 'default' if prompt_type not found in config
                - Must have a non-null schema_type for JSON parsing
                
        Returns:
            bool: True if the LLM response is successfully processed and events are written to the database, False otherwise.
        """
        # Generate prompt, query LLM, and process the response.
        prompt_text, schema_type = self.generate_prompt(url, extracted_text, prompt_type)
        max_prompt_length = int(self.config['crawling']['prompt_max_length'])
        if len(prompt_text) > max_prompt_length:
            # Fail-soft behavior: trim page text to fit prompt budget instead of dropping extraction.
            try:
                base_prompt, _ = self.generate_prompt(url, "", prompt_type)
            except Exception:
                base_prompt = ""
            overhead_chars = len(base_prompt)
            text_budget = max(5000, max_prompt_length - overhead_chars - 1000)
            trimmed_text = self._trim_extracted_text_for_budget(extracted_text, text_budget)
            prompt_text, schema_type = self.generate_prompt(url, trimmed_text, prompt_type)
            if len(prompt_text) > max_prompt_length:
                prompt_text = prompt_text[:max_prompt_length]
            logging.warning(
                "def process_llm_response: Prompt for URL %s exceeded max length; "
                "trimmed extracted text from %d to %d chars.",
                url,
                len(extracted_text or ""),
                len(trimmed_text or ""),
            )
        llm_response = self.query_llm(url, prompt_text, schema_type)

        if llm_response:
            # Check if this is a schema type that expects JSON parsing
            if schema_type is None:
                # For prompts with no schema (like relevance checks), don't try to parse JSON
                logging.warning(f"def process_llm_response: Called with schema_type=None for URL {url}. This method is for event extraction, not relevance checking.")
                return False
            
            parsed_result = self.extract_and_parse_json(llm_response, url, schema_type)

            if parsed_result:
                # If the OCR layer provided a Detected_Date/Detected_Day hint, override parsed dates to match it.
                try:
                    import re as _re
                    m = _re.search(r"(?im)^Detected_Date:\s*(\d{4}-\d{2}-\d{2})", extracted_text or "")
                    detected_date = m.group(1) if m else None
                except Exception:
                    detected_date = None
                events_df = pd.DataFrame(parsed_result)
                if detected_date:
                    try:
                        events_df['start_date'] = detected_date
                        events_df['end_date'] = events_df.get('end_date', '')
                        events_df['end_date'] = events_df['end_date'].where(events_df['end_date'].astype(str).str.len() > 0, detected_date)
                        import pandas as _pd
                        _sd = _pd.to_datetime(detected_date, errors='coerce')
                        wd = _sd.day_name() if _sd is not _pd.NaT else None
                        if wd:
                            events_df['day_of_week'] = wd
                        logging.info(f"process_llm_response: Overrode dates from Detected_Date hint: {detected_date}")
                    except Exception as _e:
                        logging.warning(f"process_llm_response: Failed to apply Detected_Date override: {_e}")
                self.db_handler.write_events_to_db(events_df, url, parent_url, source, keywords_list)
                logging.info(f"def process_llm_response: URL {url} marked as relevant with events written to the database.")
                return True
        
        else:
            logging.error(f"def process_llm_response: Failed to process LLM response for URL: {url}")
            return False

    @staticmethod
    def _trim_extracted_text_for_budget(extracted_text: str, max_chars: int) -> str:
        """
        Trim extracted page text to a max character budget while preserving
        both leading and trailing context.
        """
        if not extracted_text:
            return ""
        if max_chars <= 0:
            return ""
        if len(extracted_text) <= max_chars:
            return extracted_text
        if max_chars < 100:
            return extracted_text[:max_chars]
        head_chars = int(max_chars * 0.7)
        tail_chars = max_chars - head_chars - len("\n...[TRUNCATED]...\n")
        if tail_chars < 0:
            tail_chars = 0
        return (
            extracted_text[:head_chars]
            + "\n...[TRUNCATED]...\n"
            + (extracted_text[-tail_chars:] if tail_chars > 0 else "")
        )
        

    def generate_prompt(self, url, extracted_text, prompt_type):
        """
        Generate a prompt for a language model using extracted text and configuration details.
        
        This method implements a flexible prompt mapping system that supports both simple keys
        and URL-based prompt selection for site-specific customization.

        Args:
            url (str): The URL of the webpage from which the text was extracted.
            extracted_text (str): The text extracted from the webpage.
            prompt_type (str): Specifies which prompt to use from config['prompts']. Accepts:
                - Simple keys: 'fb', 'default', 'images', 'dedup', 'irrelevant_rows', etc.
                - Full URLs: 'https://gotothecoda.com/calendar' for site-specific prompts
                - URL-based prompts allow different websites to use specialized prompts
                - Falls back to 'default' prompt if prompt_type not found in config
                
                Examples:
                - 'fb' → uses prompts/fb_prompt.txt
                - 'https://gotothecoda.com/calendar' → uses prompts/the_coda_prompt.txt
                - 'https://www.bardandbanker.com/live-music' → uses prompts/bard_and_banker_prompt.txt

        Returns:
            tuple: (formatted_prompt_string, schema_type) where:
                - formatted_prompt_string (str): Complete prompt text with date and extracted text
                - schema_type (str): JSON schema type for structured output ('event_extraction', etc.)
        """
        # Generate the LLM prompt using the extracted text and configuration details.
        logging.info(f"def generate_prompt(): Generating prompt for URL: {url}")

        # Get prompt configuration, fallback to default if needed
        try:
            prompt_config = self.config['prompts'][prompt_type]
        except KeyError:
            # Try domain-based lookup if full URL lookup fails
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(prompt_type)
                domain = parsed_url.netloc
                if domain:
                    prompt_config = self.config['prompts'][domain]
                    logging.info(f"def generate_prompt(): Using domain-based config for '{domain}'")
                else:
                    raise KeyError("No domain found")
            except KeyError:
                prompt_config = self.config['prompts']['default']
                logging.warning(f"def generate_prompt(): Prompt type '{prompt_type}' not found, using default")
        
        # Handle both old string format and new dict format for backward compatibility
        if isinstance(prompt_config, str):
            # Old format: direct file path
            txt_file_path = prompt_config
            schema_type = None
            logging.info(f"def generate_prompt(): Using legacy config format for {prompt_type}")
        else:
            # New format: dict with file and schema
            txt_file_path = prompt_config['file']
            schema_type = prompt_config.get('schema')
        
        logging.info(f"def generate_prompt(): prompt type: {prompt_type}, file: {txt_file_path}, schema: {schema_type}")

        # Get the prompt file
        with open(txt_file_path, 'r') as file:
            is_relevant_txt = file.read()

        # For calendar venue prompts, do NOT inject the current date
        # Calendar pages have absolute dates already embedded, and injecting the current date
        # causes the LLM to incorrectly infer years (e.g., July 5 becomes 2026 instead of 2025)
        if 'calendar_venues.txt' in txt_file_path:
            logging.info(f"def generate_prompt(): Skipping date injection for calendar venues prompt")
            prompt = (
                f"{is_relevant_txt}\n"
                f"{extracted_text}\n"
            )
        else:
            # For all other prompts, generate with Pacific timezone date
            pacific_tz = ZoneInfo("America/Los_Angeles")
            today_date = datetime.now(pacific_tz).strftime("%Y-%m-%d")
            prompt = (
                f"Today's date is: {today_date}. Use this for all date calculations.\n"
                f"{is_relevant_txt}\n"
                f"{extracted_text}\n"
            )

        return prompt, schema_type

    def _is_mistral_rate_limited(self, error: Exception) -> bool:
        """Return True when the Mistral failure looks like a rate limit."""
        message = str(error).lower()
        return "429" in message or "rate limit" in message or "too many requests" in message

    def _mistral_in_cooldown(self) -> bool:
        """Return True while Mistral rate-limit cooldown window is active."""
        if self.mistral_cooldown_until is None:
            return False
        now = datetime.now()
        if now >= self.mistral_cooldown_until:
            self.mistral_cooldown_until = None
            self.mistral_rate_limit_strikes = 0
            return False
        return True

    def _record_mistral_result(self, error: Exception = None) -> None:
        """Update strike/cooldown state after a Mistral attempt."""
        if error is None:
            self.mistral_rate_limit_strikes = 0
            self.mistral_cooldown_until = None
            return
        if not self._is_mistral_rate_limited(error):
            return
        self.mistral_rate_limit_strikes += 1
        if self.mistral_rate_limit_strikes >= self.mistral_rate_limit_max_strikes:
            self.mistral_cooldown_until = datetime.now() + timedelta(seconds=self.mistral_cooldown_seconds)
            logging.warning(
                "query_llm(): Entering Mistral cooldown until %s after %s consecutive rate limits",
                self.mistral_cooldown_until.isoformat(),
                self.mistral_rate_limit_strikes,
            )

    def _fallback_enabled(self) -> bool:
        """
        Return whether cross-provider fallback is enabled.
        """
        return bool(self.config.get("llm", {}).get("fallback_enabled", True))

    def _fallback_provider_order(self) -> list[str]:
        """
        Return ordered providers to try for fallback attempts.
        """
        cfg = self.config.get("llm", {})
        configured = cfg.get("fallback_provider_order")
        if isinstance(configured, list):
            order = [str(item).strip().lower() for item in configured if str(item).strip()]
        else:
            order = ["openai", "gemini", "mistral"]
        valid = {"openai", "mistral", "gemini"}
        return [provider for provider in order if provider in valid]

    def _provider_rotation_order(self, for_tools: bool = False) -> list[str]:
        """
        Return ordered providers for round-robin primary selection.
        """
        llm_cfg = self.config.get("llm", {})
        configured = llm_cfg.get("provider_rotation_order")
        if isinstance(configured, list):
            order = [str(item).strip().lower() for item in configured if str(item).strip()]
        else:
            order = ["openai", "mistral", "gemini"]

        valid = {"openai", "mistral", "gemini"}
        filtered = [provider for provider in order if provider in valid]
        return filtered or ["openai", "mistral", "gemini"]

    def _next_round_robin_provider(self, for_tools: bool = False) -> str:
        """
        Select the next provider in round-robin order.
        """
        order = self._provider_rotation_order(for_tools=for_tools)
        provider = order[self.provider_rotation_index % len(order)]
        self.provider_rotation_index = (self.provider_rotation_index + 1) % len(order)
        return provider

    def _select_primary_provider(self, url: str, for_tools: bool = False) -> str:
        """
        Select the primary provider for a request.
        """
        lower_url = (url or "").lower()
        if "bard" in lower_url:
            return "openai"

        configured = str(self.config.get("llm", {}).get("provider", "openai")).strip().lower()
        use_rotation = self.provider_rotation_enabled or configured == "round_robin"
        if use_rotation:
            return self._next_round_robin_provider(for_tools=for_tools)
        return configured

    def _provider_model_candidates(self, provider: str) -> list[str]:
        """
        Return candidate model names for a provider in priority order.
        Supports alias-based mapping and direct provider lists.
        """
        llm_cfg = self.config.get("llm", {})
        provider = (provider or "").lower().strip()

        alias = str(llm_cfg.get("model_alias", "default_fast")).strip()
        alias_map = llm_cfg.get("model_candidates", {})
        if isinstance(alias_map, dict):
            by_provider = alias_map.get(provider, {})
            if isinstance(by_provider, dict):
                alias_candidates = by_provider.get(alias, [])
                if isinstance(alias_candidates, list) and alias_candidates:
                    return [str(m).strip() for m in alias_candidates if str(m).strip()]

        provider_list_key = f"{provider}_model_candidates"
        provider_candidates = llm_cfg.get(provider_list_key, [])
        if isinstance(provider_candidates, list) and provider_candidates:
            return [str(m).strip() for m in provider_candidates if str(m).strip()]

        single_model_map = {
            "openai": llm_cfg.get("openai_model"),
            "mistral": llm_cfg.get("mistral_model"),
            "gemini": llm_cfg.get("gemini_model"),
        }
        single = str(single_model_map.get(provider, "") or "").strip()
        return [single] if single else []

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        """
        Normalize provider model ids for comparison.
        """
        value = str(model_id or "").strip()
        if value.startswith("models/"):
            return value[len("models/"):]
        return value

    def _list_provider_models(self, provider: str) -> set[str] | None:
        """
        Best-effort listing of available models for a provider.
        Returns None if listing is unavailable/fails.
        """
        provider = (provider or "").lower().strip()
        try:
            if provider == "openai":
                resp = self.openai_client.models.list()
                return {self._normalize_model_id(getattr(m, "id", "")) for m in (resp.data or []) if getattr(m, "id", None)}

            if provider == "gemini":
                if not self.gemini_token:
                    return None
                endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
                response = requests.get(endpoint, headers={"x-goog-api-key": self.gemini_token}, timeout=self.gemini_timeout_seconds)
                response.raise_for_status()
                data = response.json()
                models = data.get("models", []) or []
                return {self._normalize_model_id(m.get("name", "")) for m in models if m.get("name")}

            if provider == "mistral":
                models_api = getattr(self.mistral_client, "models", None)
                list_fn = getattr(models_api, "list", None) if models_api else None
                if not callable(list_fn):
                    return None
                resp = list_fn()
                data = getattr(resp, "data", None) or []
                output: set[str] = set()
                for model in data:
                    model_id = getattr(model, "id", None) or getattr(model, "name", None)
                    if model_id:
                        output.add(self._normalize_model_id(model_id))
                return output or None
        except Exception as e:
            logging.warning("_list_provider_models(): could not list %s models: %s", provider, e)
            return None
        return None

    def _resolve_model_for_provider(self, provider: str, force_refresh: bool = False) -> str:
        """
        Resolve and cache a working model name from configured candidates.
        """
        provider = (provider or "").lower().strip()
        if not force_refresh and provider in self._resolved_provider_models:
            return self._resolved_provider_models[provider]

        candidates = self._provider_model_candidates(provider)
        if not candidates:
            raise ValueError(f"No model candidates configured for provider '{provider}'.")

        available = self._list_provider_models(provider)
        chosen = None
        if available:
            normalized_available = {self._normalize_model_id(model) for model in available}
            for candidate in candidates:
                if self._normalize_model_id(candidate) in normalized_available:
                    chosen = candidate
                    break
        if chosen is None:
            chosen = candidates[0]

        self._resolved_provider_models[provider] = chosen
        logging.info("_resolve_model_for_provider(): provider=%s resolved_model=%s", provider, chosen)
        return chosen

    @staticmethod
    def _is_model_not_found_error(error: Exception) -> bool:
        """
        Detect provider errors that indicate invalid/deprecated model name.
        """
        msg = str(error).lower()
        patterns = (
            "model not found",
            "unknown model",
            "does not exist",
            "no longer available",
            "404",
        )
        return any(token in msg for token in patterns)

    def resolve_models_for_startup(self) -> dict[str, str]:
        """
        Resolve model names for known providers for startup diagnostics.
        """
        resolved = {}
        for provider in ("openai", "mistral", "gemini"):
            try:
                resolved[provider] = self._resolve_model_for_provider(provider, force_refresh=True)
            except Exception as e:
                resolved[provider] = f"unresolved: {e}"
        return resolved

    def _query_provider_once(self, provider: str, prompt: str, schema_type: str | None):
        """
        Execute one provider query attempt and return text or None.
        """
        provider = (provider or "").lower().strip()
        response = None

        if provider == "openai":
            model = self._resolve_model_for_provider("openai")
            logging.info("query_llm(): Querying OpenAI")
            try:
                response = self.query_openai(prompt, model, schema_type=schema_type)
            except Exception as e:
                if self._is_model_not_found_error(e):
                    model = self._resolve_model_for_provider("openai", force_refresh=True)
                    response = self.query_openai(prompt, model, schema_type=schema_type)
                else:
                    raise
            if response:
                logging.info("query_llm(): OpenAI response received.")
            else:
                logging.warning("query_llm(): OpenAI returned no response.")
            return response

        if provider == "mistral":
            if self._mistral_in_cooldown():
                logging.warning("query_llm(): Skipping Mistral query due to active cooldown.")
                return None
            try:
                model = self._resolve_model_for_provider("mistral")
                logging.info("query_llm(): Querying Mistral")
                try:
                    response = self.query_mistral(prompt, model, schema_type=schema_type)
                except Exception as e:
                    if self._is_model_not_found_error(e):
                        model = self._resolve_model_for_provider("mistral", force_refresh=True)
                        response = self.query_mistral(prompt, model, schema_type=schema_type)
                    else:
                        raise
                self._record_mistral_result()
                if response:
                    logging.info("query_llm(): Mistral response received.")
                else:
                    logging.warning("query_llm(): Mistral returned no response.")
                return response
            except Exception as e:
                self._record_mistral_result(e)
                raise

        if provider == "gemini":
            model = self._resolve_model_for_provider("gemini")
            logging.info("query_llm(): Querying Gemini")
            try:
                response = self.query_gemini(prompt, model, schema_type=schema_type)
            except Exception as e:
                if self._is_model_not_found_error(e):
                    model = self._resolve_model_for_provider("gemini", force_refresh=True)
                    response = self.query_gemini(prompt, model, schema_type=schema_type)
                else:
                    raise
            if response:
                logging.info("query_llm(): Gemini response received.")
            else:
                logging.warning("query_llm(): Gemini returned no response.")
            return response

        raise ValueError(f"Invalid LLM provider: {provider}")


    def query_llm(self, url, prompt, schema_type=None, tools=None, max_iterations=3):
        """
        Query the configured LLM with a given prompt and return the response.
        Fallback occurs between Mistral and OpenAI if one fails.

        Supports optional function/tool calling for LLM to invoke Python functions
        (e.g., date calculations) before generating final response.

        Args:
            url (str): The URL being processed (for logging).
            prompt (str): The prompt to send to the LLM.
            schema_type (str): The schema type for structured output (optional).
            tools (list): Optional list of tool definitions for function calling.
            max_iterations (int): Maximum iterations for tool calls (default: 3).

        Returns:
            str: The response from the LLM if available, otherwise None.
        """
        if not self.config['llm']['spend_money']:
            logging.info("query_llm(): Spending money is disabled. Skipping the LLM query.")
            return None

        # If tools provided, use tool calling logic
        if tools:
            result = self._query_with_tools(url, prompt, tools, max_iterations)
            return result.get('content') if result else None
        
        provider = self._select_primary_provider(url, for_tools=False)

        fallback_enabled = self._fallback_enabled()
        candidates = [str(provider).strip().lower()]
        if fallback_enabled:
            for fallback_provider in self._fallback_provider_order():
                if fallback_provider not in candidates:
                    candidates.append(fallback_provider)
        response = None
        for candidate in candidates:
            try:
                response = self._query_provider_once(candidate, prompt, schema_type)
                if response:
                    return response
            except Exception as e:
                error_message = str(e).replace('error', 'rejection')
                logging.warning("query_llm(): %s query failed: %s", candidate.capitalize(), error_message)

        logging.error("query_llm(): All configured LLM providers failed to provide a response.")
        return None

    def _query_with_tools(self, url, prompt, tools, max_iterations=3):
        """
        Internal method: Query the LLM with function/tool calling support.

        This method allows the LLM to call Python functions (tools) to perform
        calculations or lookups before generating the final response. Supports
        iterative tool calling where the LLM can make multiple function calls.

        Args:
            url (str): The URL being processed (for logging)
            prompt (str): The prompt to send to the LLM
            tools (list): List of tool definitions (function schemas)
            max_iterations (int): Maximum number of tool call iterations to prevent loops

        Returns:
            dict: Response with keys:
                - content (str): The final LLM response
                - tool_calls (list): List of tool calls made (if any)
                - iterations (int): Number of iterations taken
        """

        # Import date calculator for tool execution
        from date_calculator import calculate_date_range

        # Map function names to actual Python functions
        available_functions = {
            "calculate_date_range": calculate_date_range
        }

        # Determine provider (same logic as query_llm)
        provider = self._select_primary_provider(url, for_tools=True)
        fallback_enabled = self._fallback_enabled()

        # Initialize conversation messages
        messages = [{"role": "user", "content": prompt}]
        all_tool_calls = []

        for iteration in range(max_iterations):
            logging.info(f"query_llm(): Tool calling iteration {iteration + 1}/{max_iterations}")

            # Query LLM with tools
            if provider == 'mistral':
                response = self._query_mistral_with_tools(messages, tools)
                # Fallback only if truly no response and no tool calls
                if not response or (not response.get("content") and not response.get("tool_calls")):
                    if fallback_enabled:
                        logging.warning("query_llm(): Mistral tools returned no content; falling back to OpenAI")
                        response = self._query_openai_with_tools(messages, tools)
                    else:
                        logging.info("query_llm(): Fallback disabled for tool calls; not falling back from Mistral.")
            elif provider == 'gemini':
                response = self._query_gemini_with_tools(messages, tools)
                # Fallback only if truly no response and no tool calls
                if not response or (not response.get("content") and not response.get("tool_calls")):
                    if fallback_enabled:
                        logging.warning("query_llm(): Gemini tools returned no content; falling back to OpenAI")
                        response = self._query_openai_with_tools(messages, tools)
                        if not response or (not response.get("content") and not response.get("tool_calls")):
                            logging.warning("query_llm(): OpenAI tools fallback returned no content; falling back to Mistral")
                            response = self._query_mistral_with_tools(messages, tools)
                    else:
                        logging.info("query_llm(): Fallback disabled for tool calls; not falling back from Gemini.")
            elif provider == 'openai':
                response = self._query_openai_with_tools(messages, tools)
                # Fallback only if truly no response and no tool calls
                if not response or (not response.get("content") and not response.get("tool_calls")):
                    if fallback_enabled:
                        logging.warning("query_llm(): OpenAI tools returned no content; falling back to Mistral")
                        response = self._query_mistral_with_tools(messages, tools)
                    else:
                        logging.info("query_llm(): Fallback disabled for tool calls; not falling back from OpenAI.")
            else:
                logging.error("query_llm(): Invalid LLM provider")
                return {"content": None, "tool_calls": [], "iterations": iteration}

            if not response:
                logging.error(f"query_llm(): No response from providers during tool call")
                return {"content": None, "tool_calls": all_tool_calls, "iterations": iteration}

            # Check if LLM wants to call a tool
            if response.get("tool_calls"):
                logging.info(f"query_llm(): LLM requested {len(response['tool_calls'])} tool call(s)")

                # Add assistant message with tool calls to conversation
                # Normalize tool_calls for providers
                # - OpenAI requires type='function'
                # - Some providers (e.g., Mistral) are strict about tool call id format
                #   → sanitize ids to a-zA-Z0-9 with length 9
                import re as _re_tc
                def _sanitize_id(i: str) -> str:
                    i = (i or "")
                    i = _re_tc.sub(r"[^A-Za-z0-9]", "", i)
                    if len(i) >= 9:
                        return i[:9]
                    # pad deterministically with zeros if too short
                    return (i + ("0" * 9))[:9]

                normalized_tool_calls = []
                id_map = {}
                for tc in response["tool_calls"]:
                    orig_id = tc.get("id", "")
                    sid = _sanitize_id(orig_id)
                    id_map[orig_id] = sid
                    norm = {
                        "id": sid,
                        "type": "function",
                        "function": tc.get("function", {})
                    }
                    normalized_tool_calls.append(norm)

                messages.append({
                    "role": "assistant",
                    "content": response.get("content"),
                    "tool_calls": normalized_tool_calls
                })

                # Execute each tool call
                for tool_call in response["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])

                    logging.info(f"query_llm(): Calling {function_name} with args: {function_args}")

                    # Execute the function
                    if function_name in available_functions:
                        try:
                            function_result = available_functions[function_name](**function_args)
                            logging.info(f"query_llm(): {function_name} returned: {function_result}")

                            # Add tool result to conversation
                            messages.append({
                                "role": "tool",
                                "tool_call_id": id_map.get(tool_call.get("id", ""), _sanitize_id(tool_call.get("id", ""))),
                                "name": function_name,
                                "content": json.dumps(function_result)
                            })

                            # Track tool call
                            all_tool_calls.append({
                                "function": function_name,
                                "arguments": function_args,
                                "result": function_result
                            })

                        except Exception as e:
                            error_msg = f"Error executing {function_name}: {str(e)}"
                            logging.error(f"query_llm(): {error_msg}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": id_map.get(tool_call.get("id", ""), _sanitize_id(tool_call.get("id", ""))),
                                "name": function_name,
                                "content": json.dumps({"error": error_msg})
                            })
                    else:
                        logging.error(f"query_llm(): Unknown function: {function_name}")

                # Continue to next iteration to get final response from LLM
                continue
            else:
                # No tool calls, we have the final response
                logging.info(f"query_llm(): Final response received after {iteration + 1} iteration(s)")
                return {
                    "content": response.get("content"),
                    "tool_calls": all_tool_calls,
                    "iterations": iteration + 1
                }

        # Max iterations reached
        logging.warning(f"query_llm(): Max iterations ({max_iterations}) reached")
        return {
            "content": messages[-1].get("content") if messages else None,
            "tool_calls": all_tool_calls,
            "iterations": max_iterations
        }

    def _query_mistral_with_tools(self, messages, tools):
        """Query Mistral with tool calling support."""
        try:
            model = self._resolve_model_for_provider("mistral")
            logging.info(f"_query_mistral_with_tools(): Querying Mistral model {model}")

            try:
                response = self.mistral_client.chat.complete(
                    model=model,
                    messages=messages,
                    tools=tools
                )
            except Exception as e:
                if self._is_model_not_found_error(e):
                    model = self._resolve_model_for_provider("mistral", force_refresh=True)
                    response = self.mistral_client.chat.complete(
                        model=model,
                        messages=messages,
                        tools=tools
                    )
                else:
                    raise

            if not response or not response.choices:
                return None

            choice = response.choices[0]
            message = choice.message

            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                }
            else:
                # No tool calls, return content
                return {"content": message.content}

        except Exception as e:
            logging.error(f"_query_mistral_with_tools(): Error: {e}")
            return None

    def _query_openai_with_tools(self, messages, tools):
        """Query OpenAI with tool calling support."""
        try:
            model = self._resolve_model_for_provider("openai")
            logging.info(f"_query_openai_with_tools(): Querying OpenAI model {model}")

            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools
                )
            except Exception as e:
                if self._is_model_not_found_error(e):
                    model = self._resolve_model_for_provider("openai", force_refresh=True)
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools
                    )
                else:
                    raise

            if not response or not response.choices:
                return None

            choice = response.choices[0]
            message = choice.message

            # Check for tool calls
            if message.tool_calls:
                return {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                }
            else:
                # No tool calls, return content
                return {"content": message.content}

        except Exception as e:
            logging.error(f"_query_openai_with_tools(): Error: {e}")
            return None

    def _convert_tools_to_gemini(self, tools):
        """
        Convert OpenAI-style tools payload to Gemini functionDeclarations format.
        """
        function_declarations = []
        for tool in tools or []:
            function_obj = None
            if isinstance(tool, dict):
                function_obj = tool.get("function") if tool.get("type") == "function" else tool.get("function", tool)
            if not isinstance(function_obj, dict):
                continue
            name = function_obj.get("name")
            if not name:
                continue
            declaration = {
                "name": name,
                "description": function_obj.get("description", ""),
                "parameters": function_obj.get("parameters", {"type": "object", "properties": {}}),
            }
            function_declarations.append(declaration)
        if not function_declarations:
            return []
        return [{"functionDeclarations": function_declarations}]

    def _convert_messages_to_gemini_contents(self, messages):
        """
        Convert internal message format to Gemini contents array.
        """
        contents = []
        for msg in messages or []:
            role = (msg.get("role") or "").lower()
            if role == "user":
                text = msg.get("content") or ""
                contents.append({"role": "user", "parts": [{"text": str(text)}]})
                continue

            if role == "assistant":
                parts = []
                content_text = msg.get("content")
                if content_text:
                    parts.append({"text": str(content_text)})
                for tool_call in msg.get("tool_calls", []) or []:
                    fn = (tool_call or {}).get("function", {})
                    fn_name = fn.get("name")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else (fn_args_raw or {})
                    except Exception:
                        fn_args = {}
                    if fn_name:
                        parts.append({"functionCall": {"name": fn_name, "args": fn_args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                fn_name = msg.get("name")
                raw_content = msg.get("content", "")
                try:
                    parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                except Exception:
                    parsed = {"result": str(raw_content)}
                if not isinstance(parsed, dict):
                    parsed = {"result": parsed}
                if fn_name:
                    contents.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": fn_name, "response": parsed}}],
                    })
                continue

        return contents

    def _query_gemini_with_tools(self, messages, tools):
        """
        Query Gemini with tool calling support using REST generateContent.
        """
        try:
            if not self.gemini_token:
                logging.error("_query_gemini_with_tools(): GEMINI_API_KEY is not configured.")
                return None

            model = self._resolve_model_for_provider("gemini")
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            payload = {
                "contents": self._convert_messages_to_gemini_contents(messages),
                "tools": self._convert_tools_to_gemini(tools),
            }
            headers = {"x-goog-api-key": self.gemini_token}
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.gemini_timeout_seconds,
            )
            try:
                response.raise_for_status()
            except Exception as e:
                if self._is_model_not_found_error(e):
                    model = self._resolve_model_for_provider("gemini", force_refresh=True)
                    endpoint = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent"
                    )
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.gemini_timeout_seconds,
                    )
                    response.raise_for_status()
                else:
                    raise
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None

            parts = (candidates[0].get("content") or {}).get("parts", [])
            text_parts = []
            tool_calls = []
            for idx, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    text_parts.append(str(part.get("text")))
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    tool_calls.append({
                        "id": f"gemini{idx:03d}",
                        "function": {
                            "name": function_call.get("name"),
                            "arguments": json.dumps(function_call.get("args", {})),
                        },
                    })

            if tool_calls:
                return {"content": "\n".join(text_parts).strip() if text_parts else "", "tool_calls": tool_calls}
            return {"content": "\n".join(text_parts).strip() if text_parts else None}
        except Exception as e:
            logging.error(f"_query_gemini_with_tools(): Error: {e}")
            return None

    def query_openai(self, prompt, model, image_url=None, schema_type=None):
        """
        Handles querying OpenAI LLM, optionally attaching an image.
        - prompt: str or list/tuple (will be normalized to str)
        - model: e.g. "o4-mini-high" or "gpt-4.1-mini"
        - image_url: optional URL string of an image to include
        - schema_type: explicit schema type (e.g. 'event_extraction', 'address_extraction', None)
        """
        # --- 1) Normalize prompt to a string ---
        if isinstance(prompt, (list, tuple)):
            prompt = "\n".join(map(str, prompt))
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        # --- 2) Build content blocks (array-of-parts is fine) ---
        content_blocks = [{"type": "text", "text": prompt}]
        if image_url:
            # OpenAI expects "input_image"
            content_blocks.append({
                "type": "input_image",
                "image_url": {"url": image_url}
            })

        # --- 3) Optional JSON schema handling ---
        json_schema = self._get_json_schema_by_type(schema_type, "openai") if schema_type else None
        response_format = None
        if json_schema:
            # Ensure it's the full JSON Schema object OpenAI expects:
            # {"type":"json_schema","json_schema":{"name":..., "schema":{...}, "strict": True}}
            if "name" in json_schema and "schema" in json_schema:
                payload_schema = json_schema
            else:
                # If your helper returns only the raw schema, wrap it
                payload_schema = {
                    "name": schema_type or "StructuredOutput",
                    "schema": json_schema,
                    "strict": True
                }
            response_format = {"type": "json_schema", "json_schema": payload_schema}

        # --- 4) Prepare and send request ---
        api_params = {
            "model": model,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        if response_format:
            api_params["response_format"] = response_format

        resp = self.openai_client.chat.completions.create(**api_params)

        # --- 5) Extract content safely ---
        if resp and getattr(resp, "choices", None):
            msg = resp.choices[0].message
            # msg.content may be a string or list of parts; normalize to string
            if isinstance(msg.content, list):
                parts = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return "\n".join(parts).strip()
            return (msg.content or "").strip()
        return None


    def query_mistral(self, prompt, model, schema_type=None):
        # 1) Normalize prompt
        if isinstance(prompt, (list, tuple)):
            prompt = "\n".join(map(str, prompt))
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        # 2) Schema wrapper
        json_schema = self._get_json_schema_by_type(schema_type, "mistral") if schema_type else None
        response_format = None
        if json_schema:
            # If helper returns the raw JSON Schema, wrap it; if it already has name/schema, keep as is
            # Note: Removed "strict": True for Mistral compatibility - Mistral doesn't support OpenAI's strict mode
            if not all(k in json_schema for k in ("name", "schema")):
                json_schema = {
                    "name": schema_type or "StructuredOutput",
                    "schema": json_schema,
                }
            response_format = {"type": "json_schema", "json_schema": json_schema}

        # 3) Call Mistral
        api_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format:
            api_params["response_format"] = response_format

        resp = self.mistral_client.chat.complete(**api_params)
        return resp.choices[0].message.content if resp and resp.choices else None

    def query_gemini(self, prompt, model, schema_type=None):
        """
        Query Google Gemini via REST API.
        """
        if isinstance(prompt, (list, tuple)):
            prompt = "\n".join(map(str, prompt))
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        if not self.gemini_token:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

        generation_config = {}
        if schema_type:
            generation_config["responseMimeType"] = "application/json"
            schema = self._get_json_schema_by_type(schema_type, "gemini")
            if schema:
                raw_schema = schema.get("schema", schema) if isinstance(schema, dict) else schema
                generation_config["responseSchema"] = raw_schema

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if generation_config:
            payload["generationConfig"] = generation_config

        headers = {"x-goog-api-key": self.gemini_token}
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.gemini_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = (candidates[0].get("content") or {}).get("parts", [])
        texts = []
        for part in parts:
            text_value = part.get("text") if isinstance(part, dict) else None
            if text_value:
                texts.append(str(text_value))
        return "\n".join(texts).strip() if texts else None


    def _get_json_schema_by_type(self, schema_type, provider="mistral"):
        """
        Returns the appropriate JSON schema based on explicit schema type and provider.
        Different providers have different schema requirements.
        """
        if not schema_type:
            return None
        
        # Define event properties once to avoid duplication
        event_properties = {
            "source": {"type": "string"},
            "dance_style": {"type": "string"},
            "url": {"type": "string"},
            "event_type": {"type": "string"},
            "event_name": {"type": "string"},
            "day_of_week": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
            "price": {"type": "string"},
            "location": {"type": "string"},
            "description": {"type": "string"}
        }
        event_required = ["source", "dance_style", "url", "event_type", "event_name", 
                         "day_of_week", "start_date", "end_date", "start_time", "end_time", 
                         "price", "location", "description"]
            
        # Provider-specific schemas
        if provider.lower() == "openai":
            schemas = {
                "event_extraction": {
                    "name": "event_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": event_properties,
                                    "required": event_required,
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["events"],
                        "additionalProperties": False
                    }
                },
                
                "address_extraction": {
                    "name": "address_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "address_id": {"type": "integer"},
                            "full_address": {"type": "string"},
                            "building_name": {"type": ["string", "null"]},
                            "street_number": {"type": "string"},
                            "street_name": {"type": "string"},
                            "street_type": {"type": "string"},
                            "direction": {"type": ["string", "null"]},
                            "city": {"type": "string"},
                            "met_area": {"type": ["string", "null"]},
                            "province_or_state": {"type": "string"},
                            "postal_code": {"type": ["string", "null"]},
                            "country_id": {"type": "string"},
                            "time_stamp": {"type": ["string", "null"]}
                        },
                        "required": ["address_id", "full_address", "building_name", "street_number", "street_name", 
                                   "street_type", "direction", "city", "met_area", "province_or_state", 
                                   "postal_code", "country_id", "time_stamp"],
                        "additionalProperties": False
                    }
                },
                
                "deduplication_response": {
                    "name": "deduplication_response", 
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "group_id": {"type": "integer"},
                                        "event_id": {"type": "integer"},
                                        "Label": {"type": "integer"}
                                    },
                                    "required": ["group_id", "event_id", "Label"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["events"],
                        "additionalProperties": False
                    }
                },
                
                "relevance_classification": {
                    "name": "relevance_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "event_id": {"type": "integer"},
                                        "Label": {"type": "integer"},
                                        "event_type_new": {"type": "string"}
                                    },
                                    "required": ["event_id", "Label", "event_type_new"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["events"],
                        "additionalProperties": False
                    }
                },
                
                "address_deduplication": {
                    "name": "address_deduplication",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "addresses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "address_id": {"type": "integer"},
                                        "Label": {"type": "integer"}
                                    },
                                    "required": ["address_id", "Label"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["addresses"],
                        "additionalProperties": False
                    }
                }
            }
            # Gemini uses the same structural schema as OpenAI response schema,
            # but does not use OpenAI's wrapper fields like name/strict.
            if provider.lower() == "gemini":
                gemini_schemas = {}
                for schema_name, schema_payload in schemas.items():
                    if isinstance(schema_payload, dict) and "schema" in schema_payload:
                        gemini_schemas[schema_name] = schema_payload["schema"]
                    else:
                        gemini_schemas[schema_name] = schema_payload
                return gemini_schemas.get(schema_type)
        else:  # Mistral and others
            schemas = {
                "event_extraction": {
                    "name": "event_extraction",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": event_properties,
                            "required": event_required,
                            "additionalProperties": False
                        }
                    }
                },
                
                "address_extraction": {
                    "name": "address_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "address_id": {"type": "integer"},
                            "full_address": {"type": "string"},
                            "building_name": {"type": ["string", "null"]},
                            "street_number": {"type": "string"},
                            "street_name": {"type": "string"},
                            "street_type": {"type": "string"},
                            "direction": {"type": ["string", "null"]},
                            "city": {"type": "string"},
                            "met_area": {"type": ["string", "null"]},
                            "province_or_state": {"type": "string"},
                            "postal_code": {"type": ["string", "null"]},
                            "country_id": {"type": "string"},
                            "time_stamp": {"type": ["string", "null"]}
                        },
                        "required": ["address_id", "full_address", "building_name", "street_number", "street_name", 
                                   "street_type", "direction", "city", "met_area", "province_or_state", 
                                   "postal_code", "country_id", "time_stamp"],
                        "additionalProperties": False
                    }
                },
                
                "deduplication_response": {
                    "name": "deduplication_response", 
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "group_id": {"type": "integer"},
                                "event_id": {"type": "integer"},
                                "Label": {"type": "integer"}
                            },
                            "required": ["group_id", "event_id", "Label"],
                            "additionalProperties": False
                        }
                    }
                },
                
                "relevance_classification": {
                    "name": "relevance_classification",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_id": {"type": "integer"},
                                "Label": {"type": "integer"},
                                "event_type_new": {"type": "string"}
                            },
                            "required": ["event_id", "Label", "event_type_new"],
                            "additionalProperties": False
                        }
                    }
                },
                
                "address_deduplication": {
                    "name": "address_deduplication",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "address_id": {"type": "integer"},
                                "Label": {"type": "integer"}
                            },
                            "required": ["address_id", "Label"],
                            "additionalProperties": False
                        }
                    }
                }
            }
        
        return schemas.get(schema_type)


    def line_based_parse(self, raw_str: str, schema_type: str = None) -> list[dict]:
        """
        Parses a JSON-like string into a list of dictionaries, one per record.

        - Uses explicit schema_type to determine required keys, falling back to content detection if not provided.
        - Splits the input into lines and looks for lines of the form: "key": value,
        - Strips all quotes from values and collects key-value pairs into a record.
        - Only records containing all required keys are included in the output list.

        Args:
            raw_str (str): The raw string containing JSON-like records.
            schema_type (str, optional): The explicit schema type to use for key validation.

        Returns:
            list[dict]: A list of dictionaries, each representing a parsed record.
        """
        # Use explicit schema type to determine required keys
        if schema_type == "event_extraction":
            required_keys = self.EVENT_KEYS
        elif schema_type == "address_extraction":
            required_keys = self.ADDRESS_KEYS
        elif schema_type in ["deduplication_response", "relevance_classification"]:
            required_keys = self.EVENT_KEYS  # These operate on events
        elif schema_type == "address_deduplication":
            required_keys = self.ADDRESS_DEDUP_KEYS
        else:
            # Fallback to old brittle content detection for backward compatibility
            if 'Label' in raw_str and 'address_id' in raw_str:
                required_keys = self.ADDRESS_DEDUP_KEYS
            elif 'building_name' in raw_str:
                required_keys = self.ADDRESS_KEYS
            else:
                required_keys = self.EVENT_KEYS

        records = []
        current = None

        if raw_str.startswith("[") and raw_str.endswith("]"):
            raw_str = raw_str[1:-1].strip()

        # Check if this is compact JSON (single line with multiple key-value pairs)
        # Look for pattern: {"key":"value","key2":"value2",...}
        if (raw_str.startswith("{") and raw_str.endswith("}") and 
            raw_str.count('\n') == 0 and raw_str.count('":') > 1):
            logging.info("line_based_parse(): Detected compact JSON, reformatting with line breaks")
            
            # ONLY for compact JSON: Use json.loads() to reformat safely
            try:
                import json
                parsed_json = json.loads(raw_str)
                
                # Reformat as multi-line JSON for downstream processing
                formatted = "{\n"
                for i, (key, value) in enumerate(parsed_json.items()):
                    # Format value properly based on type
                    if value is None:
                        formatted_value = "null"
                    elif isinstance(value, str):
                        formatted_value = f'"{value}"'
                    else:
                        formatted_value = str(value)
                    
                    # Add comma for all but last item
                    comma = "," if i < len(parsed_json) - 1 else ""
                    formatted += f'  "{key}": {formatted_value}{comma}\n'
                
                formatted += "}"
                
                logging.debug(f"line_based_parse(): Reformatted JSON:\n{formatted}")
                raw_str = formatted
                
            except json.JSONDecodeError as e:
                logging.error(f"line_based_parse(): Failed to parse compact JSON: {e}")
                # Fall through to line-based parsing with original string

        for line in raw_str.splitlines():
            line = line.strip()

            # start of a new record
            if line.startswith('{'):
                current = {}
                continue

            # end of current record
            if line.startswith('}'):
                if current is not None:
                    missing = required_keys - current.keys()
                    if not missing:
                        logging.debug("line_based_parse: Parsed record: %s", current)
                        records.append(current)
                    else:
                        logging.debug("line_based_parse: Missing keys: %s", missing)
                        logging.warning(
                            "Skipping incomplete record, missing %s: %r",
                            missing, current
                        )
                current = None
                continue

            # match lines like: "key": raw_value,
            m = self.FIELD_RE.match(line)
            if not m or current is None:
                continue

            key, raw = m.group('key'), m.group('raw').rstrip(',')
            if key in required_keys:
                # remove all stray quotes and trim whitespace
                value = raw.replace('"', '').strip()
                # Normalize null strings to None
                null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}
                current[key] = None if value.lower() in null_strings else value

        logging.info(f"line_based_parse(): Here is what the records look like: \n{records}.")
        return records
    

    def extract_and_parse_json(self, result: str, url: str, schema_type: str = None):
        """
        1) Early-exit on no data.
        2) Bracket-match to isolate the JSON-like blob.
        3) Basic cleanup (comments, backticks, ellipses, stray commas).
        4) Line-based parse using explicit schema_type or fallback to content detection.
        
        Args:
            result (str): The LLM response string to parse
            url (str): The URL context for logging
            schema_type (str, optional): The explicit schema type for key validation
        """
        # 1) Early exits
        if result is None:
            logging.info("extract_and_parse_json(): Result is None.")
            return None
        if "No events found" in result:
            logging.info("extract_and_parse_json(): No events found.")
            return None
        # Allow shorter responses for address deduplication (typically ~50-100 chars)
        min_length = 30 if ("address_id" in result and "Label" in result) or "canonical_address_id" in result else 100
        if len(result) <= min_length:
            logging.info(f"extract_and_parse_json(): Result too short (< {min_length} chars).")
            return None

        # 2) Isolate [ … ] or wrap { … } in […]
        start = result.find('[')
        if start == -1:
            start = 0
        end = None
        depth = 0
        in_str = False
        escape = False

        for i, ch in enumerate(result[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

        if end is None or end <= start:
            fb, lb = result.find('{'), result.rfind('}')
            if fb != -1 and lb > fb:
                blob = '[' + result[fb:lb+1] + ']'
            else:
                logging.error(
                    f"extract_and_parse_json(): Couldn't isolate JSON blob from {url}"
                )
                return None
        else:
            blob = result[start:end]

        # Allow shorter blobs for address deduplication 
        min_blob_length = 30 if ("address_id" in blob and "Label" in blob) or "canonical_address_id" in blob else 100
        if len(blob) < min_blob_length:
            logging.info(f"extract_and_parse_json(): Blob too short (< {min_blob_length} chars):\n{blob}")
            return None

        # 3) Basic cleanup
        cleaned = re.sub(r'(?<!:)//.*', '', blob)
        cleaned = cleaned.replace('...', '').strip()
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = cleaned.replace("```json", "").replace("```", "")

        # 4) Try JSON parsing first (for structured output), then fall back to line-based parsing
        try:
            json_data = json.loads(cleaned)
            
            # Handle different response formats
            if isinstance(json_data, dict):
                # Check for OpenAI wrapped formats first
                if "events" in json_data and isinstance(json_data["events"], list):
                    return json_data["events"]
                elif "addresses" in json_data and isinstance(json_data["addresses"], list):
                    return json_data["addresses"]
                # Single object - check if it looks like an address or event
                elif "address_id" in json_data or "full_address" in json_data:
                    return [json_data]  # Address object
                elif "event_id" in json_data or "event_name" in json_data:
                    return [json_data]  # Event object
                else:
                    # Unknown single object, wrap in list
                    return [json_data]
            # Handle direct array format (Mistral or legacy)
            elif isinstance(json_data, list):
                return json_data
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.info(f"extract_and_parse_json(): JSON parsing failed, falling back to line-based parsing: {e}")
        
        # 5) Fallback to line-based parse using explicit schema type
        records = self.line_based_parse(cleaned, schema_type)
        if not records:
            logging.info("extract_and_parse_json(): No complete records parsed.")
            return None

        return records
    

    def is_incomplete_address(self, row):
        """Check if any required field is missing or empty."""
        required_fields = [
            "full_address", "street_number", "street_name", 
            "city", "province_or_state", "postal_code", "country_id"
        ]
        return any(
            pd.isna(row[col]) or str(row[col]).strip() == ""
            for col in required_fields
        )


    def fix_incomplete_addresses_batch(self, limit: int = 10):
        """
        Batch process to fix incomplete addresses in the database using OpenAI LLM.
        Only rows missing required address fields will be processed.
        Processes up to `limit` rows. Logs updated rows and tracks failed address_ids.
        """
        # 1. Load all address rows
        sql = "SELECT * FROM address"
        df = pd.read_sql(sql, self.db_handler.conn)
        logging.info("Loaded %d address rows from database.", len(df))

        # 2. Filter for incomplete addresses using instance method
        incomplete_df = df[df.apply(self.is_incomplete_address, axis=1)].copy()
        logging.info("Identified %d incomplete address rows.", len(incomplete_df))

        # 3. Limit the number of rows to process
        incomplete_df = incomplete_df.head(limit)
        logging.info("Processing up to %d incomplete rows.", len(incomplete_df))

        updated_rows = []
        failed_ids = []

        for _, row in incomplete_df.iterrows():
            address_id = row["address_id"]
            location_str = row.get("full_address") or row.get("building_name") or ""

            # Skip empty or short location strings
            if not location_str or len(location_str.strip()) < 15:
                logging.info("Skipping address_id=%d due to short location string.", address_id)
                failed_ids.append(address_id)
                continue

            try:
                # 4. Fix address using LLM (OpenAI forced)
                parsed_address = self.parse_location_with_llm(location_str, prompt_type="address_internet_fix")

                if not parsed_address or parsed_address.get("address_id", 0) == 0:
                    logging.warning("LLM returned invalid result for address_id=%d", address_id)
                    failed_ids.append(address_id)
                    continue

                # 5. Assign correct address_id to target row
                parsed_address["address_id"] = address_id

                # 6. Update using execute_query
                update_sql = """
                    UPDATE address
                    SET full_address = :full_address,
                        building_name = :building_name,
                        street_number = :street_number,
                        street_name = :street_name,
                        street_type = :street_type,
                        direction = :direction,
                        city = :city,
                        met_area = :met_area,
                        province_or_state = :province_or_state,
                        postal_code = :postal_code,
                        country_id = :country_id,
                        time_stamp = CURRENT_TIMESTAMP
                    WHERE address_id = :address_id
                """

                result = self.db_handler.execute_query(update_sql, parsed_address)
                if result is None:
                    logging.warning("Update failed for address_id=%d", address_id)
                    failed_ids.append(address_id)
                    continue

                updated_rows.append(parsed_address)
                logging.info("Updated address_id=%d successfully.", address_id)

            except Exception as e:
                logging.exception("Exception while processing address_id=%d: %s", address_id, str(e))
                failed_ids.append(address_id)

        # 7. Save successful results to CSV
        if updated_rows:
            os.makedirs("output/test", exist_ok=True)
            pd.DataFrame(updated_rows).to_csv("output/test/batch_address_llm_update.csv", index=False)
            logging.info("Saved %d parsed addresses to audit CSV.", len(updated_rows))

        # 8. Log failures
        if failed_ids:
            logging.warning("Failed to update %d address rows. IDs: %s", len(failed_ids), failed_ids)
        else:
            logging.info("All rows processed successfully.")

    
    def parse_location_with_llm(self, location_str: str, prompt_type: str = "address_internet_fix") -> dict:
        """
        Sends the location string to the LLM with a specified prompt type and returns a parsed address dictionary.
        Forces OpenAI use for address_internet_fix only, without changing global config.
        """
        if not location_str or len(location_str.strip()) < 15:
            logging.info("parse_location_with_llm: Location string too short, creating minimal address: %s", location_str)
            # Create minimal address for short location strings
            minimal_address = {
                "building_name": location_str.strip()[:50] if location_str else "Unknown",
                "city": "Unknown",
                "province_or_state": "BC",
                "country_id": "CA"
            }
            address_id = self.db_handler.resolve_or_insert_address(minimal_address)
            return {"address_id": address_id} if address_id else None

        # Load the prompt text from config
        try:
            prompt_text = self.config["prompts"][prompt_type]
        except KeyError:
            logging.error("parse_location_with_llm: No prompt found in config for prompt_type: %s", prompt_type)
            return None

        prompt = f"{prompt_text}\n\n{location_str.strip()}"
        logging.info("parse_location_with_llm: Prompting LLM with: %s", prompt_type)

        # --- Temporarily override LLM provider if needed ---
        original_provider = self.config["llm"].get("provider")
        if prompt_type == "address_internet_fix":
            self.config["llm"]["provider"] = "openai"

        # Query the LLM
        llm_response = self.query_llm('address_fix', prompt)

        if not llm_response:
            logging.error("parse_location_with_llm: LLM returned no response for location: %s", location_str)
            return None

        # Restore original provider
        self.config["llm"]["provider"] = original_provider

        # Parse the response
        parsed_address = self.extract_and_parse_json(llm_response, "address_fix", "address_extraction")
        logging.info("parse_location_with_llm: Parsed address from LLM:\n%s", json.dumps(parsed_address, indent=2))

        if not parsed_address:
            logging.warning("parse_location_with_llm: No valid address returned by LLM")
            return None

        # Fill postal code if missing
        # Check if postal_code is missing or a null string
        postal_code = parsed_address.get("postal_code")
        null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}
        if not postal_code or (isinstance(postal_code, str) and postal_code.lower() in null_strings):
            postal = self.lookup_postal_code_external_db(
                street_number=parsed_address.get("street_number", ""),
                street_name=parsed_address.get("street_name", ""),
                city=parsed_address.get("city", "")
            )
            if postal:
                parsed_address["postal_code"] = postal
                logging.info("parse_location_with_llm: Filled missing postal_code from external DB: %s", postal)

        # Insert into address table (or get existing ID)
        address_id = self.db_handler.resolve_or_insert_address(parsed_address)
        parsed_address["address_id"] = address_id

        return parsed_address


    def parse_location(self, location_str: str, prompt_type: str = "address") -> dict:
        """
        Uses the LLM to parse a free-form location string into a structured address dict.
        """
        prompt, schema_type = self.generate_prompt("synthetic_url_for_address", location_str, prompt_type=prompt_type)
        response = self.query_llm("address_fix", prompt, schema_type)

        if response:
            results = self.extract_and_parse_json(response, "synthetic_url_for_address", "address_extraction")
            if results and isinstance(results, list):
                return results[0]  # Use the first valid address

        return {}
        

# Run the LLM
if __name__ == "__main__":

    # Setup centralized logging
    from logging_config import setup_logging
    setup_logging('llm')

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    logging.info("\n\nllm.py starting...")

    # Get the start time
    start_time = datetime.now()
    logging.info(f"__main__: Starting the crawler process at {start_time}")

    # Instantiate the LLM handler
    llm_handler = LLMHandler(config_path="config/config.yaml")

    # Instantiate the database handler
    db_handler = DatabaseHandler(config)

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before llm.py
    start_df = db_handler.count_events_urls_start(file_name)

    # Run the LLM handler to fix incomplete addresses
    # Does not work currenctly (July 13 2025) because OpenAI does not allow api access to the internet
    # llm_handler.fix_incomplete_addresses_batch()   ***TEMP***

    # Count the event and urls after llm.py
    db_handler.count_events_urls_end(start_df, file_name)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")

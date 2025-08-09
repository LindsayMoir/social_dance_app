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
        # Load configuration and configure logging
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        logging.basicConfig(
            filename=config['logging']['log_file'],
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

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
from datetime import datetime
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

        # Set up OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        self.openai_client = OpenAI()

        # Set up Mistral client
        mistral_api_key = os.environ["MISTRAL_API_KEY"]
        self.mistral_client = Mistral(api_key=mistral_api_key)

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
        Generate a prompt, query a Language Learning Model (LLM), and process the response.

        This method generates a prompt based on the provided URL and extracted text, queries the LLM with the prompt,
        and processes the LLM's response. If the response is successfully parsed, it converts the parsed result into
        a DataFrame, writes the events to the database, and logs the relevant information.

        Args:
            url (str): The URL of the webpage being processed.
            parent_url (str): The parent URL context for the webpage.
            extracted_text (str): The text extracted from the webpage.
            source (str): The source organization name.
            keywords_list (list): A list of keywords relevant to the events.
            prompt_type (str): Specifies which prompt to use. Accepts:
                - Simple keys: 'fb', 'default', 'images', etc.
                - Full URLs: 'https://gotothecoda.com/calendar' for site-specific prompts
                - Falls back to 'default' if prompt_type not found in config
                
        Returns:
            bool: True if the LLM response is successfully processed and events are written to the database, False otherwise.
        """
        # Generate prompt, query LLM, and process the response.
        prompt_text, schema_type = self.generate_prompt(url, extracted_text, prompt_type)
        if len(prompt_text) > self.config['crawling']['prompt_max_length']:
            logging.warning(f"def process_llm_response: Prompt for URL {url} exceeds maximum length. Skipping LLM query.")
            return False
        llm_response = self.query_llm(url, prompt_text, schema_type)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url, schema_type)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                self.db_handler.write_events_to_db(events_df, url, parent_url, source, keywords_list)
                logging.info(f"def process_llm_response: URL {url} marked as relevant with events written to the database.")
                return True
        
        else:
            logging.error(f"def process_llm_response: Failed to process LLM response for URL: {url}")
            return False
        

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

        # Generate the full prompt
        today_date = datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Today's date is: {today_date}. Use this for all date calculations.\n"
            f"{is_relevant_txt}\n"
            f"{extracted_text}\n"
        )

        return prompt, schema_type


    def query_llm(self, url, prompt, schema_type=None):
        """
        Query the configured LLM with a given prompt and return the response.
        Fallback occurs between Mistral and OpenAI if one fails.

        Args:
            url (str): The URL being processed (for logging).
            prompt (str): The prompt to send to the LLM.
            schema_type (str): The schema type for structured output (optional).

        Returns:
            str: The response from the LLM if available, otherwise None.
        """
        if not self.config['llm']['spend_money']:
            logging.info("query_llm(): Spending money is disabled. Skipping the LLM query.")
            return None
        
        # Instantiate response variable
        response = None

        # Mistral does not process the Bard and Banker website correctly, so we need to check the URL
        lower = url.lower()
        if 'bard' in lower:
            provider = 'openai'
        else:
            provider = self.config['llm']['provider']

        if provider == 'openai':
            # Try OpenAI first
            try:
                model = self.config['llm']['openai_model']
                logging.info("query_llm(): Querying OpenAI")
                response = self.query_openai(prompt, model, schema_type=schema_type)
                if response:
                    logging.info(f"query_llm(): OpenAI response received: {response}")
                    return response
            except Exception as e:
                error_message = str(e).replace('error', 'rejection')
                logging.warning(f"query_llm(): OpenAI query failed: {error_message}")

            # Fallback to Mistral
            try:
                model = self.config['llm']['mistral_model']
                logging.info("query_llm(): Falling back to Mistral")
                response = self.query_mistral(prompt, model, schema_type=schema_type)
                if response:
                    logging.info(f"query_llm(): Mistral response received: {response}")
                else:
                    logging.warning("query_llm(): Mistral returned no response.")
            except Exception as e:
                error_message = str(e).replace('error', 'rejection')
                logging.warning(f"query_llm(): Mistral query failed: {error_message}")

        elif provider == 'mistral':
            # Try Mistral first
            try:
                model = self.config['llm']['mistral_model']
                logging.info("query_llm(): Querying Mistral")
                response = self.query_mistral(prompt, model, schema_type=schema_type)
                if response:
                    logging.info(f"query_llm(): Mistral response received: {response}")
                    return response
            except Exception as e:
                error_message = str(e).replace('error', 'rejection')
                logging.warning(f"query_llm(): Mistral query failed: {error_message}")

            # Fallback to OpenAI
            try:
                openai_model = self.config['llm']['openai_model']
                logging.info("query_llm(): Falling back to OpenAI")
                response = self.query_openai(prompt, openai_model, schema_type=schema_type)
                if response:
                    logging.info(f"query_llm(): OpenAI response received: {response}")
                else:
                    logging.warning("query_llm(): OpenAI returned no response.")
            except Exception as e:
                error_message = str(e).replace('error', 'rejection')
                logging.warning(f"query_llm(): OpenAI query failed: {error_message}")

        else:
            logging.error("query_llm(): Invalid LLM provider specified.")
            return None

        if response is None:
            logging.error("query_llm(): Both LLM providers failed to provide a response.")
        return response


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
        json_schema = self._get_json_schema_by_type(schema_type) if schema_type else None
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
        json_schema = self._get_json_schema_by_type(schema_type) if schema_type else None
        response_format = None
        if json_schema:
            # If helper returns the raw JSON Schema, wrap it; if it already has name/schema/strict, keep as is
            if not all(k in json_schema for k in ("name", "schema")):
                json_schema = {
                    "name": schema_type or "StructuredOutput",
                    "schema": json_schema,
                    "strict": True,
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


    def _get_json_schema_by_type(self, schema_type):
        """
        Returns the appropriate JSON schema based on explicit schema type.
        This replaces the brittle keyword detection approach.
        """
        if not schema_type:
            return None
            
        schemas = {
            "event_extraction": {
                "name": "event_extraction",
                "strict": True,
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
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
                        },
                        "required": ["source", "dance_style", "url", "event_type", "event_name", 
                                   "day_of_week", "start_date", "end_date", "start_time", "end_time", 
                                   "price", "location", "description"],
                        "additionalProperties": False
                    }
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
                    "required": ["address_id", "full_address", "street_number", "street_name", 
                               "street_type", "city", "province_or_state", "country_id"],
                    "additionalProperties": False
                }
            },
            
            "deduplication_response": {
                "name": "deduplication_response", 
                "strict": True,
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
                "strict": True,
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
                "strict": True,
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
            
            # Reformat compact JSON to have line breaks
            formatted = "{\n"
            # Split on '","' to separate key-value pairs
            pairs = raw_str[1:-1].split('","')
            for i, pair in enumerate(pairs):
                if i == 0:
                    # First pair: remove leading quote
                    pair = pair.lstrip('"')
                if i == len(pairs) - 1:
                    # Last pair: remove trailing quote and don't add comma
                    pair = pair.rstrip('"')
                    formatted += f'  "{pair}"\n'
                else:
                    # Middle pairs: add quotes back and comma
                    formatted += f'  "{pair}",\n'
            formatted += "}"
            
            logging.debug(f"line_based_parse(): Reformatted JSON:\n{formatted}")
            raw_str = formatted

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
        if "No events found" in result:
            logging.info("extract_and_parse_json(): No events found.")
            return None
        # Allow shorter responses for address deduplication (typically ~50-100 chars)
        min_length = 30 if "address_id" in result and "Label" in result else 100
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
        min_blob_length = 30 if "address_id" in blob and "Label" in blob else 100
        if len(blob) < min_blob_length:
            logging.info(f"extract_and_parse_json(): Blob too short (< {min_blob_length} chars):\n{blob}")
            return None

        # 3) Basic cleanup
        cleaned = re.sub(r'(?<!:)//.*', '', blob)
        cleaned = cleaned.replace('...', '').strip()
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = cleaned.replace("```json", "").replace("```", "")

        # 4) Line-based parse using explicit schema type
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

                if parsed_address.get("address_id", 0) == 0:
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
            logging.info("parse_location_with_llm: Skipping short/empty location string: %s", location_str)
            return {"address_id": 0}

        # Load the prompt text from config
        try:
            prompt_text = self.config["prompts"][prompt_type]
        except KeyError:
            logging.error("parse_location_with_llm: No prompt found in config for prompt_type: %s", prompt_type)
            return {"address_id": 0}

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
            return {"address_id": 0}

        # Restore original provider
        self.config["llm"]["provider"] = original_provider

        # Parse the response
        parsed_address = self.extract_and_parse_json(llm_response, "address_fix")
        logging.info("parse_location_with_llm: Parsed address from LLM:\n%s", json.dumps(parsed_address, indent=2))

        if not parsed_address:
            logging.warning("parse_location_with_llm: No valid address returned by LLM. Returning address_id=0")
            return {"address_id": 0}

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
            results = self.extract_and_parse_json(response, "synthetic_url_for_address")
            if results and isinstance(results, list):
                return results[0]  # Use the first valid address

        return {}
        

# Run the LLM
if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

     # Build log_file name
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging_file = f"logs/{script_name}_log.txt"
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
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
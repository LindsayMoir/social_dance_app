"""
llm.py

This module defines the LLMHandler class, which facilitates interactions with a 
Language Learning Model (LLM) for processing event-related data. It integrates 
with a PostgreSQL database to store and update event information, uses OpenAI 
for language model queries, and leverages configuration settings loaded from 
a YAML file.

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
from openai import OpenAI
import os
import openai
import pandas as pd
import re
import yaml

from db import DatabaseHandler


class LLMHandler():
    def __init__(self, config_path=None):
        # Calculate the path to config.yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'config.yaml')

        # Get config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Instantiate DatabaseHandler if not already in globals
        if 'db_handler' not in globals():
            global db_handler
            db_handler = DatabaseHandler(self.config)

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
        "province_or_state", "postal_code", "country_id", "time_stamp"
        }

        # Regex to pull out each `"key": <rest-of-line>` pair
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
        # Set default value of prompt
        prompt = 'default'

        # Check keywords in the extracted text
        if 'facebook' in url:
            logging.info(f"def driver(): URL {url} 'facebook' is in the URL.")
            fb_status = True
            if fb_status:
                prompt = 'fb'

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
                llm_status = self.process_llm_response(url, parent_url, extracted_text, source, found_keywords, prompt)

                if llm_status:
                    # Mark the event link as relevant
                    relevant = True
                    db_handler.write_url_to_db(url_row)
                    return True
                else:
                    # Mark the event link as irrelevant
                    relevant = False
                    db_handler.write_url_to_db(url_row)
                    return False
            else:
                # Mark the event link as irrelevant
                relevant = False
                db_handler.write_url_to_db(url_row)
                return False
        else:
            logging.info(f"def driver(): No keywords found in text for URL {url}\n search_term {search_term}.")
            # Mark the event link as irrelevant
            relevant = False
            db_handler.write_url_to_db(url_row)
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
    

    def process_llm_response(self, url, parent_url, extracted_text, source, keywords_list, prompt):
        """
        Generate a prompt, query a Language Learning Model (LLM), and process the response.

        This method generates a prompt based on the provided URL and extracted text, queries the LLM with the prompt,
        and processes the LLM's response. If the response is successfully parsed, it converts the parsed result into
        a DataFrame, writes the events to the database, and logs the relevant information.

        Args:
            url (str): The URL of the webpage being processed.
            extracted_text (str): The text extracted from the webpage.
            keywords (list): A list of keywords relevant to the events.

        Returns:
            bool: True if the LLM response is successfully processed and events are written to the database, False otherwise.
        """
        # Generate prompt, query LLM, and process the response.
        prompt = self.generate_prompt(url, extracted_text, prompt)
        if len(prompt) > self.config['crawling']['prompt_max_length']:
            logging.warning(f"def process_llm_response: Prompt for URL {url} exceeds maximum length. Skipping LLM query.")
            return False
        llm_response = self.query_llm(url, prompt)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                db_handler.write_events_to_db(events_df, url, parent_url, source, keywords_list)
                logging.info(f"def process_llm_response: URL {url} marked as relevant with events written to the database.")
                return True
        
        else:
            logging.error(f"def process_llm_response: Failed to process LLM response for URL: {url}")
            return False
        
    
    def generate_prompt(self, url, extracted_text, prompt_type):
        """
        Generate a prompt for a language model using extracted text and configuration details.

        Args:
            url (str): The URL of the webpage from which the text was extracted.
            extracted_text (str): The text extracted from the webpage.
            prompt_type (str): Chooses which prompt to use from config

        Returns:
            str: A formatted prompt string for the language model.
        """
        # Generate the LLM prompt using the extracted text and configuration details.
        logging.info(f"def generate_prompt(): Generating prompt for URL: {url}")

        # If this errors, then prompt_type is 'default'
        try:
            txt_file_path = self.config['prompts'][prompt_type]
        except KeyError:
            txt_file_path = self.config['prompts']['default']
        logging.info(f"def generate_prompt(): prompt type: text file path: {txt_file_path}")
        
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

        logging.info(f"def generate_prompt(): {txt_file_path}")

        return prompt
    
    def query_llm(self, url, prompt):
        """
        Query the configured LLM with a given prompt and return the response.
        Fallback occurs between Mistral and OpenAI if one fails.

        Args:
            prompt (str): The prompt to send to the LLM.

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
                response = self.query_openai(prompt, model)
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
                response = self.query_mistral(prompt, model)
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
                response = self.query_mistral(prompt, model)
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
                response = self.query_openai(prompt, openai_model)
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


    def query_openai(self, prompt, model, image_url=None):
        """
        Handles querying OpenAI LLM, optionally attaching an image.
        - prompt: str of the user text
        - model: e.g. "o4-mini-high" or "gpt-4.1-mini"
        - image_url: optional URL string of an image to include
        """
        # build the message content as a list of text + optional image_url blocks
        content_blocks = [
            {"type": "text", "text": prompt}
        ]
        if image_url:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # send the chat completion
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_blocks}]
        )

        # extract and return the assistant's reply
        if response and response.choices:
            return response.choices[0].message.content.strip()
        return None


    def query_mistral(self, prompt, model):
        """Handles querying Mistral LLM."""

        chat_response = self.mistral_client.chat.complete(
            model= model,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        return chat_response.choices[0].message.content


    def line_based_parse(self, raw_str: str) -> list[dict]:
        """
        Generic line-based parser:
        - Chooses required_keys = ADDRESS_KEYS if 'address_id' appears,
          otherwise EVENT_KEYS.
        - Splits on lines, extracts "key": raw, strips all '"' from values,
          and only keeps records with every required key.
        """
        # Decide whether these are addresses or events
        required_keys = (
            self.ADDRESS_KEYS
            if '"address_id"' in raw_str
            else self.EVENT_KEYS
        )

        records = []
        current = None

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
                        records.append(current)
                    else:
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
                current[key] = raw.replace('"', '').strip()

        return records

    def extract_and_parse_json(self, result: str, url: str):
        """
        1) Early-exit on no data.
        2) Bracket-match to isolate the JSON-like blob.
        3) Basic cleanup (comments, backticks, ellipses, stray commas).
        4) Line-based parse into either events or addresses.
        """
        # 1) Early exits
        if "No events found" in result:
            logging.info("extract_and_parse_json(): No events found.")
            return None
        if len(result) <= 100:
            logging.info("extract_and_parse_json(): Result too short.")
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

        if len(blob) < 100:
            logging.info(f"extract_and_parse_json(): Blob too short:\n{blob}")
            return None

        # 3) Basic cleanup
        cleaned = re.sub(r'(?<!:)//.*', '', blob)
        cleaned = cleaned.replace('...', '').strip()
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = cleaned.replace("```json", "").replace("```", "")

        # 4) Line-based parse for events or addresses
        records = self.line_based_parse(cleaned)
        if not records:
            logging.info("extract_and_parse_json(): No complete records parsed.")
            return None

        return records


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
    logging.info("llm.py starting...")

    # Get the start time
    start_time = datetime.now()
    logging.info(f"__main__: Starting the crawler process at {start_time}")

    # Instantiate the LLM handler
    llm = LLMHandler(config_path="config/config.yaml")

    # Instantiate the database handler
    db_handler = DatabaseHandler(config)

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before llm.py
    start_df = db_handler.count_events_urls_start(file_name)

    # Get a test file
    extracted_text_df = pd.read_csv(config['output']['fb_search_results'])

    # Shrink it down to just the first 5 rows
    extracted_text_df = extracted_text_df.head(5)

    # Establish the constants
    keywords = ['bachata']
    search_term = 'https://facebook.com/search/top?q=events%20victoria%20bc%20canada%20dance%20bachata'

    # Call the driver function
    results_json_list = []
    for index, row in extracted_text_df.iterrows():
        url = row['url']
        extracted_text = row['extracted_text']
        
        # Check for keywords in extracted_text
        found_keywords = [kw for kw in llm.keywords_list if kw in extracted_text.lower()]
        logging.info(f"__main__: Found keywords in text for URL {url}: {found_keywords}")
        if found_keywords:
            llm.driver(url, search_term, extracted_text, '', keywords)
        else:
            logging.info(f"__main__: No keywords found in text for URL {url}.")
            url_row = [url, search_term, 'Facebook', found_keywords, False, 1, datetime.now()]
            db_handler.write_url_to_db(url_row)

    # Count the event and urls after llm.py
    db_handler.count_events_urls_end(start_df, file_name)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")
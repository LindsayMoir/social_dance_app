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
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Instantiate the LLM handler
        llm_handler = LLMHandler(config_path="config/config.yaml")

        # Example of using the driver method with test data
        test_url = "https://example.com/event"
        search_term = "sample search term"
        extracted_text = "Sample extracted event text..."
        org_name = "Sample Organization"
        keywords = ["dance", "event"]

        llm_handler.driver(test_url, search_term, extracted_text, org_name, keywords)

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
import json
import logging
from openai import OpenAI
import os
import pandas as pd
import re
import yaml

from db import DatabaseHandler


class LLMHandler():
    def __init__(self, config_path="config/config.yaml"):
        # Initialize base class

        # Get config
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Need DataBaseHandler, if it is not already in globals
        if 'db_handler' not in globals():
            global db_handler
            db_handler = DatabaseHandler(self.config)


    def driver(self, url, search_term, extracted_text, org_name, keywords_list):
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

        keyword_status = self.check_keywords_in_text(url, extracted_text, keywords_list)
        
        if keyword_status == True or fb_status == True:
            # Call the llm to process the extracted text
            llm_status = self.process_llm_response(url, extracted_text, org_name, keywords_list, prompt)

            if llm_status:
                # Mark the event link as relevant
                db_handler.write_url_to_db('', keywords, url, search_term, relevant=True, increment_crawl_trys=1)
            
            else:
                # Mark the event link as irrelevant
                db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_trys=1)

        else:
            # Mark the event link as irrelevant
            db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_trys=1)
            

    def check_keywords_in_text(self, url, extracted_text, org_name, keywords_list):
        """
        Parameters:
        url (str): The URL of the webpage being checked.
        extracted_text (str): The text extracted from the webpage.
        keywords (list): A comma-separated list of keywords to check in the extracted text.

        Returns:
        bool: True if the text is relevant based on the presence of keywords or 'calendar' in the URL, False otherwise.
        """
        #Set default value of prompt
        prompt = 'default'

        # Check for keywords in the extracted text and determine relevance.
        if keywords_list or 'facebook' in url:
            if any(kw in extracted_text.lower() for kw in keywords_list):
                logging.info(f"def check_keywords_in_text: Keywords found in extracted text for URL: {url}")
                if 'facebook' in url:
                    prompt = 'fb'
                return self.process_llm_response(url, extracted_text, org_name, keywords_list, prompt)
            
        if 'calendar' in url:
            logging.info(f"def check_keywords_in_text: URL {url} marked as relevant because 'calendar' is in the URL.")
            return True

        logging.info(f"def check_keywords_in_text: URL {url} marked as irrelevant since there are no keywords, events, or 'calendar' in URL.")
        return False
    

    def process_llm_response(self, url, extracted_text, org_name, keywords_list, prompt):
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
        prompt = self.generate_prompt(url, extracted_text, 'default')
        llm_response = self.query_llm(prompt, url)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                db_handler.write_events_to_db(events_df, url, org_name, keywords_list)
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
        txt_file_path = self.config['prompts'][prompt_type]
        logging.info(f"def generate_prompt(): prompt type: {prompt_type}, text file path: {txt_file_path}")
        
        # Get the prompt file
        with open(txt_file_path, 'r') as file:
            is_relevant_txt = file.read()

        # Generate the full prompt
        prompt = (
            f"{is_relevant_txt}"
            f"{extracted_text}\n\n"
            
        )

        logging.info(f"def generate_prompt(): \n{txt_file_path}")

        return prompt

    def query_llm(self, prompt, url):
        """
        Args:
            prompt (str): The prompt to send to the LLM.
            url (str): The URL associated with the prompt.
        Returns:
            str: The response from the LLM if available, otherwise None.
        Raises:
            FileNotFoundError: If the keys file specified in the config is not found.
            KeyError: If the 'OpenAI' organization key is not found in the keys file.
            Exception: For any other exceptions that may occur during the API call.
        """
        # Read the API key from the security file
        keys_df = pd.read_csv(self.config['input']['keys'])
        self.api_key = keys_df.loc[keys_df['organization'] == 'OpenAI', 'key_pw'].values[0]

        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI()

        # Query the LLM
        response = self.client.chat.completions.create(
            model=self.config['llm']['url_evaluator'],
            messages=[
            {
                "role": "user", 
                "content": prompt
            }
            ]
        )

        logging.info(f"def query_llm(): LLM response content: \n{response.choices[0].message.content.strip()}\n")

        if response.choices:
            logging.info(f"def query_llm(): LLM response received based on url {url}.")
            return response.choices[0].message.content.strip()
        
        return None

    def extract_and_parse_json(self, result, url):
        """
        Parameters:
        result (str): The response string from which JSON needs to be extracted.
        url (str): The URL from which the response was obtained.
        Returns:
        list or None: Returns a list of events if JSON is successfully extracted and parsed, 
                      otherwise returns None.
        """
        if "No events found" in result:
            logging.info("def extract_and_parse_json(): No events found in result.")
            return None
        
        # Check if the response contains JSON
        if 'json' in result and len(result) > 100:
            logging.info("def extract_and_parse_json(): JSON found in result.")

            # Get just the JSON string from the response
            start_position = result.find('[')
            end_position = result.rfind(']') + 1
            json_string = result[start_position:end_position]

            # Remove single-line comments
            cleaned_str = re.sub(r'\s*//.*', '', json_string)

            # Remove ellipsis patterns (if they occur)
            cleaned_str = cleaned_str.replace('...', '')

            # Ensure the string is a valid JSON array
            cleaned_str = cleaned_str.strip()

            # Remove any trailing commas before the closing bracket
            cleaned_str = re.sub(r',\s*\]', ']', cleaned_str)

            # For debugging: print the cleaned JSON string
            logging.info(f"def extract_and_parse_json(): for url {url}, \nCleaned JSON string: \n{cleaned_str}")

            # Parse the cleaned JSON string
            try:
                # Convert JSON string to Python object
                events_json =json.loads(cleaned_str)
                return events_json
                
            except json.JSONDecodeError as e:
                logging.error(f"def extract_and_parse_json(): Error parsing JSON: {e}")
                return None
            
        else:
            logging.info("def extract_and_parse_json(): No valid events found in result.")
            return None

# Run the LLM
if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set up logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get the start time
    start_time = datetime.now()
    logging.info(f"__main__: Starting the crawler process at {start_time}")

    # Instantiate the LLM handler
    llm = LLMHandler(config_path="config/config.yaml")

    # Instantiate the database handler
    db_handler = llm.db_handler()

    # Get a test file
    extracted_text_df = pd.read_csv('output/extracted_text.csv')

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
        llm.driver(url, search_term, extracted_text, '', keywords)

    # Run deduplication and set calendar URLs
    db_handler.dedup()
    db_handler.set_calendar_urls()

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")

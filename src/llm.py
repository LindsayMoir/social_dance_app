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

        # Set up providers: primary and fallback list (if any)
        # For example, config['llm']['provider'] is the primary, and you could add fallback providers in config['llm']['fallback_providers']
        self.providers = [self.config['llm']['provider']]
        if 'fallback_providers' in self.config['llm']:
            self.providers.extend(self.config['llm']['fallback_providers'])

        # Initialize the clients for each provider
        self.clients = {}
        for provider in self.providers:
            if provider == 'openai':
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
                openai.api_key = self.openai_api_key
                self.clients['openai'] = OpenAI()
            elif provider == 'mistral':
                self.api_key = os.getenv("MISTRAL_API_KEY")
                self.clients['mistral'] = Mistral(api_key=self.api_key)
            elif provider == 'deepseek':
                # Initialize deepseek client here if applicable
                # For example:
                # self.clients['deepseek'] = Deepseek(api_key=os.getenv("DEEPEEK_API_KEY"))
                pass
            else:
                logging.error(f"LLMHandler __init__: Invalid LLM provider specified: {provider}")
                
        self.keywords_list = self.get_keywords()


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
    
        if found_keywords:
            logging.info(f"def driver(): Found keywords in text for URL {url}: {found_keywords}")
        
            if fb_status == True:
                # Call the llm to process the extracted text
                llm_status = self.process_llm_response(url, extracted_text, source, keywords_list, prompt)

                if llm_status:
                    # Mark the event link as relevant
                    db_handler.write_url_to_db('', keywords, url, search_term, relevant=True, increment_crawl_try=1)
                else:
                    # Mark the event link as irrelevant
                    db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_try=1)
            else:
                # Mark the event link as irrelevant
                db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_try=1)
        else:
            logging.info(f"def driver(): No keywords found in text for URL {url}.")
            # Mark the event link as irrelevant
            db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_try=1)
    

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
    

    def process_llm_response(self, url, extracted_text, source, keywords_list, prompt):
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
        llm_response = self.query_llm(prompt)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                db_handler.write_events_to_db(events_df, url, source, keywords_list)
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
        logging.info(f"def generate_prompt(): prompt type: {prompt_type}, text file path: {txt_file_path}")
        
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

        logging.info(f"def generate_prompt(): \n{txt_file_path}")

        return prompt
    

    def query_llm(self, prompt):
        # Get primary provider from config
        primary = self.config['llm']['provider']
        # Get fallback providers (if any) from config
        fallbacks = self.config['llm'].get('fallback_providers', [])
        # Construct an ordered list: primary first, then fallbacks
        providers_to_try = [primary] + fallbacks

        for provider in providers_to_try:
            try:
                if provider == 'openai':
                    model = self.config['llm']['openai_model']
                    logging.info(f"query_llm(): Trying {provider} with model {model}.")
                    return self._query_openai(prompt, model)
                elif provider == 'mistral':
                    model = self.config['llm']['mistral_model']
                    logging.info(f"query_llm(): Trying {provider} with model {model}.")
                    return self._query_mistral(prompt, model)
                elif provider == 'deepseek':
                    # Example placeholder for deepseek; you can implement this later.
                    logging.error("query_llm(): deepseek provider not implemented yet.")
                    continue
                else:
                    logging.error(f"query_llm(): Invalid provider {provider}.")
                    continue
            except Exception as e:
                logging.error(f"query_llm(): Provider {provider} failed with error: {e}")
                continue

        logging.error("query_llm(): All providers failed. Returning None.")
        return None


    def _query_openai(self, prompt, model):
        try:
            response = self.clients['openai'].chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            llm_response = response.choices[0].message.content.strip() if response and response.choices else None
            if llm_response:
                logging.info(f"_query_openai(): Received response.")
            else:
                logging.error("_query_openai(): No response received.")
            return llm_response
        except Exception as e:
            logging.error(f"_query_openai(): OpenAI API call failed: {e}")
            raise e
        

    def _query_mistral(self, prompt, model):
        try:
            response = self.clients['mistral'].chat.complete(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            llm_response = response.choices[0].message.content if response and response.choices else None
            if llm_response:
                logging.info(f"_query_mistral(): Received response.")
            else:
                logging.error("_query_mistral(): No response received.")
            return llm_response
        except Exception as e:
            logging.error(f"_query_mistral(): Mistral API call failed: {e}")
            raise e


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
            if len(result) > 100:
                logging.info("def extract_and_parse_json(): JSON found in result.")

                # Get just the JSON string from the response
                start_position = result.find('[')

                # If no start_position is returned, prepend the string with '['
                if start_position == -1:
                    result = '[' + result
                    start_position = result.find('[')

                # Get the end position of the JSON string
                end_position = result.rfind(']') + 1

                # if no end_position is found append the string with ']'
                if end_position == 0:
                    # This means badly formed json
                    # Find the last occurence of '}' and add ']' after that
                    end_position = result.rfind('}') + 1
                    result = result[:end_position] + ']'
                    end_position = result.rfind(']') + 1

                # Extract the JSON string
                json_string = result[start_position:end_position]
                if len(json_string) < 100:
                    logging.info(f"def extract_and_parse_json(): malformed json: \n{json_string}")
                    return None

                # Remove single-line comments
                cleaned_str = re.sub(r'(?<!:)//.*', '', json_string)

                # Remove ellipsis patterns (if they occur)
                cleaned_str = cleaned_str.replace('...', '')

                # Ensure the string is a valid JSON array
                cleaned_str = cleaned_str.strip()

                # Remove any trailing commas before the closing brackets
                cleaned_str = re.sub(r',\s*\]', ']', cleaned_str)

                # Remove '''json from the string
                cleaned_str = cleaned_str.replace("```json", "")
                cleaned_str = cleaned_str.replace("```", "")

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
            db_handler.write_url_to_db('', keywords, url, search_term, relevant=False, increment_crawl_try=1)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")

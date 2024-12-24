import base64
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import json
import logging
from openai import OpenAI
import os
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import random
import re
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (create_engine, update, MetaData)
import sys
import time
import yaml

print(os.getcwd())

class LLMHandler:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration from a YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set up logging
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info("LLMHandler initialized.")
    
    
    def driver(self, url, keywords, org_name):
        """
        Determine the relevance of a given URL based on its content, keywords, or organization name.

        Parameters:
        url (str): The URL to be evaluated.
        keywords (list of str): A list of keywords to check within the URL content.
        org_name (str): The name of the organization to check within the URL content.

        Returns:
        bool: True if the URL is relevant, False otherwise.
        """

        # Extract links from the main page but handle facebook links differently
        if 'facebook' in url:

            # Process the regular group facebook page
            # We want to see if there are any facebook event links in this url
            facebooks_events_links = self.fb_get_event_links(url)

            if facebooks_events_links:
                for event_link in facebooks_events_links:

                    # Write the event link to the database
                    write_url_to_db(org_name, keywords, event_link, other_links=url, relevant=True, increment_crawl_trys=1)

                    # Call playwright to extract the event details
                    extracted_text = self.fb_extract_text(event_link)

                    # Check keywords in the extracted text
                    keyword_status = self.check_keywords_in_text(event_link, extracted_text, keywords, org_name)

                    if keyword_status:
                        # Call the llm to process the extracted text
                        llm_status = self.process_llm_response(event_link, extracted_text, keywords, org_name)

                        if llm_status:
                            # Mark the event link as relevant
                            update_url(event_link, url, increment_crawl_trys=0, relevant=True)
                        
                        else:
                            # Mark the event link as irrelevant
                            update_url(event_link, url, relevant=False, increment_crawl_trys=0)

                    else:
                        # Mark the event link as irrelevant
                        update_url(event_link, url, relevant=False, increment_crawl_trys=0)

            else:
                # Normal facebook text. I want to process this with playwright but the version that only returns text
                extracted_text = self.fb_extract_text(url)

                # Check keywords in the extracted text
                keyword_status = self.check_keywords_in_text(url, extracted_text, keywords, org_name)

                if keyword_status:
                    # Call the llm to process the extracted text
                    llm_status = self.process_llm_response(url, extracted_text, keywords, org_name)

                    if llm_status:
                        # Mark the event link as relevant
                        update_url(url, update_other_links='', relevant=True, increment_crawl_trys=0)
                    
                    else:
                        # Mark the event link as irrelevant
                        update_url(url, update_other_links='', relevant=False, increment_crawl_trys=0)

                else:
                    logging.info(f"def parse(): URL {url} marked as irrelevant since there are no keywords and/or events in URL.")
                    update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)

        else:
            # Process non-facebook links
            extracted_text = self.extract_text_with_playwright(url)

            # Check keywords in the extracted text
            keyword_status = self.check_keywords_in_text(url, extracted_text, keywords, org_name)

            if keyword_status:
                # Call the llm to process the extracted text
                llm_status = self.process_llm_response(url, extracted_text, keywords, org_name)

                if llm_status:
                    # Mark the event link as relevant
                    update_url(url, update_other_links='', relevant=True, increment_crawl_trys=0)

                else:
                    # Mark the event link as irrelevant
                    update_url(url, update_other_links='', relevant=False, increment_crawl_trys=0) 

            else:
                logging.info(f"def parse(): URL {url} marked as irrelevant since there are no keywords.")
                update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)


    def check_keywords_in_text(self, url, extracted_text, keywords, org_name):
        """
        Parameters:
        url (str): The URL of the webpage being checked.
        extracted_text (str): The text extracted from the webpage.
        keywords (str, optional): A comma-separated string of keywords to check in the extracted text. Defaults to None.
        org_name (str, optional): The name of the organization, used for further processing if keywords are found. Defaults to None.

        Returns:
        bool: True if the text is relevant based on the presence of keywords or 'calendar' in the URL, False otherwise.
        """
        # Check for keywords in the extracted text and determine relevance.
        if keywords:
            keywords_list = [kw.strip().lower() for kw in keywords.split(',')]
            if any(kw in extracted_text.lower() for kw in keywords_list):
                logging.info(f"def check_keywords_in_text: Keywords found in extracted text for URL: {url}")
                return self.process_llm_response(url, extracted_text, keywords, org_name)

        if 'calendar' in url:
            logging.info(f"def check_keywords_in_text: URL {url} marked as relevant because 'calendar' is in the URL.")
            return True

        logging.info(f"def check_keywords_in_text: URL {url} marked as irrelevant since there are no keywords, events, or 'calendar' in URL.")
        return False
    

    def process_llm_response(self, url, extracted_text, keywords, org_name):
        """
        Generate a prompt, query a Language Learning Model (LLM), and process the response.

        This method generates a prompt based on the provided URL and extracted text, queries the LLM with the prompt,
        and processes the LLM's response. If the response is successfully parsed, it converts the parsed result into
        a DataFrame, writes the events to the database, and logs the relevant information.

        Args:
            url (str): The URL of the webpage being processed.
            extracted_text (str): The text extracted from the webpage.
            keywords (list): A list of keywords relevant to the events.
            org_name (str): The name of the organization associated with the events.

        Returns:
            bool: True if the LLM response is successfully processed and events are written to the database, False otherwise.
        """
        # Generate prompt, query LLM, and process the response.
        prompt = self.generate_prompt(url, extracted_text)
        llm_response = self.query_llm(prompt, url)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)
            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                write_events_to_db(events_df, url, keywords, org_name)
                logging.info(f"def process_llm_response: URL {url} marked as relevant with events written to the database.")
                return True
        
        else:
            logging.error(f"def process_llm_response: Failed to process LLM response for URL: {url}")
            return False
        
    
    def generate_prompt(self, url, extracted_text):
        """
        Generate a prompt for a language model using extracted text and configuration details.

        Args:
            url (str): The URL of the webpage from which the text was extracted.
            extracted_text (str): The text extracted from the webpage.

        Returns:
            str: A formatted prompt string for the language model.
        """
        #Generate the LLM prompt using the extracted text and configuration details.
        txt_file_path = config['prompts']['is_relevant']
        with open(txt_file_path, 'r') as file:
            is_relevant_txt = file.read()

        # Define the date range for event identification
        start_date = datetime.now()
        end_date = start_date + pd.DateOffset(months=config['date_range']['months'])

        # Generate the full prompt
        prompt = (
            f"The following text was extracted from a webpage {url}\n\n"
            f"{extracted_text}\n\n"
            f"Identify any events (social dances, classes, workshops) within the date range "
            f"{start_date} to {end_date}."
            f"{is_relevant_txt}"
        )

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

        print('\ndef query_llm(): *************prompt************\n', prompt)

        # Read the API key from the security file
        keys_df = pd.read_csv(config['input']['keys'])
        api_key = keys_df.loc[keys_df['Organization'] == 'OpenAI', 'Key'].values[0]

        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()

        # Query the LLM
        response = client.chat.completions.create(
            model=config['llm']['url_evaluator'],
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )

        print('\ndef query_llm(): *****response.choices[0].message.content.strip()*******\n', response.choices[0].message.content.strip())

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
        if 'json' in result:
            start_position = result.find('[')
            end_position = result.find(']') + 1
            if start_position != -1 and end_position != -1:
                # Extract JSON part
                json_string = result[start_position:end_position]

                try:
                    # Convert JSON string to Python object
                    logging.info("def extract_and_parse_json(): JSON found in result.")
                    events_json =json.loads(json_string)
                    return events_json
                    
                except json.JSONDecodeError as e:
                    logging.error(f"def extract_and_parse_json(): Error parsing JSON: {e}")
                    return None

        logging.info("def extract_and_parse_json(): No JSON found in result.")
        
        return None
    
   
# Run the crawler
if __name__ == "__main__":

    # Get the start time
    start_time = datetime.now()
    logging.info(f"__main__: Starting the crawler process at {start_time}")

    # Instantiate the LLM handler
    llm = LLMHandler(config_path="config/config.yaml")

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")

from datetime import datetime
import json
import logging
from openai import OpenAI
import os
import pandas as pd
import yaml


from db import DatabaseHandler

class LLMHandler:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration from a YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        logging.info("LLMHandler initialized.")
    
    
    def driver(self, db_handler, url, search_term, extracted_text, keywords):
        """
        Determine the relevance of a given URL based on its content, keywords, or organization name.

        Parameters:
        url (str): The URL to be evaluated.
        keywords (list of str): A list of keywords to check within the URL content.

        Returns:
        bool: True if the URL is relevant, False otherwise.
        """
        # Check keywords in the extracted text
        keyword_status = self.check_keywords_in_text(db_handler, url, extracted_text, keywords)

        if keyword_status:
            # Call the llm to process the extracted text
            llm_status = self.process_llm_response(db_handler, url, extracted_text, keywords)

            if llm_status:
                # Mark the event link as relevant
                db_handler.write_url_to_db(keywords, url, search_term, relevant=False, increment_crawl_trys=1)
            
            else:
                # Mark the event link as irrelevant
                db_handler.write_url_to_db(keywords, url, search_term, relevant=True, increment_crawl_trys=1)

        else:
            # Mark the event link as irrelevant
            db_handler.write_url_to_db(keywords, url, search_term, relevant=True, increment_crawl_trys=1)


    def check_keywords_in_text(self, db_handler, url, extracted_text, keywords_list):
        """
        Parameters:
        url (str): The URL of the webpage being checked.
        extracted_text (str): The text extracted from the webpage.
        keywords (list, optional): A comma-separated list of keywords to check in the extracted text. Defaults to None.

        Returns:
        bool: True if the text is relevant based on the presence of keywords or 'calendar' in the URL, False otherwise.
        """
        # Check for keywords in the extracted text and determine relevance.
        if keywords_list:
            if any(kw in extracted_text.lower() for kw in keywords_list):
                logging.info(f"def check_keywords_in_text: Keywords found in extracted text for URL: {url}")
                return self.process_llm_response(db_handler, url, extracted_text, keywords_list)

        if 'calendar' in url:
            logging.info(f"def check_keywords_in_text: URL {url} marked as relevant because 'calendar' is in the URL.")
            return True

        logging.info(f"def check_keywords_in_text: URL {url} marked as irrelevant since there are no keywords, events, or 'calendar' in URL.")
        return False
    

    def process_llm_response(self, db_handler, url, extracted_text, keywords):
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
        prompt = self.generate_prompt(url, extracted_text)
        llm_response = self.query_llm(prompt, url)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)

            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                db_handler.write_events_to_db(events_df, url, keywords)
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
        # Generate the LLM prompt using the extracted text and configuration details.

        print('******url******\n', url)

        if 'facebook' in url:
            txt_file_path = self.config['prompts']['fb_prompt']
        else:
            txt_file_path = self.config['prompts']['is_relevant']
        with open(txt_file_path, 'r') as file:
            is_relevant_txt = file.read()

        # Define the date range for event identification
        start_date = datetime.now()
        end_date = start_date + pd.DateOffset(months=self.config['date_range']['months'])

        # Generate the full prompt
        prompt = (
            f"The following text was extracted from a webpage {url}\n\n"
            f"{extracted_text}\n\n"
            f"Identify any events (social dances, classes, workshops) within the date range "
            f"start_date ({start_date}) to end_date ({end_date})."
            f"{is_relevant_txt}"
            f"6. Location:"
            f"•	Anything futher than {self.config['location']['distance']} kilometers from {self.config['location']['epicentre']}"
            f"  should not be included in your response."
            f"•	Use your knowledge of places, and streets and avenues in {self.config['location']['epicentre']} to make sure that "
            f"  the event is actually in this location. If it is not in this area, do not include the event in your json response."
        )

        print(prompt)

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
        if not hasattr(self, 'api_key'):
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
                    events_json = json.loads(json_string)
                    return events_json
                    
                except json.JSONDecodeError as e:
                    logging.error(f"def extract_and_parse_json(): Error parsing JSON: {e}")
                    return None

        logging.info("def extract_and_parse_json(): No JSON found in result.")
        
        return None
    
   
# Run the LLM
if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(
        filename="logs/llm.log",
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
    db_handler = DatabaseHandler(llm.config)

    # Get a test file
    extracted_text_df = pd.read_csv('output/extracted_text.csv')

    # Shrink it down to just the first 5 rows
    extracted_text_df = extracted_text_df.head(5)

    # Establish the constants
    keywords = 'bachata'
    search_term = 'https://facebook.com/search/top?q=events%20victoria%20bc%20canada%20dance%20bachata'

    # Call the driver function
    results_json_list = []
    for index, row in extracted_text_df.iterrows():
        url = row['url']
        extracted_text = row['extracted_text']
        llm.driver(db_handler, url, search_term, extracted_text, keywords)

    # Run deduplication and set calendar URLs
    db_handler.dedup()
    db_handler.set_calendar_urls()

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}")

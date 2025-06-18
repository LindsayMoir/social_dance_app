"""
This script defines a GmailProcessor class that handles the authentication, fetching, and processing of emails from a 
Gmail account using the Gmail API. The class is initialized with a configuration file and an optional LLMHandler instance 
for processing email content.

Classes:
    GmailProcessor: Handles Gmail API authentication, email fetching, and processing.

Methods:
    __init__(self, config_path="config/config.yaml", llm_handler=None):

    load_config(self, config_path):

    setup_logging(self):

    authenticate_gmail(self):

    fetch_latest_email(self, email_regex):

    driver(self, csv_path):

    main():

Usage:
    Run the script to initialize the GmailProcessor and process emails from a CSV file specified in the configuration.
"""
import base64
from dotenv import load_dotenv
from email import message_from_bytes
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging
import os
import pandas as pd
import re
import yaml

with open('config/config.yaml', "r") as file:
    config = yaml.safe_load(file)

from db import DatabaseHandler

 # Build log_file name
script_name = os.path.splitext(os.path.basename(__file__))[0]
logging_file = f"{script_name}_log" 
logging.basicConfig(
    filename=logging_file,
    filemode="a",  # Append mode to preserve logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

logging.info("\n\nStarting Gmail Processor...")

# Load environment variables from .env
load_dotenv()

class GmailProcessor:
    def __init__(self, llm_handler=None):
        self.llm_handler = llm_handler
        self.client_secret_path = os.getenv("GMAIL_CLIENT_SECRET_PATH")
        self.token_path = os.getenv("GMAIL_TOKEN_PATH")
        self.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        self.service = self.authenticate_gmail()


    def authenticate_gmail(self):
        """
        Authenticates with Gmail API using OAuth2 and returns the service instance.
        If the token is expired or revoked, it will automatically trigger a new authentication flow.
        """
        creds = None

        try:
            if os.path.exists(self.token_path):
                creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
        except Exception as e:
            logging.error(f"Error loading token file: {e}")

        if not creds or not creds.valid:
            try:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    logging.info("def authenticate_gmail(): Refreshed expired credentials.")
                else:
                    raise RefreshError("Token is invalid or expired.")  # Force a new authentication flow
            except RefreshError as e:
                logging.warning(f"Token refresh failed: {e}. Initiating new OAuth authentication flow.")
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_path, self.scopes)
                creds = flow.run_local_server(port=0)  # Launches browser for authentication
                logging.info("def authenticate_gmail(): OAuth authentication completed successfully.")

                # Save new credentials
                with open(self.token_path, "w") as token_file:
                    token_file.write(creds.to_json())
                    logging.info(f"Saved new token file at {self.token_path}")

        logging.info("Successfully authenticated with Gmail API.")
        return build("gmail", "v1", credentials=creds)


    def fetch_latest_email(self, email_regex):
        """
        Fetches the most recent email matching the given regex pattern.

        :param email_regex: Regular expression pattern for filtering emails.
        :return: Extracted text from the email or None if not found.
        """
        logging.info(f"def fetch_latest_email(): Searching for latest email from: {email_regex}")
        query = f"from:{email_regex} label:archive"
        results = self.service.users().messages().list(userId="me", q=query, maxResults=1).execute()

        if "messages" not in results:
            raise Exception(f"def fetch_latest_email(): No emails found matching: {email_regex}")

        message_id = results["messages"][0]["id"]
        message = self.service.users().messages().get(userId="me", id=message_id, format="raw").execute()
        raw_email = base64.urlsafe_b64decode(message["raw"].encode("ASCII"))
        msg = message_from_bytes(raw_email)

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    extracted_text = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    return extracted_text
        else:
            extracted_text = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            return extracted_text

        return None


    def driver(self, csv_path):
        """
        Reads a CSV file and processes emails in a loop.

        :param csv_path: Path to the CSV file with columns: email, source, keywords, prompt.
        """
        logging.info(f"def driver(): Processing emails from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            email, source, keywords, prompt = row
            extracted_text = self.fetch_latest_email(email)

            if extracted_text:
                # Process extracted text with LLMHandler
                parent_url = 'email inbox'
                llm_status = self.llm_handler.process_llm_response(email, parent_url, extracted_text, source, keywords, prompt)
                if llm_status:
                    logging.info(f"def driver(): process_llm_response success for email: {email}")
                else:
                    logging.warning(f"def driver(): process_llm_response failed for email: {email}")
            else:
                logging.error(f"def driver(): No extracted text pulled from email: {email}")

        logging.info(f"Processed {idx} emails from emails .csv")
        
        return
    

    @staticmethod
    def main():
        """
        Main function that initializes the Gmail processor and processes emails from a CSV file.
        """
        from llm import LLMHandler  # Import here to avoid circular dependency

        # Instantiate LLM handler with config
        llm_handler = LLMHandler(config_path="config/config.yaml")

        # Create GmailProcessor with shared authentication
        gmail_processor = GmailProcessor(llm_handler=llm_handler)

        # Process all emails from CSV
        gmail_processor.driver(config["input"]["emails"])


if __name__ == "__main__":
    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before emails.py
    start_df = db_handler.count_events_urls_start(file_name)

    GmailProcessor.main()

    # Count events and urls after emails.py
    db_handler.count_events_urls_end(start_df, file_name)

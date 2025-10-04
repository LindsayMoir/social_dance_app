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
logging_file = f"logs/{script_name}_log.txt" 
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
        from secret_paths import is_render_environment

        self.llm_handler = llm_handler
        self.is_render = is_render_environment()

        # On Render, skip Gmail authentication entirely
        if self.is_render:
            logging.info("Running on Render - will use CSV file instead of Gmail API")
            self.service = None
            return

        # Local environment - authenticate with Gmail
        from secret_paths import get_secret_path

        # Get paths from env vars (for local development)
        local_client_secret = os.getenv("GMAIL_CLIENT_SECRET_PATH")
        local_token = os.getenv("GMAIL_TOKEN_PATH")

        # Use Render secret paths if available, otherwise use local paths
        self.client_secret_path = get_secret_path("desktop_client_secret.json", local_client_secret)
        self.token_path = get_secret_path("desktop_client_secret_token.json", local_token)

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

                # Sync to database
                from secret_paths import sync_auth_to_db
                sync_auth_to_db(self.token_path, 'google')

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

        On Render: Reads events from output/email_events.csv and writes to database
        Locally: Processes emails from Gmail and writes to BOTH database AND output/email_events.csv

        :param csv_path: Path to the CSV file with columns: email, source, keywords, prompt.
        """
        # Initialize database handler
        db_handler = DatabaseHandler(config)

        # RENDER MODE: Read from CSV and load into database
        if self.is_render:
            return self._process_from_csv(db_handler)

        # LOCAL MODE: Process emails and write to both database and CSV
        return self._process_from_gmail(csv_path, db_handler)


    def _process_from_csv(self, db_handler):
        """
        Render mode: Read events from configured CSV path and insert into database.
        """
        csv_file = config['input']['email_events']

        if not os.path.exists(csv_file):
            logging.warning(f"def _process_from_csv(): CSV file not found: {csv_file}")
            logging.warning("No email events to process on Render. Run emails.py locally first to generate the CSV.")
            return

        logging.info(f"def _process_from_csv(): Reading email events from {csv_file}")
        events_df = pd.read_csv(csv_file)

        if events_df.empty:
            logging.info("def _process_from_csv(): No events found in CSV file")
            return

        # Insert events into database using the same method as write_events_to_db
        try:
            events_df.to_sql('events', db_handler.conn, if_exists='append', index=False, method='multi')
            logging.info(f"def _process_from_csv(): Successfully inserted {len(events_df)} events from CSV into database")
        except Exception as e:
            logging.error(f"def _process_from_csv(): Failed to insert events from CSV: {e}")
            logging.info("def _process_from_csv(): Attempting individual row insertion using multiple_db_inserts...")
            try:
                # Convert DataFrame to list of dicts for multiple_db_inserts
                events_list = events_df.to_dict('records')
                # Convert NaN to None
                events_list = [{k: (None if pd.isna(v) else v) for k, v in event.items()} for event in events_list]
                db_handler.multiple_db_inserts('events', events_list)
                logging.info(f"def _process_from_csv(): Successfully inserted {len(events_list)} events using multiple_db_inserts")
            except Exception as fallback_error:
                logging.error(f"def _process_from_csv(): Fallback insertion also failed: {fallback_error}")
        return


    def _process_from_gmail(self, csv_path, db_handler):
        """
        Local mode: Process emails from Gmail and write to both database and CSV.
        """
        logging.info(f"def _process_from_gmail(): Processing emails from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Track results per email
        email_results = {}
        successful_emails = []
        failed_emails = []

        # Track total events before processing
        total_events_before = db_handler.execute_query("SELECT COUNT(*) FROM events")[0][0]

        for idx, row in df.iterrows():
            email, source, keywords, prompt_type = row

            # Get event count before processing this email
            events_before = db_handler.execute_query("SELECT COUNT(*) FROM events")[0][0]

            extracted_text = self.fetch_latest_email(email)

            if extracted_text:
                # Process extracted text with LLMHandler (this will call write_events_to_db)
                parent_url = 'email inbox'
                llm_status = self.llm_handler.process_llm_response(email, parent_url, extracted_text, source, keywords, prompt_type)

                # Get event count after processing this email
                events_after = db_handler.execute_query("SELECT COUNT(*) FROM events")[0][0]
                events_added = events_after - events_before

                if llm_status:
                    logging.info(f"def _process_from_gmail(): process_llm_response success for email: {email}")
                    successful_emails.append(email)
                    email_results[email] = events_added
                    logging.info(f"Email {email}: Added {events_added} events to database")
                else:
                    logging.warning(f"def _process_from_gmail(): process_llm_response failed for email: {email}")
                    failed_emails.append(email)
                    email_results[email] = 0
            else:
                logging.error(f"def _process_from_gmail(): No extracted text pulled from email: {email}")
                failed_emails.append(email)
                email_results[email] = 0

        # Query all newly added events from database and export to CSV
        total_events_after = db_handler.execute_query("SELECT COUNT(*) FROM events")[0][0]
        total_events_added = total_events_after - total_events_before

        csv_output_path = config['input']['email_events']
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

        if total_events_added > 0:
            # Get the most recent events that were just added
            query = f"SELECT * FROM events ORDER BY event_id DESC LIMIT {total_events_added}"
            new_events_df = pd.read_sql(query, db_handler.conn)
            # Reverse so oldest events are first (matching insertion order)
            new_events_df = new_events_df.iloc[::-1].reset_index(drop=True)
            new_events_df.to_csv(csv_output_path, index=False)
            logging.info(f"def _process_from_gmail(): Exported {len(new_events_df)} events to {csv_output_path}")
        else:
            # Create empty file if no events
            pd.DataFrame().to_csv(csv_output_path, index=False)
            logging.info(f"def _process_from_gmail(): No events to export, created empty {csv_output_path}")

        # Summary logging
        total_emails = len(df)
        total_events_added = sum(email_results.values())

        logging.info(f"Email processing summary:")
        logging.info(f"Total emails in CSV: {total_emails}")
        logging.info(f"Successfully processed: {len(successful_emails)} emails: {successful_emails}")
        logging.info(f"Failed to process: {len(failed_emails)} emails: {failed_emails}")
        logging.info(f"Events added per email:")
        for email, count in email_results.items():
            logging.info(f"  {email}: {count} events added")
        logging.info(f"Total events added across all emails: {total_events_added}")

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

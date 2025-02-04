import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

def get_credentials(organization):
    """
    Retrieves credentials for a given organization from environment variables.

    Args:
        organization (str): The organization for which to retrieve credentials.

    Returns:
        tuple: (appid_uid, key_pw, cse_id)
    """
    # Convert organization name to uppercase and replace spaces with underscores
    org_prefix = organization.upper().replace(" ", "_")

    # Construct environment variable names dynamically
    appid_uid = os.getenv(f"{org_prefix}_APPID_UID")
    key_pw = os.getenv(f"{org_prefix}_KEY_PW")
    cse_id = os.getenv(f"{org_prefix}_CSE_ID")

    # Check if any credential is missing
    if not all([appid_uid, key_pw, cse_id]):
        logging.error(f"Missing credentials for organization: {organization}")
        raise ValueError(f"Missing environment variables for organization: {organization}")

    logging.info(f"def get_credentials(): Retrieved credentials for {organization}.")
    return key_pw, appid_uid, cse_id

if __name__ == "__main__":
    # Test the function by retrieving credentials for an organization
    appid_uid, key_pw, cse_id = get_credentials('Google')
    print(appid_uid, key_pw, cse_id)
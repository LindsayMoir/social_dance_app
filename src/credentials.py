# credentials.py
import pandas as pd
import logging

def get_credentials(config, organization):
    """
    Retrieves credentials for a given organization from the keys file using the provided configuration.

    Args:
        config (dict): Configuration containing the path to the keys file.
        organization (str): The organization for which to retrieve credentials.

    Returns:
        tuple: appid_uid, key_pw, cse_id for the organization.
    """
    keys_df = pd.read_csv(config['input']['keys'])
    keys_df = keys_df[keys_df['organization'] == organization]
    appid_uid, key_pw, cse_id = keys_df.iloc[0][['appid_uid', 'key_pw', 'cse_id']]
    logging.info(f"Retrieved credentials for {organization}.")
    
    return appid_uid, key_pw, cse_id

# pipeline.py

import argparse
import copy
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import logging
import os

# Load .env from src directory (where this script is located)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
import pandas as pd
import shutil
import subprocess
import sys
import time
import yaml

# Setup centralized logging (logging_config.py is in the same directory)
from logging_config import setup_logging
setup_logging('pipeline')

# Configure Prefect based on environment
if os.getenv('RENDER') == 'true':
    # On Render: Use Prefect Cloud for remote monitoring
    # The PREFECT_API_CLOUD_URL and PREFECT_API_KEY from .env will be used
    os.environ['PREFECT_API_URL'] = os.getenv('PREFECT_API_CLOUD_URL', '')
    # Remove local server URL if set
    os.environ.pop('PREFECT_SERVER_DATABASE_CONNECTION_URL', None)
    logging.info("Prefect configured for Render (using Prefect Cloud)")

    # Set Playwright browser path for Render environment
    # This ensures all subprocess calls (scraper.py, fb.py, etc.) can find the browsers
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/render/project/src/.playwright'
    logging.info("Playwright browser path set to /opt/render/project/src/.playwright")

    # Reduce only the most verbose Prefect internal logging
    # Keep INFO level for application logs, but reduce Prefect framework noise
    logging.getLogger('prefect.flow_engine').setLevel(logging.WARNING)
    logging.getLogger('prefect.task_engine').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Silence HTTP request logs
    logging.getLogger('httpcore').setLevel(logging.WARNING)
else:
    # Local: Use local Prefect server
    logging.info("Prefect configured for local server")

from prefect import flow, task


# ─── 1) Load YAML config ─────────────────────────────────────────────────────
CONFIG_PATH = "config/config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

log_cfg = cfg.get("logging", {})
# fallback to a "dir" or default "logs" folder
log_dir = log_cfg.get("dir") or os.path.dirname(log_cfg.get("log_file", "")) or "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)

# Define common configuration updates for all pipeline steps
COMMON_CONFIG_UPDATES = {
    "testing": {"drop_tables": False},
    "crawling": {
         "headless": True,
         "max_website_urls": 10,
         "urls_run_limit": 500,  # default for all steps
    },
    "llm": {
        "provider": "round_robin",
        "provider_rotation_enabled": True,
        "provider_rotation_order": ["openai", "mistral", "gemini"],
        "fallback_enabled": True,
        "fallback_provider_order": ["openai", "gemini", "mistral"],
        "spend_money": True,
    }
}

PARALLEL_CRAWL_CONFIG_UPDATES = copy.deepcopy(COMMON_CONFIG_UPDATES)
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["urls_run_limit"] = 500
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["max_website_urls"] = 10
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_download_timeout_seconds"] = 35
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_playwright_timeout_ms"] = 35000
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_retry_times"] = 1
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_post_load_wait_ms"] = 1000
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_concurrent_requests"] = 16
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_concurrent_requests_per_domain"] = 8
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_base_urls_limit"] = 180
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_event_links_per_base_limit"] = 20
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_post_nav_wait_ms"] = 1800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_post_expand_wait_ms"] = 900
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_final_wait_ms"] = 700
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_failures_before_cooldown"] = 2
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_cooldown_base_seconds"] = 300
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_cooldown_max_seconds"] = 1800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_state_max_scopes"] = 800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_state_ttl_days"] = 45

# ------------------------
# HELPER TASKS: Backup and Restore Config
# ------------------------

@task
def backup_and_update_config(step: str, updates: dict) -> dict:
    with open(CONFIG_PATH, "r") as f:
        original_config = yaml.safe_load(f)
    logger.info("def backup_and_update_config(): Original config loaded.")
    logger.info("def backup_and_update_config(): Starting pipeline.py")
    updated_config = copy.deepcopy(original_config)
    for key, value in updates.items():
        if key in updated_config and isinstance(updated_config[key], dict) and isinstance(value, dict):
            updated_config[key].update(value)
        else:
            updated_config[key] = value
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(updated_config, f)
    logger.info(f"def backup_and_update_config(): Updated config for step '{step}' written to disk with updates: {updates}")
    return original_config

@task
def restore_config(original_config: dict, step: str):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(original_config, f)
    logger.info(f"def restore_config(): Original config restored after step '{step}'.")

# ------------------------
# HELPER TASK: Write Run-Specific Config (for traceability)
# ------------------------
@task
def write_run_config(script_name: str, cfg: dict):
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{script_name}_{run_time}.yaml"
    folder = os.path.join("config", "run_specific_configs")
    os.makedirs(folder, exist_ok=True)
    file_path_config = os.path.join(folder, filename)
    with open(file_path_config, "w") as f:
        yaml.dump(cfg, f)
    logger.info(f"def write_run_config(): Run config for {script_name} written to {file_path_config}")
    return file_path_config

# ------------------------
# GENERIC DUMMY TASKS (for steps with no special pre/post-processing)
# ------------------------
@task
def dummy_pre_process(step: str) -> bool:
    logger.info(f"def dummy_pre_process(): {step} pre-processing: no checks required.")
    return True

@task
def dummy_post_process(step: str) -> bool:
    logger.info(f"def dummy_post_process(): {step} post-processing: no checks required.")
    return True

# ------------------------
# TASKS FOR CREDENTIAL VALIDATION STEP
# ------------------------
@task
def pre_process_credential_validation():
    """Pre-process for credential validation - always returns True (no prerequisites)."""
    logger.info("def pre_process_credential_validation(): No prerequisites required.")
    return True

@task
def run_credential_validation():
    """
    Validates Gmail, Eventbrite, and Facebook credentials before pipeline execution.
    Runs credential_validator.py as a subprocess with its own log file.
    Runs with headless=False to allow user interaction for OAuth, 2FA, CAPTCHAs.
    Returns "Script completed successfully" if all validations pass, raises Exception otherwise.
    """
    try:
        logger.info("def run_credential_validation(): Executing credential_validator.py as subprocess...")
        result = subprocess.run([sys.executable, "src/credential_validator.py"], check=True)
        logger.info("def run_credential_validation(): credential_validator.py executed successfully.")
        logger.info("def run_credential_validation(): All credentials validated - pipeline will continue with headless=True")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"credential_validator.py failed with return code: {e.returncode}"
        logger.error(f"def run_credential_validation(): {error_message}")
        logger.error("def run_credential_validation(): Check logs/credential_validator_log.txt for details")
        raise Exception(error_message)

@task
def post_process_credential_validation():
    """Post-process for credential validation - always returns True."""
    logger.info("def post_process_credential_validation(): Credential validation completed successfully.")
    return True

@flow(name="Credential Validation Step")
def credential_validation_step():
    """
    Pipeline step for validating credentials before execution.
    NOTE: This step does NOT use COMMON_CONFIG_UPDATES because it needs headless=False,
    while the rest of the pipeline uses headless=True.
    """
    logger.info("=" * 70)
    logger.info("CREDENTIAL VALIDATION STEP")
    logger.info("Validating Gmail, Eventbrite, and Facebook credentials")
    logger.info("Browser will open for user interaction if needed")
    logger.info("=" * 70)

    # Pre-process
    pre_result = pre_process_credential_validation()
    logger.info(f"credential_validation_step: pre_process returned: {pre_result}")
    if not pre_result:
        raise Exception("Credential validation pre-processing failed. Pipeline stopped.")

    # Main validation - this should BLOCK until complete
    logger.info("credential_validation_step: About to call run_credential_validation()")
    validation_result = run_credential_validation()
    logger.info(f"credential_validation_step: run_credential_validation returned: {validation_result}")

    # Post-process
    post_result = post_process_credential_validation()
    logger.info(f"credential_validation_step: post_process returned: {post_result}")
    if not post_result:
        raise Exception("Credential validation post-processing failed. Pipeline stopped.")

    logger.info("credential_validation_step: Step completed successfully")
    return True

# ------------------------
# TASK: COPY LOG FILES
# ------------------------
@task
def copy_log_files():
    """Move all log files to a timestamped folder in logs directory."""
    # Create timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_folder = f"logs/logs_{timestamp}"
    
    # Create the archive folder
    os.makedirs(archive_folder, exist_ok=True)
    logger.info(f"def copy_log_files(): Created archive folder: {archive_folder}")
    
    # Get all log files from logs directory
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        logger.warning(f"def copy_log_files(): Logs directory {logs_dir} does not exist.")
        return True
    
    log_files_moved = 0
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log') or filename.endswith('.txt'):
            # Only move files, not subdirectories
            source_path = os.path.join(logs_dir, filename)
            if os.path.isfile(source_path):
                dest_path = os.path.join(archive_folder, filename)
                try:
                    shutil.move(source_path, dest_path)
                    logger.info(f"def copy_log_files(): Moved {filename} to {archive_folder}")
                    log_files_moved += 1
                except Exception as e:
                    logger.error(f"def copy_log_files(): Failed to move {filename}: {e}")
    
    logger.info(f"def copy_log_files(): Successfully moved {log_files_moved} log files to {archive_folder}")
    return True

# ------------------------
# TASK: COPY, DROP, AND CREATE EVENTS TABLE
# ------------------------
@task
def copy_drop_create_events():
    # Use the centralized database configuration
    sys.path.insert(0, 'src')
    from db_config import get_database_config
    db_conn_str, env_name = get_database_config()
    logger.info(f"def copy_drop_create_events(): Using database: {env_name}")
    
    # Compose the multi-statement SQL command.
    # First, check if events_history table exists
    check_table_exists_sql = """
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema='public' AND table_name='events_history'
    """

    check_command = f'psql -d "{db_conn_str}" -t -c "{check_table_exists_sql}"'
    try:
        result = subprocess.run(check_command, shell=True, check=True, capture_output=True, text=True)
        table_exists = int(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        table_exists = False

    # Only migrate if table exists
    if table_exists:
        # Check if migration is needed (original_event_id column missing)
        check_migration_sql = """
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name='events_history' AND column_name='original_event_id'
        """

        check_command = f'psql -d "{db_conn_str}" -t -c "{check_migration_sql}"'
        try:
            result = subprocess.run(check_command, shell=True, check=True, capture_output=True, text=True)
            needs_migration = int(result.stdout.strip()) == 0
        except subprocess.CalledProcessError:
            needs_migration = True

        if needs_migration:
            logger.info("def copy_drop_create_events(): Migrating events_history table schema...")
            migration_sql = (
                "BEGIN; "
                "CREATE TABLE events_history_new ("
                    "event_id SERIAL PRIMARY KEY, "
                    "original_event_id INTEGER, "
                    "event_name TEXT, "
                    "dance_style TEXT, "
                    "description TEXT, "
                    "day_of_week TEXT, "
                    "start_date DATE, "
                    "end_date DATE, "
                    "start_time TIME, "
                    "end_time TIME, "
                    "source TEXT, "
                    "location TEXT, "
                    "price TEXT, "
                    "url TEXT, "
                    "event_type TEXT, "
                    "address_id INTEGER, "
                    "time_stamp TIMESTAMP"
                "); "
                "INSERT INTO events_history_new (original_event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp) "
                "SELECT event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp FROM events_history; "
                "DROP TABLE events_history; "
                "ALTER TABLE events_history_new RENAME TO events_history; "
                "COMMIT;"
            )
            migration_command = f'psql -d "{db_conn_str}" -c "{migration_sql}"'
            try:
                result = subprocess.run(migration_command, shell=True, check=True, capture_output=True, text=True)
                logger.info(f"def copy_drop_create_events(): Schema migration completed: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"def copy_drop_create_events(): Schema migration failed: {e.stderr}")
                raise e
        else:
            logger.info("def copy_drop_create_events(): events_history table already has correct schema, skipping migration")
    else:
        logger.info("def copy_drop_create_events(): events_history table does not exist, will be created fresh")

    # Check if events table exists before trying to copy it
    check_events_exists_sql = """
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema='public' AND table_name='events'
    """

    check_command = f'psql -d "{db_conn_str}" -t -c "{check_events_exists_sql}"'
    try:
        result = subprocess.run(check_command, shell=True, check=True, capture_output=True, text=True)
        events_table_exists = int(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        events_table_exists = False

    # Build SQL based on whether events table exists
    if events_table_exists:
        # Events table exists - copy to history then recreate
        logger.info("def copy_drop_create_events(): Events table exists, copying to events_history")
        sql = (
            "BEGIN; "
            "CREATE TABLE IF NOT EXISTS events_history ("
                "event_id SERIAL PRIMARY KEY, "
                "original_event_id INTEGER, "
                "event_name TEXT, "
                "dance_style TEXT, "
                "description TEXT, "
                "day_of_week TEXT, "
                "start_date DATE, "
                "end_date DATE, "
                "start_time TIME, "
                "end_time TIME, "
                "source TEXT, "
                "location TEXT, "
                "price TEXT, "
                "url TEXT, "
                "event_type TEXT, "
                "address_id INTEGER, "
                "time_stamp TIMESTAMP"
            "); "
            "INSERT INTO events_history (original_event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp) "
            "SELECT event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp FROM events; "
            "DROP TABLE IF EXISTS events; "
            "CREATE TABLE IF NOT EXISTS events ("
                "event_id SERIAL PRIMARY KEY, "
                "event_name TEXT, "
                "dance_style TEXT, "
                "description TEXT, "
                "day_of_week TEXT, "
                "start_date DATE, "
                "end_date DATE, "
                "start_time TIME, "
                "end_time TIME, "
                "source TEXT, "
                "location TEXT, "
                "price TEXT, "
                "url TEXT, "
                "event_type TEXT, "
                "address_id INTEGER, "
                "time_stamp TIMESTAMP"
            "); COMMIT;"
        )
    else:
        # Events table doesn't exist - just create both tables fresh
        logger.info("def copy_drop_create_events(): Events table does not exist, creating fresh tables")
        sql = (
            "BEGIN; "
            "CREATE TABLE IF NOT EXISTS events_history ("
                "event_id SERIAL PRIMARY KEY, "
                "original_event_id INTEGER, "
                "event_name TEXT, "
                "dance_style TEXT, "
                "description TEXT, "
                "day_of_week TEXT, "
                "start_date DATE, "
                "end_date DATE, "
                "start_time TIME, "
                "end_time TIME, "
                "source TEXT, "
                "location TEXT, "
                "price TEXT, "
                "url TEXT, "
                "event_type TEXT, "
                "address_id INTEGER, "
                "time_stamp TIMESTAMP"
            "); "
            "CREATE TABLE IF NOT EXISTS events ("
                "event_id SERIAL PRIMARY KEY, "
                "event_name TEXT, "
                "dance_style TEXT, "
                "description TEXT, "
                "day_of_week TEXT, "
                "start_date DATE, "
                "end_date DATE, "
                "start_time TIME, "
                "end_time TIME, "
                "source TEXT, "
                "location TEXT, "
                "price TEXT, "
                "url TEXT, "
                "event_type TEXT, "
                "address_id INTEGER, "
                "time_stamp TIMESTAMP"
            "); COMMIT;"
        )
    command = f'psql -d "{db_conn_str}" -c "{sql}"'
    logger.info(f"def copy_drop_create_events(): Running SQL command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"def copy_drop_create_events(): SQL command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"def copy_drop_create_events(): SQL command failed: {e.stderr}")
        raise e
    return True

# ------------------------
# TASK: SYNC ADDRESS SEQUENCE
# ------------------------
@task
def sync_address_sequence():
    """Synchronizes the address sequence with the current maximum address_id to prevent unique constraint violations."""
    # Use the centralized database configuration
    sys.path.insert(0, 'src')
    from db_config import get_database_config
    db_conn_str, env_name = get_database_config()
    logger.info(f"def sync_address_sequence(): Using database: {env_name}")
    
    # SQL to sync the sequence with current maximum address_id
    # First create address table if it doesn't exist, then sync sequence
    sql = (
        "CREATE TABLE IF NOT EXISTS address ("
            "address_id SERIAL PRIMARY KEY, "
            "full_address TEXT UNIQUE, "
            "building_name TEXT, "
            "street_number TEXT, "
            "street_name TEXT, "
            "street_type TEXT, "
            "direction TEXT, "
            "city TEXT, "
            "met_area TEXT, "
            "province_or_state TEXT, "
            "postal_code TEXT, "
            "country_id TEXT, "
            "time_stamp TIMESTAMP"
        "); "
        "SELECT setval('address_address_id_seq', COALESCE((SELECT MAX(address_id) FROM address), 0) + 1, false);"
    )
    command = f'psql -d "{db_conn_str}" -c "{sql}"'
    logger.info(f"def sync_address_sequence(): Syncing address sequence with command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"def sync_address_sequence(): Address sequence synced successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"def sync_address_sequence(): Address sequence sync failed: {e.stderr}")
        raise e
    return True

# ------------------------
# TASKS FOR GS.PY STEP
# ------------------------
@task
def pre_process_gs():
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    file_path = current_config['input']['data_keywords']
    if os.path.exists(file_path):
        logger.info(f"def pre_process_gs(): gs step: keywords file {file_path} exists.")
        return True
    else:
        logger.error(f"def pre_process_gs(): gs step: keywords file {file_path} does not exist.")
        return False

@task
def run_gs_script():
    try:
        result = subprocess.run([sys.executable, "src/gs.py"], check=True)
        logger.info("def run_gs_script(): gs.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"gs.py failed with return code: {e.returncode}"
        logger.error(f"def run_gs_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_gs():
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    file_path = current_config['input']['gs_urls']
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        logger.info(f"def post_process_gs(): gs step: File {file_path} exists with size {size} bytes.")
        if size > 1024:
            logger.info("def post_process_gs(): gs step: File size check passed.")
            return True
        else:
            logger.error("def post_process_gs(): gs step: File size is below 1KB.")
            return False
    else:
        logger.error("def post_process_gs(): gs step: gs_search_results file does not exist.")
        return False

@flow(name="GS Step")
def gs_step():
    original_config = backup_and_update_config("gs", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("gs", original_config)
    if not pre_process_gs():
        send_text_message("gs.py pre-processing failed: keywords file missing.")
        restore_config(original_config, "gs")
        raise Exception("gs.py pre-processing failed. Pipeline stopped.")
    run_gs_script()
    gs_ok = post_process_gs()
    if not gs_ok:
        send_text_message("gs.py post-processing failed: gs_search_results file missing or too small.")
        restore_config(original_config, "gs")
        raise Exception("gs.py post-processing failed. Pipeline stopped.")
    restore_config(original_config, "gs")
    return True

# ------------------------
# TASKS FOR EBS.PY STEP
# ------------------------
@task
def pre_process_ebs():
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    file_path = current_config['input']['data_keywords']
    if os.path.exists(file_path):
        logger.info(f"def pre_process_ebs(): ebs step: keywords file {file_path} exists.")
        return True
    else:
        logger.error(f"def pre_process_ebs(): ebs step: keywords file {file_path} does not exist.")
        return False

@task
def run_ebs_script():
    try:
        result = subprocess.run([sys.executable, "src/ebs.py"], check=True)
        logger.info("def run_ebs_script(): ebs.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"ebs.py failed with return code: {e.returncode}"
        logger.error(f"def run_ebs_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_ebs():
    return True

@flow(name="EBS Step")
def ebs_step():
    ebs_updates = copy.deepcopy(COMMON_CONFIG_UPDATES)
    ebs_updates["crawling"]["urls_run_limit"] = 250
    original_config = backup_and_update_config("ebs", updates=ebs_updates)
    write_run_config.submit("ebs", original_config)
    if not pre_process_ebs():
        send_text_message("ebs.py pre-processing failed: keywords file missing.")
        restore_config(original_config, "ebs")
        raise Exception("ebs.py pre-processing failed. Pipeline stopped.")
    run_ebs_script()
    post_process_ebs()
    restore_config(original_config, "ebs")
    return True

# ------------------------
# TASKS FOR EMAILS.PY STEP
# ------------------------
@task
def pre_process_emails():
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    file_path = current_config['input']['emails']
    if os.path.exists(file_path):
        logger.info(f"def pre_process_emails(): emails step: Emails file {file_path} exists.")
        return True
    else:
        logger.error(f"def pre_process_emails(): emails step: Emails file {file_path} does not exist.")
        return False

@task
def run_emails_script():
    try:
        result = subprocess.run([sys.executable, "src/emails.py"], check=True)
        logger.info("def run_emails_script(): emails.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"emails.py failed with return code: {e.returncode}"
        logger.error(f"def run_emails_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_emails():
    return True

@flow(name="Emails Step")
def emails_step():
    original_config = backup_and_update_config("emails", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("emails", original_config)
    if not pre_process_emails():
        send_text_message("emails.py pre-processing failed: emails file missing.")
        restore_config(original_config, "emails")
        raise Exception("emails.py pre-processing failed. Pipeline stopped.")
    run_emails_script()
    post_process_emails()
    restore_config(original_config, "emails")
    return True

# ------------------------
# TASKS FOR RD_EXT.PY STEP
# ------------------------
@task
def pre_process_rd_ext():
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    file_path = current_config['input']['edge_cases']
    if os.path.exists(file_path):
        logger.info(f"def pre_process_rd_ext(): rd_ext step: edge_cases file {file_path} exists.")
        return True
    else:
        logger.error(f"def pre_process_rd_ext(): rd_ext step: edge_cases file {file_path} does not exist.")
        return False

@task
def run_rd_ext_script():
    try:
        result = subprocess.run([sys.executable, "src/rd_ext.py"], check=True)
        logger.info("def run_rd_ext_script(): rd_ext.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"rd_ext.py failed with return code: {e.returncode}"
        logger.error(f"def run_rd_ext_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_rd_ext():
    return True

@flow(name="RD_EXT Step")
def rd_ext_step():
    original_config = backup_and_update_config("rd_ext", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("rd_ext", original_config)
    if not pre_process_rd_ext():
        send_text_message("rd_ext.py pre-processing failed: edge_cases file missing.")
        restore_config(original_config, "rd_ext")
        raise Exception("rd_ext.py pre-processing failed. Pipeline stopped.")
    run_rd_ext_script()
    post_process_rd_ext()
    restore_config(original_config, "rd_ext")
    return True

# ------------------------
# TASKS FOR SCRAPER.PY STEP
# ------------------------
@task
def pre_process_scraper():
    logger.info("def pre_process_scraper(): scraper step: Pre-processing complete with crawling.headless = True.")
    return True

@task
def run_scraper_script():
    try:
        result = subprocess.run([sys.executable, "src/scraper.py"], check=True)
        logger.info("def run_scraper_script(): scraper.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"scraper.py failed with return code: {e.returncode}"
        logger.error(f"def run_scraper_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_scraper():
    return True

@flow(name="Scraper Step")
def scraper_step():
    scraper_updates = copy.deepcopy(COMMON_CONFIG_UPDATES)
    scraper_updates["crawling"]["urls_run_limit"] = 900
    scraper_updates["crawling"]["scraper_download_timeout_seconds"] = 35
    scraper_updates["crawling"]["scraper_playwright_timeout_ms"] = 35000
    scraper_updates["crawling"]["scraper_retry_times"] = 1
    scraper_updates["crawling"]["scraper_post_load_wait_ms"] = 1000
    scraper_updates["crawling"]["scraper_concurrent_requests"] = 16
    scraper_updates["crawling"]["scraper_concurrent_requests_per_domain"] = 8
    original_config = backup_and_update_config("scraper", updates=scraper_updates)
    write_run_config.submit("scraper", original_config)
    if not pre_process_scraper():
        send_text_message("scraper.py pre-processing failed.")
        restore_config(original_config, "scraper")
        raise Exception("scraper.py pre-processing failed. Pipeline stopped.")
    run_scraper_script()
    post_process_scraper()
    restore_config(original_config, "scraper")
    return True

# ------------------------
# TASKS FOR FB.PY STEP
# ------------------------
@task
def run_fb_script():
    try:
        result = subprocess.run([sys.executable, "src/fb.py"], check=True)
        logger.info("def run_fb_script(): fb.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"fb.py failed with return code: {e.returncode}"
        logger.error(f"def run_fb_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_fb():
    return True

@flow(name="FB Step")
def fb_step():
    fb_updates = copy.deepcopy(COMMON_CONFIG_UPDATES)
    fb_updates["crawling"]["urls_run_limit"] = 500
    fb_updates["crawling"]["fb_base_urls_limit"] = 180
    fb_updates["crawling"]["fb_event_links_per_base_limit"] = 20
    fb_updates["crawling"]["fb_post_nav_wait_ms"] = 1800
    fb_updates["crawling"]["fb_post_expand_wait_ms"] = 900
    fb_updates["crawling"]["fb_final_wait_ms"] = 700
    fb_updates["crawling"]["fb_block_failures_before_cooldown"] = 2
    fb_updates["crawling"]["fb_block_cooldown_base_seconds"] = 300
    fb_updates["crawling"]["fb_block_cooldown_max_seconds"] = 1800
    fb_updates["crawling"]["fb_block_state_max_scopes"] = 800
    fb_updates["crawling"]["fb_block_state_ttl_days"] = 45
    original_config = backup_and_update_config("fb", updates=fb_updates)
    write_run_config.submit("fb", original_config)
    run_fb_script()
    post_process_fb()
    restore_config(original_config, "fb")
    return True


@task
def run_parallel_crawlers_script(script_path: str) -> str:
    """
    Execute a crawler script in a subprocess.
    Returns a success marker string or raises on failure.
    """
    logger.info(f"run_parallel_crawlers_script(): starting {script_path}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        logger.info(f"run_parallel_crawlers_script(): completed {script_path}")
        return f"{script_path}:ok"
    except subprocess.CalledProcessError as e:
        error_message = f"{script_path} failed with return code: {e.returncode}"
        logger.error(f"run_parallel_crawlers_script(): {error_message}")
        raise Exception(error_message)


@flow(name="Parallel Crawlers Step")
def parallel_crawlers_step():
    """
    Run ebs.py, scraper.py, and fb.py concurrently using one shared config snapshot.
    This avoids config race conditions from each individual step wrapper.
    """
    scripts = ["src/ebs.py", "src/scraper.py", "src/fb.py"]
    original_config = backup_and_update_config("parallel_crawlers", updates=PARALLEL_CRAWL_CONFIG_UPDATES)
    write_run_config.submit("parallel_crawlers", original_config)

    failures: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_script = {
                executor.submit(run_parallel_crawlers_script.fn, script): script
                for script in scripts
            }
            for future in as_completed(future_to_script):
                script = future_to_script[future]
                try:
                    _ = future.result()
                except Exception as e:
                    failures.append(f"{script}: {e}")

        if failures:
            raise Exception("parallel_crawlers_step failures: " + "; ".join(failures))
        return True
    finally:
        restore_config(original_config, "parallel_crawlers")

# ------------------------
# TASKS FOR IMAGES.PY STEP
# ------------------------
@task
def run_images_script():
    try:
        result = subprocess.run(
            [sys.executable, "src/images.py"],
            check=True
        )
        logger.info("def run_images_script(): images.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"images.py failed with return code: {e.returncode}"
        logger.error(f"def run_images_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_images():
    return True

@flow(name="Images Step")
def images_step():
    original_config = backup_and_update_config("images", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("images", original_config)
    run_images_script()
    post_process_images()
    restore_config(original_config, "images")
    return True

# ------------------------
# NEW TASKS FOR READ_PDFS.PY STEP
# ------------------------
@task
def run_read_pdfs_script():
    try:
        result = subprocess.run(
            [sys.executable, "src/read_pdfs.py"],
            check=True
        )
        logger.info("def run_read_pdfs_script(): read_pdfs.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"read_pdfs.py failed with return code: {e.returncode}"
        logger.error(f"def run_read_pdfs_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_read_pdfs():
    return True

@flow(name="Read PDFs Step")
def read_pdfs_step():
    original_config = backup_and_update_config("read_pdfs", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("read_pdfs", original_config)
    run_read_pdfs_script()
    post_process_read_pdfs()
    restore_config(original_config, "read_pdfs")
    return True

# ------------------------
# TASKS FOR DB.PY STEP
# ------------------------
@task
def run_db_script():
    try:
        result = subprocess.run([sys.executable, "src/db.py"], check=True)
        logger.info("def run_db_script(): db.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"db.py failed with return code: {e.returncode}"
        logger.error(f"def run_db_script(): {error_message}")
        raise Exception(error_message)

@flow(name="DB Step")
def db_step():
    original_config = backup_and_update_config("db", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("db", original_config)
    if not dummy_pre_process("db"):
        send_text_message("db.py pre-processing failed.")
        restore_config(original_config, "db")
        raise Exception("db.py pre-processing failed. Pipeline stopped.")
    run_db_script()
    dummy_post_process("db")
    restore_config(original_config, "db")
    # After the first run, if drop_tables is True, update the config file to set it to False.
    with open(CONFIG_PATH, "r") as f:
        updated_config = yaml.safe_load(f)
    if updated_config['testing'].get('drop_tables', False):
        updated_config['testing']['drop_tables'] = False
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(updated_config, f)
        logger.info("db_step: Updated config['testing']['drop_tables'] to False after first run.")
    return True

# ------------------------
# TASK FOR DATABASE BACKUP STEP (Using .dump)
# ------------------------
@task
def backup_db_step():
    # Use the centralized database configuration
    sys.path.insert(0, 'src')
    from db_config import get_database_config
    db_conn_str, env_name = get_database_config()
    logger.info(f"def backup_db_step(): Using database: {env_name}")

    # Parse connection string to extract components
    # Expected format: postgresql://USER:PASS@HOST:PORT/DATABASE
    import re
    match = re.match(r'postgresql://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)', db_conn_str)
    if not match:
        logger.error(f"def backup_db_step(): Invalid database connection string format")
        raise Exception("Invalid database connection string")

    user, password, host, port, dbname = match.groups()
    port = port or "5432"

    backup_cmd = f"pg_dump -U {user} -h {host} -p {port} -F c -b -v -f 'backups/checkpoint.dump' {dbname}"

    env = os.environ.copy()
    env["PGPASSWORD"] = password
    logger.info(f"def backup_db_step(): Backing up database with command: {backup_cmd}")
    try:
        result_backup = subprocess.run(backup_cmd, shell=True, check=True, capture_output=True, text=True, env=env)
        logger.info(f"def backup_db_step(): Database backup completed: {result_backup.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"def backup_db_step(): Database backup failed: {e.stderr}")
        raise e
    return True

# ------------------------
# TASKS FOR CLEAN_UP.PY STEP
# ------------------------
@task
def run_clean_up_script():
    try:
        result = subprocess.run([sys.executable, "src/clean_up.py"], check=True)
        logger.info("def run_clean_up_script(): clean_up.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"clean_up.py failed with return code: {e.returncode}"
        logger.error(f"def run_clean_up_script(): {error_message}")
        raise Exception(error_message)

@flow(name="Clean Up Step")
def clean_up_step():
    original_config = backup_and_update_config("clean_up", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("clean_up", original_config)
    if not dummy_pre_process("clean_up"):
        send_text_message("clean_up.py pre-processing failed.")
        restore_config(original_config, "clean_up")
        raise Exception("clean_up.py pre-processing failed. Pipeline stopped.")
    run_clean_up_script()
    dummy_post_process("clean_up")
    restore_config(original_config, "clean_up")
    return True

# ------------------------
# TASKS FOR DEDUP_LLM.PY STEP
# ------------------------
@task
def run_dedup_llm_script():
    try:
        result = subprocess.run([sys.executable, "src/dedup_llm.py"], check=True)
        logger.info("def run_dedup_llm_script(): dedup_llm.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"dedup_llm.py failed with return code: {e.returncode}"
        logger.error(f"def run_dedup_llm_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_dedup_llm() -> bool:
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    output_file = current_config['output']['dedup']
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        logger.error(f"def post_process_dedup_llm(): Could not read CSV at {output_file}: {e}")
        return False
    if 'Label' not in df.columns:
        logger.error("def post_process_dedup_llm(): 'Label' column not found in output CSV.")
        return False

    # Check for NaN values in Label column
    nan_count = df['Label'].isna().sum()
    if nan_count > 0:
        logger.warning(f"def post_process_dedup_llm(): Found {nan_count} NaN values in Label column")
        nan_rows = df[df['Label'].isna()][['event_id', 'event_name', 'Label']]
        logger.warning(f"def post_process_dedup_llm(): Rows with NaN Labels:\n{nan_rows.to_string()}")
        # Fill NaN with 0 (treat as unique/not duplicate)
        df['Label'] = df['Label'].fillna(0)
        df.to_csv(output_file, index=False)
        logger.info(f"def post_process_dedup_llm(): Filled {nan_count} NaN values with 0 and saved to {output_file}")

    # Check if the deduplication process completed successfully
    # The presence of both 0s and 1s in Label column is expected (0=unique, 1=duplicate)
    total_rows = len(df)
    duplicates_found = (df['Label'] == 1).sum()
    unique_events = (df['Label'] == 0).sum()

    logger.info(f"def post_process_dedup_llm(): Processed {total_rows} events: {unique_events} unique, {duplicates_found} duplicates found")

    # Success criteria: we have data and Label column contains valid values (0 or 1)
    if total_rows > 0 and df['Label'].isin([0, 1]).all():
        logger.info("def post_process_dedup_llm(): Deduplication completed successfully.")
        return True
    else:
        # Report which values are invalid
        invalid_values = df[~df['Label'].isin([0, 1])]['Label'].unique()
        logger.error(f"def post_process_dedup_llm(): Invalid Label values found: {invalid_values}")
        return False

@flow(name="Dedup LLM Step")
def dedup_llm_step():
    original_config = backup_and_update_config("dedup_llm", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("dedup_llm", original_config)
    run_dedup_llm_script()
    success = post_process_dedup_llm()
    if not success:
        send_text_message("dedup_llm.py post-processing failed: 'Label' column not all 0 after run.")
        restore_config(original_config, "dedup_llm")
        raise Exception("dedup_llm.py post-processing failed. Pipeline stopped.")
    restore_config(original_config, "dedup_llm")
    return True

# ------------------------
# TASKS FOR IRRELEVANT_ROWS.PY STEP
# ------------------------
@task
def pre_process_irrelevant_rows():
    logger.info("def pre_process_irrelevant_rows(): irrelevant_rows step: No special pre-processing required.")
    return True

@task
def run_irrelevant_rows_script():
    try:
        result = subprocess.run([sys.executable, "src/irrelevant_rows.py"], check=True)
        logger.info("def run_irrelevant_rows_script(): irrelevant_rows.py executed successfully.")
        return "Script completed successfully"
    except subprocess.CalledProcessError as e:
        error_message = f"irrelevant_rows.py failed with return code: {e.returncode}"
        logger.error(f"def run_irrelevant_rows_script(): {error_message}")
        raise Exception(error_message)

@task
def post_process_irrelevant_rows():
    return True

@flow(name="Irrelevant Rows Step")
def irrelevant_rows_step():
    original_config = backup_and_update_config("irrelevant_rows", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("irrelevant_rows", original_config)
    if not pre_process_irrelevant_rows():
        send_text_message("irrelevant_rows.py pre-processing failed.")
        restore_config(original_config, "irrelevant_rows")
        raise Exception("irrelevant_rows.py pre-processing failed. Pipeline stopped.")
    run_irrelevant_rows_script()
    post_process_irrelevant_rows()
    restore_config(original_config, "irrelevant_rows")
    return True

# ------------------------
# VALIDATION STEP
# Pre-commit validation: scraping health + chatbot quality checks
# ------------------------
@task
def run_validation_tests():
    """Run pre-commit validation tests (scraping + chatbot)."""
    try:
        # Note: No check=True - don't raise on failure (non-blocking)
        result = subprocess.run(
            [sys.executable, "tests/validation/test_runner.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min (LLM scoring takes time)
        )

        if result.returncode == 0:
            logger.info("def run_validation_tests(): Validation completed successfully")
            logger.info(result.stdout)
            return "Validation passed"
        else:
            logger.warning(f"def run_validation_tests(): Validation completed with issues (exit code {result.returncode})")
            logger.warning(result.stderr)
            logger.warning("def run_validation_tests(): Continuing pipeline despite validation issues")
            return "Validation completed with warnings"

    except subprocess.TimeoutExpired:
        logger.error("def run_validation_tests(): Validation tests timed out after 30 minutes")
        logger.warning("def run_validation_tests(): Continuing pipeline despite timeout")
        return "Validation timed out"
    except Exception as e:
        logger.error(f"def run_validation_tests(): Unexpected error: {e}")
        logger.warning("def run_validation_tests(): Continuing pipeline despite error")
        return "Validation encountered error"

@flow(name="Validation Step")
def validation_step():
    """
    Pre-commit validation: scraping health + chatbot quality checks.

    NOTE: This step does NOT use config backup/restore or COMMON_CONFIG_UPDATES
    because it only reads config and doesn't modify it.
    """
    logger.info("=" * 70)
    logger.info("VALIDATION STEP")
    logger.info("Pre-commit validation: scraping health + chatbot quality")
    logger.info("=" * 70)

    # Run validation tests (non-blocking - won't stop pipeline on failure)
    validation_result = run_validation_tests()
    logger.info(f"validation_step: run_validation_tests returned: {validation_result}")

    logger.info("validation_step: Step completed")
    return True

# ------------------------
# RESULT ANALYZER STEP
# Analyzes validation test results using LLM to identify patterns and priorities
# ------------------------
@task
def run_result_analyzer():
    """Run LLM-based analysis of validation test results."""
    try:
        # Note: No check=True - don't raise on failure (non-blocking)
        result = subprocess.run(
            [sys.executable, "tests/validation/result_analyzer.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 min (LLM API call for analysis)
        )

        if result.returncode == 0:
            logger.info("def run_result_analyzer(): Result analysis completed successfully")
            # Log the analysis summary (last 50 lines which contain the report)
            output_lines = result.stdout.strip().split('\n')
            summary_start = -1
            for i, line in enumerate(output_lines):
                if "CHATBOT TEST RESULTS ANALYSIS" in line:
                    summary_start = i
                    break
            if summary_start >= 0:
                summary = '\n'.join(output_lines[summary_start:])
                logger.info(f"Analysis summary:\n{summary}")
            return "Analysis completed successfully"
        else:
            logger.warning(f"def run_result_analyzer(): Analysis completed with issues (exit code {result.returncode})")
            logger.warning(result.stderr)
            logger.warning("def run_result_analyzer(): Continuing pipeline despite analysis issues")
            return "Analysis completed with warnings"

    except subprocess.TimeoutExpired:
        logger.error("def run_result_analyzer(): Result analysis timed out after 10 minutes")
        logger.warning("def run_result_analyzer(): Continuing pipeline despite timeout")
        return "Analysis timed out"
    except Exception as e:
        logger.error(f"def run_result_analyzer(): Unexpected error: {e}")
        logger.warning("def run_result_analyzer(): Continuing pipeline despite error")
        return "Analysis encountered error"

@flow(name="Result Analyzer Step")
def result_analyzer_step():
    """
    Analyze validation test results using LLM to identify patterns and priorities.

    NOTE: This step does NOT use config backup/restore or COMMON_CONFIG_UPDATES
    because it only reads test results and doesn't modify config.
    """
    logger.info("=" * 70)
    logger.info("RESULT ANALYZER STEP")
    logger.info("LLM-based analysis of validation test results")
    logger.info("=" * 70)

    # Run result analyzer (non-blocking - won't stop pipeline on failure)
    analyzer_result = run_result_analyzer()
    logger.info(f"result_analyzer_step: run_result_analyzer returned: {analyzer_result}")

    logger.info("result_analyzer_step: Step completed")
    return True

# ------------------------
# COPY DEV DATABASE TO PRODUCTION DATABASE STEP
# This step copies the working database (local or render_dev) to production
# ------------------------
def run_command_with_retry(command, logger, attempts=3, delay=5, env=None, timeout=30):
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env, timeout=timeout)
            return result
        except subprocess.CalledProcessError as e:
            stderr_lower = e.stderr.lower() if e.stderr else ""
            if "database is locked" in stderr_lower:
                logger.warning(f"def run_command_with_retry(): Attempt {attempt} failed due to database lock. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"def run_command_with_retry(): Command failed on attempt {attempt}: {e.stderr}")
                raise e
        except subprocess.TimeoutExpired as te:
            logger.error(f"def run_command_with_retry(): Command timed out on attempt {attempt}: {te}")
            raise te
    raise Exception(f"Command failed after {attempts} attempts due to persistent database lock errors.")

@flow(name="Copy Dev to Prod Step")
def copy_dev_db_to_prod_db_step():
    """
    Copy only the required tables from development to production database.

    Tables copied:
    - events: Main event data
    - conversations: User conversation tracking
    - conversation_messages: Conversation history

    Tables NOT copied (only used during pipeline):
    - address, raw_locations, events_history, auth_storage

    Logic:
    - SOURCE: Uses current DATABASE_TARGET setting (local or render_dev)
    - TARGET: ALWAYS Render Production (render_prod)
    - If DATABASE_TARGET=render_prod, skips (source and target are the same)

    This means you never need to change DATABASE_TARGET - just set it to where
    you're working (local or render_dev) and this step will copy to production.
    """
    from db_config import get_database_config, get_production_database_url
    from urllib.parse import urlparse

    logger.info("def copy_dev_db_to_prod_db_step(): Starting table copy to production.")

    # Get source database based on current DATABASE_TARGET
    source_conn_str, source_env_name = get_database_config()
    parsed_source = urlparse(source_conn_str)

    # Get production database (always the target)
    prod_conn_str = get_production_database_url()
    parsed_prod = urlparse(prod_conn_str)

    # Check if source and target are the same
    if parsed_source.hostname == parsed_prod.hostname and parsed_source.path == parsed_prod.path:
        logger.info(f"def copy_dev_db_to_prod_db_step(): Source and target are the same (both production). Skipping copy.")
        logger.info(f"def copy_dev_db_to_prod_db_step(): Source: {source_env_name}")
        return True

    logger.info(f"def copy_dev_db_to_prod_db_step(): Source: {source_env_name} ({parsed_source.hostname}/{parsed_source.path[1:]})")
    logger.info(f"def copy_dev_db_to_prod_db_step(): Target: Render Production Database ({parsed_prod.hostname}/{parsed_prod.path[1:]})")

    # Auto-detect PostgreSQL version and use matching tools
    env_source = os.environ.copy()
    env_source["PGPASSWORD"] = parsed_source.password

    version_command = (
        f"psql -h {parsed_source.hostname} -U {parsed_source.username} "
        f"-d {parsed_source.path[1:]} -t -c 'SHOW server_version_num;'"
    )

    try:
        version_result = subprocess.run(version_command, shell=True, check=True, capture_output=True, text=True, env=env_source, timeout=10)
        server_version_num = int(version_result.stdout.strip())
        server_version = server_version_num // 10000  # 170006 -> 17
        logger.info(f"def copy_dev_db_to_prod_db_step(): Detected PostgreSQL version {server_version}")

        # Use version-specific tools
        pg_dump_path = f"/usr/lib/postgresql/{server_version}/bin/pg_dump"
        pg_restore_path = f"/usr/lib/postgresql/{server_version}/bin/pg_restore"

        # Verify tools exist
        if not os.path.exists(pg_dump_path):
            logger.warning(f"def copy_dev_db_to_prod_db_step(): Version-specific pg_dump not found at {pg_dump_path}, falling back to system pg_dump")
            pg_dump_path = "pg_dump"
            pg_restore_path = "pg_restore"
    except Exception as e:
        logger.warning(f"def copy_dev_db_to_prod_db_step(): Could not detect PostgreSQL version: {e}. Using system pg_dump/pg_restore")
        pg_dump_path = "pg_dump"
        pg_restore_path = "pg_restore"

    # Tables to copy to production (only what's needed for web service)
    REQUIRED_TABLES = ['events', 'conversations', 'conversation_messages']

    # Step 1: Dump only required tables from source database
    dump_file = 'backups/dev_to_prod_backup.dump'
    os.makedirs('backups', exist_ok=True)

    # Build table list for pg_dump
    table_args = ' '.join([f"-t {table}" for table in REQUIRED_TABLES])

    dump_command = (
        f"{pg_dump_path} -h {parsed_source.hostname} -U {parsed_source.username} "
        f"-d {parsed_source.path[1:]} {table_args} -F c -b -v -f '{dump_file}'"
    )

    # Get source row counts before dump
    source_counts = {}
    for table in REQUIRED_TABLES:
        count_command = f"psql -h {parsed_source.hostname} -U {parsed_source.username} -d {parsed_source.path[1:]} -t -c 'SELECT COUNT(*) FROM {table};'"
        try:
            result = subprocess.run(count_command, shell=True, check=True, capture_output=True, text=True, env=env_source, timeout=10)
            source_counts[table] = int(result.stdout.strip())
            logger.info(f"def copy_dev_db_to_prod_db_step(): Source {table} count: {source_counts[table]}")
        except Exception as e:
            logger.error(f"def copy_dev_db_to_prod_db_step(): Failed to get source count for {table}: {e}")
            raise e

    logger.info(f"def copy_dev_db_to_prod_db_step(): Dumping tables {REQUIRED_TABLES} using {pg_dump_path}...")
    try:
        result = subprocess.run(dump_command, shell=True, check=True, capture_output=True, text=True, env=env_source, timeout=120)
        logger.info("def copy_dev_db_to_prod_db_step(): Table dump completed successfully.")
        if result.stderr:
            logger.info(f"def copy_dev_db_to_prod_db_step(): Dump stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table dump failed: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired as te:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table dump timed out: {te}")
        raise te

    # Step 2: Restore to production database
    restore_command = (
        f"{pg_restore_path} -h {parsed_prod.hostname} -U {parsed_prod.username} "
        f"-d {parsed_prod.path[1:]} --no-owner --clean --if-exists -v -c '{dump_file}'"
    )

    env_prod = os.environ.copy()
    env_prod["PGPASSWORD"] = parsed_prod.password

    logger.info(f"def copy_dev_db_to_prod_db_step(): Restoring tables to production database...")
    restore_success = False
    restore_error = None
    try:
        result = subprocess.run(restore_command, shell=True, check=True, capture_output=True, text=True, env=env_prod, timeout=120)
        logger.info("def copy_dev_db_to_prod_db_step(): Table restore completed successfully.")
        if result.stderr:
            logger.info(f"def copy_dev_db_to_prod_db_step(): Restore stderr: {result.stderr}")
        restore_success = True
    except subprocess.TimeoutExpired as te:
        restore_error = f"Table restore timed out: {te}"
        logger.error(f"def copy_dev_db_to_prod_db_step(): {restore_error}")
    except subprocess.CalledProcessError as e:
        restore_error = f"Table restore failed with return code {e.returncode}: {e.stderr}"
        logger.error(f"def copy_dev_db_to_prod_db_step(): {restore_error}")

    # Step 2.5: Validate row counts match
    if restore_success:
        logger.info("def copy_dev_db_to_prod_db_step(): Validating row counts...")
        validation_failed = False
        for table in REQUIRED_TABLES:
            count_command = f"psql -h {parsed_prod.hostname} -U {parsed_prod.username} -d {parsed_prod.path[1:]} -t -c 'SELECT COUNT(*) FROM {table};'"
            try:
                result = subprocess.run(count_command, shell=True, check=True, capture_output=True, text=True, env=env_prod, timeout=10)
                prod_count = int(result.stdout.strip())
                source_count = source_counts[table]
                logger.info(f"def copy_dev_db_to_prod_db_step(): {table} - Source: {source_count}, Production: {prod_count}")
                if prod_count != source_count:
                    validation_failed = True
                    logger.error(f"def copy_dev_db_to_prod_db_step(): ❌ ROW COUNT MISMATCH for {table}! Source: {source_count}, Production: {prod_count}")
            except Exception as e:
                validation_failed = True
                logger.error(f"def copy_dev_db_to_prod_db_step(): Failed to validate count for {table}: {e}")

        if validation_failed:
            raise Exception("Row count validation failed! Production database does not match source. Copy was unsuccessful.")
        else:
            logger.info("def copy_dev_db_to_prod_db_step(): ✓ Row count validation passed - all tables match!")
    else:
        # Restore failed - raise exception
        raise Exception(f"Database restore failed: {restore_error}")

    # Step 3: Set timezone on production
    alter_command = f"ALTER DATABASE {parsed_prod.path[1:]} SET TIME ZONE 'PST8PDT';"
    psql_command = (
        f"psql -h {parsed_prod.hostname} -U {parsed_prod.username} "
        f"-d {parsed_prod.path[1:]} -c \"{alter_command}\""
    )

    logger.info(f"def copy_dev_db_to_prod_db_step(): Setting production timezone...")
    try:
        subprocess.run(psql_command, shell=True, check=True, capture_output=True, text=True, env=env_prod, timeout=30)
        logger.info("def copy_dev_db_to_prod_db_step(): Production timezone set to PST8PDT.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"def copy_dev_db_to_prod_db_step(): Timezone setting failed (non-fatal): {e.stderr}")
    except subprocess.TimeoutExpired as te:
        logger.warning(f"def copy_dev_db_to_prod_db_step(): Timezone setting timed out (non-fatal): {te}")

    logger.info(f"def copy_dev_db_to_prod_db_step(): ✓ Table copy to production completed! Copied tables: {REQUIRED_TABLES}")
    return True

# ------------------------
# DOWNLOAD RENDER CRON LOGS STEP
# ------------------------
@flow(name="Download Render Logs Step")
def download_render_logs_step():
    """
    Download Render cron job logs using the Render API.

    This step only runs when on Render (checks RENDER environment variable).
    Requires RENDER_API_KEY to be set in environment variables.

    Saves logs to: logs/render_logs/cron_job_log_<timestamp>.txt
    """
    # Only run this step when on Render
    if not os.getenv('RENDER'):
        logger.info("def download_render_logs_step(): Not running on Render, skipping log download.")
        return True

    # Check for RENDER_API_KEY
    render_api_key = os.getenv('RENDER_API_KEY')
    if not render_api_key:
        logger.warning("def download_render_logs_step(): RENDER_API_KEY not found. Cannot download logs.")
        logger.info("def download_render_logs_step(): Set RENDER_API_KEY environment variable to enable automatic log downloads.")
        return True

    # Get the service name from environment (or use default)
    service_name = os.getenv('RENDER_SERVICE_NAME', 'social_dance_app_cron')

    # Create logs directory
    log_dir = 'logs/render_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'cron_job_log_{timestamp}.txt')

    # Download logs using Render API
    logger.info(f"def download_render_logs_step(): Downloading logs for service: {service_name}")

    try:
        import requests

        # Set up API headers
        headers = {
            'Authorization': f'Bearer {render_api_key}',
            'Accept': 'application/json'
        }

        # Step 1: Find service ID and owner ID by service name
        services_url = 'https://api.render.com/v1/services'
        response = requests.get(services_url, headers=headers, params={'limit': 100})

        if response.status_code != 200:
            logger.error(f"def download_render_logs_step(): Failed to list services: {response.status_code} - {response.text}")
            return True  # Don't fail pipeline

        services = response.json()
        service_id = None
        owner_id = None

        for service in services:
            if service['service']['name'] == service_name:
                service_id = service['service']['id']
                owner_id = service['service'].get('ownerId')
                break

        if not service_id:
            logger.error(f"def download_render_logs_step(): Service '{service_name}' not found")
            return True  # Don't fail pipeline

        logger.info(f"def download_render_logs_step(): Found service ID: {service_id}")

        # Step 2: Get logs from Render API using /v1/logs endpoint
        from datetime import datetime, timedelta

        # Get logs from the last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        # Format as RFC3339 (ISO 8601 with Z for UTC)
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        logs_url = 'https://api.render.com/v1/logs'
        log_params = {
            'resource': service_id,
            'ownerId': owner_id,
            'startTime': start_time_str,
            'endTime': end_time_str,
            'limit': 1000
        }

        logger.info(f"def download_render_logs_step(): Retrieving logs from last 24 hours...")
        log_response = requests.get(logs_url, headers=headers, params=log_params)

        if log_response.status_code != 200:
            logger.error(f"def download_render_logs_step(): Failed to get logs: {log_response.status_code} - {log_response.text}")
            return True  # Don't fail pipeline

        # Extract log text from API response
        log_data = log_response.json()

        if isinstance(log_data, list):
            logs_text = '\n'.join([entry.get('text', entry.get('message', str(entry))) for entry in log_data])
        elif isinstance(log_data, dict):
            logs_text = '\n'.join([entry.get('text', entry.get('message', str(entry)))
                                  for entry in log_data.get('logs', log_data.get('entries', []))])
        else:
            logs_text = str(log_data)

        if not logs_text:
            logger.warning("def download_render_logs_step(): No logs returned from Render API")
            return True  # Don't fail pipeline

        # Save logs to file
        with open(log_file, 'w') as f:
            f.write(logs_text)

        logger.info(f"def download_render_logs_step(): ✓ Logs downloaded successfully to {log_file}")
        logger.info(f"def download_render_logs_step(): Downloaded {len(logs_text)} characters ({len(logs_text.splitlines())} lines)")

        return True

    except Exception as e:
        logger.error(f"def download_render_logs_step(): Unexpected error downloading logs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return True  # Don't fail the pipeline on unexpected errors

# ------------------------
# STUB FOR TEXT MESSAGING
# ------------------------
@task
def send_text_message(message: str):
    logger.info(f"def send_text_message(): Sending text message: {message}")
    # TODO: Integrate with an SMS API like Twilio

# ------------------------
# PIPELINE EXECUTION
# ------------------------
PIPELINE_STEPS = [
    ("copy_log_files", copy_log_files),
    ("credential_validation", credential_validation_step),
    ("copy_drop_create_events", copy_drop_create_events),
    ("sync_address_sequence", sync_address_sequence),
    ("emails", emails_step),
    ("gs", gs_step),
    ("rd_ext", rd_ext_step),
    ("ebs", ebs_step),
    ("scraper", scraper_step),
    ("fb", fb_step),
    ("images", images_step),
    ("read_pdfs", read_pdfs_step),
    ("backup_db", backup_db_step),
    ("db", db_step),
    ("clean_up", clean_up_step),
    ("dedup_llm", dedup_llm_step),
    ("irrelevant_rows", irrelevant_rows_step),
    ("validation", validation_step),  # Pre-commit validation before prod deployment
    ("result_analyzer", result_analyzer_step),  # LLM analysis of validation results
    ("copy_dev_to_prod", copy_dev_db_to_prod_db_step),
    ("download_render_logs", download_render_logs_step)
]

def list_available_steps():
    print("Available steps:")
    for i, (name, _) in enumerate(PIPELINE_STEPS, start=1):
        print(f" {i}. {name}")

def run_pipeline(start_step: str, end_step: str = None, parallel_crawlers: bool = False):
    step_names = [name for name, _ in PIPELINE_STEPS]
    if start_step not in step_names:
        print(f"Error: start step '{start_step}' not found.")
        sys.exit(1)
    if end_step and end_step not in step_names:
        print(f"Error: end step '{end_step}' not found.")
        sys.exit(1)
    start_idx = step_names.index(start_step)
    end_idx = step_names.index(end_step) if end_step else len(PIPELINE_STEPS) - 1
    if start_idx > end_idx:
        print("Error: start step occurs after end step.")
        sys.exit(1)
    selected_names = [name for name, _ in PIPELINE_STEPS[start_idx:end_idx + 1]]
    skip_steps: set[str] = set()

    for name, step_flow in PIPELINE_STEPS[start_idx:end_idx+1]:
        if name in skip_steps:
            logger.info(f"run_pipeline(): Skipping step '{name}' because it was executed in parallel block.")
            continue

        if (
            parallel_crawlers
            and name == "ebs"
            and {"ebs", "scraper", "fb"}.issubset(set(selected_names))
        ):
            print("Running parallel crawler group: ebs + scraper + fb")
            retry_count = 0
            while retry_count < 3:
                try:
                    parallel_crawlers_step()
                    skip_steps.update({"scraper", "fb"})
                    break
                except Exception as e:
                    if "database is locked" in str(e).lower():
                        logger.error(
                            "parallel crawler group encountered database lock, retrying in 5 seconds. Attempt %d of 3.",
                            retry_count + 1,
                        )
                        time.sleep(5)
                        retry_count += 1
                    else:
                        import traceback
                        logger.error(f"❌ Parallel crawler group failed: {str(e)}")
                        logger.error(traceback.format_exc())
                        sys.exit(1)
            else:
                logger.error("Parallel crawler group failed after 3 retries due to database locked errors.")
                sys.exit(1)
            continue

        print(f"Running step: {name}")
        retry_count = 0
        while retry_count < 3:
            try:
                step_flow()
                break
            except Exception as e:
                if "database is locked" in str(e).lower():
                    logger.error(f"Step {name} encountered a database locked error, retrying in 5 seconds. Attempt {retry_count+1} of 3.")
                    time.sleep(5)
                    retry_count += 1
                else:
                    # Log error with full details for troubleshooting
                    import traceback
                    logger.error(f"❌ Step '{name}' failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    sys.exit(1)
        else:
            logger.error(f"Step {name} failed after 3 retries due to database locked errors.")
            sys.exit(1)

def prompt_user():
    print("Select pipeline execution mode:")
    print("1. Run the entire pipeline (default)")
    print("2. Run just one part")
    print("3. Start at one part and continue to the end")
    print("4. Start at one part and stop at a specified later part")
    mode = input("Enter option number (1-4): ").strip() or "1"
    list_available_steps()
    if mode == "1":
        start_index = 0
        end_index = len(PIPELINE_STEPS) - 1
    elif mode == "2":
        try:
            step_number = int(input("Enter the step number to run: ").strip())
            start_index = step_number - 1
            end_index = step_number - 1
        except ValueError:
            print("Invalid input. Running entire pipeline.")
            start_index = 0
            end_index = len(PIPELINE_STEPS) - 1
    elif mode == "3":
        try:
            step_number = int(input("Enter the starting step number: ").strip())
            start_index = step_number - 1
            end_index = len(PIPELINE_STEPS) - 1
        except ValueError:
            print("Invalid input. Running entire pipeline.")
            start_index = 0
            end_index = len(PIPELINE_STEPS) - 1
    elif mode == "4":
        try:
            start_number = int(input("Enter the starting step number: ").strip())
            end_number = int(input("Enter the ending step number: ").strip())
            start_index = start_number - 1
            end_index = end_number - 1
        except ValueError:
            print("Invalid input. Running entire pipeline.")
            start_index = 0
            end_index = len(PIPELINE_STEPS) - 1
    else:
        print("Invalid option. Running entire pipeline.")
        start_index = 0
        end_index = len(PIPELINE_STEPS) - 1
    start = PIPELINE_STEPS[start_index][0]
    end = PIPELINE_STEPS[end_index][0]
    print(f"Pipeline will run from '{start}' to '{end}'.")
    run_pipeline(start, end)

def move_temp_files_back():
    # After a successful run, move CSV files from temp back to the original URL directory.
    try:
        with open(CONFIG_PATH, "r") as f:
            current_config = yaml.safe_load(f)
        urls_dir = current_config['input']['urls']
        temp_dir = os.path.join(urls_dir, "temp")
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                if filename.endswith(".csv"):
                    src = os.path.join(temp_dir, filename)
                    dst = os.path.join(urls_dir, filename)
                    shutil.move(src, dst)
                    logger.info(f"move_temp_files_back(): Moved {filename} from temp back to {urls_dir}.")
            logger.info("move_temp_files_back(): Completed moving temp files back.")
        else:
            logger.info("move_temp_files_back(): No temp directory found; nothing to move.")
    except Exception as e:
        logger.error(f"move_temp_files_back(): Failed to move temp files back: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run pipeline with command line options or interactive input.")
    parser.add_argument('--mode', choices=['1', '2', '3', '4'],
                        help="Execution mode: 1 (entire pipeline), 2 (one step), 3 (start at a step), 4 (start and stop at specified steps).")
    parser.add_argument('--step', type=int,
                        help="Step number to run (required for mode 2).")
    parser.add_argument('--start_step', type=int,
                        help="Starting step number (required for mode 3 and 4).")
    parser.add_argument('--end_step', type=int,
                        help="Ending step number (required for mode 4).")
    parser.add_argument('--parallel_crawlers', action='store_true',
                        help="Run ebs.py, scraper.py, and fb.py in parallel when the selected range includes all three.")
    args = parser.parse_args()
    if args.mode:
        mode = args.mode
        if mode == "1":
            start_index = 0
            end_index = len(PIPELINE_STEPS) - 1
        elif mode == "2":
            if args.step is None:
                sys.exit("For mode 2, please provide --step argument.")
            start_index = args.step - 1
            end_index = args.step - 1
        elif mode == "3":
            if args.start_step is None:
                sys.exit("For mode 3, please provide --start_step argument.")
            start_index = args.start_step - 1
            end_index = len(PIPELINE_STEPS) - 1
        elif mode == "4":
            if args.start_step is None or args.end_step is None:
                sys.exit("For mode 4, please provide both --start_step and --end_step arguments.")
            start_index = args.start_step - 1
            end_index = args.end_step - 1
        else:
            print("Invalid mode. Running entire pipeline.")
            start_index = 0
            end_index = len(PIPELINE_STEPS) - 1
        start = PIPELINE_STEPS[start_index][0]
        end = PIPELINE_STEPS[end_index][0]
        print(f"Pipeline will run from '{start}' to '{end}'.")
        run_pipeline(start, end, parallel_crawlers=args.parallel_crawlers)
    else:
        prompt_user()
    
    # After successful pipeline execution, move any CSV files from temp back to the original URLs directory.
    try:
        move_temp_files_back()
    except Exception as e:
        logger.error(f"main(): Failed to move temp files back: {e}")

if __name__ == "__main__":
    main()

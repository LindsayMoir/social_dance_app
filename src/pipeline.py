# pipeline.py

import argparse
import copy
import datetime
from dotenv import load_dotenv
load_dotenv()
import logging
import os
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
    "llm": {"provider": "mistral", "spend_money": True}
}

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
    original_config = backup_and_update_config("ebs", updates=COMMON_CONFIG_UPDATES)
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
    scraper_updates["crawling"]["urls_run_limit"] = 1500
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
    original_config = backup_and_update_config("fb", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("fb", original_config)
    run_fb_script()
    post_process_fb()
    with open(CONFIG_PATH, "r") as f:
        current_config = yaml.safe_load(f)
    current_config['llm']['provider'] = 'mistral'
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(current_config, f)
    logger.info("def fb_step(): Updated config llm.provider to mistral for the remaining pipeline run.")
    return True

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
        logger.error("def post_process_dedup_llm(): Invalid Label values found in output.")
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

    Works with DATABASE_TARGET to determine source:
    - DATABASE_TARGET=local: Copies from local PostgreSQL to Render Production
    - DATABASE_TARGET=render_dev: Copies from Render Dev to Render Production
    - DATABASE_TARGET=render_prod: Skips (already on production)
    """
    from db_config import get_database_config, get_production_database_url, is_production_target
    from urllib.parse import urlparse

    logger.info("def copy_dev_db_to_prod_db_step(): Starting table copy to production.")

    # Check if already targeting production
    if is_production_target():
        logger.info("def copy_dev_db_to_prod_db_step(): DATABASE_TARGET is 'render_prod'. Skipping copy (already on production).")
        return True

    # Get source database connection details
    source_conn_str, source_env_name = get_database_config()
    parsed_source = urlparse(source_conn_str)

    logger.info(f"def copy_dev_db_to_prod_db_step(): Source: {source_env_name}")
    logger.info(f"def copy_dev_db_to_prod_db_step(): Target: Render Production Database")

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

    logger.info(f"def copy_dev_db_to_prod_db_step(): Dumping tables {REQUIRED_TABLES} using {pg_dump_path}...")
    try:
        subprocess.run(dump_command, shell=True, check=True, capture_output=True, text=True, env=env_source, timeout=120)
        logger.info("def copy_dev_db_to_prod_db_step(): Table dump completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table dump failed: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired as te:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table dump timed out: {te}")
        raise te

    # Step 2: Restore to production database
    prod_url = get_production_database_url()
    parsed_prod = urlparse(prod_url)

    restore_command = (
        f"{pg_restore_path} -h {parsed_prod.hostname} -U {parsed_prod.username} "
        f"-d {parsed_prod.path[1:]} --no-owner --clean --if-exists -v -c '{dump_file}'"
    )

    env_prod = os.environ.copy()
    env_prod["PGPASSWORD"] = parsed_prod.password

    logger.info(f"def copy_dev_db_to_prod_db_step(): Restoring tables to production database...")
    try:
        subprocess.run(restore_command, shell=True, check=True, capture_output=True, text=True, env=env_prod, timeout=120)
        logger.info("def copy_dev_db_to_prod_db_step(): Table restore completed successfully.")
    except subprocess.TimeoutExpired as te:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table restore timed out: {te}")
        # Don't raise - just log (production might be slow)
    except subprocess.CalledProcessError as e:
        logger.error(f"def copy_dev_db_to_prod_db_step(): Table restore failed: {e.stderr}")
        # Don't raise - production restore errors are common and often non-fatal

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
    ("copy_drop_create_events", copy_drop_create_events),
    ("sync_address_sequence", sync_address_sequence),
    ("emails", emails_step),
    ("gs", gs_step),
    ("ebs", ebs_step),
    ("rd_ext", rd_ext_step),
    ("scraper", scraper_step),
    ("fb", fb_step),
    ("images", images_step),
    ("read_pdfs", read_pdfs_step),
    ("backup_db", backup_db_step),
    ("db", db_step),
    ("clean_up", clean_up_step),
    ("dedup_llm", dedup_llm_step),
    ("irrelevant_rows", irrelevant_rows_step),
    ("copy_dev_to_prod", copy_dev_db_to_prod_db_step)
]

def list_available_steps():
    print("Available steps:")
    for i, (name, _) in enumerate(PIPELINE_STEPS, start=1):
        print(f" {i}. {name}")

def run_pipeline(start_step: str, end_step: str = None):
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
    for name, step_flow in PIPELINE_STEPS[start_idx:end_idx+1]:
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
                    raise e
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
        run_pipeline(start, end)
    else:
        prompt_user()
    
    # After successful pipeline execution, move any CSV files from temp back to the original URLs directory.
    try:
        move_temp_files_back()
    except Exception as e:
        logger.error(f"main(): Failed to move temp files back: {e}")

if __name__ == "__main__":
    main()

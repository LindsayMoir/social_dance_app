from dotenv import load_dotenv
load_dotenv()
import os
import subprocess
import sys
import yaml
import datetime
import copy
import pandas as pd
import argparse
import time
import shutil

# Global logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"

# Define common configuration updates for all pipeline steps
COMMON_CONFIG_UPDATES = {
    "testing": {"drop_tables": False},
    "crawling": {
         "headless": True,
         "max_website_urls": 10,
         "urls_run_limit": 500,  # default for all steps
    },
    "llm": {"provider": "openai", "spend_money": True}
}

# ------------------------
# HELPER TASKS: Backup and Restore Config
# ------------------------
from prefect import flow, task

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
# TASK FOR EVENTS TABLE BACKUP AND DROP STEP
# ------------------------
@task
def events_table_backup_and_drop():
    db_conn_str = os.getenv("DATABASE_CONNECTION_STRING")
    if not db_conn_str:
        logger.error("def events_table_backup_and_drop(): DATABASE_CONNECTION_STRING environment variable not set.")
        raise Exception("Missing DATABASE_CONNECTION_STRING in environment.")
    
    backup_cmd = f'pg_dump "{db_conn_str}" -F c -b -v -f "backups/local_backup.dump"'
    logger.info(f"def events_table_backup_and_drop(): Backing up database with command: {backup_cmd}")
    
    # Retry backup command 3 times with 5-second delay
    for attempt in range(3):
        try:
            result_backup = subprocess.run(backup_cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"def events_table_backup_and_drop(): Database backup completed: {result_backup.stdout}")
            break
        except subprocess.CalledProcessError as e:
            logger.error(f"def events_table_backup_and_drop(): Database backup failed on attempt {attempt+1}: {e.stderr}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise e

    drop_cmd = f'psql -d "{db_conn_str}" -c "DROP TABLE IF EXISTS events;"'
    logger.info(f"def events_table_backup_and_drop(): Dropping events table with command: {drop_cmd}")
    
    # Retry drop command 3 times with 5-second delay
    for attempt in range(3):
        try:
            result_drop = subprocess.run(drop_cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"def events_table_backup_and_drop(): Events table drop result: {result_drop.stdout}")
            break
        except subprocess.CalledProcessError as e:
            logger.error(f"def events_table_backup_and_drop(): Failed to drop events table on attempt {attempt+1}: {e.stderr}")
            if attempt < 2:
                time.sleep(5)
            else:
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
        result = subprocess.run([sys.executable, "src/gs.py"], check=True, capture_output=True, text=True)
        logger.info("def run_gs_script(): gs.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"gs.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/ebs.py"], check=True, capture_output=True, text=True)
        logger.info("def run_ebs_script(): ebs.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"ebs.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/emails.py"], check=True, capture_output=True, text=True)
        logger.info("def run_emails_script(): emails.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"emails.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/rd_ext.py"], check=True, capture_output=True, text=True)
        logger.info("def run_rd_ext_script(): rd_ext.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"rd_ext.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/scraper.py"], check=True, capture_output=True, text=True)
        logger.info("def run_scraper_script(): scraper.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"scraper.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/fb.py"], check=True, capture_output=True, text=True)
        logger.info("def run_fb_script(): fb.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"fb.py failed with error: {e.stderr}"
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
# TASKS FOR DB.PY STEP
# ------------------------
@task
def run_db_script():
    try:
        result = subprocess.run([sys.executable, "src/db.py"], check=True, capture_output=True, text=True)
        logger.info("def run_db_script(): db.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"db.py failed with error: {e.stderr}"
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
    return True

# ------------------------
# TASK FOR DATABASE BACKUP STEP (Using .dump)
# ------------------------
@task
def backup_db_step():
    backup_cmd = "pg_dump -U postgres -h localhost -F c -b -v -f 'backups/checkpoint.dump' social_dance_db"
    db_password = os.getenv("DATABASE_PASSWORD")
    if not db_password:
        logger.error("def backup_db_step(): DATABASE_PASSWORD environment variable not set.")
        raise Exception("Missing DATABASE_PASSWORD.")
    env = os.environ.copy()
    env["PGPASSWORD"] = db_password
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
        result = subprocess.run([sys.executable, "src/clean_up.py"], check=True, capture_output=True, text=True)
        logger.info("def run_clean_up_script(): clean_up.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"clean_up.py failed with error: {e.stderr}"
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
        result = subprocess.run([sys.executable, "src/dedup_llm.py"], check=True, capture_output=True, text=True)
        logger.info("def run_dedup_llm_script(): dedup_llm.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"dedup_llm.py failed with error: {e.stderr}"
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
    if (df['Label'] == 0).all():
        logger.info("def post_process_dedup_llm(): 'Label' column is all zeros.")
        return True
    else:
        logger.warning("def post_process_dedup_llm(): 'Label' column is not all zeros.")
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
        result = subprocess.run([sys.executable, "src/irrelevant_rows.py"], check=True, capture_output=True, text=True)
        logger.info("def run_irrelevant_rows_script(): irrelevant_rows.py executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"irrelevant_rows.py failed with error: {e.stderr}"
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
# TASKS FOR DB MAINTENANCE STEP (Using .dump and pg_restore)
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

@flow(name="DB Maintenance Step")
def db_maintenance_step():
    logger.info("def db_maintenance_step(): Starting DB Maintenance Step.")
    env_local = os.environ.copy()
    local_database_password = os.getenv("DATABASE_PASSWORD")
    if not local_database_password:
        logger.error("def db_maintenance_step(): DATABASE_PASSWORD environment variable not set.")
        raise Exception("Missing DATABASE_PASSWORD.")
    env_local["PGPASSWORD"] = local_database_password

    dump_command = "pg_dump -U postgres -h localhost -F c -b -v -f 'backups/local_backup.dump' social_dance_db"
    try:
        subprocess.run(dump_command, shell=True, check=True, capture_output=True, text=True, env=env_local, timeout=30)
        logger.info("def db_maintenance_step(): Database dump (.dump) completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"def db_maintenance_step(): Database dump failed: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired as te:
        logger.error(f"def db_maintenance_step(): Database dump timed out: {te}")
        raise te

    render_pg_pass = os.getenv("RENDER_PG_PASS")
    if not render_pg_pass:
        logger.error("def db_maintenance_step(): RENDER_PG_PASS environment variable not set.")
        raise Exception("Missing RENDER_PG_PASS.")

    copy_command = (
        f"PGPASSWORD={render_pg_pass} pg_restore --no-owner --clean --if-exists "
        f"--dbname=postgresql://social_dance_db_user:{render_pg_pass}"
        f"@dpg-culu0r1u0jms73bgrcdg-a.oregon-postgres.render.com:5432/social_dance_db_eimr?sslmode=require "
        f"-v -c 'backups/local_backup.dump'"
    )
    logger.info(f"def db_maintenance_step(): Copy command: {copy_command}")
    try:
        subprocess.run(copy_command, shell=True, check=True, capture_output=True, text=True, env=os.environ.copy(), timeout=30)
        logger.info("def db_maintenance_step(): Database copy (restore) command executed successfully.")
    except subprocess.TimeoutExpired as te:
        logger.error(f"def db_maintenance_step(): Database copy timed out: {te}")
        # Trap the timeout error and log it.
    except subprocess.CalledProcessError as e:
        logger.error(f"def db_maintenance_step(): Database copy failed: {e.stderr}")
        # Trap the error and log it without crashing the pipeline.

    alter_command = (
        "psql -h dpg-culu0r1u0jms73bgrcdg-a.oregon-postgres.render.com "
        "-U social_dance_db_user -d social_dance_db_eimr "
        "-c \"ALTER DATABASE social_dance_db_eimr SET TIME ZONE 'PST8PDT';\""
    )
    env_render = os.environ.copy()
    env_render["PGPASSWORD"] = render_pg_pass
    try:
        subprocess.run(alter_command, shell=True, check=True, capture_output=True, text=True, env=env_render, timeout=30)
        logger.info("def db_maintenance_step(): Database time zone altered successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"def db_maintenance_step(): Altering time zone failed: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired as te:
        logger.error(f"def db_maintenance_step(): Altering time zone timed out: {te}")
        raise te

    show_command = (
        "psql -h dpg-culu0r1u0jms73bgrcdg-a.oregon-postgres.render.com "
        "-U social_dance_db_user -d social_dance_db_eimr "
        "-c \"SHOW TIME ZONE;\""
    )
    try:
        result = subprocess.run(show_command, shell=True, check=True, capture_output=True, text=True, env=env_render, timeout=30)
        logger.info(f"def db_maintenance_step(): Database time zone: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"def db_maintenance_step(): Showing time zone failed: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired as te:
        logger.error(f"def db_maintenance_step(): Showing time zone timed out: {te}")
        raise te

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
    ("events_table_backup_and_drop", events_table_backup_and_drop),
    ("db", db_step),
    ("emails", emails_step),
    ("gs", gs_step),
    ("ebs", ebs_step),
    ("rd_ext", rd_ext_step),
    ("scraper", scraper_step),
    ("fb", fb_step),
    ("backup_db", backup_db_step),
    ("db", db_step),
    ("clean_up", clean_up_step),
    ("dedup_llm", dedup_llm_step),
    ("irrelevant_rows", irrelevant_rows_step),
    ("db_maintenance", db_maintenance_step)
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

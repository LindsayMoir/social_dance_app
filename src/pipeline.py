# pipeline.py

import argparse
import csv
import copy
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import logging
import os
import json
import re
import uuid
from sqlalchemy import create_engine, text

# Load .env from src directory (where this script is located)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
import pandas as pd
import shutil
import subprocess
import sys
import time
import yaml
from utils.crawl_telemetry_tuning import tune_crawl_config_from_first_pass
from utils.chatbot_metrics_sync_utils import (
    count_nullish_datetime_values,
    safe_db_target_label,
    sanitize_records_for_sql,
    utc_now_iso_seconds,
)
from classifier_training_promoter import (
    parse_manual_review_label,
    promote_manual_review_training_rows,
    summarize_manual_review_csv,
)
from db import ensure_chatbot_metrics_schema
from db import EVENTS_HISTORY_TABLE_SCHEMA_SQL, EVENTS_TABLE_SCHEMA_SQL, ADDRESS_TABLE_SCHEMA_SQL

# Setup centralized logging (logging_config.py is in the same directory)
from logging_config import setup_logging
setup_logging('pipeline')
from output_paths import codex_review_path, reports_path

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
_STEP_RUNTIME_CONFIG_PATHS: dict[str, str] = {}

_TRANSIENT_DB_ERROR_MARKERS = (
    "database is locked",
    "operationalerror",
    "could not connect to server",
    "connection refused",
    "connection reset by peer",
    "server closed the connection unexpectedly",
    "terminating connection due to administrator command",
    "remaining connection slots are reserved",
    "too many clients already",
    "temporary failure in name resolution",
    "name or service not known",
    "timeout expired",
    "connection timed out",
    "ssl syscall error: eof detected",
    "could not translate host name",
    "the database system is starting up",
)
CHATBOT_METRICS_SYNC_LOG_PATH = os.path.join(log_dir, "chatbot_metrics_sync_log.txt")
RUN_SCORECARD_PATH = codex_review_path("run_scorecard.json")
VALIDATION_REPORT_PATH = reports_path("comprehensive_test_report.html")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL_COVERAGE_AUDIT_PATH = os.path.join(REPO_ROOT, "data", "evaluation", "manual_coverage_audit.csv")
DATABASE_ACCURACY_MANUAL_REVIEW_PATH = codex_review_path("database_accuracy_manual_review.csv")
URL_ARCHETYPE_ML_CLASSIFIER_REVIEW_PATH = codex_review_path("url_archetype_ml_classifier_review.csv")
CHATBOT_EVALUATION_REVIEW_PATH = codex_review_path("chatbot_evaluation_review.csv")
CLASSIFIER_TRAINING_CSV_PATH = os.path.join(REPO_ROOT, "ml", "training_data", "original_td.csv")
_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _is_transient_database_error(message: str) -> bool:
    """Return True if text looks like a retriable database failure."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_DB_ERROR_MARKERS)


def _append_chatbot_sync_log(event: str, details: dict | None = None, level: str = "INFO") -> None:
    """Append one structured line for chatbot metrics sync diagnostics."""
    payload = {
        "timestamp_utc": utc_now_iso_seconds(),
        "level": str(level or "INFO").upper(),
        "event": event,
        "details": details or {},
    }
    serialized = json.dumps(payload, ensure_ascii=True) + "\n"
    targets = [CHATBOT_METRICS_SYNC_LOG_PATH]
    archive_dir = os.getenv("DS_LOG_ARCHIVE_DIR", "").strip()
    if archive_dir:
        targets.append(os.path.join(archive_dir, "chatbot_metrics_sync_log.txt"))
    try:
        for target in targets:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(serialized)
    except Exception as e:
        logger.warning("_append_chatbot_sync_log(): failed to append diagnostic log: %s", e)


def _derive_log_archive_timestamp(logs_dir: str) -> str:
    """Return an archive folder timestamp derived from existing step logs when possible."""
    candidate_names = [
        "credential_validator_log.txt",
        "pipeline_log.txt",
    ]
    try:
        available_names = sorted(
            name
            for name in os.listdir(logs_dir)
            if name.endswith(".log") or name.endswith(".txt")
        )
    except Exception:
        available_names = []

    for name in available_names:
        if name not in candidate_names:
            candidate_names.append(name)

    for filename in candidate_names:
        path = os.path.join(logs_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    match = _LOG_TIMESTAMP_RE.match(line.strip())
                    if not match:
                        continue
                    parsed = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    return parsed.strftime("%Y%m%d_%H%M%S")
        except Exception as exc:
            logger.warning("_derive_log_archive_timestamp(): failed to read %s: %s", path, exc)

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_run_scorecard() -> dict:
    """Load the latest validation run scorecard from disk when available."""
    try:
        with open(RUN_SCORECARD_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_validation_timeout_seconds() -> int:
    """Return the subprocess timeout for the validation runner."""
    raw_value = (
        cfg.get("testing", {})
        .get("validation", {})
        .get("pipeline_timeout_seconds", 5400)
    )
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError):
        timeout_seconds = 5400
    return max(300, timeout_seconds)


def _safe_file_mtime(path: str) -> float | None:
    """Return file mtime as epoch seconds when the file exists."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _file_contains_text(path: str, needle: str) -> bool:
    """Return True when a text file contains the provided substring."""
    if not needle:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return needle in handle.read()
    except OSError:
        return False


def _validation_report_was_regenerated(
    report_path: str,
    previous_mtime: float | None,
    started_at_epoch: float,
    run_id: str,
) -> bool:
    """Return True when the validation HTML report was rewritten for the current run."""
    current_mtime = _safe_file_mtime(report_path)
    if current_mtime is None:
        return False
    if previous_mtime is not None and current_mtime <= previous_mtime:
        return False
    if current_mtime < started_at_epoch:
        return False
    if run_id and not _file_contains_text(report_path, run_id):
        return False
    return True


def _sql_one_line(statement: str) -> str:
    """Collapse SQL into a single line suitable for psql -c usage."""
    return " ".join(str(statement or "").split())


def _database_accuracy_manual_review_status(csv_path: str | None = None) -> dict:
    """Return completion status for the prior database accuracy manual review CSV."""
    target_path = os.path.abspath(str(csv_path or DATABASE_ACCURACY_MANUAL_REVIEW_PATH))
    summary = summarize_manual_review_csv(target_path)
    if not summary.get("exists"):
        return {
            "path": target_path,
            "exists": False,
            "rows_total": 0,
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "false_rows_missing_truth": 0,
            "correctness_pct": None,
            "complete": True,
            "reason": "missing_file",
        }

    rows_total = int(summary.get("rows_total", 0) or 0)
    if rows_total <= 0:
        return {
            "path": target_path,
            "exists": True,
            "rows_total": 0,
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "false_rows_missing_truth": 0,
            "correctness_pct": None,
            "complete": True,
            "reason": "empty_file",
        }
    rows_missing_label = int(summary.get("rows_missing_label", 0) or 0)
    false_rows_missing_truth = int(summary.get("false_rows_missing_truth", 0) or 0)
    complete = rows_total > 0 and rows_missing_label == 0 and false_rows_missing_truth == 0
    return {
        "path": target_path,
        "exists": True,
        "rows_total": rows_total,
        "rows_completed": int(summary.get("rows_completed", 0) or 0),
        "rows_missing_label": rows_missing_label,
        "rows_true": int(summary.get("rows_true", 0) or 0),
        "rows_false": int(summary.get("rows_false", 0) or 0),
        "false_rows_missing_truth": false_rows_missing_truth,
        "correctness_pct": summary.get("correctness_pct"),
        "complete": complete,
        "reason": (
            "complete"
            if complete
            else ("missing_false_truth_labels" if false_rows_missing_truth else "missing_human_labels")
        ),
    }


def _database_event_accuracy_manual_review_status(csv_path: str | None = None) -> dict:
    """Return completion status for the prior event-accuracy manual review CSV."""
    target_path = os.path.abspath(str(csv_path or DATABASE_ACCURACY_MANUAL_REVIEW_PATH))
    if not os.path.exists(target_path):
        return {
            "path": target_path,
            "exists": False,
            "rows_total": 0,
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "correctness_pct": None,
            "complete": True,
            "reason": "missing_file",
        }

    with open(target_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows_total = len(rows)
    if rows_total <= 0:
        return {
            "path": target_path,
            "exists": True,
            "rows_total": 0,
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "correctness_pct": None,
            "complete": True,
            "reason": "empty_file",
        }
    rows_completed = 0
    rows_missing_label = 0
    rows_true = 0
    rows_false = 0
    for row in rows:
        parsed = parse_manual_review_label(row.get("human_label"))
        if parsed is None:
            if str(row.get("human_label") or "").strip():
                rows_completed += 1
            else:
                rows_missing_label += 1
            continue
        rows_completed += 1
        if parsed:
            rows_true += 1
        else:
            rows_false += 1

    labeled_rows = rows_true + rows_false
    correctness_pct = round((rows_true / labeled_rows) * 100.0, 2) if labeled_rows else None
    complete = rows_total > 0 and rows_missing_label == 0
    return {
        "path": target_path,
        "exists": True,
        "rows_total": rows_total,
        "rows_completed": rows_completed,
        "rows_missing_label": rows_missing_label,
        "rows_true": rows_true,
        "rows_false": rows_false,
        "correctness_pct": correctness_pct,
        "complete": complete,
        "reason": "complete" if complete else "missing_human_labels",
    }


def _chatbot_evaluation_manual_review_status(csv_path: str | None = None) -> dict:
    """Return completion status for the prior chatbot evaluation manual review CSV."""
    target_path = os.path.abspath(str(csv_path or CHATBOT_EVALUATION_REVIEW_PATH))
    if not os.path.exists(target_path):
        return _database_event_accuracy_manual_review_status(target_path)

    with open(target_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "human_label" not in fieldnames:
        return {
            "path": target_path,
            "exists": True,
            "rows_total": len(rows),
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "correctness_pct": None,
            "complete": True,
            "reason": "legacy_missing_review_columns",
        }

    return _database_event_accuracy_manual_review_status(target_path)


def _persist_manual_review_classifier_accuracy(status: dict, db_handler=None) -> None:
    """Persist classifier correctness derived from the scored manual-review CSV."""
    correctness_pct = status.get("correctness_pct")
    rows_true = int(status.get("rows_true", 0) or 0)
    rows_false = int(status.get("rows_false", 0) or 0)
    labeled_rows = rows_true + rows_false
    if correctness_pct is None or labeled_rows <= 0:
        return

    from db import DatabaseHandler

    handler = db_handler if db_handler is not None else DatabaseHandler(cfg)
    run_id = (
        str(os.getenv("DS_RUN_ID", "")).strip()
        or f"manual-review-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    handler.record_metric_observation(
        run_id=run_id,
        metric_key="classifier_manual_correctness_pct",
        metric_value_numeric=float(correctness_pct),
        metric_unit="percent",
        description="Share of reviewed classifier audit URLs marked correct by human scoring",
        higher_is_better=True,
        notes={
            "review_csv_path": str(status.get("path") or ""),
            "labeled_rows": labeled_rows,
            "true_count": rows_true,
            "false_count": rows_false,
        },
    )


def refresh_manual_coverage_audit_csv(
    sample_size: int = 100,
    output_path: str | None = None,
    db_handler=None,
) -> dict:
    """Replace the manual coverage audit CSV with a fresh near-future sample from current events."""
    from db import DatabaseHandler

    safe_sample_size = max(1, int(sample_size or 1))
    safe_output_path = output_path or MANUAL_COVERAGE_AUDIT_PATH
    handler = db_handler if db_handler is not None else DatabaseHandler(cfg)
    rows = handler.execute_query(
        """
        SELECT
            COALESCE(NULLIF(TRIM(source), ''), COALESCE(NULLIF(TRIM(location), ''), NULLIF(TRIM(url), ''))) AS source_name,
            url,
            event_name,
            start_date
        FROM events
        WHERE start_date IS NOT NULL
          AND start_date >= CURRENT_DATE + INTERVAL '7 days'
          AND start_date <= CURRENT_DATE + INTERVAL '21 days'
          AND COALESCE(NULLIF(TRIM(url), ''), '') <> ''
          AND COALESCE(NULLIF(TRIM(event_name), ''), '') <> ''
        ORDER BY random()
        LIMIT :limit
        """,
        {"limit": safe_sample_size},
    ) or []
    fieldnames = [
        "source_name",
        "source_url",
        "event_name",
        "start_date",
        "expected_present",
        "active",
        "notes",
    ]
    generated_on = datetime.datetime.now().date().isoformat()
    normalized_rows: list[dict[str, str]] = []
    for source_name, source_url, event_name, start_date in rows:
        normalized_rows.append(
            {
                "source_name": str(source_name or source_url or "").strip(),
                "source_url": str(source_url or "").strip(),
                "event_name": str(event_name or "").strip(),
                "start_date": str(start_date or "").strip(),
                "expected_present": "True",
                "active": "True",
                "notes": f"Auto-generated from current events table on {generated_on} after copy_dev_to_prod (start_date +7d to +21d window)",
            }
        )

    os.makedirs(os.path.dirname(safe_output_path), exist_ok=True)
    with open(safe_output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    logger.info(
        "refresh_manual_coverage_audit_csv(): wrote %s rows to %s",
        len(normalized_rows),
        safe_output_path,
    )
    return {
        "output_path": safe_output_path,
        "rows_written": len(normalized_rows),
        "sample_size_requested": safe_sample_size,
    }


def _scorecard_guardrails_allow(action: str) -> bool:
    """Return True when the latest scorecard permits the requested action."""
    scorecard = _load_run_scorecard()
    if not scorecard:
        logger.warning("_scorecard_guardrails_allow(): No run scorecard found for action=%s", action)
        return False
    guardrails = scorecard.get("guardrails", {}) if isinstance(scorecard.get("guardrails"), dict) else {}
    status = str(guardrails.get("status", "UNKNOWN") or "UNKNOWN").upper()
    if status == "PASS":
        return True
    logger.warning(
        "_scorecard_guardrails_allow(): Blocking %s because guardrails status is %s",
        action,
        status,
    )
    violations = guardrails.get("violations", []) if isinstance(guardrails.get("violations"), list) else []
    for violation in violations[:10]:
        logger.warning("_scorecard_guardrails_allow(): violation=%s", violation)
    return False


def _scorecard_has_required_evaluation_scope(action: str) -> bool:
    """Return True when the latest scorecard contains usable dev and holdout evaluation results."""
    scorecard = _load_run_scorecard()
    if not scorecard:
        logger.warning("_scorecard_has_required_evaluation_scope(): No run scorecard found for action=%s", action)
        return False
    evaluation_scope = scorecard.get("evaluation_scope", {}) if isinstance(scorecard.get("evaluation_scope"), dict) else {}
    uses_dev_split = bool(evaluation_scope.get("uses_dev_split"))
    uses_holdout = bool(evaluation_scope.get("uses_holdout"))
    dev_summary = evaluation_scope.get("dev_summary", {}) if isinstance(evaluation_scope.get("dev_summary"), dict) else {}
    holdout_summary = evaluation_scope.get("holdout_summary", {}) if isinstance(evaluation_scope.get("holdout_summary"), dict) else {}
    dev_accuracy = dev_summary.get("replay_url_accuracy_pct")
    holdout_accuracy = holdout_summary.get("replay_url_accuracy_pct")
    if uses_dev_split and uses_holdout and dev_accuracy is not None and holdout_accuracy is not None:
        return True
    logger.warning(
        "_scorecard_has_required_evaluation_scope(): Blocking %s because dev/holdout evaluation is incomplete "
        "(uses_dev_split=%s, uses_holdout=%s, dev_accuracy=%s, holdout_accuracy=%s)",
        action,
        uses_dev_split,
        uses_holdout,
        dev_accuracy,
        holdout_accuracy,
    )
    return False


def _scorecard_evaluation_deltas_allow(action: str) -> bool:
    """Return True when dev and holdout comparisons do not show regressions."""
    scorecard = _load_run_scorecard()
    if not scorecard:
        logger.warning("_scorecard_evaluation_deltas_allow(): No run scorecard found for action=%s", action)
        return False
    comparison_summary = scorecard.get("comparison_summary", {}) if isinstance(scorecard.get("comparison_summary"), dict) else {}
    previous_run = comparison_summary.get("previous_run", {}) if isinstance(comparison_summary.get("previous_run"), dict) else {}
    holdout_baseline = comparison_summary.get("holdout_baseline", {}) if isinstance(comparison_summary.get("holdout_baseline"), dict) else {}
    if not previous_run.get("available") or not holdout_baseline.get("available"):
        logger.warning(
            "_scorecard_evaluation_deltas_allow(): Blocking %s because previous-run or holdout-baseline comparison is unavailable",
            action,
        )
        return False

    def _metric_regressed(comparison: dict, metric_key: str) -> bool:
        for item in comparison.get("metric_deltas", []) if isinstance(comparison.get("metric_deltas"), list) else []:
            if str(item.get("metric_key") or "") != metric_key:
                continue
            return str(item.get("direction") or "").lower() == "regressed"
        return True

    dev_regressed = _metric_regressed(previous_run, "dev_replay_url_accuracy_pct")
    holdout_regressed = _metric_regressed(holdout_baseline, "holdout_replay_url_accuracy_pct")
    if not dev_regressed and not holdout_regressed:
        return True
    logger.warning(
        "_scorecard_evaluation_deltas_allow(): Blocking %s because evaluation regressed "
        "(dev_regressed=%s, holdout_regressed=%s)",
        action,
        dev_regressed,
        holdout_regressed,
    )
    return False


def _log_copy_dev_to_prod_evaluation_warnings() -> None:
    """Log evaluation warnings for explicit dev-to-prod copies without blocking the copy."""
    if not _scorecard_guardrails_allow("copy_dev_to_prod"):
        logger.warning(
            "_log_copy_dev_to_prod_evaluation_warnings(): proceeding with explicit copy_dev_to_prod "
            "despite failing or unavailable guardrails"
        )
    if not _scorecard_has_required_evaluation_scope("copy_dev_to_prod"):
        logger.warning(
            "_log_copy_dev_to_prod_evaluation_warnings(): proceeding with explicit copy_dev_to_prod "
            "without complete dev/holdout evaluation scope"
        )
    if not _scorecard_evaluation_deltas_allow("copy_dev_to_prod"):
        logger.warning(
            "_log_copy_dev_to_prod_evaluation_warnings(): proceeding with explicit copy_dev_to_prod "
            "despite dev/holdout regression or missing comparison data"
        )

# Define common configuration updates for all pipeline steps
COMMON_CONFIG_UPDATES = {
    "testing": {"drop_tables": False},
    "crawling": {
         "headless": True,
         "max_website_urls": 10,
         "urls_run_limit": 500,  # default for all steps
    },
    "llm": {
        "provider": "openrouter",
        "provider_rotation_enabled": False,
        "provider_rotation_order": ["openrouter", "openai", "gemini"],
        "fallback_enabled": True,
        "fallback_provider_order": ["openrouter", "openai", "gemini"],
        "provider_exclusions": ["mistral"],
        "regular_openai_first_every_n_requests": 0,
        "spend_money": True,
    }
}

PARALLEL_CRAWL_CONFIG_UPDATES = copy.deepcopy(COMMON_CONFIG_UPDATES)
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["urls_run_limit"] = 1500
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["max_website_urls"] = 10
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_download_timeout_seconds"] = 50
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_playwright_timeout_ms"] = 35000
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_retry_times"] = 2
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_post_load_wait_ms"] = 1000
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_concurrent_requests"] = 8
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["scraper_concurrent_requests_per_domain"] = 2
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_base_urls_limit"] = 180
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_event_links_per_base_limit"] = 20
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_post_nav_wait_ms"] = 1800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_post_expand_wait_ms"] = 900
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_final_wait_ms"] = 700

PARALLEL_CRAWLER_STEPS = {"rd_ext", "ebs", "scraper", "fb", "images"}
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_failures_before_cooldown"] = 2
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_cooldown_base_seconds"] = 300
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_cooldown_max_seconds"] = 1800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_state_max_scopes"] = 800
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_block_state_ttl_days"] = 45
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_temp_block_policy_enabled"] = True
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_temp_block_wait_min_seconds"] = 300
PARALLEL_CRAWL_CONFIG_UPDATES["crawling"]["fb_temp_block_wait_max_seconds"] = 600


def _merge_nested_updates(base: dict, overlay: dict) -> dict:
    """Merge nested dict updates without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

# ------------------------
# HELPER TASKS: Backup and Restore Config
# ------------------------

@task
def backup_and_update_config(step: str, updates: dict) -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        original_config = yaml.safe_load(f)
    logger.info("def backup_and_update_config(): Original config loaded.")
    logger.info("def backup_and_update_config(): Starting pipeline.py")
    updated_config = copy.deepcopy(original_config)
    for key, value in updates.items():
        if key in updated_config and isinstance(updated_config[key], dict) and isinstance(value, dict):
            updated_config[key].update(value)
        else:
            updated_config[key] = value
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_config_name = f"runtime_config_{step}_{run_time}_{uuid.uuid4().hex[:8]}.yaml"
    runtime_config_dir = os.path.join("config", "run_specific_configs")
    os.makedirs(runtime_config_dir, exist_ok=True)
    runtime_config_path = os.path.join(runtime_config_dir, runtime_config_name)
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(updated_config, f, sort_keys=False)
    _STEP_RUNTIME_CONFIG_PATHS[step] = runtime_config_path
    os.environ["DS_CONFIG_PATH"] = runtime_config_path
    logger.info(
        "def backup_and_update_config(): Runtime config for step '%s' written to %s with updates: %s",
        step,
        runtime_config_path,
        updates,
    )
    return original_config

@task
def restore_config(original_config: dict, step: str):
    _ = original_config
    runtime_config_path = _STEP_RUNTIME_CONFIG_PATHS.pop(step, "")
    if runtime_config_path:
        try:
            os.remove(runtime_config_path)
            logger.info("def restore_config(): Removed runtime config for step '%s': %s", step, runtime_config_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("def restore_config(): Could not remove runtime config '%s': %s", runtime_config_path, e)
    os.environ.pop("DS_CONFIG_PATH", None)
    logger.info(f"def restore_config(): Cleared runtime config override after step '{step}'.")

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
    logger.info("Validating Gmail, Eventbrite, Facebook, and Instagram credentials")
    logger.info("Browser will open for user interaction if needed")
    logger.info("=" * 70)

    original_config = backup_and_update_config("credential_validation", updates={})
    write_run_config.submit("credential_validation", original_config)
    try:
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
    finally:
        restore_config(original_config, "credential_validation")

# ------------------------
# TASK: COPY LOG FILES
# ------------------------
@task
def copy_log_files():
    """Move all log files to a timestamped folder in logs directory."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        logger.warning(f"def copy_log_files(): Logs directory {logs_dir} does not exist.")
        return True

    timestamp = _derive_log_archive_timestamp(logs_dir)
    archive_folder = f"logs/logs_{timestamp}"

    # Create the archive folder
    os.makedirs(archive_folder, exist_ok=True)
    os.environ["DS_LOG_ARCHIVE_DIR"] = archive_folder
    logger.info(f"def copy_log_files(): Created archive folder: {archive_folder}")

    # Get all log files from logs directory
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


@flow(name="Database Accuracy Manual Review Gate Step")
def database_accuracy_manual_review_gate_step():
    """Block the pipeline until the prior event, classifier, and chatbot review CSVs are completed."""
    logger.info("=" * 70)
    logger.info("DATABASE ACCURACY MANUAL REVIEW GATE STEP")
    logger.info("Checking whether the prior database accuracy, URL classifier review, and chatbot review CSVs are complete")
    logger.info("=" * 70)

    while True:
        event_status = _database_event_accuracy_manual_review_status()
        classifier_status = _database_accuracy_manual_review_status(URL_ARCHETYPE_ML_CLASSIFIER_REVIEW_PATH)
        chatbot_status = _chatbot_evaluation_manual_review_status()
        logger.info(
            "database_accuracy_manual_review_gate_step(): event_status=%s classifier_status=%s chatbot_status=%s",
            event_status,
            classifier_status,
            chatbot_status,
        )

        event_complete = bool(event_status.get("complete"))
        classifier_complete = bool(classifier_status.get("complete"))
        chatbot_complete = bool(chatbot_status.get("complete"))

        if event_complete and classifier_complete and chatbot_complete:
            no_prior_review_reasons = {"missing_file", "empty_file"}
            if (
                event_status.get("reason") in no_prior_review_reasons
                and classifier_status.get("reason") in no_prior_review_reasons
                and chatbot_status.get("reason") in no_prior_review_reasons
            ):
                logger.info(
                    "database_accuracy_manual_review_gate_step(): No prior completed manual review CSVs found; continuing."
                )
                return True

            _persist_manual_review_classifier_accuracy(classifier_status)
            promotion_summary = promote_manual_review_training_rows(
                review_csv_path=str(classifier_status.get("path") or URL_ARCHETYPE_ML_CLASSIFIER_REVIEW_PATH),
                training_csv_path=CLASSIFIER_TRAINING_CSV_PATH,
                true_limit=2,
                false_limit=2,
            )
            logger.info(
                "database_accuracy_manual_review_gate_step(): Prior manual review CSVs complete. "
                "Database accuracy=%s%% (%s/%s rows). Classifier correctness=%s%% (%s/%s rows). "
                "Chatbot correctness=%s%% (%s/%s rows). "
                "Promoted %s review rows into training CSV (%s true, %s false). Continuing.",
                event_status.get("correctness_pct"),
                event_status.get("rows_completed", 0),
                event_status.get("rows_total", 0),
                classifier_status.get("correctness_pct"),
                classifier_status.get("rows_completed", 0),
                classifier_status.get("rows_total", 0),
                chatbot_status.get("correctness_pct"),
                chatbot_status.get("rows_completed", 0),
                chatbot_status.get("rows_total", 0),
                promotion_summary.get("promoted_count", 0),
                promotion_summary.get("promoted_true_count", 0),
                promotion_summary.get("promoted_false_count", 0),
            )
            return True

        print("\nManual review requirements are not complete.")
        print("Database Accuracy Manual Review:")
        print(f"CSV: {event_status.get('path')}")
        print(
            f"Rows completed: {event_status.get('rows_completed', 0)} / {event_status.get('rows_total', 0)}; "
            f"missing human_label: {event_status.get('rows_missing_label', 0)}"
        )
        print("URL Archetype ML Classifier Review:")
        print(f"CSV: {classifier_status.get('path')}")
        print(
            f"Rows completed: {classifier_status.get('rows_completed', 0)} / {classifier_status.get('rows_total', 0)}; "
            f"missing human_label: {classifier_status.get('rows_missing_label', 0)}; "
            f"false rows missing truth labels: {classifier_status.get('false_rows_missing_truth', 0)}"
        )
        print("Chatbot Evaluation Manual Review:")
        print(f"CSV: {chatbot_status.get('path')}")
        print(
            f"Rows completed: {chatbot_status.get('rows_completed', 0)} / {chatbot_status.get('rows_total', 0)}; "
            f"missing human_label: {chatbot_status.get('rows_missing_label', 0)}"
        )
        acknowledgment = input(
            "Complete the required CSVs, then type COMPLETE to re-check: "
        ).strip()
        if acknowledgment.upper() != "COMPLETE":
            print("Acknowledgment not received. The pipeline will continue waiting.")

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
                + _sql_one_line(EVENTS_HISTORY_TABLE_SCHEMA_SQL.replace("events_history", "events_history_new", 1))
                + "; "
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
            f"{_sql_one_line(EVENTS_HISTORY_TABLE_SCHEMA_SQL)}; "
            "INSERT INTO events_history (original_event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp) "
            "SELECT event_id, event_name, dance_style, description, day_of_week, start_date, end_date, start_time, end_time, source, location, price, url, event_type, address_id, time_stamp FROM events; "
            "DROP TABLE IF EXISTS events; "
            f"{_sql_one_line(EVENTS_TABLE_SCHEMA_SQL)}; COMMIT;"
        )
    else:
        # Events table doesn't exist - just create both tables fresh
        logger.info("def copy_drop_create_events(): Events table does not exist, creating fresh tables")
        sql = (
            "BEGIN; "
            f"{_sql_one_line(EVENTS_HISTORY_TABLE_SCHEMA_SQL)}; "
            f"{_sql_one_line(EVENTS_TABLE_SCHEMA_SQL)}; COMMIT;"
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
        f"{_sql_one_line(ADDRESS_TABLE_SCHEMA_SQL)}; "
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
    scraper_updates["crawling"]["urls_run_limit"] = 1500
    scraper_updates["crawling"]["scraper_download_timeout_seconds"] = 50
    scraper_updates["crawling"]["scraper_playwright_timeout_ms"] = 35000
    scraper_updates["crawling"]["scraper_retry_times"] = 2
    scraper_updates["crawling"]["scraper_post_load_wait_ms"] = 1000
    scraper_updates["crawling"]["scraper_concurrent_requests"] = 8
    scraper_updates["crawling"]["scraper_concurrent_requests_per_domain"] = 2
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
    fb_updates["crawling"]["fb_temp_block_policy_enabled"] = True
    fb_updates["crawling"]["fb_temp_block_wait_min_seconds"] = 300
    fb_updates["crawling"]["fb_temp_block_wait_max_seconds"] = 600
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
    logger.info("run_parallel_crawlers_script(): starting %s", script_path)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info(
                    "run_parallel_crawlers_script(): %s stdout tail:\n%s",
                    script_path,
                    result.stdout[-1000:],
                )
            logger.info("run_parallel_crawlers_script(): completed %s", script_path)
            return f"{script_path}:ok"
        except subprocess.CalledProcessError as e:
            stderr_text = (e.stderr or "").strip()
            stdout_text = (e.stdout or "").strip()
            combined_output = "\n".join(part for part in (stderr_text, stdout_text) if part)
            if _is_transient_database_error(combined_output) and attempt < max_attempts:
                delay_seconds = 5 * attempt
                logger.warning(
                    "run_parallel_crawlers_script(): transient DB failure for %s on attempt %d/%d, retrying in %ds.",
                    script_path,
                    attempt,
                    max_attempts,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                continue

            output_tail = combined_output[-1200:] if combined_output else "<no subprocess output captured>"
            error_message = (
                f"{script_path} failed with return code: {e.returncode} "
                f"(attempt {attempt}/{max_attempts}). Output tail:\n{output_tail}"
            )
            logger.error("run_parallel_crawlers_script(): %s", error_message)
            raise Exception(error_message)


@flow(name="Parallel Crawlers Step")
def parallel_crawlers_step():
    """
    Run rd_ext.py, ebs.py, scraper.py, fb.py, and images.py concurrently using one shared config snapshot.
    This avoids config race conditions from each individual step wrapper.
    """
    scripts = ["src/rd_ext.py", "src/ebs.py", "src/scraper.py", "src/fb.py", "src/images.py"]
    scraper_log_path = str(cfg.get("logging", {}).get("scraper_log_file", "logs/scraper_log.txt"))
    fb_log_path = os.path.join(os.path.dirname(scraper_log_path), "fb_log.txt")
    tuning = tune_crawl_config_from_first_pass(
        current_updates=PARALLEL_CRAWL_CONFIG_UPDATES,
        scraper_log_path=scraper_log_path,
        fb_log_path=fb_log_path,
    )
    runtime_updates = _merge_nested_updates(PARALLEL_CRAWL_CONFIG_UPDATES, tuning.updates)
    logger.info("parallel_crawlers_step(): phase3 telemetry=%s", tuning.telemetry)
    for decision in tuning.decisions:
        logger.info("parallel_crawlers_step(): phase3 decision: %s", decision)

    original_config = backup_and_update_config("parallel_crawlers", updates=runtime_updates)
    write_run_config.submit("parallel_crawlers", original_config)

    failures: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=len(scripts)) as executor:
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
    # Do not mutate config/config.yaml from pipeline runtime.
    logger.info("db_step: Skipping persistent config mutation for testing.drop_tables.")
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
    run_id = str(os.getenv("DS_RUN_ID", "") or "").strip()
    timeout_seconds = _get_validation_timeout_seconds()
    started_at_epoch = time.time()
    previous_report_mtime = _safe_file_mtime(VALIDATION_REPORT_PATH)
    try:
        # Note: No check=True - don't raise on failure (non-blocking)
        result = subprocess.run(
            [sys.executable, "tests/validation/test_runner.py"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        report_generated = _validation_report_was_regenerated(
            report_path=VALIDATION_REPORT_PATH,
            previous_mtime=previous_report_mtime,
            started_at_epoch=started_at_epoch,
            run_id=run_id,
        )

        if result.returncode == 0:
            logger.info("def run_validation_tests(): Validation completed successfully")
            logger.info(result.stdout)
            if not report_generated:
                logger.warning(
                    "def run_validation_tests(): Validation exited successfully but did not regenerate %s",
                    VALIDATION_REPORT_PATH,
                )
            return {
                "status": "success",
                "message": "Validation passed",
                "report_generated": report_generated,
                "report_path": VALIDATION_REPORT_PATH,
                "timeout_seconds": timeout_seconds,
            }
        else:
            logger.warning(f"def run_validation_tests(): Validation completed with issues (exit code {result.returncode})")
            logger.warning(result.stderr)
            logger.warning("def run_validation_tests(): Continuing pipeline despite validation issues")
            if not report_generated:
                logger.warning(
                    "def run_validation_tests(): Validation returned non-zero and did not regenerate %s",
                    VALIDATION_REPORT_PATH,
                )
            return {
                "status": "warning",
                "message": "Validation completed with warnings",
                "returncode": result.returncode,
                "report_generated": report_generated,
                "report_path": VALIDATION_REPORT_PATH,
                "timeout_seconds": timeout_seconds,
            }

    except subprocess.TimeoutExpired as exc:
        report_generated = _validation_report_was_regenerated(
            report_path=VALIDATION_REPORT_PATH,
            previous_mtime=previous_report_mtime,
            started_at_epoch=started_at_epoch,
            run_id=run_id,
        )
        logger.error(
            "def run_validation_tests(): Validation tests timed out after %s seconds",
            timeout_seconds,
        )
        if exc.stdout:
            logger.warning("def run_validation_tests(): Validation stdout before timeout:\n%s", exc.stdout[-4000:])
        if exc.stderr:
            logger.warning("def run_validation_tests(): Validation stderr before timeout:\n%s", exc.stderr[-4000:])
        if not report_generated:
            logger.warning(
                "def run_validation_tests(): No fresh validation report was generated before timeout; keeping existing artifact untouched"
            )
        logger.warning("def run_validation_tests(): Continuing pipeline despite timeout")
        return {
            "status": "timeout",
            "message": "Validation timed out",
            "report_generated": report_generated,
            "report_path": VALIDATION_REPORT_PATH,
            "timeout_seconds": timeout_seconds,
        }
    except Exception as e:
        logger.error(f"def run_validation_tests(): Unexpected error: {e}")
        logger.warning("def run_validation_tests(): Continuing pipeline despite error")
        return {
            "status": "error",
            "message": "Validation encountered error",
            "report_generated": False,
            "report_path": VALIDATION_REPORT_PATH,
            "timeout_seconds": timeout_seconds,
        }

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
# CLASSIFIER TRAINING PROMOTION STEP
# Promote safe URL-level replay candidates into the classifier training CSV
# ------------------------
@task
def run_classifier_training_promotion():
    """Promote auto-positive classifier queue candidates into the training CSV."""
    try:
        command = [
            sys.executable,
            "utilities/promote_classifier_training_candidates.py",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = _parse_json_subprocess_stdout(result.stdout)
        if result.returncode == 0:
            logger.info("run_classifier_training_promotion(): Promotion completed successfully")
            if result.stdout:
                logger.info(result.stdout.strip())
            return {
                "status": "success",
                "message": "Classifier training promotion completed",
                "payload": payload,
            }
        logger.warning(
            "run_classifier_training_promotion(): Promotion completed with issues (exit code %s)",
            result.returncode,
        )
        if result.stderr:
            logger.warning(result.stderr.strip())
        if result.stdout:
            logger.warning(result.stdout.strip())
        logger.warning("run_classifier_training_promotion(): Continuing pipeline despite promotion issues")
        return {
            "status": "warning",
            "message": "Classifier training promotion completed with warnings",
            "payload": payload,
        }
    except subprocess.TimeoutExpired:
        logger.error("run_classifier_training_promotion(): Promotion timed out after 120 seconds")
        logger.warning("run_classifier_training_promotion(): Continuing pipeline despite timeout")
        return {
            "status": "timeout",
            "message": "Classifier training promotion timed out",
            "payload": {},
        }
    except Exception as e:
        logger.error("run_classifier_training_promotion(): Unexpected error: %s", e)
        logger.warning("run_classifier_training_promotion(): Continuing pipeline despite error")
        return {
            "status": "error",
            "message": "Classifier training promotion encountered error",
            "payload": {},
        }


@task
def run_page_classifier_model_training():
    """Retrain the page-classifier ML artifact from the current training CSV."""
    try:
        command = [
            sys.executable,
            "utilities/train_page_classifier_models.py",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        payload = _parse_json_subprocess_stdout(result.stdout)
        if result.returncode == 0:
            logger.info("run_page_classifier_model_training(): Training completed successfully")
            if result.stdout:
                logger.info(result.stdout.strip())
            return {
                "status": "success",
                "message": "Page classifier model training completed",
                "payload": payload,
            }
        logger.warning(
            "run_page_classifier_model_training(): Training completed with issues (exit code %s)",
            result.returncode,
        )
        if result.stderr:
            logger.warning(result.stderr.strip())
        if result.stdout:
            logger.warning(result.stdout.strip())
        logger.warning("run_page_classifier_model_training(): Continuing pipeline despite training issues")
        return {
            "status": "warning",
            "message": "Page classifier model training completed with warnings",
            "payload": payload,
        }
    except subprocess.TimeoutExpired:
        logger.error("run_page_classifier_model_training(): Training timed out after 300 seconds")
        logger.warning("run_page_classifier_model_training(): Continuing pipeline despite timeout")
        return {
            "status": "timeout",
            "message": "Page classifier model training timed out",
            "payload": {},
        }
    except Exception as e:
        logger.error("run_page_classifier_model_training(): Unexpected error: %s", e)
        logger.warning("run_page_classifier_model_training(): Continuing pipeline despite error")
        return {
            "status": "error",
            "message": "Page classifier model training encountered error",
            "payload": {},
        }


@flow(name="Classifier Training Promotion Step")
def classifier_training_promotion_step():
    """
    Promote safe replay-derived URL candidates into the classifier training CSV.

    NOTE: This updates the local training dataset but does not modify production DB state.
    """
    logger.info("=" * 70)
    logger.info("CLASSIFIER TRAINING PROMOTION STEP")
    logger.info("Promote safe replay-derived URL candidates into training CSV")
    logger.info("=" * 70)
    if not _scorecard_guardrails_allow("classifier_training_promotion"):
        logger.warning("classifier_training_promotion_step: Guardrails failed; skipping promotion step")
        return True
    if not _scorecard_has_required_evaluation_scope("classifier_training_promotion"):
        logger.warning("classifier_training_promotion_step: Dev/holdout evaluation incomplete; skipping promotion step")
        return True
    if not _scorecard_evaluation_deltas_allow("classifier_training_promotion"):
        logger.warning("classifier_training_promotion_step: Dev/holdout comparisons regressed; skipping promotion step")
        return True

    promotion_result = run_classifier_training_promotion()
    logger.info(
        "classifier_training_promotion_step: run_classifier_training_promotion returned: %s",
        promotion_result,
    )
    promoted_count = int(((promotion_result or {}).get("payload") or {}).get("promoted_count") or 0)
    if promoted_count > 0:
        logger.info(
            "classifier_training_promotion_step: %s rows promoted; retraining page-classifier ML artifact",
            promoted_count,
        )
        training_result = run_page_classifier_model_training()
        logger.info(
            "classifier_training_promotion_step: run_page_classifier_model_training returned: %s",
            training_result,
        )
    else:
        logger.info(
            "classifier_training_promotion_step: No rows promoted; skipping page-classifier ML retraining"
        )
    logger.info("classifier_training_promotion_step: Step completed")
    return True


def _parse_json_subprocess_stdout(stdout: str) -> dict:
    """Return parsed JSON from subprocess stdout when available."""
    if not stdout or not str(stdout).strip():
        return {}
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}

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
            stderr_text = e.stderr or ""
            if _is_transient_database_error(stderr_text):
                logger.warning(
                    f"def run_command_with_retry(): Attempt {attempt} failed due to transient DB error. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"def run_command_with_retry(): Command failed on attempt {attempt}: {e.stderr}")
                raise e
        except subprocess.TimeoutExpired as te:
            logger.error(f"def run_command_with_retry(): Command timed out on attempt {attempt}: {te}")
            raise te
    raise Exception(f"Command failed after {attempts} attempts due to persistent transient DB errors.")

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
    _log_copy_dev_to_prod_evaluation_warnings()

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


@flow(name="Refresh Manual Coverage Audit Step")
def manual_coverage_audit_refresh_step():
    """Refresh the manual coverage audit CSV from a random sample of current events."""
    result = refresh_manual_coverage_audit_csv(sample_size=100)
    logger.info("manual_coverage_audit_refresh_step(): result=%s", result)
    return True

# ------------------------
# STUB FOR TEXT MESSAGING
# ------------------------
@task
def send_text_message(message: str):
    logger.info(f"def send_text_message(): Sending text message: {message}")
    # TODO: Integrate with an SMS API like Twilio


# ------------------------
# TASKS FOR LLM MODEL VALIDATION STEP
# ------------------------
@task
def run_validate_llm_models():
    """
    Resolve provider model names at startup so model deprecations fail fast.
    """
    try:
        from llm import LLMHandler
        handler = LLMHandler(config_path='config/config.yaml')
        resolved = handler.resolve_models_for_startup()
        logger.info("run_validate_llm_models(): Resolved models: %s", resolved)
        unresolved = {k: v for k, v in resolved.items() if str(v).startswith("unresolved:")}
        if unresolved:
            logger.warning("run_validate_llm_models(): Some providers unresolved: %s", unresolved)
        return resolved
    except Exception as e:
        logger.error("run_validate_llm_models(): failed: %s", e)
        raise


@flow(name="Validate LLM Models Step")
def validate_llm_models_step():
    original_config = backup_and_update_config("validate_llm_models", updates=COMMON_CONFIG_UPDATES)
    write_run_config.submit("validate_llm_models", original_config)
    try:
        run_validate_llm_models()
        return True
    finally:
        restore_config(original_config, "validate_llm_models")


# ------------------------
# TASKS FOR RENDER -> LOCAL CHATBOT METRICS SYNC
# ------------------------
@task
def sync_render_chatbot_metrics_last_90_days():
    """
    Pull chatbot metrics from Render production DB and upsert into local DB for last 90 days.
    This keeps local reporting close to production behavior while remaining idempotent.
    """
    _append_chatbot_sync_log("sync_started")

    # This sync is intended for local runs. On Render there is no local target DB.
    if os.getenv("RENDER", "").lower() == "true":
        logger.info("sync_render_chatbot_metrics_last_90_days(): Running on Render; skipping local sync.")
        _append_chatbot_sync_log(
            "sync_skipped",
            {"reason": "render_environment"},
            level="WARNING",
        )
        return {
            "status": "skipped",
            "reason": "render_environment",
        }

    from db_config import get_production_database_url

    local_db_url = os.getenv("DATABASE_CONNECTION_STRING")
    if not local_db_url:
        logger.warning("sync_render_chatbot_metrics_last_90_days(): DATABASE_CONNECTION_STRING not set; skipping.")
        _append_chatbot_sync_log(
            "sync_skipped",
            {"reason": "missing_local_database_connection_string"},
            level="WARNING",
        )
        return {
            "status": "skipped",
            "reason": "missing_local_database_connection_string",
        }

    render_db_url = get_production_database_url()
    if not render_db_url:
        logger.warning("sync_render_chatbot_metrics_last_90_days(): Render production DB URL unavailable; skipping.")
        _append_chatbot_sync_log(
            "sync_skipped",
            {"reason": "missing_render_database_url"},
            level="WARNING",
        )
        return {
            "status": "skipped",
            "reason": "missing_render_database_url",
        }

    window_days = 90
    start_ts = datetime.datetime.utcnow() - datetime.timedelta(days=window_days)
    logger.info(
        "sync_render_chatbot_metrics_last_90_days(): Syncing from Render -> local for rows since %s (UTC).",
        start_ts.isoformat(timespec="seconds"),
    )
    _append_chatbot_sync_log(
        "sync_config",
        {
            "window_days": window_days,
            "start_ts_utc": start_ts.isoformat(timespec="seconds"),
            "source_target": safe_db_target_label(render_db_url),
            "local_target": safe_db_target_label(local_db_url),
        },
    )
    source_engine = None
    target_engine = None

    select_requests_sql = text(
        """
        SELECT
            request_id, endpoint, session_suffix, started_at, finished_at, duration_ms,
            result_type, user_input, sql_snippet, has_response, created_at, updated_at
        FROM chatbot_request_metrics
        WHERE started_at >= :start_ts
        ORDER BY started_at ASC
        """
    )
    select_stages_sql = text(
        """
        SELECT
            request_id, endpoint, stage, started_at, finished_at, duration_ms, metadata_json, created_at
        FROM chatbot_stage_metrics
        WHERE started_at >= :start_ts
        ORDER BY started_at ASC
        """
    )

    upsert_request_sql = text(
        """
        INSERT INTO chatbot_request_metrics (
            request_id, endpoint, session_suffix, started_at, finished_at, duration_ms,
            result_type, user_input, sql_snippet, has_response, created_at, updated_at
        ) VALUES (
            :request_id, :endpoint, :session_suffix, :started_at, :finished_at, :duration_ms,
            :result_type, :user_input, :sql_snippet, :has_response, :created_at, :updated_at
        )
        ON CONFLICT (request_id)
        DO UPDATE SET
            endpoint = EXCLUDED.endpoint,
            session_suffix = COALESCE(EXCLUDED.session_suffix, chatbot_request_metrics.session_suffix),
            started_at = COALESCE(EXCLUDED.started_at, chatbot_request_metrics.started_at),
            finished_at = COALESCE(EXCLUDED.finished_at, chatbot_request_metrics.finished_at),
            duration_ms = COALESCE(EXCLUDED.duration_ms, chatbot_request_metrics.duration_ms),
            result_type = COALESCE(EXCLUDED.result_type, chatbot_request_metrics.result_type),
            user_input = COALESCE(EXCLUDED.user_input, chatbot_request_metrics.user_input),
            sql_snippet = COALESCE(EXCLUDED.sql_snippet, chatbot_request_metrics.sql_snippet),
            has_response = COALESCE(EXCLUDED.has_response, chatbot_request_metrics.has_response),
            created_at = COALESCE(chatbot_request_metrics.created_at, EXCLUDED.created_at),
            updated_at = COALESCE(EXCLUDED.updated_at, chatbot_request_metrics.updated_at)
        """
    )
    upsert_stage_sql = text(
        """
        INSERT INTO chatbot_stage_metrics (
            request_id, endpoint, stage, started_at, finished_at, duration_ms, metadata_json, created_at
        ) VALUES (
            :request_id, :endpoint, :stage, :started_at, :finished_at, :duration_ms, :metadata_json, :created_at
        )
        ON CONFLICT (request_id, endpoint, stage, started_at, duration_ms)
        DO UPDATE SET
            finished_at = COALESCE(EXCLUDED.finished_at, chatbot_stage_metrics.finished_at),
            metadata_json = COALESCE(EXCLUDED.metadata_json, chatbot_stage_metrics.metadata_json),
            created_at = COALESCE(chatbot_stage_metrics.created_at, EXCLUDED.created_at)
        """
    )

    try:
        source_engine = create_engine(render_db_url)
        target_engine = create_engine(local_db_url)
        ensure_chatbot_metrics_schema(target_engine)

        requests_df = pd.read_sql(select_requests_sql, source_engine, params={"start_ts": start_ts})
        stages_df = pd.read_sql(select_stages_sql, source_engine, params={"start_ts": start_ts})
        _append_chatbot_sync_log(
            "source_rows_fetched",
            {
                "requests_rows": int(len(requests_df)),
                "stages_rows": int(len(stages_df)),
                "request_nullish_datetime_counts": count_nullish_datetime_values(
                    requests_df,
                    ["started_at", "finished_at", "created_at", "updated_at"],
                ),
                "stage_nullish_datetime_counts": count_nullish_datetime_values(
                    stages_df,
                    ["started_at", "finished_at", "created_at"],
                ),
            },
        )

        request_records = sanitize_records_for_sql(requests_df)
        stage_records = sanitize_records_for_sql(stages_df)
        _append_chatbot_sync_log(
            "records_sanitized",
            {
                "requests_records": int(len(request_records)),
                "stages_records": int(len(stage_records)),
            },
        )

        with target_engine.begin() as target_conn:
            if request_records:
                target_conn.execute(upsert_request_sql, request_records)
            if stage_records:
                target_conn.execute(upsert_stage_sql, stage_records)
            request_window_count = int(
                target_conn.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM chatbot_request_metrics
                        WHERE started_at >= :start_ts
                        """
                    ),
                    {"start_ts": start_ts},
                ).scalar()
                or 0
            )
            stage_window_count = int(
                target_conn.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM chatbot_stage_metrics
                        WHERE started_at >= :start_ts
                        """
                    ),
                    {"start_ts": start_ts},
                ).scalar()
                or 0
            )
            request_total_count = int(
                target_conn.execute(
                    text("SELECT COUNT(*) FROM chatbot_request_metrics")
                ).scalar()
                or 0
            )
            stage_total_count = int(
                target_conn.execute(
                    text("SELECT COUNT(*) FROM chatbot_stage_metrics")
                ).scalar()
                or 0
            )

        summary = {
            "status": "ok",
            "window_days": window_days,
            "start_ts_utc": start_ts.isoformat(timespec="seconds"),
            "requests_fetched": len(request_records),
            "stages_fetched": len(stage_records),
            "local_post_sync_counts": {
                "chatbot_request_metrics": {
                    "window_90d": request_window_count,
                    "total": request_total_count,
                },
                "chatbot_stage_metrics": {
                    "window_90d": stage_window_count,
                    "total": stage_total_count,
                },
            },
        }
        logger.info("sync_render_chatbot_metrics_last_90_days(): %s", summary)
        _append_chatbot_sync_log("sync_completed", summary)
        return summary
    except Exception as e:
        import traceback

        error_payload = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        logger.exception("sync_render_chatbot_metrics_last_90_days(): failed")
        _append_chatbot_sync_log("sync_failed", error_payload, level="ERROR")
        raise
    finally:
        if source_engine is not None:
            source_engine.dispose()
        if target_engine is not None:
            target_engine.dispose()


@flow(name="Sync Render Chatbot Metrics Step")
def sync_render_chatbot_metrics_step():
    """Final pipeline step: pull last 90 days of chatbot metrics from Render into local DB."""
    logger.info("=" * 70)
    logger.info("SYNC RENDER CHATBOT METRICS STEP")
    logger.info("Upsert last 90 days of chatbot metrics from Render production DB into local DB")
    logger.info("=" * 70)
    result = sync_render_chatbot_metrics_last_90_days()
    logger.info("sync_render_chatbot_metrics_step: result=%s", result)
    return True

# ------------------------
# PIPELINE EXECUTION
# ------------------------
PIPELINE_STEPS = [
    ("copy_log_files", copy_log_files),
    ("database_accuracy_manual_review_gate", database_accuracy_manual_review_gate_step),
    ("credential_validation", credential_validation_step),
    ("validate_llm_models", validate_llm_models_step),
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
    ("classifier_training_promotion", classifier_training_promotion_step),  # Promote safe replay candidates into training data
    ("copy_dev_to_prod", copy_dev_db_to_prod_db_step),
    ("refresh_manual_coverage_audit", manual_coverage_audit_refresh_step),
    ("sync_render_chatbot_metrics", sync_render_chatbot_metrics_step),
]

def list_available_steps():
    print("Available steps:")
    for i, (name, _) in enumerate(PIPELINE_STEPS, start=1):
        print(f" {i}. {name}")

def run_pipeline(start_step: str, end_step: str = None, parallel_crawlers: bool = False):
    run_id = os.getenv("DS_RUN_ID")
    if not run_id:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        os.environ["DS_RUN_ID"] = run_id
    os.environ["DS_STEP_NAME"] = "pipeline"
    setup_logging('pipeline')
    logger.info("run_pipeline(): initialized run context run_id=%s", run_id)

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
            and name == "rd_ext"
            and PARALLEL_CRAWLER_STEPS.issubset(set(selected_names))
        ):
            print("Running parallel crawler group: rd_ext.py + ebs.py + scraper.py + fb.py + images.py")
            retry_count = 0
            while retry_count < 3:
                try:
                    parallel_crawlers_step()
                    skip_steps.update(PARALLEL_CRAWLER_STEPS - {"rd_ext"})
                    break
                except Exception as e:
                    if _is_transient_database_error(str(e)):
                        logger.error(
                            "parallel crawler group encountered transient DB error, retrying in 5 seconds. Attempt %d of 3.",
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
                logger.error("Parallel crawler group failed after 3 retries due to transient DB errors.")
                sys.exit(1)
            continue

        print(f"Running step: {name}")
        retry_count = 0
        while retry_count < 3:
            try:
                step_flow()
                break
            except Exception as e:
                if _is_transient_database_error(str(e)):
                    logger.error(f"Step {name} encountered a transient DB error, retrying in 5 seconds. Attempt {retry_count+1} of 3.")
                    time.sleep(5)
                    retry_count += 1
                else:
                    # Log error with full details for troubleshooting
                    import traceback
                    logger.error(f"❌ Step '{name}' failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    sys.exit(1)
        else:
            logger.error(f"Step {name} failed after 3 retries due to transient DB errors.")
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
    selected_steps = [name for name, _ in PIPELINE_STEPS[start_index:end_index + 1]]
    can_run_parallel = all(step in selected_steps for step in PARALLEL_CRAWLER_STEPS)
    parallel_crawlers = False
    if can_run_parallel:
        parallel_answer = input(
            "Run rd_ext/ebs/scraper/fb/images in parallel? (y/N): "
        ).strip().lower()
        parallel_crawlers = parallel_answer in {"y", "yes"}
        print(f"Parallel crawlers: {'enabled' if parallel_crawlers else 'disabled'}")
    run_pipeline(start, end, parallel_crawlers=parallel_crawlers)

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

"""
db.py
This module provides a DatabaseHandler class for managing database connections and operations.
It includes methods for creating tables, writing URLs and events to the database, and handling address deduplication.
It also includes methods for loading blacklisted domains and checking URLs against them.
It uses SQLAlchemy for database interactions and pandas for data manipulation.
It supports both local and Render environments, with configurations loaded from environment variables.
It also includes methods for fuzzy matching addresses and deduplicating the address table.
                    WHERE address_id = :old_id
"""

from datetime import date, datetime, timedelta
from collections import Counter
import csv
from dotenv import load_dotenv
load_dotenv()
from rapidfuzz import fuzz
import json
import logging
import numpy as np
import os
import pandas as pd
from rapidfuzz.fuzz import ratio
import re  # Added missing import
import requests
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import PendingRollbackError, SQLAlchemyError
from typing import Final, Optional, List, Dict, Any
from urllib.parse import urlparse
import sys
import yaml
import warnings

from config_runtime import load_config
# Import database configuration utility
from db_config import get_database_config
from evaluation_holdout import is_holdout_url
from output_paths import duplicates_path, events_path
from page_classifier import classify_page

_US_STATE_OR_TERRITORY_CODES: Final[frozenset[str]] = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC", "PR", "VI", "GU", "AS", "MP",
    }
)

CHATBOT_METRICS_SCHEMA_QUERIES: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS chatbot_request_metrics (
        id SERIAL PRIMARY KEY,
        request_id TEXT UNIQUE NOT NULL,
        endpoint TEXT NOT NULL,
        session_suffix TEXT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        duration_ms DOUBLE PRECISION,
        result_type TEXT,
        user_input TEXT,
        sql_snippet TEXT,
        has_response BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chatbot_stage_metrics (
        id SERIAL PRIMARY KEY,
        request_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        stage TEXT NOT NULL,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        duration_ms DOUBLE PRECISION,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chatbot_request_metrics_started_at ON chatbot_request_metrics(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_chatbot_request_metrics_endpoint ON chatbot_request_metrics(endpoint)",
    "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_started_at ON chatbot_stage_metrics(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_stage ON chatbot_stage_metrics(stage)",
    "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_request_id ON chatbot_stage_metrics(request_id)",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_chatbot_stage_metrics_nk "
        "ON chatbot_stage_metrics(request_id, endpoint, stage, started_at, duration_ms)"
    ),
)

EVENTS_TABLE_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    event_name TEXT,
    dance_style TEXT,
    description TEXT,
    day_of_week TEXT,
    start_date DATE,
    end_date DATE,
    start_time TIME,
    end_time TIME,
    source TEXT,
    location TEXT,
    price TEXT,
    url TEXT,
    event_type TEXT,
    address_id INTEGER,
    time_stamp TIMESTAMP
)
"""

EVENTS_HISTORY_TABLE_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS events_history (
    event_id SERIAL PRIMARY KEY,
    original_event_id INTEGER,
    event_name TEXT,
    dance_style TEXT,
    description TEXT,
    day_of_week TEXT,
    start_date DATE,
    end_date DATE,
    start_time TIME,
    end_time TIME,
    source TEXT,
    location TEXT,
    price TEXT,
    url TEXT,
    event_type TEXT,
    address_id INTEGER,
    time_stamp TIMESTAMP
)
"""

ADDRESS_TABLE_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS address (
    address_id SERIAL PRIMARY KEY,
    full_address TEXT UNIQUE,
    building_name TEXT,
    street_number TEXT,
    street_name TEXT,
    street_type TEXT,
    direction TEXT,
    city TEXT,
    met_area TEXT,
    province_or_state TEXT,
    postal_code TEXT,
    country_id TEXT,
    time_stamp TIMESTAMP
)
"""

URLS_TABLE_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS urls (
    link_id SERIAL PRIMARY KEY,
    link TEXT,
    parent_url TEXT,
    source TEXT,
    keywords TEXT,
    relevant BOOLEAN,
    crawl_try INTEGER,
    time_stamp TIMESTAMP
)
"""

RUNS_TABLE_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS runs (
    run_id SERIAL PRIMARY KEY,
    run_name TEXT UNIQUE,
    run_description TEXT,
    start_time TEXT,
    end_time TEXT,
    elapsed_time TEXT,
    python_file_name TEXT,
    unique_urls_count INTEGER,
    total_url_attempts INTEGER,
    urls_with_extracted_text INTEGER,
    urls_with_found_keywords INTEGER,
    events_written_to_db INTEGER,
    time_stamp TIMESTAMP
)
"""


def ensure_chatbot_metrics_schema(engine) -> None:
    """Ensure chatbot metrics schema exists using a SQLAlchemy engine."""
    with engine.begin() as conn:
        for query in CHATBOT_METRICS_SCHEMA_QUERIES:
            conn.execute(text(query))


class DatabaseHandler():
    _DELETE_REASON_CODE_MAP: Final[Dict[str, str]] = {
        "orphaned_address_id_reference": "orphaned_address_id_reference",
        "exact_time_window_duplicate": "exact_time_window_duplicate",
        "fuzzy_duplicate_merged": "fuzzy_duplicate_merged",
        "empty_source_dance_style_url_no_address": "empty_source_dance_style_url_no_address",
        "outside_bc_filter": "outside_bc_filter",
        "outside_canada_filter": "outside_canada_filter",
        "empty_dance_style_url_other_no_location_description": "empty_dance_style_url_other_no_location_description",
        "manual_delete_by_name_and_start_date": "manual_delete_by_name_and_start_date",
        "null_start_date_start_time_or_null_start_end_time": "null_start_date_start_time_or_null_start_end_time",
    }
    _SHOULD_PROCESS_MIN_HIT_RATIO = 0.5
    _SHOULD_PROCESS_MAX_RETRIES_FOR_IRRELEVANT = 2
    _LOW_VALUE_PATH_SEGMENTS = {
        "about",
        "contact",
        "faculty",
        "faq",
        "staff",
        "student-wellness",
        "team",
        "vocational-division",
        "open-division",
        "international-waitlist",
    }
    _VENUE_TOKEN_STOPWORDS = {
        "the", "and", "of", "in", "at", "on",
        "victoria", "bc", "canada", "ca",
        "venue", "club", "dance", "studio", "society", "association", "group",
        "hall", "center", "centre",
    }
    _stale_raw_location_warnings: Final[set[tuple[str, int | None, str]]] = set()
    _DANCE_STYLE_TOKENS = {
        "argentine tango",
        "tango",
        "salsa",
        "bachata",
        "kizomba",
        "semba",
        "urban kiz",
        "tarraxo",
        "tarraxa",
        "tarraxinha",
        "merengue",
        "rumba",
        "swing",
        "west coast swing",
        "wcs",
        "east coast swing",
        "lindy",
        "lindy hop",
        "balboa",
    }
    _SOURCE_PLACEHOLDER_VALUES = {
        "",
        "unknown",
        "unknown source",
        "none",
        "null",
        "n/a",
        "na",
        "source",
        "extracted text",
        "text extracted",
        "extracted_text",
    }
    _MAX_AUTOFILL_KEYWORD_STYLES = 4

    def __init__(self, config):
        """
        Initializes the DatabaseHandler instance with the provided configuration.

        This constructor sets up the database connections based on the environment (Render or local),
        loads the blacklist domains, initializes SQLAlchemy metadata, retrieves the Google API key,
        and prepares DataFrames for URL analysis.

            config (dict): Configuration parameters for the database connection.

        Raises:
            ConnectionError: If the database connection could not be established.

        Side Effects:
            - Loads blacklist domains.
            - Establishes connections to the main and address databases.
            - Reflects the existing database schema into SQLAlchemy metadata.
            - Retrieves the Google API key from environment variables.
            - Creates a DataFrame from the URLs table and computes grouped statistics for URL usefulness.
        """
        self.config = config
        self.event_overrides = self._load_event_overrides()
        self.address_aliases = self._load_address_aliases()
        self.load_blacklist_domains()
        # Pre-load whitelist entries and normalize them for robust checks
        try:
            self._whitelist_set = set()
            whitelist_path = os.path.join(self.config['input']['urls'], 'aaa_urls.csv')
            if os.path.exists(whitelist_path):
                df = pd.read_csv(whitelist_path)
                if 'link' in df.columns:
                    links = [str(u).strip() for u in df['link'].dropna().tolist()]
                    self._whitelist_set = {self._normalize_for_compare(u) for u in links}
                    logging.info(f"Loaded whitelist with {len(self._whitelist_set)} entries from {whitelist_path}")
        except Exception as e:
            logging.warning(f"Failed to preload whitelist: {e}")

        # Get database configuration using centralized utility
        # This automatically handles local, render_dev, and render_prod environments
        connection_string, env_name = get_database_config()
        self.conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
        logging.info(f"def __init__(): Database connection established: {env_name}")

        # Note: The 'locations' table (Canadian postal code database) is now part of social_dance_db
        # Previously this was in a separate address_db, but has been consolidated for simplicity

        if self.conn is None:
                raise ConnectionError("def __init__(): DatabaseHandler: Failed to establish a database connection.")

        self.ensure_core_application_tables()

        self.metadata = MetaData()
        # Reflect the existing database schema into metadata
        self.metadata.reflect(bind=self.conn)
        self.ensure_urls_decision_reason_column()
        self.ensure_url_scrape_metrics_table()
        self.ensure_event_attribution_tables()
        self.ensure_validation_metric_tables()
        self.ensure_source_distribution_history_tables()
        self.ensure_fb_block_triage_table()
        self.ensure_fb_block_occurrences_table()

        # Get google api key
        self.google_api_key = os.getenv("GOOGLE_KEY_PW")

        # Create df from urls table (only if not on production - production doesn't have urls table)
        from db_config import is_production_target
        if not is_production_target():
            self.urls_df = self.create_urls_df()
            logging.info("__init__(): URLs DataFrame created with %d rows.", len(self.urls_df))
        else:
            self.urls_df = pd.DataFrame()
            logging.info("__init__(): Skipping URLs table load on production (not needed for web service)")

        def _compute_hit_ratio(x):
            true_count = x.sum()
            false_count = (~x).sum()

            # Case 1: at least one True and one False
            if true_count > 0 and false_count > 0:
                return true_count / false_count

            # Case 2: 0 Trues and all False
            if true_count == 0:
                return 0.0

            # Case 3: all Trues and no False
            return 1.0

        # Create a groupby that gives a hit_ratio and a sum of crawl_try for how useful the URL is
        if not is_production_target():
            self.urls_gb = (
                self.urls_df
                .groupby('link')
                .agg(
                    hit_ratio=('relevant', _compute_hit_ratio),
                    crawl_try=('crawl_try', 'sum')
                )
                .reset_index()
            )
            logging.info(f"__init__(): urls_gb has {len(self.urls_gb)} rows and {len(self.urls_gb.columns)} columns.")
            # Create raw_locations table for caching location strings (only needed for pipeline)
            self.create_raw_locations_table()
        else:
            self.urls_gb = pd.DataFrame()
            logging.info("__init__(): Skipping urls_gb and raw_locations table creation on production")
        self._should_process_decision_counters: Counter = Counter()
        self._last_should_process_reason_by_url: Dict[str, str] = {}


    def set_llm_handler(self, llm_handler):
        """
        Inject an instance of LLMHandler after both classes are constructed.
        """
        self.llm_handler = llm_handler

    def ensure_urls_decision_reason_column(self) -> None:
        """
        Ensure urls.decision_reason exists so skip/process decisions can be audited.
        """
        try:
            self.execute_query("ALTER TABLE urls ADD COLUMN IF NOT EXISTS decision_reason TEXT")
            logging.info("ensure_urls_decision_reason_column: ensured urls.decision_reason exists")
        except Exception as e:
            logging.warning("ensure_urls_decision_reason_column: could not ensure column (continuing): %s", e)

    def ensure_url_scrape_metrics_table(self) -> None:
        """Ensure per-URL scrape telemetry table exists for run-over-run diagnostics."""
        query = """
            CREATE TABLE IF NOT EXISTS url_scrape_metrics (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                step_name TEXT,
                link TEXT,
                parent_url TEXT,
                source TEXT,
                keywords TEXT,
                archetype TEXT,
                extraction_attempted BOOLEAN,
                extraction_succeeded BOOLEAN,
                extraction_skipped BOOLEAN,
                decision_reason TEXT,
                handled_by TEXT,
                routing_reason TEXT,
                classification_confidence DOUBLE PRECISION,
                classification_stage TEXT,
                classification_owner_step TEXT,
                classification_subtype TEXT,
                classification_features_json TEXT,
                access_attempted BOOLEAN,
                access_succeeded BOOLEAN,
                text_extracted BOOLEAN,
                keywords_found BOOLEAN,
                events_written INTEGER,
                ocr_attempted BOOLEAN,
                ocr_succeeded BOOLEAN,
                vision_attempted BOOLEAN,
                vision_succeeded BOOLEAN,
                fallback_used BOOLEAN,
                links_discovered INTEGER,
                links_followed INTEGER,
                time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        try:
            self.execute_query(query)
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS handled_by TEXT")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS routing_reason TEXT")
            self.execute_query(
                "ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS classification_confidence DOUBLE PRECISION"
            )
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS classification_stage TEXT")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS classification_owner_step TEXT")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS classification_subtype TEXT")
            self.execute_query(
                "ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS classification_features_json TEXT"
            )
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS access_attempted BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS access_succeeded BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS text_extracted BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS keywords_found BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS events_written INTEGER")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS ocr_attempted BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS ocr_succeeded BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS vision_attempted BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS vision_succeeded BOOLEAN")
            self.execute_query("ALTER TABLE url_scrape_metrics ADD COLUMN IF NOT EXISTS fallback_used BOOLEAN")
            self.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_url_scrape_metrics_run_id ON url_scrape_metrics(run_id)"
            )
            self.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_url_scrape_metrics_link ON url_scrape_metrics(link)"
            )
            self.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_url_scrape_metrics_time_stamp ON url_scrape_metrics(time_stamp)"
            )
            logging.info("ensure_url_scrape_metrics_table: ensured url_scrape_metrics exists")
        except Exception as e:
            logging.warning("ensure_url_scrape_metrics_table: could not ensure table (continuing): %s", e)

    def ensure_event_attribution_tables(self) -> None:
        """Ensure canonical event write/delete attribution tables exist."""
        create_queries = [
            """
            CREATE TABLE IF NOT EXISTS event_write_attribution (
                attribution_id SERIAL PRIMARY KEY,
                event_id INTEGER NOT NULL UNIQUE,
                run_id TEXT,
                step_name TEXT,
                url TEXT,
                parent_url TEXT,
                source TEXT,
                write_method TEXT,
                provider TEXT,
                model TEXT,
                prompt_type TEXT,
                decision_reason TEXT,
                details_json JSONB,
                time_stamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS event_delete_attribution (
                attribution_id SERIAL PRIMARY KEY,
                event_id INTEGER NOT NULL UNIQUE,
                run_id TEXT,
                step_name TEXT,
                delete_reason_code TEXT NOT NULL,
                raw_delete_reason TEXT,
                delete_method TEXT,
                source_url TEXT,
                created_by_step TEXT,
                reason_registered BOOLEAN NOT NULL DEFAULT TRUE,
                details_json JSONB,
                time_stamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_event_write_attribution_run_id ON event_write_attribution(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_write_attribution_step_name ON event_write_attribution(step_name)",
            "CREATE INDEX IF NOT EXISTS idx_event_write_attribution_url ON event_write_attribution(url)",
            "CREATE INDEX IF NOT EXISTS idx_event_delete_attribution_run_id ON event_delete_attribution(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_event_delete_attribution_step_name ON event_delete_attribution(step_name)",
            "CREATE INDEX IF NOT EXISTS idx_event_delete_attribution_reason_code ON event_delete_attribution(delete_reason_code)",
        ]
        try:
            for query in create_queries:
                self.execute_query(query)
            logging.info("ensure_event_attribution_tables: ensured canonical event attribution tables exist")
        except Exception as e:
            logging.warning("ensure_event_attribution_tables: could not ensure tables (continuing): %s", e)

    def ensure_validation_metric_tables(self) -> None:
        """
        Ensure normalized validation metric tables exist.

        Tables:
            - metric_definitions: stable metric registry (one row per metric key)
            - metric_observations: numeric observations over time/run windows
            - accuracy_replay_results: row-level replay audit records
            - classifier_training_url_candidates: URL-level replay aggregation for training curation
            - validation_run_artifacts: JSON run artifacts keyed by run_id + artifact_type
        """
        create_metric_definitions = """
            CREATE TABLE IF NOT EXISTS metric_definitions (
                metric_id SERIAL PRIMARY KEY,
                metric_key TEXT UNIQUE NOT NULL,
                metric_unit TEXT,
                description TEXT,
                higher_is_better BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        create_metric_observations = """
            CREATE TABLE IF NOT EXISTS metric_observations (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                metric_id INTEGER NOT NULL REFERENCES metric_definitions(metric_id),
                metric_value_numeric DOUBLE PRECISION NOT NULL,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                notes_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        create_accuracy_replay_results = """
            CREATE TABLE IF NOT EXISTS accuracy_replay_results (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                baseline_event_id INTEGER,
                baseline_url TEXT,
                baseline_snapshot_json TEXT,
                replay_snapshot_json TEXT,
                is_match BOOLEAN NOT NULL,
                mismatch_category TEXT,
                mismatch_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        create_classifier_training_url_candidates = """
            CREATE TABLE IF NOT EXISTS classifier_training_url_candidates (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                query_text TEXT,
                normalized_url TEXT NOT NULL,
                domain TEXT,
                total_rows INTEGER NOT NULL,
                true_count INTEGER NOT NULL,
                false_count INTEGER NOT NULL,
                match_rate_pct DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL,
                recommended_action TEXT,
                training_eligible BOOLEAN NOT NULL DEFAULT FALSE,
                recommended_archetype TEXT,
                recommended_owner_step TEXT,
                recommended_subtype TEXT,
                priority_score INTEGER NOT NULL DEFAULT 0,
                mismatch_category_counts_json TEXT,
                baseline_event_ids_json TEXT,
                sample_baseline_json TEXT,
                sample_replay_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        create_validation_run_artifacts = """
            CREATE TABLE IF NOT EXISTS validation_run_artifacts (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (run_id, artifact_type)
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_metric_observations_run_id ON metric_observations(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_metric_observations_metric_id ON metric_observations(metric_id)",
            "CREATE INDEX IF NOT EXISTS idx_metric_observations_created_at ON metric_observations(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_replay_results_run_id ON accuracy_replay_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_replay_results_is_match ON accuracy_replay_results(is_match)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_replay_results_category ON accuracy_replay_results(mismatch_category)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_replay_results_event_id ON accuracy_replay_results(baseline_event_id)",
            "CREATE INDEX IF NOT EXISTS idx_classifier_training_candidates_run_id ON classifier_training_url_candidates(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_classifier_training_candidates_status ON classifier_training_url_candidates(status)",
            "CREATE INDEX IF NOT EXISTS idx_classifier_training_candidates_url ON classifier_training_url_candidates(normalized_url)",
            "CREATE INDEX IF NOT EXISTS idx_validation_run_artifacts_run_id ON validation_run_artifacts(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_validation_run_artifacts_type ON validation_run_artifacts(artifact_type)",
        ]
        try:
            self.execute_query(create_metric_definitions)
            self.execute_query(create_metric_observations)
            self.execute_query(create_accuracy_replay_results)
            self.execute_query(create_classifier_training_url_candidates)
            self.execute_query(create_validation_run_artifacts)
            for index_sql in indexes:
                self.execute_query(index_sql)
            logging.info("ensure_validation_metric_tables: ensured normalized validation metric tables exist")
        except Exception as e:
            logging.warning("ensure_validation_metric_tables: could not ensure tables (continuing): %s", e)

    def ensure_source_distribution_history_tables(self) -> None:
        """Ensure source-distribution trend history tables exist for validation reporting."""
        create_counts = """
            CREATE TABLE IF NOT EXISTS source_event_counts_history (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                rank_in_run INTEGER,
                total_events INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, source)
            )
        """
        create_alerts = """
            CREATE TABLE IF NOT EXISTS source_distribution_alerts_history (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                current_count INTEGER,
                baseline_avg DOUBLE PRECISION,
                pct_change DOUBLE PRECISION,
                details_json TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_source_event_counts_history_run_ts ON source_event_counts_history(run_ts)",
            "CREATE INDEX IF NOT EXISTS idx_source_event_counts_history_source ON source_event_counts_history(source)",
            "CREATE INDEX IF NOT EXISTS idx_source_event_counts_history_run_id ON source_event_counts_history(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_source_distribution_alerts_history_run_ts ON source_distribution_alerts_history(run_ts)",
            "CREATE INDEX IF NOT EXISTS idx_source_distribution_alerts_history_source ON source_distribution_alerts_history(source)",
            "CREATE INDEX IF NOT EXISTS idx_source_distribution_alerts_history_run_id ON source_distribution_alerts_history(run_id)",
        ]
        try:
            self.execute_query(create_counts)
            self.execute_query(create_alerts)
            for index_sql in indexes:
                self.execute_query(index_sql)
            logging.info(
                "ensure_source_distribution_history_tables: ensured source distribution history tables exist"
            )
        except Exception as e:
            logging.warning(
                "ensure_source_distribution_history_tables: could not ensure tables (continuing): %s",
                e,
            )

    def ensure_fb_block_triage_table(self) -> None:
        """Ensure aggregated Facebook blocked-content triage facts exist for run-over-run analysis."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS fb_block_triage (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                blocked_reason TEXT NOT NULL,
                source_key TEXT NOT NULL,
                block_category TEXT NOT NULL,
                unique_url_count INTEGER NOT NULL,
                blocked_attempt_count INTEGER NOT NULL,
                sample_url TEXT,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (run_id, blocked_reason, source_key, block_category)
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_fb_block_triage_run_id ON fb_block_triage(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_triage_reason ON fb_block_triage(blocked_reason)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_triage_category ON fb_block_triage(block_category)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_triage_source_key ON fb_block_triage(source_key)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_triage_created_at ON fb_block_triage(created_at)",
        ]
        try:
            self.execute_query(create_table_sql)
            for index_sql in indexes:
                self.execute_query(index_sql)
            logging.info("ensure_fb_block_triage_table: ensured fb_block_triage exists")
        except Exception as e:
            logging.warning("ensure_fb_block_triage_table: could not ensure table (continuing): %s", e)

    def ensure_fb_block_occurrences_table(self) -> None:
        """Ensure raw Facebook blocked URL occurrences exist for detailed troubleshooting."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS fb_block_occurrences (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                blocked_reason TEXT NOT NULL,
                requested_url TEXT NOT NULL,
                source_key TEXT NOT NULL,
                block_category TEXT NOT NULL,
                occurrence_ts TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (run_id, blocked_reason, requested_url, source_key, block_category, occurrence_ts)
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_fb_block_occurrences_run_id ON fb_block_occurrences(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_occurrences_reason ON fb_block_occurrences(blocked_reason)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_occurrences_source_key ON fb_block_occurrences(source_key)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_occurrences_category ON fb_block_occurrences(block_category)",
            "CREATE INDEX IF NOT EXISTS idx_fb_block_occurrences_occurrence_ts ON fb_block_occurrences(occurrence_ts)",
        ]
        try:
            self.execute_query(create_table_sql)
            for index_sql in indexes:
                self.execute_query(index_sql)
            logging.info("ensure_fb_block_occurrences_table: ensured fb_block_occurrences exists")
        except Exception as e:
            logging.warning("ensure_fb_block_occurrences_table: could not ensure table (continuing): %s", e)
        

    def load_blacklist_domains(self):
        """
        Loads a set of blacklisted domains from a CSV file specified in the configuration.

        The CSV file path is retrieved from self.config['constants']['black_list_domains'].
        The CSV is expected to have a column named 'Domain'. All domain names are converted
        to lowercase and stripped of whitespace before being added to the blacklist set.

        The resulting set is stored in self.blacklisted_domains.

        Logs the number of loaded blacklisted domains at the INFO level.
        """
        csv_path = self.config['constants']['black_list_domains']
        df = pd.read_csv(csv_path)
        self.blacklisted_domains = set(df['Domain'].str.lower().str.strip())
        logging.info(f"Loaded {len(self.blacklisted_domains)} blacklisted domains.")

    def avoid_domains(self, url):
        """
        Check if the given URL contains any blacklisted domain.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL contains any domain from the blacklist, False otherwise.

        Note:
            The check is case-insensitive.
        """
        """ Check if URL contains any blacklisted domain. """
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.blacklisted_domains)
    


    def _normalize_for_compare(self, url: str) -> str:
        try:
            from urllib.parse import urlparse, urlunparse
            p = urlparse(url)
            scheme = (p.scheme or 'https').lower()
            netloc = (p.netloc or '').lower()
            if netloc.endswith(':80') and scheme == 'http':
                netloc = netloc[:-3]
            if netloc.endswith(':443') and scheme == 'https':
                netloc = netloc[:-4]
            path = (p.path or '').rstrip('/')
            return urlunparse((scheme, netloc, path, '', p.query, ''))
        except Exception:
            return (url or '').strip().lower().rstrip('/')

    def is_whitelisted_url(self, url: str) -> bool:
        try:
            if not hasattr(self, '_whitelist_set') or not self._whitelist_set:
                return False
            u_norm = self._normalize_for_compare(url)
            if u_norm in self._whitelist_set:
                return True
            from urllib.parse import urlparse
            pu = urlparse(u_norm)
            for w in self._whitelist_set:
                pw = urlparse(w)
                if pu.scheme == pw.scheme and pu.netloc == pw.netloc:
                    if not pw.path or pu.path.startswith(pw.path):
                        return True
            return False
        except Exception as e:
            logging.warning(f'is_whitelisted_url: error {e}')
            return False

    def _load_event_overrides(self) -> List[Dict[str, Any]]:
        """
        Load optional event normalization overrides from config.

        Expected config shape:
            normalization:
              event_overrides:
                - name: some_rule
                  match:
                    url_contains: example.com/path
                  set:
                    event_type: "social dance, live music"
        """
        try:
            overrides = (
                self.config
                .get("normalization", {})
                .get("event_overrides", [])
            )
            if not isinstance(overrides, list):
                logging.warning("_load_event_overrides: Expected list, got %s", type(overrides).__name__)
                return []
            valid_overrides = [o for o in overrides if isinstance(o, dict)]
            if len(valid_overrides) != len(overrides):
                logging.warning("_load_event_overrides: Ignoring non-dict override entries.")
            logging.info("_load_event_overrides: Loaded %d event override rule(s)", len(valid_overrides))
            return valid_overrides
        except Exception as e:
            logging.warning("_load_event_overrides: Failed to load overrides: %s", e)
            return []

    @staticmethod
    def _normalize_address_alias_text(value: str) -> str:
        """Normalize venue/address text for robust alias comparison."""
        if not value:
            return ""
        normalized = str(value).lower().strip()
        normalized = normalized.replace("&", " and ")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _load_address_aliases(self) -> List[Dict[str, Any]]:
        """
        Load optional address alias normalization rules from config.

        Expected shape:
            normalization:
              address_aliases:
                - name: some_alias_rule
                  aliases: [studio 919, the strath]
                  canonical:
                    address_id: 109
                    full_address: ...
        """
        try:
            alias_entries = (
                self.config
                .get("normalization", {})
                .get("address_aliases", [])
            )
            if not isinstance(alias_entries, list):
                logging.warning("_load_address_aliases: Expected list, got %s", type(alias_entries).__name__)
                return []

            normalized_rules: List[Dict[str, Any]] = []
            for entry in alias_entries:
                if not isinstance(entry, dict):
                    continue

                aliases = entry.get("aliases", entry.get("match_any", []))
                canonical = entry.get("canonical", {})
                match_rule = entry.get("match", {})
                if not isinstance(aliases, list) or not isinstance(canonical, dict):
                    continue
                if match_rule and not isinstance(match_rule, dict):
                    continue

                normalized_aliases = {
                    self._normalize_address_alias_text(alias)
                    for alias in aliases
                    if isinstance(alias, str) and self._normalize_address_alias_text(alias)
                }
                if not normalized_aliases or not canonical:
                    continue

                normalized_rules.append({
                    "name": entry.get("name", "unnamed_address_alias"),
                    "aliases": normalized_aliases,
                    "canonical": canonical,
                    "match": match_rule,
                })

            logging.info("_load_address_aliases: Loaded %d address alias rule(s)", len(normalized_rules))
            return normalized_rules
        except Exception as e:
            logging.warning("_load_address_aliases: Failed to load aliases: %s", e)
            return []

    @staticmethod
    def _is_specific_alias_for_substring(alias: str) -> bool:
        """Require alias specificity before allowing substring matching."""
        alias_tokens = alias.split()
        return len(alias) >= 10 or len(alias_tokens) >= 2

    @staticmethod
    def _normalize_postal_code(value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", "", str(value).upper().strip())

    @staticmethod
    def _safe_lower(value: Any) -> str:
        return str(value or "").strip().lower()

    def _address_alias_context_matches(self, match_rule: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Check optional alias rule constraints against event/address context."""
        if not match_rule:
            return True
        if not isinstance(context, dict):
            return False

        source_equals = self._safe_lower(match_rule.get("source_equals"))
        if source_equals:
            source_value = self._safe_lower(context.get("source"))
            if source_value != source_equals:
                return False

        url_contains = self._safe_lower(match_rule.get("url_contains"))
        if url_contains:
            url_value = self._safe_lower(context.get("url"))
            if url_contains not in url_value:
                return False

        return True

    def _find_address_alias_match(
        self,
        candidate_texts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return alias match details when any configured alias matches candidate text."""
        if not candidate_texts or not self.address_aliases:
            return None

        normalized_candidates = [
            self._normalize_address_alias_text(text)
            for text in candidate_texts
            if isinstance(text, str) and text.strip()
        ]
        if not normalized_candidates:
            return None

        # Pass 1: exact normalized equality only.
        for rule in self.address_aliases:
            aliases = rule.get("aliases", set())
            canonical = rule.get("canonical", {})
            match_rule = rule.get("match", {})
            if not aliases or not canonical:
                continue
            if not self._address_alias_context_matches(match_rule, context):
                continue

            for candidate in normalized_candidates:
                for alias in aliases:
                    if alias == candidate:
                        return {
                            "canonical": canonical,
                            "rule_name": rule.get("name", "unnamed_address_alias"),
                            "matched_alias": alias,
                            "candidate": candidate,
                            "match_type": "exact",
                        }

        # Pass 2: constrained substring matching.
        for rule in self.address_aliases:
            aliases = rule.get("aliases", set())
            canonical = rule.get("canonical", {})
            match_rule = rule.get("match", {})
            if not aliases or not canonical:
                continue
            if not self._address_alias_context_matches(match_rule, context):
                continue

            for candidate in normalized_candidates:
                for alias in aliases:
                    if not self._is_specific_alias_for_substring(alias):
                        continue
                    # Guard against generic candidates (for example "victoria bc")
                    # matching long canonical aliases by substring.
                    # Exact matching is already handled in pass 1 above.
                    if alias in candidate:
                        return {
                            "canonical": canonical,
                            "rule_name": rule.get("name", "unnamed_address_alias"),
                            "matched_alias": alias,
                            "candidate": candidate,
                            "match_type": "substring",
                        }

        return None

    def _audit_address_alias_hit(self, payload: Dict[str, Any]) -> None:
        """Append alias-match telemetry to CSV for traceability."""
        try:
            output_cfg = self.config.get("output", {})
            audit_path = output_cfg.get("address_alias_audit", duplicates_path("address_alias_hits.csv"))
            audit_dir = os.path.dirname(audit_path)
            if audit_dir:
                os.makedirs(audit_dir, exist_ok=True)

            columns = [
                "timestamp",
                "decision",
                "rule_name",
                "match_type",
                "matched_alias",
                "candidate",
                "canonical_address_id",
                "canonical_full_address",
                "url",
                "source",
            ]

            file_exists = os.path.exists(audit_path)
            with open(audit_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "decision": payload.get("decision", ""),
                    "rule_name": payload.get("rule_name", ""),
                    "match_type": payload.get("match_type", ""),
                    "matched_alias": payload.get("matched_alias", ""),
                    "candidate": payload.get("candidate", ""),
                    "canonical_address_id": payload.get("canonical_address_id", ""),
                    "canonical_full_address": payload.get("canonical_full_address", ""),
                    "url": payload.get("url", ""),
                    "source": payload.get("source", ""),
                })
        except Exception as e:
            logging.warning("_audit_address_alias_hit: Failed to write audit row: %s", e)

    def _alias_conflicts_with_parsed_address(self, parsed_address: Dict[str, Any], canonical: Dict[str, Any]) -> bool:
        """Fail-safe guard: skip alias forcing when parsed street/postal strongly disagree."""
        parsed_street_number = self._safe_lower(parsed_address.get("street_number"))
        canonical_street_number = self._safe_lower(canonical.get("street_number"))
        if parsed_street_number and canonical_street_number and parsed_street_number != canonical_street_number:
            return True

        parsed_postal_code = self._normalize_postal_code(parsed_address.get("postal_code"))
        canonical_postal_code = self._normalize_postal_code(canonical.get("postal_code"))
        if parsed_postal_code and canonical_postal_code and parsed_postal_code != canonical_postal_code:
            return True

        return False

    def _get_alias_canonical_address_id(self, canonical: Dict[str, Any]) -> Optional[int]:
        """Resolve canonical alias mapping to address_id, preferring configured address_id when present."""
        if not isinstance(canonical, dict) or not canonical:
            return None

        configured_id = canonical.get("address_id")
        try:
            configured_id_int = int(configured_id) if configured_id is not None else None
        except (TypeError, ValueError):
            configured_id_int = None

        if configured_id_int and configured_id_int > 0:
            existing = self.execute_query(
                "SELECT address_id FROM address WHERE address_id = :address_id",
                {"address_id": configured_id_int},
            )
            if existing:
                return configured_id_int

        canonical_copy = dict(canonical)
        canonical_copy.pop("address_id", None)
        return self.resolve_or_insert_address(canonical_copy, skip_alias_normalization=True)

    @staticmethod
    def _url_matches_rule(normalized_url: str, match_rule: Dict[str, Any]) -> bool:
        """
        Evaluate whether a normalized URL matches a rule.

        Supported match keys:
            - url_contains: substring match
            - url_equals: exact normalized URL match
            - domain_equals: exact hostname match
        """
        if not normalized_url or not isinstance(match_rule, dict):
            return False

        contains_value = str(match_rule.get("url_contains", "")).strip().lower()
        if contains_value and contains_value in normalized_url:
            return True

        equals_value = str(match_rule.get("url_equals", "")).strip().lower().rstrip("/")
        if equals_value and normalized_url.rstrip("/") == equals_value:
            return True

        domain_value = str(match_rule.get("domain_equals", "")).strip().lower()
        if domain_value:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(normalized_url)
                if parsed.netloc == domain_value:
                    return True
            except Exception:
                return False

        return False

    def _apply_event_overrides(self, df: pd.DataFrame, url: str, parent_url: str) -> pd.DataFrame:
        """
        Apply config-driven field overrides for specific URL rules.
        """
        if df.empty or not self.event_overrides:
            return df

        normalized_candidates = {
            self._normalize_for_compare(str(url or "")),
            self._normalize_for_compare(str(parent_url or "")),
        }
        normalized_candidates.discard("")

        if not normalized_candidates:
            return df

        for override in self.event_overrides:
            match_rule = override.get("match", {})
            set_fields = override.get("set", {})
            rule_name = override.get("name", "unnamed_rule")

            if not isinstance(set_fields, dict) or not set_fields:
                continue

            if any(self._url_matches_rule(candidate, match_rule) for candidate in normalized_candidates):
                for col, value in set_fields.items():
                    df[col] = value
                logging.info(
                    "_apply_event_overrides: Applied override '%s' to %d event row(s).",
                    rule_name,
                    len(df),
                )

        return df

    def get_db_connection(self):
        """
        Establish and return a SQLAlchemy engine for the PostgreSQL database.

        DEPRECATED: This method now uses get_database_config() internally.
        Kept for backward compatibility with existing code.

        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine instance if connection is successful.
            None: If the connection could not be established.

        Note:
            New code should use get_database_config() from db_config module directly.
            This method is maintained for backward compatibility with:
            - ebs.py, clean_up.py, fb.py, scraper.py, irrelevant_rows.py
        """
        try:
            # Use centralized database configuration
            connection_string, env_name = get_database_config()
            logging.info(f"get_db_connection(): Connecting to {env_name}")

            # Create and return the SQLAlchemy engine
            engine = create_engine(connection_string, isolation_level="AUTOCOMMIT")
            return engine

        except Exception as e:
            logging.error("DatabaseHandler: Database connection failed: %s", e)
            return None

    def get_historical_classifier_memory(
        self,
        url: str,
        *,
        min_samples: int = 3,
        dominance_threshold: float = 0.80,
        success_rate_threshold: float = 0.70,
        max_runs: int = 24,
    ) -> Dict[str, Any] | None:
        """
        Return a conservative routing-memory hint for an exact normalized URL.

        This only returns a hint when recent telemetry shows strong agreement on the
        same archetype/owner-step/subtype combination and replay validation
        indicates that route actually worked.
        """
        normalized_url = self._normalize_for_compare(self.normalize_url(url))
        if not normalized_url:
            return None
        if is_holdout_url(normalized_url):
            return None
        try:
            rows = self.execute_query(
                """
                WITH replay_rollup AS (
                    SELECT
                        run_id,
                        baseline_url,
                        BOOL_AND(is_match) AS replay_url_success
                    FROM accuracy_replay_results
                    WHERE baseline_url = :link
                    GROUP BY run_id, baseline_url
                ),
                latest_scrape_per_run AS (
                    SELECT DISTINCT ON (run_id, link)
                        run_id,
                        link,
                        archetype,
                        classification_owner_step,
                        classification_subtype,
                        classification_stage,
                        classification_confidence,
                        time_stamp
                    FROM url_scrape_metrics
                    WHERE link = :link
                      AND step_name = 'scraper'
                      AND archetype IS NOT NULL
                      AND classification_owner_step IS NOT NULL
                    ORDER BY run_id, link, time_stamp DESC
                )
                SELECT
                    ls.archetype,
                    ls.classification_owner_step,
                    ls.classification_subtype,
                    ls.classification_stage,
                    ls.classification_confidence,
                    rr.replay_url_success
                FROM latest_scrape_per_run ls
                JOIN replay_rollup rr
                  ON rr.run_id = ls.run_id
                 AND rr.baseline_url = ls.link
                ORDER BY ls.time_stamp DESC
                LIMIT :limit
                """,
                {"link": normalized_url, "limit": int(max(1, max_runs))},
            ) or []
        except Exception as e:
            logging.warning("get_historical_classifier_memory(): query failed for %s: %s", normalized_url, e)
            return None

        if len(rows) < int(max(1, min_samples)):
            return None

        combo_counts: Counter[tuple[str, str, str]] = Counter()
        combo_success_counts: Counter[tuple[str, str, str]] = Counter()
        combo_confidences: Dict[tuple[str, str, str], list[float]] = {}
        combo_stages: Dict[tuple[str, str, str], Counter[str]] = {}
        for row in rows:
            archetype = str(row[0] or "").strip()
            owner_step = str(row[1] or "").strip()
            subtype = str(row[2] or "").strip()
            stage = str(row[3] or "").strip()
            confidence = row[4]
            replay_url_success = bool(row[5])
            if not archetype or not owner_step:
                continue
            combo = (archetype, owner_step, subtype)
            combo_counts.update([combo])
            if replay_url_success:
                combo_success_counts.update([combo])
            combo_confidences.setdefault(combo, [])
            combo_stages.setdefault(combo, Counter())
            combo_stages[combo].update([stage or "unknown"])
            try:
                if confidence is not None:
                    combo_confidences[combo].append(float(confidence))
            except Exception:
                pass

        if not combo_counts:
            return None

        top_combo, top_count = combo_counts.most_common(1)[0]
        total = sum(combo_counts.values())
        dominance = (top_count / total) if total else 0.0
        if top_count < int(max(1, min_samples)) or dominance < float(dominance_threshold):
            return None
        success_count = int(combo_success_counts.get(top_combo, 0))
        success_rate = (success_count / top_count) if top_count else 0.0
        if success_count < int(max(1, min_samples)) or success_rate < float(success_rate_threshold):
            return None

        confidence_values = combo_confidences.get(top_combo, [])
        stage_counter = combo_stages.get(top_combo, Counter())
        archetype, owner_step, subtype = top_combo
        return {
            "url": normalized_url,
            "archetype": archetype,
            "owner_step": owner_step,
            "subtype": subtype,
            "sample_count": int(top_count),
            "success_count": success_count,
            "success_rate": float(round(success_rate, 4)),
            "dominance": float(round(dominance, 4)),
            "avg_confidence": (
                float(round(sum(confidence_values) / len(confidence_values), 4))
                if confidence_values
                else None
            ),
            "stage_mode": stage_counter.most_common(1)[0][0] if stage_counter else "unknown",
        }
        

    def create_tables(self):
        """
        Creates the required tables in the database if they do not already exist.

        Tables created:
            - urls
            - events
            - address
            - runs

        If config['testing']['drop_tables'] is True, existing tables are dropped before creation,
        and the current 'events' table is backed up to 'events_history'.

        This method ensures all necessary tables for the application are present and logs the process.
        """
        
        # Check if we need to drop tables as per configuration
        if self.config['testing']['drop_tables'] == True:
            drop_queries = [
                "DROP TABLE IF EXISTS events CASCADE;"
            ]
            # Copy events to events_history
            sql = 'SELECT * FROM events;'
            events_df = self.read_sql_df(sql)
            events_df.to_sql('events_history', self.conn, if_exists='append', index=False)

        else:
            # Don't drop any tables
            drop_queries = []

        if drop_queries:
            for query in drop_queries:
                self.execute_query(query)
                logging.info(
                    f"create_tables: Existing tables dropped as per configuration value of "
                    f"'{self.config['testing']['drop_tables']}'."
                )
        else:
            pass

        # Create the 'urls' table
        self.execute_query(URLS_TABLE_SCHEMA_SQL)
        logging.info("create_tables: 'urls' table created or already exists.")

        # Create the 'events' table
        self.execute_query(EVENTS_TABLE_SCHEMA_SQL)
        logging.info("create_tables: 'events' table created or already exists.")

        # Create the 'address' table
        self.execute_query(ADDRESS_TABLE_SCHEMA_SQL)
        logging.info("create_tables: 'address' table created or already exists.")

        # Create the 'runs' table
        self.execute_query(RUNS_TABLE_SCHEMA_SQL)
        logging.info("create_tables: 'address' table created or already exists.")

        # Create chatbot performance metric tables
        self.create_chatbot_metrics_tables()

        # See if this worked.
        query = """
                SELECT table_schema,
                    table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
        rows = self.execute_query(query)
        if rows:
            for schema, table in rows:
                logging.info("Schema: %s, Table: %s", schema, table)
        else:
            logging.info("No tables found or query failed.")


    def create_chatbot_metrics_tables(self) -> None:
        """Create chatbot performance metric tables used for long-term latency reporting."""
        for query in CHATBOT_METRICS_SCHEMA_QUERIES:
            self.execute_query(query)
        logging.info("create_chatbot_metrics_tables: chatbot metric tables created or already exist.")

    def ensure_core_event_tables(self) -> None:
        """Ensure canonical events, events_history, and address tables exist."""
        self.execute_query(EVENTS_HISTORY_TABLE_SCHEMA_SQL)
        self.execute_query("ALTER TABLE events_history ADD COLUMN IF NOT EXISTS original_event_id INTEGER")
        self.execute_query(EVENTS_TABLE_SCHEMA_SQL)
        self.execute_query(ADDRESS_TABLE_SCHEMA_SQL)

    def ensure_core_application_tables(self) -> None:
        """Ensure the base application tables and required core columns exist."""
        self.execute_query(URLS_TABLE_SCHEMA_SQL)
        self.execute_query(RUNS_TABLE_SCHEMA_SQL)
        self.ensure_core_event_tables()

    def ensure_address_sequence(self, start_with: int) -> None:
        """Ensure the address primary-key sequence exists and is attached to address.address_id."""
        safe_start_with = max(1, int(start_with or 1))
        create_seq_sql = f"""
            CREATE SEQUENCE IF NOT EXISTS address_address_id_seq
            START WITH {safe_start_with}
            INCREMENT BY 1
            NO MINVALUE
            NO MAXVALUE
            CACHE 1;
        """
        alter_col_sql = """
            ALTER TABLE address
            ALTER COLUMN address_id SET DEFAULT nextval('address_address_id_seq');
        """
        alter_seq_sql = """
            ALTER SEQUENCE address_address_id_seq OWNED BY address.address_id;
        """
        self.execute_query(create_seq_sql)
        self.execute_query(alter_col_sql)
        self.execute_query(alter_seq_sql)


    def create_urls_df(self):
        """
        Creates and returns a pandas DataFrame from the 'urls' table in the database.

        Returns:
            pandas.DataFrame: DataFrame containing all rows from the 'urls' table.
        Notes:
            - If the table is empty or an error occurs, returns an empty DataFrame.
            - Logs the number of rows loaded or any errors encountered.
        """
        query = "SELECT * FROM urls;"
        try:
            urls_df = self.read_sql_df(query)
            logging.info("create_urls_df: Successfully created DataFrame from 'urls' table.")
            if urls_df.empty:
                logging.warning("create_urls_df: 'urls' table is empty.")
            else:
                logging.info("create_urls_df: 'urls' table contains %d rows.", len(urls_df))
            return urls_df
        except SQLAlchemyError as e:
            logging.error("create_urls_df: Failed to create DataFrame from 'urls' table: %s", e)
            return pd.DataFrame()

    def read_sql_df(
        self,
        query: Any,
        params: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Execute a pandas SQL read with recovery from invalid pooled transaction state."""
        try:
            with self.conn.connect() as connection:
                return pd.read_sql(query, connection, params=params)
        except PendingRollbackError as exc:
            logging.warning(
                "read_sql_df(): Connection was left in an invalid transaction state; "
                "disposing engine and retrying query. Error: %s",
                exc,
            )
            self.conn.dispose()
            with self.conn.connect() as connection:
                return pd.read_sql(query, connection, params=params)
        

    def execute_query(self, query, params=None, statement_timeout_ms: Optional[int] = None):
        """
        Executes a given SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Dictionary of parameters for parameterized queries.
            statement_timeout_ms (int, optional): PostgreSQL statement timeout in milliseconds
                for this query only. When provided, the query will fail fast instead of
                waiting indefinitely on locks or blocked execution.

        Returns:
            list: List of rows (as tuples) if the query returns rows.
            int: Number of rows affected for non-select queries.
            None: If the query fails or there is no database connection.
        """
        if self.conn is None:
            logging.error("execute_query: No database connection available.")
            return None

        # Handle NaN values in params (such as address_id)
        if params:
            for key, value in params.items():
                if isinstance(value, (list, np.ndarray, pd.Series)):
                    if pd.isna(value).any():
                        params[key] = None
                else:
                    if pd.isna(value):
                        params[key] = None

        try:
            with self.conn.connect() as connection:
                if statement_timeout_ms is not None:
                    timeout_ms = max(int(statement_timeout_ms), 1)
                    connection.execute(
                        text("SELECT set_config('statement_timeout', :timeout_value, true)"),
                        {"timeout_value": f"{timeout_ms}ms"},
                    )
                    logging.info(
                        "execute_query(): applied statement_timeout=%dms for query starting with %s",
                        timeout_ms,
                        query.strip().split()[0].upper() if query and query.strip() else "UNKNOWN",
                    )
                result = connection.execute(text(query), params or {})

                if result.returns_rows:
                    rows = result.fetchall()
                    # Extract query type and key info for logging
                    query_type = query.strip().split()[0].upper()
                    if query_type == "SELECT":
                        # Extract table name from SELECT query
                        table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): SELECT from %s returned %d rows", 
                            table_name, len(rows)
                        )
                    else:
                        logging.info(
                            "execute_query(): %s query returned %d rows", 
                            query_type, len(rows)
                        )
                    return rows
                else:
                    affected = result.rowcount
                    connection.commit()
                    # Extract query type for non-select queries
                    query_type = query.strip().split()[0].upper()
                    if query_type == "INSERT":
                        # Extract table name from INSERT query
                        table_match = re.search(r'INSERT\s+(?:INTO\s+)?(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): INSERT into %s affected %d rows", 
                            table_name, affected
                        )
                    elif query_type == "UPDATE":
                        # Extract table name from UPDATE query
                        table_match = re.search(r'UPDATE\s+(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): UPDATE %s affected %d rows", 
                            table_name, affected
                        )
                    else:
                        logging.info(
                            "execute_query(): %s query affected %d rows", 
                            query_type, affected
                        )
                    return affected

        except SQLAlchemyError as e:
            # Handle unique constraint violations for address table gracefully
            error_str = str(e)
            if "UniqueViolation" in error_str and ("unique_full_address" in error_str or "address_full_address_key" in error_str):
                logging.info(
                    "execute_query(): Address already exists (unique constraint), skipping insert"
                )
                return None
            else:
                logging.error(
                    "execute_query(): Query execution failed (%s)\nQuery was: %s",
                    e, query
                )
                return None

    def _deleted_events_audit_path(self) -> str:
        """Return JSONL path for deleted event audit records."""
        return (
            self.config.get("output", {}).get("deleted_events_audit")
            or os.path.join("logs", "deleted_events_audit.jsonl")
        )

    def _write_deleted_event_audit_record(
        self,
        event_row: Dict[str, Any],
        deletion_source: str,
        reason: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist a restore-ready audit record for one deleted event row.

        Each line is valid JSON and includes:
        - metadata (timestamp/source/reason/context)
        - full deleted event row payload
        """
        audit_record: Dict[str, Any] = {
            "deleted_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "deletion_source": deletion_source,
            "deletion_reason": reason,
            "extra_context": extra_context or {},
            "event": event_row,
        }
        path = self._deleted_events_audit_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record, default=str) + "\n")

    def _write_deleted_event_to_history(self, event_row: Dict[str, Any]) -> None:
        """
        Persist a deleted events row into events_history.

        events_history uses its own serial primary key (event_id), so the deleted
        events.event_id is stored in original_event_id.
        """
        if not isinstance(event_row, dict):
            return

        insert_sql = text(
            """
            INSERT INTO events_history (
                original_event_id, event_name, dance_style, description, day_of_week,
                start_date, end_date, start_time, end_time, source, location, price,
                url, event_type, address_id, time_stamp
            ) VALUES (
                :original_event_id, :event_name, :dance_style, :description, :day_of_week,
                :start_date, :end_date, :start_time, :end_time, :source, :location, :price,
                :url, :event_type, :address_id, :time_stamp
            )
            """
        )
        params = {
            "original_event_id": event_row.get("event_id"),
            "event_name": event_row.get("event_name"),
            "dance_style": event_row.get("dance_style"),
            "description": event_row.get("description"),
            "day_of_week": event_row.get("day_of_week"),
            "start_date": event_row.get("start_date"),
            "end_date": event_row.get("end_date"),
            "start_time": event_row.get("start_time"),
            "end_time": event_row.get("end_time"),
            "source": event_row.get("source"),
            "location": event_row.get("location"),
            "price": event_row.get("price"),
            "url": event_row.get("url"),
            "event_type": event_row.get("event_type"),
            "address_id": event_row.get("address_id"),
            "time_stamp": event_row.get("time_stamp"),
        }

        # Keep delete flow resilient: history write failures must not block deletion.
        try:
            with self.conn.begin() as conn:
                conn.execute(insert_sql, params)
        except Exception as e:
            logging.warning(
                "_write_deleted_event_to_history: Failed to write deleted event to events_history "
                "(original_event_id=%s): %s",
                params.get("original_event_id"),
                e,
            )

    @staticmethod
    def _current_run_context() -> Dict[str, Optional[str]]:
        """Return the active run context from environment variables."""
        run_id = str(os.getenv("DS_RUN_ID", "") or "").strip() or None
        step_name = str(os.getenv("DS_STEP_NAME", "") or "").strip() or None
        return {
            "run_id": run_id,
            "step_name": step_name,
        }

    def _normalize_delete_reason_code(self, reason: Any) -> Dict[str, Any]:
        """Normalize raw delete reasons into a controlled vocabulary."""
        raw_reason = str(reason or "").strip().lower()
        if not raw_reason:
            return {
                "delete_reason_code": "unregistered_delete_reason",
                "raw_delete_reason": "",
                "reason_registered": False,
            }
        if raw_reason.startswith("end_date_older_than_") and raw_reason.endswith("_days"):
            return {
                "delete_reason_code": "end_date_older_than_days",
                "raw_delete_reason": raw_reason,
                "reason_registered": True,
            }
        normalized_reason = self._DELETE_REASON_CODE_MAP.get(raw_reason)
        if normalized_reason:
            return {
                "delete_reason_code": normalized_reason,
                "raw_delete_reason": raw_reason,
                "reason_registered": True,
            }
        logging.warning("_normalize_delete_reason_code: unregistered delete reason=%s", raw_reason)
        return {
            "delete_reason_code": "unregistered_delete_reason",
            "raw_delete_reason": raw_reason,
            "reason_registered": False,
        }

    def _lookup_created_by_step(self, event_id: Any) -> Optional[str]:
        """Look up the writer step for an event prior to deletion."""
        if event_id is None:
            return None
        rows = self.execute_query(
            """
            SELECT step_name
            FROM event_write_attribution
            WHERE event_id = :event_id
            LIMIT 1
            """,
            {"event_id": int(event_id)},
        ) or []
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, tuple):
            return str(row[0]).strip() if row and row[0] else None
        if isinstance(row, dict):
            value = row.get("step_name")
            return str(value).strip() if value else None
        return None

    def _write_event_delete_attribution(
        self,
        event_row: Dict[str, Any],
        deletion_source: str,
        reason: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist one canonical delete attribution row."""
        if not isinstance(event_row, dict):
            return
        event_id = event_row.get("event_id")
        if event_id is None:
            logging.warning("_write_event_delete_attribution: skipped row with no event_id")
            return

        run_context = self._current_run_context()
        normalized_reason = self._normalize_delete_reason_code(reason)
        params = {
            "event_id": int(event_id),
            "run_id": run_context["run_id"],
            "step_name": run_context["step_name"] or str(deletion_source or "").strip() or None,
            "delete_reason_code": normalized_reason["delete_reason_code"],
            "raw_delete_reason": normalized_reason["raw_delete_reason"] or None,
            "delete_method": str(deletion_source or "").strip() or None,
            "source_url": str(event_row.get("url") or "").strip() or None,
            "created_by_step": self._lookup_created_by_step(event_id),
            "reason_registered": bool(normalized_reason["reason_registered"]),
            "details_json": json.dumps(extra_context or {}, default=str),
            "time_stamp": datetime.now(),
        }
        query = """
            INSERT INTO event_delete_attribution (
                event_id, run_id, step_name, delete_reason_code, raw_delete_reason,
                delete_method, source_url, created_by_step, reason_registered,
                details_json, time_stamp
            ) VALUES (
                :event_id, :run_id, :step_name, :delete_reason_code, :raw_delete_reason,
                :delete_method, :source_url, :created_by_step, :reason_registered,
                CAST(:details_json AS JSONB), :time_stamp
            )
            ON CONFLICT (event_id)
            DO UPDATE SET
                run_id = EXCLUDED.run_id,
                step_name = EXCLUDED.step_name,
                delete_reason_code = EXCLUDED.delete_reason_code,
                raw_delete_reason = EXCLUDED.raw_delete_reason,
                delete_method = EXCLUDED.delete_method,
                source_url = EXCLUDED.source_url,
                created_by_step = EXCLUDED.created_by_step,
                reason_registered = EXCLUDED.reason_registered,
                details_json = EXCLUDED.details_json,
                time_stamp = EXCLUDED.time_stamp
        """
        self.execute_query(query, params)

    def _coerce_event_records_for_insert(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert the sanitized event DataFrame into DB-ready records."""
        records: List[Dict[str, Any]] = []
        for raw_record in df.to_dict("records"):
            record: Dict[str, Any] = {}
            for key, value in raw_record.items():
                record[key] = None if pd.isna(value) else value
            records.append(record)
        return records

    def _infer_event_write_method(
        self,
        url: str,
        parent_url: str,
        explicit_write_method: Optional[str] = None,
        prompt_type: Optional[str] = None,
    ) -> str:
        """Infer a stable write method label for canonical attribution."""
        if explicit_write_method:
            return str(explicit_write_method).strip()
        prompt_basis = str(prompt_type or "").strip().lower()
        if prompt_basis == "vision_extraction":
            return "vision_extraction"
        combined = " ".join(part for part in [url, parent_url] if part)
        if "calendar" in combined.lower():
            return "google_calendar_fetch"
        return "llm_extraction"

    def _write_event_write_attribution_rows(
        self,
        event_ids: List[int],
        url: str,
        parent_url: str,
        source: str,
        write_method: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_type: Optional[str] = None,
        decision_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist canonical event write attribution rows for inserted events."""
        if not event_ids:
            return
        run_context = self._current_run_context()
        inferred_write_method = self._infer_event_write_method(
            url=url,
            parent_url=parent_url,
            explicit_write_method=write_method,
            prompt_type=prompt_type,
        )
        query = """
            INSERT INTO event_write_attribution (
                event_id, run_id, step_name, url, parent_url, source, write_method,
                provider, model, prompt_type, decision_reason, details_json, time_stamp
            ) VALUES (
                :event_id, :run_id, :step_name, :url, :parent_url, :source, :write_method,
                :provider, :model, :prompt_type, :decision_reason, CAST(:details_json AS JSONB), :time_stamp
            )
            ON CONFLICT (event_id)
            DO UPDATE SET
                run_id = EXCLUDED.run_id,
                step_name = EXCLUDED.step_name,
                url = EXCLUDED.url,
                parent_url = EXCLUDED.parent_url,
                source = EXCLUDED.source,
                write_method = EXCLUDED.write_method,
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                prompt_type = EXCLUDED.prompt_type,
                decision_reason = EXCLUDED.decision_reason,
                details_json = EXCLUDED.details_json,
                time_stamp = EXCLUDED.time_stamp
        """
        timestamp = datetime.now()
        serialized_details = json.dumps(details or {}, default=str)
        for event_id in event_ids:
            params = {
                "event_id": int(event_id),
                "run_id": run_context["run_id"],
                "step_name": run_context["step_name"],
                "url": url or None,
                "parent_url": parent_url or None,
                "source": source or None,
                "write_method": inferred_write_method or None,
                "provider": provider or None,
                "model": model or None,
                "prompt_type": prompt_type or None,
                "decision_reason": decision_reason or None,
                "details_json": serialized_details,
                "time_stamp": timestamp,
            }
            self.execute_query(query, params)

    def _insert_events_and_return_ids(self, df: pd.DataFrame) -> List[int]:
        """Insert events and return the inserted event IDs."""
        records = self._coerce_event_records_for_insert(df)
        if not records:
            return []
        events_table = Table("events", MetaData(), autoload_with=self.conn)
        insert_stmt = insert(events_table).returning(events_table.c.event_id)
        with self.conn.begin() as connection:
            result = connection.execute(insert_stmt, records)
            return [int(row[0]) for row in result.fetchall()]

    @staticmethod
    def _normalize_telemetry_step_name(step_name: Any) -> str:
        """Normalize step/file names into one stable telemetry grouping key."""
        normalized = str(step_name or "").strip().lower()
        if normalized.endswith(".py"):
            normalized = normalized[:-3]
        return normalized

    @staticmethod
    def _coerce_query_row_to_mapping(row: Any, columns: List[str]) -> Dict[str, Any]:
        """Convert one DB row into a plain mapping using fallback column names when needed."""
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
        if isinstance(row, dict):
            return dict(row)
        if isinstance(row, (tuple, list)):
            return dict(zip(columns, row))
        return {}

    def build_phase1_telemetry_integrity_report(self, run_id: str) -> Dict[str, Any]:
        """
        Build the canonical Phase 1 telemetry integrity report for one run.

        The report reconciles `url_scrape_metrics`, `event_write_attribution`, and
        `event_delete_attribution` using normalized step names.
        """
        safe_run_id = str(run_id or "").strip()
        if not safe_run_id:
            return {
                "available": False,
                "run_id": "",
                "status": "FAIL",
                "violations": ["missing_run_id"],
                "summary": {},
                "steps": {},
                "reconciliation_queries": [],
            }

        metrics_query = """
            SELECT
                COALESCE(
                    NULLIF(REPLACE(COALESCE(handled_by, ''), '.py', ''), ''),
                    NULLIF(step_name, '')
                ) AS step_norm,
                SUM(COALESCE(events_written, 0)) AS metrics_events_written_total,
                SUM(CASE WHEN COALESCE(events_written, 0) > 0 THEN 1 ELSE 0 END) AS metrics_urls_with_events_count
            FROM url_scrape_metrics
            WHERE run_id = :run_id
            GROUP BY 1
        """
        write_attr_query = """
            SELECT
                COALESCE(NULLIF(step_name, ''), 'unknown') AS step_norm,
                COUNT(*) AS write_attribution_count,
                COUNT(DISTINCT event_id) AS distinct_event_id_count
            FROM event_write_attribution
            WHERE run_id = :run_id
            GROUP BY 1
        """
        delete_attr_query = """
            SELECT
                COALESCE(NULLIF(step_name, ''), REPLACE(COALESCE(created_by_step, ''), '.py', '')) AS step_norm,
                COUNT(*) AS delete_attribution_count,
                SUM(CASE WHEN reason_registered THEN 0 ELSE 1 END) AS unknown_delete_reason_count
            FROM event_delete_attribution
            WHERE run_id = :run_id
            GROUP BY 1
        """
        duplicate_query = """
            SELECT
                (SELECT COUNT(*) FROM event_write_attribution WHERE run_id = :run_id) AS write_rows,
                (SELECT COUNT(DISTINCT event_id) FROM event_write_attribution WHERE run_id = :run_id) AS write_distinct_events,
                (SELECT COUNT(*) FROM event_delete_attribution WHERE run_id = :run_id) AS delete_rows,
                (SELECT COUNT(DISTINCT event_id) FROM event_delete_attribution WHERE run_id = :run_id) AS delete_distinct_events,
                (SELECT COUNT(*) FROM event_delete_attribution WHERE run_id = :run_id AND NOT reason_registered) AS unknown_delete_reason_total
        """

        try:
            metrics_rows = self.execute_query(metrics_query, {"run_id": safe_run_id}) or []
            write_rows = self.execute_query(write_attr_query, {"run_id": safe_run_id}) or []
            delete_rows = self.execute_query(delete_attr_query, {"run_id": safe_run_id}) or []
            duplicate_rows = self.execute_query(duplicate_query, {"run_id": safe_run_id}) or []
        except Exception as exc:
            return {
                "available": False,
                "run_id": safe_run_id,
                "status": "FAIL",
                "violations": [f"query_error:{exc}"],
                "summary": {},
                "steps": {},
                "reconciliation_queries": [
                    {"name": "metrics_query", "sql": metrics_query.strip()},
                    {"name": "write_attribution_query", "sql": write_attr_query.strip()},
                    {"name": "delete_attribution_query", "sql": delete_attr_query.strip()},
                    {"name": "duplicate_query", "sql": duplicate_query.strip()},
                ],
            }

        steps: Dict[str, Dict[str, Any]] = {}
        for row in metrics_rows:
            mapping = self._coerce_query_row_to_mapping(
                row,
                ["step_norm", "metrics_events_written_total", "metrics_urls_with_events_count"],
            )
            step_name = self._normalize_telemetry_step_name(mapping.get("step_norm"))
            if not step_name:
                continue
            steps.setdefault(step_name, {})
            steps[step_name]["metrics_events_written_total"] = int(mapping.get("metrics_events_written_total", 0) or 0)
            steps[step_name]["metrics_urls_with_events_count"] = int(mapping.get("metrics_urls_with_events_count", 0) or 0)

        for row in write_rows:
            mapping = self._coerce_query_row_to_mapping(
                row,
                ["step_norm", "write_attribution_count", "distinct_event_id_count"],
            )
            step_name = self._normalize_telemetry_step_name(mapping.get("step_norm"))
            if not step_name:
                continue
            steps.setdefault(step_name, {})
            steps[step_name]["write_attribution_count"] = int(mapping.get("write_attribution_count", 0) or 0)
            steps[step_name]["write_distinct_event_id_count"] = int(mapping.get("distinct_event_id_count", 0) or 0)

        for row in delete_rows:
            mapping = self._coerce_query_row_to_mapping(
                row,
                ["step_norm", "delete_attribution_count", "unknown_delete_reason_count"],
            )
            step_name = self._normalize_telemetry_step_name(mapping.get("step_norm"))
            if not step_name:
                continue
            steps.setdefault(step_name, {})
            steps[step_name]["delete_attribution_count"] = int(mapping.get("delete_attribution_count", 0) or 0)
            steps[step_name]["unknown_delete_reason_count"] = int(mapping.get("unknown_delete_reason_count", 0) or 0)

        violations: List[str] = []
        advisories: List[str] = []
        for step_name, payload in steps.items():
            metrics_events = int(payload.get("metrics_events_written_total", 0) or 0)
            write_events = int(payload.get("write_attribution_count", 0) or 0)
            payload["delta_metrics_vs_write_attribution"] = metrics_events - write_events
            payload["status"] = "PASS"
            if metrics_events != write_events:
                payload["status"] = "WARN"
                advisories.append(
                    f"step_mismatch:{step_name}:metrics_events={metrics_events}:write_attribution={write_events}"
                )
            if int(payload.get("unknown_delete_reason_count", 0) or 0) > 0:
                payload["status"] = "FAIL"
                violations.append(
                    f"unknown_delete_reasons:{step_name}:{int(payload.get('unknown_delete_reason_count', 0) or 0)}"
                )

        duplicate_mapping = self._coerce_query_row_to_mapping(
            duplicate_rows[0] if duplicate_rows else {},
            [
                "write_rows",
                "write_distinct_events",
                "delete_rows",
                "delete_distinct_events",
                "unknown_delete_reason_total",
            ],
        )
        write_rows_total = int(duplicate_mapping.get("write_rows", 0) or 0)
        write_distinct_total = int(duplicate_mapping.get("write_distinct_events", 0) or 0)
        delete_rows_total = int(duplicate_mapping.get("delete_rows", 0) or 0)
        delete_distinct_total = int(duplicate_mapping.get("delete_distinct_events", 0) or 0)
        unknown_delete_reason_total = int(duplicate_mapping.get("unknown_delete_reason_total", 0) or 0)

        if write_rows_total != write_distinct_total:
            violations.append(
                f"duplicate_write_attribution_rows:{write_rows_total - write_distinct_total}"
            )
        if delete_rows_total != delete_distinct_total:
            violations.append(
                f"duplicate_delete_attribution_rows:{delete_rows_total - delete_distinct_total}"
            )
        if unknown_delete_reason_total > 0:
            violations.append(f"unknown_delete_reason_total:{unknown_delete_reason_total}")

        status = "PASS" if not violations else "FAIL"
        return {
            "available": bool(steps or duplicate_rows),
            "run_id": safe_run_id,
            "status": status,
            "violations": violations,
            "advisories": advisories,
            "summary": {
                "steps_with_metrics": len(steps),
                "write_attribution_rows": write_rows_total,
                "write_attribution_distinct_event_ids": write_distinct_total,
                "delete_attribution_rows": delete_rows_total,
                "delete_attribution_distinct_event_ids": delete_distinct_total,
                "unknown_delete_reason_total": unknown_delete_reason_total,
                "step_mismatch_advisory_count": len(advisories),
            },
            "steps": dict(sorted(steps.items())),
            "reconciliation_queries": [
                {"name": "metrics_query", "sql": metrics_query.strip()},
                {"name": "write_attribution_query", "sql": write_attr_query.strip()},
                {"name": "delete_attribution_query", "sql": delete_attr_query.strip()},
                {"name": "duplicate_query", "sql": duplicate_query.strip()},
            ],
        }

    def _delete_events_with_audit(
        self,
        delete_sql_without_returning: str,
        params: Optional[Dict[str, Any]],
        deletion_source: str,
        reason: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a DELETE on events, return deleted rows, and write row-level audit.

        Args:
            delete_sql_without_returning: DELETE statement targeting events table,
                without trailing RETURNING clause.
        """
        base_sql = delete_sql_without_returning.strip().rstrip(";")
        wrapped_sql = f"""
        WITH deleted_rows AS (
            {base_sql}
            RETURNING *
        )
        SELECT row_to_json(deleted_rows) AS deleted_event
        FROM deleted_rows;
        """
        rows = self.execute_query(wrapped_sql, params or {}) or []
        deleted_events: List[Dict[str, Any]] = []
        for row in rows:
            # row shape is typically tuple(dict,) from SQLAlchemy fetchall()
            payload = row[0] if isinstance(row, tuple) else row
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"raw": payload}
            if isinstance(payload, dict):
                deleted_events.append(payload)
                self._write_deleted_event_audit_record(
                    event_row=payload,
                    deletion_source=deletion_source,
                    reason=reason,
                    extra_context=extra_context,
                )
                self._write_deleted_event_to_history(payload)
                self._write_event_delete_attribution(
                    event_row=payload,
                    deletion_source=deletion_source,
                    reason=reason,
                    extra_context=extra_context,
                )
            else:
                deleted_events.append({"raw": str(payload)})
        if deleted_events:
            logging.info(
                "_delete_events_with_audit: audited %d deleted event(s) for %s (%s)",
                len(deleted_events),
                deletion_source,
                reason,
            )
        return deleted_events

    
    def close_connection(self):
        """
        Closes the database connection if it exists.

        This method attempts to properly dispose of the current database connection.
        If the connection is successfully closed, an informational log message is recorded.
        If an error occurs during the closing process, the exception is logged as an error.
        If there is no active connection, a warning is logged indicating that there is no connection to close.
        """
        if self.conn:
            try:
                self.conn.dispose()
                logging.info("close_connection: Database connection closed successfully.")
            except Exception as e:
                logging.error("close_connection: Failed to close database connection: %s", e)
        else:
            logging.warning("close_connection: No database connection to close.")


    def write_url_to_db(self, url_row):
        """
        Appends a new URL activity record to the 'urls' table in the database.

        This method processes and normalizes the provided URL data, especially the 'keywords' field,
        ensuring it is stored as a clean, comma-separated string. The data is then inserted as a new row
        into the 'urls' table using pandas' DataFrame and SQL interface.

            url_row (tuple): A tuple containing the following fields in order:
                - link (str): The URL to be logged.
                - parent_url (str): The parent URL from which this link was found.
                - source (str): The source or context of the URL.
                - keywords (str | list | tuple | set): Associated keywords, which can be a string or an iterable.
                - relevant (bool | int): Indicator of relevance.
                - crawl_try (int): Number of crawl attempts.
                - time_stamp (str | datetime): Timestamp of the activity.
                - decision_reason (str, optional): Structured reason for process/skip decision.

        Raises:
            Exception: Logs an error if the database insertion fails.

        Side Effects:
            - Appends a new row to the 'urls' table in the connected database.
            - Logs success or failure of the operation.
        """
        # 1) Unpack (support backward-compatible 7-item rows)
        if len(url_row) >= 8:
            link, parent_url, source, keywords, relevant, crawl_try, time_stamp, decision_reason = url_row[:8]
        else:
            link, parent_url, source, keywords, relevant, crawl_try, time_stamp = url_row
            decision_reason = None

        # 2) Normalize keywords into a simple comma-separated string
        if not isinstance(keywords, str):
            if isinstance(keywords, (list, tuple, set)):
                keywords = ','.join(map(str, keywords))
            else:
                keywords = str(keywords)

        # 3) Strip out braces/brackets/quotes and trim each term
        cleaned = re.sub(r'[\{\}\[\]\"]', '', keywords)
        parts = [p.strip() for p in cleaned.split(',') if p.strip()]
        keywords = ', '.join(parts)

        # 4) Build a one-row DataFrame
        df = pd.DataFrame([{
            'link':           link,
            'parent_url':     parent_url,
            'source':         source,
            'keywords':       keywords,
            'relevant':       relevant,
            'crawl_try':      crawl_try,
            'time_stamp':     time_stamp,
            'decision_reason': str(decision_reason or "").strip() or None,
        }])

        # 5) Append to the table
        try:
            df.to_sql('urls', con=self.conn, if_exists='append', index=False)
            logging.info("write_url_to_db(): appended URL '%s'", link)
        except Exception as e:
            logging.error("write_url_to_db(): failed to append URL '%s': %s", link, e)

    def write_url_scrape_metric(self, metric: Dict[str, Any]) -> None:
        """
        Persist per-URL scrape telemetry for post-run diagnostics and trend reporting.
        """
        if not isinstance(metric, dict):
            return

        keywords = metric.get("keywords", "")
        if not isinstance(keywords, str):
            if isinstance(keywords, (list, tuple, set)):
                keywords = ", ".join(str(k).strip() for k in keywords if str(k).strip())
            else:
                keywords = str(keywords)

        row = {
            "run_id": str(metric.get("run_id", "") or "").strip() or None,
            "step_name": str(metric.get("step_name", "") or "").strip() or None,
            "link": str(metric.get("link", "") or "").strip() or None,
            "parent_url": str(metric.get("parent_url", "") or "").strip() or None,
            "source": str(metric.get("source", "") or "").strip() or None,
            "keywords": keywords,
            "archetype": str(metric.get("archetype", "") or "").strip() or None,
            "extraction_attempted": bool(metric.get("extraction_attempted", False)),
            "extraction_succeeded": bool(metric.get("extraction_succeeded", False)),
            "extraction_skipped": bool(metric.get("extraction_skipped", False)),
            "decision_reason": str(metric.get("decision_reason", "") or "").strip() or None,
            "handled_by": str(metric.get("handled_by", "") or "").strip() or None,
            "routing_reason": str(metric.get("routing_reason", "") or "").strip() or None,
            "classification_confidence": (
                float(metric.get("classification_confidence"))
                if metric.get("classification_confidence") is not None
                else None
            ),
            "classification_stage": str(metric.get("classification_stage", "") or "").strip() or None,
            "classification_owner_step": str(metric.get("classification_owner_step", "") or "").strip() or None,
            "classification_subtype": str(metric.get("classification_subtype", "") or "").strip() or None,
            "classification_features_json": (
                str(metric.get("classification_features_json", "") or "").strip() or None
            ),
            "access_attempted": bool(metric.get("access_attempted", False)),
            "access_succeeded": bool(metric.get("access_succeeded", False)),
            "text_extracted": bool(metric.get("text_extracted", False)),
            "keywords_found": bool(metric.get("keywords_found", False)),
            "events_written": int(metric.get("events_written", 0) or 0),
            "ocr_attempted": bool(metric.get("ocr_attempted", False)),
            "ocr_succeeded": bool(metric.get("ocr_succeeded", False)),
            "vision_attempted": bool(metric.get("vision_attempted", False)),
            "vision_succeeded": bool(metric.get("vision_succeeded", False)),
            "fallback_used": bool(metric.get("fallback_used", False)),
            "links_discovered": int(metric.get("links_discovered", 0) or 0),
            "links_followed": int(metric.get("links_followed", 0) or 0),
            "time_stamp": metric.get("time_stamp", datetime.now()),
        }
        try:
            pd.DataFrame([row]).to_sql('url_scrape_metrics', con=self.conn, if_exists='append', index=False)
        except Exception as e:
            logging.warning(
                "write_url_scrape_metric(): failed for link=%s reason=%s",
                row.get("link"),
                e,
            )

    def record_metric_observation(
        self,
        run_id: str,
        metric_key: str,
        metric_value_numeric: float,
        metric_unit: str = "",
        description: str = "",
        higher_is_better: bool = True,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
        notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert metric definition and insert one metric observation.
        """
        safe_run_id = str(run_id or "").strip()
        safe_metric_key = str(metric_key or "").strip()
        if not safe_run_id or not safe_metric_key:
            return
        try:
            self.execute_query(
                """
                INSERT INTO metric_definitions (metric_key, metric_unit, description, higher_is_better, updated_at)
                VALUES (:metric_key, :metric_unit, :description, :higher_is_better, :updated_at)
                ON CONFLICT (metric_key)
                DO UPDATE SET
                    metric_unit = COALESCE(EXCLUDED.metric_unit, metric_definitions.metric_unit),
                    description = COALESCE(EXCLUDED.description, metric_definitions.description),
                    higher_is_better = EXCLUDED.higher_is_better,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "metric_key": safe_metric_key,
                    "metric_unit": str(metric_unit or "").strip() or None,
                    "description": str(description or "").strip() or None,
                    "higher_is_better": bool(higher_is_better),
                    "updated_at": datetime.now(),
                },
            )
            metric_rows = self.execute_query(
                "SELECT metric_id FROM metric_definitions WHERE metric_key = :metric_key LIMIT 1",
                {"metric_key": safe_metric_key},
            )
            if not metric_rows:
                return
            metric_id = int(metric_rows[0][0])
            self.execute_query(
                """
                INSERT INTO metric_observations (
                    run_id, metric_id, metric_value_numeric, window_start, window_end, notes_json
                )
                VALUES (
                    :run_id, :metric_id, :metric_value_numeric, :window_start, :window_end, :notes_json
                )
                """,
                {
                    "run_id": safe_run_id,
                    "metric_id": metric_id,
                    "metric_value_numeric": float(metric_value_numeric),
                    "window_start": window_start,
                    "window_end": window_end,
                    "notes_json": json.dumps(notes or {}, ensure_ascii=True) if notes else None,
                },
            )
        except Exception as e:
            logging.warning(
                "record_metric_observation(): failed for run_id=%s metric_key=%s: %s",
                safe_run_id,
                safe_metric_key,
                e,
            )

    def record_accuracy_replay_result(
        self,
        run_id: str,
        query_text: str,
        baseline_event_id: Optional[int],
        baseline_url: str,
        baseline_snapshot: Optional[Dict[str, Any]],
        replay_snapshot: Optional[Dict[str, Any]],
        is_match: bool,
        mismatch_category: str,
        mismatch_details: str,
    ) -> None:
        """
        Insert one row-level replay accuracy record.
        """
        safe_run_id = str(run_id or "").strip()
        safe_query_text = str(query_text or "").strip()
        if not safe_run_id or not safe_query_text:
            return
        try:
            event_id_value = None
            if baseline_event_id is not None:
                event_id_value = int(baseline_event_id)
            self.execute_query(
                """
                INSERT INTO accuracy_replay_results (
                    run_id, query_text, baseline_event_id, baseline_url, baseline_snapshot_json,
                    replay_snapshot_json, is_match, mismatch_category, mismatch_details
                ) VALUES (
                    :run_id, :query_text, :baseline_event_id, :baseline_url, :baseline_snapshot_json,
                    :replay_snapshot_json, :is_match, :mismatch_category, :mismatch_details
                )
                """,
                {
                    "run_id": safe_run_id,
                    "query_text": safe_query_text,
                    "baseline_event_id": event_id_value,
                    "baseline_url": str(baseline_url or "").strip() or None,
                    "baseline_snapshot_json": json.dumps(baseline_snapshot or {}, ensure_ascii=True),
                    "replay_snapshot_json": json.dumps(replay_snapshot or {}, ensure_ascii=True),
                    "is_match": bool(is_match),
                    "mismatch_category": str(mismatch_category or "").strip() or None,
                    "mismatch_details": str(mismatch_details or "").strip() or None,
                },
            )
        except Exception as e:
            logging.warning(
                "record_accuracy_replay_result(): failed for run_id=%s baseline_event_id=%s: %s",
                safe_run_id,
                baseline_event_id,
                e,
            )

    def record_validation_run_artifact(
        self,
        run_id: str,
        artifact_type: str,
        artifact_payload: Dict[str, Any],
    ) -> None:
        """Upsert one JSON validation artifact keyed by run_id and artifact_type."""
        safe_run_id = str(run_id or "").strip()
        safe_artifact_type = str(artifact_type or "").strip()
        if not safe_run_id or not safe_artifact_type or not isinstance(artifact_payload, dict):
            return
        try:
            self.execute_query(
                """
                INSERT INTO validation_run_artifacts (
                    run_id, artifact_type, artifact_json, created_at, updated_at
                )
                VALUES (
                    :run_id, :artifact_type, :artifact_json, :created_at, :updated_at
                )
                ON CONFLICT (run_id, artifact_type)
                DO UPDATE SET
                    artifact_json = EXCLUDED.artifact_json,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "run_id": safe_run_id,
                    "artifact_type": safe_artifact_type,
                    "artifact_json": json.dumps(artifact_payload, ensure_ascii=True),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                },
            )
        except Exception as e:
            logging.warning(
                "record_validation_run_artifact(): failed for run_id=%s artifact_type=%s: %s",
                safe_run_id,
                safe_artifact_type,
                e,
            )

    def record_fb_block_triage_rows(
        self,
        *,
        run_id: str,
        blocked_reason: str,
        rows: List[Dict[str, Any]],
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> None:
        """Upsert aggregated Facebook blocked-content triage rows for one validation run."""
        safe_run_id = str(run_id or "").strip()
        safe_blocked_reason = str(blocked_reason or "").strip()
        if not safe_run_id or not safe_blocked_reason or not isinstance(rows, list):
            return
        try:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                source_key = str(row.get("source_key", "") or "").strip()
                block_category = str(row.get("category", "") or "").strip()
                if not source_key or not block_category:
                    continue
                self.execute_query(
                    """
                    INSERT INTO fb_block_triage (
                        run_id,
                        blocked_reason,
                        source_key,
                        block_category,
                        unique_url_count,
                        blocked_attempt_count,
                        sample_url,
                        window_start,
                        window_end,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :run_id,
                        :blocked_reason,
                        :source_key,
                        :block_category,
                        :unique_url_count,
                        :blocked_attempt_count,
                        :sample_url,
                        :window_start,
                        :window_end,
                        :created_at,
                        :updated_at
                    )
                    ON CONFLICT (run_id, blocked_reason, source_key, block_category)
                    DO UPDATE SET
                        unique_url_count = EXCLUDED.unique_url_count,
                        blocked_attempt_count = EXCLUDED.blocked_attempt_count,
                        sample_url = EXCLUDED.sample_url,
                        window_start = EXCLUDED.window_start,
                        window_end = EXCLUDED.window_end,
                        updated_at = EXCLUDED.updated_at
                    """,
                    {
                        "run_id": safe_run_id,
                        "blocked_reason": safe_blocked_reason,
                        "source_key": source_key,
                        "block_category": block_category,
                        "unique_url_count": int(row.get("unique_urls", 0) or 0),
                        "blocked_attempt_count": int(row.get("attempt_count", 0) or 0),
                        "sample_url": str(row.get("sample_url", "") or "").strip() or None,
                        "window_start": window_start,
                        "window_end": window_end,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now(),
                    },
                )
        except Exception as e:
            logging.warning(
                "record_fb_block_triage_rows(): failed for run_id=%s blocked_reason=%s: %s",
                safe_run_id,
                safe_blocked_reason,
                e,
            )

    def record_fb_block_occurrences(
        self,
        *,
        run_id: str,
        blocked_reason: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Insert raw Facebook blocked URL occurrences for one validation run."""
        safe_run_id = str(run_id or "").strip()
        safe_blocked_reason = str(blocked_reason or "").strip()
        if not safe_run_id or not safe_blocked_reason or not isinstance(rows, list):
            return
        try:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                requested_url = str(row.get("requested_url", "") or "").strip()
                source_key = str(row.get("source_key", "") or "").strip()
                block_category = str(row.get("category", "") or "").strip()
                occurrence_raw = str(row.get("occurrence_ts", "") or "").strip()
                if not requested_url or not source_key or not block_category:
                    continue
                occurrence_ts = None
                if occurrence_raw:
                    try:
                        occurrence_ts = datetime.fromisoformat(occurrence_raw)
                    except ValueError:
                        occurrence_ts = None
                self.execute_query(
                    """
                    INSERT INTO fb_block_occurrences (
                        run_id,
                        blocked_reason,
                        requested_url,
                        source_key,
                        block_category,
                        occurrence_ts,
                        created_at
                    )
                    VALUES (
                        :run_id,
                        :blocked_reason,
                        :requested_url,
                        :source_key,
                        :block_category,
                        :occurrence_ts,
                        :created_at
                    )
                    ON CONFLICT (run_id, blocked_reason, requested_url, source_key, block_category, occurrence_ts)
                    DO NOTHING
                    """,
                    {
                        "run_id": safe_run_id,
                        "blocked_reason": safe_blocked_reason,
                        "requested_url": requested_url,
                        "source_key": source_key,
                        "block_category": block_category,
                        "occurrence_ts": occurrence_ts,
                        "created_at": datetime.now(),
                    },
                )
        except Exception as e:
            logging.warning(
                "record_fb_block_occurrences(): failed for run_id=%s blocked_reason=%s: %s",
                safe_run_id,
                safe_blocked_reason,
                e,
            )

    def record_classifier_training_url_candidate(
        self,
        *,
        run_id: str,
        candidate: Dict[str, Any],
    ) -> None:
        """
        Insert one URL-level classifier training candidate derived from replay validation.
        """
        safe_run_id = str(run_id or "").strip()
        normalized_url = str((candidate or {}).get("normalized_url") or "").strip()
        if not safe_run_id or not normalized_url:
            return
        try:
            self.execute_query(
                """
                INSERT INTO classifier_training_url_candidates (
                    run_id, query_text, normalized_url, domain, total_rows, true_count, false_count,
                    match_rate_pct, status, recommended_action, training_eligible, recommended_archetype,
                    recommended_owner_step, recommended_subtype, priority_score, mismatch_category_counts_json,
                    baseline_event_ids_json, sample_baseline_json, sample_replay_json
                ) VALUES (
                    :run_id, :query_text, :normalized_url, :domain, :total_rows, :true_count, :false_count,
                    :match_rate_pct, :status, :recommended_action, :training_eligible, :recommended_archetype,
                    :recommended_owner_step, :recommended_subtype, :priority_score, :mismatch_category_counts_json,
                    :baseline_event_ids_json, :sample_baseline_json, :sample_replay_json
                )
                """,
                {
                    "run_id": safe_run_id,
                    "query_text": str(candidate.get("query_text") or "").strip() or None,
                    "normalized_url": normalized_url,
                    "domain": str(candidate.get("domain") or "").strip() or None,
                    "total_rows": int(candidate.get("total_rows", 0) or 0),
                    "true_count": int(candidate.get("true_count", 0) or 0),
                    "false_count": int(candidate.get("false_count", 0) or 0),
                    "match_rate_pct": float(candidate.get("match_rate_pct", 0.0) or 0.0),
                    "status": str(candidate.get("status") or "").strip() or "manual_review_needed",
                    "recommended_action": str(candidate.get("recommended_action") or "").strip() or None,
                    "training_eligible": bool(candidate.get("training_eligible")),
                    "recommended_archetype": str(candidate.get("recommended_archetype") or "").strip() or None,
                    "recommended_owner_step": str(candidate.get("recommended_owner_step") or "").strip() or None,
                    "recommended_subtype": str(candidate.get("recommended_subtype") or "").strip() or None,
                    "priority_score": int(candidate.get("priority_score", 0) or 0),
                    "mismatch_category_counts_json": json.dumps(
                        candidate.get("mismatch_category_counts") or {},
                        ensure_ascii=True,
                    ),
                    "baseline_event_ids_json": json.dumps(
                        candidate.get("baseline_event_ids") or [],
                        ensure_ascii=True,
                    ),
                    "sample_baseline_json": json.dumps(candidate.get("sample_baseline") or {}, ensure_ascii=True),
                    "sample_replay_json": json.dumps(candidate.get("sample_replay") or {}, ensure_ascii=True),
                },
            )
        except Exception as e:
            logging.warning(
                "record_classifier_training_url_candidate(): failed for run_id=%s url=%s: %s",
                safe_run_id,
                normalized_url,
                e,
            )
    

    def create_address_dict(self, full_address, street_number, street_name, street_type, postal_box, city, province_or_state, postal_code, country_id):
        """
        Creates an address dictionary with the given parameters.

        Args:
            full_address (str): The full address.
            street_number (str): The street number.
            street_name (str): The street name.
            street_type (str): The street type.
            postal_box (str): The postal box.
            city (str): The city.
            province_or_state (str): The province or state.
            postal_code (str): The postal code.
            country_id (str): The country ID.

        Returns:
            dict: The address dictionary.
        """
        return {
            'full_address': full_address,
            'street_number': street_number,
            'street_name': street_name,
            'street_type': street_type,
            'postal_box': postal_box,
            'city': city,
            'province_or_state': province_or_state,
            'postal_code': postal_code,
            'country_id': country_id
        }
    

    def clean_up_address_basic(self, events_df):
        """
        Cleans events using only local DB and regex methods (no Foursquare).
        """
        logging.info("clean_up_address_basic(): Starting with shape %s", events_df.shape)

        address_df = self.read_sql_df("SELECT * FROM address")

        for index, row in events_df.iterrows():
            event_id = row.get('event_id')
            location = row.get('location')

            address_id, new_location = self.try_resolve_address(
                event_id, location, address_df=address_df, use_foursquare=False
            )

            if address_id:
                events_df.at[index, 'address_id'] = address_id
                events_df.at[index, 'location'] = new_location

        return events_df
    

    def try_resolve_address(self, event_id, location, address_df=None, use_foursquare=False):
        """
        Tries to resolve an address_id for a given location.
        Can optionally use Foursquare fallbacks.

        Args:
            event_id: ID of the event being processed.
            location: String location field from the event.
            address_df: Optional DataFrame of known addresses.
            use_foursquare: Whether to attempt external lookups via Foursquare.

        Returns:
            (address_id, new_location): Tuple containing the resolved address_id (or None)
                                        and the updated location string (or None).
        """
        if not location or pd.isna(location):
            return None, None
        
        if location is None:
            pass  # Keep it as None
        elif isinstance(location, str):
            location = location.strip()

        # 1. Try DB match using street number/name
        if address_df is not None:
            update_list = self.get_address_update_for_event(event_id, location, address_df)
            if update_list:
                return update_list[0]['address_id'], location

        # 2. Try local regex postal code extraction
        postal_code = self.extract_canadian_postal_code(location)
        if postal_code:
            updated_location, address_id = self.populate_from_db_or_fallback(location, postal_code)
            if address_id:
                return address_id, updated_location

        if use_foursquare:
            # 3. Try Foursquare postal code
            postal_code = self.get_postal_code_foursquare(location)
            if postal_code and self.is_canadian_postal_code(postal_code):
                updated_location, address_id = self.populate_from_db_or_fallback(location, postal_code)
                if address_id:
                    return address_id, updated_location

            # 4. Try Foursquare municipality fallback
            updated_location, address_id = self.fallback_with_municipality(location)
            if address_id:
                return address_id, updated_location

        # Final fallback: ensure consistent return type
        return None, None
            

    def get_address_update_for_event(self, event_id, location, address_df):
        """
        Given an event's ID, its location string, and a DataFrame of addresses, 
        this method attempts to extract the street number from the location using a regular expression. 
        It then searches the address DataFrame for a row where the 'street_number' matches the extracted number 
        and the 'street_name' is present in the location string.
        Parameters:
            event_id (Any): The unique identifier for the event.
            location (str): The location string containing address information.
            address_df (pd.DataFrame): A DataFrame containing address data with at least 
                'street_number', 'street_name', and 'address_id' columns.
        Returns:
            List[dict]: A list containing a single dictionary with 'event_id' and 'address_id' keys 
            if a matching address is found; otherwise, an empty list.
        """
        updates = []
        if location is None:
            return updates
        match = re.search(r'\d+', location)
        if match:
            extracted_number = match.group()
            matching_addresses = address_df[address_df['street_number'] == extracted_number]
            for _, addr_row in matching_addresses.iterrows():
                if pd.notnull(addr_row['street_name']) and addr_row['street_name'] in location:
                    new_address_id = addr_row['address_id']
                    updates.append({
                        "event_id": event_id,
                        "address_id": new_address_id
                    })
                    break  # Stop after the first match is found.
        return updates
    

    def extract_canadian_postal_code(self, location_str):
        """
        Extracts a valid Canadian postal code from a given location string.

        This method uses a regular expression to search for a Canadian postal code pattern
        within the provided string. If a match is found, it removes any spaces from the
        postal code and validates it using the `is_canadian_postal_code` method. If the
        postal code is valid, it returns the cleaned postal code; otherwise, it returns None.

        Args:
            location_str (str): The input string potentially containing a Canadian postal code.

        Returns:
            str or None: The valid Canadian postal code without spaces if found and valid, otherwise None.
        """
        match = re.search(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d', location_str)
        if match:
            possible_pc = match.group().replace(' ', '')
            if self.is_canadian_postal_code(possible_pc):
                return possible_pc
        return None
    

    def is_canadian_postal_code(self, postal_code):
        """
        Determines whether the provided postal code matches the Canadian postal code format.

        A valid Canadian postal code follows the pattern: A1A 1A1, where 'A' is a letter and '1' is a digit.
        The space between the third and fourth characters is optional.

        Args:
            postal_code (str): The postal code string to validate.

        Returns:
            bool: True if the postal code matches the Canadian format, False otherwise.
        """
        pattern = r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$'
        return bool(re.match(pattern, postal_code.strip()))

    def extract_us_postal_code(self, location_str: str) -> str | None:
        """
        Extract a likely US ZIP code only when accompanied by a strong US-address signal.
        """
        text = str(location_str or "").strip()
        if not text:
            return None
        normalized = re.sub(r"\s+", " ", text.upper())
        state_codes = "|".join(sorted(_US_STATE_OR_TERRITORY_CODES))
        state_zip_match = re.search(rf"\b(?:{state_codes})\b[\s,]+(\d{{5}}(?:-\d{{4}})?)\b", normalized)
        if state_zip_match:
            return state_zip_match.group(1)
        country_zip_match = re.search(
            r"\b(?:USA|U\.S\.A\.|UNITED STATES|UNITED STATES OF AMERICA)\b.*?\b(\d{5}(?:-\d{4})?)\b",
            normalized,
        )
        if country_zip_match:
            return country_zip_match.group(1)
        return None

    def _drop_us_postal_code_events(self, df: pd.DataFrame, *, context: str) -> pd.DataFrame:
        """Drop rows that clearly describe US addresses so they never reach event insertion."""
        if df.empty:
            return df
        working_df = df.copy()

        def _row_has_us_address_signal(row: pd.Series) -> bool:
            postal_code = str(row.get("postal_code") or "").strip()
            if postal_code and re.fullmatch(r"\d{5}(?:-\d{4})?", postal_code):
                province_or_state = str(row.get("province_or_state") or "").strip().upper()
                country_id = str(row.get("country_id") or "").strip().upper()
                if province_or_state in _US_STATE_OR_TERRITORY_CODES or country_id in {"US", "USA", "UNITED STATES"}:
                    return True
            return self.extract_us_postal_code(str(row.get("location") or "")) is not None

        disqualify_mask = working_df.apply(_row_has_us_address_signal, axis=1)
        disqualified_count = int(disqualify_mask.sum())
        if not disqualified_count:
            return working_df
        sample_locations = working_df.loc[disqualify_mask, "location"].astype(str).head(3).tolist()
        logging.warning(
            "_drop_us_postal_code_events: Dropping %d event(s) during %s due to US ZIP/location signals. samples=%s",
            disqualified_count,
            context,
            sample_locations,
        )
        return working_df.loc[~disqualify_mask].reset_index(drop=True)
    

    def populate_from_db_or_fallback(self, location_str, postal_code):
        """
        Attempts to populate a formatted address using a Canadian postal code by querying the local address database.
        If no match is found in the database, returns a fallback value (currently returns None, None).

        Args:
            location_str (str): The raw location string, typically containing civic number and street information.
            postal_code (str): The Canadian postal code to look up in the database.

            tuple:
                updated_location (str or None): The formatted address string if found, otherwise None.
                address_id (Any or None): The unique identifier for the address if found, otherwise None.

        Notes:
            - If multiple database rows match the postal code, attempts to match the civic number from the location string.
            - If no valid match is found, returns (None, None).
            - Logging is used to provide information and warnings about the lookup process.
            - Fallback logic for missing database entries.
        """
        numbers = re.findall(r'\d+', location_str)
        query = """
            SELECT
                civic_no,
                civic_no_suffix,
                official_street_name,
                official_street_type,
                official_street_dir,
                mail_mun_name,
                mail_prov_abvn,
                mail_postal_code
            FROM locations
            WHERE mail_postal_code = %s;
        """
        df = self.read_sql_df(query, params=(postal_code,))

        # Single or multiple rows
        if df.empty:
            return None, None

        if df.shape[0] == 1:
            row = df.iloc[0]
        else:
            match_index = self.match_civic_number(df, numbers)
            if match_index is None or match_index not in df.index:
                logging.warning("populate_from_db_or_fallback(): No valid match found.")
                return None, None
            row = df.loc[match_index]

        updated_location = self.format_address_from_db_row(row)

        address_dict = self.create_address_dict(
            updated_location, str(row.civic_no) if row.civic_no else None, row.official_street_name,
            row.official_street_type, None, row.mail_mun_name, row.mail_prov_abvn, row.mail_postal_code, 'CA'
        )
        address_id = self.resolve_or_insert_address(address_dict)
        logging.info("Populated from DB for postal code '%s': '%s'", postal_code, updated_location)
        return updated_location, address_id
    

    def resolve_or_insert_address(self, parsed_address: dict, skip_alias_normalization: bool = False) -> Optional[int]:
        """
        Resolves an address by checking multiple matching strategies in order of specificity.
        Uses improved fuzzy matching to prevent duplicate addresses.

        Args:
            parsed_address (dict): Dictionary of parsed address fields.
            skip_alias_normalization (bool): Internal flag to bypass alias mapping recursion.

        Returns:
            int or None: The address_id of the matched or newly inserted address.
        """
        if not parsed_address:
            logging.info("resolve_or_insert_address: No parsed address provided.")
            return None

        if not skip_alias_normalization:
            alias_candidates: List[str] = [
                parsed_address.get("full_address", ""),
                parsed_address.get("building_name", ""),
            ]
            street_number_candidate = (parsed_address.get("street_number") or "").strip()
            street_name_candidate = (parsed_address.get("street_name") or "").strip()
            if street_number_candidate or street_name_candidate:
                alias_candidates.append(f"{street_number_candidate} {street_name_candidate}".strip())

            alias_match = self._find_address_alias_match(alias_candidates)
            if alias_match:
                canonical = alias_match["canonical"]
                if self._alias_conflicts_with_parsed_address(parsed_address, canonical):
                    self._audit_address_alias_hit({
                        "decision": "skipped_conflict",
                        "rule_name": alias_match.get("rule_name"),
                        "match_type": alias_match.get("match_type"),
                        "matched_alias": alias_match.get("matched_alias"),
                        "candidate": alias_match.get("candidate"),
                        "canonical_address_id": canonical.get("address_id"),
                        "canonical_full_address": canonical.get("full_address"),
                    })
                    logging.warning(
                        "resolve_or_insert_address: Skipping alias '%s' due to postal/street conflict.",
                        alias_match.get("rule_name", "unnamed_address_alias"),
                    )
                else:
                    canonical_address_id = self._get_alias_canonical_address_id(canonical)
                    if canonical_address_id:
                        self._audit_address_alias_hit({
                            "decision": "applied",
                            "rule_name": alias_match.get("rule_name"),
                            "match_type": alias_match.get("match_type"),
                            "matched_alias": alias_match.get("matched_alias"),
                            "candidate": alias_match.get("candidate"),
                            "canonical_address_id": canonical_address_id,
                            "canonical_full_address": canonical.get("full_address"),
                        })
                        return canonical_address_id
                    parsed_address = {**parsed_address, **canonical}

        building_name = (parsed_address.get("building_name") or "").strip()
        street_number = (parsed_address.get("street_number") or "").strip()
        street_name = (parsed_address.get("street_name") or "").strip()
        postal_code = (parsed_address.get("postal_code") or "").strip()
        city = (parsed_address.get("city") or "").strip()
        country_id = (parsed_address.get("country_id") or "").strip()

        # Step 1: Exact match on postal code + street number (most specific)
        if postal_code and street_number:
            logging.debug(f"resolve_or_insert_address: Trying postal_code + street_number match: {postal_code}, {street_number}")
            postal_match_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(postal_code) = LOWER(:postal_code)
                AND LOWER(street_number) = LOWER(:street_number)
            """
            postal_matches = self.execute_query(postal_match_query, {
                "postal_code": postal_code,
                "street_number": street_number
            })

            # CRITICAL FIX: Count matches to prevent cache corruption
            postal_match_count = len(postal_matches) if postal_matches else 0

            for addr_id, b_name, s_num, s_name, p_code in postal_matches or []:
                if building_name and b_name:
                    if not self._has_meaningful_token_overlap(building_name, b_name):
                        continue
                    # Use multiple fuzzy matching algorithms
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    # More sophisticated matching: any high score indicates a match
                    if ratio_score >= 85 or partial_score >= 95 or token_set_score >= 90:
                        logging.debug(f"Postal+street+fuzzy building match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id
                elif not building_name and not b_name and postal_match_count == 1:
                    # CRITICAL: Only match without building name if there's exactly ONE address
                    # This prevents cache corruption where different buildings at same address
                    # get incorrectly matched to the first result
                    logging.debug(f"Postal+street match (no building comparison, single match) → address_id={addr_id}")
                    return addr_id

        # Step 2: Street number + street name match with improved building name fuzzy matching
        if street_number and street_name:
            select_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(street_number) = LOWER(:street_number)
                AND (LOWER(street_name) = LOWER(:street_name) OR LOWER(street_name) = LOWER(:street_name_alt))
            """
            # Handle common street name variations (Niagra vs Niagara)
            street_name_alt = street_name.replace('Niagra', 'Niagara').replace('Niagara', 'Niagra')
            
            street_matches = self.execute_query(select_query, {
                "street_number": street_number,
                "street_name": street_name,
                "street_name_alt": street_name_alt
            })

            # CRITICAL FIX: Count matches to prevent cache corruption
            street_match_count = len(street_matches) if street_matches else 0

            for addr_id, b_name, s_num, s_name, p_code in street_matches or []:
                if building_name and b_name:
                    if not self._has_meaningful_token_overlap(building_name, b_name):
                        continue
                    # Multiple fuzzy algorithms
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    if ratio_score >= 75 or partial_score >= 90 or token_set_score >= 85:
                        logging.debug(f"Street+fuzzy building match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id
                elif not building_name and not b_name and street_match_count == 1:
                    # CRITICAL: Only match without building name if there's exactly ONE address
                    # This prevents cache corruption where different buildings at same address
                    # get incorrectly matched to the first result
                    logging.debug(f"Street match (no building comparison, single match) → address_id={addr_id}")
                    return addr_id
        else:
            logging.debug("resolve_or_insert_address: Missing street_number or street_name; skipping street match")

        # Step 3: City + building name fuzzy match (broader search)
        if city and building_name:
            logging.debug(f"resolve_or_insert_address: Trying city + building_name match: {city}, {building_name}")
            city_building_query = """
                SELECT address_id, building_name, city, postal_code
                FROM address
                WHERE LOWER(city) = LOWER(:city) AND building_name IS NOT NULL
            """
            city_matches = self.execute_query(city_building_query, {"city": city})

            for addr_id, b_name, addr_city, p_code in city_matches or []:
                if b_name:
                    if not self._has_meaningful_token_overlap(building_name, b_name):
                        continue
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    # Higher thresholds for city-only matches to avoid false positives
                    if ratio_score >= 90 or partial_score >= 95 or token_set_score >= 95:
                        logging.debug(f"City+building fuzzy match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id

        # Step 4: Legacy building name-only fuzzy match (least reliable)
        if building_name:
            logging.debug(f"resolve_or_insert_address: Trying building_name-only fuzzy match: {building_name}")
            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            candidates = self.execute_query(query)

            for addr_id, existing_name in candidates or []:
                if existing_name:
                    if not self._has_meaningful_token_overlap(building_name, existing_name):
                        continue
                    ratio_score = ratio(building_name, existing_name)
                    partial_score = fuzz.partial_ratio(building_name, existing_name)
                    token_set_score = fuzz.token_set_ratio(building_name, existing_name)

                    # Very high thresholds for building-name-only matches
                    if ratio_score >= 95 or (partial_score >= 98 and token_set_score >= 95):
                        logging.debug(f"Building-name-only fuzzy match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id

        # Step 3: Normalize null values and prepare required fields for insert
        parsed_address = self.normalize_nulls(parsed_address)
        
        # Ensure ALL fields expected by INSERT query are present (set to None if missing)
        required_fields = [
            "building_name", "street_number", "street_name", "street_type", "direction", "city",
            "province_or_state", "postal_code", "country_id"
        ]
        
        for field in required_fields:
            if field not in parsed_address:
                parsed_address[field] = None
        
        # Set specific fields from extracted values
        parsed_address["building_name"] = building_name or parsed_address.get("building_name")
        parsed_address["street_number"] = street_number or parsed_address.get("street_number")
        parsed_address["street_name"] = street_name or parsed_address.get("street_name")
        parsed_address["street_type"] = parsed_address.get("street_type")
        parsed_address["direction"] = parsed_address.get("direction")
        parsed_address["country_id"] = country_id or parsed_address.get("country_id")

        # Build standardized full_address from components
        standardized_full_address = self.build_full_address(
            building_name=parsed_address.get("building_name"),
            street_number=parsed_address.get("street_number"),
            street_name=parsed_address.get("street_name"),
            street_type=parsed_address.get("street_type"),
            direction=parsed_address.get("direction"),
            city=parsed_address.get("city"),
            province_or_state=parsed_address.get("province_or_state"),
            postal_code=parsed_address.get("postal_code"),
            country_id=parsed_address.get("country_id")
        )
        parsed_address["full_address"] = standardized_full_address

        # Set time_stamp for the new address
        parsed_address["time_stamp"] = datetime.now().isoformat()

        # FINAL DEDUPLICATION CHECK: Before inserting, check if building_name already exists
        building_name = (parsed_address.get("building_name") or "").strip()
        if building_name and len(building_name) > 2:
            # Try to find existing address with same building name
            existing_addr_id = self.find_address_by_building_name(building_name, threshold=80)
            if existing_addr_id:
                logging.info(f"resolve_or_insert_address: Found existing address (dedup) with building_name='{building_name}' → address_id={existing_addr_id}")
                return existing_addr_id

        insert_query = """
            INSERT INTO address (
                building_name, street_number, street_name, street_type, direction, city,
                province_or_state, postal_code, country_id, full_address, time_stamp
            ) VALUES (
                :building_name, :street_number, :street_name, :street_type, :direction, :city,
                :province_or_state, :postal_code, :country_id, :full_address, :time_stamp
            )
            RETURNING address_id;
        """

        result = self.execute_query(insert_query, parsed_address)
        if result:
            address_id = result[0][0]
            logging.info(f"Inserted new address with address_id: {address_id}")
            return address_id
        else:
            # If insert failed (likely due to unique constraint), try to find existing address
            full_address = parsed_address.get("full_address")
            if full_address:
                lookup_query = "SELECT address_id FROM address WHERE full_address = :full_address"
                lookup_result = self.execute_query(lookup_query, {"full_address": full_address})
                if lookup_result:
                    address_id = lookup_result[0][0]
                    logging.info(f"Found existing address with address_id: {address_id}")
                    return address_id
            
            logging.error("resolve_or_insert_address: Failed to insert or find existing address")
            return None
    

    def build_full_address(self, building_name: str = None, street_number: str = None, 
                          street_name: str = None, street_type: str = None, 
                          direction: str = None,
                          city: str = None, province_or_state: str = None, 
                          postal_code: str = None, country_id: str = None) -> str:
        """
        Builds a standardized full_address string from address components.
        
        Format: "building_name, street_number street_name street_type, city, province_or_state postal_code, country_id"
        
        Args:
            building_name: Building or venue name (optional)
            street_number: Street number
            street_name: Street name  
            street_type: Street type (St, Ave, Rd, etc.)
            direction: Street direction token (E, W, N, S, etc.)
            city: City name
            province_or_state: Province or state
            postal_code: Postal code
            country_id: Country code
            
        Returns:
            str: Formatted full address
        """
        address_parts = []
        
        # Add building name first if present
        if building_name and building_name.strip():
            address_parts.append(building_name.strip())
        
        # Build street address
        street_parts = []
        if street_number and street_number.strip():
            street_parts.append(street_number.strip())
        if direction and direction.strip():
            street_parts.append(direction.strip())
        if street_name and street_name.strip():
            street_parts.append(street_name.strip())
        if street_type and street_type.strip():
            street_parts.append(street_type.strip())
        
        if street_parts:
            address_parts.append(' '.join(street_parts))
        
        # Add city
        if city and city.strip():
            address_parts.append(city.strip())
        
        # Add province/state and postal code
        if province_or_state and province_or_state.strip():
            if postal_code and postal_code.strip():
                address_parts.append(f"{province_or_state.strip()} {postal_code.strip()}")
            else:
                address_parts.append(province_or_state.strip())
        elif postal_code and postal_code.strip():
            address_parts.append(postal_code.strip())
        
        # Add country
        if country_id and country_id.strip():
            address_parts.append(country_id.strip())
        
        return ', '.join(address_parts)

    def get_full_address_from_id(self, address_id: int) -> Optional[str]:
        """
        Returns the full_address from the address table for the given address_id.
        """
        query = "SELECT full_address FROM address WHERE address_id = :address_id"
        result = self.execute_query(query, {"address_id": address_id})
        return result[0][0] if result else None


    def format_address_from_db_row(self, db_row):
        """
        Constructs a formatted address string from a database row.

        This method takes a database row object containing address components and constructs
        a single formatted address string. The formatted address includes the street address,
        city (municipality), province abbreviation, postal code, and country ("CA").
        Missing components are omitted gracefully.

        Args:
            db_row: An object representing a database row with address fields. Expected attributes are:
                - civic_no
                - civic_no_suffix
                - official_street_name
                - official_street_type
                - official_street_dir
                - mail_mun_name (city/municipality)
                - mail_prov_abvn (province abbreviation)
                - mail_postal_code

        Returns:
            str: A formatted address string in the form:
                "<street address>, <city>, <province abbreviation>, <postal code>, CA"
            Any missing components are omitted from the output.
        """
        # Build the street portion
        parts = [
            str(db_row.civic_no) if db_row.civic_no else "",
            str(db_row.civic_no_suffix) if db_row.civic_no_suffix else "",
            db_row.official_street_name or "",
            db_row.official_street_type or "",
            db_row.official_street_dir or ""
        ]
        street_address = " ".join(part for part in parts if part).strip()

        # Insert the city if available
        city = db_row.mail_mun_name or ""

        # Construct final location string
        formatted = (
            f"{street_address}, "
            f"{city}, "                      # <─ Include municipality
            f"{db_row.mail_prov_abvn or ''}, "
            f"{db_row.mail_postal_code or ''}, CA"
        )
        # Clean up spacing
        formatted = re.sub(r'\s+,', ',', formatted)
        formatted = re.sub(r',\s+,', ',', formatted)
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        return formatted
    

    def write_events_to_db(
        self,
        df,
        url,
        parent_url,
        source,
        keywords,
        *,
        write_method: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_type: Optional[str] = None,
        decision_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Processes and writes event data to the 'events' table in the database.
        This method performs several data cleaning and transformation steps on the input DataFrame,
        including renaming columns, handling missing values, formatting dates and times, and removing
        outdated or incomplete events. It also logs relevant information and writes a record of the
        processed URL to a separate table.
            df (pandas.DataFrame): DataFrame containing raw event data to be processed and stored.
            url (str): The URL from which the events data was sourced.
            parent_url (str): The parent URL, if applicable, for hierarchical event sources.
            source (str): The source identifier for the event data. If empty, it will be inferred from the URL.
            keywords (str or list): Keywords or dance styles associated with the events. If a list, it will be joined into a string.
            - The method automatically renames columns to match the database schema if the data is from a Google Calendar source.
            - Missing or empty 'source' and 'url' fields are filled with appropriate values.
            - Dates and times are coerced into standard formats; warnings during parsing are suppressed.
            - The 'price' column is ensured to exist and is treated as text.
            - A 'time_stamp' column is added to record the time of data insertion.
            - The method cleans up the 'location' field and updates address IDs via a helper method.
            - Rows with all important fields missing are dropped.
            - Events older than a configurable number of days are excluded.
            - If no valid events remain after cleaning, the method logs this and records the URL as not relevant.
            - Cleaned data is saved to a CSV file for debugging and then written to the 'events' table in the database.
            - The method logs key actions and outcomes for traceability.
        Returns:
            int: Number of events written to the database.
        """
        url = '' if pd.isna(url) else str(url)
        parent_url = '' if pd.isna(parent_url) else str(parent_url)

        calendar_style_columns = {
            'URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week',
            'Start_Date', 'End_Date', 'Start_Time', 'End_Time',
            'Price', 'Location', 'Description',
        }
        is_calendar_payload = bool(calendar_style_columns.intersection(set(df.columns)))
        if 'calendar' in url or 'calendar' in parent_url or is_calendar_payload:
            df = self._rename_google_calendar_columns(df)
            # Do not blindly overwrite dance_style from source keyword bundles.
            # Only backfill missing dance_style when keyword styles are specific and small.
            keyword_styles = self._keywords_to_specific_dance_styles(keywords)
            if "dance_style" not in df.columns:
                df["dance_style"] = pd.NA
            if keyword_styles:
                dance_style_series = df["dance_style"].fillna("").astype(str).str.strip()
                missing_style_mask = dance_style_series.eq("")
                if missing_style_mask.any():
                    df.loc[missing_style_mask, "dance_style"] = keyword_styles

        source = self._resolve_event_source_label(source=source, url=url, parent_url=parent_url)
        df['source'] = df.get('source', pd.Series([''] * len(df))).replace('', source).fillna(source)
        df = self._enforce_event_source_values(df, source)
        df['url'] = df.get('url', pd.Series([''] * len(df))).replace('', url).fillna(url)
        df = self._enforce_event_url_values(df, default_url=url, parent_url=parent_url, source=source)
        df = self._apply_event_overrides(df, url=url, parent_url=parent_url)

        try:
            self._convert_datetime_fields(df)
        except KeyError as exc:
            sample_row = df.head(1).to_dict('records') if not df.empty else []
            logging.error(
                "write_events_to_db: datetime conversion KeyError=%s url=%s parent_url=%s columns=%s sample=%s",
                exc,
                url,
                parent_url,
                list(df.columns),
                sample_row,
            )
            for col in ['start_date', 'end_date', 'start_time', 'end_time']:
                if col not in df.columns:
                    df[col] = pd.NA
            try:
                self._convert_datetime_fields(df)
            except Exception as retry_exc:
                logging.error(
                    "write_events_to_db: datetime conversion retry failed url=%s parent_url=%s error=%s",
                    url,
                    parent_url,
                    retry_exc,
                )
                self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now()])
                return 0

        df = self._normalize_overnight_end_dates(df)

        if 'price' not in df.columns:
            logging.warning("write_events_to_db: 'price' column is missing. Filling with empty string.")
            df['price'] = ''

        df['time_stamp'] = datetime.now()

        # Clean day_of_week field to handle compound/invalid values
        df = self._clean_day_of_week_field(df)
        df = self._align_recurring_weekday_dates(df, url=url)
        # Live-music policy: require explicit dance-style evidence in event text.
        df = self._enforce_live_music_dance_style_policy(df)

        # Basic location cleanup
        df = self.clean_up_address_basic(df)
        df = self._drop_us_postal_code_events(df, context="pre_address")
        if df.empty:
            logging.info("write_events_to_db: No events remain after US ZIP pre-filter, skipping address processing.")
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now(), "us_postal_code_location"])
            return 0

        # Drop old events before expensive address resolution.
        rows_before_old_filter = len(df)
        old_only_rejection_reason = self._old_only_rejection_reason_for_url(self.normalize_url(url))
        df = self._drop_old_events_by_date(df, context="pre_address")
        if df.empty:
            logging.info("write_events_to_db: No events remain after early old-date filtering, skipping address processing.")
            decision_reason = old_only_rejection_reason if rows_before_old_filter > 0 and old_only_rejection_reason else None
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now(), decision_reason])
            return 0

        # Resolve structured addresses using LLM + match/insert logic
        updated_rows = []
        for i, row in df.iterrows():
            event_dict = row.to_dict()
            event_dict = self.normalize_nulls(event_dict)
            updated_event = self.process_event_address(event_dict)
            for key in ["address_id", "location"]:
                if key in updated_event:
                    df.at[i, key] = updated_event[key]
            updated_rows.append(updated_event)

        logging.info(f"write_events_to_db: Address processing complete for {len(updated_rows)} events.")
        df = self._drop_us_postal_code_events(df, context="post_address")
        if df.empty:
            logging.info("write_events_to_db: No events remain after US ZIP post-filter, skipping write.")
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now(), "us_postal_code_location"])
            return 0

        # Remove rows that are incomplete after address normalization.
        df = self._filter_events(df, apply_date_filter=False)

        if df.empty:
            logging.info("write_events_to_db: No events remain after filtering, skipping write.")
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now()])
            return 0

        # Final type guard before insert to keep address_id nullable-int and text fields scalar.
        df = self._sanitize_events_dataframe_for_insert(df)
        if df.empty:
            logging.info("write_events_to_db: No events remain after type sanitization, skipping write.")
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now()])
            return 0

        # Write debug CSV (only locally, not on Render)
        if os.getenv('RENDER') != 'true':
            df.to_csv(events_path('cleaned_events.csv'), index=False)

        logging.info(f"write_events_to_db: Number of events to write: {len(df)}")

        inserted_event_ids = self._insert_events_and_return_ids(df)
        self._write_event_write_attribution_rows(
            event_ids=inserted_event_ids,
            url=url,
            parent_url=parent_url,
            source=source,
            write_method=write_method,
            provider=provider,
            model=model,
            prompt_type=prompt_type,
            decision_reason=decision_reason,
            details=details,
        )
        self.write_url_to_db([url, parent_url, source, keywords, True, 1, datetime.now()])
        logging.info(
            "write_events_to_db: Events data written to the 'events' table with %d attribution row(s).",
            len(inserted_event_ids),
        )
        return len(inserted_event_ids)

    def _keywords_to_specific_dance_styles(self, keywords: Any) -> str:
        """
        Convert crawl keywords into a safe dance_style autofill string.

        Returns empty string when keywords are broad/generic (to avoid style pollution).
        """
        if keywords is None:
            return ""
        if isinstance(keywords, list):
            raw_parts = [str(item).strip().lower() for item in keywords if str(item).strip()]
        else:
            raw_parts = [part.strip().lower() for part in str(keywords).split(",") if part.strip()]

        if not raw_parts:
            return ""

        matched: list[str] = []
        for part in raw_parts:
            if part in self._DANCE_STYLE_TOKENS:
                matched.append(part)

        deduped: list[str] = []
        for style in matched:
            if style not in deduped:
                deduped.append(style)

        if not deduped:
            return ""
        if len(deduped) > self._MAX_AUTOFILL_KEYWORD_STYLES:
            return ""
        return ", ".join(deduped)

    @classmethod
    def _is_placeholder_source(cls, value: Any) -> bool:
        """Return True when source label is blank or a known placeholder."""
        if value is None:
            return True
        source_text = str(value).strip().lower()
        if source_text in cls._SOURCE_PLACEHOLDER_VALUES:
            return True
        if "extracted text" in source_text:
            return True
        return False

    @staticmethod
    def _source_from_url(url: str) -> str:
        """Derive a stable source label from URL host when possible."""
        try:
            parsed = urlparse(str(url or "").strip())
            host = (parsed.netloc or "").strip().lower()
            if not host:
                return ""
            if host.startswith("www."):
                host = host[4:]
            if ":" in host:
                host = host.split(":", 1)[0]
            labels = [label for label in host.split(".") if label]
            if not labels:
                return ""
            if len(labels) >= 2:
                return labels[-2]
            return labels[0]
        except Exception:
            return ""

    @staticmethod
    def _normalize_source_url_affinity_text(value: Any) -> str:
        """Normalize text for lightweight source-to-URL affinity checks."""
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text)

    @classmethod
    def _source_matches_event_url(cls, source: Any, url: Any) -> bool:
        """Return True when the declared source clearly matches the event URL host label."""
        source_norm = cls._normalize_source_url_affinity_text(source)
        url_source_norm = cls._normalize_source_url_affinity_text(cls._source_from_url(str(url or "")))
        if not source_norm or not url_source_norm:
            return False
        return source_norm == url_source_norm or source_norm in url_source_norm or url_source_norm in source_norm

    def _resolve_event_source_label(self, source: Any, url: str, parent_url: str) -> str:
        """
        Resolve canonical source label for event writes.

        Priority:
        1) Non-placeholder explicit source argument
        2) URL host label
        3) Parent URL host label
        4) 'unknown'
        """
        source_text = "" if source is None else str(source).strip()
        if not self._is_placeholder_source(source_text):
            return source_text
        from_url = self._source_from_url(url)
        if from_url:
            return from_url
        from_parent = self._source_from_url(parent_url)
        if from_parent:
            return from_parent
        return "unknown"

    def _enforce_event_source_values(self, df: pd.DataFrame, fallback_source: str) -> pd.DataFrame:
        """
        Replace placeholder row-level source labels with the resolved source context.
        """
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        working_df = df.copy()
        if "source" not in working_df.columns:
            working_df["source"] = fallback_source
            return working_df

        replacements = 0
        normalized_source_values: List[str] = []
        for raw_source in working_df["source"].tolist():
            if self._is_placeholder_source(raw_source):
                normalized_source_values.append(fallback_source)
                replacements += 1
            else:
                normalized_source_values.append(str(raw_source).strip())
        working_df["source"] = normalized_source_values
        if replacements:
            logging.info(
                "_enforce_event_source_values: replaced %d placeholder source values with '%s'",
                replacements,
                fallback_source,
            )
        return working_df

    @staticmethod
    def _looks_like_http_url(value: Any) -> bool:
        """Return True when value resembles an HTTP(S) URL."""
        if value is None:
            return False
        value_str = str(value).strip().lower()
        return value_str.startswith("http://") or value_str.startswith("https://")

    @staticmethod
    def _extract_email(value: Any) -> Optional[str]:
        """Return normalized email address when present in value; otherwise None."""
        if value is None:
            return None
        value_str = str(value).strip().lower()
        match = re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", value_str)
        if not match:
            return None
        return match.group(0)

    def _enforce_event_url_values(
        self,
        df: pd.DataFrame,
        default_url: str,
        parent_url: str,
        source: str,
    ) -> pd.DataFrame:
        """
        Ensure event rows have a durable URL-like identifier before insert.

        Rules:
        - Prefer row URL when present.
        - Fallback to call-level `default_url` (the page/event URL being processed).
        - Fallback to `parent_url` when it is HTTP(S).
        - For email ingestion rows, allow storing an email address when no HTTP URL exists.
        - Drop non-email rows that still have no URL after fallback.
        """
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df

        working_df = df.copy()
        if "url" not in working_df.columns:
            working_df["url"] = pd.NA
        if "source" not in working_df.columns:
            working_df["source"] = source or ""

        default_url_norm = str(default_url or "").strip()
        parent_url_norm = str(parent_url or "").strip()
        source_norm = str(source or "").strip().lower()

        fallback_url = ""
        if self._looks_like_http_url(default_url_norm):
            fallback_url = default_url_norm
        elif self._looks_like_http_url(parent_url_norm):
            fallback_url = parent_url_norm

        default_email = self._extract_email(default_url_norm)
        source_email = self._extract_email(source_norm)

        dropped_non_email = 0
        dropped_email = 0
        keep_rows: List[bool] = []
        normalized_urls: List[Optional[str]] = []

        for _, row in working_df.iterrows():
            row_url = "" if pd.isna(row.get("url")) else str(row.get("url", "")).strip()
            row_source = "" if pd.isna(row.get("source")) else str(row.get("source", "")).strip().lower()

            is_email_context = (
                "email" in row_source
                or "email" in source_norm
                or "email inbox" in str(parent_url_norm).lower()
            )

            final_url = row_url
            if not final_url:
                if fallback_url:
                    final_url = fallback_url
                elif is_email_context:
                    row_email = self._extract_email(row_url) or self._extract_email(row_source)
                    final_url = row_email or default_email or source_email or ""

            if final_url:
                keep_rows.append(True)
                normalized_urls.append(final_url)
                continue

            if is_email_context:
                dropped_email += 1
            else:
                dropped_non_email += 1
            keep_rows.append(False)
            normalized_urls.append(None)

        if dropped_non_email or dropped_email:
            logging.warning(
                "_enforce_event_url_values: dropped rows with missing URL (non_email=%d email=%d)",
                dropped_non_email,
                dropped_email,
            )

        filtered_df = working_df.loc[keep_rows].copy().reset_index(drop=True)
        filtered_urls = [u for u, keep in zip(normalized_urls, keep_rows) if keep]
        filtered_df["url"] = filtered_urls
        return filtered_df


    def _rename_google_calendar_columns(self, df):
        """
        Renames columns of a DataFrame containing Google Calendar event data to standardized column names.
        Parameters:
            df (pandas.DataFrame): The input DataFrame with original Google Calendar column names.
        Returns:
            pandas.DataFrame: A DataFrame with columns renamed to standardized names:
                - 'URL' -> 'url'
                - 'Type_of_Event' -> 'event_type'
                - 'Name_of_the_Event' -> 'event_name'
                - 'Day_of_Week' -> 'day_of_week'
                - 'Start_Date' -> 'start_date'
                - 'End_Date' -> 'end_date'
                - 'Start_Time' -> 'start_time'
                - 'End_Time' -> 'end_time'
                - 'Price' -> 'price'
                - 'Location' -> 'location'
                - 'Description' -> 'description'
        """
        return df.rename(columns={
            'URL': 'url', 'Type_of_Event': 'event_type', 'Name_of_the_Event': 'event_name',
            'Day_of_Week': 'day_of_week', 'Start_Date': 'start_date', 'End_Date': 'end_date',
            'Start_Time': 'start_time', 'End_Time': 'end_time', 'Price': 'price',
            'Location': 'location', 'Description': 'description'
        })

    def _convert_datetime_fields(self, df):
        """
        Converts specific datetime-related columns in a pandas DataFrame to appropriate date and time types.
        This method processes the following columns:
            - 'start_date' and 'end_date': Converts to `datetime.date` objects.
            - 'start_time' and 'end_time': Converts to `datetime.time` objects.
        Any parsing errors are coerced to NaT/NaN. UserWarnings during conversion are suppressed.
        Args:
            df (pandas.DataFrame): The DataFrame containing the columns to convert.
        Returns:
            None: The DataFrame is modified in place.
        """
        # Normalize common alternate datetime column names before conversion.
        alias_map = {
            'Start_Date': 'start_date',
            'End_Date': 'end_date',
            'Start_Time': 'start_time',
            'End_Time': 'end_time',
        }
        for alias, canonical in alias_map.items():
            if canonical not in df.columns and alias in df.columns:
                df[canonical] = df[alias]

        # Ensure canonical columns exist to avoid hard-fail KeyError on imperfect payloads.
        for col in ['start_date', 'end_date', 'start_time', 'end_time']:
            if col not in df.columns:
                df[col] = pd.NA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for col in ['start_date', 'end_date']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            for col in ['start_time', 'end_time']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
            warnings.resetwarnings()

    def _normalize_overnight_end_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roll end_date forward for overnight events whose end_time crosses midnight."""
        if df is None or df.empty:
            return df
        working_df = df.copy()
        required_columns = {"start_date", "end_date", "start_time", "end_time"}
        if not required_columns.issubset(set(working_df.columns)):
            return working_df

        rollover_count = 0
        for idx, row in working_df.iterrows():
            start_date = row.get("start_date")
            end_date = row.get("end_date")
            start_time = row.get("start_time")
            end_time = row.get("end_time")
            if pd.isna(start_date) or pd.isna(start_time) or pd.isna(end_time):
                continue
            if not end_date or pd.isna(end_date):
                end_date = start_date
            if end_date != start_date:
                continue
            try:
                if end_time < start_time:
                    start_date_value = pd.to_datetime(start_date, errors="coerce")
                    if pd.isna(start_date_value):
                        continue
                    working_df.at[idx, "end_date"] = (start_date_value + timedelta(days=1)).date()
                    rollover_count += 1
            except Exception:
                continue

        if rollover_count:
            logging.info(
                "_normalize_overnight_end_dates: Rolled %d event(s) to next-day end_date.",
                rollover_count,
            )
        return working_df

    @staticmethod
    def _address_text_supports_candidate(raw_location: str, candidate: dict[str, Any]) -> bool:
        """Return True when raw location text plausibly supports the candidate address row."""
        raw_text = str(raw_location or "").strip().lower()
        if not raw_text:
            return True

        building_name = str(candidate.get("building_name") or "").strip()
        street_number = str(candidate.get("street_number") or "").strip().lower()
        street_name = str(candidate.get("street_name") or "").strip().lower()
        direction = str(candidate.get("direction") or "").strip().lower()
        full_address = str(candidate.get("full_address") or "").strip().lower()

        building_supported = bool(
            building_name and DatabaseHandler._has_meaningful_token_overlap(raw_text, building_name)
        )

        if street_number and street_number not in raw_text:
            return False if (street_name or full_address) else building_supported
        if street_name:
            street_name_tokens = [token for token in re.split(r"[\s,/.-]+", street_name) if token]
            significant_tokens = [token for token in street_name_tokens if len(token) > 2 or token.isdigit()]
            if significant_tokens and not all(token in raw_text for token in significant_tokens):
                return False if (street_number or full_address) else building_supported
        if direction:
            directional_pattern = re.compile(
                rf"\b{re.escape(street_number)}\s+{re.escape(direction)}\b" if street_number else rf"\b{re.escape(direction)}\b",
                re.IGNORECASE,
            )
            if direction in {"n", "s", "e", "w"} and directional_pattern.search(raw_text) is None:
                return False
        if street_name and street_number:
            return True
        return building_supported or bool(street_name or street_number)

    def _clean_day_of_week_field(self, df):
        """
        Cleans and standardizes the day_of_week field to handle compound/invalid values.
        
        This method fixes common issues with day_of_week values:
        - Compound values like "Friday, Saturday" -> takes first day ("Friday")
        - Special values like "Daily" -> converts to empty string
        - Normalizes case and whitespace
        
        Args:
            df (pd.DataFrame): DataFrame containing event data with day_of_week column
            
        Returns:
            pd.DataFrame: DataFrame with cleaned day_of_week values
        """
        if 'day_of_week' not in df.columns:
            return df
            
        original_count = len(df)
        logging.info(f"_clean_day_of_week_field: Processing {original_count} events")
        
        # Track changes for logging
        changes_made = 0
        
        for i, row in df.iterrows():
            original_value = row.get('day_of_week', '')
            if pd.isna(original_value) or not str(original_value).strip():
                continue
                
            day_str = str(original_value).strip()
            cleaned_value = original_value
            
            # Handle compound values like "Friday, Saturday" - take first day
            if ',' in day_str:
                cleaned_value = day_str.split(',')[0].strip()
                changes_made += 1
                logging.info(f"_clean_day_of_week_field: Changed compound day '{original_value}' to '{cleaned_value}' for event at index {i}")
                
            # Handle special values like "Daily"
            elif day_str.lower() in ['daily', 'every day', 'everyday']:
                cleaned_value = ''  # Set to empty, will be handled by validation later
                changes_made += 1
                logging.info(f"_clean_day_of_week_field: Changed special day '{original_value}' to empty for event at index {i}")
                
            # Normalize standard day names (capitalize first letter)
            else:
                valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                day_lower = day_str.lower()
                if day_lower in valid_days:
                    cleaned_value = day_lower.capitalize()
                    if cleaned_value != original_value:
                        changes_made += 1
                
            # Update the DataFrame if value changed
            if cleaned_value != original_value:
                df.at[i, 'day_of_week'] = cleaned_value
        
        logging.info(f"_clean_day_of_week_field: Made {changes_made} changes to day_of_week values")
        return df

    @staticmethod
    def _has_recurrence_signal(*texts: Any) -> bool:
        """Return True when text strongly suggests a recurring schedule."""
        combined = " ".join(str(text or "") for text in texts).strip().lower()
        if not combined:
            return False
        recurrence_patterns = (
            r"\bevery\b",
            r"\bweekly\b",
            r"\brecurs?\b",
            r"\beach\b",
            r"\bmondays?\b",
            r"\btuesdays?\b",
            r"\bwednesdays?\b",
            r"\bthursdays?\b",
            r"\bfridays?\b",
            r"\bsaturdays?\b",
            r"\bsundays?\b",
        )
        return any(re.search(pattern, combined, re.IGNORECASE) for pattern in recurrence_patterns)

    @staticmethod
    def _nearest_weekday_shift(current_date: date, target_weekday_name: str) -> int:
        """Return the signed day shift needed to align current_date to target weekday."""
        weekday_lookup = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target = weekday_lookup.get(str(target_weekday_name or "").strip().lower())
        if target is None:
            return 0
        current = current_date.weekday()
        forward = (target - current) % 7
        backward = forward - 7
        return backward if abs(backward) <= abs(forward) else forward

    def _align_recurring_weekday_dates(self, df: pd.DataFrame, url: str) -> pd.DataFrame:
        """
        Align recurring event dates to their stated weekday before insert.

        This only applies when a row has recurrence evidence from its text or from
        repeated event rows in the same write batch. One-off events are left intact.
        """
        if df is None or df.empty:
            return df
        if "day_of_week" not in df.columns or "start_date" not in df.columns:
            return df

        working_df = df.copy()
        recurrence_group_sizes: dict[tuple[str, str, str, str], int] = {}
        if {"event_name", "start_time", "source"}.issubset(working_df.columns):
            grouped = (
                working_df.assign(
                    _event_key=working_df["event_name"].fillna("").astype(str).str.strip().str.lower(),
                    _time_key=working_df["start_time"].fillna("").astype(str).str.strip(),
                    _source_key=working_df["source"].fillna("").astype(str).str.strip().str.lower(),
                    _dow_key=working_df["day_of_week"].fillna("").astype(str).str.strip().str.lower(),
                )
                .groupby(["_event_key", "_time_key", "_source_key", "_dow_key"], dropna=False)
                .size()
            )
            recurrence_group_sizes = {tuple(key): int(size) for key, size in grouped.items()}

        shifts_applied = 0
        for idx, row in working_df.iterrows():
            start_date_value = row.get("start_date")
            if pd.isna(start_date_value) or start_date_value is None:
                continue

            day_of_week_value = str(row.get("day_of_week", "") or "").strip()
            if not day_of_week_value:
                continue

            recurrence_signal = self._has_recurrence_signal(
                row.get("event_name", ""),
                row.get("description", ""),
                row.get("location", ""),
            )
            if recurrence_group_sizes:
                group_key = (
                    str(row.get("event_name", "") or "").strip().lower(),
                    str(row.get("start_time", "") or "").strip(),
                    str(row.get("source", "") or "").strip().lower(),
                    day_of_week_value.lower(),
                )
                recurrence_signal = recurrence_signal or recurrence_group_sizes.get(group_key, 0) > 1

            if not recurrence_signal:
                continue

            shift_days = self._nearest_weekday_shift(start_date_value, day_of_week_value)
            if shift_days == 0:
                continue

            original_start = start_date_value
            working_df.at[idx, "start_date"] = original_start + timedelta(days=shift_days)
            end_date_value = row.get("end_date")
            if pd.notna(end_date_value) and end_date_value is not None:
                working_df.at[idx, "end_date"] = end_date_value + timedelta(days=shift_days)
            shifts_applied += 1
            logging.info(
                "_align_recurring_weekday_dates: shifted row index=%d url=%s event_name=%s start_date=%s->%s weekday=%s",
                idx,
                url,
                str(row.get("event_name", "") or ""),
                original_start,
                working_df.at[idx, "start_date"],
                day_of_week_value,
            )

        if shifts_applied:
            logging.info(
                "_align_recurring_weekday_dates: aligned %d recurring event date(s) for url=%s",
                shifts_applied,
                url,
            )
        return working_df

    def _enforce_live_music_dance_style_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For live-music events, keep dance_style only when explicitly evidenced.

        Evidence is limited to event_name and description text. If no dance-style
        token appears there, dance_style is set to NULL.
        """
        if df is None or df.empty:
            return df
        if "event_type" not in df.columns or "dance_style" not in df.columns:
            return df

        normalized = df.copy()
        nullified = 0
        explicit = 0
        for idx, row in normalized.iterrows():
            event_type = str(row.get("event_type", "") or "").strip().lower()
            if "live music" not in event_type:
                continue

            event_name = str(row.get("event_name", "") or "").lower()
            description = str(row.get("description", "") or "").lower()
            evidence_text = f"{event_name} {description}".strip()

            matched_styles: list[str] = []
            for token in sorted(self._DANCE_STYLE_TOKENS, key=len, reverse=True):
                if token in evidence_text:
                    matched_styles.append(token)
            deduped_styles: list[str] = []
            seen: set[str] = set()
            for token in matched_styles:
                if token not in seen:
                    deduped_styles.append(token)
                    seen.add(token)

            if deduped_styles:
                normalized.at[idx, "dance_style"] = ", ".join(deduped_styles)
                explicit += 1
            else:
                normalized.at[idx, "dance_style"] = pd.NA
                nullified += 1

        if nullified or explicit:
            logging.info(
                "_enforce_live_music_dance_style_policy: live-music rows updated (nullified=%d explicit=%d)",
                nullified,
                explicit,
            )
        return normalized

    def _drop_old_events_by_date(self, df: pd.DataFrame, context: str = "default") -> pd.DataFrame:
        """
        Drop events whose end_date is older than the configured cutoff.
        """
        if 'end_date' not in df.columns:
            logging.warning("_drop_old_events_by_date(%s): Missing 'end_date' column.", context)
            return df

        old_days = int(self.config.get('clean_up', {}).get('old_events', 0) or 0)
        if old_days <= 0:
            return df

        working_df = df.copy()
        working_df['end_date'] = pd.to_datetime(working_df['end_date'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=old_days)
        rows_before = len(working_df)
        filtered_df = working_df[working_df['end_date'] >= cutoff].reset_index(drop=True)
        dropped = rows_before - len(filtered_df)
        if dropped:
            logging.info(
                "_drop_old_events_by_date(%s): Dropped %s events older than %s",
                context,
                dropped,
                cutoff.date(),
            )
        return filtered_df

    def _filter_events(self, df, apply_date_filter: bool = True):
        """
        Filters a DataFrame of events by removing rows with all important columns empty and excluding old events.
        This method performs the following steps:
        1. Replaces empty or whitespace-only strings in important columns with pandas NA values.
        2. Drops rows where all important columns ('start_date', 'end_date', 'start_time', 'end_time', 'location', 'description') are missing.
        3. Optionally removes events whose 'end_date' is older than the cutoff date,
           defined as current time minus config['clean_up']['old_events'] days.
        Args:
            df (pd.DataFrame): DataFrame containing event data.
        Returns:
            pd.DataFrame: Filtered DataFrame with only relevant and recent events.
        """
        logging.info(f"_filter_events: Input DataFrame has {len(df)} events")
        
        important_cols = ['start_date', 'end_date', 'start_time', 'end_time', 'location', 'description']
        
        # Log missing columns
        missing_cols = [col for col in important_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"_filter_events: Missing important columns: {missing_cols}")
        
        df[important_cols] = df[important_cols].replace(r'^\s*$', pd.NA, regex=True).infer_objects(copy=False)
        rows_before_dropna = len(df)
        df = df.dropna(subset=important_cols, how='all')
        rows_after_dropna = len(df)
        
        if rows_before_dropna != rows_after_dropna:
            logging.info(f"_filter_events: Dropped {rows_before_dropna - rows_after_dropna} rows with all important columns empty")

        filtered_df = df
        if apply_date_filter:
            filtered_df = self._drop_old_events_by_date(df, context="filter_events")
        
        logging.info(f"_filter_events: Output DataFrame has {len(filtered_df)} events")
        return filtered_df
    
    
    def update_event(self, event_identifier, new_data, best_url):
        """
        Update an existing event in the database by overlaying new data and setting the best URL.
        This method locates an event row in the 'events' table using the provided event_identifier criteria.
        It overlays the values from new_data onto the existing row, replacing only the fields present and non-empty in new_data.
        The event's URL is updated to the provided best_url. The update is performed in-place; if no matching event is found,
        the method logs an error and returns False.
            event_identifier (dict): Dictionary specifying the criteria to uniquely identify the event row.
                Example: {'event_name': ..., 'start_date': ..., 'start_time': ...}
            new_data (dict): Dictionary containing new values to update in the event record. Only non-empty and non-null
                values will overwrite existing fields.
            best_url (str): The URL to set as the event's 'url' field.
        Returns:
            bool: True if the event was found and updated successfully, False otherwise.
        Logs:
            - Error if no matching event is found.
            - Info when an event is successfully updated.
        """
        select_query = """
        SELECT * FROM events
        WHERE event_name = :event_name
        AND start_date = :start_date
        AND start_time = :start_time
        """
        result = self.execute_query(select_query, event_identifier)
        existing_row = result[0] if result else None
        if not existing_row:
            logging.error("update_event: No matching event found for identifier: %s", event_identifier)
            return False
        
        # Overlay new data onto existing row
        updated_data = dict(existing_row)
        for col, new_val in new_data.items():
            if new_val not in [None, '', pd.NaT]:
                updated_data[col] = new_val
        
        # Update URL
        updated_data['url'] = best_url

        update_cols = [col for col in updated_data.keys() if col != 'event_id']
        set_clause = ", ".join([f"{col} = :{col}" for col in update_cols])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        updated_data['event_id'] = existing_row['event_id']

        self.execute_query(update_query, updated_data)
        logging.info("update_event: Updated event %s", updated_data)
        return True


    def fuzzy_match(self, a: str, b: str, threshold: int = 85) -> bool:
        """
        Returns True if the fuzzy match score between two strings exceeds the threshold.
        Uses token sort ratio for better match on rearranged terms.
        """
        score = fuzz.token_sort_ratio(a, b)
        return score >= threshold

    @classmethod
    def _venue_tokens(cls, name: str) -> set[str]:
        """
        Tokenize venue/building names into meaningful comparison tokens.
        """
        if not name:
            return set()
        tokens = re.findall(r"[a-z0-9]+", str(name).lower())
        return {
            token for token in tokens
            if len(token) >= 3 and token not in cls._VENUE_TOKEN_STOPWORDS
        }

    @classmethod
    def _has_meaningful_token_overlap(cls, a: str, b: str) -> bool:
        """
        Require overlap on non-generic venue tokens to avoid near-miss false matches.

        Example prevented: "The Loft Victoria" vs "The Lab Victoria"
        (shared generic token 'victoria' only).
        """
        a_tokens = cls._venue_tokens(a)
        b_tokens = cls._venue_tokens(b)
        if not a_tokens or not b_tokens:
            # If either side has no meaningful tokens, do not block fuzzy fallback.
            return True
        return len(a_tokens.intersection(b_tokens)) > 0


        

    def match_civic_number(self, df, numbers):
        """
        Attempts to match the first numeric string from a list to the 'civic_no' column in a DataFrame of addresses.
        Parameters:
            df (pd.DataFrame): DataFrame containing address information, including a 'civic_no' column.
            numbers (list of str): List of numeric strings extracted from a location.
        Returns:
            int or None: The index of the row in the DataFrame where the first number matches the 'civic_no'.
                         If no match is found, returns the index of the first row.
                         Returns None if the DataFrame is empty.
        """
        if df.empty:
            logging.warning("match_civic_number(): Received empty DataFrame.")
            return None
        
        if not numbers:
            return df.index[0]
        for i, addr_row in df.iterrows():
            if addr_row.civic_no is not None:
                try:
                    if int(numbers[0]) == int(addr_row.civic_no):
                        return i
                except ValueError:
                    continue

        return df.index[0]

    def _get_building_name_dictionary(self):
        """
        Creates and caches a dictionary mapping building names to address_ids from the address table.
        
        Returns:
            dict: Dictionary with building_name (lowercase) as keys and address_id as values
        """
        if not hasattr(self, '_building_name_cache'):
            logging.info("_get_building_name_dictionary: Building building name lookup cache")
            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL AND building_name != ''"
            results = self.execute_query(query)
            
            self._building_name_cache = {}
            if results:
                for address_id, building_name in results:
                    if building_name and building_name.strip():
                        # Use lowercase for case-insensitive matching
                        self._building_name_cache[building_name.lower().strip()] = address_id
                        
            logging.info(f"_get_building_name_dictionary: Cached {len(self._building_name_cache)} building names")
        
        return self._building_name_cache

    def _extract_address_from_event_details(self, event: Dict[str, Any]) -> Optional[int]:
        """
        Attempts to extract building names from event_name and description, 
        then matches them against existing addresses in the database.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            int or None: address_id if a match is found, None otherwise
        """
        building_dict = self._get_building_name_dictionary()
        if not building_dict:
            return None
            
        # Collect text to search from event details
        search_texts = []
        event_name = event.get("event_name", "")
        description = event.get("description", "")
        
        if event_name:
            search_texts.append(str(event_name))
        if description:
            search_texts.append(str(description))
            
        if not search_texts:
            return None
            
        # Search for building names in the text
        combined_text = " ".join(search_texts).lower()
        
        # First try exact matches
        for building_name, address_id in building_dict.items():
            if building_name in combined_text:
                logging.info(f"_extract_address_from_event_details: Found exact match '{building_name}' -> address_id={address_id}")
                return address_id
                
        # Then try fuzzy matching for partial matches
        best_match = None
        best_score = 0
        
        for building_name, address_id in building_dict.items():
            # Skip very short building names for fuzzy matching to avoid false positives
            if len(building_name) < 6:
                continue
                
            score = fuzz.partial_ratio(building_name, combined_text)
            if score > 80 and score > best_score:  # High threshold for fuzzy matching
                best_score = score
                best_match = (building_name, address_id)
                
        if best_match:
            building_name, address_id = best_match
            logging.info(f"_extract_address_from_event_details: Found fuzzy match '{building_name}' (score: {best_score}) -> address_id={address_id}")
            return address_id
            
        return None

    def process_event_address(self, event: dict) -> dict:
        """
        Uses the LLM to parse a structured address from the location, inserts or reuses the address in the DB,
        and updates the event with address_id and location = full_address from address table.
        """
        location = event.get("location", None)
        event_name = event.get("event_name", "Unknown Event")
        source = event.get("source", "Unknown Source")

        if location is None:
            pass  # Keep it as None
        elif isinstance(location, str):
            location = location.strip()

        # Handle case where location might be NaN (float), empty string, or 'Unknown'
        if (location is None or pd.isna(location) or not isinstance(location, str) or
            len(location) < 5 or 'Unknown' in str(location)):
            logging.info(
                "process_event_address: Location missing/invalid for event '%s' from %s, attempting building name extraction",
                event_name, source
            )

            # Try to extract building name from event details and match to existing addresses
            extracted_address_id = self._extract_address_from_event_details(event)
            if extracted_address_id:
                event["address_id"] = extracted_address_id
                full_address = self.get_full_address_from_id(extracted_address_id)
                if full_address:
                    event["location"] = full_address
                    logging.info(f"process_event_address: Found existing address via building name extraction: address_id={extracted_address_id}")
                    return event

            # DEDUPLICATION CHECK: Before creating a new address, check if source/event_name matches existing building
            dedup_addr_id = self.find_address_by_building_name(source, threshold=75)
            if dedup_addr_id:
                event["address_id"] = dedup_addr_id
                full_address = self.get_full_address_from_id(dedup_addr_id)
                if full_address:
                    event["location"] = full_address
                    logging.info(f"process_event_address: Found existing address via deduplication check: source='{source}' → address_id={dedup_addr_id}")
                    return event

            # If no match found, create a minimal but valid address entry
            minimal_address = {
                "address_id": 0,
                "full_address": f"Location details unavailable - {source}",
                "building_name": str(event_name)[:50],  # Use event name as building
                "street_number": "",
                "street_name": "",
                "street_type": "",
                "direction": None,
                "city": "Unknown",
                "met_area": None,
                "province_or_state": "BC", 
                "postal_code": None,
                "country_id": "CA",
                "time_stamp": None
            }
            
            address_id = self.resolve_or_insert_address(minimal_address)
            if address_id:
                event["address_id"] = address_id
                full_address = self.get_full_address_from_id(address_id)
                if full_address:
                    event["location"] = full_address
                else:
                    event["location"] = minimal_address["full_address"]  # Fallback to our description
                logging.info(f"process_event_address: Created minimal address entry with address_id={address_id}")
                return event
            else:
                logging.error("process_event_address: Failed to create minimal address entry, setting default values")
                # Keep nullable semantics for unresolved addresses (avoid pseudo-id 0).
                event["address_id"] = None
                event["location"] = f"Location unavailable - {source}"
                return event

        # Normalize known venue aliases before cache and fuzzy lookup to enforce canonical address.
        alias_context = {
            "source": source,
            "url": event.get("url", ""),
        }
        location_alias_match = self._find_address_alias_match([location], context=alias_context)
        if location_alias_match:
            location_alias_canonical = location_alias_match["canonical"]
            alias_address_id = self._get_alias_canonical_address_id(location_alias_canonical)
            if alias_address_id:
                event["address_id"] = alias_address_id
                canonical_full_address = location_alias_canonical.get("full_address")
                db_full_address = self.get_full_address_from_id(alias_address_id)
                event["location"] = db_full_address or canonical_full_address or location
                self.cache_raw_location(location, alias_address_id)
                self._audit_address_alias_hit({
                    "decision": "applied",
                    "rule_name": location_alias_match.get("rule_name"),
                    "match_type": location_alias_match.get("match_type"),
                    "matched_alias": location_alias_match.get("matched_alias"),
                    "candidate": location_alias_match.get("candidate"),
                    "canonical_address_id": alias_address_id,
                    "canonical_full_address": location_alias_canonical.get("full_address"),
                    "url": event.get("url", ""),
                    "source": source,
                })
                logging.info(
                    "process_event_address: Alias match forced canonical venue for '%s' -> address_id=%s",
                    location,
                    alias_address_id,
                )
                return event

        # STEP 1: Check raw_locations cache (fastest - exact string match)
        cached_addr_id = self.lookup_raw_location(location)
        if cached_addr_id:
            full_address = self.get_full_address_from_id(cached_addr_id)
            if self._address_text_supports_candidate(
                location,
                {"full_address": full_address},
            ):
                event["address_id"] = cached_addr_id
                if full_address:
                    event["location"] = full_address
                logging.info(f"process_event_address: Cache hit for '{location}' → address_id={cached_addr_id}")
                return event
            logging.info(
                "process_event_address: Ignoring low-confidence cache hit for '%s' → address_id=%s",
                location,
                cached_addr_id,
            )

        # STEP 2: Try intelligent address parsing (fuzzy matching, regex)
        quick_addr_id = self.quick_address_lookup(location)
        if quick_addr_id:
            full_address = self.get_full_address_from_id(quick_addr_id)
            if self._address_text_supports_candidate(
                location,
                {"full_address": full_address},
            ):
                # Cache this mapping for future use
                self.cache_raw_location(location, quick_addr_id)
                event["address_id"] = quick_addr_id
                if full_address:
                    event["location"] = full_address
                logging.info(f"process_event_address: Quick lookup found address_id={quick_addr_id} for '{location}'")
                return event
            logging.info(
                "process_event_address: Ignoring low-confidence quick lookup for '%s' → address_id=%s",
                location,
                quick_addr_id,
            )

        # STEP 3: LLM processing (last resort)
        # Generate the LLM prompt
        prompt, schema_type = self.llm_handler.generate_prompt(event.get("url", "address_fix"), location, "address_internet_fix")

        # Query the LLM
        llm_response = self.llm_handler.query_llm((event.get("url") or "").strip(), prompt, schema_type)

        # Parse the LLM response into a usable dict
        parsed_results = self.llm_handler.extract_and_parse_json(llm_response, "address_fix", schema_type)
        if not parsed_results or not isinstance(parsed_results, list) or not isinstance(parsed_results[0], dict):
            logging.warning("process_event_address: Could not parse address from LLM response, creating minimal address")
            # Create minimal address using event name and location
            event_name = event.get("event_name") or "Unknown Event"
            minimal_address = {
                "building_name": str(event_name)[:50],
                "street_name": location[:50] if location else "Unknown Location",
                "city": "Unknown",
                "province_or_state": "BC",
                "country_id": "Canada"
            }
            address_id = self.resolve_or_insert_address(minimal_address)
            if address_id:
                event["address_id"] = address_id
                return event
            else:
                logging.error("process_event_address: Failed to create minimal address")
                return event

        # ✅ Normalize null-like strings in one place
        parsed_address = self.normalize_nulls(parsed_results[0])

        # Step 4: Get or insert address_id
        address_id = self.resolve_or_insert_address(parsed_address)
        
        # Ensure we got a valid address_id  
        if not address_id:
            logging.warning("process_event_address: resolve_or_insert_address failed, creating minimal address as last resort")
            # Last resort: create very minimal address entry
            event_name = event.get("event_name") or "Event"
            minimal_address = {
                "building_name": str(event_name)[:50],
                "city": "Location Unknown", 
                "province_or_state": "BC",
                "country_id": "Canada"
            }
            address_id = self.resolve_or_insert_address(minimal_address)
            if not address_id:
                logging.error("process_event_address: All address resolution attempts failed")
                return event

        # STEP 3: Cache the raw location → address_id mapping for future use
        self.cache_raw_location(location, address_id)

        # Step 5: Force consistency: always use address.full_address
        full_address = self.get_full_address_from_id(address_id)
        event["address_id"] = address_id
        if full_address:
            event["location"] = full_address

        return event

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        """Coerce value to int when valid, else return None."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            if pd.isna(value):
                return None
            if float(value).is_integer():
                int_value = int(value)
                return int_value if int_value > 0 else None
            return None
        value_str = str(value).strip()
        if not value_str:
            return None
        if re.fullmatch(r"\d+", value_str):
            int_value = int(value_str)
            return int_value if int_value > 0 else None
        return None

    def _sanitize_events_dataframe_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize event payload types before INSERT.

        - Keep only known event table columns.
        - Coerce address_id to nullable positive integer.
        - Force text columns to plain strings where values are non-null.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        expected_columns = [
            "event_name", "dance_style", "description", "day_of_week",
            "start_date", "end_date", "start_time", "end_time", "source",
            "location", "price", "url", "event_type", "address_id", "time_stamp",
        ]
        working_df = df.copy()

        for col in expected_columns:
            if col not in working_df.columns:
                working_df[col] = pd.NA
        working_df = working_df[expected_columns]

        working_df["address_id"] = working_df["address_id"].apply(self._coerce_optional_int)

        text_columns = [
            "event_name", "dance_style", "description", "day_of_week",
            "source", "location", "price", "url", "event_type",
        ]
        for col in text_columns:
            working_df[col] = working_df[col].apply(
                lambda value: None if pd.isna(value) else str(value)
            )

        invalid_addr_count = int(working_df["address_id"].isna().sum())
        if invalid_addr_count:
            logging.info(
                "_sanitize_events_dataframe_for_insert: Normalized %d events to NULL address_id.",
                invalid_addr_count,
            )

        return working_df


    def find_address_by_building_name(self, building_name: str, threshold: int = 75) -> Optional[int]:
        """
        Find an existing address by fuzzy matching on building_name.
        Prevents creation of duplicate addresses with the same venue name.

        Args:
            building_name (str): The venue/building name to search for
            threshold (int): Fuzzy match score threshold (0-100)

        Returns:
            address_id if found, None otherwise
        """
        from fuzzywuzzy import fuzz

        if not building_name or not isinstance(building_name, str):
            return None

        building_name = building_name.strip()

        try:
            # Query all addresses with building names
            building_matches = self.execute_query(
                "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            )

            best_score = 0
            best_addr_id = None

            for addr_id, existing_building in building_matches or []:
                if existing_building and existing_building.strip():
                    if not self._has_meaningful_token_overlap(building_name, existing_building):
                        continue
                    # Use partial_ratio which is more lenient for substring matches
                    score = fuzz.partial_ratio(building_name.lower().strip(), existing_building.lower().strip())

                    if score >= threshold and score > best_score:
                        best_score = score
                        best_addr_id = addr_id
                        logging.debug(f"find_address_by_building_name: '{building_name}' vs '{existing_building}' = {score}")

            if best_addr_id:
                logging.info(f"find_address_by_building_name: Found address_id={best_addr_id} for '{building_name}' (score={best_score})")
                return best_addr_id

            logging.debug(f"find_address_by_building_name: No match found for '{building_name}'")
            return None

        except Exception as e:
            logging.warning(f"find_address_by_building_name: Error looking up '{building_name}': {e}")
            return None

    def quick_address_lookup(self, location: str) -> Optional[int]:
        """
        Attempts to find an existing address without using LLM by:
        1. Exact string match on full_address
        2. Regex parsing to extract street_number + street_name for exact match
        3. Fuzzy matching on building names for the same street
        
        Returns address_id if found, None if LLM is needed
        """
        from fuzzywuzzy import fuzz
        import re
        
        # Step 1: Exact string match (already implemented)
        exact_match = self.execute_query(
            "SELECT address_id FROM address WHERE LOWER(full_address) = LOWER(:location)",
            {"location": location}
        )
        if exact_match:
            logging.info(f"quick_address_lookup: Exact match → address_id={exact_match[0][0]}")
            return exact_match[0][0]
        
        # Step 2: Parse basic components with regex
        street_pattern = r'(\d+)\s+([A-Za-z\s]+?)(?:,|\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd))'
        street_match = re.search(street_pattern, location, re.IGNORECASE)
        
        if street_match:
            street_number = street_match.group(1).strip()
            street_name_raw = street_match.group(2).strip()
            
            # Clean street name (remove common suffixes if they got included)
            street_name = re.sub(r'\b(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd)\b', 
                               '', street_name_raw, flags=re.IGNORECASE).strip()
            
            # Step 3: Find addresses with same street_number + street_name
            street_matches = self.execute_query("""
                SELECT address_id, building_name, full_address 
                FROM address 
                WHERE LOWER(street_number) = LOWER(:street_number) 
                AND LOWER(street_name) = LOWER(:street_name)
            """, {"street_number": street_number, "street_name": street_name})
            
            if street_matches:
                # Step 4: If only one match, use it regardless of building name
                if len(street_matches) == 1:
                    addr_id, building_name, full_addr = street_matches[0]
                    logging.info(f"quick_address_lookup: Single street match → address_id={addr_id}")
                    return addr_id
                
                # Step 5: Try fuzzy matching on building names
                building_pattern = r'^([^,\d]+?)(?:,|\s+\d+)'  # Text before first comma or number
                building_match = re.search(building_pattern, location.strip())
                
                if building_match:
                    location_building = building_match.group(1).strip()
                    
                    best_score = 0
                    best_addr_id = None
                    
                    for addr_id, existing_building, full_addr in street_matches:
                        if existing_building and existing_building.strip():
                            if not self._has_meaningful_token_overlap(location_building, existing_building):
                                continue
                            score = fuzz.ratio(location_building.lower(), existing_building.lower())
                            if score >= 85 and score > best_score:
                                best_score = score
                                best_addr_id = addr_id
                    
                    if best_addr_id:
                        logging.info(f"quick_address_lookup: Fuzzy building match (score={best_score}) → address_id={best_addr_id}")
                        return best_addr_id
        
        # Step 6: Fuzzy match on building names for locations without street numbers
        if not street_match:
            building_matches = self.execute_query(
                "SELECT address_id, building_name, full_address FROM address WHERE building_name IS NOT NULL"
            )
            
            best_score = 0
            best_addr_id = None
            
            for addr_id, building_name, full_addr in building_matches or []:
                if building_name and building_name.strip():
                    if not self._has_meaningful_token_overlap(location, building_name):
                        continue
                    # Check if location is contained in building name or vice versa
                    score = fuzz.ratio(location.lower().strip(), building_name.lower().strip())
                    partial_score = fuzz.partial_ratio(location.lower().strip(), building_name.lower().strip())
                    
                    # Use the higher score
                    final_score = max(score, partial_score)
                    
                    if final_score >= 80 and final_score > best_score:
                        best_score = final_score
                        best_addr_id = addr_id
                        logging.debug(f"Building match candidate: '{location}' vs '{building_name}' = {final_score}")
            
            if best_addr_id:
                logging.info(f"quick_address_lookup: Fuzzy building name match (score={best_score}) → address_id={best_addr_id}")
                return best_addr_id
        
        # Step 7: Last resort - fuzzy match on full addresses for very similar ones
        all_addresses = self.execute_query("SELECT address_id, full_address, building_name FROM address")
        for addr_id, full_addr, building_name in all_addresses or []:
            if full_addr and fuzz.ratio(location.lower(), full_addr.lower()) >= 90:
                if building_name and not self._has_meaningful_token_overlap(location, building_name):
                    continue
                logging.info(f"quick_address_lookup: Fuzzy full address match → address_id={addr_id}")
                return addr_id
        
        logging.info(f"quick_address_lookup: No match found for '{location}', LLM required")
        return None

    def cache_raw_location(self, raw_location: str, address_id: int):
        """
        Cache a raw location string to address_id mapping for fast future lookups.
        Uses PostgreSQL ON CONFLICT to avoid duplicate key errors.
        """
        try:
            # PostgreSQL syntax: INSERT ... ON CONFLICT DO NOTHING
            insert_query = """
                INSERT INTO raw_locations (raw_location, address_id, created_at)
                VALUES (:raw_location, :address_id, :created_at)
                ON CONFLICT (raw_location) DO NOTHING
            """
            result = self.execute_query(insert_query, {
                "raw_location": raw_location,
                "address_id": address_id,
                "created_at": datetime.now()
            })
            logging.info(f"cache_raw_location: Cached '{raw_location}' → address_id={address_id}")
        except Exception as e:
            logging.warning(f"cache_raw_location: Failed to cache '{raw_location}': {e}")

    def lookup_raw_location(self, raw_location: str) -> Optional[int]:
        """
        Look up a raw location string in the cache to get its address_id.
        Returns address_id if found, None if not cached.
        """
        try:
            result = self.execute_query(
                """
                SELECT rl.address_id, a.building_name, a.street_number, a.street_name, a.direction, a.full_address
                FROM raw_locations rl
                LEFT JOIN address a ON a.address_id = rl.address_id
                WHERE rl.raw_location = :raw_location
                """,
                {"raw_location": raw_location}
            )
            if result:
                address_id, building_name, street_number, street_name, direction, full_address = result[0]
                candidate = {
                    "building_name": building_name,
                    "street_number": street_number,
                    "street_name": street_name,
                    "direction": direction,
                    "full_address": full_address,
                }
                if not self._address_text_supports_candidate(raw_location, candidate):
                    warning_key = (
                        str(raw_location),
                        int(address_id) if address_id is not None else None,
                        str(building_name or full_address),
                    )
                    if warning_key not in self._stale_raw_location_warnings:
                        logging.warning(
                            "lookup_raw_location: Ignoring stale/mismatched cache mapping '%s' -> address_id=%s (%s)",
                            raw_location,
                            address_id,
                            building_name or full_address,
                        )
                        self._stale_raw_location_warnings.add(warning_key)
                    try:
                        self.execute_query(
                            "DELETE FROM raw_locations WHERE raw_location = :raw_location",
                            {"raw_location": raw_location},
                        )
                    except Exception as cleanup_err:
                        logging.warning(
                            "lookup_raw_location: Failed to remove stale cache mapping '%s': %s",
                            raw_location,
                            cleanup_err,
                        )
                    return None
                logging.info(f"lookup_raw_location: Cache hit for '{raw_location}' → address_id={address_id}")
                return address_id
            return None
        except Exception as e:
            logging.warning(f"lookup_raw_location: Cache lookup failed for '{raw_location}': {e}")
            return None

    def create_raw_locations_table(self):
        """
        Create the raw_locations table for caching location string to address_id mappings.
        Creates address table first if it doesn't exist to satisfy foreign key constraint.
        """
        # First ensure address table exists for foreign key constraint
        address_table_query = """
            CREATE TABLE IF NOT EXISTS address (
                address_id SERIAL PRIMARY KEY,
                full_address TEXT UNIQUE,
                building_name TEXT,
                street_number TEXT,
                street_name TEXT,
                street_type TEXT,
                direction TEXT,
                city TEXT,
                met_area TEXT,
                province_or_state TEXT,
                postal_code TEXT,
                country_id TEXT,
                time_stamp TIMESTAMP
            )
        """

        # PostgreSQL syntax (not SQLite)
        create_table_query = """
            CREATE TABLE IF NOT EXISTS raw_locations (
                raw_location_id SERIAL PRIMARY KEY,
                raw_location TEXT NOT NULL UNIQUE,
                address_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (address_id) REFERENCES address(address_id)
            )
        """
        try:
            # Create address table first
            self.execute_query(address_table_query)
            logging.info("create_raw_locations_table: Address table creation/verification completed")

            # Then create raw_locations table with foreign key
            self.execute_query(create_table_query)
            logging.info("create_raw_locations_table: Raw locations table creation/verification completed")
            
            # Create index for faster lookups
            index_query = "CREATE INDEX IF NOT EXISTS idx_raw_location ON raw_locations(raw_location)"
            self.execute_query(index_query)
            logging.info("create_raw_locations_table: Index creation/verification completed")
        except Exception as e:
            logging.error(f"create_raw_locations_table: Failed to create table: {e}")
            logging.error(f"create_raw_locations_table: SQL was: {create_table_query}")
    

    def sync_event_locations_with_address_table(self):
        """
        Updates all events so that location = full_address from the address table for consistency.
        """
        query = """
            UPDATE events e
            SET location = a.full_address
            FROM address a
            WHERE e.address_id = a.address_id
            AND (e.location IS DISTINCT FROM a.full_address);
        """
        affected_rows = self.execute_query(query)
        logging.info(f"sync_event_locations_with_address_table(): Updated {affected_rows} events to use canonical full_address.")

    def clean_orphaned_references(self):
        """
        Clean up orphaned references in related tables to maintain referential integrity.

        This function removes:
        1. raw_locations records that reference non-existent addresses
        2. events records that reference non-existent addresses (if any)

        Returns:
            dict: Count of cleaned up records by table
        """
        cleanup_counts = {}

        try:
            # Clean up orphaned raw_locations
            cleanup_raw_locations_sql = """
            DELETE FROM raw_locations
            WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            raw_locations_count = self.execute_query(cleanup_raw_locations_sql)
            cleanup_counts['raw_locations'] = raw_locations_count or 0

            # Clean up events with non-existent address_ids (should be rare)
            cleanup_events_sql = """
            DELETE FROM events
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            deleted_events = self._delete_events_with_audit(
                delete_sql_without_returning=cleanup_events_sql,
                params=None,
                deletion_source="db.clean_orphaned_references",
                reason="orphaned_address_id_reference",
            )
            cleanup_counts['events'] = len(deleted_events)

            # Clean up events_history with non-existent address_ids (critical for preventing corruption)
            cleanup_events_history_sql = """
            DELETE FROM events_history
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            events_history_count = self.execute_query(cleanup_events_history_sql)
            cleanup_counts['events_history'] = events_history_count or 0

            logging.info(f"clean_orphaned_references(): Cleaned up {cleanup_counts['raw_locations']} raw_locations, {cleanup_counts['events']} events, and {cleanup_counts['events_history']} events_history records with orphaned address references")

        except Exception as e:
            logging.error(f"clean_orphaned_references(): Error cleaning orphaned references: {e}")

        return cleanup_counts

    def dedup(self):
        """
        Removes duplicate entries from the 'events' table in the database.

        Duplicates in the 'events' table are identified based on matching 'address_id', 'start_date', 'end_date',
        and start/end times within 15 minutes (900 seconds) of each other. Only the latest entry (with the highest event_id)
        is retained for each group of duplicates; all others are deleted.

        Returns:
            int: The number of rows deleted from the 'events' table during deduplication.

        Raises:
            Exception: If an error occurs during the deduplication process, it is logged and re-raised.
        """
        try:
            # Deduplicate 'events' table based on 'Name_of_the_Event' and 'Start_Date'
            dedup_events_query = """
                DELETE FROM events e1
                USING events e2
                WHERE e1.event_id < e2.event_id
                    AND e1.address_id = e2.address_id
                    AND e1.start_date = e2.start_date
                    AND e1.end_date = e2.end_date
                    AND ABS(EXTRACT(EPOCH FROM (e1.start_time - e2.start_time))) <= 900
                    AND ABS(EXTRACT(EPOCH FROM (e1.end_time - e2.end_time))) <= 900;
            """
            deleted_rows = self._delete_events_with_audit(
                delete_sql_without_returning=dedup_events_query,
                params=None,
                deletion_source="db.dedup",
                reason="exact_time_window_duplicate",
            )
            deleted_count = len(deleted_rows)
            logging.info("def dedup(): Deduplicated events table successfully. Rows deleted: %d", deleted_count)

            # Clean up any orphaned references that might have been created
            self.clean_orphaned_references()

        except Exception as e:
            logging.error("def dedup(): Failed to deduplicate tables: %s", e)


    def fetch_events_dataframe(self):
        """
        Fetch all events from the database and return them as a pandas DataFrame sorted by start date and time.

        Returns:
            pandas.DataFrame: A DataFrame containing all events from the 'events' table,
            sorted by 'start_date' and 'start_time' columns.
        """
        query = "SELECT * FROM events"
        df = self.read_sql_df(query)
        df.sort_values(by=['start_date', 'start_time'], inplace=True)
        return df

    def decide_preferred_row(self, row1, row2):
        """
        Determines the preferred row between two given rows based on the following criteria:
            1. Prefer the row whose declared source clearly matches its own URL host.
            2. Otherwise prefer the row with a non-empty 'url' field.
            3. If both or neither have a 'url', prefer the row with more filled (non-empty) columns, excluding 'event_id'.
            4. If still tied, prefer the row with the most recent 'time_stamp'.
        Args:
            row1 (pandas.Series): The first row to compare.
            row2 (pandas.Series): The second row to compare.
            tuple:
                preferred_row (pandas.Series): The row selected as preferred based on the criteria.
                other_row (pandas.Series): The other row that was not preferred.
        """
        row1_source_matches_url = self._source_matches_event_url(row1.get('source'), row1.get('url'))
        row2_source_matches_url = self._source_matches_event_url(row2.get('source'), row2.get('url'))
        if row1_source_matches_url and not row2_source_matches_url:
            return row1, row2
        if row2_source_matches_url and not row1_source_matches_url:
            return row2, row1

        # Prefer row with URL
        if row1['url'] and not row2['url']:
            return row1, row2
        if row2['url'] and not row1['url']:
            return row2, row1

        # Count filled columns (excluding event_id)
        filled_columns = lambda row: row.drop(labels='event_id').count()
        count1 = filled_columns(row1)
        count2 = filled_columns(row2)
        
        if count1 > count2:
            return row1, row2
        elif count2 > count1:
            return row2, row1
        else:
            # If still tied, choose the most recent based on time_stamp
            if row1['time_stamp'] >= row2['time_stamp']:
                return row1, row2
            else:
                return row2, row1

    def update_preferred_row_from_other(self, preferred, other, columns):
        """
        Update missing values in the preferred row with corresponding values from the other row for specified columns.

        For each column in `columns`, if the value in `preferred` is missing (NaN or empty string), 
        and the value in `other` is present (not NaN and not empty string), the value from `other` 
        is copied to `preferred`.

        Args:
            preferred (pd.Series): The row to be updated, typically the preferred or primary row.
            other (pd.Series): The row to use as a source for missing values.
            columns (Iterable[str]): List or iterable of column names to check and update.

        Returns:
            pd.Series: The updated preferred row with missing values filled from the other row where applicable.
        """
        for col in columns:
            if pd.isna(preferred[col]) or preferred[col] == '':
                if not (pd.isna(other[col]) or other[col] == ''):
                    preferred[col] = other[col]
        return preferred


    def fuzzy_duplicates(self):
        """
        Identifies and removes fuzzy duplicate events from the events table in the database.
        This method performs the following steps:
            1. Fetches all events and sorts them by 'start_date' and 'start_time'.
            2. Groups events that share the same 'start_date' and 'start_time'.
            3. Within each group, compares event names using fuzzy string matching.
            4. If two events have a fuzzy match score greater than 80:
                a. Determines which event to keep based on the following criteria:
                    i. Prefer the event with a URL.
                    ii. If neither has a URL, prefer the event with more filled columns.
                    iii. If still tied, prefer the most recently updated event.
                b. Updates the kept event with any missing data from the duplicate.
                c. Updates the database with the merged event and deletes the duplicate.
            5. Logs actions taken during the process.
        Returns:
            None. The method updates the database in place and logs the actions performed.
        """
        # Fetch and sort events using self.conn
        events_df = self.fetch_events_dataframe()
        grouped = events_df.groupby(['start_date', 'start_time'])

        for _, group in grouped:
            if len(group) < 2:
                continue  # Skip if no duplicates possible

            # Check for duplicates within the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row_i = group.iloc[i].copy()
                    row_j = group.iloc[j].copy()
                    score = fuzz.ratio(row_i['event_name'], row_j['event_name'])
                    if score > 80:
                        # Decide which row to keep
                        preferred, other = self.decide_preferred_row(row_i, row_j)

                        # Update preferred row with missing columns from other row
                        preferred = self.update_preferred_row_from_other(preferred, other, events_df.columns)

                        # Prepare update query parameters
                        update_columns = [col for col in events_df.columns if col != 'event_id']
                        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
                        update_query = f"""
                            UPDATE events
                            SET {set_clause}
                            WHERE event_id = :event_id
                        """

                        update_params = {}
                        for col in update_columns:
                            value = preferred[col]
                            if isinstance(value, (np.generic, np.ndarray)):
                                try:
                                    value = value.item() if hasattr(value, 'item') else value.tolist()
                                except Exception:
                                    pass
                            update_params[col] = value
                        update_params['event_id'] = int(preferred['event_id'])

                        # Execute update query using self.execute_query
                        self.execute_query(update_query, update_params)
                        logging.info("fuzzy_duplicates: Kept row with event_id %s", preferred['event_id'])

                        # Delete duplicate row from database
                        delete_query = "DELETE FROM events WHERE event_id = :event_id"
                        self._delete_events_with_audit(
                            delete_sql_without_returning=delete_query,
                            params={'event_id': int(other['event_id'])},
                            deletion_source="db.fuzzy_duplicates",
                            reason="fuzzy_duplicate_merged",
                            extra_context={
                                "kept_event_id": int(preferred['event_id']),
                                "deleted_event_id": int(other['event_id']),
                                "fuzzy_score": int(score),
                            },
                        )
                        logging.info("fuzzy_duplicates: Deleted duplicate row with event_id %s", other['event_id'])

        logging.info("fuzzy_duplicates: Fuzzy duplicate removal completed successfully.")


    def delete_old_events(self):
        """
        The number of days is retrieved from the configuration under 'clean_up' -> 'old_events'.
        Events with an 'End_Date' earlier than the current date minus the specified number of days are deleted.

        Returns:
            int: The number of events deleted from the database.

        Raises:
            Exception: If an error occurs during the deletion process, it is logged and re-raised.
        """
        try:
            days = int(self.config['clean_up']['old_events'])
            delete_query = """
                DELETE FROM events
                WHERE End_Date < CURRENT_DATE - INTERVAL '%s days';
            """ % days
            deleted_rows = self._delete_events_with_audit(
                delete_sql_without_returning=delete_query,
                params=None,
                deletion_source="db.delete_old_events",
                reason=f"end_date_older_than_{days}_days",
                extra_context={"days": days},
            )
            deleted_count = len(deleted_rows)
            logging.info(f"delete_old_events: Deleted {deleted_count} events older than {days} days.")
        except Exception as e:
            logging.error("delete_old_events: Failed to delete old events: %s", e)


    def delete_likely_dud_events(self):
        """
        Deletes likely invalid or irrelevant events from the database based on several criteria:
        1. Deletes events where 'source', 'dance_style', and 'url' are empty strings, unless the event has an associated 'address_id'.
        2. Deletes events whose associated address (if present) has a 'province_or_state' that is not 'BC' (British Columbia).
        3. Deletes events whose associated address (if present) has a 'country_id' that is not 'CA' (Canada).
        4. Deletes events where 'dance_style' and 'url' are empty strings, 'event_type' is 'other', and both 'location' and 'description' are NULL.
        For each deletion step, logs the number of events deleted.
        Returns:
            None
        """
        # 1. Delete events where source, dance_style, and url are empty, unless they have an address_id
        delete_query_1 = """
        DELETE FROM events
        WHERE source = :source
        AND dance_style = :dance_style
        AND url = :url
        AND address_id IS NULL;
        """
        params = {
            'source': '',
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }

        deleted_events = self._delete_events_with_audit(
            delete_sql_without_returning=delete_query_1,
            params=params,
            deletion_source="db.delete_likely_dud_events",
            reason="empty_source_dance_style_url_no_address",
        )
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events with empty source, dance_style, and url, and no address_id.", deleted_count)

        # 2. Delete events outside of British Columbia (BC)
        delete_query_2 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE province_or_state IS NOT NULL
            AND province_or_state != :province_or_state
        );
        """
        params = {
            'province_or_state': 'BC'
            }
        
        deleted_events = self._delete_events_with_audit(
            delete_sql_without_returning=delete_query_2,
            params=params,
            deletion_source="db.delete_likely_dud_events",
            reason="outside_bc_filter",
            extra_context={"province_or_state_expected": "BC"},
        )
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events outside of British Columbia (BC).", deleted_count)

        # 3. Delete events that are not in Canada
        delete_query_3 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE country_id IS NOT NULL
            AND country_id != :country_id
        );
        """
        params = {
            'country_id': 'CA'
            }
        
        deleted_events = self._delete_events_with_audit(
            delete_sql_without_returning=delete_query_3,
            params=params,
            deletion_source="db.delete_likely_dud_events",
            reason="outside_canada_filter",
            extra_context={"country_expected": "CA"},
        )
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events that are not in Canada (CA).", deleted_count)

        # 4. Delete rows in events where dance_style and url are == '' AND event_type == 'other' AND location IS NULL and description IS NULL
        delete_query_4 = """
        DELETE FROM events
        WHERE dance_style = :dance_style
            AND url = :url
            AND event_type = :event_type
            AND location IS NULL
            AND description IS NULL;
        """
        params = {
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }
        
        deleted_events = self._delete_events_with_audit(
            delete_sql_without_returning=delete_query_4,
            params=params,
            deletion_source="db.delete_likely_dud_events",
            reason="empty_dance_style_url_other_no_location_description",
        )
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info(
            "def delete_likely_dud_events(): Deleted %d events with empty "
            "dance_style, url, event_type 'other', and null location "
            "and description.",
            deleted_count
        )
        

    def delete_event(self, url, event_name, start_date):
        """
        Deletes an event from the 'events' table based on the provided event name and start date.

            url (str): The URL of the event to be deleted. (Note: This parameter is currently unused in the deletion query.)

        Returns:
            None

        Raises:
            Exception: If an error occurs during the deletion process, it is logged and the exception is propagated.
        """
        try:
            logging.info("delete_event: Deleting event with URL: %s, Event Name: %s, Start Date: %s", url, event_name, start_date)

            # Delete the event from 'events' table
            delete_event_query = """
                DELETE FROM events
                WHERE Name_of_the_Event = :event_name
                  AND Start_Date = :start_date
            """
            params = {'event_name': event_name, 'start_date': start_date}
            self._delete_events_with_audit(
                delete_sql_without_returning=delete_event_query,
                params=params,
                deletion_source="db.delete_event",
                reason="manual_delete_by_name_and_start_date",
                extra_context={"url_arg": url},
            )
            logging.info("delete_event: Deleted event from 'events' table.")

        except Exception as e:
            logging.error("delete_event: Failed to delete event: %s", e)


    def delete_events_with_nulls(self):
        """
        Deletes events from the 'events' table where both 'start_date' and 'start_time' are NULL,
        or both 'start_time' and 'end_time' are NULL.

        Returns:
            int: The number of events deleted from the table.

        Raises:
            Exception: If an error occurs during the deletion process.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE (start_date IS NULL AND start_time IS NULL) OR 
            (start_time IS NULL AND end_time IS NULL)
            """
            deleted_rows = self._delete_events_with_audit(
                delete_sql_without_returning=delete_query,
                params=None,
                deletion_source="db.delete_events_with_nulls",
                reason="null_start_date_start_time_or_null_start_end_time",
            )
            deleted_count = len(deleted_rows)
            logging.info("def delete_events_with_nulls(): Deleted %d events where (start_date and start_time are null).", deleted_count)
        except Exception as e:
            logging.error("def delete_events_with_nulls(): Failed to delete events with nulls: %s", e)


    def delete_event_with_event_id(
        self,
        event_id,
        deletion_source: str = "db.delete_event_with_event_id",
        deletion_reason: str = "delete_by_event_id",
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Deletes an event from the 'events' table based on the provided event_id.

        Args:
            event_id (int): The unique identifier of the event to be deleted.

        Returns:
            None

        Raises:
            Exception: If the deletion fails, an exception is logged and propagated.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE event_id = :event_id
            """
            params = {'event_id': event_id}
            merged_context: Dict[str, Any] = {"event_id": int(event_id)}
            if extra_context:
                merged_context.update(extra_context)

            self._delete_events_with_audit(
                delete_sql_without_returning=delete_query,
                params=params,
                deletion_source=deletion_source,
                reason=deletion_reason,
                extra_context=merged_context,
            )
            logging.info("delete_event_with_event_id: Deleted event with event_id %d successfully.", event_id)
        except Exception as e:
            logging.error("delete_event_with_event_id: Failed to delete event with event_id %d: %s", event_id, e)

    def delete_events_by_filter(
        self,
        where_clause: str,
        params: Optional[Dict[str, Any]] = None,
        deletion_source: str = "db.delete_events_by_filter",
        deletion_reason: str = "delete_by_filter",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete events matching a WHERE clause and audit every deleted row.

        Args:
            where_clause: SQL predicate used after `WHERE` in a DELETE statement.
            params: Optional bound parameters for the predicate.
            deletion_source: Logical source identifier for audit trail.
            deletion_reason: Short reason tag for audit trail.
            extra_context: Optional JSON-serializable context attached to each audit row.

        Returns:
            Number of deleted rows.
        """
        try:
            predicate = (where_clause or "").strip().strip(";")
            if not predicate:
                logging.warning("delete_events_by_filter: Empty where_clause, skipping delete.")
                return 0

            delete_query = f"""
            DELETE FROM events
            WHERE {predicate}
            """
            deleted_rows = self._delete_events_with_audit(
                delete_sql_without_returning=delete_query,
                params=params or {},
                deletion_source=deletion_source,
                reason=deletion_reason,
                extra_context=extra_context,
            )
            deleted_count = len(deleted_rows)
            logging.info(
                "delete_events_by_filter: Deleted %d event(s) for %s (%s)",
                deleted_count,
                deletion_source,
                deletion_reason,
            )
            return deleted_count
        except Exception as e:
            logging.error(
                "delete_events_by_filter: Failed delete for %s (%s): %s",
                deletion_source,
                deletion_reason,
                e,
            )
            return 0

    
    def delete_multiple_events(
        self,
        event_ids,
        deletion_source: str = "db.delete_multiple_events",
        deletion_reason: str = "bulk_delete_by_event_ids",
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Deletes multiple events from the 'events' table based on a list of event IDs.
            event_ids (list): A list of event IDs (int) to be deleted from the database.
            bool: 
                - True if all specified events were successfully deleted.
                - False if one or more deletions failed, or if the input list is empty.
        Logs:
            - A warning if no event IDs are provided.
            - An error for each event ID that fails to be deleted.
            - An info message summarizing the number of successful deletions.
        """
        if not event_ids:
            logging.warning("delete_multiple_events: No event_ids provided for deletion.")
            return False

        success_count = 0
        for event_id in event_ids:
            try:
                per_event_context = dict(extra_context or {})
                per_event_context["batch_size"] = len(event_ids)
                self.delete_event_with_event_id(
                    event_id,
                    deletion_source=deletion_source,
                    deletion_reason=deletion_reason,
                    extra_context=per_event_context,
                )
                success_count += 1
            except Exception as e:
                logging.error("delete_multiple_events: Error deleting event_id %d: %s", event_id, e)

        logging.info("delete_multiple_events: Deleted %d out of %d events.", success_count, len(event_ids))
        
        return success_count == len(event_ids)  # Return True if all deletions were successful


    def multiple_db_inserts(self, table_name, values):
        """
        Inserts or updates multiple records in the specified table using an upsert strategy.

            table_name (str): The name of the table to insert or update. Supported values are "address" and "events".
            values (list of dict): A list of dictionaries, each representing a row to insert or update. Each dictionary's keys should match the table's column names.

        Returns:
            None

        Raises:
            ValueError: If the specified table_name is not supported.
            Exception: If an error occurs during the insert or update operation.

        Logs:
            - An info message if no values are provided.
            - An info message upon successful insertion or update.
            - An error message if an exception occurs during the operation.
        """
        if not values:
            logging.info("multiple_db_inserts(): No values to insert or update.")
            return

        try:
            table = Table(table_name, self.metadata, autoload_with=self.conn)
            with self.conn.begin() as conn:
                for row in values:
                    stmt = insert(table).values(row)
                    if table_name == "address":
                        pk = "address_id"
                    elif table_name == "events":
                        pk = "event_id"
                    else:
                        raise ValueError(f"Unsupported table: {table_name}")
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[pk],
                        set_={col: stmt.excluded[col] for col in row.keys() if col != pk}
                    )
                    conn.execute(stmt)
            logging.info(f"multiple_db_inserts(): Successfully inserted/updated {len(values)} rows in {table_name} table.")
        except Exception as e:
            logging.error(f"multiple_db_inserts(): Error inserting/updating records in {table_name} table - {e}")


    def is_foreign(self):
        """
        Determines which events are likely not in British Columbia (BC) by comparing the event location
        against a list of known municipalities and street names, returning all event columns for context.
        
        The method:
            1. Loads all events from the "events" table into a DataFrame.
            2. Loads street names from the "address" table into a DataFrame.
            3. Reads a list of municipalities from a text file.
            4. Filters the events DataFrame to include only those rows whose 'location' field does not 
            contain any municipality name (from muni_list) or street name (from street_list), case-insensitively.
            5. Deletes the identified events from the "events" table.
        
        Returns:
            pd.DataFrame: A DataFrame containing all columns from the events table for events that are
                        likely not located in BC.
        """
        # 1. Load events from the database.
        events_sql = "SELECT * FROM events"
        events_df = self.read_sql_df(events_sql)
        logging.info("is_foreign(): Loaded %d records from events.", len(events_df))

        # 2. Load address street names.
        address_sql = "SELECT street_name FROM address"
        address_df = self.read_sql_df(address_sql)
        street_list = address_df['street_name'].tolist()
        logging.info("is_foreign(): Loaded %d street names from address.", len(street_list))

        # 3. Read municipalities from file.
        with open(self.config['input']['municipalities'], 'r', encoding='utf-8') as f:
            muni_list = [line.strip() for line in f if line.strip()]
        logging.info("is_foreign(): Loaded %d municipalities from file.", len(muni_list))

        # Read countries from .csv file
        countries_df = pd.read_csv(self.config['input']['countries'])
        countries_list = countries_df['country_names'].tolist()
        blocked_cities = [
            str(city).strip().lower()
            for city in self.config.get("location", {}).get("blacklist_cities", [])
            if str(city).strip()
        ]

        # 4. Filtering logic: if the location or description contains a foreign country, mark it as foreign.
        def is_foreign_location(row):
            location = row['location'] if row['location'] else ''
            description = row['description'] if row['description'] else ''
            source = row['source'] if row['source'] else ''
            combined_text = f"{location} {description} {source}".lower()

            # Explicitly blocked BC cities are treated as out-of-area.
            blocked_city_found = any(city in combined_text for city in blocked_cities)
            if blocked_city_found:
                return True
            
            # Check if any known foreign country appears in the text.
            country_found = any(country.lower() in combined_text for country in countries_list)
            # Check if any BC municipality appears.
            muni_found = any(muni.lower() in combined_text for muni in muni_list if muni)
            
            # If a foreign country is found, consider it foreign.
            if country_found:
                return True
            # If a BC municipality is found, it's likely not foreign.
            if muni_found:
                return False
            # Default to False if neither is found.
            return False

        # Create a boolean mask for events that are likely foreign.
        mask = events_df.apply(is_foreign_location, axis=1)
        foreign_events_df = events_df[mask].copy()
        logging.info("is_foreign(): Found %d events likely in foreign countries or municipalities.", len(foreign_events_df))

        # 5. Delete the identified events from the database.
        if not foreign_events_df.empty:
            event_ids = foreign_events_df['event_id'].tolist()
            self.delete_multiple_events(event_ids)
            logging.info("is_foreign(): Deleted %d events from the database.", len(event_ids))

        return foreign_events_df
    

    def groupby_source(self):
        """
        Executes a SQL query to aggregate and count the number of events per source in the events table.

        Returns:
            pandas.DataFrame: A DataFrame containing two columns:
                - 'source': The source of each event.
                - 'counted': The number of events associated with each source, sorted in descending order.
        """
        query = "SELECT source, COUNT(*) AS counted FROM events GROUP BY source ORDER BY counted DESC"
        groupby_df = pd.read_sql_query(query, self.conn)
        logging.info(f"def groupby_source(): Retrieved groupby results from events table.")
        return groupby_df


    def count_events_urls_start(self, file_name):
        """
        Counts the number of events and distinct URLs in the database at the start time and returns a DataFrame with the results.

        Args:
            file_name (str): The name of the .py file initiating the count.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
            - file_name (str): The name of the .py file.
            - start_time (datetime): The timestamp when the count was initiated.
            - events_count_start (int): The number of events in the database at the start time.
            - urls_count_start (int): The number of URLs in the database at the start time.
        """

        # Add a df for the name of the .py file
        file_name_df = pd.DataFrame([[file_name]], columns=["file_name"])

        # Get start_time
        start_time = datetime.now()
        start_time_df = pd.DataFrame([[start_time]], columns=["start_time"])

        # Count events in db at start
        sql = "SELECT COUNT(*) as events_count_start FROM events"
        events_count_start_df = self.read_sql_df(sql)

        # Count events in db at start
        sql = "SELECT COUNT(DISTINCT link) as urls_count_start FROM urls"
        urls_count_start_df = self.read_sql_df(sql)

        # Concatenate the above dataframes into a new dataframe called start_df
        start_df = pd.concat([file_name_df, start_time_df, events_count_start_df, urls_count_start_df], axis=1)
        start_df.columns = ['file_name', 'start_time_df', 'events_count_start', 'urls_count_start']

        return start_df
    

    def count_events_urls_end(self, start_df, file_name):
        """
        Counts the number of events and URLs in the database at the end of a process, compares them with the counts at the start, 
        and writes the results to a CSV file.

        Parameters:
        start_df (pd.DataFrame): A DataFrame containing the initial counts of events and URLs, as well as the file name and start time.

        Returns:
        None

        The function performs the following steps:
        1. Executes SQL queries to count the number of events and URLs in the database at the end of the process.
        2. Concatenates the initial counts with the new counts into a single DataFrame.
        3. Calculates the number of new events and URLs added to the database.
        4. Adds a timestamp and calculates the elapsed time since the start.
        5. Writes the resulting DataFrame to a CSV file, appending if the file already exists.
        6. Logs the file name where the statistics were written.
        """

        # Count events in db at end
        sql = "SELECT COUNT(*) as events_count_end FROM events"
        events_count_end_df = self.read_sql_df(sql)

        # Count events in db at end
        sql = "SELECT COUNT(DISTINCT link) as urls_count_end FROM urls"
        urls_count_end_df = self.read_sql_df(sql)

        # Create the dataframe
        results_df = pd.concat([start_df, events_count_end_df, urls_count_end_df], axis=1)
        results_df.columns = ['file_name', 'start_time_df', 'events_count_start', 'urls_count_start', 'events_count_end', 'urls_count_end']
        results_df['new_events_in_db'] = results_df['events_count_end'] - results_df['events_count_start']
        results_df['new_urls_in_db'] = results_df['urls_count_end'] - results_df['urls_count_start']
        results_df['time_stamp'] = datetime.now()
        results_df['elapsed_time'] = results_df['time_stamp'] - results_df['start_time_df']

        # Write the df to a csv file (only locally, not on Render)
        if os.getenv('RENDER') != 'true':
            output_file = self.config['output']['events_urls_diff']
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if not os.path.isfile(output_file):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a', header=False, index=False)
            logging.info(f"def count_events_urls_end(): Wrote events and urls statistics to: {output_file}")
        else:
            logging.info(f"def count_events_urls_end(): Skipping CSV write on Render (ephemeral filesystem)")


    def stale_date(self, url):
        """
        Determines whether the most recent event associated with the given URL is considered "stale" based on a configurable age threshold.

        Args:
            url (str): The URL whose most recent event's staleness is to be checked.

        Returns:
            bool: 
                - True if there are no events for the URL, the most recent event's start date is older than the configured threshold, 
                  or if an error occurs during the check (defaulting to stale).
                - False if the most recent event's start date is within the allowed threshold (i.e., not stale).

        Raises:
            None: All exceptions are caught internally and logged; the method will return True in case of errors.
        """

        try:
            # 1. Fetch the most recent start_date for this URL
            query = """
                SELECT start_date
                FROM events_history
                WHERE url = :url
                ORDER BY start_date DESC
                LIMIT 1;
            """
            params = {'url': url}
            result = self.execute_query(query, params)

            # 2. If no rows returned, nothing has been recorded for this URL ⇒ treat as “stale”
            if not result:
                return True

            latest_start_date = result[0][0]
            # 3. If, for some reason, start_date is NULL in the DB, treat as “stale”
            if latest_start_date is None:
                return True

            # 3a. Convert whatever was returned into a Python date
            #     (pd.to_datetime handles string, datetime, or pandas.Timestamp)
            latest_date = pd.to_datetime(latest_start_date).date()

            # 4. Compute cutoff_date = today – N days
            days_threshold = int(self.config['clean_up']['old_events'])
            cutoff_date = datetime.now().date() - timedelta(days=days_threshold)

            # 4a. If the event’s date is older than cutoff_date, it’s stale → return True
            return latest_date < cutoff_date

        except Exception as e:
            logging.error(f"stale_date: Error checking stale date for url {url}: {e}")
            # In case of any error, default to True (safer to re‐process)
            return True

    @staticmethod
    def _facebook_event_id_from_url(url: str) -> Optional[str]:
        """Return Facebook event id for /events/<id>/ URLs."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url or "")
            host = (parsed.netloc or "").lower()
            if "facebook.com" not in host:
                return None
            match = re.search(r"/events/(\d+)", parsed.path or "")
            if not match:
                return None
            return match.group(1)
        except Exception:
            return None

    @staticmethod
    def _eventbrite_ticket_id_from_url(url: str) -> Optional[str]:
        """Return Eventbrite ticket id from event-detail URLs."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url or "")
            host = (parsed.netloc or "").lower()
            if "eventbrite." not in host:
                return None
            match = re.search(r"-tickets-(\d+)", parsed.path or "")
            if not match:
                return None
            return match.group(1)
        except Exception:
            return None

    @staticmethod
    def _instagram_post_id_from_url(url: str) -> Optional[tuple[str, str]]:
        """
        Return (kind, id) for Instagram post-detail URLs.

        Supported kinds:
            - p (standard post)
            - reel
            - tv
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url or "")
            host = (parsed.netloc or "").lower()
            if "instagram.com" not in host:
                return None

            path = (parsed.path or "").strip("/")
            if not path:
                return None
            parts = [p for p in path.split("/") if p]
            if len(parts) < 2:
                return None
            kind = parts[0].lower()
            if kind not in {"p", "reel", "tv"}:
                return None
            post_id = (parts[1] or "").strip()
            if not post_id:
                return None
            return kind, post_id
        except Exception:
            return None

    def _classify_static_event_detail_url(self, normalized_url: str) -> tuple[Optional[str], List[str]]:
        """
        Classify URL as a static event-detail URL and return lookup candidates for history checks.

        Returns:
            tuple[str|None, list[str]]:
                - kind: one of {"facebook_event_detail", "eventbrite_event_detail", "instagram_post_detail"} or None
                - lookup candidates used to search historical events for the same detail page
        """
        try:
            from urllib.parse import urlparse, urlunparse
        except Exception:
            return None, []

        parsed = urlparse(normalized_url or "")
        host = (parsed.netloc or "").lower()
        scheme = (parsed.scheme or "https").lower()
        path = parsed.path or ""
        candidates: List[str] = []

        classification = classify_page(url=normalized_url)
        subtype = classification.subtype

        fb_event_id = self._facebook_event_id_from_url(normalized_url)
        if subtype == "facebook_event_detail" and fb_event_id:
            canonical = f"{scheme}://www.facebook.com/events/{fb_event_id}/"
            candidates.extend([normalized_url, canonical, canonical.rstrip("/")])
            return "facebook_event_detail", list(dict.fromkeys(candidates))

        eb_ticket_id = self._eventbrite_ticket_id_from_url(normalized_url)
        if subtype == "eventbrite_event_detail" and eb_ticket_id and "eventbrite." in host:
            no_query = urlunparse((scheme, parsed.netloc, path, "", "", ""))
            candidates.extend([normalized_url, no_query, no_query.rstrip("/")])
            return "eventbrite_event_detail", list(dict.fromkeys(candidates))

        ig_post = self._instagram_post_id_from_url(normalized_url)
        if subtype == "instagram_post_detail" and ig_post:
            ig_kind, ig_id = ig_post
            no_query = urlunparse((scheme, parsed.netloc, path, "", "", ""))
            canonical = f"{scheme}://www.instagram.com/{ig_kind}/{ig_id}/"
            candidates.extend([normalized_url, no_query, no_query.rstrip("/"), canonical, canonical.rstrip("/")])
            return "instagram_post_detail", list(dict.fromkeys(candidates))

        return None, []

    def _latest_event_start_date_for_urls(self, url_candidates: List[str]) -> Optional[date]:
        """Return latest historical start_date across provided URL candidates."""
        latest_date: Optional[date] = None
        for candidate in url_candidates:
            try:
                rows = self.execute_query(
                    """
                    SELECT start_date
                    FROM events_history
                    WHERE url = :url
                    ORDER BY start_date DESC
                    LIMIT 1
                    """,
                    {"url": candidate},
                )
                if not rows:
                    continue
                row_start = rows[0][0]
                if row_start is None:
                    continue
                parsed_date = pd.to_datetime(row_start).date()
                if latest_date is None or parsed_date > latest_date:
                    latest_date = parsed_date
            except Exception:
                continue
        return latest_date

    def maybe_reuse_static_event_detail_from_history(
        self,
        *,
        url: str,
        rescrape_window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Reuse one upcoming static event-detail row from events_history when safe.

        Rules:
        - Only applies to static event-detail URLs recognized by the classifier.
        - Only reuses one unambiguous upcoming event identity for the URL.
        - Events within the rescrape window are not reused.
        """
        normalized_url = self.normalize_url(str(url or "").strip())
        kind, candidates = self._classify_static_event_detail_url(normalized_url)
        if not kind or not candidates:
            return {"reused": False, "reason": "not_static_event_detail"}

        cutoff_date = datetime.now().date() + timedelta(days=max(0, int(rescrape_window_days or 0)))
        history_rows = self._load_events_history_rows_for_urls(candidates)
        if not history_rows:
            return {"reused": False, "reason": "no_history_rows"}

        eligible_rows = [
            row for row in history_rows
            if isinstance(row.get("start_date"), date) and row["start_date"] > cutoff_date
        ]
        if not eligible_rows:
            return {"reused": False, "reason": "no_upcoming_history_outside_rescrape_window"}

        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in eligible_rows:
            grouped.setdefault(self._history_event_identity_key(row), []).append(row)

        if len(grouped) != 1:
            return {"reused": False, "reason": "ambiguous_history_match", "match_count": len(grouped)}

        selected_rows = next(iter(grouped.values()))
        selected = max(
            selected_rows,
            key=lambda row: (
                self._coerce_history_sort_value(row.get("time_stamp")),
                int(row.get("event_id") or 0),
            ),
        )

        event_payload = {
            "event_name": selected.get("event_name"),
            "dance_style": selected.get("dance_style"),
            "description": selected.get("description"),
            "day_of_week": selected.get("day_of_week"),
            "start_date": selected.get("start_date"),
            "end_date": selected.get("end_date"),
            "start_time": selected.get("start_time"),
            "end_time": selected.get("end_time"),
            "source": selected.get("source"),
            "location": selected.get("location"),
            "price": selected.get("price"),
            "url": selected.get("url") or normalized_url,
            "event_type": selected.get("event_type"),
            # Never trust historical address_id directly; re-resolve it from the event payload.
            "address_id": None,
            "time_stamp": datetime.now(),
        }
        event_payload = self.normalize_nulls(event_payload)
        try:
            event_payload = self.process_event_address(event_payload)
        except Exception as e:
            logging.warning(
                "maybe_reuse_static_event_detail_from_history: address refresh failed for %s: %s",
                normalized_url,
                e,
            )
        sanitized = self._sanitize_events_dataframe_for_insert(pd.DataFrame([event_payload]))
        sanitized = self._drop_us_postal_code_events(sanitized, context="history_reuse")
        if sanitized.empty:
            return {"reused": False, "reason": "history_payload_disqualified"}

        sanitized.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        return {
            "reused": True,
            "reason": "history_reuse_static_event_detail",
            "history_kind": kind,
            "event_count": int(len(sanitized)),
            "events_history_event_id": selected.get("event_id"),
            "events_history_original_event_id": selected.get("original_event_id"),
            "history_time_stamp": self._coerce_history_sort_value(selected.get("time_stamp")).isoformat(),
            "start_date": selected["start_date"].isoformat() if isinstance(selected.get("start_date"), date) else None,
            "days_until_event": int((selected["start_date"] - datetime.now().date()).days),
            "url": normalized_url,
        }

    def _load_events_history_rows_for_urls(self, url_candidates: List[str]) -> List[Dict[str, Any]]:
        """Load events_history rows for the provided URLs."""
        loaded: List[Dict[str, Any]] = []
        seen_row_keys: set[tuple] = set()
        query = """
            SELECT
                event_id,
                original_event_id,
                event_name,
                dance_style,
                description,
                day_of_week,
                start_date,
                end_date,
                start_time,
                end_time,
                source,
                location,
                price,
                url,
                event_type,
                address_id,
                time_stamp
            FROM events_history
            WHERE url = :url
        """
        for candidate in url_candidates:
            rows = self.execute_query(query, {"url": candidate}) or []
            for row in rows:
                parsed = self._coerce_events_history_row(row)
                if not parsed:
                    continue
                row_key = (
                    parsed.get("event_id"),
                    parsed.get("original_event_id"),
                    parsed.get("url"),
                    parsed.get("start_date"),
                    parsed.get("start_time"),
                )
                if row_key in seen_row_keys:
                    continue
                seen_row_keys.add(row_key)
                loaded.append(parsed)
        return loaded

    def _coerce_events_history_row(self, row: Any) -> Optional[Dict[str, Any]]:
        """Normalize an events_history row into a dict."""
        columns = [
            "event_id",
            "original_event_id",
            "event_name",
            "dance_style",
            "description",
            "day_of_week",
            "start_date",
            "end_date",
            "start_time",
            "end_time",
            "source",
            "location",
            "price",
            "url",
            "event_type",
            "address_id",
            "time_stamp",
        ]
        if hasattr(row, "_mapping"):
            raw = dict(row._mapping)
        elif isinstance(row, dict):
            raw = dict(row)
        elif isinstance(row, (tuple, list)) and len(row) == len(columns):
            raw = dict(zip(columns, row))
        else:
            return None

        raw["start_date"] = self._coerce_date_value(raw.get("start_date"))
        raw["end_date"] = self._coerce_date_value(raw.get("end_date"))
        raw["time_stamp"] = self._coerce_history_sort_value(raw.get("time_stamp"))
        return raw

    @staticmethod
    def _history_event_identity_key(row: Dict[str, Any]) -> tuple:
        """Build a stable identity key for one historical event detail row."""
        return (
            str(row.get("url") or "").strip(),
            str(row.get("event_name") or "").strip().lower(),
            row.get("start_date"),
            str(row.get("start_time") or "").strip(),
            row.get("end_date"),
            str(row.get("end_time") or "").strip(),
        )

    @staticmethod
    def _coerce_date_value(value: Any) -> Optional[date]:
        """Coerce a value to a date when possible."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime().date()
            except Exception:
                pass
        if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
            try:
                return datetime(value.year, value.month, value.day).date()
            except Exception:
                pass
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.date()
        except Exception:
            return None

    @staticmethod
    def _coerce_history_sort_value(value: Any) -> datetime:
        """Return a sortable datetime for history rows."""
        if isinstance(value, datetime):
            return value
        if value is None or value == "":
            return datetime.min
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return datetime.min
            return parsed.to_pydatetime()
        except Exception:
            return datetime.min

    def _record_should_process_decision(self, decision_key: str) -> None:
        """Track should_process_url decision counts for runtime reporting."""
        counters = getattr(self, "_should_process_decision_counters", None)
        if counters is None:
            self._should_process_decision_counters = Counter()
            counters = self._should_process_decision_counters
        counters[str(decision_key)] += 1

    def _set_last_should_process_reason(self, normalized_url: str, decision_key: str) -> None:
        """Store most recent should_process_url reason for a normalized URL."""
        if not normalized_url:
            return
        if not hasattr(self, "_last_should_process_reason_by_url"):
            self._last_should_process_reason_by_url = {}
        self._last_should_process_reason_by_url[str(normalized_url)] = str(decision_key)

    def get_should_process_decision_counts(self) -> Dict[str, int]:
        """Return a snapshot of should_process_url decision counters."""
        counters = getattr(self, "_should_process_decision_counters", None)
        if counters is None:
            return {}
        return {str(k): int(v) for k, v in counters.items()}

    def get_should_process_decision_reason(self, url: str) -> Optional[str]:
        """Return the most recent decision reason for URL seen in should_process_url."""
        normalized_url = self._normalize_for_compare(self.normalize_url(url))
        decision_map = getattr(self, "_last_should_process_reason_by_url", {})
        return decision_map.get(normalized_url)

    def _should_skip_stale_static_event_url(self, normalized_url: str) -> tuple[bool, Optional[str]]:
        """
        Return skip decision for already-seen static event-detail URLs.

        For Facebook/Eventbrite event-detail URLs, if history shows the latest start_date is
        before today, skip reprocessing.
        """
        kind, candidates = self._classify_static_event_detail_url(normalized_url)
        if not kind or not candidates:
            return False, None
        latest_date = self._latest_event_start_date_for_urls(candidates)
        if latest_date is None:
            return False, None
        if latest_date < datetime.now().date():
            reason_map = {
                "facebook_event_detail": "skip_stale_facebook_event_detail",
                "eventbrite_event_detail": "skip_stale_eventbrite_event_detail",
                "instagram_post_detail": "skip_stale_instagram_post_detail",
            }
            reason = reason_map.get(kind, "skip_stale_static_event_detail")
            return True, reason
        return False, None


    def _old_only_rejection_reason_for_url(self, normalized_url: str) -> Optional[str]:
        """
        Return a structured reason when a static event-detail URL produced only old events.
        """
        kind, _ = self._classify_static_event_detail_url(normalized_url)
        reason_map = {
            "facebook_event_detail": "rejected_old_facebook_event_detail",
            "eventbrite_event_detail": "rejected_old_eventbrite_event_detail",
            "instagram_post_detail": "rejected_old_instagram_post_detail",
        }
        return reason_map.get(kind)


    def _should_skip_old_only_static_rejection(
        self,
        df_url: pd.DataFrame,
        normalized_url: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Skip static event-detail URLs that were previously scraped and rejected as old-only.
        """
        if df_url is None or df_url.empty:
            return False, None
        rejection_reason = self._old_only_rejection_reason_for_url(normalized_url)
        if not rejection_reason:
            return False, None
        latest_reason = str(df_url.iloc[-1].get("decision_reason") or "").strip()
        if latest_reason != rejection_reason:
            return False, None
        return True, rejection_reason.replace("rejected_old_", "skip_rejected_old_", 1)


    def normalize_url(self, url):
        """
        Normalize URLs by removing dynamic cache parameters that don't affect the underlying content.
        
        This is particularly important for Instagram and Facebook CDN URLs that include
        dynamic parameters like _nc_gid, _nc_ohc, oh, oe, etc. that change between requests
        but point to the same underlying image.
        
        Args:
            url (str): The original URL with potentially dynamic parameters
            
        Returns:
            str: Normalized URL with dynamic parameters removed
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(url)
        
        # Check if this is an Instagram or Facebook CDN URL
        # Be specific about Instagram/FB domains to avoid affecting other CDN URLs
        instagram_domains = {
            'instagram.com',
            'www.instagram.com', 
            'scontent.cdninstagram.com',
            'instagram.fcxh2-1.fna.fbcdn.net',
            'scontent.cdninstagram.com'
        }
        
        fb_cdn_domains = {
            domain for domain in [parsed.netloc] 
            if 'fbcdn.net' in domain and ('instagram' in domain or 'scontent' in domain)
        }
        
        is_instagram_cdn = (parsed.netloc in instagram_domains or 
                           any(domain in parsed.netloc for domain in instagram_domains) or
                           bool(fb_cdn_domains))
        
        if not is_instagram_cdn:
            return url
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # List of dynamic parameters to remove for Instagram/FB CDN URLs
        dynamic_params = {
            '_nc_gid',     # Cache group ID - changes between sessions
            '_nc_ohc',     # Cache hash - changes between requests  
            '_nc_oc',      # Cache parameter - changes between requests
            'oh',          # Hash parameter - changes between requests
            'oe',          # Expiration parameter - changes over time
            '_nc_zt',      # Zoom/time parameter
            '_nc_ad',      # Ad parameter
            '_nc_cid',     # Cache ID
            'ccb',         # Cache control parameter (sometimes)
        }
        
        # Remove dynamic parameters
        filtered_params = {k: v for k, v in query_params.items() 
                          if k not in dynamic_params}
        
        # Reconstruct URL with filtered parameters
        new_query = urlencode(filtered_params, doseq=True)
        normalized_url = urlunparse((
            parsed.scheme,
            parsed.netloc, 
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return normalized_url

    def should_process_url(self, url):
        """
        Determines whether a given URL should be processed based on its history in the database.

        The decision is made according to the following rules:
        0. If the URL is in the whitelist (data/urls/aaa_urls.csv), it should ALWAYS be processed.
        1. If the URL has never been seen before (i.e., no records in `self.urls_df`), it should be processed.
        2. If the most recent record for the URL has 'relevant' set to True, it should be processed.
        3. If the most recent record for the URL has 'relevant' set to False, the method checks the grouped statistics in `self.urls_gb`:
            - If the `hit_ratio` for the URL is greater than 0.1, or
            - If the number of crawl attempts (`crawl_try`) is less than or equal to 3,
            then the URL should be processed.
        4. If none of the above conditions are met, the URL should not be processed.

        Args:
             url (str): The URL to evaluate for processing.

        Returns:
             bool: True if the URL should be processed according to the criteria above, False otherwise.
        """
        # Normalize URL to handle Instagram/FB CDN dynamic parameters
        normalized_url = self.normalize_url(url)
        generic_norm = self._normalize_for_compare(normalized_url)

        # If urls_df is empty (e.g., on production), always process
        if self.urls_df.empty:
            logging.info(f"should_process_url: URLs table not loaded (production mode), processing URL.")
            decision = "process_urls_df_not_loaded"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return True

        # Log normalization if URL changed
        if normalized_url != url:
            logging.info(f"should_process_url: Normalized Instagram URL for comparison")

        # 0. Whitelist precedence: always process whitelisted URLs
        try:
            if self.is_whitelisted_url(url) or self.is_whitelisted_url(normalized_url):
                logging.info(f"should_process_url: URL {normalized_url[:100]}... is whitelisted, processing it.")
                decision = "process_whitelisted"
                self._record_should_process_decision(decision)
                self._set_last_should_process_reason(generic_norm, decision)
                return True
        except Exception as e:
            logging.warning(f"should_process_url: whitelist check error: {e}")
    
        # 1. Filter all rows for this normalized URL
        df_url = self.urls_df[self.urls_df['link'] == normalized_url]
        # If we've never recorded this normalized URL, process it
        if df_url.empty:
            logging.info(f"should_process_url: URL {normalized_url[:100]}... has never been seen before, processing it.")
            decision = "process_never_seen"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return True

        # 1b. For static event-detail URLs, skip when the last scrape already rejected the page as old-only.
        should_skip_old_rejection, old_rejection_reason = self._should_skip_old_only_static_rejection(
            df_url,
            normalized_url,
        )
        if should_skip_old_rejection:
            logging.info(
                "should_process_url: URL %s skipped due to prior old-only rejection (%s).",
                normalized_url[:100] + "...",
                old_rejection_reason,
            )
            decision = old_rejection_reason or "skip_rejected_old_static_event_detail"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return False

        # 1c. For static event-detail URLs (Facebook/Eventbrite), skip when historical event is stale.
        should_skip_stale, stale_reason = self._should_skip_stale_static_event_url(normalized_url)
        if should_skip_stale:
            logging.info(
                "should_process_url: URL %s skipped due to stale static event-detail URL (%s).",
                normalized_url[:100] + "...",
                stale_reason,
            )
            decision = stale_reason or "skip_stale_static_event_detail"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return False

        # 2. Look at the most recent "relevant" value
        last_relevant = df_url.iloc[-1]['relevant']
        if last_relevant and self.stale_date(normalized_url):
            logging.info(f"should_process_url: URL {normalized_url[:100]}... was last seen as relevant, processing it.")
            decision = "process_last_relevant_and_stale_date"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return True

        # 3. Last was False → check hit_ratio in self.urls_gb
        if not last_relevant and self._is_low_value_scrape_path(normalized_url):
            logging.info(
                "should_process_url: URL %s skipped due to low-value path after prior irrelevant result.",
                normalized_url[:100] + "...",
            )
            decision = "skip_low_value_path_after_irrelevant"
            self._record_should_process_decision(decision)
            self._set_last_should_process_reason(generic_norm, decision)
            return False

        hit_row = self.urls_gb[self.urls_gb['link'] == normalized_url]

        if not hit_row.empty:
            # Extract scalars from the grouped DataFrame
            hit_ratio = hit_row.iloc[0]['hit_ratio']
            crawl_trys = hit_row.iloc[0]['crawl_try']

            if (
                hit_ratio >= self._SHOULD_PROCESS_MIN_HIT_RATIO
                or crawl_trys <= self._SHOULD_PROCESS_MAX_RETRIES_FOR_IRRELEVANT
            ):
                logging.info(
                    "should_process_url: URL %s was last seen as not relevant "
                    "but hit_ratio (%.2f) >= %.2f or crawl_try (%d) ≤ %d, processing it.",
                    normalized_url[:100] + "...",
                    hit_ratio,
                    self._SHOULD_PROCESS_MIN_HIT_RATIO,
                    crawl_trys,
                    self._SHOULD_PROCESS_MAX_RETRIES_FOR_IRRELEVANT,
                )
                decision = "process_strong_hit_ratio_or_early_retry"
                self._record_should_process_decision(decision)
                self._set_last_should_process_reason(generic_norm, decision)
                return True

        # 4. Otherwise, do not process this URL
        logging.info(f"should_process_url: URL {normalized_url[:100]}... does not meet criteria for processing, skipping it.")
        decision = "skip_default_rules"
        self._record_should_process_decision(decision)
        self._set_last_should_process_reason(generic_norm, decision)
        return False

    @classmethod
    def _is_low_value_scrape_path(cls, url: str) -> bool:
        """Return True for path segments that are consistently low-value crawl targets."""
        try:
            path = (urlparse(url).path or "").strip("/").lower()
        except Exception:
            return False
        if not path:
            return False
        segments = [segment for segment in path.split("/") if segment]
        return any(segment in cls._LOW_VALUE_PATH_SEGMENTS for segment in segments)


    def update_day_of_week(self, event_id: int, corrected_day: str) -> bool:
        """
        Updates only the day_of_week field for a single event.

        This is intentionally safer than mutating dates: start_date/end_date are
        treated as the source of truth once parsed from source content.

        Args:
            event_id (int): Unique event identifier.
            corrected_day (str): Canonical weekday name (e.g., "Wednesday").

        Returns:
            bool: True if update executed.
        """
        update_query = """
            UPDATE events
               SET day_of_week = :corrected_day
             WHERE event_id    = :event_id
        """
        params = {
            "corrected_day": corrected_day,
            "event_id": event_id,
        }
        self.execute_query(update_query, params)
        return True


    def check_dow_date_consistent(self) -> None:
        """
        Ensures day_of_week matches start_date without mutating event dates.

        This method performs the following steps:
            1. Retrieves all events' event_id, start_date, and day_of_week from the database.
            2. Derives canonical weekday from start_date.
            3. If stored day_of_week is missing, invalid, or inconsistent, updates day_of_week.
            4. Never shifts start_date/end_date, preventing recurrence-text induced date drift.

        Returns:
            None
        """
        select_query = """
            SELECT event_id, start_date, day_of_week
              FROM events
        """
        rows = self.execute_query(select_query)
        # rows is a list of tuples: (event_id, start_date, day_of_week)

        valid_days = {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }

        for row in rows:
            event_id = row[0]
            orig_date = row[1]  # DATE
            dow_text = row[2]  # TEXT

            if orig_date is None:
                continue

            canonical_day = orig_date.strftime("%A")
            key = str(dow_text or "").strip().lower()
            current_day_valid = key in valid_days
            current_day_canonical = key.capitalize() if current_day_valid else ""

            if current_day_canonical == canonical_day:
                continue

            if not current_day_valid and key:
                logging.warning(
                    "check_dow_date_consistent: event_id %s has unrecognized day_of_week '%s'; replacing with '%s'.",
                    event_id,
                    dow_text,
                    canonical_day,
                )

            success = self.update_day_of_week(event_id, canonical_day)
            if success:
                logging.info(
                    "check_dow_date_consistent: event_id %d day_of_week changed from '%s' to '%s' (start_date=%s).",
                    event_id,
                    str(dow_text or ""),
                    canonical_day,
                    orig_date.isoformat(),
                )
            else:
                logging.error(
                    "check_dow_date_consistent: failed to update day_of_week for event_id %d",
                    event_id,
                )


    def check_image_events_exist(self, image_url: str) -> bool:
        """
        Always returns False to force re-scraping of images/PDFs.

        DISABLED: Previously checked events table and copied from events_history, but this
        caused data corruption issues. All images/PDFs are now re-scraped on every run to
        ensure fresh, accurate data with correct address normalization.

        Args:
            image_url (str): The URL of the image to check for associated events.

        Returns:
            bool: Always returns False to force re-scraping.
        """
        logging.info(f"check_image_events_exist(): Forcing re-scrape for URL: {image_url}")
        return False

        # DISABLED CODE - DO NOT USE (address_ids in events_history are corrupted):
        # # 2) Check history table
        # sql_hist = """
        # SELECT COUNT(*)
        # FROM events_history
        # WHERE url = :url
        # """
        # hist = self.execute_query(sql_hist, params)
        # if not (hist and hist[0][0] > 0):
        #     logging.info(f"check_image_events_exist(): No history events for URL: {image_url}")
        #     return False
        #
        # # 3) Copy only the most‐recent history row per unique event into events
        # sql_copy = """
        # INSERT INTO events (
        #     event_name, dance_style, description, day_of_week,
        #     start_date, end_date, start_time, end_time,
        #     source, location, price, url,
        #     event_type, address_id, time_stamp
        # )
        # SELECT
        #     sub.event_name, sub.dance_style, sub.description, sub.day_of_week,
        #     sub.start_date, sub.end_date, sub.start_time, sub.end_time,
        #     sub.source, sub.location, sub.price, sub.url,
        #     sub.event_type, sub.address_id, sub.time_stamp
        # FROM (
        #     SELECT DISTINCT ON (
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id
        #     )
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id, time_stamp
        #     FROM events_history
        #     WHERE url = :url
        #     AND start_date >= (CURRENT_DATE - (:days * INTERVAL '1 day'))
        #     ORDER BY
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id,
        #         time_stamp DESC
        # ) AS sub
        # """
        # params_copy = {
        #     'url':  image_url,
        #     'days': self.config['clean_up']['old_events']  # e.g. 3 → includes start_date ≥ today−3d
        # }
        # self.execute_query(sql_copy, params_copy)
        #
        # logging.info(f"check_image_events_exist(): Copied most‐recent history events into events for URL: {image_url}")
        # return True
    

    def sql_input(self, file_path: str):
        """
        Reads a JSON file containing a flat dictionary of SQL fixes and executes them.

        Args:
            file_path (str): Path to the .json file containing SQL statements.
        """
        logging.info("sql_input(): Starting SQL input execution from %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sql_dict = json.load(f)
            logging.info("sql_input(): Successfully loaded %d SQL entries", len(sql_dict))
        except Exception as e:
            logging.error("sql_input(): Failed to load or parse JSON file: %s", e)
            return

        for name, query in sql_dict.items():
            logging.info("sql_input(): Executing [%s]: %s", name, query)
            result = self.execute_query(query)
            if result is None:
                logging.error("sql_input(): Failed to execute [%s]", name)
            else:
                logging.info("sql_input(): Successfully executed [%s]", name)

        logging.info("sql_input(): All queries processed.")

    
    def normalize_nulls(self, record: dict) -> dict:
        """
        Replaces string values like 'null', 'none', 'nan', or empty strings with Python None (i.e., SQL NULL).
        Also handles pandas NaN values and various null representations.
        Applies to all keys in the given dictionary.
        """
        import pandas as pd
        import numpy as np
        
        cleaned = {}
        null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}
        
        for key, value in record.items():
            # Handle pandas/numpy NaN
            if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = None
            # Handle string nulls (case insensitive, strip whitespace)
            elif isinstance(value, str) and value.strip().lower() in null_strings:
                cleaned[key] = None
            # Handle numeric 0 that shouldn't be null (keep as is)
            elif value == 0:
                cleaned[key] = value
            # Handle other falsy values that should be None
            elif not value and value != 0:
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    

    def clean_null_strings_in_address(self):
        """
        Replaces string 'null', 'none', 'nan', and '' with actual SQL NULLs in address table.
        """
        fields = [
            "full_address", "building_name", "street_number", "street_name", "direction",
            "city", "met_area", "province_or_state", "postal_code", "country_id"
        ]
        for field in fields:
            query = f"""
                UPDATE address
                SET {field} = NULL
                WHERE TRIM(LOWER({field})) IN ('null', 'none', 'nan', '', '[null]', '(null)', 'n/a', 'na');
            """
            self.execute_query(query)
        logging.info("Cleaned up string 'null's in address table.")
    
    
    def standardize_postal_codes(self):
        """
        Standardizes Canadian postal codes to format V8N 1S3 (with space).
        """
        query = """
            UPDATE address
            SET postal_code = UPPER(
                CASE 
                    WHEN LENGTH(REPLACE(postal_code, ' ', '')) = 6 
                    THEN SUBSTRING(REPLACE(postal_code, ' ', ''), 1, 3) || ' ' || SUBSTRING(REPLACE(postal_code, ' ', ''), 4, 3)
                    ELSE postal_code
                END
            )
            WHERE postal_code IS NOT NULL 
            AND postal_code ~ '^[A-Za-z][0-9][A-Za-z][ ]?[0-9][A-Za-z][0-9]$'
        """
        result = self.execute_query(query)
        logging.info("Standardized postal code formats to V8N 1S3 pattern.")
        return result


    def driver(self):
        """
        Main driver function to perform database operations based on configuration.

        If the 'drop_tables' flag in the testing configuration is set to True,
        the function will recreate the database tables. Otherwise, it will perform
        a series of data cleaning and maintenance operations, including deduplication,
        deletion of old or invalid events, fuzzy duplicate detection, foreign key checks,
        address deduplication, and fixing address IDs in events.

        At the end of the process, the database connection is properly closed.

        Returns:
            None
        """
        if self.config['testing']['drop_tables'] == True:
            self.create_tables()
        else:
            self.check_dow_date_consistent()
            self.delete_old_events()
            self.delete_events_with_nulls()
            self.delete_likely_dud_events()
            self.fuzzy_duplicates()
            self.is_foreign()
            self.sql_input(self.config['input']['sql_input'])
            self.sync_event_locations_with_address_table()
            self.clean_null_strings_in_address()
            self.dedup()
            # Clean up any remaining orphaned references before sequence reset
            self.clean_orphaned_references()

            # COMMENTED OUT: reset_address_id_sequence() causes race conditions with concurrent processes
            # This function renumbers all address_ids sequentially (1, 2, 3...) and updates all foreign keys.
            # Problem: If multiple processes (pipeline, web service, or manual operations) access the database
            # simultaneously, the renumbering creates corruption as IDs change mid-operation.
            #
            # To re-enable safely:
            # 1. Ensure NO other processes are accessing the database (stop web service, CRON jobs, manual queries)
            # 2. Uncomment the line below
            # 3. Run the pipeline once for maintenance
            # 4. Re-comment this line before resuming normal operations
            #
            # self.reset_address_id_sequence()

            self.update_full_address_with_building_names()
            
            # Fix events with address_id = 0 using existing deduplication logic
            logging.info("driver(): Creating DeduplicationHandler for fix_problem_events")
            from dedup_llm import DeduplicationHandler
            dedup_handler = DeduplicationHandler(config_path='config/config.yaml')
            dedup_handler.fix_problem_events(dry_run=False)
            logging.info("driver(): Completed fix_problem_events via DeduplicationHandler")

        # Close the database connection
        self.conn.dispose()  # Using dispose() for SQLAlchemy Engine
        logging.info("driver(): Database operations completed successfully.")

    def reset_address_id_sequence(self):
        """
        Reset the address_id sequence to start from 1, updating all references in the events table.

        This method:
        0. Cleans up orphaned raw_locations records pointing to non-existent addresses
        1. Creates a mapping of old address_ids to new sequential IDs (1, 2, 3, ...)
        2. Updates all events table records with the new address_id values
        3. Updates the address table with new sequential IDs
        4. Resets the PostgreSQL sequence to continue from the max ID + 1

        Returns:
            int: Number of addresses that were renumbered
        """
        try:
            logging.info("reset_address_id_sequence(): Starting address ID sequence reset...")

            # Step 0: Clean up orphaned raw_locations records that reference non-existent addresses
            cleanup_orphaned_sql = """
            DELETE FROM raw_locations
            WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            orphaned_count = self.execute_query(cleanup_orphaned_sql)
            logging.info(f"reset_address_id_sequence(): Cleaned up {orphaned_count} orphaned raw_locations records")

            # Step 1: Get current addresses ordered by address_id and create mapping
            get_addresses_sql = """
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, met_area, province_or_state, 
                   postal_code, country_id, time_stamp
            FROM address 
            ORDER BY address_id;
            """
            
            addresses_df = self.read_sql_df(get_addresses_sql)
            
            if addresses_df.empty:
                logging.info("reset_address_id_sequence(): No addresses found to renumber.")
                return 0
            
            # Create mapping from old address_id to new sequential ID
            address_mapping = {}
            for idx, row in addresses_df.iterrows():
                old_id = row['address_id']
                new_id = idx + 1  # Start from 1
                address_mapping[old_id] = new_id
            
            logging.info(f"reset_address_id_sequence(): Created mapping for {len(address_mapping)} addresses")
            
            # Step 2: Create temporary table with new sequential IDs
            create_temp_table_sql = """
            CREATE TEMPORARY TABLE address_temp AS 
            SELECT * FROM address WHERE 1=0;
            """
            self.execute_query(create_temp_table_sql)
            
            # Insert addresses with new sequential IDs
            for idx, row in addresses_df.iterrows():
                new_id = idx + 1
                insert_temp_sql = """
                INSERT INTO address_temp (address_id, full_address, building_name, street_number, 
                                        street_name, street_type, direction, city, met_area, 
                                        province_or_state, postal_code, country_id, time_stamp)
                VALUES (:new_id, :full_address, :building_name, :street_number, :street_name, 
                        :street_type, :direction, :city, :met_area, :province_or_state, 
                        :postal_code, :country_id, :time_stamp);
                """
                params = {
                    'new_id': new_id,
                    'full_address': row['full_address'],
                    'building_name': row['building_name'],
                    'street_number': row['street_number'],
                    'street_name': row['street_name'],
                    'street_type': row['street_type'],
                    'direction': row['direction'],
                    'city': row['city'],
                    'met_area': row['met_area'],
                    'province_or_state': row['province_or_state'],
                    'postal_code': row['postal_code'],
                    'country_id': row['country_id'],
                    'time_stamp': row['time_stamp']
                }
                self.execute_query(insert_temp_sql, params)
            
            # Step 3: Update all tables that reference address_id with new address_ids
            events_updated = 0
            events_history_updated = 0
            raw_locations_updated = 0
            
            for old_id, new_id in address_mapping.items():
                # Update events table
                update_events_sql = """
                UPDATE events 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_events_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_updated += 1
                
                # Update events_history table (only for address_ids that exist)
                update_events_history_sql = """
                UPDATE events_history 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_events_history_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_history_updated += 1
                
                # Update raw_locations table (this has the foreign key constraint)
                update_raw_locations_sql = """
                UPDATE raw_locations 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_raw_locations_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    raw_locations_updated += 1
            
            logging.info(f"reset_address_id_sequence(): Updated address_id in events table for {events_updated} different address IDs")
            logging.info(f"reset_address_id_sequence(): Updated address_id in events_history table for {events_history_updated} different address IDs")
            logging.info(f"reset_address_id_sequence(): Updated address_id in raw_locations table for {raw_locations_updated} different address IDs")
            
            # Step 4: Replace original address table with renumbered version
            # Delete all from original table
            self.execute_query("DELETE FROM address;")
            
            # Insert from temp table
            copy_back_sql = """
            INSERT INTO address (address_id, full_address, building_name, street_number, 
                               street_name, street_type, direction, city, met_area, 
                               province_or_state, postal_code, country_id, time_stamp)
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, met_area, province_or_state, 
                   postal_code, country_id, time_stamp
            FROM address_temp;
            """
            self.execute_query(copy_back_sql)
            
            # Step 5: Reset the PostgreSQL sequence
            max_id = len(addresses_df)
            
            # First, get the actual sequence name for address_id
            sequence_query = "SELECT pg_get_serial_sequence('address', 'address_id');"
            sequence_result = self.execute_query(sequence_query)
            
            if sequence_result and sequence_result[0][0]:
                sequence_name = sequence_result[0][0].split('.')[-1]  # Remove schema prefix if present
                reset_sequence_sql = f"SELECT setval('{sequence_name}', {max_id}, true);"
                self.execute_query(reset_sequence_sql)
                logging.info(f"reset_address_id_sequence(): Reset sequence '{sequence_name}' to {max_id}")
            else:
                # Create proper sequence if it doesn't exist
                self.ensure_address_sequence(max_id + 1)
                logging.info(f"reset_address_id_sequence(): Created new sequence 'address_address_id_seq' starting from {max_id + 1}")
            
            # Clean up temp table
            self.execute_query("DROP TABLE address_temp;")
            
            logging.info(f"reset_address_id_sequence(): Successfully reset address_id sequence. "
                        f"Renumbered {len(address_mapping)} addresses, sequence reset to start from {max_id + 1}")
            
            return len(address_mapping)
            
        except Exception as e:
            logging.error(f"reset_address_id_sequence(): Error during address ID sequence reset: {e}")
            # Clean up temp table if it exists
            try:
                self.execute_query("DROP TABLE IF EXISTS address_temp;")
            except:
                pass
            raise

    def update_full_address_with_building_names(self):
        """
        Update existing full_address records using the standardized format.
        
        Rebuilds full_address for all records using the build_full_address method
        to ensure consistency across the database.
        
        Returns:
            int: Number of addresses updated
        """
        try:
            logging.info("update_full_address_with_building_names(): Starting full_address standardization...")
            
            # Get all addresses to standardize their full_address
            find_addresses_sql = """
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, province_or_state, postal_code, country_id
            FROM address 
            ORDER BY address_id;
            """
            
            addresses_df = self.read_sql_df(find_addresses_sql)
            
            if addresses_df.empty:
                logging.info("update_full_address_with_building_names(): No addresses found.")
                return 0
            
            logging.info(f"update_full_address_with_building_names(): Processing {len(addresses_df)} addresses for standardization")
            
            updated_count = 0
            for _, row in addresses_df.iterrows():
                # Build standardized full_address using the new method
                new_full_address = self.build_full_address(
                    building_name=row['building_name'],
                    street_number=row['street_number'],
                    street_name=row['street_name'],
                    street_type=row['street_type'],
                    city=row['city'],
                    province_or_state=row['province_or_state'],
                    postal_code=row['postal_code'],
                    country_id=row['country_id']
                )
                
                # Only update if the new address is different from current
                current_address = row['full_address'] or ""
                if new_full_address != current_address:
                    update_sql = """
                    UPDATE address 
                    SET full_address = :new_full_address 
                    WHERE address_id = :address_id;
                    """
                    
                    result = self.execute_query(update_sql, {
                        'new_full_address': new_full_address,
                        'address_id': row['address_id']
                    })
                    
                    if result is not None:  # Query executed successfully
                        updated_count += 1
                        logging.debug(f"Updated address_id {row['address_id']}: '{current_address}' -> '{new_full_address}'")
            
            logging.info(f"update_full_address_with_building_names(): Successfully updated {updated_count} addresses")
            return updated_count
            
        except Exception as e:
            logging.error(f"update_full_address_with_building_names(): Error updating full_address records: {e}")
            raise
        

if __name__ == "__main__":
    # Load configuration from a YAML file
    # Setup centralized logging
    from logging_config import setup_logging
    setup_logging('db')

    config = load_config()

    logging.info("\n\ndb.py starting...")

    start_time = datetime.now()
    logging.info("\n\nMain: Started the process at %s", start_time)

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    # Perform deduplication and delete old events
    db_handler.driver()

    # Count events and urls after cleanup
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info("Main: Finished the process at %s", end_time)
    logging.info("Main: Total time taken: %s", end_time - start_time)

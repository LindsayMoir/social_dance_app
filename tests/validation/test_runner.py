#!/usr/bin/env python3
"""
Pre-commit Validation Test Runner

Main orchestrator for validation testing framework.
Runs both scraping validation and chatbot batch testing.

This script can be run standalone or called from pipeline.py.
Exit codes:
- 0: Tests passed or completed with warnings
- 1: Tests failed (critical issues found)

Author: Claude Code
Version: 1.0.0
"""

import sys
import os
import csv
import json
import ast
import asyncio
import subprocess
from collections import Counter
import re
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from difflib import SequenceMatcher

# Add src to path for imports (calculate path relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, script_dir)  # Also add tests/validation for local imports

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('src/.env')  # Load from src/.env since that's where credentials are stored

from datetime import datetime, timedelta
import logging
import yaml
import random
import requests
from bs4 import BeautifulSoup, Comment

# Import validation modules
from scraping_validator import ScrapingValidator
from chatbot_evaluator import (
    TestQuestionGenerator,
    ChatbotTestExecutor,
    ChatbotScorer,
    generate_chatbot_report
)

# Import existing utilities
from classifier_training_queue import (
    build_classifier_training_queue,
    filter_classifier_review_candidates,
)
from db import DatabaseHandler
from evaluation_holdout import load_dev_urls, load_gold_holdout_urls, normalize_evaluation_url
from llm import LLMHandler
from logging_config import setup_logging
from email_notifier import send_report_email
from output_paths import (
    chatbot_path,
    classifier_path,
    codex_review_path,
    duplicates_path,
    replay_path,
    reports_path,
)
from page_classifier import (
    classify_page_with_confidence,
    is_email_like_input,
    is_google_calendar_like_url,
    is_social_url,
    resolve_prompt_type,
)


class ValidationTestRunner:
    """
    Main orchestrator for validation testing.

    Runs both scraping validation and chatbot testing,
    generates reports, and returns overall status.

    Attributes:
        config (dict): Configuration from config.yaml
        db_handler: DatabaseHandler instance
        llm_handler: LLMHandler instance
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the ValidationTestRunner.

        Args:
            config_path (str): Path to config.yaml file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path

        # Initialize database and LLM handlers
        self.db_handler = DatabaseHandler(self.config)
        self.llm_handler = LLMHandler(config_path=config_path)
        self._replay_fb_scraper = None
        self._replay_fb_module = None
        self._replay_rd_ext_extractor = None
        self.training_csv_path = os.path.join(repo_root, "ml", "training_data", "original_td.csv")

        # Get validation configuration
        self.validation_config = self.config.get('testing', {}).get('validation', {})

        logging.info("ValidationTestRunner initialized")

    @staticmethod
    def _is_clarification_candidate_question(question_text: str) -> bool:
        """
        Heuristic: classify prompts likely to exercise clarification/temporal disambiguation paths.
        """
        text = str(question_text or "").strip().lower()
        if not text:
            return False
        markers = (
            "this weekend",
            "next weekend",
            "this week",
            "next week",
            "tomorrow night",
            "tonight",
            "tomorrow",
            "next month",
            "this month",
            "best place",
            "what's happening",
            "what dance events are happening",
            "where can i dance",
        )
        return any(marker in text for marker in markers)

    def _sample_questions_with_clarification_quota(
        self,
        questions: list[dict],
        sample_n: int,
        clarification_ratio: float,
    ) -> tuple[list[dict], int, int]:
        """
        Sample questions while ensuring a minimum clarification-candidate share.
        """
        if sample_n <= 0 or len(questions) <= sample_n:
            selected = list(questions)
            clar_count = sum(
                1 for q in selected
                if self._is_clarification_candidate_question((q or {}).get("question", ""))
            )
            return selected, clar_count, len(selected)

        ratio = max(0.0, min(1.0, float(clarification_ratio)))
        required_clar = int(round(sample_n * ratio))

        clar_pool: list[dict] = []
        non_clar_pool: list[dict] = []
        for question in questions:
            q_text = (question or {}).get("question", "")
            if self._is_clarification_candidate_question(q_text):
                clar_pool.append(question)
            else:
                non_clar_pool.append(question)

        random.shuffle(clar_pool)
        random.shuffle(non_clar_pool)
        selected: list[dict] = []
        selected.extend(clar_pool[:required_clar])
        remaining = sample_n - len(selected)

        if remaining > 0:
            selected.extend(non_clar_pool[:remaining])
            remaining = sample_n - len(selected)

        if remaining > 0:
            clar_overflow_start = min(required_clar, len(clar_pool))
            selected.extend(clar_pool[clar_overflow_start:clar_overflow_start + remaining])

        random.shuffle(selected)
        selected = selected[:sample_n]
        clar_count = sum(
            1 for q in selected
            if self._is_clarification_candidate_question((q or {}).get("question", ""))
        )
        return selected, clar_count, len(selected)

    def run_all_validations(self) -> dict:
        """
        Run both scraping validation and chatbot testing.

        Returns:
            dict: Combined results with overall status
        """
        logging.info("=" * 80)
        logging.info("PRE-COMMIT VALIDATION TESTS")
        logging.info("=" * 80)

        results = {
            'timestamp': datetime.now().isoformat(),
            'scraping_validation': None,
            'chatbot_testing': None,
            'accuracy_replay': None,
            'chatbot_problem_category_gate': None,
            'overall_status': 'PASS',
            'reliability_gates': None,
        }

        # 1. SCRAPING VALIDATION
        try:
            logging.info("\n" + "=" * 70)
            logging.info("SCRAPING VALIDATION")
            logging.info("=" * 70)

            validator = ScrapingValidator(self.db_handler, self.config)

            # Classify important URLs
            important_urls = validator.classify_important_urls()
            logging.info(f"Identified {len(important_urls)} important URLs")

            # Check for failures
            failures = validator.check_scraping_failures(important_urls)
            logging.info(f"Found {len(failures)} scraping failures")

            # Check source distribution
            distribution_check = validator.check_source_distribution()

            # Generate report
            scraping_report = validator.generate_report(failures, important_urls)
            scraping_report['source_distribution'] = distribution_check  # Add to report
            results['scraping_validation'] = scraping_report

            # Check for critical failures
            whitelist_failures = scraping_report['summary'].get('whitelist_failures', 0)
            edge_case_failures = scraping_report['summary'].get('edge_case_failures', 0)

            if whitelist_failures > 0 or edge_case_failures > 0:
                results['overall_status'] = 'WARNING'
                logging.warning(
                    f"⚠️  Critical failures detected: {whitelist_failures} whitelist, "
                    f"{edge_case_failures} edge case URLs failed"
                )

            # Check source distribution status
            if distribution_check['status'] == 'FAIL':
                results['overall_status'] = 'FAIL'
                logging.error("❌ Source distribution check FAILED")
            elif distribution_check['status'] == 'WARNING':
                if results['overall_status'] == 'PASS':
                    results['overall_status'] = 'WARNING'
                logging.warning("⚠️  Source distribution check has warnings")

        except Exception as e:
            logging.error(f"Scraping validation failed: {e}", exc_info=True)
            results['scraping_validation'] = {'error': str(e)}
            results['overall_status'] = 'WARNING'  # Don't fail entire validation

        # 2. CHATBOT TESTING
        try:
            logging.info("\n" + "=" * 70)
            logging.info("CHATBOT BATCH TESTING")
            logging.info("=" * 70)

            # Generate questions
            chatbot_config = self.validation_config.get('chatbot', {})
            template_file = chatbot_config.get(
                'question_templates',
                'tests/data/chatbot_test_questions.yaml'
            )

            question_gen = TestQuestionGenerator(template_file)
            questions = question_gen.generate_all_questions()
            logging.info(f"Generated {len(questions)} test questions")

            # Optionally sample a random subset of questions based on config.crawling.random_test_limit
            try:
                sample_n = int(self.config.get('crawling', {}).get('random_test_limit', 0) or 0)
            except Exception:
                sample_n = 0

            if sample_n > 0 and len(questions) > sample_n:
                clarification_ratio = float(chatbot_config.get("clarification_sample_ratio", 0.50) or 0.50)
                questions, clar_count, selected_count = self._sample_questions_with_clarification_quota(
                    questions=questions,
                    sample_n=sample_n,
                    clarification_ratio=clarification_ratio,
                )
                logging.info(
                    "Randomly selected %s questions (random_test_limit=%s, clarification_ratio_target=%.0f%%, actual=%.0f%% [%s/%s])",
                    len(questions),
                    sample_n,
                    clarification_ratio * 100.0,
                    (clar_count / selected_count * 100.0) if selected_count else 0.0,
                    clar_count,
                    selected_count,
                )

            # Execute tests
            executor = ChatbotTestExecutor(self.config, self.db_handler)
            test_results = executor.execute_all_tests(questions)
            logging.info(f"Executed {len(test_results)} tests")

            # Score results
            scorer = ChatbotScorer(self.llm_handler)
            scored_results = scorer.score_all_results(test_results)
            logging.info("Scoring complete")
            self._persist_validation_chatbot_metrics(test_results)

            # Generate report
            output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')
            chatbot_report = generate_chatbot_report(scored_results, output_dir)
            results['chatbot_testing'] = chatbot_report

            # Check thresholds
            avg_score = chatbot_report['summary']['average_score']
            exec_rate = chatbot_report['summary']['execution_success_rate']

            score_threshold = chatbot_config.get('score_threshold', 70)
            exec_threshold = chatbot_config.get('execution_threshold', 0.90)
            problem_gate = self._evaluate_chatbot_problem_category_gate(chatbot_report, chatbot_config)
            chatbot_report['category_gate'] = problem_gate
            results['chatbot_problem_category_gate'] = problem_gate
            self._write_chatbot_problem_category_regressions(output_dir, chatbot_report)

            if avg_score < score_threshold:
                results['overall_status'] = 'FAIL'
                logging.error(
                    f"❌ Chatbot average score ({avg_score:.1f}) "
                    f"below threshold ({score_threshold})"
                )
            elif exec_rate < exec_threshold:
                if results['overall_status'] == 'PASS':
                    results['overall_status'] = 'WARNING'
                logging.warning(
                    f"⚠️  SQL execution rate ({exec_rate:.2%}) "
                    f"below threshold ({exec_threshold:.0%})"
                )
            else:
                logging.info(
                    f"✅ Chatbot tests passed: avg_score={avg_score:.1f}, "
                    f"exec_rate={exec_rate:.2%}"
                )

            if problem_gate.get("status") == "FAIL":
                results['overall_status'] = 'FAIL'
                logging.error("❌ Chatbot category gate failed: %s", problem_gate.get("detail", ""))
            elif problem_gate.get("status") == "WARNING" and results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'
                logging.warning("⚠️  Chatbot category gate warning: %s", problem_gate.get("detail", ""))

        except Exception as e:
            logging.error(f"Chatbot testing failed: {e}", exc_info=True)
            results['chatbot_testing'] = {'error': str(e)}
            results['overall_status'] = 'FAIL'

        # 2.5 REPLAY ACCURACY (query -> re-scrape -> row-level compare)
        try:
            replay_summary = self._run_accuracy_replay_assessment(results)
            results["accuracy_replay"] = replay_summary
            results["classifier_training_queue"] = self._build_classifier_training_queue_summary(replay_summary)
            results["classifier_performance"] = self._summarize_classifier_performance(
                run_id=str(replay_summary.get("run_id") or "").strip()
            )
        except Exception as e:
            logging.error("Replay accuracy assessment failed: %s", e, exc_info=True)
            results["accuracy_replay"] = {"error": str(e), "status": "ERROR"}
            results["classifier_training_queue"] = {"error": str(e), "status": "ERROR", "candidates": []}
            results["classifier_performance"] = {"error": str(e), "status": "ERROR"}
            if results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'

        # 3. RELIABILITY GATE EVALUATION (PHASE 2)
        try:
            alias_audit_summary = self._summarize_address_alias_audit()
            llm_activity_summary = self._summarize_llm_provider_activity(results.get('timestamp'))
            scraper_network_summary = self._summarize_scraper_network_health(results.get('timestamp'))
            fb_block_summary = self._summarize_fb_block_health(results.get('timestamp'))
            reliability_scorecard = self._summarize_reliability_scorecard(
                results, llm_activity_summary, alias_audit_summary, scraper_network_summary, fb_block_summary
            )
            gate_eval = self._evaluate_reliability_gates(reliability_scorecard)
            results['reliability_gates'] = gate_eval

            if gate_eval.get('status') == 'FAIL':
                results['overall_status'] = 'FAIL'
                failed = gate_eval.get('failed_gates', [])
                logging.error(
                    "❌ Reliability gates failed (%d): %s",
                    len(failed),
                    ", ".join(g.get('name', 'unknown') for g in failed),
                )
            elif gate_eval.get('status') == 'WARNING' and results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'
        except Exception as e:
            logging.error(f"Reliability gate evaluation failed: {e}", exc_info=True)
            if results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'
            results['reliability_gates'] = {
                'enabled': True,
                'status': 'ERROR',
                'error': str(e),
                'failed_gates': [],
                'gate_results': [],
                'thresholds': {},
            }

        # 4. GENERATE COMPREHENSIVE HTML REPORT
        try:
            if self.validation_config.get('reporting', {}).get('generate_html', True):
                self._generate_html_report(results)
        except Exception as e:
            logging.error(f"HTML report generation failed: {e}", exc_info=True)

        # 5. EMAIL NOTIFICATION
        # Email delivery is handled by pipeline remediation_planner_step so
        # attachments can include both remediation_plan.md and the HTML report.
        logging.info("Validation email notification skipped (handled in remediation_planner_step).")

        # Final status
        logging.info("\n" + "=" * 80)
        logging.info(f"VALIDATION COMPLETE - STATUS: {results['overall_status']}")
        logging.info("=" * 80)
        self._close_replay_fb_scraper()
        self._close_replay_rd_ext_extractor()

        return results

    def _persist_validation_chatbot_metrics(self, test_results: list[dict]) -> None:
        """
        Persist synthetic chatbot performance rows for validation-simulated queries.

        Validation runs bypass FastAPI `/query` and `/confirm`, so they would otherwise
        not contribute to chatbot_request_metrics/chatbot_stage_metrics.
        """
        if not test_results:
            return

        request_upsert_sql = """
            INSERT INTO chatbot_request_metrics
                (request_id, endpoint, started_at, finished_at, duration_ms, result_type, user_input, sql_snippet, has_response, updated_at)
            VALUES
                (:request_id, :endpoint, :started_at, :finished_at, :duration_ms, :result_type, :user_input, :sql_snippet, :has_response, :updated_at)
            ON CONFLICT (request_id)
            DO UPDATE SET
                endpoint = EXCLUDED.endpoint,
                started_at = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at,
                duration_ms = EXCLUDED.duration_ms,
                result_type = EXCLUDED.result_type,
                user_input = EXCLUDED.user_input,
                sql_snippet = EXCLUDED.sql_snippet,
                has_response = EXCLUDED.has_response,
                updated_at = EXCLUDED.updated_at
        """
        stage_insert_sql = """
            INSERT INTO chatbot_stage_metrics
                (request_id, endpoint, stage, started_at, finished_at, duration_ms, metadata_json)
            VALUES
                (:request_id, :endpoint, :stage, :started_at, :finished_at, :duration_ms, :metadata_json)
        """

        run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
        persisted_requests = 0
        persisted_stages = 0

        for idx, result in enumerate(test_results, start=1):
            request_id = f"validation-{run_tag}-{idx:04d}"
            endpoint = "/query"
            finished_at = self._parse_iso_datetime(str(result.get("timestamp") or "")) or datetime.now()
            duration_ms = float(result.get("execution_duration_ms") or 0.0)
            if duration_ms <= 0:
                duration_ms = 1.0
            started_at = finished_at - timedelta(milliseconds=duration_ms)
            sql_query = str(result.get("sql_query") or "")
            sql_snippet = sql_query[:1000] if sql_query else ""
            result_type = "validation_success" if bool(result.get("execution_success")) else "validation_failure"
            has_response = bool((result.get("result_count") or 0) > 0)

            try:
                self.db_handler.execute_query(
                    request_upsert_sql,
                    {
                        "request_id": request_id,
                        "endpoint": endpoint,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_ms": duration_ms,
                        "result_type": result_type,
                        "user_input": str(result.get("question") or ""),
                        "sql_snippet": sql_snippet,
                        "has_response": has_response,
                        "updated_at": datetime.now(),
                    },
                )
                persisted_requests += 1
            except Exception as e:
                logging.warning("_persist_validation_chatbot_metrics: request upsert failed for %s: %s", request_id, e)
                continue

            stage_meta = {
                "category": str(result.get("category") or ""),
                "execution_success": bool(result.get("execution_success")),
                "clarification_depth_target": int(result.get("clarification_depth_target") or 0),
                "clarification_depth_achieved": int(result.get("clarification_depth_achieved") or 0),
            }
            try:
                self.db_handler.execute_query(
                    stage_insert_sql,
                    {
                        "request_id": request_id,
                        "endpoint": endpoint,
                        "stage": "validation_execute_test",
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_ms": duration_ms,
                        "metadata_json": json.dumps(stage_meta),
                    },
                )
                persisted_stages += 1
            except Exception as e:
                logging.warning("_persist_validation_chatbot_metrics: stage insert failed for %s: %s", request_id, e)

            clarification_turns = int(result.get("clarification_turns_executed") or 0)
            if clarification_turns <= 0:
                continue
            per_turn_ms = max(1.0, round(duration_ms / max(clarification_turns, 1), 1))
            for turn in range(clarification_turns):
                confirm_request_id = f"{request_id}-confirm-{turn + 1}"
                confirm_finished_at = finished_at
                confirm_started_at = confirm_finished_at - timedelta(milliseconds=per_turn_ms)
                try:
                    self.db_handler.execute_query(
                        request_upsert_sql,
                        {
                            "request_id": confirm_request_id,
                            "endpoint": "/confirm",
                            "started_at": confirm_started_at,
                            "finished_at": confirm_finished_at,
                            "duration_ms": per_turn_ms,
                            "result_type": "validation_clarification_turn",
                            "user_input": str(result.get("question") or ""),
                            "sql_snippet": "",
                            "has_response": has_response,
                            "updated_at": datetime.now(),
                        },
                    )
                    persisted_requests += 1
                except Exception as e:
                    logging.warning(
                        "_persist_validation_chatbot_metrics: confirm request upsert failed for %s: %s",
                        confirm_request_id,
                        e,
                    )
                    continue
                try:
                    self.db_handler.execute_query(
                        stage_insert_sql,
                        {
                            "request_id": confirm_request_id,
                            "endpoint": "/confirm",
                            "stage": "validation_clarification_turn",
                            "started_at": confirm_started_at,
                            "finished_at": confirm_finished_at,
                            "duration_ms": per_turn_ms,
                            "metadata_json": json.dumps({"turn": turn + 1, "parent_request_id": request_id}),
                        },
                    )
                    persisted_stages += 1
                except Exception as e:
                    logging.warning(
                        "_persist_validation_chatbot_metrics: confirm stage insert failed for %s: %s",
                        confirm_request_id,
                        e,
                    )

        logging.info(
            "_persist_validation_chatbot_metrics: persisted request rows=%d stage rows=%d from validation results=%d",
            persisted_requests,
            persisted_stages,
            len(test_results),
        )

    @staticmethod
    def _normalize_text_value(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"", "none", "null", "nan", "<na>"}:
            return ""
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _normalize_optional_int(value: Any) -> int | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            coerced = int(value)
            return coerced if coerced > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_dance_style_tokens(value: Any) -> tuple[str, ...]:
        text = str(value or "").strip().lower()
        if not text:
            return tuple()
        text = text.replace("&", ",").replace("/", ",")
        parts = [re.sub(r"\s+", " ", token).strip() for token in text.split(",")]
        normalized = sorted({token for token in parts if token})
        return tuple(normalized)

    @staticmethod
    def _descriptions_equivalent(baseline_description: str, replay_description: str) -> bool:
        if not baseline_description or not replay_description:
            return False
        if baseline_description == replay_description:
            return True
        if baseline_description in replay_description or replay_description in baseline_description:
            return True
        return SequenceMatcher(None, baseline_description, replay_description).ratio() >= 0.85

    def _resolve_replay_address_id(self, replay_row: dict) -> int | None:
        """Resolve replay location text to an existing address_id without invoking LLM paths."""
        if not isinstance(replay_row, dict) or not getattr(self, "db_handler", None):
            return None
        location = str(replay_row.get("location") or "").strip()
        if not location:
            return None
        lookup_methods = [
            getattr(self.db_handler, "lookup_raw_location", None),
            getattr(self.db_handler, "quick_address_lookup", None),
        ]
        for method in lookup_methods:
            if not callable(method):
                continue
            try:
                address_id = method(location)
            except Exception:
                continue
            normalized = self._normalize_optional_int(address_id)
            if normalized is not None:
                return normalized
        return None

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, str(a or ""), str(b or "")).ratio()

    @staticmethod
    def _name_contains_variant(a: str, b: str) -> bool:
        aa = re.sub(r"\s+", " ", str(a or "").strip().lower())
        bb = re.sub(r"\s+", " ", str(b or "").strip().lower())
        if not aa or not bb:
            return False
        if aa in bb or bb in aa:
            return True
        aa_base = aa.split("@", 1)[0].strip()
        bb_base = bb.split("@", 1)[0].strip()
        if aa_base and bb_base and (aa_base in bb_base or bb_base in aa_base):
            return True
        return False

    @staticmethod
    def _is_placeholder_source(value: str) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized in {"", "unknown", "none", "null", "extracted_text", "extracted text", "text extracted"}

    @staticmethod
    def _is_calendar_event_url(url: str) -> bool:
        return is_google_calendar_like_url(url)

    @staticmethod
    def _normalize_url_value(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = urlparse(text)
            scheme = (parsed.scheme or "https").lower()
            netloc = (parsed.netloc or "").lower()
            path = (parsed.path or "").rstrip("/")
            return urlunparse((scheme, netloc, path, "", parsed.query, ""))
        except Exception:
            return text.lower().rstrip("/")

    @staticmethod
    def _normalize_date_value(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except Exception:
            return text[:10]

    @staticmethod
    def _normalize_time_value(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        # Handle explicit 12-hour clock representations first.
        for fmt in ("%I:%M:%S %p", "%I:%M %p"):
            try:
                parsed = datetime.strptime(text[:11] if fmt == "%I:%M:%S %p" else text[:8], fmt)
                return parsed.strftime("%H:%M:%S")
            except Exception:
                continue
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(text[:8], fmt) if fmt == "%H:%M:%S" else datetime.strptime(text[:5], fmt)
                return parsed.strftime("%H:%M:%S")
            except Exception:
                continue
        return text

    @staticmethod
    def _times_equivalent_with_12h_guard(left: str, right: str) -> bool:
        if not left or not right:
            return left == right
        if left == right:
            return True
        try:
            lt = datetime.strptime(left[:8], "%H:%M:%S")
            rt = datetime.strptime(right[:8], "%H:%M:%S")
            diff = abs((lt - rt).total_seconds())
            # Catch common 12h-vs-24h normalization drift (e.g., 02:00 vs 14:00).
            return diff in {12 * 3600}
        except Exception:
            return False

    @staticmethod
    def _is_social_platform_url(url: str) -> bool:
        return is_social_url(url)

    def _extract_visible_text_for_replay(self, html: str) -> str:
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "template", "svg"]):
                tag.decompose()
            for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
                comment.extract()
            main_node = soup.select_one("main, article, [role='main'], .event, .event-content")
            text_chunks: list[str] = []
            if main_node is not None:
                text_chunks.append(" ".join(main_node.stripped_strings))
            text_chunks.append(" ".join(soup.stripped_strings))
            visible = " ".join(chunk for chunk in text_chunks if chunk).strip()
            return visible[:20000]
        except Exception:
            return str(html or "")[:20000]

    def _get_replay_fb_scraper(self):
        """
        Lazily initialize fb.py scraper for replay of Facebook/Instagram URLs.
        Reuses one authenticated browser session across replay rows.
        """
        if self._replay_fb_scraper is not None:
            return self._replay_fb_scraper
        try:
            import fb as fb_module

            self._replay_fb_module = fb_module
            fb_module.llm_handler = self.llm_handler
            fb_module.db_handler = self.db_handler
            self._replay_fb_scraper = fb_module.FacebookEventScraper(config_path=self.config_path)
            return self._replay_fb_scraper
        except Exception as e:
            logging.warning("Replay fb.py initialization failed: %s", e)
            self._replay_fb_scraper = None
            return None

    def _close_replay_fb_scraper(self) -> None:
        scraper = self._replay_fb_scraper
        self._replay_fb_scraper = None
        if scraper is None:
            return
        try:
            if hasattr(scraper, "_safe_shutdown_browser"):
                scraper._safe_shutdown_browser()
        except Exception as e:
            logging.warning("Replay fb.py shutdown failed: %s", e)

    def _get_replay_rd_ext_extractor(self):
        """
        Lazily initialize rd_ext extractor for replay of listing/calendar pages.
        """
        if self._replay_rd_ext_extractor is not None:
            return self._replay_rd_ext_extractor
        try:
            from rd_ext import ReadExtract

            extractor = ReadExtract(config_path=self.config_path)
            asyncio.run(extractor.init_browser())
            self._replay_rd_ext_extractor = extractor
            return extractor
        except Exception as e:
            logging.warning("Replay rd_ext initialization failed: %s", e)
            self._replay_rd_ext_extractor = None
            return None

    def _close_replay_rd_ext_extractor(self) -> None:
        extractor = self._replay_rd_ext_extractor
        self._replay_rd_ext_extractor = None
        if extractor is None:
            return
        try:
            asyncio.run(extractor.close())
        except Exception as e:
            logging.warning("Replay rd_ext shutdown failed: %s", e)

    @staticmethod
    def _extract_replay_links(url: str, html: str) -> list[str]:
        """
        Extract normalized href targets from replay HTML for classifier/routing use.
        """
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []
        links: list[str] = []
        for node in soup.find_all("a", href=True):
            href = str(node.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(url, href)
            elif not href.startswith("http"):
                href = urljoin(url, href)
            links.append(href)
        return links

    def _parse_replay_events_from_text(self, source_url: str, extracted_text: str, replay_url: str | None = None) -> list[dict]:
        """
        Parse one extracted event text block into normalized replay event rows.
        """
        text = str(extracted_text or "").strip()
        if not text:
            return []
        prompt, schema_type = self.llm_handler.generate_prompt(source_url, text, source_url)
        if schema_type is None:
            prompt, schema_type = self.llm_handler.generate_prompt(source_url, text, "default")
        if schema_type is None:
            return []

        parsed = None
        for _attempt in range(1, 3):
            llm_response = self.llm_handler.query_llm(source_url, prompt, schema_type)
            if not llm_response:
                continue
            parsed = self.llm_handler.extract_and_parse_json(llm_response, source_url, schema_type)
            if parsed:
                break
        if not parsed:
            return []

        normalized_url = self._normalize_url_value(replay_url or source_url)
        normalized_events: list[dict] = []
        for event in parsed:
            if not isinstance(event, dict):
                continue
            event_raw = dict(event)
            if replay_url:
                event_raw["replay_child_url"] = replay_url
            normalized_events.append(
                {
                    "event_name": str(event.get("event_name") or "").strip(),
                    "start_date": self._normalize_date_value(event.get("start_date")),
                    "start_time": self._normalize_time_value(event.get("start_time")),
                    "source": str(event.get("source") or "").strip(),
                    "location": str(event.get("location") or "").strip(),
                    "url": normalized_url,
                    "raw": event_raw,
                }
            )
        return normalized_events

    def _fetch_replay_events_via_rd_ext(self, url: str) -> dict:
        """
        Replay fetch via rd_ext for listing/calendar pages with child event links.
        """
        extractor = self._get_replay_rd_ext_extractor()
        if extractor is None:
            return {
                "ok": False,
                "category": "parser_or_llm_failure",
                "details": "rd_ext_unavailable_for_replay",
                "events": [],
            }

        try:
            event_payloads = asyncio.run(
                extractor.extract_calendar_events(url, venue_name=(urlparse(url).netloc or "calendar"))
            )
        except Exception as e:
            return {
                "ok": False,
                "category": "parser_or_llm_failure",
                "details": f"rd_ext_replay_exception:{e}",
                "events": [],
            }

        normalized_events: list[dict] = []
        for child_url, event_text in event_payloads or []:
            normalized_events.extend(
                self._parse_replay_events_from_text(
                    source_url=child_url or url,
                    extracted_text=event_text,
                    replay_url=url,
                )
            )

        if not normalized_events:
            return {
                "ok": False,
                "category": "no_event_extracted_replay",
                "details": "rd_ext_no_normalized_events",
                "events": [],
            }
        return {"ok": True, "category": "", "details": "rd_ext_replay", "events": normalized_events}

    def _fetch_replay_events_for_social_url(self, url: str) -> dict:
        """
        Replay fetch for Facebook/Instagram URLs via fb.py methods.
        """
        fb_scraper = self._get_replay_fb_scraper()
        if fb_scraper is None:
            return {
                "ok": False,
                "category": "social_platform_scraper_init_failed",
                "details": "fb.py scraper unavailable for replay",
                "events": [],
            }

        try:
            if not fb_scraper.navigate_and_maybe_login(url):
                reason = "navigation_failed"
                try:
                    reason = str(fb_scraper._get_last_access_reason(url))
                except Exception:
                    pass
                return {
                    "ok": False,
                    "category": "url_unreachable_replay",
                    "details": f"social_access_failed:{reason}",
                    "events": [],
                }

            extracted_text = fb_scraper.extract_event_text(url, assume_navigated=True)
            if not extracted_text:
                return {
                    "ok": False,
                    "category": "no_event_extracted_replay",
                    "details": "social_no_text",
                    "events": [],
                }

            # For non-event Facebook pages, try fb.py's relevance slicer first.
            if "facebook.com" in str(url).lower() and "/events/" not in str(url).lower():
                try:
                    relevant = fb_scraper.extract_relevant_text(extracted_text, url)
                    if relevant:
                        extracted_text = relevant
                except Exception:
                    pass

            prompt_type = resolve_prompt_type(url, fallback_prompt_type="fb")
            prompt, schema_type = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
            if schema_type is None:
                prompt, schema_type = self.llm_handler.generate_prompt(url, extracted_text, "default")
            if schema_type is None:
                return {"ok": False, "category": "parser_or_llm_failure", "details": "missing_schema_type", "events": []}

            parsed = None
            last_llm_error = ""
            for _attempt in range(1, 3):
                llm_response = self.llm_handler.query_llm(url, prompt, schema_type)
                if not llm_response:
                    last_llm_error = "empty_llm_response"
                    continue
                parsed = self.llm_handler.extract_and_parse_json(llm_response, url, schema_type)
                if parsed:
                    break
                last_llm_error = "parsed_empty"

            if not parsed:
                category = "no_event_extracted_replay" if last_llm_error == "parsed_empty" else "parser_or_llm_failure"
                return {"ok": False, "category": category, "details": last_llm_error or "llm_failed", "events": []}

            normalized_events: list[dict] = []
            for event in parsed:
                if not isinstance(event, dict):
                    continue
                normalized_events.append(
                    {
                        "event_name": str(event.get("event_name") or "").strip(),
                        "start_date": self._normalize_date_value(event.get("start_date")),
                        "start_time": self._normalize_time_value(event.get("start_time")),
                        "source": str(event.get("source") or "").strip(),
                        "location": str(event.get("location") or "").strip(),
                        "url": self._normalize_url_value(event.get("url") or url),
                        "raw": event,
                    }
                )
            if not normalized_events:
                return {"ok": False, "category": "no_event_extracted_replay", "details": "no_normalized_events", "events": []}
            return {"ok": True, "category": "", "details": "", "events": normalized_events}
        except Exception as e:
            return {
                "ok": False,
                "category": "parser_or_llm_failure",
                "details": f"social_replay_exception:{e}",
                "events": [],
            }

    def _fetch_replay_events_for_url(self, url: str) -> dict:
        if self._is_social_platform_url(url):
            return self._fetch_replay_events_for_social_url(url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        response = None
        last_request_error = ""
        for _attempt in range(1, 4):
            try:
                candidate = requests.get(url, timeout=25, headers=headers)
                if candidate.status_code < 400:
                    response = candidate
                    break
                last_request_error = f"http_status={candidate.status_code}"
            except Exception as e:
                last_request_error = str(e)
        if response is None:
            return {
                "ok": False,
                "category": "url_unreachable_replay",
                "details": last_request_error or "request_failed",
                "events": [],
            }

        extracted_text = self._extract_visible_text_for_replay(response.text)
        if not extracted_text:
            return {"ok": False, "category": "no_event_extracted_replay", "details": "no_visible_text", "events": []}

        replay_links = self._extract_replay_links(url, response.text)
        try:
            class_decision = classify_page_with_confidence(
                url=url,
                visible_text=extracted_text,
                links=replay_links,
            )
        except Exception:
            class_decision = {}
        if str((class_decision or {}).get("owner_step") or "").strip().lower() == "rd_ext.py":
            rd_ext_payload = self._fetch_replay_events_via_rd_ext(url)
            if rd_ext_payload.get("ok"):
                return rd_ext_payload

        normalized_events = self._parse_replay_events_from_text(source_url=url, extracted_text=extracted_text)
        if not normalized_events:
            return {"ok": False, "category": "no_event_extracted_replay", "details": "no_normalized_events", "events": []}
        return {"ok": True, "category": "", "details": "", "events": normalized_events}

    def _llm_adjudicate_event_match(self, baseline: dict, replay: dict) -> tuple[bool, str]:
        """
        Ask LLM to judge if baseline/replay represent the same event.
        """
        prompt = (
            "Determine if these two rows are the same real-world event despite metadata differences.\n"
            "Return ONLY JSON: {\"same_event\": true|false, \"reason\": \"...\"}\n\n"
            f"Baseline: {json.dumps(baseline, ensure_ascii=True)}\n"
            f"Replay: {json.dumps(replay, ensure_ascii=True)}\n"
            "Use practical tolerance for source/location wording differences."
        )
        try:
            response = self.llm_handler.query_llm("", prompt)
            if not response:
                return False, "llm_empty"
            text = str(response).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return False, "llm_non_json"
            payload = json.loads(text[start:end + 1])
            same_event = bool(payload.get("same_event"))
            reason = str(payload.get("reason") or "")
            return same_event, reason
        except Exception as e:
            return False, f"llm_error:{e}"

    def _resolve_baseline_event_id(self, baseline: dict) -> int | None:
        try:
            rows = self.db_handler.execute_query(
                """
                SELECT event_id
                FROM events
                WHERE
                    LOWER(COALESCE(url, '')) = :url
                    AND LOWER(COALESCE(event_name, '')) = :event_name
                    AND start_date::text = :start_date
                    AND COALESCE(start_time::text, '') = :start_time
                ORDER BY event_id DESC
                LIMIT 1
                """,
                {
                    "url": str(baseline.get("url", "")).lower(),
                    "event_name": str(baseline.get("event_name", "")).lower(),
                    "start_date": str(baseline.get("start_date", "")),
                    "start_time": str(baseline.get("start_time", "")),
                },
            )
            if rows:
                return int(rows[0][0])
        except Exception:
            return None
        return None

    def _compare_replay_row(self, baseline_row: dict, replay_payload: dict, strict_time_match: bool) -> dict:
        baseline = {
            "event_name": self._normalize_text_value(baseline_row.get("event_name")),
            "start_date": self._normalize_date_value(baseline_row.get("start_date")),
            "start_time": self._normalize_time_value(baseline_row.get("start_time")),
            "source": self._normalize_text_value(baseline_row.get("source")),
            "location": self._normalize_text_value(baseline_row.get("location")),
            "url": self._normalize_url_value(baseline_row.get("url")),
            "address_id": self._normalize_optional_int(baseline_row.get("address_id")),
            "dance_style": self._normalize_text_value(baseline_row.get("dance_style")),
            "description": self._normalize_text_value(baseline_row.get("description")),
        }

        if not baseline["url"]:
            return {
                "is_match": False,
                "category": "missing_url_baseline",
                "details": "baseline row has empty URL",
                "baseline": baseline,
                "replay": {},
            }

        if self._is_calendar_event_url(baseline["url"]):
            return {
                "is_match": True,
                "category": "",
                "details": "calendar_event_auto_true",
                "baseline": baseline,
                "replay": {},
            }

        if not replay_payload.get("ok"):
            return {
                "is_match": False,
                "category": str(replay_payload.get("category") or "parser_or_llm_failure"),
                "details": str(replay_payload.get("details") or "replay failed"),
                "baseline": baseline,
                "replay": {},
            }

        events = replay_payload.get("events", []) or []
        if not events:
            return {
                "is_match": False,
                "category": "no_event_extracted_replay",
                "details": "replay returned zero events",
                "baseline": baseline,
                "replay": {},
            }

        def score_candidate(candidate: dict) -> int:
            score = 0
            if self._normalize_text_value(candidate.get("event_name")) == baseline["event_name"]:
                score += 4
            if self._normalize_date_value(candidate.get("start_date")) == baseline["start_date"]:
                score += 4
            if self._normalize_url_value(candidate.get("url")) == baseline["url"]:
                score += 3
            if self._normalize_text_value(candidate.get("source")) == baseline["source"]:
                score += 2
            cand_time = self._normalize_time_value(candidate.get("start_time"))
            if cand_time and baseline["start_time"] and cand_time == baseline["start_time"]:
                score += 2
            return score

        ranked = sorted(events, key=score_candidate, reverse=True)
        best = ranked[0]
        best_raw = best.get("raw", {}) if isinstance(best.get("raw"), dict) else {}
        replay = {
            "event_name": self._normalize_text_value(best.get("event_name")),
            "start_date": self._normalize_date_value(best.get("start_date")),
            "start_time": self._normalize_time_value(best.get("start_time")),
            "source": self._normalize_text_value(best.get("source")),
            "location": self._normalize_text_value(best.get("location")),
            "url": self._normalize_url_value(best.get("url")),
            "address_id": self._normalize_optional_int(best_raw.get("address_id")),
            "dance_style": self._normalize_text_value(best_raw.get("dance_style")),
            "description": self._normalize_text_value(best_raw.get("description")),
        }

        duplicate_identity_count = sum(
            1
            for candidate in events
            if (
                self._normalize_text_value(candidate.get("event_name")) == baseline["event_name"]
                and self._normalize_date_value(candidate.get("start_date")) == baseline["start_date"]
                and self._normalize_time_value(candidate.get("start_time")) == baseline["start_time"]
            )
        )
        if duplicate_identity_count > 1:
            return {
                "is_match": False,
                "category": "duplicate_event_identity",
                "details": f"replay produced {duplicate_identity_count} duplicates for identity keys",
                "baseline": baseline,
                "replay": replay,
            }

        name_similarity = self._name_similarity(baseline["event_name"], replay["event_name"])
        name_contains_variant = self._name_contains_variant(baseline["event_name"], replay["event_name"])
        same_date = replay["start_date"] == baseline["start_date"]
        same_time = self._times_equivalent_with_12h_guard(replay["start_time"], baseline["start_time"])
        core_match = (
            replay["url"] == baseline["url"]
            and same_date
            and (name_similarity >= 0.90 or name_contains_variant)
        )
        if strict_time_match:
            core_match = core_match and same_time

        if core_match:
            return {
                "is_match": True,
                "category": "",
                "details": "baseline and replay match on strict core fields",
                "baseline": baseline,
                "replay": replay,
            }

        # Ambiguous differences: ask LLM to adjudicate.
        # This intentionally allows URL/source drift if event identity is otherwise strong.
        if (
            same_date
            and ((not strict_time_match) or same_time)
            and (name_similarity >= 0.75 or name_contains_variant)
        ):
            llm_same, llm_reason = self._llm_adjudicate_event_match(baseline, replay)
            if llm_same:
                return {
                    "is_match": True,
                    "category": "",
                    "details": f"llm_adjudicated_match:{llm_reason}",
                    "baseline": baseline,
                    "replay": replay,
                }

        if replay["start_date"] != baseline["start_date"]:
            category = "wrong_date"
        elif strict_time_match and replay["start_time"] != baseline["start_time"]:
            category = "wrong_time"
        elif baseline["location"] and replay["location"] and replay["location"] != baseline["location"]:
            category = "wrong_location_or_address"
        elif (
            replay["source"] != baseline["source"]
            and not self._is_placeholder_source(replay["source"])
            and not self._is_placeholder_source(baseline["source"])
        ):
            category = "wrong_source"
        elif name_similarity < 0.60:
            category = "content_drift_major"
        else:
            category = "other_mismatch"

        return {
            "is_match": False,
            "category": category,
            "details": "core field mismatch",
            "baseline": baseline,
            "replay": replay,
        }

    def _load_metric_history(self, metric_key: str, days: int = 90) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=max(1, int(days or 90)))
        try:
            rows = self.db_handler.execute_query(
                """
                SELECT mo.created_at, mo.metric_value_numeric
                FROM metric_observations mo
                JOIN metric_definitions md ON md.metric_id = mo.metric_id
                WHERE md.metric_key = :metric_key
                  AND mo.created_at >= :cutoff
                ORDER BY mo.created_at
                """,
                {"metric_key": metric_key, "cutoff": cutoff},
            )
        except Exception:
            rows = []
        history: list[dict] = []
        for row in rows or []:
            ts = row[0]
            val = row[1]
            try:
                history.append(
                    {
                        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "value": float(val),
                    }
                )
            except Exception:
                continue
        return history

    def _summarize_scraper_step_telemetry(self, run_id: str) -> dict:
        """Aggregate per-step scraper telemetry for the requested run from url_scrape_metrics."""
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            return {"available": False, "run_id": normalized_run_id, "steps": {}}

        query = """
            SELECT
                COALESCE(NULLIF(step_name, ''), REPLACE(COALESCE(handled_by, ''), '.py', '')) AS step_norm,
                COUNT(*) AS total_urls,
                SUM(CASE WHEN access_succeeded THEN 1 ELSE 0 END) AS access_success_count,
                SUM(CASE WHEN text_extracted THEN 1 ELSE 0 END) AS text_extracted_count,
                SUM(CASE WHEN keywords_found THEN 1 ELSE 0 END) AS keywords_found_count,
                SUM(CASE WHEN extraction_attempted THEN 1 ELSE 0 END) AS extraction_attempted_count,
                SUM(CASE WHEN extraction_attempted AND extraction_succeeded THEN 1 ELSE 0 END) AS extraction_success_count,
                SUM(CASE WHEN events_written > 0 THEN 1 ELSE 0 END) AS urls_with_events_count,
                SUM(COALESCE(events_written, 0)) AS events_written_total,
                SUM(CASE WHEN ocr_attempted THEN 1 ELSE 0 END) AS ocr_attempted_count,
                SUM(CASE WHEN ocr_succeeded THEN 1 ELSE 0 END) AS ocr_success_count,
                SUM(CASE WHEN vision_attempted THEN 1 ELSE 0 END) AS vision_attempted_count,
                SUM(CASE WHEN vision_succeeded THEN 1 ELSE 0 END) AS vision_success_count,
                SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END) AS fallback_used_count
            FROM url_scrape_metrics
            WHERE run_id = :run_id
            GROUP BY 1
        """
        try:
            rows = self.db_handler.execute_query(query, {"run_id": normalized_run_id}) or []
        except Exception as exc:
            return {"available": False, "run_id": normalized_run_id, "error": str(exc), "steps": {}}

        steps: dict[str, dict[str, Any]] = {}
        for row in rows:
            mapping = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
            step_name = str(mapping.get("step_norm", "") or "").strip()
            if not step_name:
                continue
            total_urls = int(mapping.get("total_urls", 0) or 0)
            extraction_attempted_count = int(mapping.get("extraction_attempted_count", 0) or 0)
            ocr_attempted_count = int(mapping.get("ocr_attempted_count", 0) or 0)
            vision_attempted_count = int(mapping.get("vision_attempted_count", 0) or 0)

            def _pct(numerator: str, denominator: int) -> float | None:
                if denominator <= 0:
                    return None
                return round((int(mapping.get(numerator, 0) or 0) / denominator) * 100.0, 2)

            steps[step_name] = {
                "total_urls": total_urls,
                "access_success_count": int(mapping.get("access_success_count", 0) or 0),
                "text_extracted_count": int(mapping.get("text_extracted_count", 0) or 0),
                "keywords_found_count": int(mapping.get("keywords_found_count", 0) or 0),
                "extraction_attempted_count": extraction_attempted_count,
                "extraction_success_count": int(mapping.get("extraction_success_count", 0) or 0),
                "urls_with_events_count": int(mapping.get("urls_with_events_count", 0) or 0),
                "events_written_total": int(mapping.get("events_written_total", 0) or 0),
                "ocr_attempted_count": ocr_attempted_count,
                "ocr_success_count": int(mapping.get("ocr_success_count", 0) or 0),
                "vision_attempted_count": vision_attempted_count,
                "vision_success_count": int(mapping.get("vision_success_count", 0) or 0),
                "fallback_used_count": int(mapping.get("fallback_used_count", 0) or 0),
                "access_success_rate_pct": _pct("access_success_count", total_urls),
                "text_extracted_rate_pct": _pct("text_extracted_count", total_urls),
                "keyword_hit_rate_pct": _pct("keywords_found_count", total_urls),
                "extraction_success_rate_pct": _pct("extraction_success_count", extraction_attempted_count),
                "url_event_hit_rate_pct": _pct("urls_with_events_count", total_urls),
                "ocr_success_rate_pct": _pct("ocr_success_count", ocr_attempted_count),
                "vision_success_rate_pct": _pct("vision_success_count", vision_attempted_count),
                "fallback_usage_rate_pct": _pct("fallback_used_count", total_urls),
            }

        return {
            "available": bool(steps),
            "run_id": normalized_run_id,
            "steps": steps,
        }

    def _persist_scraper_step_telemetry_metrics(
        self,
        *,
        run_id: str,
        scraper_telemetry_summary: dict,
        runtime_summary: dict,
    ) -> None:
        """Persist per-scraper telemetry rates for DB-backed trend charts."""
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "record_metric_observation"):
            return
        if not isinstance(scraper_telemetry_summary, dict) or not scraper_telemetry_summary.get("available"):
            return
        window_start = self._parse_iso_datetime(str(runtime_summary.get("start_ts", "") or ""))
        window_end = self._parse_iso_datetime(str(runtime_summary.get("end_ts", "") or "")) or datetime.now()
        description_map = {
            "access_success_rate_pct": ("percentage of scraper URLs reached successfully", True),
            "text_extracted_rate_pct": ("percentage of scraper URLs yielding extracted text", True),
            "keyword_hit_rate_pct": ("percentage of scraper URLs containing target keywords", True),
            "extraction_success_rate_pct": ("percentage of attempted scraper URLs producing extracted events", True),
            "url_event_hit_rate_pct": ("percentage of scraper URLs yielding at least one event", True),
            "ocr_success_rate_pct": ("percentage of OCR attempts succeeding", True),
            "vision_success_rate_pct": ("percentage of vision attempts succeeding", True),
            "fallback_usage_rate_pct": ("percentage of scraper URLs using fallback logic", False),
        }
        for step_name, payload in (scraper_telemetry_summary.get("steps") or {}).items():
            for metric_name, (description_suffix, higher_is_better) in description_map.items():
                value = payload.get(metric_name)
                if value is None:
                    continue
                self.db_handler.record_metric_observation(
                    run_id=str(run_id or ""),
                    metric_key=f"scraper_{step_name}_{metric_name}",
                    metric_value_numeric=float(value),
                    metric_unit="percent",
                    description=f"Scraper telemetry for {step_name}: {description_suffix}",
                    higher_is_better=higher_is_better,
                    window_start=window_start,
                    window_end=window_end,
                )

    def _build_phase1_telemetry_integrity_summary(self, run_id: str) -> dict:
        """Load the canonical Phase 1 telemetry integrity report from the DB layer."""
        safe_run_id = str(run_id or "").strip()
        if not safe_run_id:
            return {"available": False, "run_id": "", "status": "FAIL", "violations": ["missing_run_id"]}
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "build_phase1_telemetry_integrity_report"):
            return {
                "available": False,
                "run_id": safe_run_id,
                "status": "FAIL",
                "violations": ["db_handler_missing_phase1_telemetry_integrity_report"],
            }
        try:
            summary = self.db_handler.build_phase1_telemetry_integrity_report(safe_run_id)
        except Exception as exc:
            return {
                "available": False,
                "run_id": safe_run_id,
                "status": "FAIL",
                "violations": [f"phase1_telemetry_integrity_error:{exc}"],
            }
        if not isinstance(summary, dict):
            return {
                "available": False,
                "run_id": safe_run_id,
                "status": "FAIL",
                "violations": ["phase1_telemetry_integrity_invalid_payload"],
            }
        return summary

    def _build_scraper_telemetry_html(self, scraper_telemetry_summary: dict | None) -> str:
        """Render current-run scraper telemetry plus DB-backed trend charts."""
        if not isinstance(scraper_telemetry_summary, dict) or not scraper_telemetry_summary:
            return "<p class='error-box'>❌ Scraper telemetry unavailable</p>"
        if scraper_telemetry_summary.get("error"):
            return (
                "<p class='error-box'>❌ Scraper telemetry failed: "
                f"{self._escape_html(str(scraper_telemetry_summary.get('error')))}</p>"
            )
        steps = scraper_telemetry_summary.get("steps") if isinstance(scraper_telemetry_summary.get("steps"), dict) else {}
        if not steps:
            return "<p class='error-box'>❌ No scraper telemetry rows were recorded for this run.</p>"

        current_rows = []
        for step_name in ("scraper", "fb", "rd_ext", "ebs", "images"):
            payload = steps.get(step_name)
            if not payload:
                continue
            current_rows.append(
                "<tr>"
                f"<td>{self._escape_html(step_name)}</td>"
                f"<td>{int(payload.get('total_urls', 0) or 0)}</td>"
                f"<td>{self._escape_html(str(payload.get('access_success_rate_pct', 'N/A')))}%</td>"
                f"<td>{self._escape_html(str(payload.get('text_extracted_rate_pct', 'N/A')))}%</td>"
                f"<td>{self._escape_html(str(payload.get('keyword_hit_rate_pct', 'N/A')))}%</td>"
                f"<td>{self._escape_html(str(payload.get('url_event_hit_rate_pct', 'N/A')))}%</td>"
                f"<td>{self._escape_html(str(payload.get('extraction_success_rate_pct', 'N/A')))}%</td>"
                f"<td>{int(payload.get('events_written_total', 0) or 0)}</td>"
                "</tr>"
            )

        images_payload = steps.get("images") or {}
        image_detail_html = ""
        if images_payload:
            image_detail_html = (
                "<h3>Images OCR / Vision Detail</h3>"
                "<table><tr><th>Metric</th><th>Count</th><th>Rate</th></tr>"
                f"<tr><td>OCR Attempts</td><td>{int(images_payload.get('ocr_attempted_count', 0) or 0)}</td><td>{self._escape_html(str(images_payload.get('ocr_success_rate_pct', 'N/A')))}%</td></tr>"
                f"<tr><td>Vision Attempts</td><td>{int(images_payload.get('vision_attempted_count', 0) or 0)}</td><td>{self._escape_html(str(images_payload.get('vision_success_rate_pct', 'N/A')))}%</td></tr>"
                f"<tr><td>Fallback Used</td><td>{int(images_payload.get('fallback_used_count', 0) or 0)}</td><td>{self._escape_html(str(images_payload.get('fallback_usage_rate_pct', 'N/A')))}%</td></tr>"
                "</table>"
            )

        trend_sections = [
            self._build_multi_metric_trend_svg(
                title="Scraper Access Success % Trend",
                days=180,
                value_format="percent",
                series=[
                    {"metric_key": "scraper_scraper_access_success_rate_pct", "label": "scraper", "color": "#1f77b4"},
                    {"metric_key": "scraper_fb_access_success_rate_pct", "label": "fb", "color": "#ff7f0e"},
                    {"metric_key": "scraper_rd_ext_access_success_rate_pct", "label": "rd_ext", "color": "#2ca02c"},
                    {"metric_key": "scraper_ebs_access_success_rate_pct", "label": "ebs", "color": "#d62728"},
                    {"metric_key": "scraper_images_access_success_rate_pct", "label": "images", "color": "#9467bd"},
                ],
            ),
            self._build_multi_metric_trend_svg(
                title="Scraper Text Extraction % Trend",
                days=180,
                value_format="percent",
                series=[
                    {"metric_key": "scraper_scraper_text_extracted_rate_pct", "label": "scraper", "color": "#1f77b4"},
                    {"metric_key": "scraper_fb_text_extracted_rate_pct", "label": "fb", "color": "#ff7f0e"},
                    {"metric_key": "scraper_rd_ext_text_extracted_rate_pct", "label": "rd_ext", "color": "#2ca02c"},
                    {"metric_key": "scraper_ebs_text_extracted_rate_pct", "label": "ebs", "color": "#d62728"},
                    {"metric_key": "scraper_images_text_extracted_rate_pct", "label": "images", "color": "#9467bd"},
                ],
            ),
            self._build_multi_metric_trend_svg(
                title="Scraper Keyword Hit % Trend",
                days=180,
                value_format="percent",
                series=[
                    {"metric_key": "scraper_scraper_keyword_hit_rate_pct", "label": "scraper", "color": "#1f77b4"},
                    {"metric_key": "scraper_fb_keyword_hit_rate_pct", "label": "fb", "color": "#ff7f0e"},
                    {"metric_key": "scraper_rd_ext_keyword_hit_rate_pct", "label": "rd_ext", "color": "#2ca02c"},
                    {"metric_key": "scraper_ebs_keyword_hit_rate_pct", "label": "ebs", "color": "#d62728"},
                    {"metric_key": "scraper_images_keyword_hit_rate_pct", "label": "images", "color": "#9467bd"},
                ],
            ),
            self._build_multi_metric_trend_svg(
                title="Scraper Event Extraction Success % Trend",
                days=180,
                value_format="percent",
                series=[
                    {"metric_key": "scraper_scraper_extraction_success_rate_pct", "label": "scraper", "color": "#1f77b4"},
                    {"metric_key": "scraper_fb_extraction_success_rate_pct", "label": "fb", "color": "#ff7f0e"},
                    {"metric_key": "scraper_rd_ext_extraction_success_rate_pct", "label": "rd_ext", "color": "#2ca02c"},
                    {"metric_key": "scraper_ebs_extraction_success_rate_pct", "label": "ebs", "color": "#d62728"},
                    {"metric_key": "scraper_images_extraction_success_rate_pct", "label": "images", "color": "#9467bd"},
                ],
            ),
            self._build_multi_metric_trend_svg(
                title="Images OCR / Vision Success % Trend",
                days=180,
                value_format="percent",
                series=[
                    {"metric_key": "scraper_images_ocr_success_rate_pct", "label": "images OCR", "color": "#17becf"},
                    {"metric_key": "scraper_images_vision_success_rate_pct", "label": "images vision", "color": "#bcbd22"},
                ],
            ),
        ]

        return (
            "<div class='metric-container'>"
            f"<div class='metric'><div class='metric-value'>{len(steps)}</div><div class='metric-label'>Scrapers Reporting</div></div>"
            f"<div class='metric'><div class='metric-value'>{self._escape_html(str(scraper_telemetry_summary.get('run_id', '')))}</div><div class='metric-label'>Run ID</div></div>"
            "</div>"
            "<h3>Current Run Scraper Telemetry</h3>"
            "<table><tr><th>Scraper</th><th>Total URLs</th><th>Access %</th><th>Text %</th><th>Keyword %</th><th>URLs With Events %</th><th>Extraction Success %</th><th>Events Written</th></tr>"
            f"{''.join(current_rows)}"
            "</table>"
            f"{image_detail_html}"
            f"{''.join(trend_sections)}"
        )

    def _should_auto_delete_replay_mismatch(self, comparison: dict) -> bool:
        """
        High-confidence delete condition: baseline row appears unrelated to URL content.
        """
        if bool(comparison.get("is_match")):
            return False
        category = str(comparison.get("category") or "")
        if category not in {"wrong_date", "content_drift_major"}:
            return False
        baseline = comparison.get("baseline", {}) if isinstance(comparison.get("baseline"), dict) else {}
        replay = comparison.get("replay", {}) if isinstance(comparison.get("replay"), dict) else {}
        if not baseline or not replay:
            return False
        if self._is_calendar_event_url(str(baseline.get("url", ""))):
            return False
        name_similarity = self._name_similarity(
            str(baseline.get("event_name", "")),
            str(replay.get("event_name", "")),
        )
        same_date = str(baseline.get("start_date", "")) == str(replay.get("start_date", ""))
        return (name_similarity < 0.55) and (not same_date)

    def _run_accuracy_replay_assessment(self, results: dict) -> dict:
        replay_cfg = (
            self.config.get("testing", {})
            .get("validation", {})
            .get("accuracy_replay", {})
        )
        if not bool(replay_cfg.get("enabled", True)):
            return {"status": "DISABLED", "enabled": False}

        query_text = str(
            replay_cfg.get("query_text", "Where can I dance next week. Please include live music events.")
        ).strip()
        max_events = int(replay_cfg.get("max_events", 20) or 20)
        strict_time_match = bool(replay_cfg.get("strict_time_match", True))
        trend_days = int(replay_cfg.get("trend_days", 90) or 90)

        run_id = str(os.getenv("DS_RUN_ID", "")).strip() or self._infer_latest_pipeline_run_id(results.get("timestamp"))
        if not run_id:
            run_id = f"validation-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        executor = ChatbotTestExecutor(self.config, self.db_handler)
        question_dict = {
            "question": query_text,
            "category": "accuracy_replay",
            "parameters": {},
            "expected_criteria": {},
        }
        query_result = executor.execute_test_question(question_dict)
        sql_query = str(query_result.get("sql_query") or "").strip()
        if not bool(query_result.get("execution_success")) or not sql_query or sql_query.startswith("CLARIFICATION:"):
            return {
                "status": "ERROR",
                "query_text": query_text,
                "run_id": run_id,
                "error": "baseline query did not execute successfully",
                "sql_query": sql_query,
                "coverage_accuracy_pct": 0.0,
                "replay_accuracy_pct": 0.0,
                "total_rows": 0,
                "true_count": 0,
                "false_count": 0,
                "category_counts": {},
                "rows": [],
                "trend": self._load_metric_history("validation_replay_accuracy_pct", trend_days),
                "remediation_actions": [],
            }

        rows = self.db_handler.execute_query(sql_query) or []
        baseline_rows: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
            raw_url = str(payload.get("url", "") or "").strip()
            if is_email_like_input(raw_url):
                continue
            normalized_url = self._normalize_url_value(raw_url)
            if is_email_like_input(normalized_url):
                continue
            payload["url"] = normalized_url or raw_url
            baseline_rows.append(payload)
            if len(baseline_rows) >= max_events:
                break
        if not baseline_rows:
            return {
                "status": "WARN",
                "query_text": query_text,
                "run_id": run_id,
                "error": "baseline query returned no rows",
                "sql_query": sql_query,
                "coverage_accuracy_pct": 0.0,
                "replay_accuracy_pct": 0.0,
                "total_rows": 0,
                "true_count": 0,
                "false_count": 0,
                "category_counts": {},
                "rows": [],
                "trend": self._load_metric_history("validation_replay_accuracy_pct", trend_days),
                "remediation_actions": [],
            }

        url_replay_cache: dict[str, dict] = {}
        scored_rows: list[dict] = []
        category_counter: Counter = Counter()
        true_count = 0
        url_present_count = 0
        deleted_event_ids: set[int] = set()

        for baseline_row in baseline_rows:
            baseline_url = self._normalize_url_value(baseline_row.get("url"))
            if baseline_url:
                url_present_count += 1
                if baseline_url not in url_replay_cache:
                    url_replay_cache[baseline_url] = self._fetch_replay_events_for_url(baseline_url)
            replay_payload = url_replay_cache.get(baseline_url, {"ok": False, "category": "missing_url_baseline", "details": ""})
            comparison = self._compare_replay_row(
                baseline_row=baseline_row,
                replay_payload=replay_payload,
                strict_time_match=strict_time_match,
            )
            is_match = bool(comparison.get("is_match"))
            if is_match:
                true_count += 1
            category = str(comparison.get("category") or "match")
            if not is_match:
                category_counter.update([category])

            event_id = self._resolve_baseline_event_id(comparison.get("baseline") or {})
            action_taken = ""
            if (
                event_id is not None
                and event_id not in deleted_event_ids
                and self._should_auto_delete_replay_mismatch(comparison)
            ):
                try:
                    self.db_handler.delete_event_with_event_id(
                        int(event_id),
                        deletion_source="validation.replay_accuracy",
                        deletion_reason="replay_high_confidence_wrong_url_mismatch",
                        extra_context={
                            "query_text": query_text,
                            "mismatch_category": str(comparison.get("category", "")),
                            "baseline": comparison.get("baseline", {}),
                            "replay": comparison.get("replay", {}),
                        },
                    )
                    deleted_event_ids.add(int(event_id))
                    action_taken = "deleted"
                except Exception as delete_error:
                    action_taken = f"delete_failed:{delete_error}"
            self.db_handler.record_accuracy_replay_result(
                run_id=run_id,
                query_text=query_text,
                baseline_event_id=event_id,
                baseline_url=baseline_url,
                baseline_snapshot=comparison.get("baseline") or {},
                replay_snapshot=comparison.get("replay") or {},
                is_match=is_match,
                mismatch_category=category if not is_match else "",
                mismatch_details=str(comparison.get("details") or ""),
            )
            scored_rows.append(
                {
                    "is_match": is_match,
                    "mismatch_category": category if not is_match else "",
                    "mismatch_details": str(comparison.get("details") or ""),
                    "baseline": comparison.get("baseline") or {},
                    "replay": comparison.get("replay") or {},
                    "baseline_event_id": event_id,
                    "action_taken": action_taken,
                }
            )

        total_rows = len(scored_rows)
        false_count = max(0, total_rows - true_count)
        coverage_accuracy_pct = (true_count / total_rows * 100.0) if total_rows else 0.0
        replay_accuracy_pct = (true_count / url_present_count * 100.0) if url_present_count else 0.0

        remediation_actions: list[str] = []
        for category, _count in category_counter.most_common():
            if category == "missing_url_baseline":
                remediation_actions.append("Persist source URL for every non-email event row before DB write; enforce non-empty URL guard.")
            elif category == "wrong_date":
                remediation_actions.append("Tighten date extraction precedence and reject thumbnail/subsidiary dates when primary event block disagrees.")
            elif category == "wrong_time":
                remediation_actions.append("Enforce strict Pacific-time parsing in extraction and remove heuristic time fallbacks.")
            elif category == "wrong_location_or_address":
                remediation_actions.append("Harden venue/address resolver to prioritize explicit venue tokens over fuzzy nearest-name matches.")
            elif category == "wrong_source":
                remediation_actions.append("Improve source attribution from URL/domain and page ownership signals before write.")
            elif category == "duplicate_event_identity":
                remediation_actions.append("Strengthen dedup identity keys on URL+event_name+start_date+start_time before insert.")
            elif category == "no_event_extracted_replay":
                remediation_actions.append("Add site-template fallback extraction for low-text pages and enforce second-pass link follow for incomplete pages.")
            elif category == "url_unreachable_replay":
                remediation_actions.append("Add retry/backoff and classify persistent unreachable domains for manual review.")
            elif category == "social_platform_scraper_init_failed":
                remediation_actions.append("Repair fb.py replay initialization (credentials/session/bootstrap) so social URLs are replay-scraped.")
            else:
                remediation_actions.append("Review mismatch samples and add deterministic parser guardrails for recurring patterns.")

        self.db_handler.record_metric_observation(
            run_id=run_id,
            metric_key="validation_replay_accuracy_pct",
            metric_value_numeric=coverage_accuracy_pct,
            metric_unit="percent",
            description="Replay validation accuracy: strict row match / total baseline rows",
            higher_is_better=True,
            notes={
                "query_text": query_text,
                "total_rows": total_rows,
                "true_count": true_count,
                "false_count": false_count,
                "category_counts": dict(category_counter),
                "strict_time_match": strict_time_match,
                "auto_deleted_count": len(deleted_event_ids),
            },
        )
        self.db_handler.record_metric_observation(
            run_id=run_id,
            metric_key="validation_replay_accuracy_pct_url_present",
            metric_value_numeric=replay_accuracy_pct,
            metric_unit="percent",
            description="Replay validation accuracy: strict row match / baseline rows with URL",
            higher_is_better=True,
            notes={
                "query_text": query_text,
                "url_present_count": url_present_count,
                "true_count": true_count,
            },
        )

        return {
            "status": "PASS" if false_count == 0 else "WARN",
            "query_text": query_text,
            "sql_query": sql_query,
            "run_id": run_id,
            "strict_time_match": strict_time_match,
            "coverage_accuracy_pct": round(coverage_accuracy_pct, 2),
            "replay_accuracy_pct": round(replay_accuracy_pct, 2),
            "total_rows": total_rows,
            "url_present_count": url_present_count,
            "true_count": true_count,
            "false_count": false_count,
            "category_counts": dict(category_counter),
            "rows": scored_rows,
            "auto_deleted_count": len(deleted_event_ids),
            "remediation_actions": remediation_actions[:10],
            "trend": self._load_metric_history("validation_replay_accuracy_pct", trend_days),
        }

    def _build_classifier_training_queue_summary(self, accuracy_replay_summary: dict) -> dict:
        """
        Convert row-level replay results into one URL-level labeling candidate per normalized URL.
        """
        if not isinstance(accuracy_replay_summary, dict) or accuracy_replay_summary.get("error"):
            return {
                "status": "ERROR",
                "error": str((accuracy_replay_summary or {}).get("error", "accuracy replay unavailable")),
                "candidates": [],
            }

        replay_rows = accuracy_replay_summary.get("rows")
        if not isinstance(replay_rows, list):
            replay_rows = []

        queue_summary = build_classifier_training_queue(
            replay_rows,
            training_csv_path=self.training_csv_path,
            query_text=str(accuracy_replay_summary.get("query_text") or ""),
        )
        queue_summary["status"] = "OK"
        queue_summary["run_id"] = str(accuracy_replay_summary.get("run_id") or "").strip()

        run_id = queue_summary["run_id"]
        if run_id:
            for candidate in queue_summary.get("candidates", []):
                self.db_handler.record_classifier_training_url_candidate(
                    run_id=run_id,
                    candidate=candidate,
                )
            self.db_handler.record_metric_observation(
                run_id=run_id,
                metric_key="classifier_training_queue_total_urls",
                metric_value_numeric=float(queue_summary.get("total_urls", 0) or 0),
                metric_unit="count",
                description="URL-level replay candidates considered for classifier labeling",
                higher_is_better=False,
                notes={
                    "status_counts": queue_summary.get("status_counts") or {},
                    "training_ready_count": int(queue_summary.get("training_ready_count", 0) or 0),
                },
            )
            self.db_handler.record_metric_observation(
                run_id=run_id,
                metric_key="classifier_training_queue_training_ready_urls",
                metric_value_numeric=float(queue_summary.get("training_ready_count", 0) or 0),
                metric_unit="count",
                description="URL-level replay candidates eligible for positive classifier labeling review",
                higher_is_better=True,
                notes={
                    "review_required_count": int(queue_summary.get("review_required_count", 0) or 0),
                },
            )
        return queue_summary

    def _summarize_classifier_performance(self, run_id: str) -> dict:
        """
        Summarize classifier-stage usage and replay accuracy honestly at the URL level.
        """
        safe_run_id = str(run_id or "").strip()
        if not safe_run_id:
            return {"status": "ERROR", "error": "missing_run_id"}

        try:
            rows = self.db_handler.execute_query(
                """
                WITH latest_scrape AS (
                    SELECT DISTINCT ON (link)
                        link,
                        archetype,
                        classification_stage,
                        classification_confidence,
                        classification_owner_step,
                        classification_subtype,
                        classification_features_json,
                        time_stamp
                    FROM url_scrape_metrics
                    WHERE run_id = :run_id
                      AND step_name = 'scraper'
                      AND link IS NOT NULL
                      AND classification_stage IS NOT NULL
                    ORDER BY link, time_stamp DESC
                ),
                replay_rollup AS (
                    SELECT
                        baseline_url,
                        COUNT(*) AS replay_row_count,
                        SUM(CASE WHEN is_match THEN 1 ELSE 0 END) AS replay_true_count,
                        BOOL_AND(is_match) AS replay_url_success
                    FROM accuracy_replay_results
                    WHERE run_id = :run_id
                      AND baseline_url IS NOT NULL
                    GROUP BY baseline_url
                )
                SELECT
                    ls.link,
                    ls.archetype,
                    ls.classification_stage,
                    ls.classification_confidence,
                    ls.classification_owner_step,
                    ls.classification_subtype,
                    rr.replay_row_count,
                    rr.replay_true_count,
                    rr.replay_url_success
                FROM latest_scrape ls
                LEFT JOIN replay_rollup rr
                  ON rr.baseline_url = ls.link
                """,
                {"run_id": safe_run_id},
            ) or []
        except Exception as e:
            return {"status": "ERROR", "error": f"classifier_performance_query_failed:{e}"}

        stage_counts: Counter[str] = Counter()
        stage_confidences: dict[str, list[float]] = {}
        stage_replay_url_totals: Counter[str] = Counter()
        stage_replay_url_success: Counter[str] = Counter()
        stage_replay_row_totals: Counter[str] = Counter()
        stage_replay_row_success: Counter[str] = Counter()
        stage_domain_url_totals: dict[str, Counter[str]] = {}
        stage_domain_url_success: dict[str, Counter[str]] = {}
        archetype_counts: Counter[str] = Counter()
        owner_counts: Counter[str] = Counter()

        total_urls = 0
        replay_url_total = 0
        replay_url_success_total = 0
        replay_row_total = 0
        replay_row_success_total = 0

        for row in rows:
            mapping = row._mapping if hasattr(row, "_mapping") else row
            stage = str(mapping.get("classification_stage") or "").strip() or "unknown"
            archetype = str(mapping.get("archetype") or "").strip() or "unknown"
            owner = str(mapping.get("classification_owner_step") or "").strip() or "unknown"
            total_urls += 1
            stage_counts.update([stage])
            archetype_counts.update([archetype])
            owner_counts.update([owner])

            confidence = self._safe_float(mapping.get("classification_confidence"))
            if confidence is not None:
                stage_confidences.setdefault(stage, []).append(confidence)

            domain = (urlparse(str(mapping.get("link") or "")).netloc or "").lower() or "(no domain)"
            replay_row_count = int(mapping.get("replay_row_count") or 0)
            replay_true_count = int(mapping.get("replay_true_count") or 0)
            replay_url_success = bool(mapping.get("replay_url_success")) if replay_row_count > 0 else None
            if replay_row_count > 0:
                replay_url_total += 1
                replay_row_total += replay_row_count
                replay_row_success_total += replay_true_count
                stage_replay_url_totals.update([stage])
                stage_replay_row_totals[stage] += replay_row_count
                stage_replay_row_success[stage] += replay_true_count
                stage_domain_url_totals.setdefault(stage, Counter()).update([domain])
                if replay_url_success:
                    replay_url_success_total += 1
                    stage_replay_url_success.update([stage])
                    stage_domain_url_success.setdefault(stage, Counter()).update([domain])

        stage_details: list[dict[str, Any]] = []
        for stage, count in stage_counts.items():
            conf_values = stage_confidences.get(stage, [])
            replay_stage_url_total = int(stage_replay_url_totals.get(stage, 0))
            replay_stage_url_success = int(stage_replay_url_success.get(stage, 0))
            replay_stage_row_total = int(stage_replay_row_totals.get(stage, 0))
            replay_stage_row_success = int(stage_replay_row_success.get(stage, 0))
            stage_details.append(
                {
                    "stage": stage,
                    "url_count": int(count),
                    "usage_pct": round((count / total_urls) * 100.0, 2) if total_urls else 0.0,
                    "avg_confidence": round(sum(conf_values) / len(conf_values), 4) if conf_values else None,
                    "replay_url_count": replay_stage_url_total,
                    "replay_url_success_count": replay_stage_url_success,
                    "replay_url_accuracy_pct": (
                        round((replay_stage_url_success / replay_stage_url_total) * 100.0, 2)
                        if replay_stage_url_total
                        else None
                    ),
                    "replay_row_count": replay_stage_row_total,
                    "replay_row_success_count": replay_stage_row_success,
                    "replay_row_accuracy_pct": (
                        round((replay_stage_row_success / replay_stage_row_total) * 100.0, 2)
                        if replay_stage_row_total
                        else None
                    ),
                }
            )
        stage_details.sort(key=lambda item: (-int(item["url_count"]), str(item["stage"])))
        stage_domain_details: list[dict[str, Any]] = []
        for stage, domain_totals in stage_domain_url_totals.items():
            domain_success = stage_domain_url_success.get(stage, Counter())
            for domain, replay_stage_domain_total in domain_totals.items():
                replay_stage_domain_success = int(domain_success.get(domain, 0))
                stage_domain_details.append(
                    {
                        "stage": stage,
                        "domain": domain,
                        "replay_url_count": int(replay_stage_domain_total),
                        "replay_url_success_count": replay_stage_domain_success,
                        "replay_url_accuracy_pct": (
                            round((replay_stage_domain_success / replay_stage_domain_total) * 100.0, 2)
                            if replay_stage_domain_total
                            else None
                        ),
                    }
                )
        stage_domain_details.sort(
            key=lambda item: (
                str(item.get("stage") or ""),
                float(item.get("replay_url_accuracy_pct") if item.get("replay_url_accuracy_pct") is not None else -1.0),
                -int(item.get("replay_url_count") or 0),
                str(item.get("domain") or ""),
            )
        )

        summary = {
            "status": "OK",
            "run_id": safe_run_id,
            "total_classified_urls": total_urls,
            "stage_counts": dict(stage_counts),
            "archetype_counts": dict(archetype_counts),
            "owner_step_counts": dict(owner_counts),
            "ml_usage_count": int(stage_counts.get("ml", 0)),
            "ml_usage_pct": round((stage_counts.get("ml", 0) / total_urls) * 100.0, 2) if total_urls else 0.0,
            "replay_url_total": replay_url_total,
            "replay_url_success_total": replay_url_success_total,
            "replay_url_accuracy_pct": round((replay_url_success_total / replay_url_total) * 100.0, 2)
            if replay_url_total
            else None,
            "replay_row_total": replay_row_total,
            "replay_row_success_total": replay_row_success_total,
            "replay_row_accuracy_pct": round((replay_row_success_total / replay_row_total) * 100.0, 2)
            if replay_row_total
            else None,
            "stage_details": stage_details,
            "stage_domain_details": stage_domain_details,
        }

        try:
            self.db_handler.record_metric_observation(
                run_id=safe_run_id,
                metric_key="classifier_ml_usage_pct",
                metric_value_numeric=float(summary["ml_usage_pct"] or 0.0),
                metric_unit="percent",
                description="Share of classified URLs handled by ML stage",
                higher_is_better=False,
                notes={"stage_counts": summary["stage_counts"]},
            )
            ml_stage = next((item for item in stage_details if item["stage"] == "ml"), None)
            if ml_stage and ml_stage.get("replay_url_accuracy_pct") is not None:
                self.db_handler.record_metric_observation(
                    run_id=safe_run_id,
                    metric_key="classifier_ml_replay_url_accuracy_pct",
                    metric_value_numeric=float(ml_stage["replay_url_accuracy_pct"]),
                    metric_unit="percent",
                    description="Replay URL accuracy for ML-classified URLs",
                    higher_is_better=True,
                    notes={"replay_url_count": int(ml_stage.get("replay_url_count", 0) or 0)},
                )
            rule_stage = next((item for item in stage_details if item["stage"] == "rule"), None)
            if rule_stage and rule_stage.get("replay_url_accuracy_pct") is not None:
                self.db_handler.record_metric_observation(
                    run_id=safe_run_id,
                    metric_key="classifier_rule_replay_url_accuracy_pct",
                    metric_value_numeric=float(rule_stage["replay_url_accuracy_pct"]),
                    metric_unit="percent",
                    description="Replay URL accuracy for rule-classified URLs",
                    higher_is_better=True,
                    notes={"replay_url_count": int(rule_stage.get("replay_url_count", 0) or 0)},
                )
        except Exception:
            pass
        return summary

    def _evaluate_chatbot_problem_category_gate(self, chatbot_report: dict, chatbot_config: dict) -> dict:
        """Evaluate category-level chatbot quality gate from problem-category buckets."""
        categories = chatbot_report.get("problem_categories", []) if isinstance(chatbot_report, dict) else []
        min_count = int(chatbot_config.get("problem_category_min_count", 2) or 2)
        max_allowed = int(chatbot_config.get("max_problem_categories", 0) or 0)
        mode = str(chatbot_config.get("problem_category_gate_mode", "warning") or "warning").lower()
        relevant = [c for c in categories if int((c or {}).get("count", 0) or 0) >= min_count]
        count = len(relevant)
        detail = (
            f"relevant_categories={count} (min_count={min_count}) max_allowed={max_allowed}; "
            f"categories={', '.join(str(c.get('name', '')) for c in relevant[:5]) or 'none'}"
        )
        if count > max_allowed:
            status = "FAIL" if mode == "fail" else "WARNING"
        else:
            status = "PASS"
        return {
            "status": status,
            "mode": mode,
            "max_allowed": max_allowed,
            "min_count": min_count,
            "relevant_count": count,
            "detail": detail,
        }

    def _write_chatbot_problem_category_regressions(self, output_dir: str, chatbot_report: dict) -> None:
        """Persist deterministic regression cases from chatbot problem categories."""
        os.makedirs(output_dir, exist_ok=True)
        cases = chatbot_report.get("problem_category_regression_cases", []) if isinstance(chatbot_report, dict) else []
        payload = {
            "generated_at": datetime.now().isoformat(),
            "count": len(cases),
            "cases": cases,
        }
        path = chatbot_path("chatbot_problem_category_regressions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Chatbot problem-category regression cases saved: %s", path)

    def _generate_html_report(self, results: dict) -> None:
        """
        Generate HTML report combining both validations.

        Args:
            results (dict): Combined validation results
        """
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')
        output_path = reports_path('comprehensive_test_report.html')
        alias_audit_summary = self._summarize_address_alias_audit()
        llm_activity_summary = self._summarize_llm_provider_activity(results.get('timestamp'))
        llm_extraction_quality = self._summarize_llm_extraction_quality(results.get('timestamp'))
        chatbot_performance_summary = self._summarize_chatbot_performance(results.get('timestamp'))
        chatbot_metrics_sync_summary = self._summarize_chatbot_metrics_sync(results.get('timestamp'))
        accuracy_replay_summary = results.get("accuracy_replay") if isinstance(results, dict) else {}
        classifier_training_queue_summary = results.get("classifier_training_queue") if isinstance(results, dict) else {}
        classifier_performance_summary = results.get("classifier_performance") if isinstance(results, dict) else {}
        scraper_network_summary = self._summarize_scraper_network_health(results.get('timestamp'))
        fb_block_summary = self._summarize_fb_block_health(results.get('timestamp'))
        fb_ig_funnel_summary = self._summarize_fb_ig_url_funnel(results.get('timestamp'), fb_block_summary)
        suspicious_deletes_summary = self._summarize_suspicious_deletes(results.get('timestamp'))
        report_run_id = str(fb_block_summary.get("run_id", "") or "").strip() or self._infer_latest_pipeline_run_id(results.get('timestamp'))
        pipeline_runtime_summary = self._summarize_pipeline_runtime(results.get('timestamp'), report_run_id)
        scraper_telemetry_summary = self._summarize_scraper_step_telemetry(report_run_id)
        telemetry_integrity_summary = self._build_phase1_telemetry_integrity_summary(report_run_id)
        openrouter_cost_summary = self._summarize_openrouter_run_cost(
            report_timestamp=results.get('timestamp'),
            pipeline_runtime=pipeline_runtime_summary,
            output_dir=output_dir,
        )
        openai_cost_summary = self._summarize_openai_run_cost(
            report_timestamp=results.get('timestamp'),
            pipeline_runtime=pipeline_runtime_summary,
            output_dir=output_dir,
        )
        runtime_summary = self._build_phase1_runtime_summary(pipeline_runtime_summary)
        llm_cost_summary = self._build_phase1_llm_cost_summary(
            openrouter_cost=openrouter_cost_summary,
            openai_cost=openai_cost_summary,
            accuracy_replay=accuracy_replay_summary,
            llm_activity_summary=llm_activity_summary,
        )
        chatbot_quality_summary = self._build_phase1_chatbot_quality_summary(
            chatbot_performance=chatbot_performance_summary,
            chatbot_testing=results.get('chatbot_testing'),
        )
        duplicate_audit_summary = self._summarize_duplicate_audit()
        event_data_quality_summary = self._summarize_event_data_quality()
        field_accuracy_summary = self._summarize_field_accuracy(accuracy_replay_summary)
        coverage_summary = self._summarize_coverage_watchlist()
        dev_summary = self._summarize_dev_replay(accuracy_replay_summary)
        holdout_summary = self._summarize_holdout_replay(accuracy_replay_summary)
        domain_capped_summary = self._summarize_domain_capped_replay(accuracy_replay_summary)
        domain_evaluation_summary = self._summarize_domain_evaluation(accuracy_replay_summary)
        run_scorecard = self._build_phase1_run_scorecard(
            run_id=report_run_id,
            report_timestamp=results.get('timestamp'),
            accuracy_replay=accuracy_replay_summary,
            scraping_results=results.get('scraping_validation'),
            runtime_summary=runtime_summary,
            llm_cost_summary=llm_cost_summary,
            chatbot_quality_summary=chatbot_quality_summary,
            classifier_performance_summary=classifier_performance_summary,
            duplicate_summary=duplicate_audit_summary,
            event_data_quality_summary=event_data_quality_summary,
            field_accuracy_summary=field_accuracy_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            holdout_summary=holdout_summary,
            domain_capped_summary=domain_capped_summary,
            telemetry_integrity_summary=telemetry_integrity_summary,
        )
        run_delta_summary = self._build_run_delta_summary(run_scorecard)
        run_scorecard["comparison_summary"] = run_delta_summary
        previous_run_delta = ((run_delta_summary.get("previous_run") or {}).get("summary") or {}).get("delta_overall_score")
        holdout_baseline_delta = ((run_delta_summary.get("holdout_baseline") or {}).get("summary") or {}).get("delta_holdout_replay_url_accuracy_pct")
        run_scorecard["overall_score"]["vs_previous_run"] = previous_run_delta
        run_scorecard["overall_score"]["vs_holdout_baseline"] = holdout_baseline_delta
        run_scorecard["guardrails"] = self._evaluate_phase3_guardrails(run_scorecard)
        run_scorecard["recommendations_input"] = self._build_recommendations_input(
            run_scorecard=run_scorecard,
            duplicate_summary=duplicate_audit_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            classifier_performance_summary=classifier_performance_summary,
            runtime_summary=runtime_summary,
            llm_cost_summary=llm_cost_summary,
            domain_evaluation_summary=domain_evaluation_summary,
            run_delta_summary=run_delta_summary,
        )
        recommendation_plan = self._build_recommendation_plan(
            run_scorecard=run_scorecard,
            classifier_performance_summary=classifier_performance_summary,
            duplicate_summary=duplicate_audit_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            runtime_summary=runtime_summary,
            llm_cost_summary=llm_cost_summary,
            domain_evaluation_summary=domain_evaluation_summary,
            run_delta_summary=run_delta_summary,
        )
        codex_review_bundle = self._build_codex_review_bundle(
            run_scorecard=run_scorecard,
            accuracy_replay_summary=accuracy_replay_summary,
            classifier_performance_summary=classifier_performance_summary,
            duplicate_summary=duplicate_audit_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            holdout_summary=holdout_summary,
            domain_capped_summary=domain_capped_summary,
            domain_evaluation_summary=domain_evaluation_summary,
            run_delta_summary=run_delta_summary,
            recommendation_plan=recommendation_plan,
        )
        if run_scorecard["guardrails"]["status"] == "PASS" and run_scorecard.get("overall_score", {}).get("value") is not None:
            run_scorecard["overall_score"]["status"] = "PASS"
        else:
            run_scorecard["overall_score"]["status"] = "FAIL"
        self._persist_phase1_scorecard_metrics(
            run_id=report_run_id,
            runtime_summary=runtime_summary,
            llm_cost_summary=llm_cost_summary,
            chatbot_quality_summary=chatbot_quality_summary,
            duplicate_summary=duplicate_audit_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            holdout_summary=holdout_summary,
            domain_capped_summary=domain_capped_summary,
            event_data_quality_summary=event_data_quality_summary,
            field_accuracy_summary=field_accuracy_summary,
            run_delta_summary=run_delta_summary,
        )
        self._persist_scraper_step_telemetry_metrics(
            run_id=report_run_id,
            scraper_telemetry_summary=scraper_telemetry_summary,
            runtime_summary=runtime_summary,
        )
        self._persist_validation_artifacts(
            run_id=report_run_id,
            artifacts={
                "run_scorecard": run_scorecard,
                "runtime_summary": runtime_summary,
                "llm_cost_summary": llm_cost_summary,
                "chatbot_quality_summary": chatbot_quality_summary,
                "duplicate_audit_summary": duplicate_audit_summary,
                "event_data_quality_summary": event_data_quality_summary,
                "field_accuracy_summary": field_accuracy_summary,
                "coverage_summary": coverage_summary,
                "dev_summary": dev_summary,
                "holdout_summary": holdout_summary,
                "domain_capped_summary": domain_capped_summary,
                "domain_evaluation_summary": domain_evaluation_summary,
                "scraper_telemetry_summary": scraper_telemetry_summary,
                "telemetry_integrity_summary": telemetry_integrity_summary,
                "run_delta_summary": run_delta_summary,
                "recommendation_plan": recommendation_plan,
                "codex_review_bundle": codex_review_bundle,
            },
        )
        reliability_scorecard = self._summarize_reliability_scorecard(
            results, llm_activity_summary, alias_audit_summary, scraper_network_summary, fb_block_summary
        )
        reliability_issues = self._extract_reliability_issues(
            results, llm_activity_summary, alias_audit_summary, scraper_network_summary, fb_block_summary
        )
        reliability_gates = results.get('reliability_gates') or self._evaluate_reliability_gates(reliability_scorecard)
        optimization_plan = self._build_optimization_plan(results, reliability_scorecard, llm_activity_summary)
        trend_summary = self._update_and_summarize_reliability_history(output_dir, reliability_scorecard)
        reliability_issues, registry_summary = self._update_reliability_issue_registry(output_dir, reliability_issues)
        action_queue = self._build_action_queue(reliability_gates, optimization_plan, reliability_issues, registry_summary)
        control_panel_summary = self._summarize_run_control_panel(
            output_dir=output_dir,
            run_id=report_run_id,
            reliability_scorecard=reliability_scorecard,
            llm_activity=llm_activity_summary,
            llm_quality=llm_extraction_quality,
            scraping_results=results.get('scraping_validation'),
            fb_ig_funnel=fb_ig_funnel_summary,
            scraper_network=scraper_network_summary,
            pipeline_runtime=pipeline_runtime_summary,
            chatbot_performance=chatbot_performance_summary,
            openrouter_cost=openrouter_cost_summary,
            openai_cost=openai_cost_summary,
            accuracy_replay=accuracy_replay_summary,
        )

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pre-Commit Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 30px; }}
        h3 {{ color: #777; margin-top: 20px; }}
        .status-pass {{ color: #28a745; font-weight: bold; font-size: 1.3em; }}
        .status-warning {{ color: #ffc107; font-weight: bold; font-size: 1.3em; }}
        .status-fail {{ color: #dc3545; font-weight: bold; font-size: 1.3em; }}
        .metric-container {{ display: flex; flex-wrap: wrap; margin: 20px 0; }}
        .metric {{ flex: 0 0 200px; margin: 10px 20px 10px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #007bff; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .problematic {{ background-color: #ffe6e6; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
        .error-box {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pre-Commit Validation Report</h1>
        <p class="timestamp"><strong>Timestamp:</strong> {results['timestamp']}</p>
        <p><strong>Overall Status:</strong>
            <span class="status-{results['overall_status'].lower()}">{results['overall_status']}</span>
        </p>

        <h2>0. Trend Dashboard</h2>
        {self._build_top_trend_dashboard_html()}

        <h2>1. Scraper Telemetry Trends</h2>
        {self._build_scraper_telemetry_html(scraper_telemetry_summary)}

        <h2>2. Replay Accuracy Row-by-Row</h2>
        {self._build_accuracy_replay_rows_html(accuracy_replay_summary)}

        <h2>3. Completeness KPIs</h2>
        {self._build_completeness_kpis_only_html(control_panel_summary)}

        <h2>4. KPI Scorecard</h2>
        {self._build_phase1_scorecard_html(run_scorecard)}

        <h2>5. Phase 2 Integrity Coverage</h2>
        {self._build_phase2_integrity_html(duplicate_audit_summary, coverage_summary)}

        <h2>6. Phase 3 Honest Evaluation</h2>
        {self._build_phase3_honest_evaluation_html(dev_summary, holdout_summary, domain_capped_summary, run_scorecard.get('guardrails', {}))}

        <h2>7. Domain Evaluation</h2>
        {self._build_domain_evaluation_html(domain_evaluation_summary)}

        <h2>8. Run Comparison</h2>
        {self._build_run_delta_html(run_delta_summary)}

        <h2>9. Recommendation Plan</h2>
        {self._build_recommendation_plan_html(recommendation_plan)}

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #999; font-size: 0.9em;">
            Generated by Claude Code Validation Framework v1.0.0
        </p>
    </div>
</body>
</html>
"""

        # Write HTML file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        scorecard_path = codex_review_path('reliability_scorecard.json')
        issues_path = codex_review_path('reliability_issues.json')
        with open(scorecard_path, 'w', encoding='utf-8') as f:
            json.dump(reliability_scorecard, f, indent=2)
        with open(issues_path, 'w', encoding='utf-8') as f:
            json.dump({"issues": reliability_issues}, f, indent=2)
        gates_path = codex_review_path('reliability_gates.json')
        with open(gates_path, 'w', encoding='utf-8') as f:
            json.dump(reliability_gates, f, indent=2)
        optimization_path = codex_review_path('reliability_optimization.json')
        with open(optimization_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_plan, f, indent=2)
        action_queue_path = codex_review_path('reliability_action_queue.json')
        with open(action_queue_path, 'w', encoding='utf-8') as f:
            json.dump(action_queue, f, indent=2)
        llm_extraction_quality_path = codex_review_path('llm_extraction_quality.json')
        with open(llm_extraction_quality_path, 'w', encoding='utf-8') as f:
            json.dump(llm_extraction_quality, f, indent=2)
        chatbot_performance_path = chatbot_path('chatbot_performance.json')
        with open(chatbot_performance_path, 'w', encoding='utf-8') as f:
            json.dump(chatbot_performance_summary, f, indent=2)
        suspicious_deletes_path = codex_review_path('suspicious_deletes.json')
        with open(suspicious_deletes_path, 'w', encoding='utf-8') as f:
            json.dump(suspicious_deletes_summary, f, indent=2)
        chatbot_metrics_sync_path = chatbot_path('chatbot_metrics_sync_summary.json')
        with open(chatbot_metrics_sync_path, 'w', encoding='utf-8') as f:
            json.dump(chatbot_metrics_sync_summary, f, indent=2)
        control_panel_path = codex_review_path('run_control_panel.json')
        with open(control_panel_path, 'w', encoding='utf-8') as f:
            json.dump(control_panel_summary, f, indent=2)
        runtime_summary_path = codex_review_path('runtime_summary.json')
        with open(runtime_summary_path, 'w', encoding='utf-8') as f:
            json.dump(runtime_summary, f, indent=2)
        llm_cost_summary_path = codex_review_path('llm_cost_summary.json')
        with open(llm_cost_summary_path, 'w', encoding='utf-8') as f:
            json.dump(llm_cost_summary, f, indent=2)
        chatbot_quality_summary_path = chatbot_path('chatbot_quality_summary.json')
        with open(chatbot_quality_summary_path, 'w', encoding='utf-8') as f:
            json.dump(chatbot_quality_summary, f, indent=2)
        duplicate_audit_summary_path = duplicates_path('duplicate_audit_summary.json')
        with open(duplicate_audit_summary_path, 'w', encoding='utf-8') as f:
            json.dump(duplicate_audit_summary, f, indent=2)
        event_data_quality_summary_path = codex_review_path('event_data_quality_summary.json')
        with open(event_data_quality_summary_path, 'w', encoding='utf-8') as f:
            json.dump(event_data_quality_summary, f, indent=2)
        field_accuracy_summary_path = codex_review_path('field_accuracy_summary.json')
        with open(field_accuracy_summary_path, 'w', encoding='utf-8') as f:
            json.dump(field_accuracy_summary, f, indent=2)
        coverage_summary_path = codex_review_path('coverage_summary.json')
        with open(coverage_summary_path, 'w', encoding='utf-8') as f:
            json.dump(coverage_summary, f, indent=2)
        dev_summary_path = codex_review_path('dev_summary.json')
        with open(dev_summary_path, 'w', encoding='utf-8') as f:
            json.dump(dev_summary, f, indent=2)
        holdout_summary_path = codex_review_path('holdout_summary.json')
        with open(holdout_summary_path, 'w', encoding='utf-8') as f:
            json.dump(holdout_summary, f, indent=2)
        domain_capped_summary_path = codex_review_path('domain_capped_summary.json')
        with open(domain_capped_summary_path, 'w', encoding='utf-8') as f:
            json.dump(domain_capped_summary, f, indent=2)
        domain_evaluation_summary_path = codex_review_path('domain_evaluation_summary.json')
        with open(domain_evaluation_summary_path, 'w', encoding='utf-8') as f:
            json.dump(domain_evaluation_summary, f, indent=2)
        run_delta_summary_path = codex_review_path('run_delta_summary.json')
        with open(run_delta_summary_path, 'w', encoding='utf-8') as f:
            json.dump(run_delta_summary, f, indent=2)
        recommendation_plan_path = codex_review_path('recommendation_plan.json')
        with open(recommendation_plan_path, 'w', encoding='utf-8') as f:
            json.dump(recommendation_plan, f, indent=2)
        run_scorecard_path = codex_review_path('run_scorecard.json')
        with open(run_scorecard_path, 'w', encoding='utf-8') as f:
            json.dump(run_scorecard, f, indent=2)
        codex_review_bundle_path = codex_review_path('codex_review_bundle.json')
        with open(codex_review_bundle_path, 'w', encoding='utf-8') as f:
            json.dump(codex_review_bundle, f, indent=2)
        accuracy_replay_path = replay_path('accuracy_replay_summary.json')
        with open(accuracy_replay_path, 'w', encoding='utf-8') as f:
            json.dump(accuracy_replay_summary, f, indent=2)
        classifier_training_queue_path = classifier_path('classifier_training_queue.json')
        with open(classifier_training_queue_path, 'w', encoding='utf-8') as f:
            json.dump(classifier_training_queue_summary, f, indent=2)
        classifier_training_queue_csv_path = classifier_path('classifier_training_queue.csv')
        self._write_classifier_training_queue_csv(
            classifier_training_queue_csv_path,
            classifier_training_queue_summary,
        )
        classifier_review_queue_summary = filter_classifier_review_candidates(classifier_training_queue_summary)
        classifier_review_queue_path = classifier_path('classifier_review_queue.json')
        with open(classifier_review_queue_path, 'w', encoding='utf-8') as f:
            json.dump(classifier_review_queue_summary, f, indent=2)
        classifier_review_queue_csv_path = classifier_path('classifier_review_queue.csv')
        self._write_classifier_training_queue_csv(
            classifier_review_queue_csv_path,
            classifier_review_queue_summary,
        )
        classifier_performance_path = classifier_path('classifier_performance_summary.json')
        with open(classifier_performance_path, 'w', encoding='utf-8') as f:
            json.dump(classifier_performance_summary, f, indent=2)
        openrouter_cost_path = codex_review_path('openrouter_run_cost.json')
        with open(openrouter_cost_path, 'w', encoding='utf-8') as f:
            json.dump(openrouter_cost_summary, f, indent=2)
        openai_cost_path = codex_review_path('openai_run_cost.json')
        with open(openai_cost_path, 'w', encoding='utf-8') as f:
            json.dump(openai_cost_summary, f, indent=2)

        logging.info(f"HTML report saved: {output_path}")
        logging.info(f"Reliability scorecard saved: {scorecard_path}")
        logging.info(f"Reliability issues saved: {issues_path}")
        logging.info(f"Reliability gates saved: {gates_path}")
        logging.info(f"Reliability optimization saved: {optimization_path}")
        logging.info(f"Reliability action queue saved: {action_queue_path}")
        logging.info(f"LLM extraction quality saved: {llm_extraction_quality_path}")
        logging.info(f"Chatbot performance summary saved: {chatbot_performance_path}")
        logging.info(f"Suspicious deletes summary saved: {suspicious_deletes_path}")
        logging.info(f"Chatbot metrics sync summary saved: {chatbot_metrics_sync_path}")
        logging.info(f"Run control panel saved: {control_panel_path}")
        logging.info(f"Runtime summary saved: {runtime_summary_path}")
        logging.info(f"LLM cost summary saved: {llm_cost_summary_path}")
        logging.info(f"Chatbot quality summary saved: {chatbot_quality_summary_path}")
        logging.info(f"Duplicate audit summary saved: {duplicate_audit_summary_path}")
        logging.info(f"Event data quality summary saved: {event_data_quality_summary_path}")
        logging.info(f"Field accuracy summary saved: {field_accuracy_summary_path}")
        logging.info(f"Coverage summary saved: {coverage_summary_path}")
        logging.info(f"Dev summary saved: {dev_summary_path}")
        logging.info(f"Holdout summary saved: {holdout_summary_path}")
        logging.info(f"Domain-capped summary saved: {domain_capped_summary_path}")
        logging.info(f"Domain evaluation summary saved: {domain_evaluation_summary_path}")
        logging.info(f"Run delta summary saved: {run_delta_summary_path}")
        logging.info(f"Recommendation plan saved: {recommendation_plan_path}")
        logging.info(f"Run scorecard saved: {run_scorecard_path}")
        logging.info(f"Codex review bundle saved: {codex_review_bundle_path}")
        logging.info(f"Accuracy replay summary saved: {accuracy_replay_path}")
        logging.info(f"Classifier training queue saved: {classifier_training_queue_path}")
        logging.info(f"Classifier training queue CSV saved: {classifier_training_queue_csv_path}")
        logging.info(f"Classifier review queue saved: {classifier_review_queue_path}")
        logging.info(f"Classifier review queue CSV saved: {classifier_review_queue_csv_path}")
        logging.info(f"Classifier performance summary saved: {classifier_performance_path}")
        logging.info(f"OpenRouter run cost summary saved: {openrouter_cost_path}")
        logging.info(f"OpenAI run cost summary saved: {openai_cost_path}")

    def _write_classifier_training_queue_csv(self, path: str, queue_summary: dict) -> None:
        """
        Write URL-level classifier labeling candidates as a flat CSV for review.
        """
        candidates = queue_summary.get("candidates", []) if isinstance(queue_summary, dict) else []
        fieldnames = [
            "normalized_url",
            "domain",
            "query_text",
            "total_rows",
            "true_count",
            "false_count",
            "match_rate_pct",
            "status",
            "recommended_action",
            "training_eligible",
            "recommended_archetype",
            "recommended_owner_step",
            "recommended_subtype",
            "priority_score",
            "mismatch_category_counts",
            "baseline_event_ids",
            "sample_event_name",
            "sample_start_date",
            "sample_start_time",
            "sample_source",
            "sample_location",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates:
                baseline = candidate.get("sample_baseline", {}) if isinstance(candidate.get("sample_baseline"), dict) else {}
                writer.writerow(
                    {
                        "normalized_url": candidate.get("normalized_url", ""),
                        "domain": candidate.get("domain", ""),
                        "query_text": candidate.get("query_text", ""),
                        "total_rows": candidate.get("total_rows", 0),
                        "true_count": candidate.get("true_count", 0),
                        "false_count": candidate.get("false_count", 0),
                        "match_rate_pct": candidate.get("match_rate_pct", 0.0),
                        "status": candidate.get("status", ""),
                        "recommended_action": candidate.get("recommended_action", ""),
                        "training_eligible": candidate.get("training_eligible", False),
                        "recommended_archetype": candidate.get("recommended_archetype", ""),
                        "recommended_owner_step": candidate.get("recommended_owner_step", ""),
                        "recommended_subtype": candidate.get("recommended_subtype", ""),
                        "priority_score": candidate.get("priority_score", 0),
                        "mismatch_category_counts": json.dumps(
                            candidate.get("mismatch_category_counts") or {},
                            ensure_ascii=True,
                        ),
                        "baseline_event_ids": json.dumps(candidate.get("baseline_event_ids") or [], ensure_ascii=True),
                        "sample_event_name": baseline.get("event_name", ""),
                        "sample_start_date": baseline.get("start_date", ""),
                        "sample_start_time": baseline.get("start_time", ""),
                        "sample_source": baseline.get("source", ""),
                        "sample_location": baseline.get("location", ""),
                    }
                )

    @staticmethod
    def _escape_html(value: str) -> str:
        """Escape untrusted text for safe HTML rendering."""
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _summarize_address_alias_audit(self) -> dict:
        """
        Summarize address alias audit CSV for report display.
        Missing/invalid files are reported as non-fatal metadata.
        """
        output_cfg = self.config.get('output', {})
        csv_path = output_cfg.get('address_alias_audit', duplicates_path('address_alias_hits.csv'))
        days_window = int(
            self.validation_config.get('reporting', {}).get('address_alias_audit_days', 14) or 14
        )

        summary = {
            'available': False,
            'path': csv_path,
            'days_window': days_window,
            'rows_analyzed': 0,
            'decision_counts': {},
            'rule_counts': [],
            'rule_decision_counts': [],
            'daily_counts': [],
            'top_candidates': [],
            'error': '',
        }

        if not os.path.exists(csv_path):
            summary['error'] = f"Audit file not found: {csv_path}"
            return summary

        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as file:
                rows = [{k: (v or '').strip() for k, v in row.items()} for row in csv.DictReader(file)]
        except Exception as e:
            summary['error'] = f"Failed reading audit file: {e}"
            return summary

        if not rows:
            summary['error'] = "Audit file is empty."
            return summary

        def extract_day(ts: str) -> str:
            ts = (ts or '').strip()
            if not ts:
                return 'unknown'
            try:
                return datetime.fromisoformat(ts).date().isoformat()
            except ValueError:
                return ts[:10] if len(ts) >= 10 else 'unknown'

        if days_window > 0:
            valid_days = sorted({extract_day(r.get('timestamp', '')) for r in rows if extract_day(r.get('timestamp', '')) != 'unknown'})
            keep_days = set(valid_days[-days_window:]) if valid_days else set()
            rows = [r for r in rows if not keep_days or extract_day(r.get('timestamp', '')) in keep_days]

        decision_counts: Counter = Counter()
        rule_counts: Counter = Counter()
        rule_decision_counts: Counter = Counter()
        daily_counts: Counter = Counter()
        candidate_counts: Counter = Counter()

        for row in rows:
            day = extract_day(row.get('timestamp', ''))
            decision = row.get('decision', '') or '(blank)'
            rule_name = row.get('rule_name', '') or '(blank)'
            candidate = row.get('candidate', '') or '(blank)'

            decision_counts.update([decision])
            rule_counts.update([rule_name])
            rule_decision_counts.update([f"{rule_name} | {decision}"])
            daily_counts.update([day])
            candidate_counts.update([candidate])

        summary['available'] = True
        summary['rows_analyzed'] = len(rows)
        summary['decision_counts'] = dict(decision_counts)
        summary['rule_counts'] = rule_counts.most_common(8)
        summary['rule_decision_counts'] = rule_decision_counts.most_common(10)
        summary['daily_counts'] = daily_counts.most_common(14)
        summary['top_candidates'] = candidate_counts.most_common(8)
        return summary

    def _build_address_alias_audit_html(self, alias_data: dict) -> str:
        """Build HTML for address alias audit section."""
        if not alias_data:
            return "<p class='error-box'>❌ Address alias audit summary unavailable</p>"

        if not alias_data.get('available'):
            err = self._escape_html(alias_data.get('error', 'Address alias audit unavailable'))
            path = self._escape_html(alias_data.get('path', ''))
            return (
                "<p class='error-box'>"
                f"⚠ Address alias audit not available.<br>{err}<br><strong>Path:</strong> {path}"
                "</p>"
            )

        decision_counts = alias_data.get('decision_counts', {})
        applied_count = int(decision_counts.get('applied', 0))
        conflict_skips = int(decision_counts.get('skipped_conflict', 0))
        rows_analyzed = int(alias_data.get('rows_analyzed', 0))
        unique_rules = len(alias_data.get('rule_counts', []))

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{rows_analyzed}</div>
                <div class="metric-label">Rows Analyzed (Recent Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{applied_count}</div>
                <div class="metric-label">Alias Matches Applied</div>
            </div>
            <div class="metric">
                <div class="metric-value">{conflict_skips}</div>
                <div class="metric-label">Conflict Skips</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unique_rules}</div>
                <div class="metric-label">Active Rules (Top List)</div>
            </div>
        </div>
        """

        if alias_data.get('daily_counts'):
            html += "<h3>Daily Volume</h3><table><tr><th>Day</th><th>Count</th></tr>"
            for day, count in alias_data['daily_counts']:
                html += f"<tr><td>{self._escape_html(day)}</td><td>{count}</td></tr>"
            html += "</table>"

        if alias_data.get('rule_decision_counts'):
            html += "<h3>Top Rule + Decision Pairs</h3><table><tr><th>Rule | Decision</th><th>Count</th></tr>"
            for key, count in alias_data['rule_decision_counts']:
                html += f"<tr><td>{self._escape_html(key)}</td><td>{count}</td></tr>"
            html += "</table>"

        if alias_data.get('top_candidates'):
            html += "<h3>Top Candidate Text</h3><table><tr><th>Candidate</th><th>Count</th></tr>"
            for candidate, count in alias_data['top_candidates']:
                html += f"<tr><td>{self._escape_html(candidate)}</td><td>{count}</td></tr>"
            html += "</table>"

        html += (
            f"<p><strong>Audit Source:</strong> {self._escape_html(alias_data.get('path', ''))}"
            f" (window: last {alias_data.get('days_window', 14)} day(s) present)</p>"
        )
        return html

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        """Parse ISO datetime/date strings safely."""
        if not value:
            return None
        text_value = str(value).strip()
        if not text_value:
            return None
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text_value)
        except ValueError:
            try:
                return datetime.fromisoformat(text_value[:10])
            except ValueError:
                return None

    def _summarize_suspicious_deletes(self, report_timestamp: str | None) -> dict:
        """
        Summarize likely-incorrect event deletes from deleted-events audit JSONL.

        Flags are heuristic and intended for human review, not auto-restore.
        """
        output_cfg = self.config.get('output', {})
        reporting_cfg = self.validation_config.get('reporting', {})
        audit_path = output_cfg.get('deleted_events_audit', 'logs/deleted_events_audit.jsonl')
        days_window = int(reporting_cfg.get('suspicious_delete_days', 14) or 14)
        max_rows = int(reporting_cfg.get('suspicious_delete_max_rows', 40) or 40)

        summary = {
            'available': False,
            'path': audit_path,
            'days_window': days_window,
            'rows_analyzed': 0,
            'window_rows': 0,
            'exact_duplicate_deletes': 0,
            'fuzzy_model_candidate_deletes': 0,
            'flagged_count': 0,
            'category_counts': [],
            'flagged_rows': [],
            'error': '',
        }

        if not os.path.exists(audit_path):
            summary['error'] = f"Deleted-events audit file not found: {audit_path}"
            return summary

        end_ts = datetime.now()
        if report_timestamp:
            parsed = self._parse_iso_datetime(report_timestamp)
            if parsed is not None:
                end_ts = parsed
        start_ts = end_ts - timedelta(days=days_window)
        today_date = end_ts.date()

        category_counts: Counter = Counter()
        flagged_rows: list[dict] = []

        try:
            with open(audit_path, 'r', encoding='utf-8') as file:
                for line in file:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    summary['rows_analyzed'] += 1

                    deleted_at = self._parse_iso_datetime(record.get('deleted_at_utc'))
                    if deleted_at is None or not (start_ts <= deleted_at <= end_ts):
                        continue
                    summary['window_rows'] += 1

                    event = record.get('event') or {}
                    if not isinstance(event, dict):
                        continue
                    deletion_source = str(record.get('deletion_source') or '')
                    deletion_reason = str(record.get('deletion_reason') or '')
                    extra_context = record.get('extra_context') if isinstance(record.get('extra_context'), dict) else {}
                    if deletion_source == 'db.dedup':
                        summary['exact_duplicate_deletes'] += 1
                    if deletion_source in {
                        'db.fuzzy_duplicates',
                        'irrelevant_rows.delete_irrelevant_rows',
                        'dedup_llm.delete_duplicates',
                        'dedup_llm.deduplicate_with_embeddings',
                        'dedup_llm.resolve_venue_time_conflicts',
                    }:
                        summary['fuzzy_model_candidate_deletes'] += 1
                    event_name = str(event.get('event_name') or '')
                    url = str(event.get('url') or '')
                    location = str(event.get('location') or '')
                    start_date_raw = str(event.get('start_date') or '')
                    time_stamp_raw = str(event.get('time_stamp') or '')
                    dance_style = str(event.get('dance_style') or '')
                    description = str(event.get('description') or '')

                    score = 0
                    signals: list[str] = []
                    category = ''
                    fuzzy_score_raw = extra_context.get('fuzzy_score')
                    try:
                        fuzzy_score = int(fuzzy_score_raw) if fuzzy_score_raw is not None else None
                    except (TypeError, ValueError):
                        fuzzy_score = None

                    if deletion_source in {
                        'irrelevant_rows.delete_irrelevant_rows',
                        'dedup_llm.delete_duplicates',
                        'dedup_llm.deduplicate_with_embeddings',
                        'dedup_llm.resolve_venue_time_conflicts',
                    }:
                        score += 2
                        category = category or 'model_driven_delete_requires_review'
                        signals.append("Delete came from model-driven/fuzzy pipeline.")

                    if deletion_source == 'db.fuzzy_duplicates':
                        if fuzzy_score is not None and fuzzy_score < 90:
                            score += 2
                            category = category or 'low_confidence_fuzzy_duplicate'
                            signals.append(f"Fuzzy duplicate merged at low score ({fuzzy_score}).")

                    start_dt = self._parse_iso_datetime(start_date_raw)
                    if start_dt is not None and start_dt.date() >= today_date:
                        if deletion_source in {
                            'db.delete_old_events',
                            'db.delete_likely_dud_events',
                            'db.delete_events_with_nulls',
                        }:
                            score += 3
                            category = 'future_event_deleted_by_cleanup'
                            signals.append("Future-dated event removed by cleanup path.")

                    if deletion_reason in {'outside_bc_filter', 'outside_canada_filter'}:
                        loc_low = location.lower()
                        if any(tok in loc_low for tok in ['victoria', 'vancouver island', ', bc', 'british columbia']):
                            score += 3
                            if not category:
                                category = 'local_looking_location_deleted_as_non_local'
                            signals.append("Location text appears local but removed by geo filter.")

                    if deletion_reason == 'empty_source_dance_style_url_no_address':
                        if url.strip() or dance_style.strip() or len(description.strip()) >= 40:
                            score += 2
                            if not category:
                                category = 'non_empty_event_deleted_as_empty_dud'
                            signals.append("Dud-delete reason conflicts with non-empty event fields.")

                    if deletion_reason == 'empty_dance_style_url_other_no_location_description':
                        if event_name.strip() and (url.strip() or len(description.strip()) >= 40):
                            score += 2
                            if not category:
                                category = 'non_empty_other_deleted_as_empty'
                            signals.append("Delete reason conflicts with populated URL/description.")

                    created_at = self._parse_iso_datetime(time_stamp_raw)
                    if created_at is not None:
                        age_hours = (deleted_at - created_at).total_seconds() / 3600.0
                        if 0 <= age_hours <= 6:
                            score += 1
                            if not category:
                                category = 'rapid_post_ingest_delete'
                            signals.append(f"Deleted {age_hours:.1f}h after ingest.")

                    # De-prioritize routine black/white dedup deletes unless they have strong contradictory evidence.
                    if deletion_source == 'db.dedup' and score < 4:
                        continue
                    if deletion_source == 'db.fuzzy_duplicates' and (fuzzy_score is None or fuzzy_score >= 90) and score < 4:
                        continue

                    if score >= 2:
                        category_counts.update([category or 'other_suspicious'])
                        flagged_rows.append({
                            'deleted_at_utc': str(record.get('deleted_at_utc') or ''),
                            'event_id': event.get('event_id'),
                            'event_name': event_name,
                            'start_date': start_date_raw,
                            'source': event.get('source'),
                            'deletion_source': deletion_source,
                            'deletion_reason': deletion_reason,
                            'url': url,
                            'location': location,
                            'suspicion_score': score,
                            'signals': signals,
                        })
        except Exception as e:
            summary['error'] = f"Failed reading deleted-events audit: {e}"
            return summary

        flagged_rows.sort(
            key=lambda row: (
                -int(row.get('suspicion_score') or 0),
                str(row.get('deleted_at_utc') or ''),
            )
        )
        summary['available'] = True
        summary['flagged_count'] = len(flagged_rows)
        summary['category_counts'] = category_counts.most_common(10)
        summary['flagged_rows'] = flagged_rows[:max_rows]
        return summary

    def _build_suspicious_deletes_html(self, suspicious_data: dict) -> str:
        """Render HTML for likely-incorrect delete review."""
        if not suspicious_data:
            return "<p class='error-box'>❌ Suspicious delete summary unavailable.</p>"
        if not suspicious_data.get('available'):
            err = self._escape_html(suspicious_data.get('error', 'Suspicious delete audit unavailable'))
            path = self._escape_html(suspicious_data.get('path', ''))
            return (
                "<p class='error-box'>"
                f"⚠ Suspicious delete audit not available.<br>{err}<br><strong>Path:</strong> {path}"
                "</p>"
            )

        flagged_rows = suspicious_data.get('flagged_rows', []) or []
        category_rows = suspicious_data.get('category_counts', []) or []
        html = (
            "<div class='metric-container'>"
            f"<div class='metric'><div class='metric-value'>{int(suspicious_data.get('rows_analyzed', 0))}</div><div class='metric-label'>Audit Rows (All Time)</div></div>"
            f"<div class='metric'><div class='metric-value'>{int(suspicious_data.get('window_rows', 0))}</div><div class='metric-label'>Rows in Window</div></div>"
            f"<div class='metric'><div class='metric-value'>{int(suspicious_data.get('exact_duplicate_deletes', 0))}</div><div class='metric-label'>Exact Duplicate Deletes</div></div>"
            f"<div class='metric'><div class='metric-value'>{int(suspicious_data.get('fuzzy_model_candidate_deletes', 0))}</div><div class='metric-label'>Fuzzy/Model Delete Candidates</div></div>"
            f"<div class='metric'><div class='metric-value'>{int(suspicious_data.get('flagged_count', 0))}</div><div class='metric-label'>Likely Incorrect Deletes</div></div>"
            "</div>"
            f"<p><strong>Window:</strong> Last {int(suspicious_data.get('days_window', 0))} day(s)</p>"
            f"<p><strong>Audit Path:</strong> {self._escape_html(suspicious_data.get('path', ''))}</p>"
        )

        if category_rows:
            html += "<h3>Top Suspicion Categories</h3><table><tr><th>Category</th><th>Count</th></tr>"
            for category, count in category_rows:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(category)}</td>"
                    f"<td>{int(count)}</td>"
                    "</tr>"
                )
            html += "</table>"

        if not flagged_rows:
            return html + "<p>No likely incorrect deletes were detected in this window.</p>"

        html += (
            "<h3>Flagged Events (Review)</h3>"
            "<table><tr>"
            "<th>Deleted At</th><th>Score</th><th>Event ID</th><th>Event</th><th>Start Date</th>"
            "<th>Delete Source</th><th>Delete Reason</th><th>Signals</th><th>URL</th>"
            "</tr>"
        )
        for row in flagged_rows:
            signals = "; ".join(row.get('signals', []) or [])
            html += (
                "<tr>"
                f"<td>{self._escape_html(row.get('deleted_at_utc', ''))}</td>"
                f"<td>{int(row.get('suspicion_score', 0))}</td>"
                f"<td>{self._escape_html(row.get('event_id', ''))}</td>"
                f"<td>{self._escape_html(row.get('event_name', ''))}</td>"
                f"<td>{self._escape_html(row.get('start_date', ''))}</td>"
                f"<td>{self._escape_html(row.get('deletion_source', ''))}</td>"
                f"<td>{self._escape_html(row.get('deletion_reason', ''))}</td>"
                f"<td>{self._escape_html(signals)}</td>"
                f"<td>{self._escape_html(row.get('url', ''))}</td>"
                "</tr>"
            )
        html += "</table>"
        return html

    def _make_scraping_validator(self) -> ScrapingValidator:
        """Build a scraping validator instance for shared URL/watchlist checks."""
        return ScrapingValidator(self.db_handler, self.config)

    def _load_coverage_watchlist_rows(self) -> tuple[list[dict[str, Any]], str]:
        """Load curated coverage watchlist rows, falling back to important-url seed files."""
        watchlist_path = os.path.join("data", "watchlists", "coverage_watchlist.csv")
        candidate_frames: list[pd.DataFrame] = []
        source_label = "coverage_watchlist_csv"

        if os.path.exists(watchlist_path):
            try:
                watchlist_df = pd.read_csv(watchlist_path)
                if not watchlist_df.empty:
                    candidate_frames.append(watchlist_df.copy())
            except Exception:
                pass

        if not candidate_frames:
            source_label = "fallback_whitelist_edge_cases"
            fallback_paths = [
                ("data/urls/aaa_urls.csv", "high"),
                ("data/other/edge_cases.csv", "medium"),
            ]
            for path, priority in fallback_paths:
                if not os.path.exists(path):
                    continue
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if df.empty:
                    continue
                if "link" not in df.columns:
                    continue
                normalized = pd.DataFrame(
                    {
                        "source_name": df.get("source", pd.Series([""] * len(df))),
                        "source_url": df["link"],
                        "source_type": pd.Series(["seed_url"] * len(df)),
                        "priority": pd.Series([priority] * len(df)),
                        "expected_frequency": pd.Series(["weekly"] * len(df)),
                        "coverage_region": pd.Series(["Greater Victoria"] * len(df)),
                        "active": pd.Series([True] * len(df)),
                    }
                )
                candidate_frames.append(normalized)

        if not candidate_frames:
            return [], source_label

        merged = pd.concat(candidate_frames, ignore_index=True)
        rows: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for _, row in merged.iterrows():
            source_url = str(row.get("source_url") or row.get("link") or "").strip()
            if not source_url or source_url in seen_urls:
                continue
            active_raw = row.get("active", True)
            active = str(active_raw).strip().lower() not in {"false", "0", "no", "n"}
            if not active:
                continue
            seen_urls.add(source_url)
            rows.append(
                {
                    "source_name": str(row.get("source_name") or row.get("source") or source_url).strip(),
                    "source_url": source_url,
                    "source_type": str(row.get("source_type") or "seed_url").strip(),
                    "priority": str(row.get("priority") or "medium").strip().lower(),
                    "expected_frequency": str(row.get("expected_frequency") or "weekly").strip(),
                    "coverage_region": str(row.get("coverage_region") or "Greater Victoria").strip(),
                    "active": True,
                }
            )
        return rows, source_label

    def _load_manual_coverage_audit_rows(self) -> tuple[list[dict[str, Any]], str]:
        """Load the periodic manual coverage audit set."""
        audit_path = os.path.join("data", "evaluation", "manual_coverage_audit.csv")
        if not os.path.exists(audit_path):
            return [], "manual_coverage_audit_missing"
        try:
            audit_df = pd.read_csv(audit_path)
        except Exception:
            return [], "manual_coverage_audit_unreadable"
        if audit_df.empty:
            return [], "manual_coverage_audit_empty"

        rows: list[dict[str, Any]] = []
        for _, row in audit_df.iterrows():
            active_raw = row.get("active", True)
            active = str(active_raw).strip().lower() not in {"false", "0", "no", "n"}
            if not active:
                continue
            rows.append(
                {
                    "source_name": str(row.get("source_name") or "").strip(),
                    "source_url": str(row.get("source_url") or "").strip(),
                    "event_name": str(row.get("event_name") or "").strip(),
                    "start_date": str(row.get("start_date") or "").strip(),
                    "expected_present": str(row.get("expected_present", "True") or "True").strip().lower() not in {"false", "0", "no", "n"},
                    "active": True,
                }
            )
        return rows, "manual_coverage_audit_csv"

    def _summarize_manual_coverage_audit(self) -> dict:
        """Summarize a curated manual audit of expected events against the current events table."""
        audit_rows, source_label = self._load_manual_coverage_audit_rows()
        summary = {
            "available": False,
            "manual_audit_source": source_label,
            "sample_size": len(audit_rows),
            "expected_present_count": 0,
            "captured_count": 0,
            "missed_count": 0,
            "missed_event_rate_manual_audit_pct": None,
            "sample_missed_events": [],
            "error": "",
        }
        if not audit_rows:
            summary["error"] = "manual coverage audit unavailable"
            return summary

        captured_count = 0
        expected_present_count = 0
        missed_rows: list[dict[str, Any]] = []
        query = """
            SELECT COUNT(*)
            FROM events
            WHERE (:event_name = '' OR LOWER(TRIM(event_name)) = LOWER(TRIM(:event_name)))
              AND (:start_date = '' OR CAST(start_date AS TEXT) = :start_date)
              AND (:source_url = '' OR LOWER(TRIM(url)) = LOWER(TRIM(:source_url)))
        """
        for row in audit_rows:
            if not bool(row.get("expected_present")):
                continue
            expected_present_count += 1
            params = {
                "event_name": str(row.get("event_name") or "").strip(),
                "start_date": str(row.get("start_date") or "").strip(),
                "source_url": str(row.get("source_url") or "").strip(),
            }
            try:
                result_rows = self.db_handler.execute_query(query, params) or []
            except Exception as e:
                summary["error"] = f"manual audit query failed: {e}"
                return summary
            match_count = int(result_rows[0][0]) if result_rows else 0
            if match_count > 0:
                captured_count += 1
            else:
                missed_rows.append(
                    {
                        "source_name": str(row.get("source_name") or ""),
                        "source_url": str(row.get("source_url") or ""),
                        "event_name": str(row.get("event_name") or ""),
                        "start_date": str(row.get("start_date") or ""),
                    }
                )

        missed_count = max(expected_present_count - captured_count, 0)
        summary.update(
            {
                "available": True,
                "expected_present_count": expected_present_count,
                "captured_count": captured_count,
                "missed_count": missed_count,
                "missed_event_rate_manual_audit_pct": round((missed_count / expected_present_count) * 100.0, 2) if expected_present_count else None,
                "sample_missed_events": missed_rows[:25],
            }
        )
        return summary

    def _summarize_coverage_watchlist(self) -> dict:
        """Summarize coverage watchlist source hits and event capture using existing scrape evidence."""
        watchlist_rows, source_label = self._load_coverage_watchlist_rows()
        manual_audit_summary = self._summarize_manual_coverage_audit()
        source_discovery_summary = self._summarize_new_source_discovery(watchlist_rows)
        summary = {
            "available": False,
            "watchlist_source": source_label,
            "watchlist_rows_total": len(watchlist_rows),
            "sources_hit": 0,
            "source_hit_rate_pct": None,
            "event_capture_hits": 0,
            "event_capture_rate_pct": None,
            "priority_sources_total": 0,
            "priority_sources_hit": 0,
            "priority_source_hit_rate_pct": None,
            "new_source_discovery_count": source_discovery_summary.get("new_source_discovery_count"),
            "missed_event_rate_manual_audit_pct": None,
            "manual_audit": manual_audit_summary,
            "new_source_discovery": source_discovery_summary,
            "missed_sources": [],
            "captured_sources": [],
            "error": "",
        }
        if not watchlist_rows:
            summary["error"] = "coverage watchlist unavailable"
            return summary

        validator = self._make_scraping_validator()
        window_days = int(getattr(validator, "days_back", 7) or 7)
        missed: list[dict[str, Any]] = []
        captured: list[dict[str, Any]] = []
        source_hits = 0
        event_hits = 0
        priority_total = 0
        priority_hits = 0

        for row in watchlist_rows:
            priority = str(row.get("priority") or "medium").strip().lower()
            if priority == "high":
                priority_total += 1

            variants = validator._normalize_link_variants(str(row.get("source_url") or ""))
            recent_rows = validator._query_recent_url_rows(variants, window_days)
            child_success_count = validator._query_recent_child_success_count(variants, window_days)
            latest_overall_row = validator._query_latest_overall_row(variants)

            has_source_hit = bool(recent_rows or latest_overall_row)
            has_event_capture = bool(child_success_count > 0 or any(bool(r[2]) for r in recent_rows))
            if has_source_hit:
                source_hits += 1
            if has_event_capture:
                event_hits += 1
            if priority == "high" and has_source_hit:
                priority_hits += 1

            payload = {
                "source_name": str(row.get("source_name") or ""),
                "source_url": str(row.get("source_url") or ""),
                "priority": priority,
                "recent_rows": int(len(recent_rows)),
                "recent_child_success_count": int(child_success_count),
                "latest_seen_time": str(latest_overall_row[3]) if latest_overall_row and latest_overall_row[3] else "",
                "status": "captured" if has_event_capture else ("seen_no_events" if has_source_hit else "missed"),
            }
            if has_event_capture or has_source_hit:
                captured.append(payload)
            else:
                missed.append(payload)

        total = len(watchlist_rows)
        summary.update(
            {
                "available": True,
                "sources_hit": source_hits,
                "source_hit_rate_pct": round((source_hits / total) * 100.0, 2) if total else None,
                "event_capture_hits": event_hits,
                "event_capture_rate_pct": round((event_hits / total) * 100.0, 2) if total else None,
                "priority_sources_total": priority_total,
                "priority_sources_hit": priority_hits,
                "priority_source_hit_rate_pct": round((priority_hits / priority_total) * 100.0, 2) if priority_total else None,
                "new_source_discovery_count": source_discovery_summary.get("new_source_discovery_count"),
                "missed_event_rate_manual_audit_pct": manual_audit_summary.get("missed_event_rate_manual_audit_pct"),
                "manual_audit": manual_audit_summary,
                "new_source_discovery": source_discovery_summary,
                "missed_sources": missed[:25],
                "captured_sources": captured[:25],
            }
        )
        return summary

    def _summarize_new_source_discovery(self, watchlist_rows: list[dict[str, Any]] | None) -> dict:
        """Summarize event-source domains currently in the DB that are not on the coverage watchlist."""
        watchlist_domains: set[str] = set()
        for row in watchlist_rows or []:
            source_url = normalize_evaluation_url((row or {}).get("source_url"))
            if not source_url:
                continue
            domain = (urlparse(source_url).netloc or "").lower().removeprefix("www.")
            if domain:
                watchlist_domains.add(domain)

        summary = {
            "available": False,
            "new_source_discovery_count": 0,
            "sample_new_source_domains": [],
            "watchlist_domains_total": len(watchlist_domains),
            "error": "",
        }
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "execute_query"):
            summary["error"] = "db_handler unavailable"
            return summary

        try:
            rows = self.db_handler.execute_query(
                """
                SELECT DISTINCT LOWER(TRIM(url)) AS url_value
                FROM events
                WHERE COALESCE(end_date, start_date) >= CURRENT_DATE
                  AND COALESCE(NULLIF(TRIM(url), ''), '') <> ''
                """
            ) or []
        except Exception as e:
            summary["error"] = f"new source discovery query failed: {e}"
            return summary

        discovered_domains: set[str] = set()
        for row in rows:
            raw_url = str(row[0] or "").strip()
            normalized_url = normalize_evaluation_url(raw_url)
            if not normalized_url:
                continue
            domain = (urlparse(normalized_url).netloc or "").lower().removeprefix("www.")
            if not domain or domain in watchlist_domains:
                continue
            discovered_domains.add(domain)

        summary.update(
            {
                "available": True,
                "new_source_discovery_count": len(discovered_domains),
                "sample_new_source_domains": sorted(discovered_domains)[:25],
            }
        )
        return summary

    def _summarize_duplicate_audit(self) -> dict:
        """Summarize duplicate patterns in the current events table."""
        summary = {
            "available": False,
            "total_events": 0,
            "duplicate_clusters_count": 0,
            "duplicate_rows": 0,
            "duplicate_rate_per_100_events": None,
            "severe_duplicate_clusters_count": 0,
            "severe_duplicate_rows": 0,
            "severe_duplicate_rate_per_100_events": None,
            "top_duplicate_domains": [],
            "sample_duplicate_clusters": [],
            "error": "",
        }
        try:
            total_rows = self.db_handler.execute_query("SELECT COUNT(*) FROM events") or []
            total_events = int(total_rows[0][0]) if total_rows else 0
            summary["total_events"] = total_events
            if total_events <= 0:
                summary["available"] = True
                return summary

            duplicate_rows = self.db_handler.execute_query(
                """
                SELECT
                    COALESCE(NULLIF(TRIM(url), ''), '(no url)') AS url,
                    COALESCE(NULLIF(TRIM(event_name), ''), '(no event name)') AS event_name,
                    start_date,
                    COALESCE(start_time::text, '') AS start_time_text,
                    COUNT(*) AS row_count
                FROM events
                GROUP BY 1, 2, 3, 4
                HAVING COUNT(*) > 1
                ORDER BY row_count DESC, event_name ASC
                LIMIT 100
                """
            ) or []

            severe_rows = self.db_handler.execute_query(
                """
                SELECT
                    COALESCE(NULLIF(TRIM(event_name), ''), '(no event name)') AS event_name,
                    start_date,
                    COALESCE(start_time::text, '') AS start_time_text,
                    COALESCE(address_id, 0) AS address_id_key,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT COALESCE(NULLIF(TRIM(url), ''), '(no url)')) AS distinct_urls
                FROM events
                GROUP BY 1, 2, 3, 4
                HAVING COUNT(*) > 1
                   AND COUNT(DISTINCT COALESCE(NULLIF(TRIM(url), ''), '(no url)')) > 1
                ORDER BY row_count DESC, event_name ASC
                LIMIT 100
                """
            ) or []
        except Exception as e:
            summary["error"] = f"duplicate audit query failed: {e}"
            return summary

        duplicate_clusters_count = len(duplicate_rows)
        duplicate_row_excess = sum(max(int(row[4]) - 1, 0) for row in duplicate_rows)
        severe_clusters_count = len(severe_rows)
        severe_row_excess = sum(max(int(row[4]) - 1, 0) for row in severe_rows)

        domain_counts: Counter[str] = Counter()
        sample_clusters: list[dict[str, Any]] = []
        for row in duplicate_rows[:15]:
            url = str(row[0] or "")
            domain = urlparse(url).netloc.lower() if url and url != "(no url)" else "(no url)"
            domain_counts.update([domain])
            sample_clusters.append(
                {
                    "kind": "exact_key_duplicate",
                    "url": url,
                    "event_name": str(row[1] or ""),
                    "start_date": str(row[2] or ""),
                    "start_time": str(row[3] or ""),
                    "row_count": int(row[4] or 0),
                }
            )
        for row in severe_rows[:10]:
            sample_clusters.append(
                {
                    "kind": "cross_url_same_event",
                    "event_name": str(row[0] or ""),
                    "start_date": str(row[1] or ""),
                    "start_time": str(row[2] or ""),
                    "address_id": int(row[3] or 0),
                    "row_count": int(row[4] or 0),
                    "distinct_urls": int(row[5] or 0),
                }
            )

        summary.update(
            {
                "available": True,
                "duplicate_clusters_count": duplicate_clusters_count,
                "duplicate_rows": duplicate_row_excess,
                "duplicate_rate_per_100_events": round((duplicate_row_excess / total_events) * 100.0, 2),
                "severe_duplicate_clusters_count": severe_clusters_count,
                "severe_duplicate_rows": severe_row_excess,
                "severe_duplicate_rate_per_100_events": round((severe_row_excess / total_events) * 100.0, 2),
                "top_duplicate_domains": [
                    {"domain": domain, "cluster_count": count}
                    for domain, count in domain_counts.most_common(10)
                ],
                "sample_duplicate_clusters": sample_clusters,
            }
        )
        return summary

    def _summarize_event_data_quality(self) -> dict:
        """Summarize stale and invalid event rates from the current events table."""
        summary = {
            "available": False,
            "total_events": 0,
            "invalid_event_count": 0,
            "invalid_event_rate_pct": None,
            "stale_event_count": 0,
            "stale_event_rate_pct": None,
            "error": "",
        }
        try:
            total_rows = self.db_handler.execute_query("SELECT COUNT(*) FROM events") or []
            total_events = int(total_rows[0][0]) if total_rows else 0
            summary["total_events"] = total_events
            if total_events <= 0:
                summary["available"] = True
                return summary

            invalid_rows = self.db_handler.execute_query(
                """
                SELECT COUNT(*)
                FROM events
                WHERE COALESCE(NULLIF(TRIM(event_name), ''), '') = ''
                   OR start_date IS NULL
                   OR COALESCE(NULLIF(TRIM(url), ''), '') = ''
                """
            ) or []
            stale_rows = self.db_handler.execute_query(
                """
                SELECT COUNT(*)
                FROM events
                WHERE COALESCE(end_date, start_date) < CURRENT_DATE
                """
            ) or []
        except Exception as e:
            summary["error"] = f"event quality query failed: {e}"
            return summary

        invalid_count = int(invalid_rows[0][0]) if invalid_rows else 0
        stale_count = int(stale_rows[0][0]) if stale_rows else 0
        summary.update(
            {
                "available": True,
                "invalid_event_count": invalid_count,
                "invalid_event_rate_pct": round((invalid_count / total_events) * 100.0, 2),
                "stale_event_count": stale_count,
                "stale_event_rate_pct": round((stale_count / total_events) * 100.0, 2),
            }
        )
        return summary

    def _summarize_field_accuracy(self, accuracy_replay_summary: dict | None) -> dict:
        """Summarize field-level replay accuracy using replay snapshots and existing address resolution."""
        rows = accuracy_replay_summary.get("rows", []) if isinstance(accuracy_replay_summary, dict) else []
        field_buckets = {
            "date": {"comparable": 0, "matches": 0},
            "time": {"comparable": 0, "matches": 0},
            "location": {"comparable": 0, "matches": 0},
            "source": {"comparable": 0, "matches": 0},
            "address_id": {"comparable": 0, "matches": 0},
            "dance_style": {"comparable": 0, "matches": 0},
            "description": {"comparable": 0, "matches": 0},
        }

        for row in rows:
            if not isinstance(row, dict):
                continue
            baseline = row.get("baseline", {}) if isinstance(row.get("baseline"), dict) else {}
            replay = row.get("replay", {}) if isinstance(row.get("replay"), dict) else {}

            baseline_date = self._normalize_date_value(baseline.get("start_date"))
            replay_date = self._normalize_date_value(replay.get("start_date"))
            if baseline_date and replay_date:
                field_buckets["date"]["comparable"] += 1
                if baseline_date == replay_date:
                    field_buckets["date"]["matches"] += 1

            baseline_time = self._normalize_time_value(baseline.get("start_time"))
            replay_time = self._normalize_time_value(replay.get("start_time"))
            if baseline_time and replay_time:
                field_buckets["time"]["comparable"] += 1
                if self._times_equivalent_with_12h_guard(replay_time, baseline_time):
                    field_buckets["time"]["matches"] += 1

            baseline_location = self._normalize_text_value(baseline.get("location"))
            replay_location = self._normalize_text_value(replay.get("location"))
            if baseline_location and replay_location:
                field_buckets["location"]["comparable"] += 1
                if baseline_location == replay_location:
                    field_buckets["location"]["matches"] += 1

            baseline_source = self._normalize_text_value(baseline.get("source"))
            replay_source = self._normalize_text_value(replay.get("source"))
            if baseline_source and replay_source and not self._is_placeholder_source(baseline_source) and not self._is_placeholder_source(replay_source):
                field_buckets["source"]["comparable"] += 1
                if baseline_source == replay_source:
                    field_buckets["source"]["matches"] += 1

            baseline_address_id = self._normalize_optional_int(baseline.get("address_id"))
            replay_address_id = self._normalize_optional_int(replay.get("address_id"))
            if replay_address_id is None:
                replay_address_id = self._resolve_replay_address_id(replay)
            if baseline_address_id is not None and replay_address_id is not None:
                field_buckets["address_id"]["comparable"] += 1
                if baseline_address_id == replay_address_id:
                    field_buckets["address_id"]["matches"] += 1

            baseline_styles = self._normalize_dance_style_tokens(baseline.get("dance_style"))
            replay_styles = self._normalize_dance_style_tokens(replay.get("dance_style"))
            if baseline_styles and replay_styles:
                field_buckets["dance_style"]["comparable"] += 1
                if baseline_styles == replay_styles:
                    field_buckets["dance_style"]["matches"] += 1

            baseline_description = self._normalize_text_value(baseline.get("description"))
            replay_description = self._normalize_text_value(replay.get("description"))
            if baseline_description and replay_description:
                field_buckets["description"]["comparable"] += 1
                if self._descriptions_equivalent(baseline_description, replay_description):
                    field_buckets["description"]["matches"] += 1

        def _pct(bucket: dict[str, int]) -> float | None:
            comparable = int(bucket.get("comparable", 0) or 0)
            matches = int(bucket.get("matches", 0) or 0)
            if comparable <= 0:
                return None
            return round((matches / comparable) * 100.0, 2)

        return {
            "available": True,
            "date_pct": _pct(field_buckets["date"]),
            "time_pct": _pct(field_buckets["time"]),
            "location_pct": _pct(field_buckets["location"]),
            "source_pct": _pct(field_buckets["source"]),
            "address_id_pct": _pct(field_buckets["address_id"]),
            "dance_style_pct": _pct(field_buckets["dance_style"]),
            "description_pct": _pct(field_buckets["description"]),
            "sample_sizes": {
                "date": field_buckets["date"]["comparable"],
                "time": field_buckets["time"]["comparable"],
                "location": field_buckets["location"]["comparable"],
                "source": field_buckets["source"]["comparable"],
                "address_id": field_buckets["address_id"]["comparable"],
                "dance_style": field_buckets["dance_style"]["comparable"],
                "description": field_buckets["description"]["comparable"],
            },
        }

    def _build_phase2_integrity_html(self, duplicate_summary: dict, coverage_summary: dict) -> str:
        """Render a compact Phase 2 integrity section for duplicates and watchlist coverage."""
        dup_rate = duplicate_summary.get("duplicate_rate_per_100_events")
        severe_rate = duplicate_summary.get("severe_duplicate_rate_per_100_events")
        source_hit = coverage_summary.get("source_hit_rate_pct")
        event_hit = coverage_summary.get("event_capture_rate_pct")
        manual_missed = coverage_summary.get("missed_event_rate_manual_audit_pct")
        html = "<div class='metric-container'>"
        cards = [
            ("Duplicate Rate /100", dup_rate),
            ("Severe Duplicates /100", severe_rate),
            ("Watchlist Source Hit %", source_hit),
            ("Watchlist Event Capture %", event_hit),
            ("Manual Audit Missed %", manual_missed),
        ]
        for label, value in cards:
            value_text = f"{float(value):.2f}" if isinstance(value, (int, float)) else "n/a"
            html += (
                "<div class='metric'>"
                f"<div class='metric-value'>{self._escape_html(value_text)}</div>"
                f"<div class='metric-label'>{self._escape_html(label)}</div>"
                "</div>"
            )
        html += "</div>"
        html += (
            f"<p><strong>Coverage watchlist source:</strong> {self._escape_html(str(coverage_summary.get('watchlist_source', '')))}</p>"
            f"<p><strong>Duplicate clusters:</strong> {int(duplicate_summary.get('duplicate_clusters_count', 0) or 0)}; "
            f"<strong>Severe clusters:</strong> {int(duplicate_summary.get('severe_duplicate_clusters_count', 0) or 0)}</p>"
        )
        return html

    def _build_phase3_honest_evaluation_html(
        self,
        dev_summary: dict,
        holdout_summary: dict,
        domain_capped_summary: dict,
        guardrails: dict,
    ) -> str:
        """Render a compact Phase 3 section for dev/holdout/domain-cap evaluation and guardrail status."""
        dev_accuracy = dev_summary.get("replay_url_accuracy_pct")
        holdout_accuracy = holdout_summary.get("replay_url_accuracy_pct")
        domain_capped_accuracy = domain_capped_summary.get("replay_url_accuracy_pct")
        guardrail_status = str(guardrails.get("status", "UNKNOWN") or "UNKNOWN")
        html = "<div class='metric-container'>"
        for label, value in (
            ("Dev Replay URL %", dev_accuracy),
            ("Holdout Replay URL %", holdout_accuracy),
            ("Domain-Capped Replay URL %", domain_capped_accuracy),
        ):
            value_text = f"{float(value):.2f}" if isinstance(value, (int, float)) else "n/a"
            html += (
                "<div class='metric'>"
                f"<div class='metric-value'>{self._escape_html(value_text)}</div>"
                f"<div class='metric-label'>{self._escape_html(label)}</div>"
                "</div>"
            )
        html += (
            "<div class='metric'>"
            f"<div class='metric-value'>{self._escape_html(guardrail_status)}</div>"
            "<div class='metric-label'>Guardrails</div>"
            "</div>"
        )
        html += "</div>"
        violations = guardrails.get("violations", []) if isinstance(guardrails.get("violations"), list) else []
        if violations:
            violation_items = "".join(
                f"<li>{self._escape_html(str(v.get('detail', '')))} "
                f"({self._escape_html(str(v.get('actual', '')))} vs {self._escape_html(str(v.get('threshold', '')))}"
                f")</li>"
                for v in violations
            )
            html += f"<p><strong>Guardrail Violations:</strong></p><ul>{violation_items}</ul>"
        else:
            html += "<p><strong>Guardrail Violations:</strong> none</p>"
        html += (
            f"<p><strong>Dev URLs seen in replay:</strong> {int(dev_summary.get('replay_urls_seen', 0) or 0)} / "
            f"{int(dev_summary.get('dev_urls_total', 0) or 0)}</p>"
        )
        html += (
            f"<p><strong>Holdout URLs seen in replay:</strong> {int(holdout_summary.get('replay_urls_seen', 0) or 0)} / "
            f"{int(holdout_summary.get('holdout_urls_total', 0) or 0)}; "
            f"<strong>Domain cap:</strong> {int(domain_capped_summary.get('per_domain_cap', 0) or 0)}</p>"
        )
        return html

    def _build_domain_evaluation_html(self, domain_evaluation_summary: dict) -> str:
        """Render a compact domain-level replay evaluation section."""
        if not isinstance(domain_evaluation_summary, dict) or not domain_evaluation_summary:
            return "<p class='error-box'>❌ Domain evaluation unavailable</p>"
        top_domains = domain_evaluation_summary.get("top_domains", []) if isinstance(domain_evaluation_summary.get("top_domains"), list) else []
        html = "<div class='metric-container'>"
        for label, value in (
            ("Replay Domains", domain_evaluation_summary.get("domain_count")),
            ("Replay URLs", domain_evaluation_summary.get("total_urls")),
        ):
            value_text = str(int(value)) if isinstance(value, int) else "n/a"
            html += (
                "<div class='metric'>"
                f"<div class='metric-value'>{self._escape_html(value_text)}</div>"
                f"<div class='metric-label'>{self._escape_html(label)}</div>"
                "</div>"
            )
        html += "</div>"
        if not top_domains:
            return html + "<p><strong>Top Domains:</strong> none</p>"
        rows = ""
        for item in top_domains[:10]:
            if not isinstance(item, dict):
                continue
            rows += (
                "<tr>"
                f"<td>{self._escape_html(str(item.get('domain', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('url_count', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('replay_url_accuracy_pct', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('top_mismatch_category', '')))}</td>"
                "</tr>"
            )
        return html + "<table><tr><th>Domain</th><th>URLs</th><th>Replay URL %</th><th>Top Mismatch</th></tr>" + rows + "</table>"

    def _build_run_delta_html(self, run_delta_summary: dict) -> str:
        """Render a compact run-to-run and holdout-baseline comparison section."""
        if not isinstance(run_delta_summary, dict) or not run_delta_summary.get("available"):
            return "<p>No previous run or holdout baseline comparison available yet.</p>"

        def _render_section(title: str, comparison: dict) -> str:
            if not isinstance(comparison, dict) or not comparison.get("available"):
                return (
                    "<div class='metric' style='flex: 1 1 320px;'>"
                    f"<div class='metric-label'><strong>{self._escape_html(title)}</strong></div>"
                    "<div class='metric-value'>n/a</div>"
                    "</div>"
                )
            summary = comparison.get("summary", {}) if isinstance(comparison.get("summary"), dict) else {}
            top_regressions = comparison.get("top_regressions", []) if isinstance(comparison.get("top_regressions"), list) else []
            regression_items = "".join(
                f"<li>{self._escape_html(str(item.get('label', '')))}: {self._escape_html(str(item.get('delta', '')))}</li>"
                for item in top_regressions[:3]
                if isinstance(item, dict)
            ) or "<li>None</li>"
            return (
                "<div class='metric' style='flex: 1 1 320px;'>"
                f"<div class='metric-label'><strong>{self._escape_html(title)}</strong> ({self._escape_html(str(comparison.get('baseline_run_id', 'unknown')))})</div>"
                f"<div>Overall delta: {self._escape_html(str(summary.get('delta_overall_score', 'n/a')))}</div>"
                f"<div>Holdout delta: {self._escape_html(str(summary.get('delta_holdout_replay_url_accuracy_pct', 'n/a')))}</div>"
                f"<div>Metric deltas: {self._escape_html(str(summary.get('metric_delta_count', '0')))}</div>"
                f"<ul>{regression_items}</ul>"
                "</div>"
            )

        return (
            "<div class='metric-container'>"
            f"{_render_section('Previous Run', run_delta_summary.get('previous_run', {}))}"
            f"{_render_section('Holdout Baseline', run_delta_summary.get('holdout_baseline', {}))}"
            "</div>"
        )

    def _classify_recommendation_expected_impact(self, issue_type: str) -> dict[str, str]:
        """Map recommendation issue types to the KPI families they are most likely to affect."""
        impact_map: dict[str, dict[str, str]] = {
            "guardrail_violation": {"database_accuracy": "high", "events_coverage": "medium", "run_time": "low", "run_costs": "low", "chatbot_quality": "medium"},
            "duplicate_hotspot": {"database_accuracy": "high", "events_coverage": "low", "run_time": "low", "run_costs": "low", "chatbot_quality": "none"},
            "coverage_gap": {"database_accuracy": "low", "events_coverage": "high", "run_time": "medium", "run_costs": "medium", "chatbot_quality": "none"},
            "runtime_bottleneck": {"database_accuracy": "low", "events_coverage": "low", "run_time": "high", "run_costs": "medium", "chatbot_quality": "low"},
            "cost_hotspot": {"database_accuracy": "low", "events_coverage": "low", "run_time": "low", "run_costs": "high", "chatbot_quality": "low"},
            "classifier_regression": {"database_accuracy": "high", "events_coverage": "medium", "run_time": "low", "run_costs": "low", "chatbot_quality": "none"},
            "domain_regression": {"database_accuracy": "medium", "events_coverage": "medium", "run_time": "low", "run_costs": "low", "chatbot_quality": "none"},
        }
        return impact_map.get(issue_type, {"database_accuracy": "medium", "events_coverage": "medium", "run_time": "low", "run_costs": "low", "chatbot_quality": "none"})

    def _build_recommendation_plan(
        self,
        *,
        run_scorecard: dict,
        classifier_performance_summary: dict,
        duplicate_summary: dict,
        coverage_summary: dict,
        dev_summary: dict,
        runtime_summary: dict,
        llm_cost_summary: dict,
        domain_evaluation_summary: dict,
        run_delta_summary: dict,
    ) -> dict:
        """Build a structured recommendation plan artifact from existing evaluation signals."""
        recommendations_input = self._build_recommendations_input(
            run_scorecard=run_scorecard,
            duplicate_summary=duplicate_summary,
            coverage_summary=coverage_summary,
            dev_summary=dev_summary,
            classifier_performance_summary=classifier_performance_summary,
            runtime_summary=runtime_summary,
            llm_cost_summary=llm_cost_summary,
            domain_evaluation_summary=domain_evaluation_summary,
            run_delta_summary=run_delta_summary,
        )
        issues: list[dict[str, Any]] = []
        for detail in recommendations_input.get("top_regressions", [])[:3]:
            issues.append(
                {
                    "issue_type": "guardrail_violation",
                    "title": str(detail),
                    "evidence": {"guardrail": str(detail)},
                    "likely_root_causes": [
                        "A protected KPI regressed below its minimum threshold.",
                        "The current run traded accuracy or coverage for speed or cost.",
                    ],
                    "recommended_actions": [
                        "Inspect the failing metric's underlying artifact before promoting any further changes.",
                        "Prioritize fixes that address the failing guardrail without expanding model scope.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("guardrail_violation"),
                    "confidence": "high",
                    "regression_risk": "medium",
                }
            )
        for item in recommendations_input.get("duplicate_hotspots", [])[:1]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_type": "duplicate_hotspot",
                    "title": f"Reduce duplicates on {item.get('domain', '(unknown domain)')}",
                    "evidence": item,
                    "likely_root_causes": [
                        "Dedup keys are too weak for this source pattern.",
                        "The same logical event is entering the pipeline through multiple URLs.",
                    ],
                    "recommended_actions": [
                        "Tighten duplicate matching for this domain's recurring event shape.",
                        "Add a domain-specific duplicate guard or normalization rule if the pattern is stable.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("duplicate_hotspot"),
                    "confidence": "medium",
                    "regression_risk": "medium",
                }
            )
        for item in recommendations_input.get("coverage_gaps", [])[:1]:
            issues.append(
                {
                    "issue_type": "coverage_gap",
                    "title": f"Recover missed coverage for {item.get('source_name', item.get('source_url', 'watchlist source'))}" if isinstance(item, dict) else f"Recover missed coverage for {item}",
                    "evidence": item,
                    "likely_root_causes": [
                        "The source is not being reached reliably during the run.",
                        "Extraction or routing is failing before events are inserted.",
                    ],
                    "recommended_actions": [
                        "Review the source in the watchlist and confirm the current route still applies.",
                        "If the source is stable, add deterministic handling before expanding ML coverage.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("coverage_gap"),
                    "confidence": "medium",
                    "regression_risk": "low",
                }
            )
        for item in recommendations_input.get("runtime_bottlenecks", [])[:1]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_type": "runtime_bottleneck",
                    "title": f"Reduce runtime in {item.get('step', '(unknown step)')}",
                    "evidence": item,
                    "likely_root_causes": [
                        "The step is dominating the critical path.",
                        "The step is doing repeated work that can be cached or short-circuited.",
                    ],
                    "recommended_actions": [
                        "Inspect the step for repeated network, LLM, or database work.",
                        "Prefer deterministic skip/reuse paths before adding concurrency or model complexity.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("runtime_bottleneck"),
                    "confidence": "medium",
                    "regression_risk": "medium",
                }
            )
        for item in recommendations_input.get("largest_cost_deltas", [])[:1]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_type": "cost_hotspot",
                    "title": f"Reduce LLM cost in {item.get('bucket', '(unknown bucket)')}",
                    "evidence": item,
                    "likely_root_causes": [
                        "This provider or model is carrying most of the run cost.",
                        "Low-value prompts are still reaching expensive LLM paths.",
                    ],
                    "recommended_actions": [
                        "Review prompt routing to ensure deterministic paths short-circuit before LLM calls.",
                        "Measure whether cheaper models or fewer retries would preserve guardrails.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("cost_hotspot"),
                    "confidence": "medium",
                    "regression_risk": "high",
                }
            )
        for item in recommendations_input.get("classifier_routing_regressions", [])[:1]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_type": "classifier_regression",
                    "title": f"Improve classifier stage {item.get('stage', '(unknown stage)')}",
                    "evidence": item,
                    "likely_root_causes": [
                        "The current stage is misrouting pages before extraction begins.",
                        "Training data or deterministic rules do not yet cover this template mix well.",
                    ],
                    "recommended_actions": [
                        "Review the stage's failed URLs in replay and decide whether rules or training labels should change.",
                        "Do not promote new classifier rows automatically unless the failure is classifier-related.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("classifier_regression"),
                    "confidence": "medium",
                    "regression_risk": "medium",
                }
            )
        for item in recommendations_input.get("domain_regressions", [])[:1]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_type": "domain_regression",
                    "title": f"Investigate replay regression on {item.get('domain', '(unknown domain)')}",
                    "evidence": item,
                    "likely_root_causes": [
                        "The domain template changed or the current parser assumptions drifted.",
                        "The domain needs deterministic routing or extraction adjustments.",
                    ],
                    "recommended_actions": [
                        "Review recent mismatches and compare them with the domain's previous successful path.",
                        "If the domain is recurring, add or tighten a deterministic path before relying on ML.",
                    ],
                    "expected_kpi_impact": self._classify_recommendation_expected_impact("domain_regression"),
                    "confidence": "medium",
                    "regression_risk": "medium",
                }
            )

        prioritized_issues = issues[:3]
        return {
            "plan_version": "v1",
            "generated_at": datetime.utcnow().isoformat(),
            "run_id": str(run_scorecard.get("run_id", "") or ""),
            "overall_status": str((run_scorecard.get("overall_score") or {}).get("status", "") or ""),
            "guardrail_status": str((run_scorecard.get("guardrails") or {}).get("status", "") or ""),
            "top_issues": prioritized_issues,
            "summary": {
                "issue_count": len(prioritized_issues),
                "has_guardrail_failure": str((run_scorecard.get("guardrails") or {}).get("status", "") or "").upper() == "FAIL",
                "focus_order": [str(item.get("issue_type", "") or "") for item in prioritized_issues],
            },
        }

    def _build_recommendation_plan_html(self, recommendation_plan: dict) -> str:
        """Render a compact recommendation plan section for the HTML report."""
        if not isinstance(recommendation_plan, dict) or not recommendation_plan:
            return "<p class='error-box'>❌ Recommendation plan unavailable</p>"
        top_issues = recommendation_plan.get("top_issues", []) if isinstance(recommendation_plan.get("top_issues"), list) else []
        if not top_issues:
            return "<p>No recommendation issues identified for this run.</p>"
        rows = ""
        for item in top_issues[:3]:
            if not isinstance(item, dict):
                continue
            actions = item.get("recommended_actions", []) if isinstance(item.get("recommended_actions"), list) else []
            actions_text = "<br>".join(self._escape_html(str(action)) for action in actions[:2]) or "n/a"
            rows += (
                "<tr>"
                f"<td>{self._escape_html(str(item.get('issue_type', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('title', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('confidence', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('regression_risk', '')))}</td>"
                f"<td>{actions_text}</td>"
                "</tr>"
            )
        return (
            "<table><tr><th>Issue Type</th><th>Title</th><th>Confidence</th><th>Regression Risk</th><th>Recommended Actions</th></tr>"
            + rows
            + "</table>"
        )

    def _build_recommendations_input(
        self,
        *,
        run_scorecard: dict,
        duplicate_summary: dict,
        coverage_summary: dict,
        dev_summary: dict,
        classifier_performance_summary: dict,
        runtime_summary: dict,
        llm_cost_summary: dict,
        domain_evaluation_summary: dict,
        run_delta_summary: dict,
    ) -> dict:
        """Build a structured recommendations_input section for Codex review."""
        guardrails = run_scorecard.get("guardrails", {}) if isinstance(run_scorecard.get("guardrails"), dict) else {}
        violations = guardrails.get("violations", []) if isinstance(guardrails.get("violations"), list) else []
        top_regressions = [str(item.get("detail") or "") for item in violations[:5] if str(item.get("detail") or "").strip()]

        largest_cost_deltas: list[dict[str, Any]] = []
        by_provider = llm_cost_summary.get("by_provider", {}) if isinstance(llm_cost_summary.get("by_provider"), dict) else {}
        for provider_key, value in by_provider.items():
            if value is None:
                continue
            largest_cost_deltas.append({"bucket": provider_key, "usd": float(value)})
        largest_cost_deltas.sort(key=lambda item: float(item.get("usd", 0.0)), reverse=True)

        stage_details = classifier_performance_summary.get("stage_details", []) if isinstance(classifier_performance_summary.get("stage_details"), list) else []
        classifier_regressions = []
        for item in stage_details:
            if not isinstance(item, dict):
                continue
            classifier_regressions.append(
                {
                    "stage": str(item.get("stage") or ""),
                    "replay_url_accuracy_pct": item.get("replay_url_accuracy_pct"),
                    "replay_url_count": item.get("replay_url_count"),
                }
            )

        duplicate_hotspots = duplicate_summary.get("top_duplicate_domains", []) if isinstance(duplicate_summary.get("top_duplicate_domains"), list) else []
        coverage_gaps = coverage_summary.get("missed_sources", []) if isinstance(coverage_summary.get("missed_sources"), list) else []
        domain_regressions = domain_evaluation_summary.get("worst_domains", []) if isinstance(domain_evaluation_summary.get("worst_domains"), list) else []
        dev_info = dev_summary if isinstance(dev_summary, dict) else {}
        previous_run_regressions = (((run_delta_summary.get("previous_run") or {}).get("top_regressions")) or []) if isinstance(run_delta_summary, dict) else []
        holdout_baseline_regressions = (((run_delta_summary.get("holdout_baseline") or {}).get("top_regressions")) or []) if isinstance(run_delta_summary, dict) else []
        bottlenecks = []
        for span in runtime_summary.get("step_spans", []) if isinstance(runtime_summary.get("step_spans"), list) else []:
            if not isinstance(span, dict):
                continue
            bottlenecks.append(
                {
                    "step": str(span.get("log_file") or ""),
                    "duration_minutes": float(span.get("duration_minutes", 0.0) or 0.0),
                }
            )
        bottlenecks.sort(key=lambda item: float(item.get("duration_minutes", 0.0)), reverse=True)

        return {
            "top_regressions": top_regressions,
            "largest_cost_deltas": largest_cost_deltas[:5],
            "runtime_bottlenecks": bottlenecks[:5],
            "duplicate_hotspots": duplicate_hotspots[:5],
            "coverage_gaps": coverage_gaps[:5],
            "domain_regressions": domain_regressions[:5],
            "previous_run_regressions": previous_run_regressions[:5],
            "holdout_baseline_regressions": holdout_baseline_regressions[:5],
            "dev_replay_summary": {
                "replay_url_accuracy_pct": dev_info.get("replay_url_accuracy_pct"),
                "matched_urls": dev_info.get("matched_urls"),
                "mismatched_urls": dev_info.get("mismatched_urls"),
                "replay_urls_seen": dev_info.get("replay_urls_seen"),
            },
            "classifier_routing_regressions": classifier_regressions[:5],
        }

    def _build_codex_review_bundle(
        self,
        *,
        run_scorecard: dict,
        accuracy_replay_summary: dict,
        classifier_performance_summary: dict,
        duplicate_summary: dict,
        coverage_summary: dict,
        dev_summary: dict,
        holdout_summary: dict,
        domain_capped_summary: dict,
        domain_evaluation_summary: dict,
        run_delta_summary: dict,
        recommendation_plan: dict,
    ) -> dict:
        """Build a stable machine-readable review bundle for Codex analysis."""
        return {
            "bundle_version": "v1",
            "generated_at": datetime.utcnow().isoformat(),
            "run_id": str(run_scorecard.get("run_id", "") or ""),
            "overall_status": str(run_scorecard.get("overall_score", {}).get("status", "") or ""),
            "guardrails": run_scorecard.get("guardrails", {}) if isinstance(run_scorecard.get("guardrails"), dict) else {},
            "recommendations_input": run_scorecard.get("recommendations_input", {}) if isinstance(run_scorecard.get("recommendations_input"), dict) else {},
            "artifacts": {
                "run_scorecard": run_scorecard,
                "accuracy_replay_summary": accuracy_replay_summary,
                "classifier_performance_summary": classifier_performance_summary,
                "duplicate_audit_summary": duplicate_summary,
                "coverage_summary": coverage_summary,
                "dev_summary": dev_summary,
                "holdout_summary": holdout_summary,
                "domain_capped_summary": domain_capped_summary,
                "domain_evaluation_summary": domain_evaluation_summary,
                "run_delta_summary": run_delta_summary,
                "recommendation_plan": recommendation_plan,
            },
        }

    def _load_recent_validation_artifacts(
        self,
        *,
        artifact_type: str,
        limit: int = 10,
        exclude_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load recent validation artifacts from the DB when available."""
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "execute_query"):
            return []
        safe_artifact_type = str(artifact_type or "").strip()
        if not safe_artifact_type:
            return []
        safe_limit = max(1, int(limit or 1))
        sql = """
            SELECT run_id, artifact_json, updated_at
            FROM validation_run_artifacts
            WHERE artifact_type = :artifact_type
              AND (:exclude_run_id IS NULL OR run_id <> :exclude_run_id)
            ORDER BY updated_at DESC, created_at DESC
            LIMIT :limit
        """
        try:
            rows = self.db_handler.execute_query(
                sql,
                {
                    "artifact_type": safe_artifact_type,
                    "exclude_run_id": str(exclude_run_id or "").strip() or None,
                    "limit": safe_limit,
                },
            )
        except Exception:
            return []
        artifacts: list[dict[str, Any]] = []
        for row in rows or []:
            if not row or len(row) < 2:
                continue
            payload = row[1]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            payload.setdefault("run_id", str(row[0] or ""))
            artifacts.append(payload)
        return artifacts

    def _get_code_version_info(self) -> dict[str, str]:
        """Return lightweight Git provenance for the current working tree when available."""
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            git_commit = ""
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            branch = ""
        return {
            "git_commit": git_commit,
            "branch": branch,
        }

    def _metric_delta(
        self,
        *,
        metric_key: str,
        label: str,
        current_value: Any,
        baseline_value: Any,
        higher_is_better: bool,
    ) -> dict[str, Any] | None:
        """Build one normalized metric delta record."""
        if current_value is None or baseline_value is None:
            return None
        try:
            current_numeric = float(current_value)
            baseline_numeric = float(baseline_value)
        except (TypeError, ValueError):
            return None
        delta = round(current_numeric - baseline_numeric, 4)
        direction = "unchanged"
        if delta > 0:
            direction = "improved" if higher_is_better else "regressed"
        elif delta < 0:
            direction = "regressed" if higher_is_better else "improved"
        return {
            "metric_key": metric_key,
            "label": label,
            "current_value": current_numeric,
            "baseline_value": baseline_numeric,
            "delta": delta,
            "higher_is_better": higher_is_better,
            "direction": direction,
        }

    def _extract_scorecard_comparison_metrics(self, run_scorecard: dict) -> list[dict[str, Any]]:
        """Extract comparable scorecard metrics for run-to-run deltas."""
        kpis = run_scorecard.get("kpis", {}) if isinstance(run_scorecard.get("kpis"), dict) else {}
        db_accuracy = (kpis.get("database_accuracy") or {}).get("summary", {})
        coverage = (kpis.get("events_coverage") or {}).get("summary", {})
        runtime = (kpis.get("run_time") or {}).get("summary", {})
        cost_summary = ((kpis.get("run_costs") or {}).get("summary") or {}).get("summary", {})
        chatbot = ((kpis.get("chatbot_quality") or {}).get("summary") or {}).get("summary", {})
        dev = (run_scorecard.get("evaluation_scope") or {}).get("dev_summary", {})
        holdout = (run_scorecard.get("evaluation_scope") or {}).get("holdout_summary", {})
        return [
            {"metric_key": "overall_score", "label": "Overall Score", "value": (run_scorecard.get("overall_score") or {}).get("value"), "higher_is_better": True},
            {"metric_key": "replay_url_accuracy_pct", "label": "Replay URL Accuracy %", "value": db_accuracy.get("replay_url_accuracy_pct"), "higher_is_better": True},
            {"metric_key": "duplicate_rate_per_100_events", "label": "Duplicate Rate / 100 Events", "value": db_accuracy.get("duplicate_rate_per_100_events"), "higher_is_better": False},
            {"metric_key": "severe_duplicate_rate_per_100_events", "label": "Severe Duplicate Rate / 100 Events", "value": db_accuracy.get("severe_duplicate_rate_per_100_events"), "higher_is_better": False},
            {"metric_key": "invalid_event_rate_pct", "label": "Invalid Event Rate %", "value": db_accuracy.get("invalid_event_rate_pct"), "higher_is_better": False},
            {"metric_key": "stale_event_rate_pct", "label": "Stale Event Rate %", "value": db_accuracy.get("stale_event_rate_pct"), "higher_is_better": False},
            {"metric_key": "watchlist_source_hit_rate_pct", "label": "Watchlist Source Hit Rate %", "value": coverage.get("watchlist_source_hit_rate_pct"), "higher_is_better": True},
            {"metric_key": "watchlist_event_capture_rate_pct", "label": "Watchlist Event Capture Rate %", "value": coverage.get("watchlist_event_capture_rate_pct"), "higher_is_better": True},
            {"metric_key": "missed_event_rate_manual_audit_pct", "label": "Manual Audit Missed Event Rate %", "value": coverage.get("missed_event_rate_manual_audit_pct"), "higher_is_better": False},
            {"metric_key": "pipeline_duration_minutes", "label": "Pipeline Duration Minutes", "value": runtime.get("pipeline_duration_minutes"), "higher_is_better": False},
            {"metric_key": "urls_processed_per_minute", "label": "URLs Processed / Minute", "value": runtime.get("urls_processed_per_minute"), "higher_is_better": True},
            {"metric_key": "events_inserted_per_minute", "label": "Events Inserted / Minute", "value": runtime.get("events_inserted_per_minute"), "higher_is_better": True},
            {"metric_key": "total_llm_cost_usd", "label": "Total LLM Cost USD", "value": cost_summary.get("total_usd"), "higher_is_better": False},
            {"metric_key": "cost_per_processed_url_usd", "label": "Cost / Processed URL USD", "value": cost_summary.get("cost_per_processed_url_usd"), "higher_is_better": False},
            {"metric_key": "cost_per_inserted_event_usd", "label": "Cost / Inserted Event USD", "value": cost_summary.get("cost_per_inserted_event_usd"), "higher_is_better": False},
            {"metric_key": "chatbot_response_within_15s_pct", "label": "Chatbot <=15s %", "value": chatbot.get("chatbot_response_within_15s_pct"), "higher_is_better": True},
            {"metric_key": "chatbot_answer_correctness_pct", "label": "Chatbot Correctness %", "value": chatbot.get("chatbot_answer_correctness_pct"), "higher_is_better": True},
            {"metric_key": "dev_replay_url_accuracy_pct", "label": "Dev Replay URL Accuracy %", "value": dev.get("replay_url_accuracy_pct"), "higher_is_better": True},
            {"metric_key": "holdout_replay_url_accuracy_pct", "label": "Holdout Replay URL Accuracy %", "value": holdout.get("replay_url_accuracy_pct"), "higher_is_better": True},
        ]

    def _compare_scorecards(self, current_run_scorecard: dict, baseline_run_scorecard: dict) -> dict:
        """Compare two run scorecards and summarize the KPI deltas."""
        current_metrics = {
            str(item.get("metric_key") or ""): item
            for item in self._extract_scorecard_comparison_metrics(current_run_scorecard)
            if str(item.get("metric_key") or "").strip()
        }
        baseline_metrics = {
            str(item.get("metric_key") or ""): item
            for item in self._extract_scorecard_comparison_metrics(baseline_run_scorecard)
            if str(item.get("metric_key") or "").strip()
        }
        metric_deltas: list[dict[str, Any]] = []
        for metric_key, current_metric in current_metrics.items():
            baseline_metric = baseline_metrics.get(metric_key)
            if not baseline_metric:
                continue
            delta = self._metric_delta(
                metric_key=metric_key,
                label=str(current_metric.get("label") or metric_key),
                current_value=current_metric.get("value"),
                baseline_value=baseline_metric.get("value"),
                higher_is_better=bool(current_metric.get("higher_is_better")),
            )
            if delta is not None:
                metric_deltas.append(delta)
        improvements = [item for item in metric_deltas if item.get("direction") == "improved"]
        regressions = [item for item in metric_deltas if item.get("direction") == "regressed"]
        improvements.sort(key=lambda item: abs(float(item.get("delta", 0.0) or 0.0)), reverse=True)
        regressions.sort(key=lambda item: abs(float(item.get("delta", 0.0) or 0.0)), reverse=True)
        current_holdout = ((current_run_scorecard.get("evaluation_scope") or {}).get("holdout_summary") or {}).get("replay_url_accuracy_pct")
        baseline_holdout = ((baseline_run_scorecard.get("evaluation_scope") or {}).get("holdout_summary") or {}).get("replay_url_accuracy_pct")
        holdout_delta = None
        if current_holdout is not None and baseline_holdout is not None:
            holdout_delta = round(float(current_holdout) - float(baseline_holdout), 4)
        delta_overall_score = None
        if (current_run_scorecard.get("overall_score") or {}).get("value") is not None and (baseline_run_scorecard.get("overall_score") or {}).get("value") is not None:
            delta_overall_score = round(
                float((current_run_scorecard.get("overall_score") or {}).get("value"))
                - float((baseline_run_scorecard.get("overall_score") or {}).get("value")),
                4,
            )
        return {
            "baseline_run_id": str(baseline_run_scorecard.get("run_id", "") or ""),
            "baseline_run_timestamp_utc": str(baseline_run_scorecard.get("run_timestamp_utc", "") or ""),
            "summary": {
                "delta_overall_score": delta_overall_score,
                "delta_holdout_replay_url_accuracy_pct": holdout_delta,
                "metric_delta_count": len(metric_deltas),
            },
            "top_improvements": improvements[:5],
            "top_regressions": regressions[:5],
            "metric_deltas": metric_deltas,
        }

    def _build_run_delta_summary(self, current_run_scorecard: dict) -> dict:
        """Build a structured previous-run and holdout-baseline comparison artifact."""
        safe_current_run_id = str(current_run_scorecard.get("run_id", "") or "").strip()
        prior_scorecards = self._load_recent_validation_artifacts(
            artifact_type="run_scorecard",
            limit=10,
            exclude_run_id=safe_current_run_id,
        )
        previous_run_scorecard = prior_scorecards[0] if prior_scorecards else None
        holdout_baseline_scorecard = next(
            (
                artifact
                for artifact in prior_scorecards
                if ((artifact.get("evaluation_scope") or {}).get("holdout_summary") or {}).get("replay_url_accuracy_pct") is not None
            ),
            None,
        )
        summary: dict[str, Any] = {
            "available": bool(previous_run_scorecard or holdout_baseline_scorecard),
            "current_run_id": safe_current_run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "previous_run": {
                "available": bool(previous_run_scorecard),
            },
            "holdout_baseline": {
                "available": bool(holdout_baseline_scorecard),
            },
        }
        if previous_run_scorecard:
            previous_run_comparison = self._compare_scorecards(current_run_scorecard, previous_run_scorecard)
            previous_run_comparison["available"] = True
            summary["previous_run"] = previous_run_comparison
        if holdout_baseline_scorecard:
            holdout_baseline_comparison = self._compare_scorecards(current_run_scorecard, holdout_baseline_scorecard)
            holdout_baseline_comparison["available"] = True
            summary["holdout_baseline"] = holdout_baseline_comparison
        return summary

    def _summarize_domain_evaluation(self, accuracy_replay_summary: dict | None) -> dict:
        """Summarize URL-level replay accuracy by domain."""
        rows = accuracy_replay_summary.get("rows", []) if isinstance(accuracy_replay_summary, dict) else []
        by_domain: dict[str, dict[str, Any]] = {}
        seen_urls: set[str] = set()
        for row in rows:
            baseline = row.get("baseline", {}) if isinstance(row, dict) else {}
            url = normalize_evaluation_url(baseline.get("url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            domain = (urlparse(url).netloc or "").lower() or "(no domain)"
            entry = by_domain.setdefault(
                domain,
                {"domain": domain, "url_count": 0, "matched_urls": 0, "mismatch_categories": Counter()},
            )
            entry["url_count"] += 1
            if bool(row.get("is_match")):
                entry["matched_urls"] += 1
            else:
                entry["mismatch_categories"].update([str(row.get("mismatch_category") or "unknown")])

        domain_rows: list[dict[str, Any]] = []
        for entry in by_domain.values():
            mismatch_categories = entry.get("mismatch_categories", Counter())
            url_count = int(entry.get("url_count", 0) or 0)
            matched_urls = int(entry.get("matched_urls", 0) or 0)
            domain_rows.append(
                {
                    "domain": str(entry.get("domain", "")),
                    "url_count": url_count,
                    "matched_urls": matched_urls,
                    "replay_url_accuracy_pct": round((matched_urls / url_count) * 100.0, 2) if url_count else None,
                    "top_mismatch_category": mismatch_categories.most_common(1)[0][0] if mismatch_categories else "",
                    "mismatch_categories": dict(mismatch_categories),
                }
            )

        top_domains = sorted(domain_rows, key=lambda item: (-int(item.get("url_count", 0) or 0), str(item.get("domain", ""))))
        worst_domains = sorted(
            [item for item in domain_rows if item.get("replay_url_accuracy_pct") is not None],
            key=lambda item: (float(item.get("replay_url_accuracy_pct", 100.0)), -int(item.get("url_count", 0) or 0)),
        )
        return {
            "available": True,
            "domain_count": len(domain_rows),
            "total_urls": len(seen_urls),
            "top_domains": top_domains[:20],
            "worst_domains": worst_domains[:20],
        }

    def _summarize_dev_replay(self, accuracy_replay_summary: dict | None) -> dict:
        """Summarize replay results restricted to the dev URL set."""
        dev_urls = load_dev_urls()
        summary = {
            "available": bool(dev_urls),
            "dev_version": "v1" if dev_urls else "",
            "dev_urls_total": len(dev_urls),
            "replay_urls_seen": 0,
            "matched_urls": 0,
            "mismatched_urls": 0,
            "replay_url_accuracy_pct": None,
            "mismatch_categories": {},
            "sample_urls": [],
            "error": "",
        }
        if not dev_urls:
            summary["error"] = "dev urls unavailable"
            return summary

        rows = accuracy_replay_summary.get("rows", []) if isinstance(accuracy_replay_summary, dict) else []
        matched_rows: list[dict[str, Any]] = []
        mismatch_categories: Counter[str] = Counter()
        seen_urls: set[str] = set()
        matched_urls = 0

        for row in rows:
            baseline = row.get("baseline", {}) if isinstance(row, dict) else {}
            url = normalize_evaluation_url(baseline.get("url"))
            if not url or url not in dev_urls or url in seen_urls:
                continue
            seen_urls.add(url)
            is_match = bool(row.get("is_match"))
            if is_match:
                matched_urls += 1
            else:
                mismatch_categories.update([str(row.get("mismatch_category") or "unknown")])
            matched_rows.append(
                {
                    "url": url,
                    "is_match": is_match,
                    "category": str(row.get("mismatch_category") or ""),
                    "details": str(row.get("details") or ""),
                }
            )

        replay_urls_seen = len(seen_urls)
        summary.update(
            {
                "replay_urls_seen": replay_urls_seen,
                "matched_urls": matched_urls,
                "mismatched_urls": max(replay_urls_seen - matched_urls, 0),
                "replay_url_accuracy_pct": round((matched_urls / replay_urls_seen) * 100.0, 2) if replay_urls_seen else None,
                "mismatch_categories": dict(mismatch_categories),
                "sample_urls": matched_rows[:20],
            }
        )
        return summary

    def _summarize_holdout_replay(self, accuracy_replay_summary: dict | None) -> dict:
        """Summarize replay results restricted to the gold holdout URL set."""
        holdout_urls = load_gold_holdout_urls()
        summary = {
            "available": bool(holdout_urls),
            "holdout_version": "v1" if holdout_urls else "",
            "holdout_urls_total": len(holdout_urls),
            "replay_urls_seen": 0,
            "matched_urls": 0,
            "mismatched_urls": 0,
            "replay_url_accuracy_pct": None,
            "mismatch_categories": {},
            "sample_urls": [],
            "error": "",
        }
        if not holdout_urls:
            summary["error"] = "gold holdout unavailable"
            return summary

        rows = accuracy_replay_summary.get("rows", []) if isinstance(accuracy_replay_summary, dict) else []
        matched_rows: list[dict[str, Any]] = []
        mismatch_categories: Counter[str] = Counter()
        seen_urls: set[str] = set()
        matched_urls = 0

        for row in rows:
            baseline = row.get("baseline", {}) if isinstance(row, dict) else {}
            url = normalize_evaluation_url(baseline.get("url"))
            if not url or url not in holdout_urls or url in seen_urls:
                continue
            seen_urls.add(url)
            is_match = bool(row.get("is_match"))
            if is_match:
                matched_urls += 1
            else:
                mismatch_categories.update([str(row.get("mismatch_category") or "unknown")])
            matched_rows.append(
                {
                    "url": url,
                    "is_match": is_match,
                    "mismatch_category": str(row.get("mismatch_category") or ""),
                    "baseline_event_id": row.get("baseline_event_id"),
                }
            )

        replay_urls_seen = len(seen_urls)
        summary.update(
            {
                "replay_urls_seen": replay_urls_seen,
                "matched_urls": matched_urls,
                "mismatched_urls": max(replay_urls_seen - matched_urls, 0),
                "replay_url_accuracy_pct": round((matched_urls / replay_urls_seen) * 100.0, 2) if replay_urls_seen else None,
                "mismatch_categories": dict(mismatch_categories),
                "sample_urls": matched_rows[:20],
            }
        )
        return summary

    def _summarize_domain_capped_replay(
        self,
        accuracy_replay_summary: dict | None,
        *,
        per_domain_cap: int = 3,
    ) -> dict:
        """Summarize replay accuracy with a per-domain sample cap to reduce template dominance."""
        rows = accuracy_replay_summary.get("rows", []) if isinstance(accuracy_replay_summary, dict) else []
        cap = max(1, int(per_domain_cap or 1))
        per_domain_counts: Counter[str] = Counter()
        sampled_rows: list[dict[str, Any]] = []
        sampled_match_count = 0
        seen_urls: set[str] = set()

        for row in rows:
            baseline = row.get("baseline", {}) if isinstance(row, dict) else {}
            url = normalize_evaluation_url(baseline.get("url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            domain = (urlparse(url).netloc or "").lower()
            if not domain:
                domain = "(no domain)"
            if per_domain_counts[domain] >= cap:
                continue
            per_domain_counts[domain] += 1
            is_match = bool(row.get("is_match"))
            if is_match:
                sampled_match_count += 1
            sampled_rows.append(
                {
                    "url": url,
                    "domain": domain,
                    "is_match": is_match,
                    "mismatch_category": str(row.get("mismatch_category") or ""),
                }
            )

        sampled_total = len(sampled_rows)
        return {
            "available": True,
            "per_domain_cap": cap,
            "sampled_rows": sampled_total,
            "sampled_domains": len(per_domain_counts),
            "replay_url_accuracy_pct": round((sampled_match_count / sampled_total) * 100.0, 2) if sampled_total else None,
            "domain_counts": dict(per_domain_counts),
            "sample_rows": sampled_rows[:20],
        }

    def _evaluate_phase3_guardrails(self, run_scorecard: dict) -> dict:
        """Evaluate explicit KPI guardrails and return pass/fail status with violations."""
        kpis = run_scorecard.get("kpis", {}) if isinstance(run_scorecard.get("kpis"), dict) else {}
        db_accuracy = (kpis.get("database_accuracy") or {}).get("summary", {})
        coverage = (kpis.get("events_coverage") or {}).get("summary", {})
        chatbot = ((kpis.get("chatbot_quality") or {}).get("summary") or {}).get("summary", {})
        holdout = (run_scorecard.get("evaluation_scope") or {}).get("holdout_summary", {})
        guardrail_specs = [
            (
                "database_accuracy_min_pct",
                db_accuracy.get("replay_url_accuracy_pct"),
                75.0,
                "min",
                "Replay URL accuracy below minimum",
            ),
            (
                "holdout_replay_url_accuracy_min_pct",
                holdout.get("replay_url_accuracy_pct"),
                70.0,
                "min",
                "Holdout replay URL accuracy below minimum",
            ),
            (
                "coverage_watchlist_source_hit_rate_min_pct",
                coverage.get("watchlist_source_hit_rate_pct"),
                60.0,
                "min",
                "Watchlist source hit rate below minimum",
            ),
            (
                "events_coverage_min_pct",
                coverage.get("watchlist_event_capture_rate_pct"),
                65.0,
                "min",
                "Watchlist event capture rate below minimum",
            ),
            (
                "severe_duplicate_rate_max_per_100_events",
                db_accuracy.get("severe_duplicate_rate_per_100_events"),
                2.0,
                "max",
                "Severe duplicate rate above maximum",
            ),
            (
                "stale_event_rate_max_pct",
                db_accuracy.get("stale_event_rate_pct"),
                5.0,
                "max",
                "Stale event rate above maximum",
            ),
            (
                "chatbot_response_within_15s_min_pct",
                chatbot.get("chatbot_response_within_15s_pct"),
                90.0,
                "min",
                "Chatbot 15s latency target missed",
            ),
            (
                "chatbot_answer_correctness_min_pct",
                chatbot.get("chatbot_answer_correctness_pct"),
                85.0,
                "min",
                "Chatbot correctness below minimum",
            ),
            (
                "chatbot_user_visible_error_rate_max_pct",
                chatbot.get("chatbot_user_visible_error_rate_pct"),
                3.0,
                "max",
                "Chatbot user-visible error rate above maximum",
            ),
        ]
        violations: list[dict[str, Any]] = []
        thresholds: dict[str, float] = {}
        for metric_key, actual_value, threshold, mode, detail in guardrail_specs:
            thresholds[metric_key] = threshold
            if actual_value is None:
                continue
            actual = float(actual_value)
            failed = actual < threshold if mode == "min" else actual > threshold
            if failed:
                violations.append(
                    {
                        "metric_key": metric_key,
                        "actual": round(actual, 2),
                        "threshold": threshold,
                        "mode": mode,
                        "detail": detail,
                    }
                )
        return {
            **thresholds,
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
        }

    @staticmethod
    def _parse_log_timestamp(line: str) -> datetime | None:
        """Parse timestamp prefix in log lines (YYYY-MM-DD HH:MM:SS)."""
        try:
            return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    @staticmethod
    def _get_llm_activity_log_files() -> list[str]:
        """Return log files used for LLM activity/reliability summarization."""
        return [
            "logs/emails_log.txt",
            "logs/gs_log.txt",
            "logs/rd_ext_log.txt",
            "logs/ebs_log.txt",
            "logs/scraper_log.txt",
            "logs/fb_log.txt",
            "logs/images_log.txt",
            "logs/read_pdfs_log.txt",
            "logs/db_log.txt",
            "logs/clean_up_log.txt",
            "logs/dedup_llm_log.txt",
            "logs/irrelevant_rows_log.txt",
            "logs/validation_tests_log.txt",
        ]

    @staticmethod
    def _parse_llm_query_provider_model(line: str) -> tuple[str, str] | None:
        """Extract provider/model from any LLM query log line shape."""
        low = line.lower()
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            marker = f"querying {provider} model "
            idx = low.find(marker)
            if idx >= 0:
                model = line[idx + len(marker):].strip().split()[0] if line[idx + len(marker):].strip() else ""
                return provider, model
        return None

    def _infer_latest_pipeline_run_id(self, report_timestamp: str | None) -> str:
        """Return latest pipeline run_id from logs/pipeline_log.txt near report timestamp."""
        path = "logs/pipeline_log.txt"
        if not os.path.exists(path):
            return ""

        end_ts = datetime.now()
        if report_timestamp:
            parsed = self._parse_iso_datetime(report_timestamp)
            if parsed is not None:
                end_ts = parsed
        start_ts = end_ts - timedelta(hours=48)
        run_re = re.compile(r"\[run_id=([^\]]+)\]")
        latest_run_id = ""
        latest_ts: datetime | None = None

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    ts = self._parse_log_timestamp(line)
                    if ts is None or ts < start_ts or ts > end_ts:
                        continue
                    m = run_re.search(line)
                    if not m:
                        continue
                    if latest_ts is None or ts >= latest_ts:
                        latest_ts = ts
                        latest_run_id = m.group(1).strip()
        except Exception:
            return ""
        return latest_run_id

    def _summarize_pipeline_runtime(self, report_timestamp: str | None, run_id: str) -> dict:
        """Approximate end-to-end runtime using top-level log timestamps for one run_id."""
        summary = {
            "available": False,
            "run_id": str(run_id or ""),
            "start_ts": "",
            "end_ts": "",
            "runtime_seconds": 0.0,
            "runtime_hours": 0.0,
            "step_log_spans": [],
            "error": "",
        }
        if not run_id:
            summary["error"] = "run_id unavailable for runtime summarization"
            return summary

        log_dir = "logs"
        if not os.path.isdir(log_dir):
            summary["error"] = "logs directory not found"
            return summary

        end_ts = datetime.now()
        if report_timestamp:
            parsed = self._parse_iso_datetime(report_timestamp)
            if parsed is not None:
                end_ts = parsed
        start_window = end_ts - timedelta(days=3)

        run_marker = f"[run_id={run_id}]"
        global_start: datetime | None = None
        global_end: datetime | None = None
        spans: list[dict] = []

        for name in sorted(os.listdir(log_dir)):
            if not name.endswith("_log.txt"):
                continue
            path = os.path.join(log_dir, name)
            if not os.path.isfile(path):
                continue
            file_start: datetime | None = None
            file_end: datetime | None = None
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        if run_marker not in line:
                            continue
                        ts = self._parse_log_timestamp(line)
                        if ts is None or ts < start_window or ts > end_ts:
                            continue
                        if file_start is None or ts < file_start:
                            file_start = ts
                        if file_end is None or ts > file_end:
                            file_end = ts
                if file_start is None or file_end is None:
                    continue
                duration_s = max(0.0, (file_end - file_start).total_seconds())
                spans.append(
                    {
                        "log_file": name,
                        "start_ts": file_start.isoformat(sep=" "),
                        "end_ts": file_end.isoformat(sep=" "),
                        "duration_seconds": round(duration_s, 1),
                        "duration_minutes": round(duration_s / 60.0, 2),
                    }
                )
                if global_start is None or file_start < global_start:
                    global_start = file_start
                if global_end is None or file_end > global_end:
                    global_end = file_end
            except Exception:
                continue

        if global_start is None or global_end is None:
            summary["error"] = f"no timestamps found for run_id={run_id}"
            return summary

        runtime_seconds = max(0.0, (global_end - global_start).total_seconds())
        spans.sort(key=lambda row: float(row.get("duration_seconds", 0.0) or 0.0), reverse=True)
        summary.update(
            {
                "available": True,
                "start_ts": global_start.isoformat(sep=" "),
                "end_ts": global_end.isoformat(sep=" "),
                "runtime_seconds": round(runtime_seconds, 1),
                "runtime_hours": round(runtime_seconds / 3600.0, 3),
                "step_log_spans": spans[:12],
            }
        )
        return summary

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert numeric-ish values to float safely."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """Convert numeric-ish values to int safely."""
        try:
            if value is None:
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _extract_openrouter_activity_totals(self, payload: Any) -> dict:
        """
        Best-effort parser for OpenRouter activity/usage payload shapes.
        Returns aggregate spend/requests/tokens if present.
        """
        totals = {"cost_usd": None, "requests": None, "tokens": None}
        if not isinstance(payload, (dict, list)):
            return totals

        def _sum_rows(rows: list[dict]) -> tuple[float | None, int | None, int | None]:
            cost_sum = 0.0
            req_sum = 0
            tok_sum = 0
            cost_seen = False
            req_seen = False
            tok_seen = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for k in ("cost", "spend", "total_cost", "usd", "usage"):
                    v = self._safe_float(row.get(k))
                    if v is not None:
                        cost_sum += v
                        cost_seen = True
                        break
                for k in ("requests", "request_count", "num_requests"):
                    v = self._safe_int(row.get(k))
                    if v is not None:
                        req_sum += v
                        req_seen = True
                        break
                token_candidates = [
                    self._safe_int(row.get("tokens")),
                    self._safe_int(row.get("total_tokens")),
                    self._safe_int(row.get("input_tokens")),
                    self._safe_int(row.get("output_tokens")),
                    self._safe_int(row.get("prompt_tokens")),
                    self._safe_int(row.get("completion_tokens")),
                    self._safe_int(row.get("reasoning_tokens")),
                ]
                token_values = [v for v in token_candidates if v is not None]
                if token_values:
                    tok_sum += sum(token_values)
                    tok_seen = True
            return (
                round(cost_sum, 6) if cost_seen else None,
                req_sum if req_seen else None,
                tok_sum if tok_seen else None,
            )

        if isinstance(payload, dict):
            for key in ("data", "results", "activity", "usage"):
                if isinstance(payload.get(key), list):
                    cost, reqs, toks = _sum_rows(payload.get(key) or [])
                    if cost is not None or reqs is not None or toks is not None:
                        totals["cost_usd"] = cost
                        totals["requests"] = reqs
                        totals["tokens"] = toks
                        return totals

            # Fallback direct keys
            for k in ("cost", "spend", "total_cost", "total_spend", "usd_spend", "usage"):
                v = self._safe_float(payload.get(k))
                if v is not None:
                    totals["cost_usd"] = v
                    break
            for k in ("requests", "request_count", "num_requests"):
                v = self._safe_int(payload.get(k))
                if v is not None:
                    totals["requests"] = v
                    break
            tokens_parts = [
                self._safe_int(payload.get("tokens")),
                self._safe_int(payload.get("total_tokens")),
                self._safe_int(payload.get("prompt_tokens")),
                self._safe_int(payload.get("completion_tokens")),
                self._safe_int(payload.get("reasoning_tokens")),
            ]
            token_values = [v for v in tokens_parts if v is not None]
            if token_values:
                totals["tokens"] = sum(token_values)
            return totals

        if isinstance(payload, list):
            cost, reqs, toks = _sum_rows(payload)
            totals["cost_usd"] = cost
            totals["requests"] = reqs
            totals["tokens"] = toks
        return totals

    def _apply_snapshot_delta_cost(self, provider: str, summary: dict, output_dir: str | None) -> dict:
        """
        Convert cumulative provider totals into per-run deltas using local snapshots.

        If no previous snapshot exists, keep raw totals and mark basis accordingly.
        """
        provider_norm = str(provider or "").strip().lower()
        if provider_norm not in {"openrouter", "openai"}:
            return summary

        raw_cost = self._safe_float(summary.get("cost_usd"))
        raw_requests = self._safe_int(summary.get("requests"))
        raw_tokens = self._safe_int(summary.get("tokens"))

        summary["raw_cost_usd"] = raw_cost
        summary["raw_requests"] = raw_requests
        summary["raw_tokens"] = raw_tokens

        if raw_cost is None and raw_requests is None and raw_tokens is None:
            summary["cost_basis"] = "unavailable"
            return summary

        run_start_ts = self._parse_iso_datetime(str(summary.get("start_ts", "") or ""))
        run_end_ts = self._parse_iso_datetime(str(summary.get("end_ts", "") or ""))
        run_id = str(os.getenv("DS_RUN_ID", "")).strip() or "validation-report"

        # Preferred path: normalized DB metrics (metric_definitions + metric_observations).
        db_supported = bool(
            getattr(self, "db_handler", None)
            and hasattr(self.db_handler, "execute_query")
            and hasattr(self.db_handler, "record_metric_observation")
        )
        if db_supported:
            metric_keys = {
                "raw_cost": f"{provider_norm}_cumulative_cost_usd",
                "raw_requests": f"{provider_norm}_cumulative_requests",
                "raw_tokens": f"{provider_norm}_cumulative_tokens",
                "run_cost": f"{provider_norm}_run_cost_usd",
                "run_requests": f"{provider_norm}_run_requests",
                "run_tokens": f"{provider_norm}_run_tokens",
            }
            prev_cost = None
            prev_requests = None
            prev_tokens = None
            prev_ts_text = ""

            if run_start_ts is not None:
                try:
                    prev_rows = self.db_handler.execute_query(
                        """
                        SELECT mo.window_start, mo.metric_value_numeric
                        FROM metric_observations mo
                        JOIN metric_definitions md ON md.metric_id = mo.metric_id
                        WHERE md.metric_key = :metric_key
                          AND mo.window_start IS NOT NULL
                          AND mo.window_start < :run_start_ts
                        ORDER BY mo.window_start DESC, mo.created_at DESC
                        LIMIT 1
                        """,
                        {
                            "metric_key": metric_keys["raw_cost"],
                            "run_start_ts": run_start_ts,
                        },
                    ) or []
                    if prev_rows:
                        prev_ts = prev_rows[0][0]
                        prev_ts_text = prev_ts.isoformat() if hasattr(prev_ts, "isoformat") else str(prev_ts)
                        prev_cost = self._safe_float(prev_rows[0][1])
                except Exception:
                    prev_cost = None
                    prev_ts_text = ""
                try:
                    prev_rows = self.db_handler.execute_query(
                        """
                        SELECT mo.metric_value_numeric
                        FROM metric_observations mo
                        JOIN metric_definitions md ON md.metric_id = mo.metric_id
                        WHERE md.metric_key = :metric_key
                          AND mo.window_start IS NOT NULL
                          AND mo.window_start < :run_start_ts
                        ORDER BY mo.window_start DESC, mo.created_at DESC
                        LIMIT 1
                        """,
                        {
                            "metric_key": metric_keys["raw_requests"],
                            "run_start_ts": run_start_ts,
                        },
                    ) or []
                    if prev_rows:
                        prev_requests = self._safe_int(prev_rows[0][0])
                except Exception:
                    prev_requests = None
                try:
                    prev_rows = self.db_handler.execute_query(
                        """
                        SELECT mo.metric_value_numeric
                        FROM metric_observations mo
                        JOIN metric_definitions md ON md.metric_id = mo.metric_id
                        WHERE md.metric_key = :metric_key
                          AND mo.window_start IS NOT NULL
                          AND mo.window_start < :run_start_ts
                        ORDER BY mo.window_start DESC, mo.created_at DESC
                        LIMIT 1
                        """,
                        {
                            "metric_key": metric_keys["raw_tokens"],
                            "run_start_ts": run_start_ts,
                        },
                    ) or []
                    if prev_rows:
                        prev_tokens = self._safe_int(prev_rows[0][0])
                except Exception:
                    prev_tokens = None

            def _delta_float(cur: float | None, prev: float | None) -> float | None:
                if cur is None or prev is None:
                    return None
                return round(cur - prev, 6)

            def _delta_int(cur: int | None, prev: int | None) -> int | None:
                if cur is None or prev is None:
                    return None
                return cur - prev

            if prev_cost is None and prev_requests is None and prev_tokens is None:
                summary["cost_basis"] = "window_total_api"
                summary["delta_reference_timestamp"] = ""
                run_cost = raw_cost
                run_requests = raw_requests
                run_tokens = raw_tokens
            else:
                run_cost = _delta_float(raw_cost, prev_cost)
                run_requests = _delta_int(raw_requests, prev_requests)
                run_tokens = _delta_int(raw_tokens, prev_tokens)
                negative_delta = any(
                    v is not None and v < 0
                    for v in (run_cost, run_requests, run_tokens)
                )
                if negative_delta:
                    summary["cost_basis"] = "window_total_api_reset_detected"
                    summary["delta_reference_timestamp"] = prev_ts_text
                    run_cost = raw_cost
                    run_requests = raw_requests
                    run_tokens = raw_tokens
                else:
                    summary["cost_basis"] = "delta_from_snapshot"
                    summary["delta_reference_timestamp"] = prev_ts_text

            summary["cost_usd"] = run_cost
            summary["requests"] = run_requests
            summary["tokens"] = run_tokens
            summary["snapshot_history_path"] = "metric_observations"

            # Persist raw cumulative metrics and derived run-window deltas.
            try:
                if raw_cost is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["raw_cost"],
                        metric_value_numeric=float(raw_cost),
                        metric_unit="usd",
                        description=f"{provider_norm} cumulative cost as reported by provider usage endpoint",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
                if raw_requests is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["raw_requests"],
                        metric_value_numeric=float(raw_requests),
                        metric_unit="count",
                        description=f"{provider_norm} cumulative request count from provider usage endpoint",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
                if raw_tokens is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["raw_tokens"],
                        metric_value_numeric=float(raw_tokens),
                        metric_unit="tokens",
                        description=f"{provider_norm} cumulative token count from provider usage endpoint",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
                if run_cost is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["run_cost"],
                        metric_value_numeric=float(run_cost),
                        metric_unit="usd",
                        description=f"{provider_norm} run-window cost delta used in control panel",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
                if run_requests is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["run_requests"],
                        metric_value_numeric=float(run_requests),
                        metric_unit="count",
                        description=f"{provider_norm} run-window request delta used in control panel",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
                if run_tokens is not None:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_keys["run_tokens"],
                        metric_value_numeric=float(run_tokens),
                        metric_unit="tokens",
                        description=f"{provider_norm} run-window token delta used in control panel",
                        higher_is_better=False,
                        window_start=run_start_ts,
                        window_end=run_end_ts,
                    )
            except Exception as e:
                summary["db_persist_warning"] = f"metric_observations_persist_failed: {e}"
            return summary

        # Fallback path when DB metric persistence is unavailable.
        if not output_dir:
            summary["cost_basis"] = summary.get("cost_basis") or "window_total_api"
            return summary

        os.makedirs(output_dir, exist_ok=True)
        history_path = codex_review_path(f"{provider_norm}_cost_snapshots.jsonl")

        previous_snapshot: dict | None = None
        snapshots: list[dict] = []
        try:
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if not isinstance(record, dict):
                                continue
                            if str(record.get("provider", "")).strip().lower() != provider_norm:
                                continue
                            snapshots.append(record)
                        except Exception:
                            continue
        except Exception:
            snapshots = []

        run_start_ts = self._parse_iso_datetime(str(summary.get("start_ts", "") or ""))
        current_window_start = run_start_ts
        if snapshots:
            if current_window_start is not None:
                # Prefer baseline snapshots from an earlier usage window start_ts.
                # This avoids collapsing deltas to zero when report generation is rerun
                # for the same run window.
                pre_window_snapshots: list[tuple[datetime, dict]] = []
                for snap in snapshots:
                    snap_start = self._parse_iso_datetime(str(snap.get("start_ts", "") or ""))
                    if snap_start is None:
                        continue
                    if snap_start < current_window_start:
                        pre_window_snapshots.append((snap_start, snap))
                if pre_window_snapshots:
                    pre_window_snapshots.sort(key=lambda item: item[0])
                    previous_snapshot = pre_window_snapshots[-1][1]
                else:
                    # Fallback to wall-clock snapshot timestamp if window start isn't available
                    # in older history rows.
                    pre_run_snapshots: list[tuple[datetime, dict]] = []
                    for snap in snapshots:
                        snap_ts = self._parse_iso_datetime(str(snap.get("timestamp", "") or ""))
                        if snap_ts is None:
                            continue
                        if snap_ts < run_start_ts:
                            pre_run_snapshots.append((snap_ts, snap))
                    if pre_run_snapshots:
                        pre_run_snapshots.sort(key=lambda item: item[0])
                        previous_snapshot = pre_run_snapshots[-1][1]
                    else:
                        # No baseline prior to run start: treat API totals as window totals.
                        previous_snapshot = None
            else:
                previous_snapshot = snapshots[-1]

        snapshot_record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider_norm,
            "start_ts": str(summary.get("start_ts", "") or ""),
            "end_ts": str(summary.get("end_ts", "") or ""),
            "endpoint_used": str(summary.get("endpoint_used", "") or ""),
            "raw_cost_usd": raw_cost,
            "raw_requests": raw_requests,
            "raw_tokens": raw_tokens,
        }
        try:
            append_record = True
            if isinstance(previous_snapshot, dict):
                if (
                    str(previous_snapshot.get("start_ts", "") or "") == snapshot_record["start_ts"]
                    and str(previous_snapshot.get("end_ts", "") or "") == snapshot_record["end_ts"]
                    and self._safe_float(previous_snapshot.get("raw_cost_usd")) == raw_cost
                    and self._safe_int(previous_snapshot.get("raw_requests")) == raw_requests
                    and self._safe_int(previous_snapshot.get("raw_tokens")) == raw_tokens
                ):
                    append_record = False
            if append_record:
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snapshot_record) + "\n")
        except Exception as e:
            summary["cost_basis"] = "window_total_api"
            summary["delta_error"] = f"snapshot_write_failed: {e}"
            return summary

        prev_cost = self._safe_float((previous_snapshot or {}).get("raw_cost_usd"))
        prev_requests = self._safe_int((previous_snapshot or {}).get("raw_requests"))
        prev_tokens = self._safe_int((previous_snapshot or {}).get("raw_tokens"))

        if prev_cost is None and prev_requests is None and prev_tokens is None:
            summary["cost_basis"] = "window_total_api"
            summary["snapshot_history_path"] = history_path
            return summary

        def _delta_float(cur: float | None, prev: float | None) -> float | None:
            if cur is None or prev is None:
                return None
            return round(cur - prev, 6)

        def _delta_int(cur: int | None, prev: int | None) -> int | None:
            if cur is None or prev is None:
                return None
            return cur - prev

        cost_delta = _delta_float(raw_cost, prev_cost)
        req_delta = _delta_int(raw_requests, prev_requests)
        tok_delta = _delta_int(raw_tokens, prev_tokens)

        negative_delta = any(
            v is not None and v < 0
            for v in (cost_delta, req_delta, tok_delta)
        )
        if negative_delta:
            summary["cost_basis"] = "window_total_api_reset_detected"
            summary["delta_reference_timestamp"] = str((previous_snapshot or {}).get("timestamp", "") or "")
            summary["snapshot_history_path"] = history_path
            return summary

        summary["cost_usd"] = cost_delta if cost_delta is not None else raw_cost
        summary["requests"] = req_delta if req_delta is not None else raw_requests
        summary["tokens"] = tok_delta if tok_delta is not None else raw_tokens
        summary["cost_basis"] = "delta_from_snapshot"
        summary["delta_reference_timestamp"] = str((previous_snapshot or {}).get("timestamp", "") or "")
        summary["snapshot_history_path"] = history_path
        return summary

    def _summarize_openrouter_run_cost(
        self,
        report_timestamp: str | None,
        pipeline_runtime: dict | None,
        output_dir: str | None = None,
    ) -> dict:
        """Fetch best-effort OpenRouter run usage/cost for the report window.

        OpenRouter account-level usage endpoints require a management key. We
        therefore prefer management-key env vars and fall back to the standard
        API key only as a best-effort attempt with explicit error reporting.
        """
        summary = {
            "available": False,
            "source": "openrouter_api",
            "start_ts": "",
            "end_ts": "",
            "cost_usd": None,
            "requests": None,
            "tokens": None,
            "endpoint_used": "",
            "error": "",
        }
        start_ts: datetime | None = None
        end_ts: datetime | None = None
        if isinstance(pipeline_runtime, dict):
            start_ts = self._parse_iso_datetime(pipeline_runtime.get("start_ts"))
            end_ts = self._parse_iso_datetime(pipeline_runtime.get("end_ts"))
        if start_ts is None or end_ts is None:
            end_ts = datetime.now()
            if report_timestamp:
                parsed = self._parse_iso_datetime(report_timestamp)
                if parsed is not None:
                    end_ts = parsed
            start_ts = end_ts - timedelta(hours=24)
        summary["start_ts"] = start_ts.isoformat(sep=" ")
        summary["end_ts"] = end_ts.isoformat(sep=" ")

        openrouter_key = (
            os.getenv("OPENROUTER_MANAGEMENT_KEY")
            or os.getenv("OPENROUTER_ADMIN_KEY")
            or os.getenv("OPENROUTER_MGMT_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_API" + "_KEY")
        )
        if not openrouter_key:
            summary["error"] = (
                "Missing OpenRouter key. Set OPENROUTER_MANAGEMENT_KEY (preferred) "
                "or OPENROUTER_API_KEY."
            )
            return summary

        try:
            import requests  # type: ignore
        except Exception as e:
            summary["error"] = f"requests import failed: {e}"
            return summary

        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
        start_iso = start_ts.isoformat()
        end_iso = end_ts.isoformat()
        start_epoch = int(start_ts.timestamp())
        end_epoch = int(end_ts.timestamp())
        endpoint_candidates = [
            ("https://openrouter.ai/api/v1/activity", {"start_time": start_iso, "end_time": end_iso}),
            ("https://openrouter.ai/api/v1/activity", {"start_time": start_epoch, "end_time": end_epoch}),
            ("https://openrouter.ai/api/v1/usage", {"start_time": start_iso, "end_time": end_iso}),
            ("https://openrouter.ai/api/v1/usage", {"start_time": start_epoch, "end_time": end_epoch}),
            ("https://openrouter.ai/api/v1/auth/key", {}),
        ]
        last_error = ""
        for endpoint, params in endpoint_candidates:
            try:
                resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
                if resp.status_code >= 400:
                    error_detail = ""
                    try:
                        payload = resp.json()
                        if isinstance(payload, dict):
                            error_obj = payload.get("error")
                            if isinstance(error_obj, dict):
                                message = str(error_obj.get("message", "")).strip()
                                if message:
                                    error_detail = message
                    except Exception:
                        error_detail = ""
                    if resp.status_code == 403 and "management" in error_detail.lower():
                        last_error = (
                            "OpenRouter usage endpoint requires a management key. "
                            "Set OPENROUTER_MANAGEMENT_KEY in src/.env."
                        )
                    else:
                        last_error = (
                            f"{endpoint} HTTP {resp.status_code}"
                            + (f": {error_detail}" if error_detail else "")
                        )
                    continue
                payload = resp.json()
                parsed = self._extract_openrouter_activity_totals(payload)
                if parsed.get("cost_usd") is None and parsed.get("requests") is None and parsed.get("tokens") is None:
                    last_error = f"{endpoint} returned no cost/usage fields for requested window"
                    continue
                summary.update(
                    {
                        "available": True,
                        "endpoint_used": endpoint,
                        "cost_usd": parsed.get("cost_usd"),
                        "requests": parsed.get("requests"),
                        "tokens": parsed.get("tokens"),
                        "error": "",
                    }
                )
                return self._apply_snapshot_delta_cost("openrouter", summary, output_dir)
            except Exception as e:
                last_error = f"{endpoint} request failed: {e}"
                continue
        summary["error"] = last_error or "No OpenRouter usage endpoint returned parseable totals"
        return summary

    def _extract_openai_cost_totals(self, payload: Any) -> dict:
        """Best-effort parser for OpenAI usage/cost payload shapes."""
        totals = {"cost_usd": None, "requests": None, "tokens": None}
        if not isinstance(payload, (dict, list)):
            return totals

        def _row_cost(row: dict) -> float | None:
            for key in ("cost", "spend", "total_cost", "usd", "amount", "value"):
                raw = row.get(key)
                if isinstance(raw, dict):
                    nested = self._safe_float(raw.get("value"))
                    if nested is not None:
                        return nested
                val = self._safe_float(raw)
                if val is not None:
                    return val
            amount_obj = row.get("amount")
            if isinstance(amount_obj, dict):
                nested = self._safe_float(amount_obj.get("value"))
                if nested is not None:
                    return nested
            return None

        def _sum_rows(rows: list[dict]) -> tuple[float | None, int | None, int | None]:
            cost_sum = 0.0
            req_sum = 0
            tok_sum = 0
            cost_seen = False
            req_seen = False
            tok_seen = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                nested_results = row.get("results")
                candidate_rows = (
                    [r for r in nested_results if isinstance(r, dict)]
                    if isinstance(nested_results, list) and nested_results
                    else [row]
                )
                for candidate in candidate_rows:
                    c = _row_cost(candidate)
                    if c is not None:
                        cost_sum += c
                        cost_seen = True
                    for k in ("requests", "request_count", "num_requests", "n_requests", "num_model_requests"):
                        v = self._safe_int(candidate.get(k))
                        if v is not None:
                            req_sum += v
                            req_seen = True
                            break
                    token_candidates = [
                        self._safe_int(candidate.get("tokens")),
                        self._safe_int(candidate.get("total_tokens")),
                        self._safe_int(candidate.get("input_tokens")),
                        self._safe_int(candidate.get("output_tokens")),
                        self._safe_int(candidate.get("prompt_tokens")),
                        self._safe_int(candidate.get("completion_tokens")),
                        self._safe_int(candidate.get("reasoning_tokens")),
                        self._safe_int(candidate.get("input_text_tokens")),
                        self._safe_int(candidate.get("output_text_tokens")),
                    ]
                    token_values = [v for v in token_candidates if v is not None]
                    if token_values:
                        tok_sum += sum(token_values)
                        tok_seen = True
            return (
                round(cost_sum, 6) if cost_seen else None,
                req_sum if req_seen else None,
                tok_sum if tok_seen else None,
            )

        if isinstance(payload, dict):
            for key in ("data", "results", "usage", "costs", "buckets"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    cost, reqs, toks = _sum_rows(rows)
                    if cost is not None or reqs is not None or toks is not None:
                        totals["cost_usd"] = cost
                        totals["requests"] = reqs
                        totals["tokens"] = toks
                        return totals
            totals["cost_usd"] = _row_cost(payload)
            for k in ("requests", "request_count", "num_requests", "n_requests"):
                v = self._safe_int(payload.get(k))
                if v is not None:
                    totals["requests"] = v
                    break
            token_candidates = [
                self._safe_int(payload.get("tokens")),
                self._safe_int(payload.get("total_tokens")),
                self._safe_int(payload.get("prompt_tokens")),
                self._safe_int(payload.get("completion_tokens")),
                self._safe_int(payload.get("reasoning_tokens")),
            ]
            token_values = [v for v in token_candidates if v is not None]
            if token_values:
                totals["tokens"] = sum(token_values)
            return totals

        if isinstance(payload, list):
            cost, reqs, toks = _sum_rows(payload)
            totals["cost_usd"] = cost
            totals["requests"] = reqs
            totals["tokens"] = toks
        return totals

    def _summarize_openai_run_cost(
        self,
        report_timestamp: str | None,
        pipeline_runtime: dict | None,
        output_dir: str | None = None,
    ) -> dict:
        """Fetch best-effort OpenAI run usage/cost totals using management key endpoints."""
        summary = {
            "available": False,
            "source": "openai_api",
            "start_ts": "",
            "end_ts": "",
            "cost_usd": None,
            "requests": None,
            "tokens": None,
            "endpoint_used": "",
            "error": "",
        }
        start_ts: datetime | None = None
        end_ts: datetime | None = None
        if isinstance(pipeline_runtime, dict):
            start_ts = self._parse_iso_datetime(pipeline_runtime.get("start_ts"))
            end_ts = self._parse_iso_datetime(pipeline_runtime.get("end_ts"))
        if start_ts is None or end_ts is None:
            end_ts = datetime.now()
            if report_timestamp:
                parsed = self._parse_iso_datetime(report_timestamp)
                if parsed is not None:
                    end_ts = parsed
            start_ts = end_ts - timedelta(hours=24)
        summary["start_ts"] = start_ts.isoformat(sep=" ")
        summary["end_ts"] = end_ts.isoformat(sep=" ")

        openai_key = (
            os.getenv("OPENAI_MANAGEMENT_KEY")
            or os.getenv("OPENAI_ADMIN_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not openai_key:
            summary["error"] = "Missing OPENAI_MANAGEMENT_KEY (preferred) or OPENAI_API_KEY."
            return summary

        try:
            import requests  # type: ignore
        except Exception as e:
            summary["error"] = f"requests import failed: {e}"
            return summary

        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        start_epoch = int(start_ts.timestamp())
        end_epoch = int(end_ts.timestamp())
        endpoint_candidates = [
            ("https://api.openai.com/v1/organization/costs", {"start_time": start_epoch, "end_time": end_epoch}),
            ("https://api.openai.com/v1/organization/usage/completions", {"start_time": start_epoch, "end_time": end_epoch}),
            ("https://api.openai.com/v1/organization/usage/responses", {"start_time": start_epoch, "end_time": end_epoch}),
            ("https://api.openai.com/v1/usage", {"start_time": start_epoch, "end_time": end_epoch}),
        ]
        last_error = ""
        for endpoint, params in endpoint_candidates:
            try:
                resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
                if resp.status_code >= 400:
                    detail = ""
                    try:
                        payload = resp.json()
                        if isinstance(payload, dict):
                            err = payload.get("error")
                            if isinstance(err, dict):
                                detail = str(err.get("message", "")).strip()
                            elif isinstance(err, str):
                                detail = err.strip()
                    except Exception:
                        detail = ""
                    last_error = (
                        f"{endpoint} HTTP {resp.status_code}"
                        + (f": {detail}" if detail else "")
                    )
                    continue
                payload = resp.json()
                parsed = self._extract_openai_cost_totals(payload)
                if parsed.get("cost_usd") is None and parsed.get("requests") is None and parsed.get("tokens") is None:
                    last_error = f"{endpoint} returned no parseable cost/usage fields for requested window"
                    continue
                summary.update(
                    {
                        "available": True,
                        "endpoint_used": endpoint,
                        "cost_usd": parsed.get("cost_usd"),
                        "requests": parsed.get("requests"),
                        "tokens": parsed.get("tokens"),
                        "error": "",
                    }
                )
                return self._apply_snapshot_delta_cost("openai", summary, output_dir)
            except Exception as e:
                last_error = f"{endpoint} request failed: {e}"
                continue
        summary["error"] = last_error or "No OpenAI usage endpoint returned parseable totals"
        return summary

    @staticmethod
    def _status_class(status: str) -> str:
        """Map PASS/WARN/FAIL-like status to CSS class suffix."""
        normalized = str(status or "").strip().upper()
        if normalized in {"PASS", "HEALTHY"}:
            return "status-pass"
        if normalized in {"WARN", "WARNING", "WATCH"}:
            return "status-warning"
        return "status-fail"

    @staticmethod
    def _status_rank(status: str) -> int:
        """Return severity rank for status comparisons."""
        normalized = str(status or "").strip().upper()
        if normalized in {"FAIL", "AT_RISK", "DEGRADED"}:
            return 2
        if normalized in {"WARN", "WARNING", "WATCH"}:
            return 1
        return 0

    def _control_status_from_checks(self, checks: list[dict]) -> str:
        """Reduce check-level statuses to one objective status."""
        worst_rank = 0
        for check in checks:
            worst_rank = max(worst_rank, self._status_rank(str(check.get("status", "PASS"))))
        if worst_rank >= 2:
            return "FAIL"
        if worst_rank == 1:
            return "WARN"
        return "PASS"

    def _update_and_summarize_run_control_history(self, output_dir: str, record: dict) -> dict:
        """Persist control-panel KPIs and return short trend deltas vs recent history."""
        run_id = str(record.get("run_id", "") or "").strip() or str(os.getenv("DS_RUN_ID", "")).strip() or "validation-report"
        run_ts = self._parse_iso_datetime(str(record.get("timestamp", "") or "")) or datetime.now()
        metric_specs = [
            (
                "runtime_hours",
                "run_control_runtime_hours",
                "hours",
                "Run-control panel: end-to-end pipeline runtime in hours",
                False,
            ),
            (
                "event_yield_rate",
                "run_control_event_yield_rate",
                "ratio",
                "Run-control panel: urls_with_events / urls_passed_for_scraping",
                True,
            ),
            (
                "llm_calls_per_event_url",
                "run_control_llm_calls_per_event_url",
                "ratio",
                "Run-control panel: LLM extraction attempts per event-producing URL",
                False,
            ),
        ]

        # Preferred path: DB-backed metric history.
        if getattr(self, "db_handler", None) and hasattr(self.db_handler, "record_metric_observation"):
            for source_key, metric_key, metric_unit, description, higher_is_better in metric_specs:
                value = record.get(source_key)
                if value is None:
                    continue
                try:
                    self.db_handler.record_metric_observation(
                        run_id=run_id,
                        metric_key=metric_key,
                        metric_value_numeric=float(value),
                        metric_unit=metric_unit,
                        description=description,
                        higher_is_better=higher_is_better,
                        window_start=run_ts,
                        window_end=run_ts,
                    )
                except Exception:
                    continue

            def _avg_prev(metric_key: str) -> tuple[float | None, int]:
                try:
                    rows = self.db_handler.execute_query(
                        """
                        SELECT mo.metric_value_numeric
                        FROM metric_observations mo
                        JOIN metric_definitions md ON md.metric_id = mo.metric_id
                        WHERE md.metric_key = :metric_key
                          AND mo.run_id <> :run_id
                        ORDER BY mo.created_at DESC
                        LIMIT 7
                        """,
                        {"metric_key": metric_key, "run_id": run_id},
                    ) or []
                except Exception:
                    rows = []
                values: list[float] = []
                for row in rows:
                    try:
                        values.append(float(row[0]))
                    except Exception:
                        continue
                if not values:
                    return None, 0
                return (sum(values) / len(values)), len(values)

            runtime_avg, runtime_n = _avg_prev("run_control_runtime_hours")
            yield_avg, yield_n = _avg_prev("run_control_event_yield_rate")
            llm_avg, llm_n = _avg_prev("run_control_llm_calls_per_event_url")
            return {
                "path": "metric_observations",
                "runs_considered": max(runtime_n, yield_n, llm_n),
                "runtime_hours_avg_prev": runtime_avg,
                "event_yield_rate_avg_prev": yield_avg,
                "llm_calls_per_event_url_avg_prev": llm_avg,
            }

        # Fallback to file history when DB metrics are unavailable.
        os.makedirs(output_dir, exist_ok=True)
        history_path = codex_review_path("run_control_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        rows: list[dict] = []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        rows.append(json.loads(raw))
                    except Exception:
                        continue
        except Exception:
            rows = [record]

        recent = rows[-8:]
        prev = recent[:-1] if len(recent) > 1 else []

        def _avg(key: str) -> float | None:
            vals: list[float] = []
            for row in prev:
                val = row.get(key)
                if val is None:
                    continue
                try:
                    vals.append(float(val))
                except Exception:
                    continue
            if not vals:
                return None
            return sum(vals) / len(vals)

        return {
            "path": history_path,
            "runs_considered": len(prev),
            "runtime_hours_avg_prev": _avg("runtime_hours"),
            "event_yield_rate_avg_prev": _avg("event_yield_rate"),
            "llm_calls_per_event_url_avg_prev": _avg("llm_calls_per_event_url"),
        }

    def _summarize_run_control_panel(
        self,
        output_dir: str,
        run_id: str,
        reliability_scorecard: dict,
        llm_activity: dict,
        llm_quality: dict,
        scraping_results: dict | None,
        fb_ig_funnel: dict,
        scraper_network: dict,
        pipeline_runtime: dict,
        chatbot_performance: dict,
        openrouter_cost: dict | None,
        openai_cost: dict | None,
        accuracy_replay: dict | None,
    ) -> dict:
        """Build cost/accuracy/completeness/runtime control panel focused on operator decisions."""
        reporting_cfg = self.validation_config.get("reporting", {}) if isinstance(self.validation_config, dict) else {}
        control_cfg = reporting_cfg.get("control_panel_targets", {}) if isinstance(reporting_cfg, dict) else {}
        cost_cfg = control_cfg.get("cost", {}) if isinstance(control_cfg, dict) else {}
        accuracy_cfg = control_cfg.get("accuracy", {}) if isinstance(control_cfg, dict) else {}
        completeness_cfg = control_cfg.get("completeness", {}) if isinstance(control_cfg, dict) else {}
        runtime_cfg = control_cfg.get("runtime", {}) if isinstance(control_cfg, dict) else {}

        llm_total_attempts = int((llm_quality or {}).get("total_attempts", 0) or 0)
        event_urls = int((llm_quality or {}).get("successful_urls", 0) or 0)
        too_short_urls = int((llm_quality or {}).get("too_short_urls", 0) or 0)
        quality_total_urls = int((llm_quality or {}).get("total_urls", 0) or 0)
        llm_calls_per_event_url = (llm_total_attempts / event_urls) if event_urls > 0 else None
        too_short_rate = (too_short_urls / quality_total_urls) if quality_total_urls > 0 else None
        pressure_level = str((llm_activity or {}).get("pressure_level", "LOW") or "LOW").upper()
        pressure_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(pressure_level, 2)

        max_calls_warn = float(cost_cfg.get("max_llm_calls_per_event_url_warn", 1.75) or 1.75)
        max_calls_fail = float(cost_cfg.get("max_llm_calls_per_event_url_fail", 2.5) or 2.5)
        max_short_warn = float(cost_cfg.get("max_too_short_rate_warn", 0.20) or 0.20)
        max_short_fail = float(cost_cfg.get("max_too_short_rate_fail", 0.35) or 0.35)
        max_cost_warn = float(cost_cfg.get("max_run_cost_usd_warn", 2.0) or 2.0)
        max_cost_fail = float(cost_cfg.get("max_run_cost_usd_fail", 5.0) or 5.0)

        cost_checks: list[dict] = []
        openrouter_cost_usd = self._safe_float((openrouter_cost or {}).get("cost_usd"))
        openai_cost_usd = self._safe_float((openai_cost or {}).get("cost_usd"))
        if openrouter_cost_usd is None:
            cost_checks.append({
                "name": "OpenRouter Run Cost (USD)",
                "actual": "N/A",
                "target": f"<= ${max_cost_warn:.2f} warn, <= ${max_cost_fail:.2f} fail",
                "delta": "N/A",
                "status": "WARN",
                "details": str((openrouter_cost or {}).get("error", "OpenRouter cost unavailable")),
            })
        else:
            run_cost_status = "PASS"
            if openrouter_cost_usd > max_cost_fail:
                run_cost_status = "FAIL"
            elif openrouter_cost_usd > max_cost_warn:
                run_cost_status = "WARN"
            cost_checks.append({
                "name": "OpenRouter Run Cost (USD)",
                "actual": f"${openrouter_cost_usd:.4f}",
                "target": f"<= ${max_cost_warn:.2f} warn, <= ${max_cost_fail:.2f} fail",
                "delta": f"{openrouter_cost_usd - max_cost_warn:+.4f} vs warn",
                "status": run_cost_status,
                "details": (
                    f"requests={int((openrouter_cost or {}).get('requests', 0) or 0)}, "
                    f"tokens={int((openrouter_cost or {}).get('tokens', 0) or 0)}, "
                    f"endpoint={str((openrouter_cost or {}).get('endpoint_used', ''))}, "
                    f"basis={str((openrouter_cost or {}).get('cost_basis', 'window_total_api'))}"
                ),
            })

        if openai_cost_usd is None:
            cost_checks.append({
                "name": "OpenAI Run Cost (USD)",
                "actual": "N/A",
                "target": "Informational",
                "delta": "N/A",
                "status": "WARN",
                "details": str((openai_cost or {}).get("error", "OpenAI cost unavailable")),
            })
        else:
            cost_checks.append({
                "name": "OpenAI Run Cost (USD)",
                "actual": f"${openai_cost_usd:.4f}",
                "target": "Informational",
                "delta": "N/A",
                "status": "PASS",
                "details": (
                    f"requests={int((openai_cost or {}).get('requests', 0) or 0)}, "
                    f"tokens={int((openai_cost or {}).get('tokens', 0) or 0)}, "
                    f"endpoint={str((openai_cost or {}).get('endpoint_used', ''))}, "
                    f"basis={str((openai_cost or {}).get('cost_basis', 'window_total_api'))}"
                ),
            })

        total_run_cost_usd = None
        if openrouter_cost_usd is not None or openai_cost_usd is not None:
            total_run_cost_usd = float((openrouter_cost_usd or 0.0) + (openai_cost_usd or 0.0))
            total_status = "PASS"
            if total_run_cost_usd > max_cost_fail:
                total_status = "FAIL"
            elif total_run_cost_usd > max_cost_warn:
                total_status = "WARN"
            cost_checks.append({
                "name": "Total Run Cost (OpenRouter + OpenAI)",
                "actual": f"${total_run_cost_usd:.4f}",
                "target": f"<= ${max_cost_warn:.2f} warn, <= ${max_cost_fail:.2f} fail",
                "delta": f"{total_run_cost_usd - max_cost_warn:+.4f} vs warn",
                "status": total_status,
                "details": "Sum of provider management-usage totals for run window.",
            })

        if llm_calls_per_event_url is None:
            cost_checks.append({
                "name": "LLM Calls / Event URL",
                "actual": "N/A",
                "target": f"<= {max_calls_warn:.2f} warn, <= {max_calls_fail:.2f} fail",
                "delta": "N/A",
                "status": "WARN",
                "details": "No successful event URLs in window.",
            })
        else:
            status = "PASS"
            if llm_calls_per_event_url > max_calls_fail:
                status = "FAIL"
            elif llm_calls_per_event_url > max_calls_warn:
                status = "WARN"
            cost_checks.append({
                "name": "LLM Calls / Event URL",
                "actual": f"{llm_calls_per_event_url:.2f}",
                "target": f"<= {max_calls_warn:.2f} warn, <= {max_calls_fail:.2f} fail",
                "delta": f"{llm_calls_per_event_url - max_calls_warn:+.2f} vs warn",
                "status": status,
                "details": f"{llm_total_attempts} attempts / {event_urls} event-producing URLs",
            })
        if too_short_rate is None:
            too_short_status = "WARN"
            too_short_actual = "N/A"
            too_short_delta = "N/A"
        else:
            too_short_status = "PASS"
            if too_short_rate > max_short_fail:
                too_short_status = "FAIL"
            elif too_short_rate > max_short_warn:
                too_short_status = "WARN"
            too_short_actual = f"{too_short_rate:.1%}"
            too_short_delta = f"{too_short_rate - max_short_warn:+.1%} vs warn"
        cost_checks.append({
            "name": "Too-Short Extraction Rate",
            "actual": too_short_actual,
            "target": f"<= {max_short_warn:.0%} warn, <= {max_short_fail:.0%} fail",
            "delta": too_short_delta,
            "status": too_short_status,
            "details": f"{too_short_urls}/{quality_total_urls} URL outcomes were too_short",
        })
        pressure_status = "PASS" if pressure_rank == 0 else ("WARN" if pressure_rank == 1 else "FAIL")
        cost_checks.append({
            "name": "LLM Cost Pressure",
            "actual": pressure_level,
            "target": "LOW",
            "delta": f"rank={pressure_rank}",
            "status": pressure_status,
            "details": "; ".join((llm_activity or {}).get("pressure_reasons", [])[:3]) or "No pressure reasons logged.",
        })
        cost_status = self._control_status_from_checks(cost_checks)

        url_level_score = float((reliability_scorecard or {}).get("url_level_score", (reliability_scorecard or {}).get("score", 0)) or 0)
        chatbot_avg_score = float(((reliability_scorecard or {}).get("metrics", {}) or {}).get("chatbot_average_score", 0) or 0)
        event_yield_rate = fb_ig_funnel.get("events_over_passed_rate")
        hard_failure_rate = float((llm_quality or {}).get("hard_failure_rate", 0.0) or 0.0)

        min_url_score_warn = float(accuracy_cfg.get("min_url_level_score_warn", 80.0) or 80.0)
        min_url_score_fail = float(accuracy_cfg.get("min_url_level_score_fail", 75.0) or 75.0)
        min_event_yield_warn = float(accuracy_cfg.get("min_event_yield_rate_warn", 0.20) or 0.20)
        min_event_yield_fail = float(accuracy_cfg.get("min_event_yield_rate_fail", 0.10) or 0.10)
        min_chatbot_warn = float(accuracy_cfg.get("min_chatbot_average_score_warn", 75.0) or 75.0)
        min_chatbot_fail = float(accuracy_cfg.get("min_chatbot_average_score_fail", 70.0) or 70.0)
        max_hard_fail_warn = float(accuracy_cfg.get("max_llm_hard_failure_rate_warn", 0.08) or 0.08)
        max_hard_fail_fail = float(accuracy_cfg.get("max_llm_hard_failure_rate_fail", 0.15) or 0.15)
        min_replay_warn = float(accuracy_cfg.get("min_replay_coverage_accuracy_warn", 85.0) or 85.0)
        min_replay_fail = float(accuracy_cfg.get("min_replay_coverage_accuracy_fail", 75.0) or 75.0)

        accuracy_checks: list[dict] = []
        replay_coverage_pct = self._safe_float((accuracy_replay or {}).get("coverage_accuracy_pct"))
        if replay_coverage_pct is None:
            accuracy_checks.append({
                "name": "Replay Coverage Accuracy (%)",
                "actual": "N/A",
                "target": f">= {min_replay_warn:.1f} warn, >= {min_replay_fail:.1f} fail",
                "delta": "N/A",
                "status": "WARN",
                "details": str((accuracy_replay or {}).get("error", "Replay accuracy unavailable")),
            })
        else:
            replay_status = "PASS"
            if replay_coverage_pct < min_replay_fail:
                replay_status = "FAIL"
            elif replay_coverage_pct < min_replay_warn:
                replay_status = "WARN"
            accuracy_checks.append({
                "name": "Replay Coverage Accuracy (%)",
                "actual": f"{replay_coverage_pct:.2f}",
                "target": f">= {min_replay_warn:.1f} warn, >= {min_replay_fail:.1f} fail",
                "delta": f"{replay_coverage_pct - min_replay_warn:+.2f} vs warn",
                "status": replay_status,
                "details": (
                    f"true={int((accuracy_replay or {}).get('true_count', 0) or 0)} / "
                    f"total={int((accuracy_replay or {}).get('total_rows', 0) or 0)}"
                ),
            })
        url_score_status = "PASS"
        if url_level_score < min_url_score_fail:
            url_score_status = "FAIL"
        elif url_level_score < min_url_score_warn:
            url_score_status = "WARN"
        accuracy_checks.append({
            "name": "URL-Level Reliability Score",
            "actual": f"{url_level_score:.1f}",
            "target": f">= {min_url_score_warn:.1f} warn, >= {min_url_score_fail:.1f} fail",
            "delta": f"{url_level_score - min_url_score_warn:+.1f} vs warn",
            "status": url_score_status,
            "details": "Primary accuracy guard for report grading.",
        })
        if event_yield_rate is None:
            event_yield_status = "WARN"
            event_yield_actual = "N/A"
            event_yield_delta = "N/A"
            event_yield_details = "No passed-for-scraping denominator."
        else:
            event_yield_status = "PASS"
            if float(event_yield_rate) < min_event_yield_fail:
                event_yield_status = "FAIL"
            elif float(event_yield_rate) < min_event_yield_warn:
                event_yield_status = "WARN"
            event_yield_actual = f"{float(event_yield_rate):.1%}"
            event_yield_delta = f"{float(event_yield_rate) - min_event_yield_warn:+.1%} vs warn"
            event_yield_details = (
                f"{int(fb_ig_funnel.get('urls_with_events', 0) or 0)} URLs with events / "
                f"{int(fb_ig_funnel.get('urls_passed_for_scraping', 0) or 0)} passed-for-scraping URLs"
            )
        accuracy_checks.append({
            "name": "Event Yield (URLs with events / URLs passed)",
            "actual": event_yield_actual,
            "target": f">= {min_event_yield_warn:.0%} warn, >= {min_event_yield_fail:.0%} fail",
            "delta": event_yield_delta,
            "status": event_yield_status,
            "details": event_yield_details,
        })
        chatbot_status = "PASS"
        if chatbot_avg_score < min_chatbot_fail:
            chatbot_status = "FAIL"
        elif chatbot_avg_score < min_chatbot_warn:
            chatbot_status = "WARN"
        accuracy_checks.append({
            "name": "Chatbot Average Score",
            "actual": f"{chatbot_avg_score:.1f}",
            "target": f">= {min_chatbot_warn:.1f} warn, >= {min_chatbot_fail:.1f} fail",
            "delta": f"{chatbot_avg_score - min_chatbot_warn:+.1f} vs warn",
            "status": chatbot_status,
            "details": "Included for end-user answer quality control.",
        })
        hard_fail_status = "PASS"
        if hard_failure_rate > max_hard_fail_fail:
            hard_fail_status = "FAIL"
        elif hard_failure_rate > max_hard_fail_warn:
            hard_fail_status = "WARN"
        accuracy_checks.append({
            "name": "LLM Hard Failure Rate",
            "actual": f"{hard_failure_rate:.1%}",
            "target": f"<= {max_hard_fail_warn:.0%} warn, <= {max_hard_fail_fail:.0%} fail",
            "delta": f"{hard_failure_rate - max_hard_fail_warn:+.1%} vs warn",
            "status": hard_fail_status,
            "details": "From llm_extract_attempt_result outcomes.",
        })
        accuracy_status = self._control_status_from_checks(accuracy_checks)

        source_distribution = ((scraping_results or {}).get("source_distribution", {}) or {})
        source_dist_status_raw = str(source_distribution.get("status", "WARNING") or "WARNING").upper()
        if source_dist_status_raw in {"PASS", "OK"}:
            source_dist_status = "PASS"
        elif source_dist_status_raw in {"WARN", "WARNING"}:
            source_dist_status = "WARN"
        else:
            source_dist_status = "FAIL"
        missing_sources = list(source_distribution.get("missing_sources", []) or [])
        trend_monitoring = source_distribution.get("trend_monitoring", {}) or {}
        trend_alerts = list(trend_monitoring.get("alerts", []) or [])
        critical_alert_count = sum(
            1 for alert in trend_alerts if str((alert or {}).get("severity", "")).lower() == "critical"
        )
        warning_alert_count = max(0, len(trend_alerts) - critical_alert_count)
        top_10_percentage = self._safe_float(source_distribution.get("top_10_percentage"))

        max_top10_warn = float(completeness_cfg.get("max_top_10_percentage_warn", 90.0) or 90.0)
        max_top10_fail = float(completeness_cfg.get("max_top_10_percentage_fail", 95.0) or 95.0)
        max_warning_alerts_warn = int(completeness_cfg.get("max_warning_trend_alerts_warn", 0) or 0)
        max_critical_alerts_fail = int(completeness_cfg.get("max_critical_trend_alerts_fail", 0) or 0)

        completeness_checks: list[dict] = []
        completeness_checks.append({
            "name": "Source Distribution Check Status",
            "actual": source_dist_status_raw,
            "target": "PASS",
            "delta": "N/A",
            "status": source_dist_status,
            "details": "Direct status from Source Distribution Check section.",
        })

        missing_sources_status = "PASS" if len(missing_sources) == 0 else "FAIL"
        completeness_checks.append({
            "name": "Missing Required Sources",
            "actual": str(len(missing_sources)),
            "target": "0",
            "delta": f"+{len(missing_sources)}",
            "status": missing_sources_status,
            "details": ", ".join(str(s) for s in missing_sources[:8]) if missing_sources else "All required sources present.",
        })

        trend_alert_status = "PASS"
        if critical_alert_count > max_critical_alerts_fail:
            trend_alert_status = "FAIL"
        elif warning_alert_count > max_warning_alerts_warn:
            trend_alert_status = "WARN"
        completeness_checks.append({
            "name": "Top-Source Trend Alerts",
            "actual": f"critical={critical_alert_count}, warning={warning_alert_count}",
            "target": (
                f"critical <= {max_critical_alerts_fail}, "
                f"warning <= {max_warning_alerts_warn}"
            ),
            "delta": (
                f"critical={critical_alert_count - max_critical_alerts_fail:+d}, "
                f"warning={warning_alert_count - max_warning_alerts_warn:+d}"
            ),
            "status": trend_alert_status,
            "details": f"history_runs_used={int(trend_monitoring.get('history_runs_used', 0) or 0)}",
        })

        if top_10_percentage is None:
            concentration_status = "WARN"
            concentration_actual = "N/A"
            concentration_delta = "N/A"
            concentration_details = "Top-10 concentration unavailable."
        else:
            concentration_status = "PASS"
            if top_10_percentage > max_top10_fail:
                concentration_status = "FAIL"
            elif top_10_percentage > max_top10_warn:
                concentration_status = "WARN"
            concentration_actual = f"{top_10_percentage:.1f}%"
            concentration_delta = f"{top_10_percentage - max_top10_warn:+.1f}% vs warn"
            concentration_details = (
                f"top_10_total={int(source_distribution.get('top_10_total', 0) or 0)}, "
                f"total_events={int(source_distribution.get('total_events', 0) or 0)}"
            )
        completeness_checks.append({
            "name": "Top 10 Source Concentration",
            "actual": concentration_actual,
            "target": f"<= {max_top10_warn:.1f}% warn, <= {max_top10_fail:.1f}% fail",
            "delta": concentration_delta,
            "status": concentration_status,
            "details": concentration_details,
        })
        completeness_status = self._control_status_from_checks(completeness_checks)

        runtime_hours = float((pipeline_runtime or {}).get("runtime_hours", 0.0) or 0.0)
        scraper_exception_rate = float((scraper_network or {}).get("exception_rate", 0.0) or 0.0)
        total_events = int((((scraping_results or {}).get("source_distribution", {}) or {}).get("total_events", 0) or 0))
        events_per_hour = (float(total_events) / runtime_hours) if runtime_hours > 0 and total_events > 0 else None
        query_p95_ms = float((((chatbot_performance or {}).get("query_latency_ms", {}) or {}).get("p95", 0.0) or 0.0))

        max_runtime_warn = float(runtime_cfg.get("max_pipeline_runtime_hours_warn", 6.0) or 6.0)
        max_runtime_fail = float(runtime_cfg.get("max_pipeline_runtime_hours_fail", 8.0) or 8.0)
        min_throughput_warn = float(runtime_cfg.get("min_events_per_hour_warn", 30.0) or 30.0)
        min_throughput_fail = float(runtime_cfg.get("min_events_per_hour_fail", 20.0) or 20.0)
        max_scraper_exc_warn = float(runtime_cfg.get("max_scraper_exception_rate_warn", 0.20) or 0.20)
        max_scraper_exc_fail = float(runtime_cfg.get("max_scraper_exception_rate_fail", 0.30) or 0.30)
        max_query_p95_warn = float(runtime_cfg.get("max_chatbot_query_p95_ms_warn", 15000.0) or 15000.0)
        max_query_p95_fail = float(runtime_cfg.get("max_chatbot_query_p95_ms_fail", 25000.0) or 25000.0)

        runtime_checks: list[dict] = []
        runtime_status_metric = "PASS"
        if runtime_hours > max_runtime_fail:
            runtime_status_metric = "FAIL"
        elif runtime_hours > max_runtime_warn:
            runtime_status_metric = "WARN"
        runtime_checks.append({
            "name": "Pipeline Runtime (hours)",
            "actual": f"{runtime_hours:.2f}",
            "target": f"<= {max_runtime_warn:.2f} warn, <= {max_runtime_fail:.2f} fail",
            "delta": f"{runtime_hours - max_runtime_warn:+.2f}h vs warn",
            "status": runtime_status_metric,
            "details": "Computed from run_id timestamps across top-level logs.",
        })
        if events_per_hour is None:
            throughput_status = "WARN"
            throughput_actual = "N/A"
            throughput_delta = "N/A"
            throughput_details = "Need runtime + total_events for throughput."
        else:
            throughput_status = "PASS"
            if events_per_hour < min_throughput_fail:
                throughput_status = "FAIL"
            elif events_per_hour < min_throughput_warn:
                throughput_status = "WARN"
            throughput_actual = f"{events_per_hour:.1f}"
            throughput_delta = f"{events_per_hour - min_throughput_warn:+.1f} vs warn"
            throughput_details = f"{total_events} events / {runtime_hours:.2f} runtime hours"
        runtime_checks.append({
            "name": "Event Throughput (events/hour)",
            "actual": throughput_actual,
            "target": f">= {min_throughput_warn:.1f} warn, >= {min_throughput_fail:.1f} fail",
            "delta": throughput_delta,
            "status": throughput_status,
            "details": throughput_details,
        })
        exc_status = "PASS"
        if scraper_exception_rate > max_scraper_exc_fail:
            exc_status = "FAIL"
        elif scraper_exception_rate > max_scraper_exc_warn:
            exc_status = "WARN"
        runtime_checks.append({
            "name": "Scraper Exception Rate (request-level)",
            "actual": f"{scraper_exception_rate:.1%}",
            "target": f"<= {max_scraper_exc_warn:.0%} warn, <= {max_scraper_exc_fail:.0%} fail",
            "delta": f"{scraper_exception_rate - max_scraper_exc_warn:+.1%} vs warn",
            "status": exc_status,
            "details": "Informational telemetry (not grade-driving).",
        })
        query_status = "PASS"
        if query_p95_ms > max_query_p95_fail:
            query_status = "FAIL"
        elif query_p95_ms > max_query_p95_warn:
            query_status = "WARN"
        runtime_checks.append({
            "name": "Chatbot Query P95 (ms)",
            "actual": f"{query_p95_ms:.0f}",
            "target": f"<= {max_query_p95_warn:.0f} warn, <= {max_query_p95_fail:.0f} fail",
            "delta": f"{query_p95_ms - max_query_p95_warn:+.0f}ms vs warn",
            "status": query_status,
            "details": "User-facing responsiveness guard.",
        })
        runtime_status = self._control_status_from_checks(runtime_checks)

        overall_status = self._control_status_from_checks(
            [
                {"status": cost_status},
                {"status": accuracy_status},
                {"status": completeness_status},
                {"status": runtime_status},
            ]
        )

        history_record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": str(run_id or ""),
            "status": overall_status,
            "cost_status": cost_status,
            "accuracy_status": accuracy_status,
            "completeness_status": completeness_status,
            "runtime_status": runtime_status,
            "runtime_hours": runtime_hours,
            "event_yield_rate": float(event_yield_rate) if event_yield_rate is not None else None,
            "llm_calls_per_event_url": float(llm_calls_per_event_url) if llm_calls_per_event_url is not None else None,
        }
        trend = self._update_and_summarize_run_control_history(output_dir, history_record)
        runtime_start_dt = self._parse_iso_datetime(str((pipeline_runtime or {}).get("start_ts", "") or ""))
        runtime_end_dt = self._parse_iso_datetime(str((pipeline_runtime or {}).get("end_ts", "") or ""))
        events_table_count = None
        try:
            rows = self.db_handler.execute_query("SELECT COUNT(*) FROM events") or []
            events_table_count = int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else None
        except Exception:
            events_table_count = None

        # Persist top-line control metrics for DB-backed trend charts.
        try:
            if total_run_cost_usd is not None:
                self.db_handler.record_metric_observation(
                    run_id=str(run_id or ""),
                    metric_key="total_llm_run_cost_usd",
                    metric_value_numeric=float(total_run_cost_usd),
                    metric_unit="usd",
                    description="Total LLM run cost (OpenRouter + OpenAI) for this validation window",
                    higher_is_better=False,
                    window_start=runtime_start_dt,
                    window_end=runtime_end_dt,
                )
            if events_table_count is not None:
                self.db_handler.record_metric_observation(
                    run_id=str(run_id or ""),
                    metric_key="run_control_events_table_count",
                    metric_value_numeric=float(events_table_count),
                    metric_unit="count",
                    description="Count of rows in events table at end of run",
                    higher_is_better=True,
                    window_start=runtime_end_dt or runtime_start_dt,
                    window_end=runtime_end_dt or runtime_start_dt,
                )
        except Exception:
            pass

        simple_summary = {
            "run_id": str(run_id or ""),
            "cost_usd": total_run_cost_usd,
            "openrouter_cost_usd": openrouter_cost_usd,
            "openai_cost_usd": openai_cost_usd,
            "accuracy_replay_coverage_pct": replay_coverage_pct,
            "accuracy_url_score": round(url_level_score, 1),
            "accuracy_chatbot_score": round(chatbot_avg_score, 1),
            "completeness_event_yield_rate": float(event_yield_rate) if event_yield_rate is not None else None,
            "completeness_urls_with_events": int(fb_ig_funnel.get("urls_with_events", 0) or 0),
            "completeness_urls_passed_for_scraping": int(fb_ig_funnel.get("urls_passed_for_scraping", 0) or 0),
            "completeness_source_distribution_status": source_dist_status_raw,
            "completeness_missing_required_sources": len(missing_sources),
            "completeness_trend_alert_count": len(trend_alerts),
            "runtime_hours": runtime_hours,
            "runtime_start_ts": str((pipeline_runtime or {}).get("start_ts", "")),
            "runtime_end_ts": str((pipeline_runtime or {}).get("end_ts", "")),
            "events_table_count": events_table_count,
        }
        return {
            "run_id": str(run_id or ""),
            "overall_status": overall_status,
            "cost": {"status": cost_status, "checks": cost_checks},
            "accuracy": {"status": accuracy_status, "checks": accuracy_checks},
            "completeness": {"status": completeness_status, "checks": completeness_checks},
            "runtime": {"status": runtime_status, "checks": runtime_checks},
            "simple_summary": simple_summary,
            "trend": trend,
            "runtime_summary": pipeline_runtime or {},
            "openrouter_cost": openrouter_cost or {},
            "openai_cost": openai_cost or {},
        }

    def _build_accuracy_replay_html(self, replay_data: dict | None) -> str:
        """Render replay-accuracy KPI, mismatch categories, and trend mini-chart."""
        if not isinstance(replay_data, dict) or not replay_data:
            return "<p class='error-box'>❌ Replay accuracy unavailable</p>"
        if replay_data.get("error"):
            return f"<p class='error-box'>❌ Replay accuracy failed: {self._escape_html(str(replay_data.get('error')))}</p>"

        coverage_pct = float(replay_data.get("coverage_accuracy_pct", 0.0) or 0.0)
        replay_pct = float(replay_data.get("replay_accuracy_pct", 0.0) or 0.0)
        total_rows = int(replay_data.get("total_rows", 0) or 0)
        true_count = int(replay_data.get("true_count", 0) or 0)
        false_count = int(replay_data.get("false_count", 0) or 0)
        auto_deleted_count = int(replay_data.get("auto_deleted_count", 0) or 0)
        query_text = self._escape_html(str(replay_data.get("query_text", "")))
        run_id = self._escape_html(str(replay_data.get("run_id", "")))
        sql_query = self._escape_html(str(replay_data.get("sql_query", "")))
        strict_time = "Yes" if bool(replay_data.get("strict_time_match", True)) else "No"

        category_counts = replay_data.get("category_counts", {}) if isinstance(replay_data.get("category_counts"), dict) else {}
        category_rows = ""
        for category, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True):
            category_rows += f"<tr><td>{self._escape_html(str(category))}</td><td>{int(count or 0)}</td></tr>"
        if not category_rows:
            category_rows = "<tr><td colspan='2'>No mismatches recorded.</td></tr>"

        actions = replay_data.get("remediation_actions", []) if isinstance(replay_data.get("remediation_actions"), list) else []
        actions_html = "".join(f"<li>{self._escape_html(str(action))}</li>" for action in actions[:8]) or "<li>No remediation actions generated.</li>"

        comparison_rows = replay_data.get("rows", []) if isinstance(replay_data.get("rows"), list) else []
        comparison_html = ""
        for idx, row in enumerate(comparison_rows, start=1):
            if not isinstance(row, dict):
                continue
            is_match = bool(row.get("is_match"))
            match_text = "True" if is_match else "False"
            baseline = row.get("baseline", {}) if isinstance(row.get("baseline"), dict) else {}
            replay = row.get("replay", {}) if isinstance(row.get("replay"), dict) else {}
            mismatch_category = self._escape_html(str(row.get("mismatch_category", "") or ""))
            mismatch_details = self._escape_html(str(row.get("mismatch_details", "") or ""))
            action_taken = self._escape_html(str(row.get("action_taken", "") or ""))
            row_class = "" if is_match else " class='problematic'"

            def _cell(payload: dict, key: str) -> str:
                return self._escape_html(str(payload.get(key, "") or ""))

            comparison_html += (
                f"<tr{row_class}>"
                f"<td>{idx}</td>"
                "<td>Original</td>"
                f"<td>{match_text}</td>"
                f"<td>{_cell(baseline, 'event_name')}</td>"
                f"<td>{_cell(baseline, 'start_date')}</td>"
                f"<td>{_cell(baseline, 'start_time')}</td>"
                f"<td>{_cell(baseline, 'source')}</td>"
                f"<td>{_cell(baseline, 'location')}</td>"
                f"<td>{_cell(baseline, 'url')}</td>"
                f"<td>{mismatch_category}</td>"
                f"<td>{mismatch_details}</td>"
                f"<td>{action_taken}</td>"
                "</tr>"
                f"<tr{row_class}>"
                f"<td>{idx}</td>"
                "<td>Re-scraped</td>"
                f"<td>{match_text}</td>"
                f"<td>{_cell(replay, 'event_name')}</td>"
                f"<td>{_cell(replay, 'start_date')}</td>"
                f"<td>{_cell(replay, 'start_time')}</td>"
                f"<td>{_cell(replay, 'source')}</td>"
                f"<td>{_cell(replay, 'location')}</td>"
                f"<td>{_cell(replay, 'url')}</td>"
                f"<td>{mismatch_category}</td>"
                f"<td>{mismatch_details}</td>"
                f"<td>{action_taken}</td>"
                "</tr>"
            )
        if not comparison_html:
            comparison_html = "<tr><td colspan='12'>No replay row comparisons available.</td></tr>"

        trend = replay_data.get("trend", []) if isinstance(replay_data.get("trend"), list) else []
        trend_points = []
        for item in trend:
            if not isinstance(item, dict):
                continue
            ts = str(item.get("timestamp", "")).strip()
            value = self._safe_float(item.get("value"))
            if not ts or value is None:
                continue
            trend_points.append({"timestamp": ts, "value": float(value)})

        chart_html = "<p>No trend history yet.</p>"
        if len(trend_points) >= 1:
            width = 860
            height = 220
            pad_left = 48
            pad_right = 10
            pad_top = 15
            pad_bottom = 28
            inner_w = max(10, width - pad_left - pad_right)
            inner_h = max(10, height - pad_top - pad_bottom)
            values = [p["value"] for p in trend_points]
            min_v = min(values)
            max_v = max(values)
            span = (max_v - min_v) if (max_v - min_v) > 1e-9 else 1.0
            coords: list[tuple[float, float]] = []
            for idx, point in enumerate(trend_points):
                x = pad_left + (idx / max(1, len(trend_points) - 1)) * inner_w
                y = pad_top + (1.0 - ((point["value"] - min_v) / span)) * inner_h
                coords.append((x, y))
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            last_label = self._escape_html(str(trend_points[-1]["timestamp"])[:10])
            first_label = self._escape_html(str(trend_points[0]["timestamp"])[:10])
            chart_html = (
                f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='Replay accuracy trend'>"
                f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff'/>"
                f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' stroke='#999'/>"
                f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#999'/>"
                f"<polyline points='{polyline}' fill='none' stroke='#007bff' stroke-width='2.5'/>"
                f"<text x='6' y='{pad_top + 8}' font-size='11' fill='#666'>{max_v:.1f}%</text>"
                f"<text x='6' y='{height - pad_bottom}' font-size='11' fill='#666'>{min_v:.1f}%</text>"
                f"<text x='{pad_left}' y='{height - 6}' font-size='11' fill='#666'>{first_label}</text>"
                f"<text x='{width - 90}' y='{height - 6}' font-size='11' fill='#666'>{last_label}</text>"
                "</svg>"
            )

        return (
            "<div class='metric-container'>"
            f"<div class='metric'><div class='metric-value'>{coverage_pct:.2f}%</div><div class='metric-label'>Coverage Accuracy (True / Total)</div></div>"
            f"<div class='metric'><div class='metric-value'>{replay_pct:.2f}%</div><div class='metric-label'>Replay Accuracy (True / URL Rows)</div></div>"
            f"<div class='metric'><div class='metric-value'>{true_count}/{total_rows}</div><div class='metric-label'>True Rows / Total Rows</div></div>"
            f"<div class='metric'><div class='metric-value'>{false_count}</div><div class='metric-label'>False Rows</div></div>"
            f"<div class='metric'><div class='metric-value'>{auto_deleted_count}</div><div class='metric-label'>Auto-deleted (audited)</div></div>"
            "</div>"
            f"<p><strong>Run ID:</strong> {run_id}<br>"
            f"<strong>Query:</strong> {query_text}<br>"
            f"<strong>Strict Time Match:</strong> {strict_time}</p>"
            f"<details><summary>Generated SQL</summary><pre>{sql_query}</pre></details>"
            "<h3>Mismatch Categories</h3>"
            "<table><tr><th>Category</th><th>Count</th></tr>"
            f"{category_rows}</table>"
            "<h3>Remediation Actions</h3>"
            f"<ol>{actions_html}</ol>"
            "<h3>Row-by-Row Original vs Re-scraped</h3>"
            "<table><tr>"
            "<th>#</th><th>Row Type</th><th>Match</th><th>Event Name</th><th>Start Date</th><th>Start Time</th>"
            "<th>Source</th><th>Location</th><th>URL</th><th>Mismatch Category</th><th>Details</th><th>Action</th>"
            "</tr>"
            f"{comparison_html}</table>"
            "<h3>Accuracy Trend (DB persisted metric_observations)</h3>"
            f"{chart_html}"
        )

    def _build_accuracy_replay_rows_html(self, replay_data: dict | None) -> str:
        """Render only the row-by-row original vs replay comparison table."""
        if not isinstance(replay_data, dict) or not replay_data:
            return "<p class='error-box'>❌ Replay accuracy unavailable</p>"
        if replay_data.get("error"):
            return f"<p class='error-box'>❌ Replay accuracy failed: {self._escape_html(str(replay_data.get('error')))}</p>"

        comparison_rows = replay_data.get("rows", []) if isinstance(replay_data.get("rows"), list) else []
        comparison_html = ""
        for idx, row in enumerate(comparison_rows, start=1):
            if not isinstance(row, dict):
                continue
            is_match = bool(row.get("is_match"))
            match_text = "True" if is_match else "False"
            baseline = row.get("baseline", {}) if isinstance(row.get("baseline"), dict) else {}
            replay = row.get("replay", {}) if isinstance(row.get("replay"), dict) else {}
            baseline_url = str(baseline.get("url", "") or "").strip()
            replay_url = str(replay.get("url", "") or "").strip()
            if is_email_like_input(baseline_url) or is_email_like_input(replay_url):
                continue
            mismatch_category = self._escape_html(str(row.get("mismatch_category", "") or ""))
            mismatch_details = self._escape_html(str(row.get("mismatch_details", "") or ""))
            action_taken = self._escape_html(str(row.get("action_taken", "") or ""))
            row_class = "" if is_match else " class='problematic'"

            def _cell(payload: dict, key: str) -> str:
                return self._escape_html(str(payload.get(key, "") or ""))

            comparison_html += (
                f"<tr{row_class}>"
                f"<td>{idx}</td>"
                "<td>Original</td>"
                f"<td>{match_text}</td>"
                f"<td>{_cell(baseline, 'event_name')}</td>"
                f"<td>{_cell(baseline, 'start_date')}</td>"
                f"<td>{_cell(baseline, 'start_time')}</td>"
                f"<td>{_cell(baseline, 'source')}</td>"
                f"<td>{_cell(baseline, 'location')}</td>"
                f"<td>{_cell(baseline, 'url')}</td>"
                f"<td>{mismatch_category}</td>"
                f"<td>{mismatch_details}</td>"
                f"<td>{action_taken}</td>"
                "</tr>"
                f"<tr{row_class}>"
                f"<td>{idx}</td>"
                "<td>Re-scraped</td>"
                f"<td>{match_text}</td>"
                f"<td>{_cell(replay, 'event_name')}</td>"
                f"<td>{_cell(replay, 'start_date')}</td>"
                f"<td>{_cell(replay, 'start_time')}</td>"
                f"<td>{_cell(replay, 'source')}</td>"
                f"<td>{_cell(replay, 'location')}</td>"
                f"<td>{_cell(replay, 'url')}</td>"
                f"<td>{mismatch_category}</td>"
                f"<td>{mismatch_details}</td>"
                f"<td>{action_taken}</td>"
                "</tr>"
            )

        if not comparison_html:
            comparison_html = "<tr><td colspan='12'>No replay row comparisons available.</td></tr>"

        return (
            "<h3>Row-by-Row Original vs Re-scraped</h3>"
            "<table><tr>"
            "<th>#</th><th>Row Type</th><th>Match</th><th>Event Name</th><th>Start Date</th><th>Start Time</th>"
            "<th>Source</th><th>Location</th><th>URL</th><th>Mismatch Category</th><th>Details</th><th>Action</th>"
            "</tr>"
            f"{comparison_html}</table>"
        )

    def _build_metric_trend_svg(
        self,
        metric_key: str,
        title: str,
        days: int = 120,
        value_format: str = "number",
    ) -> str:
        """Render one DB-backed metric trend as a compact inline SVG."""
        trend = self._load_metric_history(metric_key, days=days)
        points = []
        for item in trend:
            ts = str(item.get("timestamp", "")).strip()
            value = self._safe_float(item.get("value"))
            if not ts or value is None:
                continue
            points.append({"timestamp": ts, "value": float(value)})

        if not points:
            return f"<h3>{self._escape_html(title)}</h3><p>No DB trend data available for <code>{self._escape_html(metric_key)}</code>.</p>"

        width = 860
        height = 220
        pad_left = 52
        pad_right = 12
        pad_top = 14
        pad_bottom = 30
        inner_w = max(10, width - pad_left - pad_right)
        inner_h = max(10, height - pad_top - pad_bottom)
        values = [p["value"] for p in points]
        min_v = min(values)
        max_v = max(values)
        span = (max_v - min_v) if (max_v - min_v) > 1e-9 else 1.0
        coords: list[tuple[float, float]] = []
        for idx, point in enumerate(points):
            x = pad_left + (idx / max(1, len(points) - 1)) * inner_w
            y = pad_top + (1.0 - ((point["value"] - min_v) / span)) * inner_h
            coords.append((x, y))
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        first_label = self._escape_html(str(points[0]["timestamp"])[:10])
        last_label = self._escape_html(str(points[-1]["timestamp"])[:10])

        def _fmt(value: float) -> str:
            if value_format == "percent":
                return f"{value:.1f}%"
            if value_format == "usd":
                return f"${value:.2f}"
            if value_format == "hours":
                return f"{value:.2f}h"
            if value_format == "count":
                return f"{int(round(value))}"
            return f"{value:.2f}"

        return (
            f"<h3>{self._escape_html(title)}</h3>"
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='{self._escape_html(title)}'>"
            f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff'/>"
            f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' stroke='#999'/>"
            f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#999'/>"
            f"<polyline points='{polyline}' fill='none' stroke='#007bff' stroke-width='2.5'/>"
            f"<text x='6' y='{pad_top + 9}' font-size='11' fill='#666'>{self._escape_html(_fmt(max_v))}</text>"
            f"<text x='6' y='{height - pad_bottom}' font-size='11' fill='#666'>{self._escape_html(_fmt(min_v))}</text>"
            f"<text x='{pad_left}' y='{height - 7}' font-size='11' fill='#666'>{first_label}</text>"
            f"<text x='{width - 92}' y='{height - 7}' font-size='11' fill='#666'>{last_label}</text>"
            "</svg>"
        )

    def _build_multi_metric_trend_svg(
        self,
        *,
        series: list[dict[str, str]],
        title: str,
        days: int = 120,
        value_format: str = "percent",
    ) -> str:
        """Render multiple DB-backed metric trends on a shared inline SVG."""
        prepared_series: list[dict[str, Any]] = []
        all_values: list[float] = []
        all_labels: set[str] = set()

        for item in series:
            metric_key = str(item.get("metric_key") or "").strip()
            label = str(item.get("label") or metric_key).strip()
            color = str(item.get("color") or "#007bff").strip() or "#007bff"
            if not metric_key:
                continue
            trend = self._load_metric_history(metric_key, days=days)
            points = []
            for point in trend:
                ts = str(point.get("timestamp", "")).strip()
                value = self._safe_float(point.get("value"))
                if not ts or value is None:
                    continue
                day_label = ts[:10]
                points.append({"timestamp": ts, "day_label": day_label, "value": float(value)})
                all_values.append(float(value))
                all_labels.add(day_label)
            if points:
                prepared_series.append(
                    {
                        "metric_key": metric_key,
                        "label": label,
                        "color": color,
                        "points": points,
                    }
                )

        if not prepared_series:
            return f"<h3>{self._escape_html(title)}</h3><p>No DB trend data available for classifier metrics.</p>"

        ordered_labels = sorted(all_labels)
        if len(ordered_labels) == 1:
            label_positions = {ordered_labels[0]: 0.0}
        else:
            label_positions = {
                label: idx / max(1, len(ordered_labels) - 1)
                for idx, label in enumerate(ordered_labels)
            }

        width = 860
        height = 240
        pad_left = 52
        pad_right = 16
        pad_top = 14
        pad_bottom = 34
        inner_w = max(10, width - pad_left - pad_right)
        inner_h = max(10, height - pad_top - pad_bottom)
        min_v = min(all_values)
        max_v = max(all_values)
        span = (max_v - min_v) if (max_v - min_v) > 1e-9 else 1.0
        first_label = self._escape_html(ordered_labels[0])
        last_label = self._escape_html(ordered_labels[-1])

        def _fmt(value: float) -> str:
            if value_format == "percent":
                return f"{value:.1f}%"
            if value_format == "usd":
                return f"${value:.2f}"
            if value_format == "hours":
                return f"{value:.2f}h"
            if value_format == "count":
                return f"{int(round(value))}"
            return f"{value:.2f}"

        polylines: list[str] = []
        legend_items: list[str] = []
        for idx, item in enumerate(prepared_series):
            coords: list[tuple[float, float]] = []
            for point in item["points"]:
                x = pad_left + label_positions[point["day_label"]] * inner_w
                y = pad_top + (1.0 - ((point["value"] - min_v) / span)) * inner_h
                coords.append((x, y))
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            polylines.append(
                f"<polyline points='{polyline}' fill='none' stroke='{self._escape_html(item['color'])}' "
                f"stroke-width='2.5'/>"
            )
            legend_y = pad_top + 14 + idx * 16
            legend_items.append(
                f"<line x1='{width - 230}' y1='{legend_y}' x2='{width - 210}' y2='{legend_y}' "
                f"stroke='{self._escape_html(item['color'])}' stroke-width='2.5'/>"
                f"<text x='{width - 204}' y='{legend_y + 4}' font-size='11' fill='#444'>{self._escape_html(item['label'])}</text>"
            )

        return (
            f"<h3>{self._escape_html(title)}</h3>"
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='{self._escape_html(title)}'>"
            f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff'/>"
            f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' stroke='#999'/>"
            f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#999'/>"
            f"{''.join(polylines)}"
            f"{''.join(legend_items)}"
            f"<text x='6' y='{pad_top + 9}' font-size='11' fill='#666'>{self._escape_html(_fmt(max_v))}</text>"
            f"<text x='6' y='{height - pad_bottom}' font-size='11' fill='#666'>{self._escape_html(_fmt(min_v))}</text>"
            f"<text x='{pad_left}' y='{height - 7}' font-size='11' fill='#666'>{first_label}</text>"
            f"<text x='{width - 92}' y='{height - 7}' font-size='11' fill='#666'>{last_label}</text>"
            "</svg>"
        )

    def _build_top_trend_dashboard_html(self) -> str:
        """Build top-of-report trend dashboard from DB-persisted metric_observations."""
        parts = [
            self._build_metric_trend_svg(
                metric_key="validation_replay_accuracy_pct",
                title="Accuracy Trend (DB persisted metric_observations)",
                days=180,
                value_format="percent",
            ),
            self._build_multi_metric_trend_svg(
                title="Classifier Trends",
                days=180,
                value_format="percent",
                series=[
                    {
                        "metric_key": "classifier_ml_usage_pct",
                        "label": "ML Usage %",
                        "color": "#d94841",
                    },
                    {
                        "metric_key": "classifier_ml_replay_url_accuracy_pct",
                        "label": "ML Replay URL Accuracy %",
                        "color": "#1f77b4",
                    },
                    {
                        "metric_key": "classifier_rule_replay_url_accuracy_pct",
                        "label": "Rule Replay URL Accuracy %",
                        "color": "#2ca02c",
                    },
                ],
            ),
            self._build_metric_trend_svg(
                metric_key="total_llm_run_cost_usd",
                title="Cost Trend (Total Cost per Run)",
                days=180,
                value_format="usd",
            ),
            self._build_metric_trend_svg(
                metric_key="run_control_runtime_hours",
                title="Runtime Trend (Total Pipeline Runtime Hours)",
                days=180,
                value_format="hours",
            ),
            self._build_metric_trend_svg(
                metric_key="run_control_events_table_count",
                title="Number of Events Trend (Events Table Count at End of Run)",
                days=180,
                value_format="count",
            ),
        ]
        return "".join(parts)

    def _build_run_control_panel_html(self, panel_data: dict, action_queue: dict | None = None) -> str:
        """Render operator-first control panel for cost, accuracy, completeness, and runtime."""
        if not panel_data:
            return "<p class='error-box'>❌ Run control panel unavailable</p>"
        cost = panel_data.get("cost", {}) or {}
        accuracy = panel_data.get("accuracy", {}) or {}
        completeness = panel_data.get("completeness", {}) or {}
        runtime = panel_data.get("runtime", {}) or {}
        simple_summary = panel_data.get("simple_summary", {}) or {}
        trend = panel_data.get("trend", {}) or {}
        overall_status = str(panel_data.get("overall_status", "WARN") or "WARN")

        html = (
            "<div class='metric-container'>"
            f"<div class='metric'><div class='metric-value'><span class='{self._status_class(overall_status)}'>{self._escape_html(overall_status)}</span></div><div class='metric-label'>Control Panel Status</div></div>"
            f"<div class='metric'><div class='metric-value'><span class='{self._status_class(cost.get('status', 'WARN'))}'>{self._escape_html(str(cost.get('status', 'WARN')))}</span></div><div class='metric-label'>Cost Status</div></div>"
            f"<div class='metric'><div class='metric-value'><span class='{self._status_class(accuracy.get('status', 'WARN'))}'>{self._escape_html(str(accuracy.get('status', 'WARN')))}</span></div><div class='metric-label'>Accuracy Status</div></div>"
            f"<div class='metric'><div class='metric-value'><span class='{self._status_class(completeness.get('status', 'WARN'))}'>{self._escape_html(str(completeness.get('status', 'WARN')))}</span></div><div class='metric-label'>Completeness Status</div></div>"
            f"<div class='metric'><div class='metric-value'><span class='{self._status_class(runtime.get('status', 'WARN'))}'>{self._escape_html(str(runtime.get('status', 'WARN')))}</span></div><div class='metric-label'>Runtime Status</div></div>"
            "</div>"
        )
        html += f"<p><strong>Run ID:</strong> {self._escape_html(str(panel_data.get('run_id', '')))}</p>"
        cost_usd = simple_summary.get("cost_usd")
        cost_display = f"${float(cost_usd):.4f}" if cost_usd is not None else "N/A"
        openrouter_cost_usd = simple_summary.get("openrouter_cost_usd")
        openai_cost_usd = simple_summary.get("openai_cost_usd")
        openrouter_display = f"${float(openrouter_cost_usd):.4f}" if openrouter_cost_usd is not None else "N/A"
        openai_display = f"${float(openai_cost_usd):.4f}" if openai_cost_usd is not None else "N/A"
        event_yield = simple_summary.get("completeness_event_yield_rate")
        event_yield_display = f"{float(event_yield):.1%}" if event_yield is not None else "N/A"
        replay_accuracy = simple_summary.get("accuracy_replay_coverage_pct")
        replay_accuracy_display = f"{float(replay_accuracy):.2f}%" if replay_accuracy is not None else "N/A"
        html += (
            "<h3>Simple Run Summary</h3>"
            "<table><tr><th>Metric</th><th>Value</th><th>Formula / Notes</th></tr>"
            f"<tr><td>Cost</td><td>{self._escape_html(cost_display)}</td>"
            f"<td>Total = OpenRouter ({self._escape_html(openrouter_display)}) + "
            f"OpenAI ({self._escape_html(openai_display)}) usage windows aligned to run time</td></tr>"
            f"<tr><td>Accuracy</td><td>{float(simple_summary.get('accuracy_url_score', 0) or 0):.1f} "
            f"(replay={self._escape_html(replay_accuracy_display)})</td>"
            f"<td>URL-level reliability score plus replay coverage accuracy (strict row match / total rows)</td></tr>"
            f"<tr><td>Completeness</td><td>{self._escape_html(event_yield_display)}</td>"
            f"<td>urls_with_events / urls_passed_for_scraping = "
            f"{int(simple_summary.get('completeness_urls_with_events', 0) or 0)} / "
            f"{int(simple_summary.get('completeness_urls_passed_for_scraping', 0) or 0)}; "
            f"source_distribution_status={self._escape_html(str(simple_summary.get('completeness_source_distribution_status', 'N/A')))}; "
            f"missing_required_sources={int(simple_summary.get('completeness_missing_required_sources', 0) or 0)}; "
            f"trend_alerts={int(simple_summary.get('completeness_trend_alert_count', 0) or 0)}</td></tr>"
            f"<tr><td>Time Spent</td><td>{float(simple_summary.get('runtime_hours', 0.0) or 0.0):.2f}h</td>"
            f"<td>{self._escape_html(str(simple_summary.get('runtime_start_ts', '')))} to "
            f"{self._escape_html(str(simple_summary.get('runtime_end_ts', '')))}</td></tr>"
            "</table>"
        )

        def _build_check_table(title: str, checks: list[dict]) -> str:
            block = f"<h3>{self._escape_html(title)}</h3>"
            block += "<table><tr><th>KPI</th><th>Actual</th><th>Target</th><th>Delta</th><th>Status</th><th>Details</th></tr>"
            for row in checks or []:
                block += (
                    "<tr>"
                    f"<td>{self._escape_html(str(row.get('name', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('actual', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('target', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('delta', '')))}</td>"
                    f"<td><span class='{self._status_class(str(row.get('status', 'WARN')))}'>{self._escape_html(str(row.get('status', 'WARN')))}</span></td>"
                    f"<td>{self._escape_html(str(row.get('details', '')))}</td>"
                    "</tr>"
                )
            block += "</table>"
            return block

        html += _build_check_table("Cost KPIs", cost.get("checks", []) or [])
        html += _build_check_table("Accuracy KPIs", accuracy.get("checks", []) or [])
        html += _build_check_table("Completeness KPIs", completeness.get("checks", []) or [])
        html += _build_check_table("Runtime KPIs", runtime.get("checks", []) or [])

        runs_considered = int(trend.get("runs_considered", 0) or 0)
        html += (
            "<h3>Trend Snapshot (Last 7 Prior Runs)</h3>"
            "<table><tr><th>KPI</th><th>Current</th><th>Prev Avg</th><th>Delta</th></tr>"
        )

        def _trend_row(name: str, current: float | None, avg_prev: float | None, as_pct: bool = False) -> str:
            if current is None:
                cur_s = "N/A"
            else:
                cur_s = f"{current:.1%}" if as_pct else f"{current:.2f}"
            if avg_prev is None:
                avg_s = "N/A"
                delta_s = "N/A"
            else:
                avg_s = f"{avg_prev:.1%}" if as_pct else f"{avg_prev:.2f}"
                delta = current - avg_prev if current is not None else 0.0
                delta_s = f"{delta:+.1%}" if as_pct else f"{delta:+.2f}"
            return (
                "<tr>"
                f"<td>{self._escape_html(name)}</td>"
                f"<td>{self._escape_html(cur_s)}</td>"
                f"<td>{self._escape_html(avg_s)}</td>"
                f"<td>{self._escape_html(delta_s)}</td>"
                "</tr>"
            )

        runtime_hours_current = float((panel_data.get("runtime_summary", {}) or {}).get("runtime_hours", 0.0) or 0.0)
        event_yield_current = None
        for check in accuracy.get("checks", []) or []:
            if str(check.get("name", "")).startswith("Event Yield"):
                try:
                    raw = str(check.get("actual", "")).replace("%", "").strip()
                    event_yield_current = float(raw) / 100.0 if raw and raw != "N/A" else None
                except Exception:
                    event_yield_current = None
                break
        calls_per_event_current = None
        for check in cost.get("checks", []) or []:
            if str(check.get("name", "")).startswith("LLM Calls / Event URL"):
                try:
                    raw = str(check.get("actual", "")).strip()
                    calls_per_event_current = float(raw) if raw and raw != "N/A" else None
                except Exception:
                    calls_per_event_current = None
                break

        html += _trend_row("Pipeline Runtime (hours)", runtime_hours_current, trend.get("runtime_hours_avg_prev"), as_pct=False)
        html += _trend_row("Event Yield Rate", event_yield_current, trend.get("event_yield_rate_avg_prev"), as_pct=True)
        html += _trend_row("LLM Calls / Event URL", calls_per_event_current, trend.get("llm_calls_per_event_url_avg_prev"), as_pct=False)
        html += "</table>"
        html += (
            f"<p><strong>History Path:</strong> {self._escape_html(str(trend.get('path', '')))} | "
            f"<strong>Prior Runs Used:</strong> {runs_considered}</p>"
        )

        queue_items = (action_queue or {}).get("items", []) if isinstance(action_queue, dict) else []
        if queue_items:
            mapped: list[dict] = []
            for item in queue_items:
                source = str(item.get("source_section", "") or "")
                if source in {
                    "LLM Provider Activity",
                    "LLM Extraction Quality Scorecard",
                    "Scraper Network Reliability",
                    "FB/IG URL Funnel",
                    "Source Distribution Check",
                    "Chatbot Performance",
                    "Reliability Scorecard",
                }:
                    mapped.append(item)
            if mapped:
                html += "<h3>Priority Actions (Cost/Accuracy/Completeness/Runtime)</h3>"
                html += "<table><tr><th>Priority</th><th>Source Section</th><th>Metric Key</th><th>Action</th><th>Acceptance Test</th></tr>"
                for item in mapped[:12]:
                    html += (
                        "<tr>"
                        f"<td>{self._escape_html(str(item.get('priority', '')))}</td>"
                        f"<td>{self._escape_html(str(item.get('source_section', '')))}</td>"
                        f"<td>{self._escape_html(str(item.get('metric_key', '')))}</td>"
                        f"<td>{self._escape_html(str(item.get('title', '')))}</td>"
                        f"<td>{self._escape_html(str(item.get('acceptance_test', '')))}</td>"
                        "</tr>"
                    )
                html += "</table>"
        return html

    def _build_completeness_kpis_only_html(self, panel_data: dict | None) -> str:
        """Render only completeness KPI checks for the slim report."""
        if not panel_data:
            return "<p class='error-box'>❌ Completeness KPIs unavailable</p>"
        completeness = (panel_data.get("completeness", {}) or {})
        checks = completeness.get("checks", []) or []
        if not checks:
            return "<p>No completeness KPI checks available.</p>"

        html = "<table><tr><th>KPI</th><th>Actual</th><th>Target</th><th>Delta</th><th>Status</th><th>Details</th></tr>"
        for row in checks:
            html += (
                "<tr>"
                f"<td>{self._escape_html(str(row.get('name', '')))}</td>"
                f"<td>{self._escape_html(str(row.get('actual', '')))}</td>"
                f"<td>{self._escape_html(str(row.get('target', '')))}</td>"
                f"<td>{self._escape_html(str(row.get('delta', '')))}</td>"
                f"<td><span class='{self._status_class(str(row.get('status', 'WARN')))}'>{self._escape_html(str(row.get('status', 'WARN')))}</span></td>"
                f"<td>{self._escape_html(str(row.get('details', '')))}</td>"
                "</tr>"
            )
        html += "</table>"
        return html

    def _summarize_llm_provider_activity(self, report_timestamp: str | None) -> dict:
        """
        Summarize provider-level LLM activity from recent logs.
        """
        hours_window = int(
            self.validation_config.get('reporting', {}).get('llm_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        log_files = self._get_llm_activity_log_files()

        providers = ("openai", "openrouter", "mistral", "gemini")
        stats = {
            p: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "rate_limits": 0,
            }
            for p in providers
        }
        file_attempts: Counter = Counter()
        model_attempts: Counter = Counter()
        total_provider_exhausted = 0
        total_accesses = 0
        lines_scanned = 0

        def _inc_if(line: str, needle: str, provider: str, metric: str) -> None:
            if needle in line:
                stats[provider][metric] += 1

        for path in log_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        ts = self._parse_log_timestamp(line)
                        if ts is None or ts < start_ts or ts > end_ts:
                            continue

                        lines_scanned += 1
                        low = line.lower()

                        parsed_query = self._parse_llm_query_provider_model(line)
                        if parsed_query:
                            provider, model = parsed_query
                            total_accesses += 1
                            file_attempts[path] += 1
                            if provider in stats:
                                stats[provider]["attempts"] += 1
                                if model:
                                    model_attempts[f"{provider}:{model}"] += 1

                        _inc_if(low, "query_llm(): openai response received", "openai", "successes")
                        _inc_if(low, "query_llm(): openrouter response received", "openrouter", "successes")
                        _inc_if(low, "query_llm(): mistral response received", "mistral", "successes")
                        _inc_if(low, "query_llm(): gemini response received", "gemini", "successes")

                        _inc_if(low, "query_llm(): openai query failed", "openai", "failures")
                        _inc_if(low, "query_llm(): openrouter query failed", "openrouter", "failures")
                        _inc_if(low, "query_llm(): mistral query failed", "mistral", "failures")
                        _inc_if(low, "query_llm(): gemini query failed", "gemini", "failures")

                        if "request timed out" in low or "read timed out" in low:
                            if "openai query failed" in low:
                                stats["openai"]["timeouts"] += 1
                            if "gemini query failed" in low:
                                stats["gemini"]["timeouts"] += 1

                        if "status 429" in low or "too many requests" in low or "rate limit exceeded" in low:
                            if "openrouter query failed" in low:
                                stats["openrouter"]["rate_limits"] += 1
                            if "mistral query failed" in low:
                                stats["mistral"]["rate_limits"] += 1
                            if "openai query failed" in low:
                                stats["openai"]["rate_limits"] += 1
                            if "gemini query failed" in low:
                                stats["gemini"]["rate_limits"] += 1

                        if "all configured llm providers failed to provide a response" in low:
                            total_provider_exhausted += 1
            except Exception as e:
                logging.warning("_summarize_llm_provider_activity: Failed to parse %s: %s", path, e)

        total_provider_attempts = sum(v["attempts"] for v in stats.values())
        total_attempts = total_accesses
        total_rate_limits = sum(v["rate_limits"] for v in stats.values())
        total_timeouts = sum(v["timeouts"] for v in stats.values())
        gemini_attempts = int(stats.get("gemini", {}).get("attempts", 0))

        thresholds_cfg = (
            self.validation_config.get("reporting", {}).get("llm_activity_thresholds", {})
        )
        high_cfg = thresholds_cfg.get("high", {}) if isinstance(thresholds_cfg, dict) else {}
        medium_cfg = thresholds_cfg.get("medium", {}) if isinstance(thresholds_cfg, dict) else {}

        high_attempts = int(high_cfg.get("attempts", 150) or 150)
        high_exhausted = int(high_cfg.get("provider_exhausted", 10) or 10)
        high_rate_limits = int(high_cfg.get("rate_limits", 25) or 25)
        high_timeouts = int(high_cfg.get("timeouts", 15) or 15)
        high_gemini = int(high_cfg.get("gemini_attempts", 20) or 20)

        medium_attempts = int(medium_cfg.get("attempts", 60) or 60)
        medium_exhausted = int(medium_cfg.get("provider_exhausted", 3) or 3)
        medium_rate_limits = int(medium_cfg.get("rate_limits", 10) or 10)
        medium_timeouts = int(medium_cfg.get("timeouts", 5) or 5)
        medium_gemini = int(medium_cfg.get("gemini_attempts", 5) or 5)

        pressure_reasons: list[str] = []
        if total_attempts >= high_attempts:
            pressure_reasons.append(f"high attempt volume ({total_attempts})")
        if total_provider_exhausted >= high_exhausted:
            pressure_reasons.append(f"many full fallback failures ({total_provider_exhausted})")
        if total_rate_limits >= high_rate_limits:
            pressure_reasons.append(f"heavy rate limiting ({total_rate_limits})")
        if total_timeouts >= high_timeouts:
            pressure_reasons.append(f"many timeouts ({total_timeouts})")
        if gemini_attempts >= high_gemini:
            pressure_reasons.append(f"frequent Gemini fallback ({gemini_attempts})")

        pressure_level = "LOW"
        if pressure_reasons:
            pressure_level = "HIGH"
        else:
            medium_signals = [
                total_attempts >= medium_attempts,
                total_provider_exhausted >= medium_exhausted,
                total_rate_limits >= medium_rate_limits,
                total_timeouts >= medium_timeouts,
                gemini_attempts >= medium_gemini,
            ]
            if any(medium_signals):
                pressure_level = "MEDIUM"

        return {
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "providers": stats,
            "total_attempts": total_attempts,
            "total_accesses": total_accesses,
            "denominator_scope": "all_logs",
            "denominator_label": "Total LLM accesses across all configured logs in reporting window",
            "total_provider_attempts": total_provider_attempts,
            "provider_exhausted_count": total_provider_exhausted,
            "lines_scanned": lines_scanned,
            "top_files": file_attempts.most_common(8),
            "top_models": model_attempts.most_common(12),
            "total_rate_limits": total_rate_limits,
            "total_timeouts": total_timeouts,
            "pressure_level": pressure_level,
            "pressure_reasons": pressure_reasons,
        }

    def _summarize_llm_extraction_quality(self, report_timestamp: str | None) -> dict:
        """
        Summarize event-extraction quality from LLM logs in the recent reporting window.
        """
        hours_window = int(
            self.validation_config.get('reporting', {}).get('llm_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        log_files = self._get_llm_activity_log_files()
        pending_models: list[str] = []
        attempt_to_model: dict[tuple[str, int], str] = {}
        url_state: dict[str, dict] = {}
        model_stats: dict[str, dict[str, int]] = {}
        top_failing_urls: Counter = Counter()
        top_failing_url_model_pairs: Counter = Counter()

        query_model_re = re.compile(r"query_llm\(\): querying (\w+) model (\S+)", re.IGNORECASE)
        attempt_re = re.compile(r"def process_llm_response: URL (.+?) attempt=(\d+) llm_response_len=", re.IGNORECASE)
        no_events_re = re.compile(
            r"def process_llm_response: URL (.+?) attempt=(\d+) returned 'no events found'",
            re.IGNORECASE,
        )
        short_re = re.compile(
            r"def process_llm_response: URL (.+?) attempt=(\d+) returned too-short response",
            re.IGNORECASE,
        )
        retry_re = re.compile(
            r"def process_llm_response: URL (.+?) attempt=(\d+) returned non-parseable response; retrying once",
            re.IGNORECASE,
        )
        success_re = re.compile(
            r"def process_llm_response: URL (.+?) marked as relevant with events written to the database",
            re.IGNORECASE,
        )
        fail_re = re.compile(
            r"def process_llm_response: Failed to process LLM response for URL: (.+)",
            re.IGNORECASE,
        )
        schema_re = re.compile(
            r"write_events_to_db: datetime conversion KeyError=.* url=(\S+)\s+parent_url=",
            re.IGNORECASE,
        )
        structured_marker = "llm_extract_attempt_result:"
        structured_event_count = 0

        def _ensure_url(url: str) -> dict:
            if url not in url_state:
                url_state[url] = {
                    "attempts": 0,
                    "status": "unknown",
                    "retries": 0,
                    "too_short": 0,
                    "no_events": 0,
                    "hard_failures": 0,
                    "schema_errors": 0,
                    "last_model": "unknown",
                }
            return url_state[url]

        def _ensure_model(model_key: str) -> dict[str, int]:
            if model_key not in model_stats:
                model_stats[model_key] = {
                    "attempts": 0,
                    "successes": 0,
                    "hard_failures": 0,
                    "no_events": 0,
                    "too_short": 0,
                    "retries": 0,
                    "schema_errors": 0,
                }
            return model_stats[model_key]

        def _resolve_model(url: str, attempt_num: int | None = None) -> str:
            if attempt_num is not None:
                model = attempt_to_model.get((url, attempt_num))
                if model:
                    return model
            return _ensure_url(url).get("last_model", "unknown")

        for path in log_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        ts = self._parse_log_timestamp(line)
                        if ts is None or ts < start_ts or ts > end_ts:
                            continue
                        low = line.lower()

                        if structured_marker in low:
                            try:
                                payload_raw = line.split(structured_marker, 1)[1].strip()
                                payload = json.loads(payload_raw)
                                url = str(payload.get("url", "") or "").strip()
                                attempt_num = int(payload.get("attempt", 0) or 0)
                                provider = str(payload.get("provider", "unknown") or "unknown").strip().lower()
                                model_name = str(payload.get("model", "unknown") or "unknown").strip()
                                outcome = str(payload.get("outcome", "unknown") or "unknown").strip().lower()
                                if not url:
                                    continue
                                model_key = f"{provider}:{model_name}" if provider != "unknown" else model_name
                                state = _ensure_url(url)
                                if attempt_num > 0:
                                    attempt_to_model[(url, attempt_num)] = model_key
                                state["last_model"] = model_key
                                state["attempts"] += 1
                                model_bucket = _ensure_model(model_key)
                                model_bucket["attempts"] += 1
                                structured_event_count += 1

                                if outcome == "success":
                                    state["status"] = "success"
                                    model_bucket["successes"] += 1
                                elif outcome == "no_events":
                                    state["status"] = "no_events"
                                    state["no_events"] += 1
                                    model_bucket["no_events"] += 1
                                elif outcome == "too_short":
                                    state["status"] = "too_short"
                                    state["too_short"] += 1
                                    model_bucket["too_short"] += 1
                                elif outcome in {"parse_retry", "no_response_retry"}:
                                    state["retries"] += 1
                                    model_bucket["retries"] += 1
                                elif outcome == "hard_failure":
                                    state["status"] = "hard_failure"
                                    state["hard_failures"] += 1
                                    model_bucket["hard_failures"] += 1
                                    top_failing_urls[url] += 1
                                    top_failing_url_model_pairs[(url, str(state.get("last_model", "unknown")))] += 1
                                continue
                            except Exception:
                                pass

                        match_model = query_model_re.search(line)
                        if structured_event_count > 0:
                            # New structured extraction logs provide exact provider/model attribution.
                            # Skip heuristic queue matching once structured lines are present.
                            continue
                        if match_model:
                            provider = match_model.group(1).lower()
                            model = match_model.group(2).strip()
                            pending_models.append(f"{provider}:{model}")
                            continue

                        match_attempt = attempt_re.search(line)
                        if match_attempt:
                            url = match_attempt.group(1).strip()
                            attempt_num = int(match_attempt.group(2))
                            state = _ensure_url(url)
                            state["attempts"] += 1
                            model_key = pending_models.pop(0) if pending_models else "unknown"
                            state["last_model"] = model_key
                            attempt_to_model[(url, attempt_num)] = model_key
                            _ensure_model(model_key)["attempts"] += 1
                            continue

                        match_no_events = no_events_re.search(line)
                        if match_no_events:
                            url = match_no_events.group(1).strip()
                            attempt_num = int(match_no_events.group(2))
                            state = _ensure_url(url)
                            state["status"] = "no_events"
                            state["no_events"] += 1
                            _ensure_model(_resolve_model(url, attempt_num))["no_events"] += 1
                            continue

                        match_short = short_re.search(line)
                        if match_short:
                            url = match_short.group(1).strip()
                            attempt_num = int(match_short.group(2))
                            state = _ensure_url(url)
                            state["status"] = "too_short"
                            state["too_short"] += 1
                            _ensure_model(_resolve_model(url, attempt_num))["too_short"] += 1
                            continue

                        match_retry = retry_re.search(line)
                        if match_retry:
                            url = match_retry.group(1).strip()
                            attempt_num = int(match_retry.group(2))
                            state = _ensure_url(url)
                            state["retries"] += 1
                            _ensure_model(_resolve_model(url, attempt_num))["retries"] += 1
                            continue

                        match_success = success_re.search(line)
                        if match_success:
                            url = match_success.group(1).strip()
                            state = _ensure_url(url)
                            state["status"] = "success"
                            _ensure_model(state["last_model"])["successes"] += 1
                            continue

                        match_fail = fail_re.search(line)
                        if match_fail:
                            url = match_fail.group(1).strip()
                            state = _ensure_url(url)
                            if state["status"] != "success":
                                state["status"] = "hard_failure"
                            state["hard_failures"] += 1
                            top_failing_urls[url] += 1
                            top_failing_url_model_pairs[(url, str(state.get("last_model", "unknown")))] += 1
                            _ensure_model(state["last_model"])["hard_failures"] += 1
                            continue

                        match_schema = schema_re.search(line)
                        if match_schema:
                            url = match_schema.group(1).strip()
                            state = _ensure_url(url)
                            if state["status"] != "success":
                                state["status"] = "schema_error"
                            state["schema_errors"] += 1
                            top_failing_urls[url] += 1
                            top_failing_url_model_pairs[(url, str(state.get("last_model", "unknown")))] += 1
                            _ensure_model(state["last_model"])["schema_errors"] += 1
                            continue
            except Exception as e:
                logging.warning("_summarize_llm_extraction_quality: Failed to parse %s: %s", path, e)

        total_urls = len(url_state)
        successful_urls = sum(1 for s in url_state.values() if s.get("status") == "success")
        hard_failed_urls = sum(
            1 for s in url_state.values() if s.get("status") in {"hard_failure", "schema_error"}
        )
        no_events_urls = sum(1 for s in url_state.values() if s.get("status") == "no_events")
        too_short_urls = sum(1 for s in url_state.values() if s.get("status") == "too_short")
        total_attempts = sum(int(s.get("attempts", 0)) for s in url_state.values())
        total_retries = sum(int(s.get("retries", 0)) for s in url_state.values())
        total_schema_errors = sum(int(s.get("schema_errors", 0)) for s in url_state.values())

        parse_success_rate = (successful_urls / total_urls) if total_urls else 0.0
        hard_failure_rate = (hard_failed_urls / total_urls) if total_urls else 0.0

        top_models: list[tuple[str, dict]] = sorted(
            model_stats.items(),
            key=lambda item: (-int(item[1].get("attempts", 0)), item[0]),
        )[:15]

        return {
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "total_urls": total_urls,
            "total_attempts": total_attempts,
            "successful_urls": successful_urls,
            "hard_failed_urls": hard_failed_urls,
            "no_events_urls": no_events_urls,
            "too_short_urls": too_short_urls,
            "total_retries": total_retries,
            "schema_errors": total_schema_errors,
            "parse_success_rate": round(parse_success_rate, 4),
            "hard_failure_rate": round(hard_failure_rate, 4),
            "top_failing_urls": top_failing_urls.most_common(15),
            "top_failing_url_model_pairs": [
                {"url": str(url), "model": str(model), "count": int(count)}
                for (url, model), count in top_failing_url_model_pairs.most_common(15)
            ],
            "models": top_models,
        }

    def _summarize_chatbot_performance(self, report_timestamp: str | None) -> dict:
        """Summarize chatbot request performance from DB metrics (preferred) or timing logs."""
        reporting_cfg = self.validation_config.get("reporting", {}) if isinstance(self.validation_config, dict) else {}
        perf_cfg = reporting_cfg.get("chatbot_performance_thresholds", {}) if isinstance(reporting_cfg, dict) else {}

        hours_window = int(reporting_cfg.get("chatbot_performance_hours", reporting_cfg.get("llm_activity_hours", 24)) or 24)
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        start_re = re.compile(
            r"chatbot_timing_start:\s*request_id=(\S+)\s+endpoint=(\S+)\s+stage=(\S+)(?:\s+(.*))?",
            re.IGNORECASE,
        )
        end_re = re.compile(
            r"chatbot_timing_end:\s*request_id=(\S+)\s+endpoint=(\S+)\s+stage=(\S+)\s+duration_ms=([0-9.]+)(?:\s+(.*))?",
            re.IGNORECASE,
        )
        trace_req_re = re.compile(
            r"chatbot_trace_request:\s*request_id=(\S+)\s+endpoint=(\S+)\s+user_input=(.*)$",
            re.IGNORECASE,
        )
        trace_sql_re = re.compile(
            r"chatbot_trace_sql:\s*request_id=(\S+)\s+endpoint=(\S+)\s+stage=(\S+)\s+sql=(.*)$",
            re.IGNORECASE,
        )

        def _parse_kv(extra: str) -> dict[str, str]:
            out: dict[str, str] = {}
            if not extra:
                return out
            for k, v in re.findall(r"([a-zA-Z_]+)=([^\s]+)", extra):
                out[k] = v
            return out

        def _percentile(values: list[float], percentile: float) -> float:
            if not values:
                return 0.0
            scoped = sorted(values)
            if len(scoped) == 1:
                return scoped[0]
            idx = int(round((percentile / 100.0) * (len(scoped) - 1)))
            idx = max(0, min(len(scoped) - 1, idx))
            return float(scoped[idx])

        def _compute_status(query_latencies: list[float], unresolved_count: int) -> tuple[str, list[str], float, float, float, int, int]:
            query_p95_local = _percentile(query_latencies, 95) if query_latencies else 0.0
            slow_query_ms_local = float(perf_cfg.get("slow_query_ms", 15000) or 15000)
            warn_query_p95_ms_local = float(perf_cfg.get("warn_query_p95_ms", 10000) or 10000)
            critical_query_p95_ms_local = float(perf_cfg.get("critical_query_p95_ms", 20000) or 20000)
            warn_unresolved_count_local = int(perf_cfg.get("warn_unresolved_count", 1) or 1)
            critical_unresolved_count_local = int(perf_cfg.get("critical_unresolved_count", 3) or 3)
            status_local = "HEALTHY"
            reasons_local: list[str] = []
            if query_p95_local >= critical_query_p95_ms_local or unresolved_count >= critical_unresolved_count_local:
                status_local = "DEGRADED"
            elif query_p95_local >= warn_query_p95_ms_local or unresolved_count >= warn_unresolved_count_local:
                status_local = "WATCH"
            if query_p95_local >= warn_query_p95_ms_local:
                reasons_local.append(f"query p95 latency high ({query_p95_local:.1f}ms)")
            if unresolved_count > 0:
                reasons_local.append(f"unfinished requests ({unresolved_count})")
            return (
                status_local,
                reasons_local,
                slow_query_ms_local,
                warn_query_p95_ms_local,
                critical_query_p95_ms_local,
                warn_unresolved_count_local,
                critical_unresolved_count_local,
            )

        # Prefer DB-backed metrics for durable, longitudinal reporting.
        try:
            requests_rows = self.db_handler.execute_query(
                """
                SELECT request_id, endpoint, duration_ms, result_type, user_input, sql_snippet, started_at, finished_at
                FROM chatbot_request_metrics
                WHERE started_at >= :start_ts AND started_at <= :end_ts
                ORDER BY started_at DESC
                """,
                {"start_ts": start_ts, "end_ts": end_ts},
            ) or []
            stages_rows = self.db_handler.execute_query(
                """
                SELECT request_id, endpoint, stage, duration_ms, started_at, finished_at
                FROM chatbot_stage_metrics
                WHERE started_at >= :start_ts AND started_at <= :end_ts
                ORDER BY started_at DESC
                """,
                {"start_ts": start_ts, "end_ts": end_ts},
            ) or []
            if requests_rows or stages_rows:
                query_entries: list[dict] = []
                confirm_entries: list[dict] = []
                unresolved_ids: list[str] = []
                stage_durations: dict[str, list[float]] = {}
                for row in requests_rows:
                    req_id, endpoint, duration_ms, result_type, question, sql_snippet, started_at_row, finished_at_row = row
                    if duration_ms is None or finished_at_row is None:
                        unresolved_ids.append(str(req_id))
                        continue
                    entry = {
                        "request_id": str(req_id),
                        "endpoint": str(endpoint or ""),
                        "duration_ms": round(float(duration_ms or 0.0), 1),
                        "result_type": str(result_type or ""),
                        "question": str(question or ""),
                        "sql": str(sql_snippet or ""),
                    }
                    if str(endpoint) == "/query":
                        query_entries.append(entry)
                    elif str(endpoint) == "/confirm":
                        confirm_entries.append(entry)

                for row in stages_rows:
                    _, _, stage, duration_ms, _, _ = row
                    if stage is None or duration_ms is None:
                        continue
                    stage_durations.setdefault(str(stage), []).append(float(duration_ms))

                query_latencies = [float(e.get("duration_ms", 0.0) or 0.0) for e in query_entries]
                confirm_latencies = [float(e.get("duration_ms", 0.0) or 0.0) for e in confirm_entries]
                query_avg = (sum(query_latencies) / len(query_latencies)) if query_latencies else 0.0
                confirm_avg = (sum(confirm_latencies) / len(confirm_latencies)) if confirm_latencies else 0.0

                (
                    status,
                    reasons,
                    slow_query_ms,
                    warn_query_p95_ms,
                    critical_query_p95_ms,
                    warn_unresolved_count,
                    critical_unresolved_count,
                ) = _compute_status(query_latencies, len(unresolved_ids))
                query_p95 = _percentile(query_latencies, 95) if query_latencies else 0.0

                slow_entries = sorted(
                    [
                        e for e in (query_entries + confirm_entries)
                        if float(e.get("duration_ms", 0.0) or 0.0) >= slow_query_ms
                    ],
                    key=lambda x: float(x.get("duration_ms", 0.0) or 0.0),
                    reverse=True,
                )[:20]

                stage_summary: list[dict] = []
                for stage, values in stage_durations.items():
                    if not values:
                        continue
                    avg_val = sum(values) / len(values)
                    stage_summary.append({
                        "stage": stage,
                        "count": len(values),
                        "avg_ms": round(avg_val, 1),
                        "p95_ms": round(_percentile(values, 95), 1),
                        "max_ms": round(max(values), 1),
                    })
                stage_summary = sorted(
                    stage_summary,
                    key=lambda x: float(x.get("p95_ms", 0.0) or 0.0),
                    reverse=True,
                )[:15]

                traced_question_count = sum(1 for e in slow_entries if e.get("question"))
                traced_sql_count = sum(1 for e in slow_entries if e.get("sql"))
                return {
                    "source": "db",
                    "paths": ["chatbot_request_metrics", "chatbot_stage_metrics"],
                    "window_hours": hours_window,
                    "start_ts": start_ts.isoformat(sep=" "),
                    "end_ts": end_ts.isoformat(sep=" "),
                    "scanned_lines": 0,
                    "query_request_count": len(query_entries),
                    "confirm_request_count": len(confirm_entries),
                    "query_latency_ms": {
                        "avg": round(query_avg, 1),
                        "p50": round(_percentile(query_latencies, 50), 1) if query_latencies else 0.0,
                        "p95": round(query_p95, 1),
                        "max": round(max(query_latencies), 1) if query_latencies else 0.0,
                    },
                    "confirm_latency_ms": {
                        "avg": round(confirm_avg, 1),
                        "p50": round(_percentile(confirm_latencies, 50), 1) if confirm_latencies else 0.0,
                        "p95": round(_percentile(confirm_latencies, 95), 1) if confirm_latencies else 0.0,
                        "max": round(max(confirm_latencies), 1) if confirm_latencies else 0.0,
                    },
                    "unfinished_request_count": len(unresolved_ids),
                    "unfinished_request_ids": sorted(unresolved_ids)[:25],
                    "slow_request_threshold_ms": slow_query_ms,
                    "slow_requests": slow_entries,
                    "slow_request_trace_coverage": {
                        "question_count": traced_question_count,
                        "sql_count": traced_sql_count,
                        "slow_request_count": len(slow_entries),
                    },
                    "stage_latency_summary": stage_summary,
                    "status": status,
                    "status_reasons": reasons,
                    "thresholds": {
                        "warn_query_p95_ms": warn_query_p95_ms,
                        "critical_query_p95_ms": critical_query_p95_ms,
                        "warn_unresolved_count": warn_unresolved_count,
                        "critical_unresolved_count": critical_unresolved_count,
                        "slow_query_ms": slow_query_ms,
                    },
                }
        except Exception as e:
            logging.warning("_summarize_chatbot_performance: DB query path unavailable, falling back to logs: %s", e)

        log_paths = ["logs/main_log.txt", "logs/app_log.txt"]
        existing_paths = [p for p in log_paths if os.path.exists(p)]
        if not existing_paths:
            return {
                "error": "chatbot performance metrics unavailable in DB and logs",
                "paths": log_paths,
            }

        request_total_starts: set[str] = set()
        request_total_ends: set[str] = set()
        stage_durations: dict[str, list[float]] = {}
        query_entries: list[dict] = []
        confirm_entries: list[dict] = []
        trace_by_request: dict[str, dict[str, str]] = {}
        scanned_lines = 0

        for path in existing_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        ts = self._parse_log_timestamp(line)
                        if ts is None or ts < start_ts or ts > end_ts:
                            continue
                        scanned_lines += 1

                        m_start = start_re.search(line)
                        if m_start:
                            req_id, endpoint, stage, extra = (
                                m_start.group(1).strip(),
                                m_start.group(2).strip(),
                                m_start.group(3).strip(),
                                (m_start.group(4) or "").strip(),
                            )
                            if stage == "request_total":
                                request_total_starts.add(req_id)
                            if extra:
                                trace_by_request.setdefault(req_id, {})
                            continue

                        m_end = end_re.search(line)
                        if m_end:
                            req_id = m_end.group(1).strip()
                            endpoint = m_end.group(2).strip()
                            stage = m_end.group(3).strip()
                            duration_ms = float(m_end.group(4) or 0.0)
                            extra = (m_end.group(5) or "").strip()
                            extras = _parse_kv(extra)
                            stage_durations.setdefault(stage, []).append(duration_ms)

                            if stage == "request_total":
                                request_total_ends.add(req_id)
                                entry = {
                                    "request_id": req_id,
                                    "endpoint": endpoint,
                                    "duration_ms": round(duration_ms, 1),
                                    "result_type": extras.get("result_type", ""),
                                }
                                if endpoint == "/query":
                                    query_entries.append(entry)
                                elif endpoint == "/confirm":
                                    confirm_entries.append(entry)
                            continue

                        m_req = trace_req_re.search(line)
                        if m_req:
                            req_id = m_req.group(1).strip()
                            user_input = (m_req.group(3) or "").strip()
                            trace_by_request.setdefault(req_id, {})["question"] = user_input
                            continue

                        m_sql = trace_sql_re.search(line)
                        if m_sql:
                            req_id = m_sql.group(1).strip()
                            sql_text = (m_sql.group(4) or "").strip()
                            trace_by_request.setdefault(req_id, {})["sql"] = sql_text
                            continue
            except Exception as e:
                logging.warning("_summarize_chatbot_performance: Failed to parse %s: %s", path, e)

        unresolved_request_ids = sorted(list(request_total_starts - request_total_ends))[:25]
        unresolved_count = len(request_total_starts - request_total_ends)

        query_latencies = [float(e.get("duration_ms", 0.0) or 0.0) for e in query_entries]
        confirm_latencies = [float(e.get("duration_ms", 0.0) or 0.0) for e in confirm_entries]
        query_avg = (sum(query_latencies) / len(query_latencies)) if query_latencies else 0.0
        confirm_avg = (sum(confirm_latencies) / len(confirm_latencies)) if confirm_latencies else 0.0

        slow_query_ms = float(perf_cfg.get("slow_query_ms", 15000) or 15000)
        warn_query_p95_ms = float(perf_cfg.get("warn_query_p95_ms", 10000) or 10000)
        critical_query_p95_ms = float(perf_cfg.get("critical_query_p95_ms", 20000) or 20000)
        warn_unresolved_count = int(perf_cfg.get("warn_unresolved_count", 1) or 1)
        critical_unresolved_count = int(perf_cfg.get("critical_unresolved_count", 3) or 3)

        slow_entries: list[dict] = []
        for entry in query_entries + confirm_entries:
            if float(entry.get("duration_ms", 0.0) or 0.0) < slow_query_ms:
                continue
            req_id = str(entry.get("request_id", "") or "")
            trace = trace_by_request.get(req_id, {})
            slow_entries.append({
                **entry,
                "question": str(trace.get("question", "")),
                "sql": str(trace.get("sql", "")),
            })
        slow_entries = sorted(slow_entries, key=lambda x: float(x.get("duration_ms", 0.0) or 0.0), reverse=True)[:20]

        stage_summary: list[dict] = []
        for stage, values in stage_durations.items():
            if not values:
                continue
            avg_val = sum(values) / len(values)
            stage_summary.append({
                "stage": stage,
                "count": len(values),
                "avg_ms": round(avg_val, 1),
                "p95_ms": round(_percentile(values, 95), 1),
                "max_ms": round(max(values), 1),
            })
        stage_summary = sorted(stage_summary, key=lambda x: float(x.get("p95_ms", 0.0) or 0.0), reverse=True)[:15]

        query_p95 = _percentile(query_latencies, 95) if query_latencies else 0.0
        status = "HEALTHY"
        reasons: list[str] = []
        if query_p95 >= critical_query_p95_ms or unresolved_count >= critical_unresolved_count:
            status = "DEGRADED"
        elif query_p95 >= warn_query_p95_ms or unresolved_count >= warn_unresolved_count:
            status = "WATCH"
        if query_p95 >= warn_query_p95_ms:
            reasons.append(f"query p95 latency high ({query_p95:.1f}ms)")
        if unresolved_count > 0:
            reasons.append(f"unfinished requests ({unresolved_count})")

        traced_question_count = sum(1 for e in slow_entries if e.get("question"))
        traced_sql_count = sum(1 for e in slow_entries if e.get("sql"))

        return {
            "source": "logs",
            "paths": existing_paths,
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "scanned_lines": scanned_lines,
            "query_request_count": len(query_entries),
            "confirm_request_count": len(confirm_entries),
            "query_latency_ms": {
                "avg": round(query_avg, 1),
                "p50": round(_percentile(query_latencies, 50), 1) if query_latencies else 0.0,
                "p95": round(query_p95, 1),
                "max": round(max(query_latencies), 1) if query_latencies else 0.0,
            },
            "confirm_latency_ms": {
                "avg": round(confirm_avg, 1),
                "p50": round(_percentile(confirm_latencies, 50), 1) if confirm_latencies else 0.0,
                "p95": round(_percentile(confirm_latencies, 95), 1) if confirm_latencies else 0.0,
                "max": round(max(confirm_latencies), 1) if confirm_latencies else 0.0,
            },
            "unfinished_request_count": unresolved_count,
            "unfinished_request_ids": unresolved_request_ids,
            "slow_request_threshold_ms": slow_query_ms,
            "slow_requests": slow_entries,
            "slow_request_trace_coverage": {
                "question_count": traced_question_count,
                "sql_count": traced_sql_count,
                "slow_request_count": len(slow_entries),
            },
            "stage_latency_summary": stage_summary,
            "status": status,
            "status_reasons": reasons,
            "thresholds": {
                "warn_query_p95_ms": warn_query_p95_ms,
                "critical_query_p95_ms": critical_query_p95_ms,
                "warn_unresolved_count": warn_unresolved_count,
                "critical_unresolved_count": critical_unresolved_count,
                "slow_query_ms": slow_query_ms,
            },
        }

    def _summarize_chatbot_metrics_sync(self, report_timestamp: str | None) -> dict:
        """Summarize local chatbot metrics-table population and freshness (for sync visibility)."""
        window_days = int(
            self.validation_config.get("reporting", {}).get("chatbot_metrics_sync_days", 90) or 90
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(days=window_days)

        summary = {
            "available": False,
            "window_days": window_days,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "request_count_window": 0,
            "stage_count_window": 0,
            "request_count_total": 0,
            "stage_count_total": 0,
            "latest_request_started_at": "",
            "latest_stage_started_at": "",
            "status": "MISSING",
            "status_reasons": [],
            "error": "",
        }

        try:
            table_exists_rows = self.db_handler.execute_query(
                """
                SELECT
                    to_regclass('public.chatbot_request_metrics') IS NOT NULL AS request_exists,
                    to_regclass('public.chatbot_stage_metrics') IS NOT NULL AS stage_exists
                """
            ) or []
            request_exists = bool(table_exists_rows and table_exists_rows[0][0])
            stage_exists = bool(table_exists_rows and table_exists_rows[0][1])
            if not request_exists or not stage_exists:
                missing = []
                if not request_exists:
                    missing.append("chatbot_request_metrics")
                if not stage_exists:
                    missing.append("chatbot_stage_metrics")
                summary["status"] = "MISSING"
                summary["status_reasons"] = [f"chatbot metrics tables unavailable in local DB: {', '.join(missing)}."]
                summary["error"] = f"missing_tables={','.join(missing)}"
                return summary

            rows = self.db_handler.execute_query(
                """
                SELECT COUNT(*) FROM chatbot_request_metrics WHERE started_at >= :start_ts
                """,
                {"start_ts": start_ts},
            ) or []
            summary["request_count_window"] = int(rows[0][0]) if rows else 0

            rows = self.db_handler.execute_query(
                """
                SELECT COUNT(*) FROM chatbot_stage_metrics WHERE started_at >= :start_ts
                """,
                {"start_ts": start_ts},
            ) or []
            summary["stage_count_window"] = int(rows[0][0]) if rows else 0

            rows = self.db_handler.execute_query("SELECT COUNT(*) FROM chatbot_request_metrics") or []
            summary["request_count_total"] = int(rows[0][0]) if rows else 0

            rows = self.db_handler.execute_query("SELECT COUNT(*) FROM chatbot_stage_metrics") or []
            summary["stage_count_total"] = int(rows[0][0]) if rows else 0

            rows = self.db_handler.execute_query(
                "SELECT MAX(started_at) FROM chatbot_request_metrics"
            ) or []
            summary["latest_request_started_at"] = str(rows[0][0]) if rows and rows[0][0] else ""

            rows = self.db_handler.execute_query(
                "SELECT MAX(started_at) FROM chatbot_stage_metrics"
            ) or []
            summary["latest_stage_started_at"] = str(rows[0][0]) if rows and rows[0][0] else ""

            summary["available"] = True
            reasons: list[str] = []
            if summary["request_count_window"] <= 0:
                reasons.append("No chatbot_request_metrics rows in 90-day window.")
            if summary["stage_count_window"] <= 0:
                reasons.append("No chatbot_stage_metrics rows in 90-day window.")
            if reasons:
                summary["status"] = "STALE"
                summary["status_reasons"] = reasons
            else:
                summary["status"] = "SYNCED"
                summary["status_reasons"] = ["Local chatbot metrics tables have recent window coverage."]
            return summary
        except Exception as e:
            summary["error"] = str(e)
            summary["status"] = "MISSING"
            summary["status_reasons"] = ["chatbot metrics tables unavailable in local DB."]
            return summary

    def _summarize_scraper_network_health(self, report_timestamp: str | None) -> dict:
        """Summarize recent scraper network/transient failure reliability from scraper logs."""
        path = "logs/scraper_log.txt"
        if not os.path.exists(path):
            return {"error": "scraper log not found", "path": path}

        hours_window = int(
            self.validation_config.get('reporting', {}).get('scraper_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()

        window_lines: list[str] = []
        for line in lines:
            ts = self._parse_log_timestamp(line)
            if ts is None:
                continue
            if start_ts <= ts <= end_ts:
                window_lines.append(line)
        requested_start_ts = start_ts
        requested_end_ts = end_ts
        used_fallback_window = False
        if not window_lines:
            window_lines = lines[-4000:]
            used_fallback_window = True

        actual_timestamps: list[datetime] = []
        for line in window_lines:
            ts = self._parse_log_timestamp(line)
            if ts is not None:
                actual_timestamps.append(ts)
        if actual_timestamps:
            start_ts = min(actual_timestamps)
            end_ts = max(actual_timestamps)

        latest_values: dict[str, int] = {}
        stat_patterns = {
            "request_count": re.compile(r"'downloader/request_count':\s*(\d+)"),
            "response_count": re.compile(r"'downloader/response_count':\s*(\d+)"),
            "exception_count": re.compile(r"'downloader/exception_count':\s*(\d+)"),
            "retry_count": re.compile(r"'retry/count':\s*(\d+)"),
            "retry_max_reached": re.compile(r"'retry/max_reached':\s*(\d+)"),
            "timeout_count": re.compile(r"'retry/reason_count/twisted\.internet\.error\.TimeoutError':\s*(\d+)"),
            "connection_lost_count": re.compile(r"'retry/reason_count/twisted\.web\._newclient\.ResponseNeverReceived':\s*(\d+)"),
        }
        for line in window_lines:
            for key, pattern in stat_patterns.items():
                match = pattern.search(line)
                if match:
                    latest_values[key] = int(match.group(1))

        finish_reason = ""
        finish_re = re.compile(r"'finish_reason':\s*'([^']+)'")
        for line in window_lines:
            match = finish_re.search(line)
            if match:
                finish_reason = match.group(1)

        top_fail_domains: Counter = Counter()
        domain_timeout_top_from_summary: list[tuple[str, int]] = []
        domain_exception_top_from_summary: list[tuple[str, int]] = []
        for line in window_lines:
            if "Error downloading <GET " not in line:
                # Parse optional per-domain aggregate summaries emitted by scraper.py
                if "domain_timeout_failures_top:" in line:
                    m = re.search(r"domain_timeout_failures_top:\s*(\[.*\])", line)
                    if m:
                        try:
                            parsed = ast.literal_eval(m.group(1))
                            if isinstance(parsed, list):
                                domain_timeout_top_from_summary = [
                                    (str(item[0]), int(item[1])) for item in parsed[:10]
                                    if isinstance(item, (tuple, list)) and len(item) >= 2
                                ]
                        except Exception:
                            pass
                if "domain_exception_failures_top:" in line:
                    m = re.search(r"domain_exception_failures_top:\s*(\[.*\])", line)
                    if m:
                        try:
                            parsed = ast.literal_eval(m.group(1))
                            if isinstance(parsed, list):
                                domain_exception_top_from_summary = [
                                    (str(item[0]), int(item[1])) for item in parsed[:10]
                                    if isinstance(item, (tuple, list)) and len(item) >= 2
                                ]
                        except Exception:
                            pass
                continue
            start = line.find("<GET ")
            end = line.find(">", start)
            if start < 0 or end < 0:
                continue
            raw_url = line[start + 5:end].strip()
            try:
                domain = re.sub(r":\d+$", "", raw_url.split("/")[2].lower())
            except Exception:
                domain = ""
            if domain:
                top_fail_domains[domain] += 1

        request_count = int(latest_values.get("request_count", 0))
        exception_count = int(latest_values.get("exception_count", 0))
        retry_max_reached = int(latest_values.get("retry_max_reached", 0))
        timeout_count = int(latest_values.get("timeout_count", 0))
        connection_lost_count = int(latest_values.get("connection_lost_count", 0))
        exception_rate = (exception_count / request_count) if request_count else 0.0

        thresholds_cfg = (
            self.validation_config.get("reporting", {}).get("scraper_network_thresholds", {})
        )
        max_exception_rate = float(thresholds_cfg.get("max_exception_rate", 0.20) or 0.20)
        max_retry_max_reached = int(thresholds_cfg.get("max_retry_max_reached", 30) or 30)
        max_timeout_count = int(thresholds_cfg.get("max_timeout_count", 50) or 50)
        degraded = (
            exception_rate > max_exception_rate
            or retry_max_reached > max_retry_max_reached
            or timeout_count > max_timeout_count
        )

        return {
            "path": path,
            "informational_for_grade": True,
            "grading_note": "Request-level network telemetry is informational and excluded from URL-level grading gates.",
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "requested_start_ts": requested_start_ts.isoformat(sep=" "),
            "requested_end_ts": requested_end_ts.isoformat(sep=" "),
            "used_fallback_window": used_fallback_window,
            "finish_reason": finish_reason,
            "request_count": request_count,
            "response_count": int(latest_values.get("response_count", 0)),
            "exception_count": exception_count,
            "exception_rate": round(exception_rate, 4),
            "retry_count": int(latest_values.get("retry_count", 0)),
            "retry_max_reached": retry_max_reached,
            "timeout_count": timeout_count,
            "connection_lost_count": connection_lost_count,
            "top_failure_domains": top_fail_domains.most_common(10),
            "top_timeout_domains": domain_timeout_top_from_summary,
            "top_exception_domains": domain_exception_top_from_summary,
            "degraded": degraded,
            "thresholds": {
                "max_exception_rate": max_exception_rate,
                "max_retry_max_reached": max_retry_max_reached,
                "max_timeout_count": max_timeout_count,
            },
        }

    def _summarize_fb_block_health(self, report_timestamp: str | None) -> dict:
        """Summarize Facebook temporary-block behavior from fb logs."""
        path = "logs/fb_log.txt"
        if not os.path.exists(path):
            return {"error": "fb log not found", "path": path}

        hours_window = int(
            self.validation_config.get('reporting', {}).get('fb_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()

        window_lines: list[str] = []
        for line in lines:
            ts = self._parse_log_timestamp(line)
            if ts is None:
                continue
            if start_ts <= ts <= end_ts:
                window_lines.append(line)
        if not window_lines:
            window_lines = lines[-4000:]

        run_id_re = re.compile(r"\[run_id=([^\]]+)\]")
        retrieved_re = re.compile(r"Retrieved\s+(\d+)\s+Facebook URLs from the database\.", re.IGNORECASE)
        processing_base_re = re.compile(r"Processing base URL:\s*(https?://\S+)", re.IGNORECASE)
        processing_fb_url_re = re.compile(r"Processing Facebook URL:\s*(https?://\S+)", re.IGNORECASE)
        access_ok_re = re.compile(r"fb access check:.*requested_url=(https?://\S+).*state=ok\b", re.IGNORECASE)
        access_any_re = re.compile(r"fb access check:.*requested_url=(https?://\S+)", re.IGNORECASE)
        event_marked_processed_re = re.compile(r"Event URL marked processed:\s*(https?://\S+)", re.IGNORECASE)
        base_marked_processed_re = re.compile(r"Base URL marked processed:\s*(https?://\S+)", re.IGNORECASE)
        base_events_scraped_re = re.compile(r"Events_scraped flag set for base URL:\s*(https?://\S+)", re.IGNORECASE)

        latest_run_id = ""
        for line in window_lines:
            m = run_id_re.search(line)
            if m:
                latest_run_id = m.group(1).strip()

        scoped_fb_lines = window_lines
        if latest_run_id:
            scoped_fb_lines = [ln for ln in window_lines if f"[run_id={latest_run_id}]" in ln]
            if not scoped_fb_lines:
                scoped_fb_lines = window_lines

        strike1_count = 0
        abort_count = 0
        explicit_block_count = 0
        wait_seconds_total = 0
        wait_seconds_samples: list[int] = []
        fb_target_urls = 0
        fb_successes_before_throttle = 0
        first_throttle_seen = False
        throttle_detected = False
        unique_ok_urls: set[str] = set()
        unique_ok_urls_all: set[str] = set()
        unique_processed_urls_before_throttle: set[str] = set()
        unique_attempted_fb_urls: set[str] = set()
        unique_success_fb_urls: set[str] = set()

        wait_re = re.compile(r"fb temp block policy: strike=1 .* wait_seconds=(\d+)")
        for line in scoped_fb_lines:
            low = line.lower()
            if "reason=explicit_temp_block_regex" in low:
                explicit_block_count += 1
            if "fb temp block policy: strike=1 " in low:
                strike1_count += 1
                m = wait_re.search(line)
                if m:
                    wait_val = int(m.group(1))
                    wait_seconds_total += wait_val
                    wait_seconds_samples.append(wait_val)
            if "action=abort_fb_run" in low:
                abort_count += 1
            m_retrieved = retrieved_re.search(line)
            if m_retrieved:
                fb_target_urls = max(fb_target_urls, int(m_retrieved.group(1)))

            m_access_any = access_any_re.search(line)
            if m_access_any:
                unique_attempted_fb_urls.add(m_access_any.group(1).rstrip(".,"))
            m_ok_any = access_ok_re.search(line)
            if m_ok_any:
                unique_ok_urls_all.add(m_ok_any.group(1).rstrip(".,"))
            m_base_proc = processing_base_re.search(line)
            if m_base_proc:
                unique_attempted_fb_urls.add(m_base_proc.group(1).rstrip(".,"))
            m_fb_proc = processing_fb_url_re.search(line)
            if m_fb_proc:
                unique_attempted_fb_urls.add(m_fb_proc.group(1).rstrip(".,"))

            m_event_done = event_marked_processed_re.search(line)
            if m_event_done:
                unique_success_fb_urls.add(m_event_done.group(1).rstrip(".,"))
            m_base_done = base_marked_processed_re.search(line)
            if m_base_done:
                unique_success_fb_urls.add(m_base_done.group(1).rstrip(".,"))
            m_base_scraped = base_events_scraped_re.search(line)
            if m_base_scraped:
                unique_success_fb_urls.add(m_base_scraped.group(1).rstrip(".,"))

            throttle_now = (
                "reason=explicit_temp_block_regex" in low
                or "reason=rate_limit_or_temp_block_text" in low
                or "you are temporarily blocked" in low
            )
            if not first_throttle_seen and throttle_now:
                first_throttle_seen = True
            if throttle_now:
                throttle_detected = True

            if first_throttle_seen:
                continue

            m_ok = access_ok_re.search(line)
            if m_ok:
                unique_ok_urls.add(m_ok.group(1).rstrip(".,"))
            m_base = processing_base_re.search(line)
            if m_base:
                unique_processed_urls_before_throttle.add(m_base.group(1).rstrip(".,"))
            m_fb_url = processing_fb_url_re.search(line)
            if m_fb_url:
                unique_processed_urls_before_throttle.add(m_fb_url.group(1).rstrip(".,"))

        fb_successes_before_throttle = len(unique_ok_urls) or len(unique_processed_urls_before_throttle)

        instagram_urls_seen = 0
        scraper_path = "logs/scraper_log.txt"
        if os.path.exists(scraper_path):
            with open(scraper_path, "r", encoding="utf-8", errors="ignore") as file:
                scraper_lines = file.readlines()

            scoped_scraper_lines: list[str] = []
            for line in scraper_lines:
                ts = self._parse_log_timestamp(line)
                if ts is None:
                    continue
                if start_ts <= ts <= end_ts:
                    scoped_scraper_lines.append(line)
            if not scoped_scraper_lines:
                scoped_scraper_lines = scraper_lines[-6000:]

            if latest_run_id:
                run_scoped = [ln for ln in scoped_scraper_lines if f"[run_id={latest_run_id}]" in ln]
                if run_scoped:
                    scoped_scraper_lines = run_scoped

            ig_urls: set[str] = set()
            ig_re = re.compile(r"Skipping social media URL \(fb/ig\):\s*(https?://\S+)", re.IGNORECASE)
            for line in scoped_scraper_lines:
                m_ig = ig_re.search(line)
                if not m_ig:
                    continue
                url = m_ig.group(1).rstrip(".,")
                if "instagram" in url.lower():
                    ig_urls.add(url)
            instagram_urls_seen = len(ig_urls)

        denominator_fb_ig = max(1, fb_target_urls + instagram_urls_seen)
        numerator_fb_ig = fb_successes_before_throttle + instagram_urls_seen
        jail_progress_ratio = (numerator_fb_ig / denominator_fb_ig) if throttle_detected else None
        fb_only_denominator = max(1, fb_target_urls)
        fb_only_ratio = (fb_successes_before_throttle / fb_only_denominator) if throttle_detected else None

        attempted_fb_urls = len(unique_attempted_fb_urls)
        successful_fb_urls = len(unique_success_fb_urls)
        target_total_urls = fb_target_urls + instagram_urls_seen
        attempted_total_urls = attempted_fb_urls
        successful_total_urls = successful_fb_urls
        success_rate = (successful_total_urls / attempted_total_urls) if attempted_total_urls else 0.0
        target_coverage_rate = (successful_total_urls / max(1, target_total_urls))

        avg_wait_seconds = (wait_seconds_total / len(wait_seconds_samples)) if wait_seconds_samples else 0.0

        return {
            "path": path,
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "explicit_block_count": explicit_block_count,
            "strike1_count": strike1_count,
            "abort_count": abort_count,
            "avg_wait_seconds": round(avg_wait_seconds, 1),
            "max_wait_seconds": max(wait_seconds_samples) if wait_seconds_samples else 0,
            "min_wait_seconds": min(wait_seconds_samples) if wait_seconds_samples else 0,
            "policy_triggered": strike1_count > 0 or abort_count > 0,
            "run_id": latest_run_id,
            "fb_target_urls": fb_target_urls,
            "instagram_target_urls": instagram_urls_seen,
            "target_total_urls": target_total_urls,
            "attempted_fb_urls": attempted_fb_urls,
            "attempted_total_urls": attempted_total_urls,
            "access_ok_urls": len(unique_ok_urls_all),
            "successful_fb_urls": successful_fb_urls,
            "successful_total_urls": successful_total_urls,
            "success_rate": round(success_rate, 4),
            "target_coverage_rate": round(target_coverage_rate, 4),
            "throttle_detected": throttle_detected,
            "fb_successes_before_throttle": fb_successes_before_throttle,
            "numerator_fb_ig": numerator_fb_ig,
            "denominator_fb_ig": denominator_fb_ig,
            "jail_progress_ratio": round(jail_progress_ratio, 4) if jail_progress_ratio is not None else None,
            "fb_only_ratio": round(fb_only_ratio, 4) if fb_only_ratio is not None else None,
        }

    def _build_fb_block_health_html(self, fb_data: dict) -> str:
        """Render HTML for Facebook temporary-block policy health."""
        if not fb_data:
            return "<p class='error-box'>❌ Facebook block health summary unavailable</p>"
        if 'error' in fb_data:
            return f"<p class='error-box'>❌ Facebook block health failed: {self._escape_html(fb_data['error'])}</p>"

        explicit_blocks = int(fb_data.get("explicit_block_count", 0) or 0)
        strike1 = int(fb_data.get("strike1_count", 0) or 0)
        aborts = int(fb_data.get("abort_count", 0) or 0)
        avg_wait = float(fb_data.get("avg_wait_seconds", 0) or 0)
        fb_target_urls = int(fb_data.get("fb_target_urls", 0) or 0)
        instagram_target_urls = int(fb_data.get("instagram_target_urls", 0) or 0)
        target_total_urls = int(fb_data.get("target_total_urls", 0) or 0)
        attempted_total_urls = int(fb_data.get("attempted_total_urls", 0) or 0)
        successful_total_urls = int(fb_data.get("successful_total_urls", 0) or 0)
        success_rate = float(fb_data.get("success_rate", 0) or 0)
        target_coverage_rate = float(fb_data.get("target_coverage_rate", 0) or 0)
        throttle_detected = bool(fb_data.get("throttle_detected", False))
        fb_successes_before_throttle = int(fb_data.get("fb_successes_before_throttle", 0) or 0)
        numerator_fb_ig = int(fb_data.get("numerator_fb_ig", 0) or 0)
        denominator_fb_ig = int(fb_data.get("denominator_fb_ig", 0) or 0)
        jail_progress_ratio = fb_data.get("jail_progress_ratio")
        fb_only_ratio = fb_data.get("fb_only_ratio")
        status_class = "status-pass"
        status_text = "Healthy"
        if aborts > 0:
            status_class = "status-fail"
            status_text = "Aborted"
        elif strike1 > 0 or explicit_blocks > 0:
            status_class = "status-warning"
            status_text = "Cooling"

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{explicit_blocks}</div>
                <div class="metric-label">Explicit Temp Blocks</div>
            </div>
            <div class="metric">
                <div class="metric-value">{strike1}</div>
                <div class="metric-label">Strike-1 Cooldowns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{aborts}</div>
                <div class="metric-label">fb.py Early Aborts</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_wait:.1f}s</div>
                <div class="metric-label">Average Strike-1 Wait</div>
            </div>
            <div class="metric">
                <div class="metric-value">{target_total_urls}</div>
                <div class="metric-label">FB+IG Target URLs</div>
            </div>
            <div class="metric">
                <div class="metric-value">{attempted_total_urls}</div>
                <div class="metric-label">Attempted URLs</div>
            </div>
            <div class="metric">
                <div class="metric-value">{successful_total_urls}</div>
                <div class="metric-label">Successfully Scraped URLs</div>
            </div>
            <div class="metric">
                <div class="metric-value">{success_rate:.1%}</div>
                <div class="metric-label">Attempted→Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{target_coverage_rate:.1%}</div>
                <div class="metric-label">Target Coverage Rate</div>
            </div>
        </div>
        <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        """
        html += (
            f"<p><strong>Success Formula:</strong> "
            f"{successful_total_urls} successful / {attempted_total_urls} attempted</p>"
        )
        if throttle_detected and jail_progress_ratio is not None and fb_only_ratio is not None:
            html += (
                f"<p><strong>Jail Progress Formula (only when throttle detected):</strong> "
                f"({fb_successes_before_throttle} FB successes before first throttle + {instagram_target_urls} IG URLs seen) "
                f"/ ({fb_target_urls} FB target URLs + {instagram_target_urls} IG URLs seen) "
                f"= {float(jail_progress_ratio):.1%} (FB-only: {float(fb_only_ratio):.1%})</p>"
            )
        else:
            html += "<p><strong>Jail Progress Ratio:</strong> N/A (no throttle detected in this run/window)</p>"
        if fb_data.get("run_id"):
            html += f"<p><strong>Run ID:</strong> {self._escape_html(str(fb_data.get('run_id')))}</p>"
        html += (
            f"<p><strong>Window:</strong> {self._escape_html(str(fb_data.get('start_ts', '')))}"
            f" to {self._escape_html(str(fb_data.get('end_ts', '')))} "
            f"(last {int(fb_data.get('window_hours', 24) or 24)} hour(s))</p>"
        )
        return html

    def _summarize_fb_ig_url_funnel(self, report_timestamp: str | None, fb_data: dict | None = None) -> dict:
        """Summarize FB/IG URL funnel counts for end-of-report visibility."""
        hours_window = int(
            self.validation_config.get('reporting', {}).get('fb_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        summary = {
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "fb_ig_urls_in_db": 0,
            "urls_passed_for_scraping": int((fb_data or {}).get("access_ok_urls", 0) or 0),
            "urls_attempted_scraping": int((fb_data or {}).get("attempted_total_urls", 0) or 0),
            "urls_with_keywords": 0,
            "urls_with_events": 0,
            "events_over_passed_rate": None,
        }

        try:
            total_rows = self.db_handler.execute_query(
                """
                SELECT COUNT(DISTINCT link)
                FROM urls
                WHERE link ILIKE '%facebook.com%' OR link ILIKE '%instagram.com%'
                """
            ) or []
            if total_rows:
                summary["fb_ig_urls_in_db"] = int(total_rows[0][0] or 0)

            keywords_rows = self.db_handler.execute_query(
                """
                SELECT COUNT(DISTINCT link)
                FROM urls
                WHERE (link ILIKE '%facebook.com%' OR link ILIKE '%instagram.com%')
                  AND time_stamp >= :start_ts
                  AND time_stamp <= :end_ts
                  AND COALESCE(TRIM(keywords), '') <> ''
                """,
                {"start_ts": start_ts, "end_ts": end_ts},
            ) or []
            if keywords_rows:
                summary["urls_with_keywords"] = int(keywords_rows[0][0] or 0)

            events_rows = self.db_handler.execute_query(
                """
                SELECT COUNT(DISTINCT url)
                FROM events
                WHERE (url ILIKE '%facebook.com%' OR url ILIKE '%instagram.com%')
                  AND time_stamp >= :start_ts
                  AND time_stamp <= :end_ts
                """,
                {"start_ts": start_ts, "end_ts": end_ts},
            ) or []
            if events_rows:
                summary["urls_with_events"] = int(events_rows[0][0] or 0)

            passed = int(summary.get("urls_passed_for_scraping", 0) or 0)
            if passed > 0:
                summary["events_over_passed_rate"] = round(
                    float(summary.get("urls_with_events", 0) or 0) / float(passed),
                    4,
                )
        except Exception as e:
            summary["error"] = f"Failed to summarize FB/IG URL funnel: {e}"

        return summary

    def _build_phase1_runtime_summary(self, pipeline_runtime: dict | None) -> dict:
        """Normalize pipeline runtime into the Phase 1 scorecard shape."""
        runtime = pipeline_runtime if isinstance(pipeline_runtime, dict) else {}
        available = bool(runtime.get("available")) and not runtime.get("error")
        runtime_seconds = float(runtime.get("runtime_seconds", 0.0) or 0.0)
        step_spans = runtime.get("step_log_spans", []) if isinstance(runtime.get("step_log_spans"), list) else []
        critical_path_minutes = 0.0
        if step_spans:
            critical_path_minutes = max(
                float(span.get("duration_minutes", 0.0) or 0.0)
                for span in step_spans
                if isinstance(span, dict)
            )
        return {
            "available": available,
            "run_id": str(runtime.get("run_id", "") or ""),
            "start_ts": str(runtime.get("start_ts", "") or ""),
            "end_ts": str(runtime.get("end_ts", "") or ""),
            "pipeline_duration_seconds": round(runtime_seconds, 1),
            "pipeline_duration_minutes": round(runtime_seconds / 60.0, 2),
            "pipeline_duration_hours": round(runtime_seconds / 3600.0, 3),
            "critical_path_minutes": round(critical_path_minutes, 2),
            "parallel_crawlers_enabled": None,
            "step_spans": step_spans,
            "error": str(runtime.get("error", "") or ""),
        }

    def _build_phase1_llm_cost_summary(
        self,
        openrouter_cost: dict | None,
        openai_cost: dict | None,
        accuracy_replay: dict | None,
        llm_activity_summary: dict | None = None,
    ) -> dict:
        """Normalize provider run-cost summaries into one Phase 1 cost artifact."""
        openrouter = openrouter_cost if isinstance(openrouter_cost, dict) else {}
        openai = openai_cost if isinstance(openai_cost, dict) else {}
        replay = accuracy_replay if isinstance(accuracy_replay, dict) else {}
        llm_activity = llm_activity_summary if isinstance(llm_activity_summary, dict) else {}

        provider_costs = {
            "openrouter_usd": float(openrouter.get("cost_usd", 0.0) or 0.0),
            "openai_usd": float(openai.get("cost_usd", 0.0) or 0.0),
        }
        provider_requests = {
            "openrouter_requests": int(openrouter.get("requests", 0) or 0),
            "openai_requests": int(openai.get("requests", 0) or 0),
        }
        provider_tokens = {
            "openrouter_tokens": int(openrouter.get("tokens", 0) or 0),
            "openai_tokens": int(openai.get("tokens", 0) or 0),
        }
        total_cost_usd = round(sum(provider_costs.values()), 6)
        total_requests = sum(provider_requests.values())
        total_tokens = sum(provider_tokens.values())
        successful_replay_urls = int(replay.get("true_count", 0) or 0)
        cost_per_successful_replay_url = (
            round(total_cost_usd / successful_replay_urls, 6)
            if successful_replay_urls > 0
            else None
        )

        available = bool(openrouter.get("available")) or bool(openai.get("available"))
        top_models = llm_activity.get("top_models", []) if isinstance(llm_activity.get("top_models"), list) else []
        top_files = llm_activity.get("top_files", []) if isinstance(llm_activity.get("top_files"), list) else []
        total_model_attempts = sum(int(item[1]) for item in top_models if isinstance(item, (tuple, list)) and len(item) >= 2)
        total_file_attempts = sum(int(item[1]) for item in top_files if isinstance(item, (tuple, list)) and len(item) >= 2)
        by_model: dict[str, float] = {}
        by_step: dict[str, float] = {}
        if total_cost_usd > 0 and total_model_attempts > 0:
            for item in top_models:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                model_key = str(item[0] or "")
                attempts = int(item[1] or 0)
                if not model_key or attempts <= 0:
                    continue
                by_model[model_key] = round(total_cost_usd * (attempts / total_model_attempts), 6)
        if total_cost_usd > 0 and total_file_attempts > 0:
            for item in top_files:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                log_file = str(item[0] or "")
                attempts = int(item[1] or 0)
                if not log_file or attempts <= 0:
                    continue
                step_name = os.path.basename(log_file).replace("_log.txt", "").replace(".txt", "")
                by_step[step_name] = round(total_cost_usd * (attempts / total_file_attempts), 6)
        return {
            "available": available,
            "window": {
                "start_ts": str(openrouter.get("start_ts") or openai.get("start_ts") or ""),
                "end_ts": str(openrouter.get("end_ts") or openai.get("end_ts") or ""),
            },
            "summary": {
                "total_usd": total_cost_usd,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "cost_per_successful_replay_url_usd": cost_per_successful_replay_url,
            },
            "providers": {
                "openrouter": openrouter,
                "openai": openai,
            },
            "by_provider": provider_costs,
            "by_model": by_model,
            "by_step": by_step,
            "allocation_method": "estimated_from_llm_attempt_share",
            "request_counts": provider_requests,
            "token_counts": provider_tokens,
        }

    def _build_phase1_chatbot_quality_summary(
        self,
        chatbot_performance: dict | None,
        chatbot_testing: dict | None,
    ) -> dict:
        """Normalize chatbot latency/correctness into the Phase 1 scorecard shape."""
        performance = chatbot_performance if isinstance(chatbot_performance, dict) else {}
        testing = chatbot_testing if isinstance(chatbot_testing, dict) else {}
        testing_summary = testing.get("summary", {}) if isinstance(testing.get("summary"), dict) else {}
        query_count = int(performance.get("query_request_count", 0) or 0)
        confirm_count = int(performance.get("confirm_request_count", 0) or 0)
        total_requests = query_count + confirm_count
        slow_entries = performance.get("slow_requests", []) if isinstance(performance.get("slow_requests"), list) else []
        under_15s_count = 0
        if total_requests > 0:
            over_15s_count = 0
            for entry in slow_entries:
                if not isinstance(entry, dict):
                    continue
                duration_ms = float(entry.get("duration_ms", 0.0) or 0.0)
                if duration_ms > 15000.0:
                    over_15s_count += 1
            over_15s_count = min(over_15s_count, total_requests)
            under_15s_count = max(total_requests - over_15s_count, 0)
        within_15s_pct = round((under_15s_count / total_requests) * 100.0, 2) if total_requests > 0 else None

        correctness_pct = (
            round(float(testing_summary.get("average_score", 0.0) or 0.0), 2)
            if testing_summary
            else None
        )
        sql_validity_pct = (
            round(float(testing_summary.get("execution_success_rate", 0.0) or 0.0) * 100.0, 2)
            if testing_summary
            else None
        )
        problem_categories = testing.get("problem_categories", []) if isinstance(testing.get("problem_categories"), list) else []
        total_problem_count = sum(
            int(category.get("count", 0) or 0)
            for category in problem_categories
            if isinstance(category, dict)
        )
        hallucination_problem_count = sum(
            int(category.get("count", 0) or 0)
            for category in problem_categories
            if isinstance(category, dict)
            and any(
                token in str(category.get("name", "")).lower()
                for token in ("halluc", "fabricat", "invent", "made up")
            )
        )
        fallback_rate_pct = (
            round((confirm_count / total_requests) * 100.0, 2)
            if total_requests > 0
            else None
        )
        hallucination_rate_pct = (
            round((hallucination_problem_count / total_problem_count) * 100.0, 2)
            if total_problem_count > 0
            else 0.0
        )
        unfinished_request_count = int(performance.get("unfinished_request_count", 0) or 0)
        user_visible_error_rate_pct = (
            round((unfinished_request_count / total_requests) * 100.0, 2)
            if total_requests > 0
            else None
        )

        return {
            "available": bool(testing_summary) or bool(performance.get("query_request_count")) or bool(performance.get("confirm_request_count")),
            "window": {
                "start_ts": str(performance.get("start_ts", "") or ""),
                "end_ts": str(performance.get("end_ts", "") or ""),
                "source": str(performance.get("source", "") or ""),
            },
            "summary": {
                "chatbot_response_within_15s_pct": within_15s_pct,
                "chatbot_p50_latency_seconds": round(float(((performance.get("query_latency_ms") or {}).get("p50", 0.0) or 0.0) / 1000.0), 3),
                "chatbot_p95_latency_seconds": round(float(((performance.get("query_latency_ms") or {}).get("p95", 0.0) or 0.0) / 1000.0), 3),
                "chatbot_answer_correctness_pct": correctness_pct,
                "chatbot_sql_validity_pct": sql_validity_pct,
                "chatbot_hallucination_rate_pct": hallucination_rate_pct,
                "chatbot_fallback_rate_pct": fallback_rate_pct,
                "chatbot_user_visible_error_rate_pct": user_visible_error_rate_pct,
            },
            "sample_sizes": {
                "query_requests": query_count,
                "confirm_requests": confirm_count,
                "graded_responses": int(testing_summary.get("total_tests", 0) or 0),
            },
            "status": str(performance.get("status", "") or ""),
            "status_reasons": performance.get("status_reasons", []) if isinstance(performance.get("status_reasons"), list) else [],
            "problem_categories": testing.get("problem_categories", []) if isinstance(testing.get("problem_categories"), list) else [],
        }

    def _persist_phase1_scorecard_metrics(
        self,
        run_id: str,
        runtime_summary: dict,
        llm_cost_summary: dict,
        chatbot_quality_summary: dict,
        duplicate_summary: dict | None = None,
        coverage_summary: dict | None = None,
        dev_summary: dict | None = None,
        holdout_summary: dict | None = None,
        domain_capped_summary: dict | None = None,
        event_data_quality_summary: dict | None = None,
        field_accuracy_summary: dict | None = None,
        run_delta_summary: dict | None = None,
    ) -> None:
        """Persist key Phase 1 metrics for trend reporting."""
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "record_metric_observation"):
            return
        window_start = self._parse_iso_datetime(str(runtime_summary.get("start_ts", "") or ""))
        window_end = self._parse_iso_datetime(str(runtime_summary.get("end_ts", "") or "")) or datetime.now()
        metric_specs = [
            (
                "phase1_pipeline_duration_minutes",
                runtime_summary.get("pipeline_duration_minutes"),
                "minutes",
                "Phase 1 scorecard: end-to-end pipeline runtime in minutes",
                False,
            ),
            (
                "phase1_total_llm_cost_usd",
                ((llm_cost_summary.get("summary") or {}).get("total_usd") if isinstance(llm_cost_summary.get("summary"), dict) else None),
                "usd",
                "Phase 1 scorecard: total OpenAI + OpenRouter run cost",
                False,
            ),
            (
                "phase1_chatbot_response_within_15s_pct",
                ((chatbot_quality_summary.get("summary") or {}).get("chatbot_response_within_15s_pct") if isinstance(chatbot_quality_summary.get("summary"), dict) else None),
                "percent",
                "Phase 1 scorecard: chatbot responses completed within 15 seconds",
                True,
            ),
            (
                "phase1_chatbot_answer_correctness_pct",
                ((chatbot_quality_summary.get("summary") or {}).get("chatbot_answer_correctness_pct") if isinstance(chatbot_quality_summary.get("summary"), dict) else None),
                "percent",
                "Phase 1 scorecard: chatbot graded correctness score",
                True,
            ),
            (
                "phase1_chatbot_p95_latency_seconds",
                ((chatbot_quality_summary.get("summary") or {}).get("chatbot_p95_latency_seconds") if isinstance(chatbot_quality_summary.get("summary"), dict) else None),
                "seconds",
                "Phase 1 scorecard: chatbot p95 latency in seconds",
                False,
            ),
        ]
        duplicate_info = duplicate_summary if isinstance(duplicate_summary, dict) else {}
        coverage_info = coverage_summary if isinstance(coverage_summary, dict) else {}
        dev_info = dev_summary if isinstance(dev_summary, dict) else {}
        holdout_info = holdout_summary if isinstance(holdout_summary, dict) else {}
        domain_capped_info = domain_capped_summary if isinstance(domain_capped_summary, dict) else {}
        event_data_quality = event_data_quality_summary if isinstance(event_data_quality_summary, dict) else {}
        field_accuracy = field_accuracy_summary if isinstance(field_accuracy_summary, dict) else {}
        run_delta_info = run_delta_summary if isinstance(run_delta_summary, dict) else {}
        previous_run_info = run_delta_info.get("previous_run", {}) if isinstance(run_delta_info.get("previous_run"), dict) else {}
        holdout_baseline_info = run_delta_info.get("holdout_baseline", {}) if isinstance(run_delta_info.get("holdout_baseline"), dict) else {}
        metric_specs.extend(
            [
                (
                    "duplicate_rate_per_100_events",
                    duplicate_info.get("duplicate_rate_per_100_events"),
                    "count_per_100",
                    "Phase 2 integrity: duplicate excess rows per 100 events",
                    False,
                ),
                (
                    "severe_duplicate_rate_per_100_events",
                    duplicate_info.get("severe_duplicate_rate_per_100_events"),
                    "count_per_100",
                    "Phase 2 integrity: severe duplicate excess rows per 100 events",
                    False,
                ),
                (
                    "coverage_watchlist_source_hit_rate_pct",
                    coverage_info.get("source_hit_rate_pct"),
                    "percent",
                    "Phase 2 integrity: watchlist source hit rate",
                    True,
                ),
                (
                    "coverage_watchlist_event_capture_rate_pct",
                    coverage_info.get("event_capture_rate_pct"),
                    "percent",
                    "Phase 2 integrity: watchlist event capture rate",
                    True,
                ),
                (
                    "missed_event_rate_manual_audit_pct",
                    coverage_info.get("missed_event_rate_manual_audit_pct"),
                    "percent",
                    "Coverage manual audit: expected events missing from the current events table",
                    False,
                ),
                (
                    "dev_replay_url_accuracy_pct",
                    dev_info.get("replay_url_accuracy_pct"),
                    "percent",
                    "Phase 3 honest evaluation: dev replay URL accuracy",
                    True,
                ),
                (
                    "holdout_replay_url_accuracy_pct",
                    holdout_info.get("replay_url_accuracy_pct"),
                    "percent",
                    "Phase 3 honest evaluation: holdout replay URL accuracy",
                    True,
                ),
                (
                    "domain_capped_replay_url_accuracy_pct",
                    domain_capped_info.get("replay_url_accuracy_pct"),
                    "percent",
                    "Phase 3 honest evaluation: replay URL accuracy after per-domain cap",
                    True,
                ),
                (
                    "invalid_event_rate_pct",
                    event_data_quality.get("invalid_event_rate_pct"),
                    "percent",
                    "Event data quality: invalid events in current events table",
                    False,
                ),
                (
                    "stale_event_rate_pct",
                    event_data_quality.get("stale_event_rate_pct"),
                    "percent",
                    "Event data quality: stale events in current events table",
                    False,
                ),
                (
                    "field_accuracy_date_pct",
                    field_accuracy.get("date_pct"),
                    "percent",
                    "Replay field accuracy: start_date",
                    True,
                ),
                (
                    "field_accuracy_time_pct",
                    field_accuracy.get("time_pct"),
                    "percent",
                    "Replay field accuracy: start_time",
                    True,
                ),
                (
                    "field_accuracy_location_pct",
                    field_accuracy.get("location_pct"),
                    "percent",
                    "Replay field accuracy: location",
                    True,
                ),
                (
                    "field_accuracy_source_pct",
                    field_accuracy.get("source_pct"),
                    "percent",
                    "Replay field accuracy: source",
                    True,
                ),
                (
                    "field_accuracy_address_id_pct",
                    field_accuracy.get("address_id_pct"),
                    "percent",
                    "Replay field accuracy: resolved address_id",
                    True,
                ),
                (
                    "field_accuracy_dance_style_pct",
                    field_accuracy.get("dance_style_pct"),
                    "percent",
                    "Replay field accuracy: dance_style",
                    True,
                ),
                (
                    "field_accuracy_description_pct",
                    field_accuracy.get("description_pct"),
                    "percent",
                    "Replay field accuracy: description",
                    True,
                ),
                (
                    "overall_score_delta_vs_previous_run",
                    (previous_run_info.get("summary") or {}).get("delta_overall_score") if isinstance(previous_run_info.get("summary"), dict) else None,
                    "points",
                    "Phase 4 comparison: overall score delta versus previous run",
                    True,
                ),
                (
                    "holdout_replay_url_accuracy_delta_vs_baseline",
                    (holdout_baseline_info.get("summary") or {}).get("delta_holdout_replay_url_accuracy_pct") if isinstance(holdout_baseline_info.get("summary"), dict) else None,
                    "percent",
                    "Phase 4 comparison: holdout replay URL accuracy delta versus prior holdout baseline",
                    True,
                ),
            ]
        )
        for metric_key, value, metric_unit, description, higher_is_better in metric_specs:
            if value is None:
                continue
            try:
                self.db_handler.record_metric_observation(
                    run_id=str(run_id or ""),
                    metric_key=metric_key,
                    metric_value_numeric=float(value),
                    metric_unit=metric_unit,
                    description=description,
                    higher_is_better=higher_is_better,
                    window_start=window_start,
                    window_end=window_end,
                )
            except Exception:
                continue

    def _persist_validation_artifacts(
        self,
        run_id: str,
        artifacts: dict[str, dict],
    ) -> None:
        """Persist structured validation artifacts by run_id when DB support is available."""
        if not getattr(self, "db_handler", None) or not hasattr(self.db_handler, "record_validation_run_artifact"):
            return
        safe_run_id = str(run_id or "").strip()
        if not safe_run_id:
            return
        for artifact_type, payload in artifacts.items():
            if not isinstance(payload, dict):
                continue
            try:
                self.db_handler.record_validation_run_artifact(
                    run_id=safe_run_id,
                    artifact_type=str(artifact_type or "").strip(),
                    artifact_payload=payload,
                )
            except Exception:
                continue

    def _build_phase1_run_scorecard(
        self,
        run_id: str,
        report_timestamp: str | None,
        accuracy_replay: dict | None,
        scraping_results: dict | None,
        runtime_summary: dict,
        llm_cost_summary: dict,
        chatbot_quality_summary: dict,
        classifier_performance_summary: dict | None,
        duplicate_summary: dict | None,
        event_data_quality_summary: dict | None,
        field_accuracy_summary: dict | None,
        coverage_summary: dict | None,
        dev_summary: dict | None,
        holdout_summary: dict | None,
        domain_capped_summary: dict | None,
        telemetry_integrity_summary: dict | None = None,
    ) -> dict:
        """Build the canonical scorecard artifact."""
        replay = accuracy_replay if isinstance(accuracy_replay, dict) else {}
        scraping = scraping_results if isinstance(scraping_results, dict) else {}
        scraping_summary = scraping.get("summary", {}) if isinstance(scraping.get("summary"), dict) else {}
        source_distribution = scraping.get("source_distribution", {}) if isinstance(scraping.get("source_distribution"), dict) else {}
        classifier = classifier_performance_summary if isinstance(classifier_performance_summary, dict) else {}
        duplicate_info = duplicate_summary if isinstance(duplicate_summary, dict) else {}
        event_data_quality = event_data_quality_summary if isinstance(event_data_quality_summary, dict) else {}
        field_accuracy = field_accuracy_summary if isinstance(field_accuracy_summary, dict) else {}
        coverage_info = coverage_summary if isinstance(coverage_summary, dict) else {}
        dev_info = dev_summary if isinstance(dev_summary, dict) else {}
        holdout_info = holdout_summary if isinstance(holdout_summary, dict) else {}
        domain_capped_info = domain_capped_summary if isinstance(domain_capped_summary, dict) else {}
        telemetry_integrity = telemetry_integrity_summary if isinstance(telemetry_integrity_summary, dict) else {}
        dev_urls = load_dev_urls()

        replay_url_accuracy_pct = float(replay.get("coverage_accuracy_pct", 0.0) or 0.0)
        replay_row_accuracy_pct = float(replay.get("replay_accuracy_pct", 0.0) or 0.0)
        database_accuracy_score = round(replay_url_accuracy_pct, 2) if replay_url_accuracy_pct else None

        total_important_urls = int(scraping_summary.get("important_urls_checked", 0) or 0)
        failed_urls = int(scraping_summary.get("failed_count", 0) or 0)
        events_coverage_score = None
        if total_important_urls > 0:
            events_coverage_score = round(((total_important_urls - failed_urls) / total_important_urls) * 100.0, 2)
        elif source_distribution:
            status = str(source_distribution.get("status", "UNKNOWN") or "UNKNOWN").upper()
            if status == "PASS":
                events_coverage_score = 100.0
            elif status in {"WARN", "WARNING"}:
                events_coverage_score = 75.0
            elif status == "FAIL":
                events_coverage_score = 50.0

        runtime_minutes = runtime_summary.get("pipeline_duration_minutes")
        total_classified_urls = int(classifier.get("total_classified_urls", 0) or 0)
        total_events = int(event_data_quality.get("total_events", 0) or 0)
        runtime_score = None
        if runtime_minutes is not None:
            runtime_score = max(0.0, round(100.0 - float(runtime_minutes) / 6.0, 2))

        total_cost_usd = ((llm_cost_summary.get("summary") or {}).get("total_usd") if isinstance(llm_cost_summary.get("summary"), dict) else None)
        cost_per_processed_url_usd = (
            round(float(total_cost_usd) / total_classified_urls, 6)
            if total_cost_usd is not None and total_classified_urls > 0
            else None
        )
        cost_per_inserted_event_usd = (
            round(float(total_cost_usd) / total_events, 6)
            if total_cost_usd is not None and total_events > 0
            else None
        )
        run_cost_score = None
        if total_cost_usd is not None:
            run_cost_score = max(0.0, round(100.0 - (float(total_cost_usd) * 5.0), 2))

        chatbot_score = ((chatbot_quality_summary.get("summary") or {}).get("chatbot_answer_correctness_pct") if isinstance(chatbot_quality_summary.get("summary"), dict) else None)

        component_scores = [
            score for score in (
                database_accuracy_score,
                events_coverage_score,
                runtime_score,
                run_cost_score,
            ) if score is not None
        ]
        overall_value = round(sum(component_scores) / len(component_scores), 2) if component_scores else None

        return {
            "run_id": str(run_id or ""),
            "run_timestamp_utc": str(report_timestamp or datetime.utcnow().isoformat()),
            "code_version": self._get_code_version_info(),
            "scorecard_version": "phase4",
            "evaluation_scope": {
                "environment": str(os.getenv("RENDER_ENVIRONMENT", "local") or "local"),
                "uses_holdout": bool(holdout_info.get("available")),
                "holdout_version": str(holdout_info.get("holdout_version", "") or ""),
                "uses_dev_split": bool(dev_urls),
                "dev_version": "v1" if dev_urls else "",
                "dev_urls_total": len(dev_urls),
                "watchlist_version": str(coverage_info.get("watchlist_source", "") or ""),
                "dev_summary": dev_info,
                "holdout_summary": holdout_info,
                "domain_capped_summary": domain_capped_info,
                "notes": "Phase 4 scorecard adds train/dev/holdout separation metadata, holdout evaluation, domain-capped replay summaries, explicit guardrail evaluation, and run-to-run comparison hooks.",
            },
            "kpis": {
                "database_accuracy": {
                    "score": database_accuracy_score,
                    "summary": {
                        "replay_url_accuracy_pct": replay_url_accuracy_pct,
                        "replay_row_accuracy_pct": replay_row_accuracy_pct,
                        "total_rows": int(replay.get("total_rows", 0) or 0),
                        "true_count": int(replay.get("true_count", 0) or 0),
                        "false_count": int(replay.get("false_count", 0) or 0),
                        "duplicate_rate_per_100_events": duplicate_info.get("duplicate_rate_per_100_events"),
                        "severe_duplicate_rate_per_100_events": duplicate_info.get("severe_duplicate_rate_per_100_events"),
                        "invalid_event_rate_pct": event_data_quality.get("invalid_event_rate_pct"),
                        "stale_event_rate_pct": event_data_quality.get("stale_event_rate_pct"),
                        "classifier_effect": {
                            "ml_usage_pct": float(classifier.get("ml_usage_pct", 0.0) or 0.0),
                            "stage_counts": classifier.get("stage_counts", {}) if isinstance(classifier.get("stage_counts"), dict) else {},
                            "stage_details": classifier.get("stage_details", []) if isinstance(classifier.get("stage_details"), list) else [],
                        },
                    },
                    "field_accuracy": field_accuracy,
                },
                "events_coverage": {
                    "score": events_coverage_score,
                    "summary": {
                        "important_urls_checked": total_important_urls,
                        "failed_urls": failed_urls,
                        "whitelist_failures": int(scraping_summary.get("whitelist_failures", 0) or 0),
                        "edge_case_failures": int(scraping_summary.get("edge_case_failures", 0) or 0),
                        "source_distribution_status": str(source_distribution.get("status", "") or ""),
                        "watchlist_source_hit_rate_pct": coverage_info.get("source_hit_rate_pct"),
                        "watchlist_event_capture_rate_pct": coverage_info.get("event_capture_rate_pct"),
                        "priority_source_hit_rate_pct": coverage_info.get("priority_source_hit_rate_pct"),
                        "new_source_discovery_count": coverage_info.get("new_source_discovery_count"),
                        "missed_event_rate_manual_audit_pct": coverage_info.get("missed_event_rate_manual_audit_pct"),
                        "watchlist_source": coverage_info.get("watchlist_source"),
                        "dev_replay_url_accuracy_pct": dev_info.get("replay_url_accuracy_pct"),
                        "holdout_replay_url_accuracy_pct": holdout_info.get("replay_url_accuracy_pct"),
                        "domain_capped_replay_url_accuracy_pct": domain_capped_info.get("replay_url_accuracy_pct"),
                    },
                },
                "run_time": {
                    "score": runtime_score,
                    "summary": {
                        **runtime_summary,
                        "urls_processed_per_minute": (
                            round(total_classified_urls / float(runtime_minutes), 4)
                            if runtime_minutes not in (None, 0, 0.0) and total_classified_urls > 0
                            else None
                        ),
                        "events_inserted_per_minute": (
                            round(total_events / float(runtime_minutes), 4)
                            if runtime_minutes not in (None, 0, 0.0) and total_events > 0
                            else None
                        ),
                    },
                },
                "run_costs": {
                    "score": run_cost_score,
                    "summary": {
                        **llm_cost_summary,
                        "summary": {
                            **((llm_cost_summary.get("summary") or {}) if isinstance(llm_cost_summary.get("summary"), dict) else {}),
                            "cost_per_processed_url_usd": cost_per_processed_url_usd,
                            "cost_per_inserted_event_usd": cost_per_inserted_event_usd,
                        },
                    },
                },
                "chatbot_quality": {
                    "score": chatbot_score,
                    "summary": chatbot_quality_summary,
                },
            },
            "guardrails": {},
            "telemetry_integrity": telemetry_integrity,
            "overall_score": {
                "value": overall_value,
                "status": "PRELIMINARY",
            },
            "comparison_summary": {},
            "recommendations_input": {},
        }

    def _build_phase1_scorecard_html(self, run_scorecard: dict) -> str:
        """Render a compact scorecard section for the HTML report."""
        if not isinstance(run_scorecard, dict) or not run_scorecard:
            return "<p class='error-box'>❌ Phase 1 scorecard unavailable</p>"
        kpis = run_scorecard.get("kpis", {}) if isinstance(run_scorecard.get("kpis"), dict) else {}
        cards: list[str] = []
        for key, title in (
            ("database_accuracy", "Database Accuracy"),
            ("events_coverage", "Events Coverage"),
            ("run_time", "Run Time"),
            ("run_costs", "Run Costs"),
            ("chatbot_quality", "Chatbot Quality"),
        ):
            metric = kpis.get(key, {}) if isinstance(kpis.get(key), dict) else {}
            score = metric.get("score")
            score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "n/a"
            cards.append(
                "<div class='metric'>"
                f"<div class='metric-value'>{self._escape_html(score_text)}</div>"
                f"<div class='metric-label'>{self._escape_html(title)}</div>"
                "</div>"
            )

        chatbot_summary = ((kpis.get("chatbot_quality") or {}).get("summary") or {}).get("summary", {})
        runtime_summary = ((kpis.get("run_time") or {}).get("summary") or {})
        cost_summary = ((kpis.get("run_costs") or {}).get("summary") or {}).get("summary", {})
        telemetry_integrity = run_scorecard.get("telemetry_integrity", {}) if isinstance(run_scorecard.get("telemetry_integrity"), dict) else {}
        bullet_lines = [
            f"Pipeline duration: {float(runtime_summary.get('pipeline_duration_minutes', 0.0) or 0.0):.2f} minutes",
            f"Total LLM cost: ${float(cost_summary.get('total_usd', 0.0) or 0.0):.4f}",
            f"Telemetry integrity: {self._escape_html(str(telemetry_integrity.get('status', 'UNKNOWN') or 'UNKNOWN'))}",
        ]
        within_15s = chatbot_summary.get("chatbot_response_within_15s_pct")
        correctness = chatbot_summary.get("chatbot_answer_correctness_pct")
        holdout_summary = (run_scorecard.get("evaluation_scope") or {}).get("holdout_summary", {})
        domain_capped_summary = (run_scorecard.get("evaluation_scope") or {}).get("domain_capped_summary", {})
        if within_15s is not None:
            bullet_lines.append(f"Chatbot within 15s: {float(within_15s):.2f}%")
        if correctness is not None:
            bullet_lines.append(f"Chatbot correctness: {float(correctness):.2f}%")
        holdout_accuracy = holdout_summary.get("replay_url_accuracy_pct")
        if holdout_accuracy is not None:
            bullet_lines.append(f"Holdout replay URL accuracy: {float(holdout_accuracy):.2f}%")
        domain_capped_accuracy = domain_capped_summary.get("replay_url_accuracy_pct")
        if domain_capped_accuracy is not None:
            bullet_lines.append(f"Domain-capped replay URL accuracy: {float(domain_capped_accuracy):.2f}%")

        bullets = "".join(f"<li>{self._escape_html(line)}</li>" for line in bullet_lines)
        guardrails = run_scorecard.get("guardrails", {}) if isinstance(run_scorecard.get("guardrails"), dict) else {}
        status_text = str(run_scorecard.get("overall_score", {}).get("status", "UNKNOWN") or "UNKNOWN")
        return (
            "<div class='metric-container'>"
            + "".join(cards)
            + "</div>"
            + f"<p><strong>Status:</strong> {self._escape_html(status_text)}. "
            f"<strong>Guardrails:</strong> {self._escape_html(str(guardrails.get('status', 'UNKNOWN') or 'UNKNOWN'))}.</p>"
            + f"<ul>{bullets}</ul>"
        )

    def _build_fb_ig_url_funnel_html(self, funnel_data: dict) -> str:
        """Render HTML for FB/IG URL funnel summary."""
        if not funnel_data:
            return "<p class='error-box'>❌ FB/IG URL funnel summary unavailable</p>"
        if funnel_data.get("error"):
            return (
                "<p class='error-box'>❌ FB/IG URL funnel summary unavailable: "
                f"{self._escape_html(str(funnel_data.get('error', 'unknown')))}</p>"
            )

        passed = int(funnel_data.get("urls_passed_for_scraping", 0) or 0)
        events = int(funnel_data.get("urls_with_events", 0) or 0)
        rate = funnel_data.get("events_over_passed_rate")
        rate_display = f"{float(rate):.1%}" if rate is not None else "N/A"

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{int(funnel_data.get('fb_ig_urls_in_db', 0) or 0)}</div>
                <div class="metric-label">FB/IG URLs in DB</div>
            </div>
            <div class="metric">
                <div class="metric-value">{passed}</div>
                <div class="metric-label">URLs Passed for Scraping</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(funnel_data.get('urls_attempted_scraping', 0) or 0)}</div>
                <div class="metric-label">URLs Attempted Scraping</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(funnel_data.get('urls_with_keywords', 0) or 0)}</div>
                <div class="metric-label">URLs with Keywords</div>
            </div>
            <div class="metric">
                <div class="metric-value">{events}</div>
                <div class="metric-label">URLs with Events</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rate_display}</div>
                <div class="metric-label">Events / Passed %</div>
            </div>
        </div>
        """
        if passed > 0:
            html += (
                "<p><strong>Formula:</strong> "
                f"urls_with_events / urls_passed_for_scraping = {events} / {passed} = {rate_display}</p>"
            )
        else:
            html += "<p><strong>Formula:</strong> urls_with_events / urls_passed_for_scraping = N/A (passed=0)</p>"
        html += (
            f"<p><strong>Window:</strong> {self._escape_html(str(funnel_data.get('start_ts', '')))}"
            f" to {self._escape_html(str(funnel_data.get('end_ts', '')))} "
            f"(last {int(funnel_data.get('window_hours', 24) or 24)} hour(s))</p>"
        )
        return html

    def _build_llm_provider_activity_html(self, llm_data: dict) -> str:
        """Build HTML for LLM provider usage/cost pressure section."""
        if not llm_data:
            return "<p class='error-box'>❌ LLM activity summary unavailable</p>"

        total_attempts = int(llm_data.get("total_accesses", llm_data.get("total_attempts", 0)))
        exhausted = int(llm_data.get("provider_exhausted_count", 0))
        lines_scanned = int(llm_data.get("lines_scanned", 0))
        total_rate_limits = int(llm_data.get("total_rate_limits", 0))
        total_timeouts = int(llm_data.get("total_timeouts", 0))
        pressure_level = str(llm_data.get("pressure_level", "LOW")).upper()
        pressure_class = {
            "LOW": "status-pass",
            "MEDIUM": "status-warning",
            "HIGH": "status-fail",
        }.get(pressure_level, "status-warning")

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{total_attempts}</div>
                <div class="metric-label">LLM Accesses (Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{exhausted}</div>
                <div class="metric-label">All-Provider Exhausted</div>
            </div>
            <div class="metric">
                <div class="metric-value">{lines_scanned}</div>
                <div class="metric-label">Log Lines Scanned</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_rate_limits}</div>
                <div class="metric-label">Rate Limits (All Providers)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_timeouts}</div>
                <div class="metric-label">Timeouts (All Providers)</div>
            </div>
        </div>
        """

        html += (
            "<p><strong>Cost Pressure:</strong> "
            f"<span class='{pressure_class}'>{self._escape_html(pressure_level)}</span></p>"
        )
        reasons = llm_data.get("pressure_reasons", [])
        if reasons:
            html += "<ul>"
            for reason in reasons:
                html += f"<li>{self._escape_html(reason)}</li>"
            html += "</ul>"

        html += (
            "<p><strong>Window:</strong> "
            f"{self._escape_html(llm_data.get('start_ts', ''))} to {self._escape_html(llm_data.get('end_ts', ''))} "
            f"({int(llm_data.get('window_hours', 24))}h)</p>"
        )
        html += (
            "<p><strong>Denominator:</strong> "
            f"{self._escape_html(str(llm_data.get('denominator_label', 'Total LLM accesses in window')))}"
            f" = {total_attempts}</p>"
        )

        html += "<h3>Provider Breakdown</h3>"
        html += "<table><tr><th>Provider</th><th>Attempts</th><th>Successes</th><th>Failures</th><th>Timeouts</th><th>Rate Limits</th></tr>"
        providers = llm_data.get("providers", {})
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            p = providers.get(provider, {})
            html += (
                "<tr>"
                f"<td>{provider}</td>"
                f"<td>{int(p.get('attempts', 0))}</td>"
                f"<td>{int(p.get('successes', 0))}</td>"
                f"<td>{int(p.get('failures', 0))}</td>"
                f"<td>{int(p.get('timeouts', 0))}</td>"
                f"<td>{int(p.get('rate_limits', 0))}</td>"
                "</tr>"
            )
        html += "</table>"

        top_files = llm_data.get("top_files", [])
        if top_files:
            html += "<h3>Top Log Files by LLM Attempts</h3><table><tr><th>Log File</th><th>Attempts</th></tr>"
            for path, attempts in top_files:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(os.path.basename(path))}</td>"
                    f"<td>{int(attempts)}</td>"
                    "</tr>"
                )
            html += "</table>"

        top_models = llm_data.get("top_models", [])
        if top_models:
            html += "<h3>Top Provider/Model Attempts</h3><table><tr><th>Provider:Model</th><th>Attempts</th></tr>"
            for model_key, attempts in top_models:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(model_key)}</td>"
                    f"<td>{int(attempts)}</td>"
                    "</tr>"
                )
            html += "</table>"

        return html

    def _build_llm_extraction_quality_html(self, quality_data: dict) -> str:
        """Build HTML for LLM event-extraction quality scorecard section."""
        if not quality_data:
            return "<p class='error-box'>❌ LLM extraction quality summary unavailable</p>"

        total_urls = int(quality_data.get("total_urls", 0) or 0)
        total_attempts = int(quality_data.get("total_attempts", 0) or 0)
        successful_urls = int(quality_data.get("successful_urls", 0) or 0)
        hard_failed_urls = int(quality_data.get("hard_failed_urls", 0) or 0)
        no_events_urls = int(quality_data.get("no_events_urls", 0) or 0)
        too_short_urls = int(quality_data.get("too_short_urls", 0) or 0)
        total_retries = int(quality_data.get("total_retries", 0) or 0)
        schema_errors = int(quality_data.get("schema_errors", 0) or 0)
        parse_success_rate = float(quality_data.get("parse_success_rate", 0) or 0)
        hard_failure_rate = float(quality_data.get("hard_failure_rate", 0) or 0)

        status = "GOOD"
        if hard_failure_rate >= 0.20 or parse_success_rate < 0.40:
            status = "POOR"
        elif hard_failure_rate >= 0.10 or parse_success_rate < 0.60:
            status = "FAIR"

        status_class = {
            "GOOD": "status-pass",
            "FAIR": "status-warning",
            "POOR": "status-fail",
        }[status]

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{total_urls}</div>
                <div class="metric-label">URLs Evaluated</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_attempts}</div>
                <div class="metric-label">Extraction Attempts</div>
            </div>
            <div class="metric">
                <div class="metric-value">{successful_urls}</div>
                <div class="metric-label">Successful Writes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{hard_failed_urls}</div>
                <div class="metric-label">Hard Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{no_events_urls + too_short_urls}</div>
                <div class="metric-label">Non-Fatal Misses</div>
            </div>
            <div class="metric">
                <div class="metric-value">{schema_errors}</div>
                <div class="metric-label">Schema/Column Errors</div>
            </div>
        </div>
        """
        html += (
            "<p><strong>Quality Status:</strong> "
            f"<span class='{status_class}'>{self._escape_html(status)}</span></p>"
        )
        html += (
            "<p><strong>Window:</strong> "
            f"{self._escape_html(quality_data.get('start_ts', ''))} to {self._escape_html(quality_data.get('end_ts', ''))} "
            f"({int(quality_data.get('window_hours', 24))}h)</p>"
        )
        html += (
            "<p><strong>Parse Success Rate:</strong> "
            f"{parse_success_rate:.1%} | "
            f"<strong>Hard Failure Rate:</strong> {hard_failure_rate:.1%} | "
            f"<strong>Retries:</strong> {total_retries}</p>"
        )

        html += (
            "<h3>Model Breakdown (Estimated Mapping)</h3>"
            "<table><tr><th>Provider:Model</th><th>Attempts</th><th>Successes</th><th>Hard Failures</th>"
            "<th>No Events</th><th>Too Short</th><th>Retries</th><th>Schema Errors</th></tr>"
        )
        for model_key, stats in quality_data.get("models", []):
            html += (
                "<tr>"
                f"<td>{self._escape_html(model_key)}</td>"
                f"<td>{int(stats.get('attempts', 0))}</td>"
                f"<td>{int(stats.get('successes', 0))}</td>"
                f"<td>{int(stats.get('hard_failures', 0))}</td>"
                f"<td>{int(stats.get('no_events', 0))}</td>"
                f"<td>{int(stats.get('too_short', 0))}</td>"
                f"<td>{int(stats.get('retries', 0))}</td>"
                f"<td>{int(stats.get('schema_errors', 0))}</td>"
                "</tr>"
            )
        html += "</table>"

        top_failing = quality_data.get("top_failing_urls", [])
        if top_failing:
            html += "<h3>Top Failing URLs</h3><table><tr><th>URL</th><th>Failure Count</th></tr>"
            for url, count in top_failing[:10]:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(url)}</td>"
                    f"<td>{int(count)}</td>"
                    "</tr>"
                )
            html += "</table>"

        top_pairs = quality_data.get("top_failing_url_model_pairs", [])
        if top_pairs:
            html += "<h3>Top Failing URL/Model Pairs</h3><table><tr><th>URL</th><th>Provider:Model</th><th>Failure Count</th></tr>"
            for row in top_pairs[:10]:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(str(row.get('url', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('model', 'unknown')))}</td>"
                    f"<td>{int(row.get('count', 0) or 0)}</td>"
                    "</tr>"
                )
            html += "</table>"

        html += "<p><em>Model mapping is estimated from log sequence and intended for trend monitoring.</em></p>"
        return html

    def _summarize_reliability_scorecard(
        self,
        results: dict,
        llm_data: dict,
        alias_data: dict,
        scraper_network_data: dict | None = None,
        fb_block_data: dict | None = None,
    ) -> dict:
        """Build baseline reliability metrics from report-driving data."""
        scraping = results.get('scraping_validation') or {}
        scraping_summary = scraping.get('summary', {}) if isinstance(scraping, dict) else {}
        critical_failures = len(scraping.get('critical_failures', []) or []) if isinstance(scraping, dict) else 0
        total_important_urls = int(scraping_summary.get("total_important_urls", 0) or 0)
        attempted_url_denominator = int(scraping_summary.get("attempted_url_denominator", 0) or 0)
        post_scrape_failures = int(scraping_summary.get("post_scrape_failures", 0) or 0)
        attempted_failure_rate = float(scraping_summary.get("attempted_failure_rate", 0) or 0) if attempted_url_denominator > 0 else 0.0

        chatbot = results.get('chatbot_testing') or {}
        chatbot_summary = chatbot.get('summary', {}) if isinstance(chatbot, dict) else {}
        avg_score = float(chatbot_summary.get('average_score', 0) or 0)
        exec_rate = float(chatbot_summary.get('execution_success_rate', 0) or 0)
        chatbot_total_tests = int(chatbot_summary.get("total_tests", 0) or 0)
        chatbot_execution_success_count = int(round(exec_rate * chatbot_total_tests)) if chatbot_total_tests > 0 else 0

        total_attempts = int(
            (llm_data or {}).get('total_accesses', (llm_data or {}).get('total_attempts', 0)) or 0
        )
        providers = (llm_data or {}).get('providers', {}) or {}
        total_successes = sum(int((providers.get(p, {}) or {}).get('successes', 0) or 0) for p in ("openai", "openrouter", "mistral", "gemini"))
        total_rate_limits = int((llm_data or {}).get('total_rate_limits', 0) or 0)
        total_timeouts = int((llm_data or {}).get('total_timeouts', 0) or 0)
        exhausted = int((llm_data or {}).get('provider_exhausted_count', 0) or 0)
        llm_success_rate = (total_successes / total_attempts) if total_attempts else 0.0
        rate_limit_rate = (total_rate_limits / total_attempts) if total_attempts else 0.0

        decision_counts = (alias_data or {}).get('decision_counts', {}) or {}
        alias_conflicts = int(decision_counts.get('skipped_conflict', 0) or 0)
        scraper_network = scraper_network_data or {}
        scraper_exception_rate = float(scraper_network.get("exception_rate", 0) or 0)
        scraper_retry_max_reached = int(scraper_network.get("retry_max_reached", 0) or 0)
        scraper_timeout_count = int(scraper_network.get("timeout_count", 0) or 0)
        scraper_degraded = bool(scraper_network.get("degraded", False))
        failure_types = scraping_summary.get("failure_types", {}) if isinstance(scraping_summary, dict) else {}
        failure_stage_counts = scraping_summary.get("failure_stage_counts", {}) if isinstance(scraping_summary, dict) else {}
        scrape_not_accessed_urls = int((failure_types or {}).get("not_attempted", 0) or 0)
        scrape_keyword_misses_after_access = int(
            scraping_summary.get("keyword_failures_after_scrape", 0) or 0
        )
        attempted_no_keywords = int((failure_stage_counts or {}).get("attempted_no_keywords", 0) or 0)
        attempted_extraction_or_llm_failure = int(
            (failure_stage_counts or {}).get("attempted_extraction_or_llm_failure", 0) or 0
        )
        other_post_scrape_reasons = max(
            0,
            post_scrape_failures - attempted_no_keywords - attempted_extraction_or_llm_failure,
        )
        scrape_total_failure_reasons = (
            f"attempted_no_keywords={attempted_no_keywords}; "
            f"attempted_extraction_or_llm_failure={attempted_extraction_or_llm_failure}; "
            f"other_post_scrape={other_post_scrape_reasons}"
        )
        keyword_miss_rate = (
            scrape_keyword_misses_after_access / attempted_url_denominator
            if attempted_url_denominator > 0 else 0.0
        )
        fb_blocks = fb_block_data or {}
        fb_explicit_blocks = int(fb_blocks.get("explicit_block_count", 0) or 0)
        fb_abort_count = int(fb_blocks.get("abort_count", 0) or 0)

        # URL-level grading score: only URL/test-level quality metrics.
        url_level_score = 100.0
        url_level_score -= min(30.0, critical_failures * 3.0)
        url_level_score -= max(0.0, min(20.0, 90.0 - avg_score))
        url_level_score -= max(0.0, min(20.0, (1.0 - exec_rate) * 20.0))
        url_level_score -= min(15.0, attempted_failure_rate * 100.0)
        url_level_score -= min(10.0, keyword_miss_rate * 100.0)
        url_level_score = max(0.0, min(100.0, url_level_score))

        # Request-level telemetry score: operational network/provider pressure indicators.
        request_level_score = 100.0
        request_level_score -= min(15.0, rate_limit_rate * 100.0)
        request_level_score -= min(10.0, exhausted * 2.0)
        request_level_score -= min(20.0, scraper_exception_rate * 100.0)
        request_level_score -= min(15.0, scraper_timeout_count / 5.0)
        request_level_score -= min(8.0, fb_explicit_blocks * 1.0)
        request_level_score -= min(12.0, fb_abort_count * 6.0)
        request_level_score = max(0.0, min(100.0, request_level_score))

        # Backward-compatible top-level score now aligns to URL-level grading.
        score = url_level_score

        if score >= 85:
            status = "HEALTHY"
        elif score >= 70:
            status = "WATCH"
        else:
            status = "AT_RISK"

        return {
            "timestamp": results.get("timestamp", datetime.now().isoformat()),
            "status": status,
            "score": round(score, 1),
            "url_level_score": round(url_level_score, 1),
            "request_level_score": round(request_level_score, 1),
            "metrics": {
                "scrape_critical_failures": critical_failures,
                "scrape_total_failures": int(scraping_summary.get('total_failures', 0) or 0),
                "scrape_total_failure_reasons": scrape_total_failure_reasons,
                "scrape_total_important_urls": total_important_urls,
                "scrape_attempted_url_denominator": attempted_url_denominator,
                "scrape_attempted_failure_rate": round(attempted_failure_rate, 4) if attempted_url_denominator > 0 else None,
                "scrape_not_accessed_urls": scrape_not_accessed_urls,
                "scrape_keyword_misses_after_access": scrape_keyword_misses_after_access,
                "scrape_keyword_miss_rate_after_access": round(keyword_miss_rate, 4) if attempted_url_denominator > 0 else None,
                "chatbot_average_score": round(avg_score, 2),
                "chatbot_execution_success_rate": round(exec_rate, 4),
                "chatbot_total_tests": chatbot_total_tests,
                "chatbot_execution_success_count": chatbot_execution_success_count,
                "llm_attempts": total_attempts,
                "llm_success_rate": round(llm_success_rate, 4),
                "llm_rate_limits": total_rate_limits,
                "llm_timeouts": total_timeouts,
                "llm_provider_exhausted": exhausted,
                "llm_cost_pressure": (llm_data or {}).get("pressure_level", "LOW"),
                "address_alias_conflict_skips": alias_conflicts,
                "scraper_exception_rate": round(scraper_exception_rate, 4),
                "scraper_retry_max_reached": scraper_retry_max_reached,
                "scraper_timeout_count": scraper_timeout_count,
                "scraper_network_degraded": scraper_degraded,
                "fb_explicit_temp_blocks": fb_explicit_blocks,
                "fb_temp_block_aborts": fb_abort_count,
            },
            "metric_families": {
                "url_level_grading": {
                    "score": round(url_level_score, 1),
                    "metrics": {
                        "scrape_critical_failures": critical_failures,
                        "scrape_total_important_urls": total_important_urls,
                        "scrape_post_scrape_failures": post_scrape_failures,
                        "scrape_attempted_url_denominator": attempted_url_denominator,
                        "scrape_attempted_failure_rate": round(attempted_failure_rate, 4) if attempted_url_denominator > 0 else None,
                        "scrape_not_accessed_urls": scrape_not_accessed_urls,
                        "scrape_keyword_misses_after_access": scrape_keyword_misses_after_access,
                        "scrape_keyword_miss_rate_after_access": round(keyword_miss_rate, 4) if attempted_url_denominator > 0 else None,
                        "chatbot_average_score": round(avg_score, 2),
                        "chatbot_execution_success_rate": round(exec_rate, 4),
                        "chatbot_total_tests": chatbot_total_tests,
                        "chatbot_execution_success_count": chatbot_execution_success_count,
                    },
                },
                "request_level_telemetry": {
                    "score": round(request_level_score, 1),
                    "metrics": {
                        "llm_attempts": total_attempts,
                        "llm_success_rate": round(llm_success_rate, 4),
                        "llm_rate_limits": total_rate_limits,
                        "llm_timeouts": total_timeouts,
                        "llm_provider_exhausted": exhausted,
                        "llm_cost_pressure": (llm_data or {}).get("pressure_level", "LOW"),
                        "scraper_exception_rate": round(scraper_exception_rate, 4),
                        "scraper_retry_max_reached": scraper_retry_max_reached,
                        "scraper_timeout_count": scraper_timeout_count,
                        "scraper_network_degraded": scraper_degraded,
                        "fb_explicit_temp_blocks": fb_explicit_blocks,
                        "fb_temp_block_aborts": fb_abort_count,
                    },
                },
            },
        }

    def _extract_reliability_issues(
        self,
        results: dict,
        llm_data: dict,
        alias_data: dict,
        scraper_network_data: dict | None = None,
        fb_block_data: dict | None = None,
    ) -> list[dict]:
        """Normalize major reliability signals into a JSON-friendly issue list."""
        now_ts = str(results.get("timestamp") or datetime.now().isoformat())
        issues: list[dict] = []

        scraping = results.get('scraping_validation') or {}
        for idx, failure in enumerate((scraping.get('critical_failures') or [])[:30], start=1):
            url = str(failure.get('url', '') or '')
            reason = str(failure.get('reason_code', failure.get('failure_type', 'unknown')) or 'unknown')
            issues.append({
                "issue_id": f"SCRAPE-{idx:03d}",
                "timestamp": now_ts,
                "category": "Scrape Coverage Miss",
                "severity": "high",
                "step": "scraping_validation",
                "provider": "",
                "url": url,
                "input_signature": reason,
                "expected": "Event source should produce expected events.",
                "actual": f"Critical scrape failure detected ({reason}).",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        chatbot = results.get('chatbot_testing') or {}
        for idx, cat in enumerate((chatbot.get('problem_categories') or [])[:10], start=1):
            ex = cat.get('example', {}) or {}
            issues.append({
                "issue_id": f"CHATBOT-{idx:03d}",
                "timestamp": now_ts,
                "category": "LLM Semantic Failure",
                "severity": "medium",
                "step": "chatbot_testing",
                "provider": "",
                "url": "",
                "input_signature": str(cat.get('name', 'unknown')),
                "expected": "Chatbot intent + SQL output should meet quality threshold.",
                "actual": str(ex.get('reason', 'Category below target threshold.')),
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        perf_summary = self._summarize_chatbot_performance(results.get("timestamp"))
        if isinstance(perf_summary, dict) and not perf_summary.get("error"):
            p95_ms = float(((perf_summary.get("query_latency_ms") or {}).get("p95", 0.0) or 0.0))
            unfinished = int(perf_summary.get("unfinished_request_count", 0) or 0)
            perf_thresholds = perf_summary.get("thresholds", {}) or {}
            warn_p95 = float(perf_thresholds.get("warn_query_p95_ms", 10000) or 10000)
            warn_unfinished = int(perf_thresholds.get("warn_unresolved_count", 1) or 1)
            if p95_ms >= warn_p95 or unfinished >= warn_unfinished:
                issues.append({
                    "issue_id": "CHATBOT-PERF-001",
                    "timestamp": now_ts,
                    "category": "Chatbot Performance Degradation",
                    "severity": "high" if str(perf_summary.get("status", "WATCH")).upper() == "DEGRADED" else "medium",
                    "step": "chatbot_performance",
                    "provider": "",
                    "url": "",
                    "input_signature": "latency_or_unfinished_requests",
                    "expected": "Chatbot /query latency should remain below p95 threshold with no stuck requests.",
                    "actual": f"query_p95_ms={p95_ms}, unfinished_requests={unfinished}",
                    "status": "open",
                    "owner": "unassigned",
                    "first_seen": now_ts,
                    "last_seen": now_ts,
                })

        providers = (llm_data or {}).get("providers", {}) or {}
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            pdata = providers.get(provider, {}) or {}
            failures = int(pdata.get("failures", 0) or 0)
            timeouts = int(pdata.get("timeouts", 0) or 0)
            rate_limits = int(pdata.get("rate_limits", 0) or 0)
            if failures <= 0 and timeouts <= 0 and rate_limits <= 0:
                continue
            issues.append({
                "issue_id": f"LLM-{provider.upper()}-001",
                "timestamp": now_ts,
                "category": "Provider Reliability Failure",
                "severity": "high" if (timeouts > 0 or rate_limits > 0) else "medium",
                "step": "llm_activity",
                "provider": provider,
                "url": "",
                "input_signature": "provider_failure_signal",
                "expected": "Provider should respond within timeout and quota limits.",
                "actual": f"failures={failures}, timeouts={timeouts}, rate_limits={rate_limits}",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        decision_counts = (alias_data or {}).get('decision_counts', {}) or {}
        alias_conflicts = int(decision_counts.get('skipped_conflict', 0) or 0)
        if alias_conflicts > 0:
            issues.append({
                "issue_id": "ADDRESS-ALIAS-001",
                "timestamp": now_ts,
                "category": "Data Integrity Failure",
                "severity": "medium",
                "step": "address_alias_audit",
                "provider": "",
                "url": "",
                "input_signature": "skipped_conflict",
                "expected": "Alias normalization should avoid unresolved conflicts.",
                "actual": f"Conflict skips observed: {alias_conflicts}",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        fb_blocks = fb_block_data or {}
        fb_explicit_blocks = int(fb_blocks.get("explicit_block_count", 0) or 0)
        fb_abort_count = int(fb_blocks.get("abort_count", 0) or 0)
        if fb_explicit_blocks > 0:
            issues.append({
                "issue_id": "FB-BLOCK-001",
                "timestamp": now_ts,
                "category": "Facebook Throttle",
                "severity": "high" if fb_abort_count > 0 else "medium",
                "step": "fb.py",
                "provider": "facebook",
                "url": "",
                "input_signature": "explicit_temp_block_regex",
                "expected": "Facebook crawl should progress without temp-block triggers.",
                "actual": (
                    f"explicit_blocks={fb_explicit_blocks}, "
                    f"fb_run_aborts={fb_abort_count}"
                ),
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        gates = results.get('reliability_gates') or {}
        for gate in (gates.get('failed_gates') or []):
            gate_name = str(gate.get('name', 'unknown_gate'))
            numerator = gate.get('numerator')
            denominator = gate.get('denominator')
            detail = str(gate.get('detail', 'Gate threshold breached.'))
            if numerator and denominator and "[n=" not in detail:
                detail = f"{detail} [n={numerator} d={denominator}]"
            issues.append({
                "issue_id": f"GATE-{gate_name.upper().replace('-', '_')}",
                "timestamp": now_ts,
                "category": "Reliability Gate Failure",
                "severity": "high",
                "step": "reliability_gates",
                "provider": "",
                "url": "",
                "input_signature": gate_name,
                "expected": f"Gate '{gate_name}' threshold should pass.",
                "actual": detail,
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        return issues

    def _build_reliability_scorecard_html(
        self,
        scorecard: dict,
        issues: list[dict],
        gates: dict | None = None,
        trends: dict | None = None,
        registry_summary: dict | None = None,
    ) -> str:
        """Render baseline reliability scorecard + normalized issue summary."""
        if not scorecard:
            return "<p class='error-box'>❌ Reliability scorecard unavailable</p>"

        status = str(scorecard.get("status", "WATCH")).upper()
        status_class = {
            "HEALTHY": "status-pass",
            "WATCH": "status-warning",
            "AT_RISK": "status-fail",
        }.get(status, "status-warning")
        metrics = scorecard.get("metrics", {}) or {}
        open_issues = len(issues or [])

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{float(scorecard.get('score', 0)):.1f}</div>
                <div class="metric-label">Reliability Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self._escape_html(status)}</div>
                <div class="metric-label">Reliability Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{open_issues}</div>
                <div class="metric-label">Normalized Open Issues</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(metrics.get('llm_attempts', 0))}</div>
                <div class="metric-label">LLM Accesses (Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(metrics.get('llm_rate_limits', 0))}</div>
                <div class="metric-label">LLM Rate Limits</div>
            </div>
        </div>
        <p><strong>Status:</strong> <span class="{status_class}">{self._escape_html(status)}</span></p>
        """

        metric_descriptions = {
            "scrape_critical_failures": "Critical whitelist/edge-case scrape failures. Source scope: cross-pipeline DB validation (multi-source), not scraper_log-only.",
            "scrape_total_failures": "Post-scrape failures across important URLs (excludes not-attempted URLs; includes keyword/irrelevant failures after scrape). Source scope: cross-pipeline DB validation (multi-source), not scraper_log-only.",
            "scrape_total_failure_reasons": "Reason composition of scrape_total_failures (attempted URL failures only). Counts are categorized into no-keyword, extraction/LLM, and other post-scrape causes.",
            "scrape_not_accessed_urls": "Important URLs with no scrape attempt in-window (not_attempted). Source scope: scraping validation URL checks.",
            "scrape_keyword_misses_after_access": "Attempted URLs classified irrelevant/no-keyword after access (marked_irrelevant). Source scope: scraping validation URL checks.",
            "scraper_exception_rate": "Downloader exceptions / scraper requests. Source scope: scraper.py stats from logs/scraper_log.txt only.",
            "scraper_retry_max_reached": "Requests that hit retry max. Source scope: scraper.py stats from logs/scraper_log.txt only.",
            "scraper_timeout_count": "Timeout retry count. Source scope: scraper.py stats from logs/scraper_log.txt only.",
            "scraper_network_degraded": "Network threshold breach flag. Source scope: derived from scraper.py stats in logs/scraper_log.txt only.",
            "chatbot_average_score": "Average chatbot test score. Source scope: chatbot validation run outputs (not scraper logs).",
            "chatbot_execution_success_rate": "Chatbot test execution success ratio. Source scope: chatbot validation run outputs (not scraper logs).",
            "llm_success_rate": "LLM successes / LLM accesses. Denominator counts all query access lines across multi-log provider activity (emails/rd_ext/scraper/fb/validation_tests/etc.).",
            "llm_provider_exhausted": "Times all configured LLM providers failed in a call chain. Source scope: multi-log provider activity summary.",
            "llm_cost_pressure": "Cost/volume pressure level from recent LLM traffic. Source scope: multi-log provider activity summary.",
            "address_alias_conflict_skips": "Address alias rows skipped due to conflicts. Source scope: output/duplicates/address_alias_hits.csv audit data.",
        }

        html += "<h3>Core Reliability Metrics</h3>"
        html += "<table><tr><th>Metric</th><th>Description</th><th>Value</th></tr>"
        for key in (
            "scrape_critical_failures",
            "scrape_total_failures",
            "scrape_total_failure_reasons",
            "scrape_not_accessed_urls",
            "scrape_keyword_misses_after_access",
            "scraper_exception_rate",
            "scraper_retry_max_reached",
            "scraper_timeout_count",
            "scraper_network_degraded",
            "chatbot_average_score",
            "chatbot_execution_success_rate",
            "llm_success_rate",
            "llm_provider_exhausted",
            "llm_cost_pressure",
            "address_alias_conflict_skips",
        ):
            html += (
                "<tr>"
                f"<td>{self._escape_html(key)}</td>"
                f"<td>{self._escape_html(metric_descriptions.get(key, ''))}</td>"
                f"<td>{self._escape_html(metrics.get(key, ''))}</td>"
                "</tr>"
            )
        html += "</table>"

        gate_payload = gates or {}
        gate_status = str(gate_payload.get("status", "PASS")).upper()
        gate_status_class = {
            "PASS": "status-pass",
            "WARNING": "status-warning",
            "FAIL": "status-fail",
            "ERROR": "status-fail",
        }.get(gate_status, "status-warning")
        html += "<h3>Reliability Gates</h3>"
        html += (
            "<p><strong>Gate Status:</strong> "
            f"<span class=\"{gate_status_class}\">{self._escape_html(gate_status)}</span></p>"
        )
        gate_results = gate_payload.get("gate_results", []) or []
        if gate_results:
            html += "<table><tr><th>Gate</th><th>Status</th><th>Detail</th></tr>"
            for gate in gate_results:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(gate.get('name', ''))}</td>"
                    f"<td>{self._escape_html(gate.get('status', ''))}</td>"
                    f"<td>{self._escape_html(gate.get('detail', ''))}</td>"
                    "</tr>"
                )
            html += "</table>"

        if issues:
            html += "<h3>Top Reliability Issues</h3>"
            html += "<table><tr><th>ID</th><th>Category</th><th>Severity</th><th>Step</th><th>Provider</th><th>Occurrences</th><th>Actual</th></tr>"
            for issue in issues[:20]:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(issue.get('issue_id', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('category', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('severity', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('step', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('provider', ''))}</td>"
                    f"<td>{int(issue.get('occurrence_count', 1) or 1)}</td>"
                    f"<td>{self._escape_html(issue.get('actual', ''))}</td>"
                    "</tr>"
                )
            html += "</table>"
        else:
            html += "<p>✅ No normalized reliability issues in current window.</p>"

        trends_payload = trends or {}
        if trends_payload:
            html += "<h3>Trend Snapshot</h3>"
            html += "<table><tr><th>Window</th><th>Runs</th><th>Average Score</th><th>Latest Score</th><th>Delta</th></tr>"
            for window in ("7d", "30d"):
                t = trends_payload.get(window, {}) or {}
                html += (
                    "<tr>"
                    f"<td>{window}</td>"
                    f"<td>{int(t.get('runs', 0) or 0)}</td>"
                    f"<td>{float(t.get('average_score', 0) or 0):.1f}</td>"
                    f"<td>{float(t.get('latest_score', 0) or 0):.1f}</td>"
                    f"<td>{float(t.get('delta_latest_vs_avg', 0) or 0):+.1f}</td>"
                    "</tr>"
                )
            html += "</table>"

        reg = registry_summary or {}
        if reg:
            html += (
                f"<p><strong>Issue Registry:</strong> total_tracked={int(reg.get('total_tracked', 0) or 0)}, "
                f"recurring_in_current_run={int(reg.get('recurring_in_current_run', 0) or 0)}</p>"
            )

        return html

    def _evaluate_reliability_gates(self, scorecard: dict) -> dict:
        """Evaluate configurable reliability gates and return pass/fail details."""
        reporting_cfg = self.validation_config.get('reporting', {})
        gates_cfg = reporting_cfg.get('reliability_gates', {}) if isinstance(reporting_cfg, dict) else {}
        enabled = bool(gates_cfg.get('enabled', True))
        thresholds = gates_cfg.get('thresholds', {}) if isinstance(gates_cfg, dict) else {}
        metrics = (scorecard or {}).get('metrics', {}) or {}
        url_metrics = (
            ((scorecard or {}).get("metric_families", {}) or {})
            .get("url_level_grading", {})
            .get("metrics", {})
            or metrics
        )

        defaults = {
            "min_reliability_score": 75,
            "min_chatbot_execution_success_rate": 0.90,
            "min_chatbot_average_score": 70,
            "max_scrape_critical_failures": 0,
        }
        merged_thresholds = {**defaults, **(thresholds or {})}

        if not enabled:
            return {
                "enabled": False,
                "status": "PASS",
                "thresholds": merged_thresholds,
                "gate_results": [],
                "failed_gates": [],
                "evaluated_at": datetime.now().isoformat(),
            }

        gate_results: list[dict] = []

        def check_min(
            name: str,
            actual: float,
            minimum: float,
            numerator: str | None = None,
            denominator: str | None = None,
            family: str = "url_level_grading",
        ) -> None:
            ok = actual >= minimum
            gate_results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": (
                    f"actual={actual} min_required={minimum}"
                    + (f" [n={numerator} d={denominator}]" if numerator and denominator else "")
                ),
                "family": family,
                "numerator": numerator,
                "denominator": denominator,
            })

        def check_max(
            name: str,
            actual: float,
            maximum: float,
            numerator: str | None = None,
            denominator: str | None = None,
            family: str = "url_level_grading",
        ) -> None:
            ok = actual <= maximum
            gate_results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": (
                    f"actual={actual} max_allowed={maximum}"
                    + (f" [n={numerator} d={denominator}]" if numerator and denominator else "")
                ),
                "family": family,
                "numerator": numerator,
                "denominator": denominator,
            })

        check_min(
            "min_reliability_score",
            float((scorecard or {}).get("url_level_score", (scorecard or {}).get("score", 0)) or 0),
            float(merged_thresholds["min_reliability_score"]),
            numerator="url_level_score",
            denominator="100",
        )
        check_min(
            "min_chatbot_execution_success_rate",
            float(url_metrics.get("chatbot_execution_success_rate", 0) or 0),
            float(merged_thresholds["min_chatbot_execution_success_rate"]),
            numerator=str(int(url_metrics.get("chatbot_execution_success_count", 0) or 0)),
            denominator=str(max(1, int(url_metrics.get("chatbot_total_tests", 0) or 0))),
        )
        check_min(
            "min_chatbot_average_score",
            float(url_metrics.get("chatbot_average_score", 0) or 0),
            float(merged_thresholds["min_chatbot_average_score"]),
            numerator="chatbot_average_score",
            denominator="100",
        )
        check_max(
            "max_scrape_critical_failures",
            float(url_metrics.get("scrape_critical_failures", 0) or 0),
            float(merged_thresholds["max_scrape_critical_failures"]),
            numerator=str(int(url_metrics.get("scrape_critical_failures", 0) or 0)),
            denominator=str(max(1, int(url_metrics.get("scrape_total_important_urls", 0) or 0))),
        )

        failed_gates = [g for g in gate_results if g.get("status") == "FAIL"]
        status = "FAIL" if failed_gates else "PASS"
        return {
            "enabled": True,
            "status": status,
            "thresholds": merged_thresholds,
            "gate_results": gate_results,
            "failed_gates": failed_gates,
            "evaluated_at": datetime.now().isoformat(),
        }

    def _build_optimization_plan(self, results: dict, scorecard: dict, llm_data: dict) -> dict:
        """
        Phase 4: Build continuous optimization recommendations from observed outcomes.
        """
        providers = (llm_data or {}).get("providers", {}) or {}
        default_order = list(self.config.get("llm", {}).get("chatbot_provider_order", ["openai", "openrouter", "gemini"]))
        base_order_index = {provider: idx for idx, provider in enumerate(default_order)}
        provider_scores: dict[str, dict] = {}
        ranked: list[tuple[str, float]] = []
        recommendations: list[str] = []
        change_targets: list[dict] = []

        for provider in ("openai", "openrouter", "mistral", "gemini"):
            pdata = providers.get(provider, {}) or {}
            attempts = int(pdata.get("attempts", 0) or 0)
            successes = int(pdata.get("successes", 0) or 0)
            failures = int(pdata.get("failures", 0) or 0)
            rate_limits = int(pdata.get("rate_limits", 0) or 0)
            timeouts = int(pdata.get("timeouts", 0) or 0)
            success_rate = (successes / attempts) if attempts else 0.0
            failure_rate = (failures / attempts) if attempts else 0.0
            rate_limit_rate = (rate_limits / attempts) if attempts else 0.0
            timeout_rate = (timeouts / attempts) if attempts else 0.0
            unstable_ratio = failure_rate + rate_limit_rate + timeout_rate
            # Ratio-first ranking: prioritize low failure/rate-limit/timeout providers while still rewarding success.
            score = (success_rate * 100.0) - (failure_rate * 55.0) - (rate_limit_rate * 90.0) - (timeout_rate * 65.0)
            if attempts < 5:
                score -= 5.0
            score += max(0.0, 5.0 - float(base_order_index.get(provider, 99)))
            provider_scores[provider] = {
                "attempts": attempts,
                "success_rate": round(success_rate, 4),
                "failure_rate": round(failure_rate, 4),
                "rate_limit_rate": round(rate_limit_rate, 4),
                "timeout_rate": round(timeout_rate, 4),
                "unstable_ratio": round(unstable_ratio, 4),
                "health_score": round(score, 2),
            }
            ranked.append((provider, score))
            if attempts > 0 and rate_limit_rate >= 0.20:
                recommendations.append(
                    f"{provider}: high rate-limit ratio ({rate_limits}/{attempts}); reduce priority or increase cooldown."
                )
                change_targets.append({
                    "source_section": "LLM Provider Activity",
                    "target_type": "config",
                    "target": "llm.chatbot_provider_order",
                    "metric_key": f"llm_{provider}_rate_limit_rate",
                    "acceptance_test": f"{provider} rate_limit_rate < 0.20 in next report window",
                    "reason": f"rate_limit_ratio={rate_limits}/{attempts}",
                })
            if attempts > 0 and timeout_rate >= 0.15:
                recommendations.append(
                    f"{provider}: elevated timeout ratio ({timeouts}/{attempts}); keep as fallback until stable."
                )
                change_targets.append({
                    "source_section": "LLM Provider Activity",
                    "target_type": "function",
                    "target": "test_runner._build_optimization_plan",
                    "metric_key": f"llm_{provider}_timeout_rate",
                    "acceptance_test": f"{provider} timeout_rate < 0.15 in next report window",
                    "reason": f"timeout_ratio={timeouts}/{attempts}",
                })

        ranked.sort(key=lambda x: x[1], reverse=True)
        optimized_order = [p for p, _ in ranked]

        current_metrics = (scorecard or {}).get("metrics", {}) or {}
        if float(current_metrics.get("llm_success_rate", 0) or 0) < 0.8:
            recommendations.append("Overall LLM success rate is below 80%; tighten routing and fallback depth for chatbot requests.")
            change_targets.append({
                "source_section": "LLM Provider Activity",
                "target_type": "metric_policy",
                "target": "reliability_scorecard.llm_success_rate",
                "metric_key": "llm_success_rate",
                "acceptance_test": "llm_success_rate >= 0.80",
                "reason": "global success-rate floor breached",
            })
        if str(current_metrics.get("llm_cost_pressure", "LOW")).upper() in {"MEDIUM", "HIGH"}:
            recommendations.append("Cost pressure is elevated; prioritize stable low-cost providers and avoid deep fallback chains.")
            change_targets.append({
                "source_section": "LLM Provider Activity",
                "target_type": "config",
                "target": "testing.validation.reporting.llm_activity_thresholds",
                "metric_key": "llm_cost_pressure",
                "acceptance_test": "llm_cost_pressure in {LOW, MEDIUM} aligned with configured thresholds",
                "reason": f"cost_pressure={current_metrics.get('llm_cost_pressure', 'UNKNOWN')}",
            })
        if not recommendations:
            recommendations.append("Routing health is stable. Keep current order and continue monitoring 7-day trend deltas.")
            change_targets.append({
                "source_section": "Optimization Recommendations",
                "target_type": "monitoring",
                "target": "reliability_history.7d",
                "metric_key": "llm_provider_routing_stability",
                "acceptance_test": "No sustained degradation in 7-day trend snapshot",
                "reason": "no immediate routing changes required",
            })

        return {
            "generated_at": datetime.now().isoformat(),
            "current_chatbot_provider_order": default_order,
            "recommended_chatbot_provider_order": optimized_order,
            "provider_scores": provider_scores,
            "quality_snapshot": {
                "chatbot_average_score": current_metrics.get("chatbot_average_score", 0),
                "chatbot_execution_success_rate": current_metrics.get("chatbot_execution_success_rate", 0),
                "reliability_score": (scorecard or {}).get("score", 0),
            },
            "recommendations": recommendations,
            "change_targets": change_targets,
            "config_patch_preview": {
                "llm": {
                    "chatbot_provider_order": optimized_order
                }
            },
        }

    def _build_optimization_html(self, optimization_plan: dict) -> str:
        """Render optimization recommendations section."""
        if not optimization_plan:
            return "<p class='error-box'>❌ Optimization recommendations unavailable</p>"

        current_order = optimization_plan.get("current_chatbot_provider_order", [])
        recommended_order = optimization_plan.get("recommended_chatbot_provider_order", [])
        html = (
            f"<p><strong>Current chatbot provider order:</strong> {self._escape_html(' -> '.join(current_order))}</p>"
            f"<p><strong>Recommended order:</strong> {self._escape_html(' -> '.join(recommended_order))}</p>"
        )
        html += "<h3>Provider Health Scores</h3>"
        html += "<table><tr><th>Provider</th><th>Attempts</th><th>Success Rate</th><th>Rate Limit Rate</th><th>Timeout Rate</th><th>Health Score</th></tr>"
        provider_scores = optimization_plan.get("provider_scores", {}) or {}
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            stats = provider_scores.get(provider, {}) or {}
            html += (
                "<tr>"
                f"<td>{self._escape_html(provider)}</td>"
                f"<td>{int(stats.get('attempts', 0) or 0)}</td>"
                f"<td>{float(stats.get('success_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('rate_limit_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('timeout_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('health_score', 0) or 0):.2f}</td>"
                "</tr>"
            )
        html += "</table>"

        recs = optimization_plan.get("recommendations", []) or []
        if recs:
            html += "<h3>Recommended Actions</h3><ul>"
            for rec in recs:
                html += f"<li>{self._escape_html(rec)}</li>"
            html += "</ul>"
        change_targets = optimization_plan.get("change_targets", []) or []
        if change_targets:
            html += "<h3>Machine-Readable Change Targets</h3>"
            html += "<table><tr><th>Source Section</th><th>Target Type</th><th>Target</th><th>Metric Key</th><th>Acceptance Test</th><th>Reason</th></tr>"
            for target in change_targets:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(str(target.get('source_section', '')))}</td>"
                    f"<td>{self._escape_html(str(target.get('target_type', '')))}</td>"
                    f"<td>{self._escape_html(str(target.get('target', '')))}</td>"
                    f"<td>{self._escape_html(str(target.get('metric_key', '')))}</td>"
                    f"<td>{self._escape_html(str(target.get('acceptance_test', '')))}</td>"
                    f"<td>{self._escape_html(str(target.get('reason', '')))}</td>"
                    "</tr>"
                )
            html += "</table>"

        html += (
            "<p><strong>Patch Preview:</strong> "
            "<code>llm.chatbot_provider_order</code> "
            f"→ {self._escape_html(str(recommended_order))}</p>"
        )
        return html

    def _build_action_queue_html(self, action_queue: dict) -> str:
        """Render prioritized reliability action queue."""
        if not action_queue:
            return "<p class='error-box'>❌ Action queue unavailable</p>"
        items = action_queue.get("items", []) or []
        if not items:
            return "<p>✅ No queued reliability actions.</p>"
        html = "<table><tr><th>Priority</th><th>Status</th><th>Owner</th><th>Source Section</th><th>Metric Key</th><th>Action</th><th>Reason</th><th>Suggested Change</th><th>Acceptance Test</th></tr>"
        for item in items:
            html += (
                "<tr>"
                f"<td>{self._escape_html(item.get('priority', ''))}</td>"
                f"<td>{self._escape_html(item.get('status', 'open'))}</td>"
                f"<td>{self._escape_html(item.get('owner', 'unassigned'))}</td>"
                f"<td>{self._escape_html(item.get('source_section', ''))}</td>"
                f"<td>{self._escape_html(item.get('metric_key', ''))}</td>"
                f"<td>{self._escape_html(item.get('title', ''))}</td>"
                f"<td>{self._escape_html(item.get('reason', ''))}</td>"
                f"<td>{self._escape_html(item.get('suggested_change', ''))}</td>"
                f"<td>{self._escape_html(item.get('acceptance_test', ''))}</td>"
                "</tr>"
            )
        html += "</table>"
        return html

    def _update_and_summarize_reliability_history(self, output_dir: str, scorecard: dict) -> dict:
        """Append current scorecard and return 7d/30d trend summary."""
        os.makedirs(output_dir, exist_ok=True)
        history_path = codex_review_path("reliability_history.jsonl")
        record = {
            "timestamp": scorecard.get("timestamp", datetime.now().isoformat()),
            "score": float(scorecard.get("score", 0) or 0),
            "status": scorecard.get("status", "WATCH"),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        rows: list[dict] = []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            rows = [record]

        now = datetime.now()
        summary: dict[str, dict] = {}
        for days, label in ((7, "7d"), (30, "30d")):
            cutoff = now - timedelta(days=days)
            scoped = []
            for row in rows:
                ts = str(row.get("timestamp", "") or "")
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                if dt >= cutoff:
                    scoped.append(float(row.get("score", 0) or 0))
            avg_score = (sum(scoped) / len(scoped)) if scoped else 0.0
            latest_score = float(record.get("score", 0) or 0)
            summary[label] = {
                "runs": len(scoped),
                "average_score": round(avg_score, 2),
                "latest_score": round(latest_score, 2),
                "delta_latest_vs_avg": round(latest_score - avg_score, 2) if scoped else 0.0,
            }
        summary["path"] = history_path
        return summary

    def _update_reliability_issue_registry(self, output_dir: str, issues: list[dict]) -> tuple[list[dict], dict]:
        """Upsert issues into persistent registry and annotate current issues with occurrence counts."""
        os.makedirs(output_dir, exist_ok=True)
        registry_path = codex_review_path("reliability_issue_registry.json")
        payload = {"issues": {}, "updated_at": datetime.now().isoformat()}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and isinstance(loaded.get("issues"), dict):
                        payload = loaded
            except Exception:
                pass

        registry = payload.get("issues", {})
        now_ts = datetime.now().isoformat()
        for issue in issues:
            key = f"{issue.get('issue_id', '')}|{issue.get('input_signature', '')}"
            entry = registry.get(key, {})
            first_seen = entry.get("first_seen", issue.get("first_seen", now_ts))
            occurrence = int(entry.get("occurrence_count", 0) or 0) + 1
            registry[key] = {
                "issue_id": issue.get("issue_id", ""),
                "category": issue.get("category", ""),
                "severity": issue.get("severity", ""),
                "step": issue.get("step", ""),
                "provider": issue.get("provider", ""),
                "input_signature": issue.get("input_signature", ""),
                "first_seen": first_seen,
                "last_seen": now_ts,
                "occurrence_count": occurrence,
                "latest_actual": issue.get("actual", ""),
            }
            issue["occurrence_count"] = occurrence

        payload["issues"] = registry
        payload["updated_at"] = now_ts
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        recurring_current = sum(1 for i in issues if int(i.get("occurrence_count", 1) or 1) > 1)
        top_recurring = sorted(
            registry.values(),
            key=lambda x: int(x.get("occurrence_count", 0) or 0),
            reverse=True
        )[:10]
        return issues, {
            "path": registry_path,
            "total_tracked": len(registry),
            "recurring_in_current_run": recurring_current,
            "top_recurring": top_recurring,
        }

    def _build_action_queue(
        self,
        gates: dict,
        optimization_plan: dict,
        issues: list[dict],
        registry_summary: dict,
    ) -> dict:
        """Build prioritized action queue from gates, recurring issues, and optimization signals."""
        default_owner = str(
            self.validation_config.get("reporting", {})
            .get("action_queue", {})
            .get("default_owner", "unassigned")
        )
        default_status = str(
            self.validation_config.get("reporting", {})
            .get("action_queue", {})
            .get("default_status", "open")
        )

        def _step_to_section(step: str) -> str:
            mapping = {
                "scraping_validation": "Scraping Validation",
                "chatbot_testing": "Chatbot Testing",
                "chatbot_performance": "Chatbot Performance",
                "llm_activity": "LLM Provider Activity",
                "llm_extraction_quality": "LLM Extraction Quality Scorecard",
                "scraper_network_health": "Scraper Network Reliability",
                "fb_block_health": "Facebook Block Health",
                "fb.py": "Facebook Block Health",
                "address_alias_audit": "Address Alias Audit",
                "reliability_gates": "Reliability Scorecard",
                "fb_ig_funnel": "FB/IG URL Funnel",
            }
            return mapping.get(step, "Reliability Scorecard")

        items: list[dict] = []
        for gate in (gates or {}).get("failed_gates", []) or []:
            gate_name = str(gate.get("name", "") or "")
            items.append({
                "priority": "P0",
                "source_section": "Reliability Scorecard",
                "metric_key": gate_name,
                "title": f"Fix gate failure: {gate_name}",
                "reason": gate.get("detail", "Gate threshold breached."),
                "suggested_change": "Adjust provider routing/threshold regressions and rerun validation.",
                "acceptance_test": f"Gate `{gate_name}` is PASS in reliability_gates.json",
                "status": default_status,
                "owner": default_owner,
            })

        for recurring in (registry_summary or {}).get("top_recurring", [])[:3]:
            occ = int(recurring.get("occurrence_count", 0) or 0)
            if occ < 2:
                continue
            step = str(recurring.get("step", "") or "")
            issue_id = str(recurring.get("issue_id", "") or "")
            items.append({
                "priority": "P1",
                "source_section": _step_to_section(step),
                "metric_key": issue_id,
                "title": f"Resolve recurring issue: {issue_id}",
                "reason": f"Recurring {occ} times since {recurring.get('first_seen', '')}.",
                "suggested_change": "Add deterministic regression test and tighten parser/fallback for this signature.",
                "acceptance_test": f"`{issue_id}` occurrence trend decreases in reliability_issue_registry.json",
                "status": default_status,
                "owner": default_owner,
            })

        change_targets = (optimization_plan or {}).get("change_targets", []) or []
        for target in change_targets[:3]:
            items.append({
                "priority": "P2",
                "source_section": str(target.get("source_section", "Optimization Recommendations")),
                "metric_key": str(target.get("metric_key", "optimization_target")),
                "title": "Apply optimization recommendation",
                "reason": str(target.get("reason", "")),
                "suggested_change": "Review config_patch_preview and apply routing change if aligned with quality goals.",
                "acceptance_test": str(target.get("acceptance_test", "Target metric improves in next run")),
                "status": default_status,
                "owner": default_owner,
            })
        if not change_targets:
            for rec in (optimization_plan or {}).get("recommendations", [])[:3]:
                items.append({
                    "priority": "P2",
                    "source_section": "Optimization Recommendations",
                    "metric_key": "optimization_recommendation",
                    "title": "Apply optimization recommendation",
                    "reason": rec,
                    "suggested_change": "Review config_patch_preview and apply routing change if aligned with quality goals.",
                    "acceptance_test": "Recommendation is implemented and validated in next run.",
                    "status": default_status,
                    "owner": default_owner,
                })

        # Deduplicate by title while preserving order
        dedup: dict[str, dict] = {}
        for item in items:
            title = str(item.get("title", "") or "")
            if title and title not in dedup:
                dedup[title] = item
        return {
            "generated_at": datetime.now().isoformat(),
            "issue_count": len(issues or []),
            "items": list(dedup.values())[:12],
        }

    def _build_scraper_network_html(self, scraper_network: dict, scraping_data: dict | None = None) -> str:
        """Build HTML for scraper network reliability health section."""
        if not scraper_network:
            return "<p class='error-box'>❌ Scraper network summary unavailable</p>"
        if scraper_network.get("error"):
            return (
                "<p class='error-box'>❌ Scraper network summary unavailable: "
                f"{self._escape_html(scraper_network.get('error', 'unknown'))}</p>"
            )

        degraded = bool(scraper_network.get("degraded", False))
        status = "DEGRADED" if degraded else "HEALTHY"
        status_class = "status-fail" if degraded else "status-pass"
        scraping_summary = (scraping_data or {}).get("summary", {}) if isinstance(scraping_data, dict) else {}
        attempted_denominator = int(scraping_summary.get("attempted_url_denominator", 0) or 0)
        post_scrape_failures = int(scraping_summary.get("post_scrape_failures", 0) or 0)
        attempted_failure_rate = scraping_summary.get("attempted_failure_rate")
        url_failure_rate_display = (
            f"{float(attempted_failure_rate):.2%}" if attempted_failure_rate is not None else "N/A"
        )

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{float(scraper_network.get('exception_rate', 0) or 0):.2%}</div>
                <div class="metric-label">Request Exception Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{url_failure_rate_display}</div>
                <div class="metric-label">URL Failure Rate (Attempted)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(scraper_network.get('timeout_count', 0) or 0)}</div>
                <div class="metric-label">Timeout Retries</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(scraper_network.get('retry_max_reached', 0) or 0)}</div>
                <div class="metric-label">Retry Max Reached</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(scraper_network.get('request_count', 0) or 0)}</div>
                <div class="metric-label">Requests</div>
            </div>
        </div>
        <p><strong>Status:</strong> <span class="{status_class}">{status}</span></p>
        <p><strong>Window:</strong> {self._escape_html(scraper_network.get('start_ts', ''))} to {self._escape_html(scraper_network.get('end_ts', ''))}</p>
        <p><strong>Finish Reason:</strong> {self._escape_html(scraper_network.get('finish_reason', ''))}</p>
        <p><strong>Grading Use:</strong> {'Informational only (excluded from URL-level grade)' if bool(scraper_network.get('informational_for_grade', False)) else 'Included in grade'}</p>
        <p><strong>Request Exception Formula:</strong> exception_count / request_count = {int(scraper_network.get('exception_count', 0) or 0)} / {int(scraper_network.get('request_count', 0) or 0)} = {float(scraper_network.get('exception_rate', 0) or 0):.2%}</p>
        """
        if attempted_denominator > 0:
            html += (
                "<p><strong>URL Failure Formula:</strong> "
                f"post_scrape_failures / attempted_urls = {post_scrape_failures} / {attempted_denominator} = {url_failure_rate_display}</p>"
            )
        else:
            html += "<p><strong>URL Failure Formula:</strong> N/A (attempted URL denominator unavailable)</p>"
        html += (
            "<h3>Counts and Rates</h3>"
            "<table><tr><th>Metric</th><th>Numerator</th><th>Denominator</th><th>Rate / Value</th></tr>"
            f"<tr><td>Request Exception Rate</td><td>{int(scraper_network.get('exception_count', 0) or 0)}</td><td>{int(scraper_network.get('request_count', 0) or 0)}</td><td>{float(scraper_network.get('exception_rate', 0) or 0):.2%}</td></tr>"
            f"<tr><td>Timeout Retries</td><td>{int(scraper_network.get('timeout_count', 0) or 0)}</td><td>{int(scraper_network.get('request_count', 0) or 0)}</td><td>{(int(scraper_network.get('timeout_count', 0) or 0) / max(1, int(scraper_network.get('request_count', 0) or 0))):.2%}</td></tr>"
            f"<tr><td>Retry Max Reached</td><td>{int(scraper_network.get('retry_max_reached', 0) or 0)}</td><td>{int(scraper_network.get('request_count', 0) or 0)}</td><td>{(int(scraper_network.get('retry_max_reached', 0) or 0) / max(1, int(scraper_network.get('request_count', 0) or 0))):.2%}</td></tr>"
            f"<tr><td>URL Failure Rate (Attempted)</td><td>{post_scrape_failures}</td><td>{attempted_denominator if attempted_denominator > 0 else 'N/A'}</td><td>{url_failure_rate_display}</td></tr>"
            "</table>"
        )

        if bool(scraper_network.get("used_fallback_window", False)):
            html += (
                "<p class='error-box'>"
                "<strong>Note:</strong> No scraper_log activity was found in the requested window "
                f"({self._escape_html(scraper_network.get('requested_start_ts', ''))} to "
                f"{self._escape_html(scraper_network.get('requested_end_ts', ''))}). "
                "Displayed scraper network metrics are from the latest available scraper_log segment."
                "</p>"
            )

        top_domains = scraper_network.get("top_failure_domains") or []
        if top_domains:
            html += "<h3>Top Failing Domains</h3><table><tr><th>Domain</th><th>Transient Failures</th></tr>"
            for domain, count in top_domains:
                html += f"<tr><td>{self._escape_html(domain)}</td><td>{int(count)}</td></tr>"
            html += "</table>"

        top_timeout_domains = scraper_network.get("top_timeout_domains") or []
        if top_timeout_domains:
            html += "<h3>Top Timeout Domains</h3><table><tr><th>Domain</th><th>Timeout Failures</th></tr>"
            for domain, count in top_timeout_domains:
                html += f"<tr><td>{self._escape_html(domain)}</td><td>{int(count)}</td></tr>"
            html += "</table>"

        top_exception_domains = scraper_network.get("top_exception_domains") or []
        if top_exception_domains:
            html += "<h3>Top Exception Domains</h3><table><tr><th>Domain</th><th>Non-Timeout Exceptions</th></tr>"
            for domain, count in top_exception_domains:
                html += f"<tr><td>{self._escape_html(domain)}</td><td>{int(count)}</td></tr>"
            html += "</table>"

        thresholds = scraper_network.get("thresholds") or {}
        if thresholds:
            html += "<h3>Thresholds</h3><table><tr><th>Metric</th><th>Threshold</th></tr>"
            html += f"<tr><td>max_request_exception_rate</td><td>{float(thresholds.get('max_exception_rate', 0) or 0):.2%}</td></tr>"
            html += f"<tr><td>max_retry_max_reached</td><td>{int(thresholds.get('max_retry_max_reached', 0) or 0)}</td></tr>"
            html += f"<tr><td>max_timeout_count</td><td>{int(thresholds.get('max_timeout_count', 0) or 0)}</td></tr>"
            html += "</table>"

        return html

    def _build_scraping_html(self, scraping_data: dict) -> str:
        """Build HTML for scraping validation section."""
        if not scraping_data:
            return "<p class='error-box'>❌ Scraping validation did not run</p>"

        if 'error' in scraping_data:
            return f"<p class='error-box'>❌ Scraping validation failed: {scraping_data['error']}</p>"

        summary = scraping_data['summary']
        not_attempted_breakdown = summary.get('not_attempted_reason_breakdown', {}) if isinstance(summary, dict) else {}
        failure_stage_counts = summary.get('failure_stage_counts', {}) if isinstance(summary, dict) else {}
        attempted_denominator = int(summary.get("attempted_url_denominator", 0) or 0)
        attempted_failure_rate = summary.get("attempted_failure_rate")
        keyword_failures = int(summary.get("keyword_failures_after_scrape", 0) or 0)
        total_important_urls = int(summary.get("total_important_urls", 0) or 0)

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{summary['total_failures']}</div>
                <div class="metric-label">Post-Scrape Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{attempted_denominator}</div>
                <div class="metric-label">Attempted URL Denominator</div>
            </div>
            <div class="metric">
                <div class="metric-value">{keyword_failures}</div>
                <div class="metric-label">Keyword Failures (After Scrape)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['whitelist_failures']}</div>
                <div class="metric-label">Whitelist Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['edge_case_failures']}</div>
                <div class="metric-label">Edge Case Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['high_performer_failures']}</div>
                <div class="metric-label">High-Performer Failures</div>
            </div>
        </div>

        """
        if total_important_urls > 0:
            rate_display = f"{float(attempted_failure_rate):.1%}" if attempted_failure_rate is not None else "N/A"
            html += (
                "<p><strong>Attempted Failure Rate:</strong> "
                f"{rate_display}"
            )
            html += f" ({int(summary.get('total_failures', 0) or 0)}/{attempted_denominator}, total important URLs={total_important_urls})</p>"

        raw_total = int(summary.get("total_failures_raw", summary.get("total_failures", 0)) or 0)
        excluded_pre_scrape = int(summary.get("pre_scrape_skipped_failures_excluded", 0) or 0)
        post_scrape_failures = int(summary.get("post_scrape_failures", summary.get("total_failures", 0)) or 0)
        if excluded_pre_scrape > 0 or raw_total != post_scrape_failures:
            html += (
                "<p><strong>Total Failures Counting:</strong> "
                f"post_scrape_counted={post_scrape_failures}, "
                f"raw={raw_total}, "
                f"excluded_pre_scrape_should_process_skips={excluded_pre_scrape}, "
                f"excluded_not_attempted={int((summary.get('failure_types', {}) or {}).get('not_attempted', 0) or 0)}"
                "</p>"
            )
        attempted_no_keywords = int(failure_stage_counts.get('attempted_no_keywords', 0) or 0)
        attempted_extraction_or_llm_failure = int(
            failure_stage_counts.get('attempted_extraction_or_llm_failure', 0) or 0
        )
        other_post_scrape = max(
            0,
            post_scrape_failures - attempted_no_keywords - attempted_extraction_or_llm_failure,
        )
        html += (
            "<p><strong>Post-Scrape Failure Reasons:</strong> "
            f"attempted_no_keywords={attempted_no_keywords}; "
            f"attempted_extraction_or_llm_failure={attempted_extraction_or_llm_failure}; "
            f"other_post_scrape={other_post_scrape}"
            "</p>"
        )

        if not_attempted_breakdown:
            categories = not_attempted_breakdown.get("categories", {}) or {}
            run_limit_ctx = not_attempted_breakdown.get("run_limit_whitelist_context", {}) or {}
            pending_scraper_owned = int(run_limit_ctx.get("pending_scraper_owned_roots_max", 0) or 0)
            fb_owned_roots = int(run_limit_ctx.get("fb_owned_roots_max", 0) or 0)
            non_text_roots = int(run_limit_ctx.get("non_text_roots_max", 0) or 0)
            html += (
                "<h3>Not-Attempted Reason Breakdown</h3>"
                "<p><strong>Total Not Attempted:</strong> "
                f"{int(not_attempted_breakdown.get('total_not_attempted', 0) or 0)} | "
                "<strong>Run Ended By URL Limit:</strong> "
                f"{'Yes' if bool(not_attempted_breakdown.get('global_url_run_limit_reached', False)) else 'No'}</p>"
                "<p><strong>Run-Limit Whitelist Context (max seen in logs):</strong> "
                f"pending_scraper_owned_roots={pending_scraper_owned}; "
                f"fb_owned_roots={fb_owned_roots}; "
                f"non_text_roots={non_text_roots}</p>"
                "<table><tr><th>Reason</th><th>Affected URLs</th></tr>"
                "<tr><td>Explicit URL run-limit skip</td>"
                f"<td>{int(categories.get('explicit_url_run_limit_skip', 0) or 0)}</td></tr>"
                "<tr><td>Explicit should_process_url skip</td>"
                f"<td>{int(categories.get('explicit_should_process_url_skip', 0) or 0)}</td></tr>"
                "<tr><td>Unattributed with global URL limit reached</td>"
                f"<td>{int(categories.get('unattributed_with_global_run_limit', 0) or 0)}</td></tr>"
                "<tr><td>Other / unknown</td>"
                f"<td>{int(categories.get('other_or_unknown', 0) or 0)}</td></tr>"
                "</table>"
            )

        html += (
            "<h3>Failure Stage Breakdown</h3>"
            "<table><tr><th>Failure Stage</th><th>Affected URLs</th></tr>"
            "<tr><td>Not attempted - run limit skipped</td>"
            f"<td>{int(failure_stage_counts.get('run_limit_skipped', 0) or 0)}</td></tr>"
            "<tr><td>Not attempted - should_process_url skipped</td>"
            f"<td>{int(failure_stage_counts.get('should_process_url_skipped', 0) or 0)}</td></tr>"
            "<tr><td>Attempted - no keywords / marked irrelevant</td>"
            f"<td>{int(failure_stage_counts.get('attempted_no_keywords', 0) or 0)}</td></tr>"
            "<tr><td>Attempted - extraction/LLM failure</td>"
            f"<td>{int(failure_stage_counts.get('attempted_extraction_or_llm_failure', 0) or 0)}</td></tr>"
            "<tr><td>Other / unknown</td>"
            f"<td>{int(failure_stage_counts.get('other_or_unknown', 0) or 0)}</td></tr>"
            "</table>"
            "<p><strong>Stage Formula Notes:</strong> "
            "run_limit_skipped = explicit run-limit skips + unattributed not-attempted URLs when a global run-limit was hit; "
            "should_process_url_skipped = explicit `should_process_url` skips; "
            "attempted_no_keywords = attempted URLs marked_irrelevant/no-keyword after access; "
            "attempted_extraction_or_llm_failure = attempted URLs with retry/extraction/LLM failure signals."
            "</p>"
        )

        html += f"""
        <h3>Critical Failures</h3>
        """

        if scraping_data.get('critical_failures'):
            html += (
                "<table><tr>"
                "<th>URL</th><th>Source</th><th>Type</th><th>Failure Type</th>"
                "<th>Failure Stage</th>"
                "<th>Reason Code</th><th>Probable Cause</th>"
                "<th>Last Attempt</th><th>Child Successes</th><th>Evidence</th><th>Recommendation</th>"
                "</tr>"
            )
            for failure in scraping_data['critical_failures'][:20]:  # Limit to 20
                evidence_lines = failure.get('evidence') or []
                if evidence_lines:
                    evidence_html = "<br>".join(
                        line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        for line in evidence_lines
                    )
                else:
                    evidence_html = ""
                html += f"""
                <tr class="problematic">
                    <td>{failure['url'][:60]}...</td>
                    <td>{failure['source']}</td>
                    <td>{failure['importance']}</td>
                    <td>{failure['failure_type']}</td>
                    <td>{failure.get('failure_stage', '')}</td>
                    <td>{failure.get('reason_code', '')}</td>
                    <td>{failure.get('probable_cause', '')}</td>
                    <td>{failure.get('last_attempt_time', '')}</td>
                    <td>{failure.get('recent_relevant_child_count', '')}</td>
                    <td style="font-size:0.85em;">{evidence_html}</td>
                    <td>{failure['recommendation']}</td>
                </tr>
                """
            html += "</table>"
        else:
            html += "<p>✅ No critical failures</p>"

        # Add source distribution check
        if 'source_distribution' in scraping_data:
            dist = scraping_data['source_distribution']
            html += "<h3>Source Distribution Check</h3>"

            # Status and metrics
            status_class = dist['status'].lower()
            html += f"<p><strong>Status:</strong> <span class='status-{status_class}'>{dist['status']}</span></p>"

            if 'total_events' in dist:
                html += f"""
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{dist['total_events']}</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('top_10_total', 0)}</div>
                        <div class="metric-label">Top 10 Sources</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('top_10_percentage', 0)}%</div>
                        <div class="metric-label">Top 10 Percentage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('total_sources', 0)}</div>
                        <div class="metric-label">Total Sources</div>
                    </div>
                </div>
                """

                # Missing sources (CRITICAL)
                if dist.get('missing_sources'):
                    html += "<h4>❌ Missing Required Sources</h4>"
                    html += "<p class='error-box'>The following required sources are NOT present in the database. Scraping likely failed for these sources:</p>"
                    html += "<ul>"
                    for missing_source in dist['missing_sources']:
                        html += f"<li class='error-box'><strong>{missing_source}</strong></li>"
                    html += "</ul>"

                # Top 10 sources table
                if 'top_10_sources' in dist:
                    html += "<h4>Top 10 Event Sources</h4>"
                    html += "<table><tr><th>Rank</th><th>Source</th><th>Event Count</th></tr>"
                    for i, source_info in enumerate(dist['top_10_sources'], 1):
                        html += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{source_info['source']}</td>
                            <td>{source_info['count']}</td>
                        </tr>
                        """
                    html += "</table>"

                trend = dist.get('trend_monitoring', {}) if isinstance(dist, dict) else {}
                if trend:
                    alerts = trend.get('alerts', []) or []
                    html += "<h4>Top-Source Trend Monitoring</h4>"
                    html += (
                        f"<p><strong>Run ID:</strong> {self._escape_html(str(trend.get('run_id', '')))} | "
                        f"<strong>History Runs Used:</strong> {int(trend.get('history_runs_used', 0) or 0)} | "
                        f"<strong>Lookback Runs:</strong> {int(trend.get('lookback_runs', 0) or 0)}</p>"
                    )

                    baseline_top = trend.get('baseline_top_10', []) or []
                    if baseline_top:
                        html += "<p><strong>Baseline Top Sources (Avg Count):</strong></p>"
                        html += "<table><tr><th>Rank</th><th>Source</th><th>Baseline Avg Count</th></tr>"
                        for i, row in enumerate(baseline_top[:10], 1):
                            html += (
                                "<tr>"
                                f"<td>{i}</td>"
                                f"<td>{self._escape_html(str(row.get('source', '')))}</td>"
                                f"<td>{float(row.get('avg_count', 0) or 0):.2f}</td>"
                                "</tr>"
                            )
                        html += "</table>"

                    if alerts:
                        html += "<p class='error-box'><strong>Trend Alerts:</strong> Significant source changes detected.</p>"
                        html += (
                            "<table><tr>"
                            "<th>Source</th><th>Alert Type</th><th>Severity</th>"
                            "<th>Current Count</th><th>Baseline Avg</th><th>% Change</th>"
                            "</tr>"
                        )
                        for alert in alerts:
                            html += (
                                "<tr class='problematic'>"
                                f"<td>{self._escape_html(str(alert.get('source', '')))}</td>"
                                f"<td>{self._escape_html(str(alert.get('alert_type', '')))}</td>"
                                f"<td>{self._escape_html(str(alert.get('severity', '')))}</td>"
                                f"<td>{int(alert.get('current_count', 0) or 0)}</td>"
                                f"<td>{float(alert.get('baseline_avg', 0) or 0):.2f}</td>"
                                f"<td>{float(alert.get('pct_change', 0) or 0):.1%}</td>"
                                "</tr>"
                            )
                        html += "</table>"
                    elif int(trend.get('history_runs_used', 0) or 0) > 0:
                        html += "<p>✅ No top-source trend anomalies detected in this run.</p>"
                    else:
                        html += "<p>ℹ️ Baseline is still warming up (insufficient prior runs).</p>"

            # Warnings
            if dist.get('warnings'):
                html += "<h4>Warnings</h4><ul>"
                for warning in dist['warnings']:
                    html += f"<li class='error-box'>{warning}</li>"
                html += "</ul>"
            elif not dist.get('missing_sources'):
                html += "<p>✅ All required sources present in database</p>"

        return html

    def _build_chatbot_html(self, chatbot_data: dict) -> str:
        """Build HTML for chatbot testing section."""
        if not chatbot_data:
            return "<p class='error-box'>❌ Chatbot testing did not run</p>"

        if 'error' in chatbot_data:
            return f"<p class='error-box'>❌ Chatbot testing failed: {chatbot_data['error']}</p>"

        summary = chatbot_data['summary']

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{summary['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['average_score']:.1f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['execution_success_rate']:.1%}</div>
                <div class="metric-label">Execution Success</div>
            </div>
        </div>

        <h3>Score Distribution</h3>
        <table>
            <tr><th>Score Range</th><th>Count</th></tr>
        """

        for range_name, count in summary['score_distribution'].items():
            html += f"<tr><td>{range_name}</td><td>{count}</td></tr>"

        html += "</table>"
        category_gate = chatbot_data.get("category_gate") if isinstance(chatbot_data, dict) else None
        if isinstance(category_gate, dict):
            gate_status = str(category_gate.get("status", "PASS") or "PASS").upper()
            gate_class = {
                "PASS": "status-pass",
                "WARNING": "status-warning",
                "FAIL": "status-fail",
            }.get(gate_status, "status-warning")
            html += (
                "<p><strong>Problem Category Gate:</strong> "
                f"<span class='{gate_class}'>{self._escape_html(gate_status)}</span> "
                f"({self._escape_html(str(category_gate.get('detail', '')))}"
                ")</p>"
            )

        if chatbot_data.get('problematic_questions'):
            html += "<h3>Problematic Questions (Score < 50)</h3>"
            html += "<table><tr><th>Question</th><th>Category</th><th>Score</th><th>Reasoning</th></tr>"
            for q in chatbot_data['problematic_questions'][:10]:  # Show top 10
                html += f"""
                <tr class="problematic">
                    <td>{q['question']}</td>
                    <td>{q['category']}</td>
                    <td>{q['score']}</td>
                    <td>{q['reasoning'][:100]}...</td>
                </tr>
                """
            html += "</table>"

        # New: Problem Categories (score < 90)
        if chatbot_data.get('problem_categories'):
            html += "<h3>Problem Categories (Score < 90)</h3>"
            for cat in chatbot_data['problem_categories']:
                html += f"<h4>{cat['name']} ({cat['count']} issues)</h4>"
                example_q = cat.get('example', {}).get('question', '')
                example_reason = cat.get('example', {}).get('reason', '')
                html += f"<p><strong>Example:</strong> {example_q}<br><em>{example_reason}</em></p>"
                example_sql = cat.get('example', {}).get('sql', '')
                if example_sql:
                    html += f"<pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:10px; border-radius:4px; border:1px solid #eee;\"><code>{example_sql}</code></pre>"
                if cat.get('questions'):
                    html += "<ul>"
                    for qtext in cat['questions']:
                        html += f"<li>{qtext}</li>"
                    html += "</ul>"
                if cat.get('recommendation'):
                    html += f"<p><strong>Recommendation:</strong> {cat['recommendation']}</p>"

        return html

    def _build_chatbot_performance_html(self, perf_data: dict) -> str:
        """Build HTML for chatbot performance timing section."""
        if not perf_data:
            return "<p class='error-box'>❌ Chatbot performance summary unavailable</p>"
        if perf_data.get("error"):
            return (
                "<p class='error-box'>❌ Chatbot performance summary unavailable: "
                f"{self._escape_html(str(perf_data.get('error', 'unknown')))}</p>"
            )

        status = str(perf_data.get("status", "UNKNOWN") or "UNKNOWN").upper()
        status_class = {
            "HEALTHY": "status-pass",
            "WATCH": "status-warning",
            "DEGRADED": "status-fail",
        }.get(status, "status-warning")

        query_count = int(perf_data.get("query_request_count", 0) or 0)
        confirm_count = int(perf_data.get("confirm_request_count", 0) or 0)
        query_lat = perf_data.get("query_latency_ms", {}) or {}
        confirm_lat = perf_data.get("confirm_latency_ms", {}) or {}
        unfinished = int(perf_data.get("unfinished_request_count", 0) or 0)

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{query_count}</div>
                <div class="metric-label">/query Requests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{confirm_count}</div>
                <div class="metric-label">/confirm Requests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{float(query_lat.get('avg', 0.0) or 0.0):.1f} ms</div>
                <div class="metric-label">Query Avg Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value">{float(query_lat.get('p95', 0.0) or 0.0):.1f} ms</div>
                <div class="metric-label">Query P95 Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value">{float(query_lat.get('max', 0.0) or 0.0):.1f} ms</div>
                <div class="metric-label">Query Max Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unfinished}</div>
                <div class="metric-label">Unfinished Requests</div>
            </div>
        </div>
        """
        html += f"<p><strong>Status:</strong> <span class='{status_class}'>{self._escape_html(status)}</span></p>"
        html += (
            "<p><strong>Window:</strong> "
            f"{self._escape_html(str(perf_data.get('start_ts', '')))} to "
            f"{self._escape_html(str(perf_data.get('end_ts', '')))} "
            f"(last {int(perf_data.get('window_hours', 24) or 24)} hour(s))</p>"
        )
        html += (
            "<p><strong>Data Source:</strong> "
            f"{self._escape_html(str(perf_data.get('source', 'logs')))}</p>"
        )
        html += (
            "<p><strong>Latency Formula:</strong> "
            f"query_p95_ms computed over {query_count} request_total samples for /query endpoint.</p>"
        )
        reasons = perf_data.get("status_reasons", []) or []
        if reasons:
            html += "<ul>"
            for reason in reasons:
                html += f"<li>{self._escape_html(str(reason))}</li>"
            html += "</ul>"

        html += "<h3>Endpoint Latency Breakdown</h3>"
        html += (
            "<table><tr><th>Endpoint</th><th>Count</th><th>Avg (ms)</th><th>P50 (ms)</th><th>P95 (ms)</th><th>Max (ms)</th></tr>"
            f"<tr><td>/query</td><td>{query_count}</td><td>{float(query_lat.get('avg', 0.0) or 0.0):.1f}</td><td>{float(query_lat.get('p50', 0.0) or 0.0):.1f}</td><td>{float(query_lat.get('p95', 0.0) or 0.0):.1f}</td><td>{float(query_lat.get('max', 0.0) or 0.0):.1f}</td></tr>"
            f"<tr><td>/confirm</td><td>{confirm_count}</td><td>{float(confirm_lat.get('avg', 0.0) or 0.0):.1f}</td><td>{float(confirm_lat.get('p50', 0.0) or 0.0):.1f}</td><td>{float(confirm_lat.get('p95', 0.0) or 0.0):.1f}</td><td>{float(confirm_lat.get('max', 0.0) or 0.0):.1f}</td></tr>"
            "</table>"
        )

        stage_rows = perf_data.get("stage_latency_summary", []) or []
        if stage_rows:
            html += "<h3>Stage Latency (Top by P95)</h3>"
            html += "<table><tr><th>Stage</th><th>Count</th><th>Avg (ms)</th><th>P95 (ms)</th><th>Max (ms)</th></tr>"
            for row in stage_rows:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(str(row.get('stage', '')))}</td>"
                    f"<td>{int(row.get('count', 0) or 0)}</td>"
                    f"<td>{float(row.get('avg_ms', 0.0) or 0.0):.1f}</td>"
                    f"<td>{float(row.get('p95_ms', 0.0) or 0.0):.1f}</td>"
                    f"<td>{float(row.get('max_ms', 0.0) or 0.0):.1f}</td>"
                    "</tr>"
                )
            html += "</table>"

        unfinished_ids = perf_data.get("unfinished_request_ids", []) or []
        if unfinished_ids:
            html += "<h3>Unfinished Request IDs (Sample)</h3><ul>"
            for req_id in unfinished_ids[:20]:
                html += f"<li>{self._escape_html(str(req_id))}</li>"
            html += "</ul>"

        slow_rows = perf_data.get("slow_requests", []) or []
        if slow_rows:
            html += "<h3>Slow Requests</h3>"
            html += (
                f"<p><strong>Slow Threshold:</strong> {float(perf_data.get('slow_request_threshold_ms', 0.0) or 0.0):.0f} ms</p>"
                "<table><tr><th>Request ID</th><th>Endpoint</th><th>Duration (ms)</th><th>Result</th><th>Question</th><th>SQL Snippet</th></tr>"
            )
            for row in slow_rows:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(str(row.get('request_id', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('endpoint', '')))}</td>"
                    f"<td>{float(row.get('duration_ms', 0.0) or 0.0):.1f}</td>"
                    f"<td>{self._escape_html(str(row.get('result_type', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('question', '')))}</td>"
                    f"<td>{self._escape_html(str(row.get('sql', '')))}</td>"
                    "</tr>"
                )
            html += "</table>"
        else:
            html += "<p>No slow requests exceeded threshold in this window.</p>"

        coverage = perf_data.get("slow_request_trace_coverage", {}) or {}
        if coverage:
            html += (
                "<p><strong>Slow Request Trace Coverage:</strong> "
                f"questions={int(coverage.get('question_count', 0) or 0)}, "
                f"sql={int(coverage.get('sql_count', 0) or 0)}, "
                f"slow_requests={int(coverage.get('slow_request_count', 0) or 0)}</p>"
            )
        return html

    def _build_chatbot_metrics_sync_html(self, sync_data: dict) -> str:
        """Build HTML for chatbot metrics sync visibility section."""
        if not sync_data:
            return "<p class='error-box'>❌ Chatbot metrics sync summary unavailable</p>"
        if sync_data.get("error"):
            return (
                "<p class='error-box'>❌ Chatbot metrics sync summary unavailable: "
                f"{self._escape_html(str(sync_data.get('error', 'unknown')))}</p>"
            )

        status = str(sync_data.get("status", "UNKNOWN") or "UNKNOWN").upper()
        status_class = {
            "SYNCED": "status-pass",
            "STALE": "status-warning",
            "MISSING": "status-fail",
        }.get(status, "status-warning")

        req_window = int(sync_data.get("request_count_window", 0) or 0)
        stg_window = int(sync_data.get("stage_count_window", 0) or 0)
        req_total = int(sync_data.get("request_count_total", 0) or 0)
        stg_total = int(sync_data.get("stage_count_total", 0) or 0)

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{req_window}</div>
                <div class="metric-label">Request Rows (90d)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stg_window}</div>
                <div class="metric-label">Stage Rows (90d)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{req_total}</div>
                <div class="metric-label">Request Rows (Total)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stg_total}</div>
                <div class="metric-label">Stage Rows (Total)</div>
            </div>
        </div>
        """
        html += (
            f"<p><strong>Status:</strong> <span class='{status_class}'>{self._escape_html(status)}</span></p>"
            "<p><strong>Window:</strong> "
            f"{self._escape_html(str(sync_data.get('start_ts', '')))} to "
            f"{self._escape_html(str(sync_data.get('end_ts', '')))} "
            f"(last {int(sync_data.get('window_days', 90) or 90)} day(s))</p>"
            "<p><strong>Latest Request started_at:</strong> "
            f"{self._escape_html(str(sync_data.get('latest_request_started_at', '')))}</p>"
            "<p><strong>Latest Stage started_at:</strong> "
            f"{self._escape_html(str(sync_data.get('latest_stage_started_at', '')))}</p>"
        )

        reasons = sync_data.get("status_reasons", []) or []
        if reasons:
            html += "<ul>"
            for reason in reasons:
                html += f"<li>{self._escape_html(str(reason))}</li>"
            html += "</ul>"

        return html

    def _send_email_notification(self, results: dict) -> None:
        """
        Send email notification with validation results.

        Args:
            results (dict): Combined validation results
        """
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')

        # Build summary for email body
        summary = {}

        # Add chatbot testing summary
        if results.get('chatbot_testing') and 'summary' in results['chatbot_testing']:
            chatbot_summary = results['chatbot_testing']['summary']
            summary['total_tests'] = chatbot_summary.get('total_tests', 0)
            summary['execution_success_rate'] = chatbot_summary.get('execution_success_rate', 0)
            summary['average_score'] = chatbot_summary.get('average_score', 0)
            summary['interpretation_pass_rate'] = chatbot_summary.get('interpretation_evaluation', {}).get('interpretation_pass_rate', 0)

        # Add scraping validation summary
        if results.get('scraping_validation') and 'summary' in results['scraping_validation']:
            scraping_summary = results['scraping_validation']['summary']
            summary['total_failures'] = scraping_summary.get('total_failures', 0)
            summary['whitelist_failures'] = scraping_summary.get('whitelist_failures', 0)

        # Add overall status
        summary['overall_status'] = results['overall_status']
        summary['timestamp'] = results['timestamp']

        # Collect attachment paths — include only the comprehensive HTML report.
        html_report = reports_path('comprehensive_test_report.html')
        attachments = [html_report] if os.path.exists(html_report) else []

        # Send email
        logging.info("Attempting to send email notification...")
        success = send_report_email(
            report_summary=summary,
            attachment_paths=attachments,
            test_type="Pre-Commit Validation"
        )

        if success:
            logging.info("✓ Email notification sent successfully")
        else:
            logging.info("Email notification skipped (not configured or failed)")


def main():
    """Main entry point for standalone execution."""
    # Setup logging
    setup_logging('validation_tests')

    try:
        # Run validations
        runner = ValidationTestRunner()
        results = runner.run_all_validations()

        # Exit with appropriate code
        if results['overall_status'] == 'FAIL':
            logging.error("Validation tests FAILED")
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            logging.warning("Validation tests completed with WARNINGS")
            sys.exit(0)  # Don't block pipeline
        else:
            logging.info("Validation tests PASSED")
            sys.exit(0)

    except Exception as e:
        logging.error(f"Fatal error in validation tests: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

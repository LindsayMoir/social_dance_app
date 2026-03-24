#!/usr/bin/env python3
"""Helpers for replay row comparison and recurring schedule matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any, Callable


@dataclass(frozen=True)
class ReplayMatcher:
    """Compare baseline and replay rows using injected normalization and adjudication helpers."""

    normalize_text_value: Callable[[Any], str]
    normalize_date_value: Callable[[Any], str]
    normalize_time_value: Callable[[Any], str]
    normalize_url_value: Callable[[Any], str]
    normalize_optional_int: Callable[[Any], int | None]
    name_similarity: Callable[[str, str], float]
    name_contains_variant: Callable[[str, str], bool]
    times_equivalent_with_12h_guard: Callable[[str, str], bool]
    is_placeholder_source: Callable[[str], bool]
    is_calendar_event_url: Callable[[str], bool]
    is_social_platform_url: Callable[[str], bool]
    llm_adjudicate_event_match: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str]]

    @staticmethod
    def parse_replay_date_string(value: str) -> date | None:
        """Parse normalized replay date strings safely."""
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except Exception:
            return None

    @staticmethod
    def has_recurrence_signal(*texts: Any) -> bool:
        """Return True when text suggests a recurring schedule rather than a one-off event."""
        combined = " ".join(str(text or "").strip().lower() for text in texts if str(text or "").strip())
        if not combined:
            return False
        recurrence_patterns = [
            r"\bevery\b",
            r"\bweekly\b",
            r"\bdaily\b",
            r"\beveryday\b",
            r"\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b|\bfriday\b|\bsaturday\b|\bsunday\b",
            r"\bmon(?:day)?\s*-\s*wed(?:nesday)?\b",
            r"\bthu(?:rsday)?\s*-\s*sun(?:day)?\b",
            r"\bmon(?:day)?\s*-\s*fri(?:day)?\b",
            r"\bevery\s+[a-z]+day\b",
        ]
        return any(re.search(pattern, combined, re.IGNORECASE) for pattern in recurrence_patterns)

    def is_recurring_schedule_match(self, baseline: dict[str, Any], replay: dict[str, Any]) -> bool:
        """Treat recurring social schedule posts as a match when weekly cadence aligns."""
        baseline_url = str(baseline.get("url") or "")
        replay_url = str(replay.get("url") or "")
        if not baseline_url or baseline_url != replay_url or not self.is_social_platform_url(baseline_url):
            return False
        baseline_name = str(baseline.get("event_name") or "")
        replay_name = str(replay.get("event_name") or "")
        if not baseline_name or self.normalize_text_value(baseline_name) != self.normalize_text_value(replay_name):
            return False
        if not self.times_equivalent_with_12h_guard(str(replay.get("start_time") or ""), str(baseline.get("start_time") or "")):
            return False

        baseline_date = self.parse_replay_date_string(str(baseline.get("start_date") or ""))
        replay_date = self.parse_replay_date_string(str(replay.get("start_date") or ""))
        if baseline_date is None or replay_date is None or baseline_date == replay_date:
            return False
        delta_days = abs((baseline_date - replay_date).days)
        if delta_days <= 0 or delta_days % 7 != 0 or delta_days > 35:
            return False
        if baseline_date.weekday() != replay_date.weekday():
            return False

        recurrence_signal = self.has_recurrence_signal(
            baseline.get("event_name"),
            baseline.get("description"),
            baseline.get("day_of_week"),
            replay.get("event_name"),
            replay.get("description"),
            replay.get("day_of_week"),
        )
        return recurrence_signal or self.normalize_text_value(baseline_name) in {"live music"}

    def compare_replay_row(self, baseline_row: dict[str, Any], replay_payload: dict[str, Any], strict_time_match: bool) -> dict[str, Any]:
        """Compare one baseline row against replayed candidates and classify mismatches."""
        baseline = {
            "event_name": self.normalize_text_value(baseline_row.get("event_name")),
            "start_date": self.normalize_date_value(baseline_row.get("start_date")),
            "start_time": self.normalize_time_value(baseline_row.get("start_time")),
            "day_of_week": self.normalize_text_value(baseline_row.get("day_of_week")),
            "source": self.normalize_text_value(baseline_row.get("source")),
            "location": self.normalize_text_value(baseline_row.get("location")),
            "url": self.normalize_url_value(baseline_row.get("url")),
            "address_id": self.normalize_optional_int(baseline_row.get("address_id")),
            "dance_style": self.normalize_text_value(baseline_row.get("dance_style")),
            "description": self.normalize_text_value(baseline_row.get("description")),
        }

        if not baseline["url"]:
            return {
                "is_match": False,
                "category": "missing_url_baseline",
                "details": "baseline row has empty URL",
                "baseline": baseline,
                "replay": {},
            }

        if self.is_calendar_event_url(baseline["url"]):
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

        def score_candidate(candidate: dict[str, Any]) -> int:
            score = 0
            if self.normalize_text_value(candidate.get("event_name")) == baseline["event_name"]:
                score += 4
            if self.normalize_date_value(candidate.get("start_date")) == baseline["start_date"]:
                score += 4
            if self.normalize_url_value(candidate.get("url")) == baseline["url"]:
                score += 3
            if self.normalize_text_value(candidate.get("source")) == baseline["source"]:
                score += 2
            cand_time = self.normalize_time_value(candidate.get("start_time"))
            if cand_time and baseline["start_time"] and cand_time == baseline["start_time"]:
                score += 2
            return score

        def is_same_page_candidate(candidate: dict[str, Any]) -> bool:
            return self.normalize_url_value(candidate.get("url")) == baseline["url"]

        same_page_events = [candidate for candidate in events if isinstance(candidate, dict) and is_same_page_candidate(candidate)]
        if len(same_page_events) > 1:
            same_date_candidates = [
                candidate
                for candidate in same_page_events
                if self.normalize_date_value(candidate.get("start_date")) == baseline["start_date"]
            ]
            candidate_pool = same_date_candidates or same_page_events

            strong_name_candidates = [
                candidate
                for candidate in candidate_pool
                if (
                    self.name_similarity(
                        baseline["event_name"],
                        self.normalize_text_value(candidate.get("event_name")),
                    ) >= 0.75
                    or self.name_contains_variant(
                        baseline["event_name"],
                        self.normalize_text_value(candidate.get("event_name")),
                    )
                )
            ]
            if strong_name_candidates:
                ranked = sorted(strong_name_candidates, key=score_candidate, reverse=True)
            else:
                same_time_candidates = [
                    candidate
                    for candidate in candidate_pool
                    if self.times_equivalent_with_12h_guard(
                        self.normalize_time_value(candidate.get("start_time")),
                        baseline["start_time"],
                    )
                ]
                if same_time_candidates:
                    ranked = sorted(same_time_candidates, key=score_candidate, reverse=True)
                else:
                    ranked = sorted(candidate_pool, key=score_candidate, reverse=True)
        else:
            ranked = sorted(events, key=score_candidate, reverse=True)
        best = ranked[0]
        best_raw = best.get("raw", {}) if isinstance(best.get("raw"), dict) else {}
        replay = {
            "event_name": self.normalize_text_value(best.get("event_name")),
            "start_date": self.normalize_date_value(best.get("start_date")),
            "start_time": self.normalize_time_value(best.get("start_time")),
            "day_of_week": self.normalize_text_value(best.get("day_of_week")),
            "source": self.normalize_text_value(best.get("source")),
            "location": self.normalize_text_value(best.get("location")),
            "url": self.normalize_url_value(best.get("url")),
            "address_id": self.normalize_optional_int(best_raw.get("address_id")),
            "dance_style": self.normalize_text_value(best_raw.get("dance_style")),
            "description": self.normalize_text_value(best_raw.get("description")),
        }

        if self.is_social_platform_url(baseline["url"]) and replay["url"] != baseline["url"]:
            return {
                "is_match": False,
                "category": "wrong_replay_source_url",
                "details": "social replay candidate URL drifted from baseline source URL",
                "baseline": baseline,
                "replay": replay,
            }

        if self.is_recurring_schedule_match(baseline, replay):
            return {
                "is_match": True,
                "category": "",
                "details": "recurring_event_schedule_match",
                "baseline": baseline,
                "replay": replay,
            }

        duplicate_identity_count = sum(
            1
            for candidate in events
            if (
                self.normalize_text_value(candidate.get("event_name")) == baseline["event_name"]
                and self.normalize_date_value(candidate.get("start_date")) == baseline["start_date"]
                and self.normalize_time_value(candidate.get("start_time")) == baseline["start_time"]
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

        name_similarity = self.name_similarity(baseline["event_name"], replay["event_name"])
        name_contains_variant = self.name_contains_variant(baseline["event_name"], replay["event_name"])
        same_date = replay["start_date"] == baseline["start_date"]
        same_time = self.times_equivalent_with_12h_guard(replay["start_time"], baseline["start_time"])

        best_candidate_name_signal = max(
            (
                self.name_similarity(
                    baseline["event_name"],
                    self.normalize_text_value(candidate.get("event_name")),
                )
                for candidate in events
                if isinstance(candidate, dict)
            ),
            default=0.0,
        )
        any_name_variant_signal = any(
            self.name_contains_variant(
                baseline["event_name"],
                self.normalize_text_value(candidate.get("event_name")),
            )
            for candidate in events
            if isinstance(candidate, dict)
        )
        if (
            len(events) > 1
            and replay["url"] == baseline["url"]
            and same_date
            and best_candidate_name_signal < 0.60
            and not any_name_variant_signal
        ):
            return {
                "is_match": False,
                "category": "wrong_replay_event_selection",
                "details": "listing page replay selected a different event from the same source URL",
                "baseline": baseline,
                "replay": replay,
            }

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

        if (
            same_date
            and ((not strict_time_match) or same_time)
            and (name_similarity >= 0.75 or name_contains_variant)
        ):
            llm_same, llm_reason = self.llm_adjudicate_event_match(baseline, replay)
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
            and not self.is_placeholder_source(replay["source"])
            and not self.is_placeholder_source(baseline["source"])
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

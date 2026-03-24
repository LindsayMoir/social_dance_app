#!/usr/bin/env python3
"""Helpers for replay URL fetching and extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ReplayArtifact:
    """Normalized replay input that can come from a live fetch or stored artifact."""

    source_url: str
    fetch_method: str
    artifact_type: str
    body_text: str
    final_url: str = ""
    status_code: int | None = None
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayFetcher:
    """Fetch and normalize replay events using injected fetch, parse, and routing helpers."""

    is_social_platform_url: Callable[[str], bool]
    extract_visible_text_for_replay: Callable[[str], str]
    extract_replay_links: Callable[[str, str], list[str]]
    parse_replay_events_from_text: Callable[[str, str, str | None], list[dict[str, Any]]]
    fetch_replay_events_via_rd_ext: Callable[[str], dict[str, Any]]
    get_replay_fb_scraper: Callable[[], Any]
    normalize_url_value: Callable[[Any], str]
    normalize_date_value: Callable[[Any], str]
    normalize_time_value: Callable[[Any], str]
    resolve_prompt_type: Callable[[str], str]
    classify_page: Callable[[str, str, list[str]], dict[str, Any]]
    request_get: Callable[..., Any]
    generate_prompt: Callable[[str, str, str], tuple[str, str | None]]
    query_llm: Callable[[str, str, str | None], Any]
    extract_and_parse_json: Callable[[Any, str, str | None], Any]

    def build_live_replay_artifact(self, url: str) -> ReplayArtifact | None:
        """Fetch a live URL and normalize the response into a replay artifact."""
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
                candidate = self.request_get(url, timeout=25, headers=headers)
                if candidate.status_code < 400:
                    response = candidate
                    break
                last_request_error = f"http_status={candidate.status_code}"
            except Exception as exc:
                last_request_error = str(exc)
        if response is None:
            return ReplayArtifact(
                source_url=url,
                fetch_method="requests",
                artifact_type="raw_html",
                body_text="",
                final_url=url,
                status_code=None,
                links=[],
                metadata={"request_error": last_request_error or "request_failed"},
            )

        html_text = str(getattr(response, "text", "") or "")
        return ReplayArtifact(
            source_url=url,
            fetch_method="requests",
            artifact_type="raw_html",
            body_text=html_text,
            final_url=str(getattr(response, "url", "") or url),
            status_code=getattr(response, "status_code", None),
            links=self.extract_replay_links(url, html_text),
            metadata={"request_error": last_request_error} if last_request_error else {},
        )

    def extract_replay_events_from_artifact(self, artifact: ReplayArtifact) -> dict[str, Any]:
        """Extract replay events from a normalized artifact using shared routing logic."""
        extracted_text = self.extract_visible_text_for_replay(artifact.body_text)
        if not extracted_text:
            return {"ok": False, "category": "no_event_extracted_replay", "details": "no_visible_text", "events": []}

        try:
            class_decision = self.classify_page(
                artifact.source_url,
                extracted_text,
                list(artifact.links),
            )
        except Exception:
            class_decision = {}
        if str((class_decision or {}).get("owner_step") or "").strip().lower() == "rd_ext.py":
            rd_ext_payload = self.fetch_replay_events_via_rd_ext(artifact.source_url)
            if rd_ext_payload.get("ok"):
                return rd_ext_payload

        normalized_events = self.parse_replay_events_from_text(
            source_url=artifact.source_url,
            extracted_text=extracted_text,
            replay_url=None,
        )
        if not normalized_events:
            return {"ok": False, "category": "no_event_extracted_replay", "details": "no_normalized_events", "events": []}
        return {"ok": True, "category": "", "details": "", "events": normalized_events}

    def fetch_replay_events_for_social_url(self, url: str) -> dict[str, Any]:
        """Replay fetch for Facebook and Instagram URLs via fb.py methods."""
        fb_scraper = self.get_replay_fb_scraper()
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

            if "facebook.com" in str(url).lower() and "/events/" not in str(url).lower():
                try:
                    relevant = fb_scraper.extract_relevant_text(extracted_text, url)
                    if relevant:
                        extracted_text = relevant
                except Exception:
                    pass

            prompt_type = self.resolve_prompt_type(url)
            prompt, schema_type = self.generate_prompt(url, extracted_text, prompt_type)
            if schema_type is None:
                prompt, schema_type = self.generate_prompt(url, extracted_text, "default")
            if schema_type is None:
                return {"ok": False, "category": "parser_or_llm_failure", "details": "missing_schema_type", "events": []}

            parsed = None
            last_llm_error = ""
            for _attempt in range(1, 3):
                llm_response = self.query_llm(url, prompt, schema_type)
                if not llm_response:
                    last_llm_error = "empty_llm_response"
                    continue
                parsed = self.extract_and_parse_json(llm_response, url, schema_type)
                if parsed:
                    break
                last_llm_error = "parsed_empty"

            if not parsed:
                category = "no_event_extracted_replay" if last_llm_error == "parsed_empty" else "parser_or_llm_failure"
                return {"ok": False, "category": category, "details": last_llm_error or "llm_failed", "events": []}

            normalized_events: list[dict[str, Any]] = []
            for event in parsed:
                if not isinstance(event, dict):
                    continue
                event_raw = dict(event)
                mentioned_url = self.normalize_url_value(event.get("url"))
                if mentioned_url:
                    event_raw["mentioned_url"] = mentioned_url
                normalized_events.append(
                    {
                        "event_name": str(event.get("event_name") or "").strip(),
                        "start_date": self.normalize_date_value(event.get("start_date")),
                        "start_time": self.normalize_time_value(event.get("start_time")),
                        "source": str(event.get("source") or "").strip(),
                        "location": str(event.get("location") or "").strip(),
                        "url": self.normalize_url_value(url),
                        "raw": event_raw,
                    }
                )
            if not normalized_events:
                return {"ok": False, "category": "no_event_extracted_replay", "details": "no_normalized_events", "events": []}
            return {"ok": True, "category": "", "details": "", "events": normalized_events}
        except Exception as exc:
            return {
                "ok": False,
                "category": "parser_or_llm_failure",
                "details": f"social_replay_exception:{exc}",
                "events": [],
            }

    def fetch_replay_events_for_url(self, url: str) -> dict[str, Any]:
        """Fetch replay events for a URL using platform-specific or generic extraction."""
        if self.is_social_platform_url(url):
            return self.fetch_replay_events_for_social_url(url)

        artifact = self.build_live_replay_artifact(url)
        if artifact is None or not artifact.body_text:
            return {
                "ok": False,
                "category": "url_unreachable_replay",
                "details": str((artifact.metadata if artifact else {}).get("request_error") or "request_failed"),
                "events": [],
            }
        return self.extract_replay_events_from_artifact(artifact)

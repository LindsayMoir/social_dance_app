"""Telemetry-driven timeout/retry/pacing tuning for crawler runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Optional


@dataclass(frozen=True)
class ScraperTelemetry:
    """Subset of scraper telemetry used for runtime tuning decisions."""

    request_count: int
    timeout_retry_count: int
    retry_count: int
    retry_max_reached: int
    breaker_triggers: int

    @property
    def timeout_retry_ratio(self) -> float:
        if self.request_count <= 0:
            return 0.0
        return self.timeout_retry_count / float(self.request_count)


@dataclass(frozen=True)
class FbTelemetry:
    """Subset of FB telemetry used for runtime pacing decisions."""

    access_checks: int
    blocked_access_checks: int
    temp_block_strikes: int

    @property
    def blocked_ratio(self) -> float:
        if self.access_checks <= 0:
            return 0.0
        return self.blocked_access_checks / float(self.access_checks)


@dataclass(frozen=True)
class TelemetryTuningResult:
    """Output payload for pipeline consumption and trace logging."""

    updates: dict[str, dict[str, Any]]
    telemetry: dict[str, Any]
    decisions: list[str]


_SCRAPER_START_MARKER = "scraper.py starting"


def _tail_text(path: str, max_chars: int = 500_000) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _latest_run_slice(text: str, start_marker: str, fallback_tail_chars: int = 120_000) -> str:
    if not text:
        return ""
    idx = text.lower().rfind(start_marker.lower())
    if idx >= 0:
        return text[idx:]
    return text[-fallback_tail_chars:]


def _slice_for_latest_run_id(text: str, step_name: str) -> Optional[str]:
    """
    Return only lines for the latest run_id of a given step when present.
    """
    if not text:
        return None
    step_tag = f"[step={step_name}]"
    candidate_lines = [line for line in text.splitlines() if step_tag in line and "[run_id=" in line]
    if not candidate_lines:
        return None
    last_line = candidate_lines[-1]
    match = re.search(r"\[run_id=([^\]]+)\]", last_line)
    if not match:
        return None
    run_id = match.group(1).strip()
    if not run_id:
        return None
    filtered = [line for line in text.splitlines() if f"[run_id={run_id}]" in line and step_tag in line]
    if not filtered:
        return None
    return "\n".join(filtered)


def _extract_scraper_stats_block(scraper_slice: str) -> str:
    marker = "Dumping Scrapy stats:"
    idx = scraper_slice.rfind(marker)
    if idx < 0:
        return ""
    block = scraper_slice[idx + len(marker) :].strip()
    end_idx = block.find("Spider closed")
    if end_idx >= 0:
        block = block[:end_idx]
    return block.strip()


def _extract_stat_int(text: str, key: str) -> int:
    """
    Extract integer metric value from Scrapy stats text.
    """
    pattern = re.compile(rf"['\"]{re.escape(key)}['\"]\s*:\s*(\d+)")
    match = pattern.search(text or "")
    if not match:
        return 0
    return int(match.group(1))


def _parse_scraper_telemetry(scraper_log_text: str) -> ScraperTelemetry:
    run_text = _latest_run_slice(
        scraper_log_text,
        _SCRAPER_START_MARKER,
    )
    stats_block = _extract_scraper_stats_block(run_text) or run_text

    breaker_re = re.compile(r"domain_circuit_breaker_summary:\s*triggers=(\d+)")
    breaker_triggers = 0
    for match in breaker_re.finditer(run_text):
        breaker_triggers = int(match.group(1))

    return ScraperTelemetry(
        request_count=_extract_stat_int(stats_block, "downloader/request_count"),
        timeout_retry_count=_extract_stat_int(
            stats_block,
            "retry/reason_count/twisted.internet.error.TimeoutError",
        ),
        retry_count=_extract_stat_int(stats_block, "retry/count"),
        retry_max_reached=_extract_stat_int(stats_block, "retry/max_reached"),
        breaker_triggers=breaker_triggers,
    )


def _parse_fb_telemetry(fb_log_text: str) -> FbTelemetry:
    run_text = _slice_for_latest_run_id(fb_log_text, "fb") or fb_log_text
    access_checks = len(re.findall(r"fb access check:", run_text))
    blocked_access_checks = len(re.findall(r"state=blocked", run_text))
    temp_block_strikes = len(re.findall(r"fb temp block policy: strike=", run_text))
    return FbTelemetry(
        access_checks=access_checks,
        blocked_access_checks=blocked_access_checks,
        temp_block_strikes=temp_block_strikes,
    )


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def tune_crawl_config_from_first_pass(
    *,
    current_updates: dict[str, dict[str, Any]],
    scraper_log_path: str,
    fb_log_path: str,
) -> TelemetryTuningResult:
    """
    Tune timeout/retry/pacing settings from latest telemetry snapshot.

    The tuner is intentionally bounded and only touches runtime knobs.
    It does not modify URL limits or scope selection.
    """
    scraper_text = _tail_text(scraper_log_path)
    fb_text = _tail_text(fb_log_path)
    scraper = _parse_scraper_telemetry(scraper_text)
    fb = _parse_fb_telemetry(fb_text)

    updates: dict[str, dict[str, Any]] = {"crawling": {}}
    decisions: list[str] = []
    crawling = current_updates.get("crawling", {})

    timeout_ratio = scraper.timeout_retry_ratio
    current_timeout = int(crawling.get("scraper_download_timeout_seconds", 35) or 35)
    current_retry = int(crawling.get("scraper_retry_times", 1) or 1)
    current_concurrency = int(crawling.get("scraper_concurrent_requests", 8) or 8)
    current_per_domain = int(crawling.get("scraper_concurrent_requests_per_domain", 2) or 2)

    if scraper.request_count >= 200 and timeout_ratio >= 0.22:
        updates["crawling"]["scraper_download_timeout_seconds"] = _clamp_int(current_timeout - 10, 25, 70)
        updates["crawling"]["scraper_retry_times"] = _clamp_int(current_retry - 1, 1, 3)
        updates["crawling"]["scraper_concurrent_requests"] = _clamp_int(current_concurrency - 2, 4, 16)
        updates["crawling"]["scraper_concurrent_requests_per_domain"] = _clamp_int(current_per_domain - 1, 1, 6)
        decisions.append(
            "scraper timeout pressure high: reduce hang budget/retries and soften concurrency "
            f"(timeout_ratio={timeout_ratio:.3f}, requests={scraper.request_count})."
        )
    elif scraper.request_count >= 200 and timeout_ratio <= 0.08 and scraper.breaker_triggers == 0:
        updates["crawling"]["scraper_concurrent_requests"] = _clamp_int(current_concurrency + 2, 4, 20)
        updates["crawling"]["scraper_concurrent_requests_per_domain"] = _clamp_int(current_per_domain + 1, 1, 8)
        decisions.append(
            "scraper timeout pressure low: increase concurrency for throughput "
            f"(timeout_ratio={timeout_ratio:.3f}, requests={scraper.request_count})."
        )
    else:
        decisions.append(
            "scraper telemetry stable/insufficient for timeout-retry-concurrency retune "
            f"(timeout_ratio={timeout_ratio:.3f}, requests={scraper.request_count})."
        )

    fb_wait_min = int(crawling.get("fb_inter_request_wait_min_ms", 5000) or 5000)
    fb_wait_max = int(crawling.get("fb_inter_request_wait_max_ms", 15000) or 15000)
    fb_nav_wait = int(crawling.get("fb_post_nav_wait_ms", 1800) or 1800)
    fb_expand_wait = int(crawling.get("fb_post_expand_wait_ms", 900) or 900)
    fb_final_wait = int(crawling.get("fb_final_wait_ms", 700) or 700)

    if fb.access_checks >= 60 and fb.temp_block_strikes == 0 and fb.blocked_ratio <= 0.02:
        updates["crawling"]["fb_inter_request_wait_min_ms"] = _clamp_int(fb_wait_min - 1000, 2500, 20000)
        updates["crawling"]["fb_inter_request_wait_max_ms"] = _clamp_int(fb_wait_max - 1500, 6000, 30000)
        updates["crawling"]["fb_post_nav_wait_ms"] = _clamp_int(fb_nav_wait - 300, 1200, 5000)
        updates["crawling"]["fb_post_expand_wait_ms"] = _clamp_int(fb_expand_wait - 200, 600, 4000)
        updates["crawling"]["fb_final_wait_ms"] = _clamp_int(fb_final_wait - 150, 450, 3000)
        decisions.append(
            "fb blocking pressure low: slightly reduce pacing waits for faster throughput "
            f"(blocked_ratio={fb.blocked_ratio:.3f}, checks={fb.access_checks})."
        )
    elif fb.blocked_ratio >= 0.08 or fb.temp_block_strikes > 0:
        updates["crawling"]["fb_inter_request_wait_min_ms"] = _clamp_int(fb_wait_min + 1000, 2500, 30000)
        updates["crawling"]["fb_inter_request_wait_max_ms"] = _clamp_int(fb_wait_max + 1500, 6000, 45000)
        updates["crawling"]["fb_post_nav_wait_ms"] = _clamp_int(fb_nav_wait + 300, 1200, 7000)
        updates["crawling"]["fb_post_expand_wait_ms"] = _clamp_int(fb_expand_wait + 150, 600, 5000)
        updates["crawling"]["fb_final_wait_ms"] = _clamp_int(fb_final_wait + 150, 450, 4000)
        decisions.append(
            "fb blocking pressure elevated: increase pacing waits for safety "
            f"(blocked_ratio={fb.blocked_ratio:.3f}, strikes={fb.temp_block_strikes})."
        )
    else:
        decisions.append(
            "fb telemetry stable/insufficient for pacing retune "
            f"(blocked_ratio={fb.blocked_ratio:.3f}, checks={fb.access_checks}, strikes={fb.temp_block_strikes})."
        )

    if not updates["crawling"]:
        updates = {}

    return TelemetryTuningResult(
        updates=updates,
        telemetry={
            "scraper": {
                "request_count": scraper.request_count,
                "timeout_retry_count": scraper.timeout_retry_count,
                "retry_count": scraper.retry_count,
                "retry_max_reached": scraper.retry_max_reached,
                "breaker_triggers": scraper.breaker_triggers,
                "timeout_retry_ratio": round(scraper.timeout_retry_ratio, 4),
            },
            "fb": {
                "access_checks": fb.access_checks,
                "blocked_access_checks": fb.blocked_access_checks,
                "temp_block_strikes": fb.temp_block_strikes,
                "blocked_ratio": round(fb.blocked_ratio, 4),
            },
        },
        decisions=decisions,
    )

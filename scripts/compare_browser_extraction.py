#!/usr/bin/env python3
"""Compare DOM-text and ARIA-snapshot extraction on authenticated event pages.

This is a read-only live diagnostic. It navigates directly to the configured
event-detail URLs, captures both extraction representations from the same
rendered page, and writes one JSON artifact per URL plus a summary. It does
not call the LLM or write to the database.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from browser_extraction import (  # noqa: E402
    AriaSnapshotResult,
    capture_aria_snapshot_async,
    capture_aria_snapshot_sync,
    reduce_instagram_aria_snapshot,
)
from config_runtime import load_config  # noqa: E402
from credential_validator import _temporary_headless_config  # noqa: E402
from fb import FacebookEventScraper  # noqa: E402
from images import ImageScraper  # noqa: E402
from secret_paths import get_auth_file  # noqa: E402


DEFAULT_URLS_FILE = REPOSITORY_ROOT / "data/evaluation/browser_extraction_comparison_urls.csv"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "debug/browser_extraction_comparison"
PLATFORMS = ("facebook", "instagram", "eventbrite")


@dataclass(frozen=True)
class ComparisonTarget:
    """One live event page to inspect."""

    platform: str
    url: str
    note: str


@dataclass(frozen=True)
class ComparisonResult:
    """Representations captured from one rendered page."""

    platform: str
    requested_url: str
    final_url: str
    note: str
    dom_text: str
    aria_snapshot: str | None
    aria_snapshot_reason: str | None
    dom_text_length: int
    aria_snapshot_length: int
    dom_access_state: str
    aria_access_state: str
    error: str | None = None


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the live diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", type=Path, default=DEFAULT_URLS_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pages-per-platform", type=int, default=5)
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        action="append",
        dest="platforms",
        help="Limit the comparison to one platform; repeat to select more than one.",
    )
    parser.add_argument("--headful", action="store_true", help="Show Chromium for login/challenge diagnosis.")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--settle-ms", type=int, default=2_500)
    parser.add_argument("--snapshot-timeout-ms", type=int, default=5_000)
    parser.add_argument("--snapshot-max-chars", type=int, default=40_000)
    return parser.parse_args()


def load_targets(
    urls_file: Path,
    pages_per_platform: int,
    platforms: tuple[str, ...] = PLATFORMS,
) -> list[ComparisonTarget]:
    """Load exactly the requested number of direct event URLs per platform."""
    if pages_per_platform < 1:
        raise ValueError("pages_per_platform must be at least one")
    if not urls_file.is_file():
        raise FileNotFoundError(f"Comparison URL file does not exist: {urls_file}")

    selected_platforms = tuple(dict.fromkeys(platforms))
    targets_by_platform: dict[str, list[ComparisonTarget]] = {platform: [] for platform in selected_platforms}
    with urls_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            platform = str(row.get("platform") or "").strip().lower()
            url = str(row.get("url") or "").strip()
            if platform not in targets_by_platform or not url:
                continue
            if len(targets_by_platform[platform]) < pages_per_platform:
                targets_by_platform[platform].append(
                    ComparisonTarget(platform=platform, url=url, note=str(row.get("note") or "").strip())
                )

    missing = [platform for platform, targets in targets_by_platform.items() if len(targets) < pages_per_platform]
    if missing:
        raise ValueError(
            f"Expected {pages_per_platform} URL(s) for every platform; insufficient entries for: {', '.join(missing)}"
        )
    return [target for platform in selected_platforms for target in targets_by_platform[platform]]


def dom_text_from_html(platform: str, html: str) -> str:
    """Mirror the present deterministic text representation for each platform."""
    soup = BeautifulSoup(html, "html.parser")
    if platform == "instagram":
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    return " ".join(soup.stripped_strings)


def detect_access_state(platform: str, representation: str | None) -> str:
    """Classify common login and unavailable-content shells in a representation."""
    text = str(representation or "").lower()
    if not text:
        return "empty"
    if "this content is unavailable" in text or "can't see this content" in text:
        return "content_unavailable"
    if platform == "facebook":
        if "device-based/login" in text:
            return "login_required"
        if "you must log in to continue" in text:
            return "login_required"
        if "continue with saved account" in text:
            return "login_dialog_obscuring_content"
    if platform == "instagram" and "sign up for instagram to stay in the loop" in text:
        return "anonymous_view"
    return "event_content_visible"


def capture_comparison(
    page: Page,
    target: ComparisonTarget,
    *,
    timeout_ms: int,
    settle_ms: int,
    snapshot_timeout_ms: int,
    snapshot_max_chars: int,
) -> ComparisonResult:
    """Capture present DOM text and optional ARIA text from one direct URL."""
    try:
        page.goto(target.url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(settle_ms)
        if target.platform == "facebook":
            for button in page.query_selector_all("text=/See more/i"):
                try:
                    button.click()
                    page.wait_for_timeout(250)
                except Exception:
                    break
        snapshot = capture_aria_snapshot_sync(
            page,
            timeout_ms=snapshot_timeout_ms,
            max_chars=snapshot_max_chars,
        )
        dom_text = dom_text_from_html(target.platform, page.content())
        return ComparisonResult(
            platform=target.platform,
            requested_url=target.url,
            final_url=page.url,
            note=target.note,
            dom_text=dom_text,
            aria_snapshot=snapshot.text,
            aria_snapshot_reason=snapshot.reason,
            dom_text_length=len(dom_text),
            aria_snapshot_length=len(snapshot.text or ""),
            dom_access_state=detect_access_state(target.platform, dom_text),
            aria_access_state=detect_access_state(target.platform, snapshot.text),
        )
    except Exception as exc:
        return ComparisonResult(
            platform=target.platform,
            requested_url=target.url,
            final_url=page.url,
            note=target.note,
            dom_text="",
            aria_snapshot=None,
            aria_snapshot_reason=None,
            dom_text_length=0,
            aria_snapshot_length=0,
            dom_access_state="capture_error",
            aria_access_state="capture_error",
            error=f"{type(exc).__name__}: {exc}",
        )


def create_context(browser: Browser, platform: str) -> BrowserContext:
    """Create an isolated context using saved session state when available."""
    storage_state = Path(get_auth_file(platform))
    context_options: dict[str, Any] = {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    if storage_state.is_file():
        context_options["storage_state"] = str(storage_state)
    else:
        logging.warning("No saved %s session state at %s; results may be login shells.", platform, storage_state)
    return browser.new_context(**context_options)


def capture_instagram_comparison(
    scraper: ImageScraper,
    target: ComparisonTarget,
    *,
    timeout_ms: int,
    settle_ms: int,
    snapshot_timeout_ms: int,
    snapshot_max_chars: int,
) -> ComparisonResult:
    """Capture both Instagram representations using ImageScraper's verified session."""
    page = scraper.read_extract.page

    async def capture() -> tuple[str, Any]:
        await page.goto(target.url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(settle_ms)
        snapshot = await capture_aria_snapshot_async(
            page,
            timeout_ms=snapshot_timeout_ms,
            max_chars=snapshot_max_chars,
        )
        reduced_snapshot = reduce_instagram_aria_snapshot(snapshot.text)
        if reduced_snapshot:
            snapshot = AriaSnapshotResult(text=reduced_snapshot, reason=snapshot.reason)
        return await page.content(), snapshot

    try:
        html, snapshot = scraper.loop.run_until_complete(capture())
        dom_text = dom_text_from_html(target.platform, html)
        return ComparisonResult(
            platform=target.platform,
            requested_url=target.url,
            final_url=page.url,
            note=target.note,
            dom_text=dom_text,
            aria_snapshot=snapshot.text,
            aria_snapshot_reason=snapshot.reason,
            dom_text_length=len(dom_text),
            aria_snapshot_length=len(snapshot.text or ""),
            dom_access_state=detect_access_state(target.platform, dom_text),
            aria_access_state=detect_access_state(target.platform, snapshot.text),
        )
    except Exception as exc:
        return ComparisonResult(
            platform=target.platform,
            requested_url=target.url,
            final_url=str(getattr(page, "url", "") or ""),
            note=target.note,
            dom_text="",
            aria_snapshot=None,
            aria_snapshot_reason=None,
            dom_text_length=0,
            aria_snapshot_length=0,
            dom_access_state="capture_error",
            aria_access_state="capture_error",
            error=f"{type(exc).__name__}: {exc}",
        )


def write_results(output_dir: Path, results: list[ComparisonResult]) -> Path:
    """Write one inspectable artifact per page and a compact run summary."""
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, result in enumerate(results, start=1):
        destination = output_dir / f"{index:02d}_{result.platform}.json"
        destination.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages_requested": len(results),
        "pages_with_dom_text": sum(bool(result.dom_text) for result in results),
        "pages_with_aria_snapshot": sum(bool(result.aria_snapshot) for result in results),
        "pages_with_errors": sum(bool(result.error) for result in results),
        "results": [
            {
                "platform": result.platform,
                "requested_url": result.requested_url,
                "final_url": result.final_url,
                "dom_text_length": result.dom_text_length,
                "aria_snapshot_length": result.aria_snapshot_length,
                "aria_snapshot_reason": result.aria_snapshot_reason,
                "dom_access_state": result.dom_access_state,
                "aria_access_state": result.aria_access_state,
                "error": result.error,
            }
            for result in results
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def run_comparison(arguments: argparse.Namespace) -> Path:
    """Run the direct-page comparison and return its summary artifact path."""
    selected_platforms = tuple(arguments.platforms or PLATFORMS)
    targets = load_targets(arguments.urls_file, arguments.pages_per_platform, selected_platforms)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = arguments.output_dir or DEFAULT_OUTPUT_ROOT / run_timestamp
    config = load_config()
    headless = not arguments.headful and bool(config.get("crawling", {}).get("headless", True))
    results: list[ComparisonResult] = []

    if "facebook" in selected_platforms:
        with _temporary_headless_config(headless):
            facebook_scraper = FacebookEventScraper(config_path="config/config.yaml")
            try:
                for target in (item for item in targets if item.platform == "facebook"):
                    results.append(
                        capture_comparison(
                            facebook_scraper.page,
                            target,
                            timeout_ms=arguments.timeout_ms,
                            settle_ms=arguments.settle_ms,
                            snapshot_timeout_ms=arguments.snapshot_timeout_ms,
                            snapshot_max_chars=arguments.snapshot_max_chars,
                        )
                    )
            finally:
                facebook_scraper._safe_shutdown_browser()

    if "instagram" in selected_platforms:
        with _temporary_headless_config(headless):
            instagram_scraper = ImageScraper(load_config())
            try:
                for target in (item for item in targets if item.platform == "instagram"):
                    results.append(
                        capture_instagram_comparison(
                            instagram_scraper,
                            target,
                            timeout_ms=arguments.timeout_ms,
                            settle_ms=arguments.settle_ms,
                            snapshot_timeout_ms=arguments.snapshot_timeout_ms,
                            snapshot_max_chars=arguments.snapshot_max_chars,
                        )
                    )
            finally:
                instagram_scraper.loop.run_until_complete(instagram_scraper.read_extract.close())

    if "eventbrite" in selected_platforms:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            try:
                context = create_context(browser, "eventbrite")
                try:
                    for target in (item for item in targets if item.platform == "eventbrite"):
                        page = context.new_page()
                        try:
                            results.append(
                                capture_comparison(
                                    page,
                                    target,
                                    timeout_ms=arguments.timeout_ms,
                                    settle_ms=arguments.settle_ms,
                                    snapshot_timeout_ms=arguments.snapshot_timeout_ms,
                                    snapshot_max_chars=arguments.snapshot_max_chars,
                                )
                            )
                        finally:
                            page.close()
                finally:
                    context.close()
            finally:
                browser.close()
    return write_results(output_dir, results)


def main() -> int:
    """Run the live diagnostic from the command line."""
    arguments = parse_arguments()
    try:
        summary_path = run_comparison(arguments)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Browser extraction comparison setup error: %s", exc)
        return 2
    print(f"Comparison artifacts written to {summary_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

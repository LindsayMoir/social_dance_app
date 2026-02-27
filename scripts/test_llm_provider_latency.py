#!/usr/bin/env python3
"""
Measure response-time latency across chatbot providers:
- OpenAI
- OpenRouter DeepSeek
- Gemini

This is a live integration benchmark (calls external APIs).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

from mistralai import Mistral
import yaml
from openai import OpenAI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from llm import LLMHandler  # noqa: E402


@dataclass(frozen=True)
class ProviderRunResult:
    provider: str
    model: str
    latency_seconds: float | None
    ok: bool
    error: str | None


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def _first_non_empty(items: list[str]) -> str | None:
    for item in items:
        candidate = str(item or "").strip()
        if candidate:
            return candidate
    return None


def _resolve_models(config: dict, args: argparse.Namespace) -> dict[str, str]:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    openai_default = _first_non_empty(
        [
            args.openai_model,
            llm_cfg.get("openai_model"),
            *list(llm_cfg.get("openai_model_candidates", [])),
            "gpt-5-mini",
        ]
    )
    gemini_default = _first_non_empty(
        [
            args.gemini_model,
            *list(llm_cfg.get("gemini_model_candidates", [])),
            llm_cfg.get("gemini_model"),
            "gemini-2.5-flash-lite",
        ]
    )
    deepseek_default = _first_non_empty(
        [
            args.deepseek_model,
            *list(llm_cfg.get("chatbot_openrouter_model_candidates", [])),
            *list(llm_cfg.get("openrouter_model_candidates", [])),
            "deepseek/deepseek-v3.2",
        ]
    )
    mistral_default = _first_non_empty(
        [
            args.mistral_model,
            llm_cfg.get("mistral_model"),
            *list(llm_cfg.get("mistral_model_candidates", [])),
            "mistral-large-latest",
        ]
    )
    if not openai_default or not gemini_default or not deepseek_default or not mistral_default:
        raise ValueError("Unable to resolve provider model defaults from args/config.")
    return {
        "openai": openai_default,
        "gemini": gemini_default,
        "openrouter": deepseek_default,
        "mistral": mistral_default,
    }


def _build_handler(config: dict) -> LLMHandler:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    handler = LLMHandler.__new__(LLMHandler)
    handler.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        handler.mistral_client = Mistral(api_key=mistral_api_key, timeout_ms=60000)
    else:
        handler.mistral_client = None
    handler.openrouter_token = os.getenv("OPENROUTER_API_KEY")
    handler.openrouter_base_url = str(llm_cfg.get("openrouter_base_url", "https://openrouter.ai/api/v1"))
    handler.openrouter_timeout_seconds = int(llm_cfg.get("openrouter_timeout_seconds", 60) or 60)
    handler.gemini_token = os.getenv("GEMINI_API_KEY")
    handler.gemini_timeout_seconds = int(llm_cfg.get("gemini_timeout_seconds", 60) or 60)
    return handler


def _timed_call(provider: str, model: str, fn: Callable[[], str | None]) -> ProviderRunResult:
    start = time.perf_counter()
    try:
        response = fn()
        elapsed = time.perf_counter() - start
        if not response:
            return ProviderRunResult(provider=provider, model=model, latency_seconds=elapsed, ok=False, error="empty response")
        return ProviderRunResult(provider=provider, model=model, latency_seconds=elapsed, ok=True, error=None)
    except Exception as exc:  # pragma: no cover - live integration error path
        elapsed = time.perf_counter() - start
        return ProviderRunResult(provider=provider, model=model, latency_seconds=elapsed, ok=False, error=str(exc))


def _print_summary(results: list[ProviderRunResult], provider_order: list[str]) -> None:
    print("\nLatency Summary (seconds)")
    print("provider,model,attempts,successes,failures,avg,p50,p95,min,max,last_error")
    grouped: dict[str, list[ProviderRunResult]] = {}
    for item in results:
        grouped.setdefault(item.provider, []).append(item)

    for provider in provider_order:
        items = grouped.get(provider, [])
        if not items:
            continue
        ok_items = [entry for entry in items if entry.ok and entry.latency_seconds is not None]
        latencies = [entry.latency_seconds for entry in ok_items if entry.latency_seconds is not None]
        failures = [entry for entry in items if not entry.ok]
        avg_value = statistics.mean(latencies) if latencies else None
        p50_value = statistics.median(latencies) if latencies else None
        p95_value = (statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else None)
        min_value = min(latencies) if latencies else None
        max_value = max(latencies) if latencies else None
        last_error = failures[-1].error if failures else ""
        model = items[0].model
        print(
            f"{provider},{model},{len(items)},{len(ok_items)},{len(failures)},"
            f"{_fmt(avg_value)},{_fmt(p50_value)},{_fmt(p95_value)},"
            f"{_fmt(min_value)},{_fmt(max_value)},{last_error}"
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live latency benchmark for OpenAI, DeepSeek (OpenRouter), and Gemini.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed calls per provider.")
    parser.add_argument(
        "--prompt",
        default="Reply with JSON: {\"ok\": true, \"provider\": \"name\"}",
        help="Prompt used for all provider calls.",
    )
    parser.add_argument("--openai-model", default="", help="Override OpenAI model.")
    parser.add_argument("--deepseek-model", default="", help="Override OpenRouter model (DeepSeek).")
    parser.add_argument("--gemini-model", default="", help="Override Gemini model.")
    parser.add_argument("--mistral-model", default="", help="Override Mistral model.")
    parser.add_argument(
        "--include-mistral",
        action="store_true",
        help="Also benchmark Mistral with the same number of runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("ERROR: --runs must be >= 1")
        return 2

    config = _load_config(args.config)
    models = _resolve_models(config, args)
    handler = _build_handler(config)

    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }
    if args.include_mistral:
        required_keys["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
    missing = [key for key, value in required_keys.items() if not value]
    if missing:
        print(f"ERROR: Missing API keys: {', '.join(missing)}")
        return 2

    print("Running latency benchmark")
    print(f"runs_per_provider={args.runs}")
    print(f"openai_model={models['openai']}")
    if args.include_mistral:
        print(f"mistral_model={models['mistral']}")
    print(f"openrouter_model={models['openrouter']}")
    print(f"gemini_model={models['gemini']}")

    provider_order = ["openai", "openrouter", "gemini"]
    if args.include_mistral:
        provider_order.insert(1, "mistral")

    results: list[ProviderRunResult] = []
    for run_index in range(1, args.runs + 1):
        print(f"\nRun {run_index}/{args.runs}")
        openai_result = _timed_call(
            provider="openai",
            model=models["openai"],
            fn=lambda: handler.query_openai(prompt=args.prompt, model=models["openai"], schema_type=None),
        )
        print(f"openai: {'ok' if openai_result.ok else 'fail'} latency={_fmt(openai_result.latency_seconds)}")
        results.append(openai_result)

        if args.include_mistral:
            mistral_result = _timed_call(
                provider="mistral",
                model=models["mistral"],
                fn=lambda: handler.query_mistral(prompt=args.prompt, model=models["mistral"], schema_type=None),
            )
            print(f"mistral: {'ok' if mistral_result.ok else 'fail'} latency={_fmt(mistral_result.latency_seconds)}")
            results.append(mistral_result)

        openrouter_result = _timed_call(
            provider="openrouter",
            model=models["openrouter"],
            fn=lambda: handler.query_openrouter(prompt=args.prompt, model=models["openrouter"], schema_type=None),
        )
        print(f"openrouter: {'ok' if openrouter_result.ok else 'fail'} latency={_fmt(openrouter_result.latency_seconds)}")
        results.append(openrouter_result)

        gemini_result = _timed_call(
            provider="gemini",
            model=models["gemini"],
            fn=lambda: handler.query_gemini(prompt=args.prompt, model=models["gemini"], schema_type=None),
        )
        print(f"gemini: {'ok' if gemini_result.ok else 'fail'} latency={_fmt(gemini_result.latency_seconds)}")
        results.append(gemini_result)

    _print_summary(results, provider_order=provider_order)
    if all(not item.ok for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

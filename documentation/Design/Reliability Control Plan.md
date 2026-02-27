# Reliability Control Plan

## Purpose
Define a repeatable reliability system for DanceScoop so the pipeline can run unattended, recover from common failures, and improve over time with measurable outcomes.

## Context
The system has three persistent reliability challenges:
- LLM variability (stochastic outputs, schema drift, edge-case interpretation).
- Unreliable external dependencies (web pages change, network failures, provider rate limits/timeouts).
- Open-ended user input in chatbot flows (unexpected phrasing, ambiguous intent, SQL safety requirements).

The goal is not zero failures. The goal is fast detection, controlled degradation, automated recovery where safe, and continuous reduction of recurring failure modes.

## Reliability Principles
1. JSON-first observability: machine-readable artifacts are the source of truth.
2. HTML for review: `comprehensive_test_report.html` is a human-facing projection of JSON/log data.
3. Every incident becomes a regression test or guardrail.
4. Fallbacks are bounded, explicit, and cost-aware.
5. Promotion gates enforce reliability baselines (no silent regression).

## Source of Truth and Artifacts
Required run artifacts:
1. Validation/report JSON (driver for reliability scoring and issue extraction).
2. Raw logs (pipeline + provider-level request/response metadata).
3. `comprehensive_test_report.html` (summary, trends, action items).

Operational rule:
- Diagnose and classify from JSON/logs first.
- Use HTML as triage dashboard, not primary forensic data.

## Reliability Control Loop
1. Run scheduled pipeline + validations.
2. Generate artifacts (JSON, logs, HTML).
3. Auto-classify failures into categories.
4. Create structured issue records with reproducible inputs.
5. Implement fix with tests and guardrails.
6. Re-run targeted tests + validation suite.
7. Publish scorecard trend deltas (7-day, 30-day).
8. Promote only if reliability gates pass.

## Failure Taxonomy
Primary categories:
1. Scrape Coverage Miss: event exists on source page but missing in DB.
2. Extraction/Parsing Error: page fetched but event fields are malformed/empty.
3. LLM Structural Failure: invalid JSON/schema mismatch/tool-call failure.
4. LLM Semantic Failure: wrong interpretation, wrong SQL intent, hallucinated filters.
5. Provider Reliability Failure: timeout/rate-limit/auth/network errors.
6. Cost/Route Drift: expensive fallback usage exceeds policy.
7. Data Integrity Failure: bad/unknown address, duplicate conflict, wrong event typing.

Each issue record should contain:
- `issue_id`, `timestamp`, `category`, `severity`, `step`, `provider` (if applicable), `url/source`, `input_signature`, `expected`, `actual`, `status`, `owner`, `first_seen`, `last_seen`.

## SLOs and Core Metrics
Track and publish in report:
1. Scrape coverage rate: `% expected events captured`.
2. Structured extraction success: `% runs/pages producing valid structured rows`.
3. SQL validity rate: `% generated SQL passing deterministic validator`.
4. First-pass LLM success: `% requests resolved without provider fallback`.
5. Fallback depth: average provider/model attempts per request.
6. Provider reliability: success rate, timeout rate, 429 rate per provider/model.
7. Latency: p50/p95 end-to-end and by provider/model.
8. Cost efficiency: estimated cost per 100 requests and per successful result.
9. Regression reopen rate: `% issues reappearing after marked fixed`.

## Guardrails and Promotion Gates
Example release gates:
1. SQL validity must not regress below baseline threshold.
2. Scrape coverage must not regress by more than configured delta.
3. Provider timeout/rate-limit error rate must stay under threshold.
4. Cost per successful result must stay within budget band.
5. No new high-severity data integrity issues.

If any gate fails:
- Block promotion.
- Auto-attach top failure signatures and suggested mitigations in the report.

## Self-Healing Strategies
Safe auto-remediation patterns:
1. Provider circuit breaker/cooldown after repeated `429` or timeouts.
2. Bounded retries with jitter and max attempt caps.
3. Deterministic fallback path for high-risk operations (especially SQL generation).
4. Domain-specific scraper fallback (e.g., secondary extraction path) when known pattern fails.
5. Quarantine suspicious outputs instead of writing low-confidence data to production.

Never self-heal by silently bypassing validation for SQL or data integrity constraints.

## Test Strategy
For each reliability fix:
1. Add a deterministic regression test reproducing failure signature.
2. Add one integration check for pipeline path impact.
3. Capture before/after metric deltas in report.

Minimum automated suites:
1. Unit tests for parser/normalization/validators.
2. Integration tests for provider routing/fallback behavior.
3. Validation tests for chatbot SQL safety and semantic intent mapping.

## Report Requirements (`comprehensive_test_report.html`)
Add sections:
1. Reliability Scorecard (7-day and 30-day trends).
2. Top Regressions (new and reopened).
3. Provider Health (success, timeout, 429, p50/p95 latency).
4. Fallback and Cost Pressure indicators.
5. Action Queue (recommended fixes with severity and ownership).

## Implementation Phases
Phase 1: Observability Foundation
1. Normalize validation JSON output for issue extraction.
2. Add provider/model telemetry fields to logs.
3. Add baseline scorecard section in report.

Phase 2: Gated Reliability
1. Define threshold config for gates.
2. Enforce pass/fail at end of validation pipeline.
3. Auto-generate issue records for failed gates.

Phase 3: Targeted Self-Healing
1. Add circuit breakers and bounded retries per provider.
2. Add deterministic fallback for SQL-critical paths.
3. Add source-specific scrape fallback for high-value domains.

Phase 4: Continuous Optimization
1. Track cost/latency/quality tradeoffs by route.
2. Tune model/provider order from real outcomes.
3. Retire brittle heuristics replaced by validated rules.

## Ownership and Cadence
Weekly operating rhythm:
1. Review scorecard and top regressions.
2. Select top 3 recurring issues by impact.
3. Ship fixes with tests and close-loop verification.

Definition of Done for reliability work:
1. Reproducible failure documented.
2. Fix merged.
3. Regression test added and passing.
4. Metric improvement visible in report.

## Immediate Next Actions
1. Add reliability scorecard and provider health sections to `comprehensive_test_report.html`.
2. Add structured issue extraction from report-driving JSON.
3. Add two initial promotion gates:
   - SQL validity floor.
   - Provider timeout/rate-limit ceiling.
4. Add one self-healing policy:
   - Provider cooldown after repeated timeout/429.

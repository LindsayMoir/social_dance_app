# Reinforcement Learning Telemetry Design

## Purpose
Define the telemetry architecture required to move DanceScoop from a human-operated scraping system to a self-improving system that can:

1. Generate enough structured evidence on each run for Codex to analyze regressions and opportunities.
2. Propose targeted code or config changes based on measured outcomes.
3. Safely apply narrow changes with rollback and guardrails.
4. Progress toward a reinforcement-learning-like loop driven by explicit reward signals, not guesswork from text logs.

This document is intentionally more operational than [evaluation_framework.md](/mnt/d/GitHub/social_dance_app/documentation/Reinforcement%20Learning/evaluation_framework.md). The evaluation framework defines what success means. This document defines what telemetry must exist so that success, failure, and causality can be measured honestly enough for automation.

## Problem Statement
The current application has useful observability, but it is not yet sufficient for autonomous improvement.

Current strengths:
1. `url_scrape_metrics` exists and already captures URL-level telemetry.
2. `runs` captures some step summaries.
3. validation metric tables already exist.
4. deletion auditing exists.
5. run artifacts such as scorecards and logs exist.

Current blockers:
1. different scrapers have historically used different counting semantics
2. some telemetry is still encoded only in free-text logs
3. event writes and event deletes are not fully attributed to the responsible step and decision path
4. LLM routing decisions are not captured with enough detail to optimize providers automatically
5. validation failures are not normalized into machine-actionable reason codes
6. configuration and code version are not fully snapshotted at the level needed for causal comparison
7. human corrections are not yet a first-class feedback signal

Without fixing those gaps, Codex can help with ad hoc debugging, but it cannot safely operate as a self-healing optimizer.

## Design Principles
1. Structured records over text logs. Text logs remain useful for humans, but they are not the canonical source for automation.
2. Causal attribution. For every meaningful outcome, we must know which code path, config version, provider, and URL caused it.
3. Stable semantics. A metric like `events_written` must mean the same thing in every step.
4. Longitudinal persistence. Telemetry must live in the database, not only in local artifacts.
5. Reward-bearing outcomes. The system must record not just writes, but whether the writes were later kept, deleted, validated, or corrected.
6. Safe automation. Telemetry must support before/after comparisons, guardrails, and rollback.
7. Idempotent writes. Telemetry writes must be safe to retry and must not silently double-count.
8. Versioned contracts. Telemetry schemas, reason codes, and scorecard payloads must be explicitly versioned.
9. Least privilege and data minimization. Telemetry should capture what is needed for optimization, but should avoid unnecessary sensitive content.
10. Testability. Every telemetry path should be covered by deterministic unit or integration tests.

## Software Engineering Requirements
This telemetry system is infrastructure, not incidental logging. It must be treated with the same engineering discipline as core application code.

### 1. Schema Versioning
Every persisted telemetry contract should have an explicit version.

Required:
1. `schema_version` on major tables or payloads where shape may evolve
2. versioned reason-code registries
3. versioned `run_scorecard.json` payload contract
4. migration notes when a telemetry meaning changes

Rationale:
1. self-healing systems depend on stable semantics
2. silently changing telemetry meaning destroys longitudinal comparability

### 2. Idempotency and Deduplication
Telemetry writes must be safe under retries, step restarts, and partial failures.

Required:
1. deterministic natural keys where possible
2. upsert or uniqueness strategy for retry-prone writes
3. explicit duplicate-handling policy for per-attempt and per-event telemetry
4. run restart behavior documented for each table

Examples:
1. `llm_attempt_metrics` should have a stable uniqueness rule such as `run_id + step_name + url + attempt_number + provider + model`
2. `event_write_attribution` should avoid duplicate rows when a write is retried after transient DB failure

### 3. Data Retention and Privacy
Telemetry volume will grow quickly and may include sensitive or semi-sensitive content.

Required:
1. retention policy by table
2. archival strategy for high-volume raw telemetry
3. redaction rules for prompts, responses, email-derived content, and reviewer notes
4. explicit distinction between raw payload storage and summary storage

Rules:
1. store hashes, signatures, previews, or derived features when full raw content is not required
2. avoid storing secrets, auth tokens, cookies, or unnecessary user content
3. treat email-derived text and reviewer annotations as potentially sensitive

### 4. Performance and Cost Controls
Telemetry itself must not become a bottleneck.

Required:
1. indexes for high-frequency query paths
2. bounded payload sizes for JSON columns
3. async or buffered write strategy where synchronous writes would materially slow scraping
4. telemetry budget monitoring for row counts and storage growth

Rules:
1. heavy JSON payloads should be summarized before persistence when raw detail is not needed
2. repeated high-volume writes should support batching where practical

### 5. Migration and Backfill Strategy
This design introduces new tables and semantics. The rollout must not break historical reporting.

Required:
1. additive migrations first
2. dual-write period where needed
3. backfill strategy for recent runs when possible
4. explicit deprecation plan for replaced metrics or artifacts

Recommended rollout:
1. add new tables and fields
2. begin dual-writing from existing code paths
3. compare old and new reports for at least several runs
4. cut dashboards and reports over only after parity is confirmed

### 6. Telemetry Quality Validation
Telemetry must validate itself.

Required checks:
1. no negative counts unless intentionally defined
2. `events_written` in aggregates should reconcile with attribution tables
3. per-run totals should reconcile with scorecard summaries
4. required fields should be present for high-priority tables
5. reason codes should come from a controlled set

Recommended:
1. add telemetry integrity checks to the validation pipeline
2. fail loudly when canonical tables drift from scorecard summaries

### 7. Ownership and Operational Discipline
Telemetry systems decay if no one owns them.

Required:
1. identify canonical tables vs convenience artifacts
2. document ownership for schema evolution
3. require documentation updates when telemetry semantics change
4. require tests for new reason codes, scorecard fields, and attribution flows

### 8. Standards Alignment
Where practical, telemetry naming and metadata should align to public standards rather than inventing local conventions.

Required:
1. adopt OpenTelemetry-style resource and attribute naming for shared dimensions such as service, environment, error, URL, HTTP, DB, and model/provider metadata
2. keep a documented mapping from local tables/fields to standard semantic concepts
3. prefer standard metric units and attribute naming conventions where they fit the application

Rationale:
1. standard naming improves interoperability with off-the-shelf observability tools
2. it reduces future migration cost if telemetry is exported to tracing or metrics backends

### 9. Lineage and Provenance
Autonomous improvement requires reproducible provenance, not just aggregate metrics.

Required:
1. track lineage between run context, URL decisions, LLM attempts, event writes, validation outcomes, and delete outcomes
2. preserve source provenance for important derived artifacts such as scorecards and auto-fix candidates
3. ensure that a recommendation can be traced back to the raw evidence that produced it

Examples:
1. `auto_fix_candidates.json` should link to the exact run, domains, URLs, and metrics that triggered the recommendation
2. event-level validation should remain attributable to the original write path and model/provider when known

### 10. Telemetry Health Monitoring
The telemetry system itself must be observable and alarmable.

Required:
1. track telemetry ingest success/failure rates
2. track queue depth or write backlog if telemetry is buffered
3. alert on missing expected run artifacts or missing canonical telemetry rows
4. define SLOs for telemetry completeness on critical tables

Examples:
1. if a run completes but `url_scrape_metrics` is missing rows for a scraper step, that should be detectable
2. if attribution totals stop reconciling with scorecards, validation should fail

## Target Capability
When this design is implemented, Codex should be able to answer questions like:

1. Which domains are causing the largest runtime waste relative to event yield?
2. Which scraper heuristics should be tightened because they produce mostly validation failures or cleanup deletes?
3. Which provider and model combinations maximize event yield per cost for a given step and source type?
4. Which prompt or routing changes improved performance and which regressed it?
5. Which cleanup rules are deleting too many valid events?
6. Which manual reviewer corrections should become code or config rules?

That is the minimum viable foundation for a self-healing loop.

## Recommended First Subsystem
The first self-improving subsystem should not be the entire application. It should be the page-parsing and parser-routing subsystem.

Rationale:
1. it is one of the highest-value failure surfaces in the product
2. it has a relatively clean input-decision-output-evaluation loop
3. it is much easier to constrain safely than whole-application automation
4. it directly reduces the amount of human intervention required after each `pipeline.py` run

Scope of the first subsystem:
1. page classification and archetype assignment
2. extractor selection, such as `scraper`, `rd_ext`, `images`, or deterministic special-case paths
3. prompt selection, including URL-specific prompt choice
4. provider and model routing for extraction attempts
5. parse confidence estimation and likely-bad-output detection
6. clustering repeated parsing failures into reusable remediation categories
7. local fetch-artifact capture for offline parser iteration

This subsystem should be treated as the first practical milestone toward reinforcement-style optimization.

It is intentionally narrower than:
1. end-to-end autonomous software modification
2. fully automatic self-healing across the whole pipeline
3. unconstrained RL over all pipeline decisions

The immediate goal is not "make the whole application self-learning." The immediate goal is:
1. learn which parsing strategy works best for a given page shape
2. detect when extraction is probably wrong
3. reduce silent bad writes
4. route unfamiliar or low-confidence pages into a structured remediation loop
5. replay parser changes quickly against locally stored page artifacts instead of repeatedly hitting live URLs

Success criteria for the first subsystem:
1. fewer human-discovered parsing defects per run
2. improved replay accuracy on parser-sensitive URLs
3. fewer invalid rows that are later deleted or remediated
4. better handling of unfamiliar page shapes without immediate manual coding

Human involvement after each run should therefore focus on:
1. reviewing prioritized parsing failure clusters rather than raw logs
2. labeling the failure mode for important mismatches
3. approving or rejecting narrow remediation proposals
4. supplying occasional ground-truth correction on new page types

### Local Artifact Corpus For Parser Learning
This first subsystem should include a local corpus of fetch artifacts so parser work can be iterated quickly and reproducibly without repeatedly scraping live URLs.

Why this is important:
1. it separates parser quality from network failures, rate limits, and site drift
2. it makes tight offline iteration possible for parser rules, prompts, replay matching, and routing policies
3. it creates a stable state snapshot suitable for offline reward modeling and later reinforcement-style learning
4. it reduces turnaround time when investigating parser regressions

What should be captured:
1. for static pages: raw response body, final URL, status code, headers, fetch timestamp, and content hash
2. for JS-rendered pages: rendered HTML after Playwright, original URL, final URL, render timestamp, and content hash
3. for image-heavy pages: rendered HTML, extracted text, OCR text, image URLs, and stable image hashes or local image artifacts when needed
4. for social and listing pages: the exact page artifact used for replay comparison, not only the normalized parse output

What should not be assumed:
1. raw `requests` HTML alone is not sufficient for JS-heavy pages
2. Beautiful Soup output is not a substitute for storing the original raw or rendered artifact
3. this corpus should not start as a full capture of every URL; it should start with replay-target URLs, high-value sources, and important failures

Persistence layer:
1. persist artifact metadata and lookup fields in Postgres
2. persist large artifact payloads such as raw HTML, rendered HTML, OCR blobs, screenshots, and image binaries on the filesystem first
3. use `storage_path` in the database to point to the stored payload
4. only move heavy payloads to object storage later if local volume requires it

Initial capture policy:
1. always capture artifacts for replay-row failures used in the validation report
2. capture artifacts for a bounded allowlist of high-value or high-volume domains
3. capture artifacts for URLs that trigger new mismatch categories or low-confidence parse outcomes
4. defer full-run broad capture until storage, retention, and deduplication behavior are proven

Operational use:
1. the validation and replay framework should be able to run against local artifacts instead of live URLs
2. parser experiments should prefer the local artifact corpus for fast iteration
3. live fetching should remain available for new discovery and for confirming that a source has materially changed

Success condition:
1. a parser bug can be reproduced locally from a stored artifact
2. a proposed parser fix can be tested against that artifact set without re-hitting the source site
3. the artifact corpus becomes the stable training and evaluation bed for parser-policy improvements

### Artifact Corpus Operating Model
The artifact corpus should not create a second independent parser stack. It should reuse the same extraction logic as the live scrapers while swapping the fetch source.

Core rule:
1. same extraction methods, different fetch source

Desired shape:
1. live path: `url -> fetch/render -> normalize artifact -> extract -> write/score`
2. offline path: `stored artifact -> normalize artifact -> extract -> score`

Implication:
1. scraper code should be refactored so fetch/render is injectable
2. extraction logic should be callable from either a live URL or a stored artifact
3. artifact replay should not duplicate parsing logic in separate ad hoc scripts

Expected runners and utilities:
1. artifact capture writer that saves raw or rendered page artifacts during live runs
2. offline replay runner that loads stored artifacts and calls the same extraction/classification methods used by `rd_ext.py`, `scraper.py`, `fb.py`, and `images.py`
3. artifact evaluation runner that compares offline extraction output to expected or validated outcomes
4. targeted experiment runner for bounded slices such as one domain, one mismatch category, or one page archetype

### How This Fits With `pipeline.py`
The first implementation should not put full offline artifact replay into the main scraping critical path.

Initial integration model:
1. `pipeline.py` captures artifacts during normal live scraping for selected URLs
2. validation identifies important parser failures and high-value replay targets
3. offline replay runs after the scrape, or as a separate validation/post-run workflow, against stored artifacts
4. the report and later automation consume those offline results

Why this order is preferred:
1. it keeps production scraping focused on source discovery and insertion
2. it avoids slowing the main run with large offline experiment loops
3. it allows parser iteration to happen much more frequently than live scraping

Recommended rollout:
1. add artifact capture for replay-target URLs, high-value domains, and important failures
2. refactor one scraper family so extraction can run from stored artifacts
3. add an offline replay command for that scraper family
4. surface offline replay results in the validation report
5. only later consider a bounded `artifact_replay_validation` step in `pipeline.py`

Current implementation status:
1. the validation/replay code has been partially refactored toward this target shape
2. parser workflow guidance, replay matching policy, replay acquisition, and replay normalization utilities have been separated into dedicated helper modules rather than living only in `tests/validation/test_runner.py`
3. replay acquisition now includes a normalized replay-artifact boundary so live URL replay follows `url -> fetch -> normalize artifact -> extract`
4. this is preparatory architecture only; it does not yet implement persisted `fetch_artifacts` capture or a stored-artifact loader
5. Phase 1 remains the active operational gate because real `pipeline.py` runs are still required before this repository should treat Phase 1 as complete

### Parser Workflow Report Follow-Ups
The validation report should evolve into the operational control surface for this first subsystem. The following report improvements should be treated as planned follow-up work so they are not lost:

1. add direct links from each priority URL row to the relevant replay row number in `comprehensive_test_report.html`
2. persist the parser-improvement workflow as a JSON artifact, not only as HTML, so Codex and later automation can consume it directly
3. rank priority URLs by estimated impact, not only by mismatch frequency
4. include a `fix_owner` field such as `replay_matcher`, `scraper`, `db_write_guard`, or `normalization`

Rationale:
1. direct links reduce reviewer time spent searching the report
2. JSON persistence makes the workflow machine-actionable for later automated triage
3. impact-based ranking is a better prioritization strategy than raw mismatch counts alone
4. explicit ownership reduces ambiguity about where the next change should be made

Operational intent:
1. the report should tell the reviewer exactly which URL to inspect first
2. it should state where the likely fix belongs
3. it should provide an acceptance test for the next validation-only rerun
4. it should become progressively more structured until it can drive semi-automated remediation queues

This first subsystem aligns with the broader learning progression below:
1. evaluator-driven optimization first
2. offline reward modeling second
3. constrained policy learning later

It should be considered the recommended first thing to implement after the telemetry foundation is trustworthy enough to support it.

## Existing Telemetry Baseline
The current codebase already provides useful building blocks:

1. `url_scrape_metrics` in [src/db.py](/mnt/d/GitHub/social_dance_app/src/db.py)
2. `runs` in [src/db.py](/mnt/d/GitHub/social_dance_app/src/db.py)
3. validation tables such as `metric_observations`, `accuracy_replay_results`, and `validation_run_artifacts` in [src/db.py](/mnt/d/GitHub/social_dance_app/src/db.py)
4. deletion audit and `events_history` writes in [src/db.py](/mnt/d/GitHub/social_dance_app/src/db.py)
5. run scorecard artifacts described in [evaluation_framework.md](/mnt/d/GitHub/social_dance_app/documentation/Reinforcement%20Learning/evaluation_framework.md)

These are the correct foundation. The design below extends them rather than replacing them.

## Telemetry Gaps To Close

### 1. Event Attribution Gap
We currently know events were written, but not always exactly which URL, step, and decision path produced each event row.

This blocks:
1. source-quality attribution
2. per-step precision analysis
3. automated rollback of bad extractors
4. accurate reward assignment

### 2. Deletion Attribution Gap
Deleted rows are audited, but there is not yet a normalized delete-reason model strong enough for automated learning.

This blocks:
1. understanding whether a step is overproducing garbage
2. distinguishing beneficial cleanup from destructive cleanup
3. computing true reward for a scraper or prompt

### 3. Decision Boundary Gap
Branch decisions such as `should_process_url`, provider routing, and fanout limits are not fully persisted as structured features.

This blocks:
1. heuristic tuning
2. controlled experimentation
3. model-driven policy optimization

### 4. LLM Attempt Telemetry Gap
Provider attempts, retry reasons, parse failures, and latency are partly visible in logs but not fully normalized.

This blocks:
1. provider tuning
2. timeout optimization
3. cost optimization
4. prompt quality analysis

### 5. Validation Label Gap
Validation often tells us a run failed, but not always in a stable event- or URL-level label format suitable for automated learning.

This blocks:
1. failure clustering
2. reward shaping
3. defect trend analysis

### 6. Config Snapshot Gap
There is not yet a canonical persisted snapshot of all resolved runtime config and policy values for every run.

This blocks:
1. honest before/after comparisons
2. causal analysis of config changes
3. reproducibility

### 7. Human Feedback Gap
Manual review and user judgment are still mostly external to the telemetry model.

This blocks:
1. supervised correction loops
2. high-quality reward signals
3. safe agent training

## Canonical Telemetry Model
The canonical model should be organized into six layers.

### Layer 1. Run Context
Stores what code and config were active during a run.

Tables:
1. `run_context`
2. `run_policy_snapshot`

### Layer 2. URL Decision Telemetry
Stores what the system did for each URL and why.

Tables:
1. existing `url_scrape_metrics`
2. new `url_decision_events`
3. new `llm_attempt_metrics`
4. new `fetch_artifacts`

The `fetch_artifacts` layer should preserve the raw or rendered page state that parser decisions operated on.

Purpose:
1. replay parser decisions against stable local artifacts
2. distinguish fetch instability from extraction defects
3. support offline parser evaluation and future RL-style policy learning

Implementation requirement:
1. `fetch_artifacts` should be paired with extraction interfaces that can consume stored artifacts directly, not only live URLs

### Layer 3. Event Write and Delete Attribution
Stores lifecycle evidence for every event row.

Tables:
1. new `event_write_attribution`
2. new `event_delete_attribution`
3. existing `events_history` remains useful as raw history

### Layer 4. Validation Outcomes
Stores URL-level and event-level quality judgments.

Tables:
1. new `url_validation_results`
2. new `event_validation_results`
3. existing validation summary tables remain valid as aggregates

### Layer 5. Human Feedback
Stores manual review judgments and fixes.

Tables:
1. new `url_feedback`
2. new `event_feedback`
3. new `domain_feedback`

### Layer 6. Aggregated Scorecards
Stores derived performance summaries for agent reasoning and gating.

Tables or artifacts:
1. existing `metric_observations`
2. existing `validation_run_artifacts`
3. `run_scorecard.json`
4. new materialized per-run summaries if needed

## Required Schema Additions

### 1. `run_context`
This should be the canonical run snapshot table.

Suggested columns:
1. `run_id`
2. `started_at_utc`
3. `finished_at_utc`
4. `environment`
5. `git_commit`
6. `git_branch`
7. `config_hash`
8. `prompt_bundle_hash`
9. `input_snapshot_hash`
10. `resolved_config_json`
11. `provider_policy_json`
12. `seed_versions_json`
13. `runtime_flags_json`

Purpose:
1. compare runs honestly
2. reproduce runs
3. identify whether code or config caused a change

### 2. `run_policy_snapshot`
This table stores explicit policy and experiment values that are important for tuning.

Suggested columns:
1. `run_id`
2. `policy_name`
3. `policy_version`
4. `policy_value_json`
5. `experiment_id`
6. `treatment_group`

Examples:
1. `openai_exclusion_heavy_scrapers`
2. `instagram_keyword_filter`
3. `scraper_dynamic_fanout_cap`
4. `provider_rotation_policy`

Purpose:
1. support A/B comparisons
2. support controlled rollout
3. support automated recommendations

Implementation note:
1. this table should support additive policy evolution without rewriting prior rows
2. `policy_value_json` should be bounded and versioned

### 3. Extend `url_scrape_metrics`
`url_scrape_metrics` should remain the canonical per-URL scrape telemetry table, but it needs more fields.

Current strengths:
1. step attribution
2. `events_written`
3. OCR and vision signals
4. classification metadata

Add these columns:
1. `code_path`
2. `provider`
3. `model`
4. `prompt_type`
5. `schema_type`
6. `latency_ms`
7. `http_status`
8. `error_type`
9. `retry_count`
10. `policy_version`
11. `event_write_count` if we want an alias clearer than `events_written`
12. `event_delete_count` for steps that delete
13. `cost_usd_estimate`
14. `reward_label`

Purpose:
1. give Codex enough features to compare routes and policies
2. remove dependence on parsing logs

Engineering requirements:
1. add a clear uniqueness or deduplication policy
2. keep high-volume JSON payloads bounded
3. index common filter columns such as `run_id`, `handled_by`, `provider`, `decision_reason`, and `time_stamp`

### 4. `url_decision_events`
This is a normalized table for branch decisions.

Suggested columns:
1. `run_id`
2. `step_name`
3. `url`
4. `decision_stage`
5. `decision_name`
6. `decision_reason`
7. `decision_value`
8. `features_json`
9. `time_stamp`

Examples:
1. `decision_stage=should_process_url`, `decision_name=process_or_skip`
2. `decision_stage=provider_routing`, `decision_name=selected_provider`
3. `decision_stage=fanout`, `decision_name=max_links_to_follow`
4. `decision_stage=ownership`, `decision_name=owner_step`

Purpose:
1. learn which branch conditions predict good or bad outcomes
2. support heuristic tuning without scraping logs

### 5. `llm_attempt_metrics`
This should capture every LLM attempt, including failures and retries.

Suggested columns:
1. `run_id`
2. `step_name`
3. `url`
4. `parent_url`
5. `provider`
6. `model`
7. `attempt_number`
8. `prompt_type`
9. `schema_type`
10. `prompt_chars`
11. `response_chars`
12. `latency_ms`
13. `outcome`
14. `parse_success`
15. `events_written`
16. `error_type`
17. `retry_reason`
18. `cost_usd_estimate`
19. `time_stamp`

Purpose:
1. optimize provider selection
2. optimize timeout and fallback policy
3. identify prompt regressions

Engineering requirements:
1. avoid storing full prompts or full responses by default
2. store prompt hashes, bounded previews, token counts, and derived metadata unless full payload retention is explicitly required
3. separate raw payload retention from default longitudinal telemetry

### 5a. `fetch_artifacts`
This table or artifact registry should capture the frozen page state that parser decisions used.

Suggested columns:
1. `artifact_id`
2. `run_id`
3. `step_name`
4. `url`
5. `parent_url`
6. `fetch_method`
7. `artifact_type`
8. `original_url`
9. `final_url`
10. `http_status`
11. `content_type`
12. `fetched_at_utc`
13. `content_hash`
14. `storage_path`
15. `headers_json`
16. `artifact_meta_json`

`fetch_method` examples:
1. `requests`
2. `playwright_render`
3. `image_download`
4. `ocr_input`

`artifact_type` examples:
1. `raw_html`
2. `rendered_html`
3. `image_binary`
4. `ocr_text`
5. `linked_image_manifest`

Purpose:
1. reproduce parser defects against a stable local artifact
2. compare parser or routing variants against the same input
3. allow offline replay without repeated live fetches

Engineering requirements:
1. use content hashes to deduplicate identical artifacts
2. keep large binaries on disk or object storage and persist references in `storage_path`
3. capture enough metadata to reproduce how the artifact was obtained
4. support bounded retention so artifact storage does not grow without control

Persistence model:
1. Postgres is the canonical metadata registry for artifact discovery, provenance, and joins
2. the filesystem is the initial canonical payload store for heavy artifact bodies
3. `storage_path` should point to a deterministic local artifact location during the first implementation
4. object storage is a later optimization, not a prerequisite for the first rollout

### 6. `event_write_attribution`
This table is mandatory if the goal is reinforcement-style improvement.

Suggested columns:
1. `event_id`
2. `run_id`
3. `step_name`
4. `url`
5. `parent_url`
6. `source`
7. `write_method`
8. `provider`
9. `model`
10. `prompt_type`
11. `decision_reason`
12. `time_stamp`

`write_method` examples:
1. `llm_extraction`
2. `google_calendar_fetch`
3. `vision_extraction`
4. `history_reuse`
5. `checkpoint_restore`

Purpose:
1. directly attribute final rows to the step and method that created them
2. connect later validation or deletion back to the write path

Engineering requirements:
1. this table should support idempotent upsert behavior
2. attribution rows should be written from a centralized write path, not duplicated per scraper

### 7. `event_delete_attribution`
This table is the mirror image of write attribution.

Suggested columns:
1. `event_id`
2. `run_id`
3. `step_name`
4. `delete_reason_code`
5. `delete_method`
6. `source_url`
7. `created_by_step`
8. `time_stamp`
9. `details_json`

`delete_method` examples:
1. `clean_up.py`
2. `dedup_llm.py`
3. `irrelevant_rows.py`
4. `db.delete_old_events`

Purpose:
1. compute true reward
2. identify overly aggressive cleanup
3. spot noisy scrapers

Engineering requirements:
1. delete reason codes must come from a controlled vocabulary
2. delete attribution must preserve linkage back to the original writer when known

### 8. `url_validation_results`
This table stores URL-level quality judgments.

Suggested columns:
1. `run_id`
2. `step_name`
3. `url`
4. `validator_name`
5. `pass_fail`
6. `severity`
7. `reason_code`
8. `details_json`
9. `time_stamp`

Examples:
1. `missing_keywords`
2. `bad_calendar_id`
3. `private_social_content`
4. `scraped_non_event_page`

### 9. `event_validation_results`
This table stores event-level quality judgments.

Suggested columns:
1. `run_id`
2. `event_id`
3. `url`
4. `validator_name`
5. `pass_fail`
6. `severity`
7. `reason_code`
8. `field_name`
9. `expected_value`
10. `observed_value`
11. `details_json`
12. `time_stamp`

Examples:
1. `invalid_date`
2. `missing_time`
3. `outside_bc`
4. `duplicate_event`
5. `venue_closed`
6. `dance_irrelevant`

Purpose:
1. machine-readable reward shaping
2. failure clustering
3. targeted code changes

### 10. Human Feedback Tables
These are required if the system is expected to improve from reviewer knowledge.

#### `url_feedback`
1. `feedback_id`
2. `run_id`
3. `url`
4. `label`
5. `reason_code`
6. `reviewer`
7. `details_json`
8. `time_stamp`

#### `event_feedback`
1. `feedback_id`
2. `event_id`
3. `url`
4. `label`
5. `reason_code`
6. `reviewer`
7. `details_json`
8. `time_stamp`

#### `domain_feedback`
1. `feedback_id`
2. `domain`
3. `label`
4. `reason_code`
5. `reviewer`
6. `details_json`
7. `time_stamp`

Example labels:
1. `good_source`
2. `bad_source`
3. `invalid_url`
4. `high_priority`
5. `needs_custom_prompt`
6. `needs_blacklist`

## Reward Model
The system does not need online RL first. It needs an explicit reward model that can later support RL or bandit optimization.

### Positive Reward Signals
1. event written and survives cleanup
2. event passes validation
3. event matches replay or manual truth set
4. event comes from a priority source
5. write has low latency and low cost relative to value

### Negative Reward Signals
1. event later deleted as invalid
2. event later deleted as duplicate
3. URL consumed time/cost with zero useful output
4. provider attempt timed out or caused cooldown
5. step caused validation failures
6. manual reviewer marked URL or event as wrong

### Reward Assignment Rule
Reward must be attributable to:
1. URL
2. step
3. provider/model
4. prompt or routing policy
5. domain/source

That is why write attribution and delete attribution are non-negotiable.

## Reinforcement Learning Readiness
This section distinguishes between:
1. telemetry that is sufficient for scorecards and heuristic tuning
2. telemetry that is sufficient for offline policy learning or contextual bandits
3. telemetry that would be required before any credible move toward full RL

The distinction matters. Production systems often fail by calling a telemetry stack "RL-ready" when it only supports after-the-fact reporting.

### What RL-Like Methods Require
For policy learning from logged data, authoritative work on logged bandit feedback and offline policy learning requires that each logged interaction preserve at least:
1. context or state
2. action taken
3. reward or outcome
4. logging or behavior policy information, especially propensity when counterfactual evaluation is desired
5. enough temporal structure to connect delayed outcomes back to the original action

That requirement is consistent with logged bandit and counterfactual learning literature:
1. Swaminathan and Joachims frame batch learning from logged bandit feedback around predictions made for an input and observed feedback for the chosen action, with propensity scoring for counterfactual learning.
2. More recent offline policy learning work still assumes logged datasets contain context, action, propensity score, and feedback for each sample.

### What This Means For DanceScoop
If the system wants to use RL-like optimization for routing, provider selection, fanout, or scrape/skip decisions, then each decision must be loggable as a policy interaction.

At minimum, each interaction needs:
1. `state_features`
2. `action_taken`
3. `behavior_policy_id`
4. `action_probability` or `selection_score` if stochastic selection or policy evaluation is desired
5. `immediate_reward`
6. `delayed_reward`
7. `reward_observation_window`
8. `next_state_features` when the decision affects a later state in a trajectory
9. `terminal_flag` when the episode or sub-episode ends

Without that, the system can still optimize heuristics, but it cannot honestly claim reinforcement-learning-ready data collection.

## RL-Specific Telemetry Requirements

### 1. State Logging
Each optimization-relevant decision must record the state that was visible at decision time.

Examples:
1. URL features before `should_process_url`
2. domain history before choosing provider routing
3. page archetype, confidence, and fanout candidates before selecting crawl depth
4. OCR quality and keyword signals before choosing OCR-only vs vision fallback

Suggested fields:
1. `state_features_json`
2. `state_hash`
3. `state_version`

Purpose:
1. allow offline learning from historical decisions
2. prevent training-serving skew between optimization and live policy execution

### 2. Action Logging
The chosen action must be explicit and enumerable.

Examples of actions:
1. `provider=openai`
2. `provider=openrouter`
3. `fanout_cap=3`
4. `fanout_cap=10`
5. `process_url=true`
6. `process_url=false`
7. `route_to=fb.py`
8. `vision_fallback=true`

Suggested fields:
1. `action_type`
2. `action_value`
3. `action_features_json`

Purpose:
1. support policy comparison
2. allow action-conditional reward analysis

### 3. Behavior Policy Logging
If counterfactual evaluation or contextual bandits are a goal, the system must log the behavior policy that produced the action.

Suggested fields:
1. `behavior_policy_id`
2. `behavior_policy_version`
3. `action_probability`
4. `action_score`
5. `action_sampling_method`

Notes:
1. if actions are deterministic, true inverse-propensity-based counterfactual evaluation becomes much weaker
2. if stochastic exploration is introduced later, action propensities become mandatory

Purpose:
1. support off-policy evaluation
2. support counterfactual policy optimization

### 4. Reward Logging
Rewards should be decomposed rather than stored only as one blended scalar.

Suggested fields:
1. `reward_accuracy`
2. `reward_coverage`
3. `reward_runtime`
4. `reward_cost`
5. `reward_total`
6. `reward_version`

Notes:
1. the reward contract must be versioned
2. scalar rewards should be derivable from component rewards, not the other way around

Purpose:
1. allow changing KPI weights without losing raw signal
2. make reward shaping auditable

### 5. Delayed Reward and Credit Assignment
For this application, many rewards are delayed.

Examples:
1. an event is written now and deleted later
2. a URL appears successful now but later fails validation
3. a source appears low yield in one run but contributes to coverage over multiple runs

Required linkage:
1. `decision_id`
2. `reward_event_id`
3. `reward_observed_at`
4. `reward_window_start`
5. `reward_window_end`
6. `credit_assignment_method`

Purpose:
1. connect late outcomes to the original decisions that caused them
2. prevent false reward assignment to later cleanup steps alone

### 6. Trajectory or Episode Structure
Not every optimization problem here is a full Markov decision process, but some are multi-step.

Examples:
1. page classification -> provider selection -> extraction -> follow links -> validation outcome
2. image OCR -> keyword decision -> prompt selection -> LLM extraction -> vision fallback

Suggested fields:
1. `episode_id`
2. `parent_decision_id`
3. `step_index`
4. `next_decision_id`
5. `terminal_flag`

Purpose:
1. support sequential analysis where one action changes later opportunities
2. separate simple bandit-style decisions from multi-step flows

### 7. Candidate Action Set Logging
For optimization quality, it is useful to know which alternatives were available.

Examples:
1. available providers at decision time
2. candidate fanout caps
3. whether a URL could have been routed to multiple steps
4. whether vision fallback was available

Suggested fields:
1. `candidate_actions_json`
2. `action_constraints_json`

Purpose:
1. understand whether the chosen action was a real choice or the only feasible action
2. support constrained policy learning

### 8. Non-Stationarity Logging
Offline RL and logged-feedback systems degrade when the environment changes and the log does not capture that change.

Relevant non-stationarity in DanceScoop:
1. websites change structure
2. providers change latency and reliability
3. prompts change
4. seasonal event patterns change
5. source lists and whitelists change

Required fields:
1. `environment_snapshot_id`
2. `source_structure_signature`
3. `provider_health_snapshot_id`
4. `policy_snapshot_id`

Purpose:
1. prevent mixing incompatible historical data without context
2. support segmented offline evaluation

## KPI Readiness For RL Use
The target KPIs are:
1. event coverage
2. database accuracy
3. pipeline runtime
4. LLM cost

These are not equally ready for RL-style optimization.

### 1. Runtime
Status:
1. closest to reward-ready

Why:
1. immediately observable
2. already measurable per run and per step
3. directly attributable to many decision paths

### 2. Cost
Status:
1. close to reward-ready once provider/model/attempt costs are fully logged

Why:
1. measurable
2. attributable to providers, models, retries, and steps

### 3. Database Accuracy
Status:
1. medium-term reward-ready after event-level validation and delete attribution are complete

Why:
1. needs delayed outcome linkage
2. needs stable event-level validation labels
3. needs attribution back to the originating write path

### 4. Event Coverage
Status:
1. least ready for RL use

Why:
1. coverage is only partially observable from the system itself
2. it requires external expectation sets, watchlists, or manual audits
3. coverage reward is sparse and noisy compared with runtime and cost

Conclusion:
1. runtime and cost can support earlier policy optimization
2. accuracy can follow after attribution and validation work
3. coverage should be treated as a higher-latency supervised or evaluation signal before it is treated as a primary RL reward

## Recommended Learning Progression
The recommended progression is:

The first applied instance of this progression should be the page-parsing and parser-routing subsystem described above.

### Stage 1. Evaluator-Driven Optimization
Use telemetry and scorecards to improve heuristics and code paths manually or with Codex assistance.

### Stage 2. Offline Reward Modeling
Use the collected telemetry to estimate which decisions tend to improve runtime, cost, and accuracy.

### Stage 3. Contextual Bandits For Narrow Decisions
Apply bandit-style optimization to constrained choices such as:
1. provider selection
2. fanout caps
3. scrape vs skip thresholds
4. vision fallback invocation

### Stage 4. Delayed-Reward Policy Learning
Only after decision, attribution, and delayed reward linkage are stable should the system attempt richer RL-style policy learning for multi-step flows.

This staged path is safer and much more realistic than jumping directly to full RL.

## What Codex Needs To Operate Autonomously
To allow Codex to generate reports and automatically patch the codebase, the telemetry must support the following workflows.

### 1. Diagnostic Reporting
Codex must be able to query:
1. per-step event yield
2. per-domain event yield
3. provider success and timeout rates
4. validation failure clusters
5. cleanup delete clusters
6. top runtime bottlenecks

### 2. Change Recommendation
Codex must be able to produce evidence-based recommendations like:
1. exclude OpenAI for `scraper` but keep it for `gs`
2. reduce Instagram keyword fanout
3. raise `should_process_url` threshold for low-value domains
4. add domain-specific prompt for a source with high parse success but poor field accuracy

### 3. Safe Auto-Modification
Codex must only patch the code if:
1. the proposed change targets a known failure cluster
2. there is a measurable expected improvement
3. the change is narrow and reversible
4. post-change guardrails are checked

### 4. Rollback Support
Telemetry must support:
1. diff vs previous run
2. diff vs baseline cohort
3. guardrail regressions
4. automatic disable of bad policy changes

## Reporting Layer
Each run should materialize a machine-readable report suitable for both humans and agents.

Minimum report sections:
1. run context
2. step summaries
3. provider summaries
4. domain summaries
5. validation failures by reason
6. deletions by reason
7. top regressions vs previous run
8. top improvements vs previous run
9. candidate auto-fixes

Recommended artifacts:
1. `output/run_scorecard.json`
2. `output/reinforcement_learning_summary.json`
3. `output/auto_fix_candidates.json`

`auto_fix_candidates.json` should be explicit and narrow. Example payload:

```json
{
  "run_id": "20260322-192053-1f6b80dc",
  "candidates": [
    {
      "candidate_id": "scraper-low-value-vaballet-fanout",
      "problem_type": "runtime_waste",
      "scope": "domain",
      "domain": "vaballet.ca",
      "evidence": {
        "urls_processed": 22,
        "events_written": 0,
        "avg_latency_ms": 14320,
        "validation_failures": 0
      },
      "recommended_action": "reduce_fanout",
      "confidence": 0.91
    }
  ]
}
```

This artifact is the bridge between telemetry and automated patch planning.

## Required Code Changes
The telemetry design implies the following implementation changes.

### 1. Centralize Event Write Attribution
Modify `write_events_to_db` or a wrapper around it so every write can emit:
1. `event_write_attribution`
2. optional `url_scrape_metrics` update
3. source step and method metadata

This should be the single shared path rather than repeated scraper-local logic.

### 2. Centralize Event Delete Attribution
All deletion helpers in `db.py`, `clean_up.py`, `dedup_llm.py`, and `irrelevant_rows.py` should emit normalized delete records with reason codes.

### 3. Normalize Reason Codes
All major code paths should use stable reason codes, not only free-form strings.

Examples:
1. `no_keywords`
2. `llm_no_events`
3. `prompt_overflow`
4. `history_reuse`
5. `skip_low_value_path_after_irrelevant`
6. `private_or_unavailable_content`

These codes are essential for clustering and reporting.

### 4. Expand LLM Attempt Writes
`LLMHandler` should emit per-attempt metrics for all providers and all retries, including parse retries, timeouts, and hard failures.

### 5. Snapshot Resolved Runtime Config
`pipeline.py` already writes run-specific config files. That should also be persisted into `run_context.resolved_config_json` with a stable hash.

### 6. Add Human Review Ingestion
Provide a small mechanism for recording reviewer labels against URLs and events so they become persistent learning signals.

### 7. Add Telemetry Integrity Validation
Add explicit validation for telemetry itself.

Required:
1. reconcile `url_scrape_metrics`, attribution tables, and scorecard summaries
2. detect missing required fields in canonical telemetry tables
3. detect unknown reason codes and contract drift
4. fail the validation step when canonical telemetry becomes internally inconsistent

### 8. Add Retention and Archival Jobs
Define scheduled maintenance for telemetry tables.

Required:
1. retention windows by table type
2. compaction or archival for high-volume attempt telemetry
3. safe deletion policy for raw payloads after summary retention windows expire

## Implementation Phases

### Phase 1. Canonical Counting and Attribution
Goal:
1. reliable per-step counts
2. reliable event-write attribution

Tasks:
1. complete `url_scrape_metrics` normalization across all scrapers
2. add `event_write_attribution`
3. add `event_delete_attribution`
4. normalize reason codes
5. add telemetry integrity checks for the canonical counting path

Deliverables:
1. migrations for attribution tables and any required `url_scrape_metrics` extensions
2. centralized event write attribution path in application code
3. centralized event delete attribution path in application code
4. telemetry integrity report section in validation artifacts
5. first reconciliation query pack for per-step counts vs attribution totals

Validation:
1. per-step counts reconcile across canonical tables for at least 3 full runs
2. unknown reason codes fail validation
3. duplicate attribution rows do not appear under retry/restart scenarios

Exit criteria:
1. `events_urls_diff.csv` is no longer needed as the canonical source for per-step event counts
2. all scraper steps write trustworthy event-count telemetry
3. all delete paths produce normalized delete attribution

Current status:
1. The Phase 1 code implementation is substantially complete in the repository:
   `event_write_attribution` and `event_delete_attribution` exist, centralized write/delete attribution paths are implemented, delete reason normalization is in place, the telemetry integrity report exists, and telemetry integrity is wired into validation guardrails.
2. Phase 1 is not yet operationally complete. Real `pipeline.py` runs are still required to prove the validation and exit criteria.
3. The remaining work for Phase 1 is run-based verification:
   `per-step counts reconcile across canonical tables for at least 3 full runs`, duplicate attribution rows do not appear under retry/restart scenarios in practice, all scraper steps show trustworthy live event-count telemetry, and only then should `events_urls_diff.csv` stop being treated as the canonical source.
4. Until those runs have been reviewed and pass cleanly, Phase 1 should be treated as `implemented, pending operational verification`, not `fully complete`.

### Phase 2. Decision and LLM Telemetry
Goal:
1. enough evidence to tune routing and heuristics automatically

Tasks:
1. add `url_decision_events`
2. add `llm_attempt_metrics`
3. add `fetch_artifacts` capture for selected parser-learning URLs
4. add cost and latency fields
5. add policy snapshots
6. define retention and preview/redaction rules for raw prompt/response telemetry
7. add at least one offline replay runner that consumes stored artifacts through shared extraction interfaces

Deliverables:
1. migrations for `url_decision_events`, `llm_attempt_metrics`, `fetch_artifacts`, `run_context`, and `run_policy_snapshot`
2. provider-attempt logging in `LLMHandler`
3. decision logging for `should_process_url`, provider routing, ownership routing, and fanout
4. initial artifact storage convention and capture path for selected URLs
5. one bounded offline replay path for a selected scraper family
6. persisted resolved runtime config snapshot per run
7. retention and redaction policy document or section linked from implementation notes

Validation:
1. provider-attempt counts reconcile with run summaries
2. latency and cost fields are populated for the majority of LLM attempts
3. decision rows exist for the major branch points in `scraper.py`, `images.py`, `fb.py`, and `rd_ext.py`
4. at least one parser defect can be reproduced locally from a stored artifact

Exit criteria:
1. Codex can query provider/model outcomes without parsing text logs
2. major routing and fanout decisions are recoverable from structured telemetry
3. each run has a stable config and policy snapshot
4. at least one scraper family supports offline artifact replay using shared extraction logic

Current status:
1. structural replay refactoring has started and now supports a normalized artifact boundary in the validation/replay layer
2. this partially advances Deliverable 5 and Exit criterion 4 at the code-structure level, but not at the operational level
3. `fetch_artifacts` persistence, artifact storage conventions, and stored-artifact replay are still not implemented end-to-end
4. because Phase 1 is still pending operational verification, Phase 2 should be treated as `early architectural preparation`, not `in implementation completion`

### Phase 3. Validation Labels and Reward Signals
Goal:
1. create machine-readable reward signals

Tasks:
1. add `url_validation_results`
2. add `event_validation_results`
3. connect deletion and validation outcomes to writes
4. compute run-level reward summaries
5. add reward contract versioning

Deliverables:
1. migrations for validation result tables
2. reward computation module or pipeline step
3. reward contract definition with versioned KPI weights
4. run artifact containing decomposed reward components
5. linkage from validation and delete outcomes back to `decision_id` or attribution rows

Validation:
1. event validation rows can be joined back to original write attribution
2. reward component totals reconcile with validation and delete evidence
3. reward version is persisted and queryable

Exit criteria:
1. runtime, cost, and accuracy rewards are machine-readable and attributable
2. delayed negative outcomes such as later delete/validation failures can be assigned back to origin decisions
3. reward summaries are stable enough for offline policy analysis

### Phase 4. Human Feedback Loop
Goal:
1. incorporate reviewer judgment

Tasks:
1. add feedback tables
2. add import flow for manual feedback
3. add reports that merge manual and automated signals
4. define reviewer workflow and label governance

Deliverables:
1. migrations for `url_feedback`, `event_feedback`, and `domain_feedback`
2. a lightweight ingestion path for reviewer labels
3. reviewer label dictionary and governance rules
4. merged reporting that distinguishes manual vs automated signals

Validation:
1. manual feedback can be linked to runs, URLs, and events
2. reviewer labels are validated against an approved vocabulary
3. reports show how manual feedback changes reward interpretation

Exit criteria:
1. high-value manual corrections become queryable training signals
2. feedback ingestion is simple enough to be used consistently
3. reviewer labels can be incorporated into scorecards and auto-fix candidate generation

### Phase 5. Safe Auto-Fix Loop
Goal:
1. allow Codex to generate and apply narrow patches automatically

Tasks:
1. generate `auto_fix_candidates.json`
2. add promotion and rollback guardrails
3. define policy for auto-apply vs review-required changes
4. require before/after scorecard comparison before auto-merge behavior is considered

Deliverables:
1. `auto_fix_candidates.json`
2. guardrail policy for auto-fix eligibility
3. before/after comparison report template
4. rollback rule set and failure thresholds
5. audit trail for generated and applied code changes

Validation:
1. every candidate fix includes evidence, expected metric improvement, and rollback condition
2. no auto-fix is eligible without passing telemetry integrity checks
3. regression detection blocks auto-apply when guardrails fail

Exit criteria:
1. Codex can propose narrow fixes from structured evidence
2. every applied change is auditable and reversible
3. auto-fix behavior remains gated behind measured scorecard improvements

## Phase Dependencies
The phases are intentionally ordered and should not be collapsed.

Dependencies:
1. Phase 2 depends on Phase 1 because decision telemetry is not useful if counts and attribution are untrustworthy.
2. Phase 3 depends on Phases 1 and 2 because reward needs both trustworthy attribution and explicit decision context.
3. Phase 4 depends on Phase 3 because manual feedback should join onto validated event and URL records, not float independently.
4. Phase 5 depends on all earlier phases because safe automation requires trustworthy telemetry, reward interpretation, and guardrails.

Recommended sequencing rule:
1. do not start auto-fix implementation before at least one full cycle of telemetry integrity, validation labeling, and reward reporting has been proven stable across multiple runs.

## Phase Kickoff Checklist
This document is sufficient to begin implementation, but each phase should still start with a short phase-specific implementation spec.

Before starting any phase, define:
1. exact SQL DDL for new or changed tables
2. exact uniqueness and upsert rules
3. exact reason-code registry additions or changes
4. exact retention and archival values for any new telemetry
5. exact ownership of canonical tables, reports, and validation checks
6. exact test plan for the phase
7. exact rollback or cutover plan if the phase changes canonical reporting

Output of the kickoff checklist:
1. a small implementation note or engineering ticket set
2. migration sequence
3. validation queries
4. success criteria tied to the phase exit criteria already defined above

Use this rule when revisiting the document later:
1. the design doc defines what must exist and in what order
2. the phase-specific implementation spec defines exactly how the next phase will be built

Recommended next step from the current state:
1. start Phase 1 with a concrete implementation spec for `event_write_attribution`, `event_delete_attribution`, `url_scrape_metrics` extensions, telemetry integrity validation, and the controlled reason-code registry

## Safety Requirements For Self-Healing
This system must not auto-patch the codebase without strong safeguards.

Required guardrails:
1. no auto-change without persisted evidence from at least one full run
2. no auto-change that touches more than a narrow scope unless explicitly approved
3. no auto-change if validation or scorecard guardrails are failing globally
4. every auto-change must record:
   1. what evidence triggered it
   2. what files changed
   3. what metrics it expected to improve
   4. what rollback condition applies

This is not optional. A self-healing system without rollback discipline is a self-corrupting system.

## Immediate Recommendations
The highest-value next telemetry work is:

1. make `event_write_attribution` and `event_delete_attribution`
2. add `llm_attempt_metrics`
3. persist `run_context` with resolved config and git hash
4. add normalized validation result tables
5. normalize reason codes across all major decision points
6. emit `auto_fix_candidates.json` from scorecard generation

If only one thing is done next, it should be event lifecycle attribution. Without that, there is no honest reward signal.

## Summary
To make DanceScoop a self-healing, reinforcement-oriented application, we do not primarily need more text logs. We need a telemetry model that preserves:

1. run context
2. decision boundaries
3. provider attempts
4. event write attribution
5. event delete attribution
6. validation outcomes
7. human feedback
8. stable scorecards and guardrails

Once those exist, Codex can move from reactive debugging to evidence-based automatic improvement.

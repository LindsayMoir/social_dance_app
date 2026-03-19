# Evaluation Framework

## Purpose
Define a stable, honest evaluation framework for DanceScoop that optimizes the four core KPIs without allowing the system to improve one metric by silently damaging another.

The four KPIs are:
1. Database Accuracy
2. Events Coverage
3. Run Time
4. Run Costs

This framework is intended to support:
1. Human review
2. Codex-assisted analysis and planning
3. Controlled iterative improvement
4. Promotion gating

It is not a reinforcement learning design. It is an evaluator-driven improvement loop with explicit scorecards, guardrails, and anti-cheating controls.

## Problem Statement
DanceScoop is a heterogeneous scraping and extraction system. Different sources behave differently:
1. Some URLs are single-event detail pages.
2. Some are listing pages.
3. Some are route-through pages that require click expansion.
4. Some are social-platform pages with unstable structure.
5. Some are repeated templates that benefit from historical routing memory.

Because of that heterogeneity, overall pipeline success cannot be judged honestly with one metric alone.

Examples of cheating or false confidence:
1. Replay row accuracy can look good while duplicate rates remain high.
2. Runtime can improve because the system skipped hard pages.
3. Cost can improve because extraction quality silently regressed.
4. Coverage can appear stable while priority community sources are missed.
5. A classifier can appear accurate if evaluation is dominated by easy deterministic URLs.

The framework must prevent those outcomes.

## Evaluation Principles
1. JSON-first observability: machine-readable artifacts are the source of truth.
2. HTML is for review, not for primary computation.
3. No single KPI is allowed to hide regressions in another KPI.
4. Duplicates must be measured explicitly, not inferred from replay accuracy.
5. Coverage must be measured against an external expectation set.
6. Training, tuning, and evaluation data must remain separated.
7. Holdout evaluation must exist for honest comparison over time.
8. Faster or cheaper is not a win if accuracy or coverage materially regress.

## KPI Definitions

### 1. Database Accuracy
Database Accuracy answers:
"How correct is the data currently stored in the events database?"

This KPI must include:
1. Correct dates
2. Correct times
3. Correct locations
4. Correct address resolution
5. Correct dance styles
6. Correct descriptions
7. Duplicate control
8. Invalid-event control
9. Stale-event control

Database Accuracy must not be represented as a single replay percentage alone.

Recommended submetrics:
1. `replay_url_accuracy_pct`
2. `replay_row_accuracy_pct`
3. `field_accuracy_date_pct`
4. `field_accuracy_time_pct`
5. `field_accuracy_location_pct`
6. `field_accuracy_address_id_pct`
7. `field_accuracy_dance_style_pct`
8. `field_accuracy_description_pct`
9. `duplicate_rate_per_100_events`
10. `severe_duplicate_rate_per_100_events`
11. `invalid_event_rate_pct`
12. `stale_event_rate_pct`

Recommended interpretation:
1. Replay metrics measure extraction correctness on sampled truth cases.
2. Field-level metrics identify where correctness is failing.
3. Duplicate metrics measure database integrity not captured by replay.
4. Invalid/stale metrics prevent silent accumulation of bad records.

### 2. Events Coverage
Events Coverage answers:
"How completely does DanceScoop capture relevant events in the Greater Victoria social dance community?"

Coverage cannot be measured solely from events already inside the database. It requires an external expectation set.

Recommended submetrics:
1. `watchlist_source_hit_rate_pct`
2. `watchlist_event_capture_rate_pct`
3. `priority_source_hit_rate_pct`
4. `new_source_discovery_count`
5. `missed_event_rate_manual_audit_pct`

Coverage sources should include:
1. Curated venue sites
2. Community calendars
3. Facebook event sources
4. Eventbrite sources
5. Other recurring local social dance sources

Recommended interpretation:
1. Watchlist metrics measure whether important sources are being represented.
2. Manual audit metrics estimate misses that automation does not detect.
3. Coverage is a separate KPI from accuracy; getting a smaller set of events perfectly correct is not full success.

### 3. Run Time
Run Time answers:
"How long does a full pipeline run take, and where is the time spent?"

Recommended submetrics:
1. `pipeline_duration_minutes`
2. `critical_path_minutes`
3. `parallel_crawlers_enabled`
4. `urls_processed_per_minute`
5. `events_inserted_per_minute`
6. Step durations for:
   - `emails`
   - `gs`
   - `rd_ext`
   - `ebs`
   - `scraper`
   - `fb`
   - `images`
   - `validation`

Recommended interpretation:
1. Total duration matters.
2. Step duration trends matter because they identify bottlenecks.
3. Throughput matters because absolute runtime can vary by input volume.

### 4. Run Costs
Run Costs answers:
"How much does each pipeline run cost in LLM usage, and where are those costs incurred?"

Recommended submetrics:
1. `total_usd`
2. `cost_per_processed_url_usd`
3. `cost_per_inserted_event_usd`
4. Cost by provider
5. Cost by model
6. Cost by step
7. Retry/fallback cost pressure

Recommended interpretation:
1. Total run cost is useful for budgeting.
2. Cost per URL and cost per inserted event make runs comparable over time.
3. Step-level and provider-level costs identify where optimization will matter most.

## KPI Weighting
The framework should compute both:
1. Individual KPI scores
2. A weighted overall score

Recommended initial weights:
1. Database Accuracy: `0.45`
2. Events Coverage: `0.30`
3. Run Time: `0.15`
4. Run Costs: `0.10`

Rationale:
1. Accuracy is the most important KPI.
2. Coverage is second because missing major community events undermines the product.
3. Runtime matters operationally but should not dominate product quality.
4. Costs matter, but they should not be optimized ahead of correctness and coverage.

These weights can change, but changes should be explicit and versioned.

## Required Artifact Set
Each pipeline run should produce a stable set of machine-readable artifacts.

Canonical artifact:
1. `output/run_scorecard.json`

Supporting artifacts:
1. `output/accuracy_replay_summary.json`
2. `output/classifier_performance_summary.json`
3. `output/duplicate_audit_summary.json`
4. `output/coverage_summary.json`
5. `output/runtime_summary.json`
6. `output/llm_cost_summary.json`
7. `output/classifier_training_queue.json`
8. `output/classifier_review_queue.json`
9. `output/comprehensive_test_report.html`

Operational rule:
1. JSON artifacts are the source of truth.
2. HTML is a human-facing projection of JSON and logs.

## Persistence Requirements
Evaluation data must be persisted, not just written to transient files.

Required persistence targets:
1. Run-level metrics in the existing metrics/telemetry tables
2. Row-level replay audit results
3. URL-level classifier telemetry
4. URL-level training candidate queue rows
5. Historical routing-memory evidence

Persistence rules:
1. KPI trend graphs must be driven from persisted data, not reconstructed ad hoc from local files.
2. URL-level classifier decisions should include stage, confidence, owner step, subtype, and feature payload.
3. Historical routing memory must be based on persisted replay-backed outcomes, not only repeated prior choices.
4. Files in `output/` are review artifacts; the database is the longitudinal record.

## Canonical Scorecard Schema
`run_scorecard.json` should contain the minimum structure below.

```json
{
  "run_id": "20260318-101855-3d22d728",
  "run_timestamp_utc": "2026-03-18T17:25:10Z",
  "code_version": {
    "git_commit": "abc1234",
    "branch": "main"
  },
  "evaluation_scope": {
    "environment": "dev",
    "uses_holdout": true,
    "holdout_version": "v1",
    "watchlist_version": "v1"
  },
  "kpis": {
    "database_accuracy": {},
    "events_coverage": {},
    "run_time": {},
    "run_costs": {}
  },
  "guardrails": {},
  "overall_score": {},
  "recommendations_input": {}
}
```

The `recommendations_input` section should point Codex toward:
1. Top regressions
2. Largest cost deltas
3. Runtime bottlenecks
4. Duplicate hotspots
5. Coverage gaps
6. Classifier routing regressions

## Example KPI Structures

### Database Accuracy Example
```json
{
  "score": 81.4,
  "weight": 0.45,
  "summary": {
    "replay_url_accuracy_pct": 76.2,
    "replay_row_accuracy_pct": 72.8,
    "duplicate_rate_per_100_events": 6.4,
    "severe_duplicate_rate_per_100_events": 1.2,
    "invalid_event_rate_pct": 3.1,
    "stale_event_rate_pct": 2.4
  },
  "field_accuracy": {
    "date_pct": 91.0,
    "time_pct": 84.5,
    "location_pct": 80.2,
    "address_id_pct": 88.1,
    "dance_style_pct": 74.0,
    "description_pct": 69.5
  },
  "classifier_effect": {
    "rule_replay_url_accuracy_pct": 83.0,
    "ml_replay_url_accuracy_pct": 68.0,
    "memory_replay_url_accuracy_pct": 86.0,
    "ml_usage_pct": 19.0
  },
  "sample_sizes": {
    "replay_urls": 21,
    "replay_rows": 53,
    "duplicate_audit_events": 410
  }
}
```

### Events Coverage Example
```json
{
  "score": 73.6,
  "weight": 0.30,
  "summary": {
    "watchlist_source_hit_rate_pct": 88.0,
    "watchlist_event_capture_rate_pct": 71.0,
    "priority_source_hit_rate_pct": 95.0,
    "new_source_discovery_count": 4,
    "missed_event_rate_manual_audit_pct": 18.0
  },
  "watchlist": {
    "sources_total": 75,
    "sources_hit": 66,
    "priority_sources_total": 20,
    "priority_sources_hit": 19
  },
  "manual_audit": {
    "sample_size": 22,
    "missed_events": 4
  }
}
```

### Run Time Example
```json
{
  "score": 87.2,
  "weight": 0.15,
  "summary": {
    "pipeline_duration_minutes": 221.4,
    "critical_path_minutes": 198.2,
    "parallel_crawlers_enabled": true
  },
  "steps": {
    "emails_minutes": 8.5,
    "gs_minutes": 12.0,
    "rd_ext_minutes": 64.0,
    "ebs_minutes": 25.0,
    "scraper_minutes": 79.0,
    "fb_minutes": 43.0,
    "validation_minutes": 18.0
  },
  "throughput": {
    "urls_processed_per_minute": 14.8,
    "events_inserted_per_minute": 3.1
  }
}
```

### Run Costs Example
```json
{
  "score": 78.5,
  "weight": 0.10,
  "summary": {
    "total_usd": 11.84,
    "cost_per_processed_url_usd": 0.021,
    "cost_per_inserted_event_usd": 0.049
  },
  "by_provider": {
    "openai_usd": 8.14,
    "anthropic_usd": 3.70
  },
  "by_model": {
    "gpt_5_mini_usd": 4.91,
    "gpt_5_usd": 3.23,
    "claude_usd": 3.70
  },
  "by_step": {
    "scraper_usd": 5.10,
    "fb_usd": 2.65,
    "read_pdfs_usd": 1.20
  }
}
```

## Anti-Cheating Controls
This section is mandatory. Without it, the system will optimize toward misleading improvements.

### 1. Separate Train, Dev, and Holdout
Maintain three evaluation pools:
1. `train`
2. `dev`
3. `gold_holdout`

Rules:
1. `train` may be used for classifier training and prompt-tuning.
2. `dev` may be used for regular iteration.
3. `gold_holdout` must never be used for model training, tuning, or routing-memory seeding.
4. Promotions should be judged against both `dev` and `gold_holdout`.

Recommended additional separation:
1. `review_queue` for ambiguous or suspicious cases that require human labeling
2. `auto_positive_queue` for conservative URL-level promotions that are safe to add automatically

### 2. Evaluate at URL Level
Many URLs generate multiple rows. Row-level metrics overweight repeated templates.

Rules:
1. Primary classifier and replay metrics should be URL-level.
2. Row-level metrics should still be tracked, but not treated as the only truth.
3. One URL should not dominate performance reporting simply because it emitted many rows.
4. Training examples should be added at one row per normalized URL, not one row per emitted event record.
5. Domain/template caps should be applied during both promotion and evaluation summarization.

### 2a. URL-Level Training Promotion Rules
Promotion into the classifier training set must be conservative.

Rules:
1. Replay output should be aggregated into one candidate per normalized URL.
2. A URL should not be auto-promoted merely because some replay rows were `True`.
3. `True` replay rows are evidence for candidate correctness.
4. `False` replay rows are review signals first, not automatic negative training labels.
5. Auto-promotion should be limited to safe `auto_positive_candidate` URLs.
6. `classifier_review_needed` URLs should go into a separate review queue, not directly into the training set.
7. Training-set balance should be enforced by class and capped by domain/template.
8. After promotion adds new rows, the page-classifier model should be retrained automatically.

### 3. Measure Duplicates Explicitly
Replay matching does not fully capture duplicate problems.

Rules:
1. Duplicate rate must be computed independently of replay metrics.
2. Severe duplicates and harmless duplicates should be separated.
3. Duplicate metrics must influence `database_accuracy`.

### 4. Use External Coverage Expectations
Coverage cannot be scored from internal outputs alone.

Rules:
1. Maintain a curated source watchlist.
2. Maintain a periodic manual audit set.
3. Treat watchlist misses as coverage failures even if the database otherwise looks internally consistent.

### 4a. Distinguish Classifier Failures From Extractor Failures
Not every replay failure means the page classifier was wrong.

Rules:
1. `no_event_extracted_replay` often indicates a routing or expansion problem and should be reviewed for classifier impact.
2. `wrong_date`, `wrong_time`, `wrong_location_or_address`, and similar mismatches often indicate extraction quality problems after routing.
3. Duplicate identity failures are primarily data integrity problems, not classifier labels.
4. Classifier training rows should only be created from reviewed truth, not from raw replay failure categories alone.

### 5. Enforce Guardrails
No change should count as an improvement if it violates hard quality thresholds.

Example guardrails:
1. `database_accuracy_min_pct = 75.0`
2. `events_coverage_min_pct = 65.0`
3. `severe_duplicate_rate_max_per_100_events = 2.0`
4. `stale_event_rate_max_pct = 5.0`

### 6. Cap Domain Influence
Some domains recur frequently and can distort both training and evaluation.

Rules:
1. Cap training examples per domain/template.
2. Publish domain-level evaluation summaries.
3. Use domain-diverse holdout sets.

## Coverage Watchlist Design
Create a curated watchlist file:
1. `data/watchlists/coverage_watchlist.csv`

Recommended columns:
1. `source_name`
2. `source_url`
3. `source_type`
4. `priority`
5. `expected_frequency`
6. `coverage_region`
7. `active`

Use cases:
1. Coverage scoring
2. Priority source monitoring
3. Gap analysis
4. New source discovery comparison

Recommended watchlist operating rules:
1. The watchlist should include a mix of easy deterministic sources and hard long-tail sources.
2. Priority sources should be explicitly flagged and tracked separately.
3. Watchlist changes must be versioned because they change KPI interpretation over time.

## Duplicate Audit Design
Create a duplicate summary artifact:
1. `output/duplicate_audit_summary.json`

Recommended content:
1. `duplicate_clusters_count`
2. `duplicate_rate_per_100_events`
3. `severe_duplicate_rate_per_100_events`
4. `top_duplicate_patterns`
5. `top_duplicate_domains`
6. `sample_duplicate_clusters`

Recommended duplicate categories:
1. Exact duplicate
2. Same event, minor formatting variance
3. Same source repeated across crawlers
4. Same event under multiple URLs
5. Harmless repeat vs harmful duplicate

## Classifier Evaluation Requirements
The page classifier must be evaluated honestly and separately from overall pipeline success.

Required classifier metrics:
1. `ml_usage_pct`
2. `rule_replay_url_accuracy_pct`
3. `ml_replay_url_accuracy_pct`
4. `memory_replay_url_accuracy_pct`
5. Confidence distribution by stage
6. Stage usage counts
7. Stage accuracy by domain

Required classifier principles:
1. Deterministic rules remain first for obvious cases.
2. Runtime routing order should be:
   - deterministic rules
   - exact-URL replay-backed memory
   - ML for ambiguous web pages
   - heuristic fallback
3. Exact-URL routing memory should only activate when backed by successful replay history.
4. ML should be evaluated primarily on ambiguous cases, not just across easy deterministic traffic.
5. Email-address inputs are not part of the web-page classifier problem.
6. Email-address inputs should be routed deterministically and excluded from the web-page ML training problem.
7. Deterministic platform cases such as obvious Google Calendar, Eventbrite, and social-platform detail pages should continue to short-circuit where appropriate.

Recommended classifier telemetry:
1. `classification_stage`
2. `classification_confidence`
3. `classification_owner_step`
4. `classification_subtype`
5. `classification_features_json`
6. URL-level replay outcome joined back to the classification decision

Recommended classifier trend graphs:
1. `ML usage %`
2. `ML replay URL accuracy %`
3. `rule replay URL accuracy %`

These should be plotted by run date from persisted metrics.

## Routing Memory Requirements
Historical routing memory is valuable, but only if it remembers successful behavior rather than repeated mistakes.

Rules:
1. Memory should be exact normalized URL only in the initial design.
2. Memory should not override strong deterministic rules.
3. Memory should require replay-backed evidence, not just repeated prior routing choices.
4. Repeated failures for the same route should weaken or block memory activation.
5. Sparse replay coverage should simply mean memory remains inactive.
6. Domain-wide or template-wide memory should not be enabled until exact-URL memory is trusted.

Recommended replay-backed activation thresholds:
1. At least `3` replay-backed samples for the exact normalized URL
2. At least `80%` agreement on the same route
3. At least `70%` replay-backed URL success for that route
4. At least `3` successful replay-backed samples

These thresholds may change, but they must remain explicit and versioned.

## Historical Reuse Requirements
Some runtime improvements should come from safe reuse, not repeated scraping.

Current design direction:
1. Reuse historical event records only for clearly static single-event detail URLs.
2. Match by normalized event URL in `events_history`.
3. Reuse only when the event is still in the future and more than `7` days away.
4. Re-scrape when the event is within `7` days because details can still change.
5. Never apply this reuse policy broadly to rolling listing pages or generic pages.

Data integrity rule:
1. Reused events must not blindly trust historical `address_id`.
2. Address resolution should be refreshed during reuse before insertion into `events`.

Telemetry rule:
1. Historical reuse should be recorded in the telemetry tables with explicit provenance.

## Runtime and Cost Reporting Requirements
The framework should publish both total and granular runtime/cost metrics.

Required runtime outputs:
1. Total pipeline duration
2. Step durations
3. Critical path duration
4. Throughput metrics

Required cost outputs:
1. Total LLM cost
2. Cost by provider
3. Cost by model
4. Cost by step
5. Cost per processed URL
6. Cost per inserted event

Interpretation rule:
1. Cost reductions are only wins if accuracy and coverage guardrails still pass.
2. Runtime reductions are only wins if accuracy, coverage, and severe-duplicate guardrails still pass.

Recommended optimization priorities:
1. Prefer skipping redundant work only for sources with strong identity guarantees or validated history.
2. Prefer deterministic short-circuits over expensive LLM calls where the route is already known.
3. Prefer reuse of already-known stable event detail records over repeated full extraction when policy conditions are met.

## Codex Review Loop
Codex should not be allowed to optimize directly against vague feedback. It should read a stable evaluation bundle and produce a constrained plan.

Recommended review inputs:
1. `output/run_scorecard.json`
2. `output/accuracy_replay_summary.json`
3. `output/classifier_performance_summary.json`
4. `output/duplicate_audit_summary.json`
5. `output/coverage_summary.json`
6. `output/runtime_summary.json`
7. `output/llm_cost_summary.json`
8. `output/comprehensive_test_report.html`
9. Relevant pipeline logs

Codex review discipline:
1. Codex should propose changes against the KPI scorecard, not against subjective impressions from one run.
2. Codex should name expected tradeoffs for each recommendation.
3. Codex should distinguish between classifier, extractor, deduplication, and coverage interventions.
4. Codex should not recommend auto-promoting raw replay failures into training without reviewed truth labeling.

Recommended Codex outputs:
1. Top 3 regressions
2. Top 3 likely root causes
3. Top 3 proposed changes
4. Expected KPI impact per proposed change
5. Confidence rating
6. Regression risk
7. Suggested tests or guardrails

Recommended operational loop:
1. Run `pipeline.py`.
2. Generate scorecard and supporting artifacts.
3. Compare against previous run and holdout baseline.
4. Have Codex produce a KPI-aware recommendation plan.
5. Human approves only the highest-value changes.
6. Implement changes with tests.
7. Re-run and compare deltas.
8. Promote only if guardrails pass.

Recommended human approval boundary:
1. Automatic promotion may update the training queue and safe positive labels.
2. Human review should remain required for ambiguous labels, suspicious replay failures, and taxonomy changes.
3. Metric definition changes, weight changes, and holdout changes should require explicit human approval.

## Why This Is Not Reinforcement Learning
This framework should not be described as reinforcement learning in the strict technical sense.

Reasons:
1. The environment is slow, noisy, and partially observed.
2. Feedback is delayed and incomplete.
3. The system has multiple competing objectives.
4. There is significant risk of reward hacking.

The correct design is:
1. Fixed metrics
2. Fixed guardrails
3. Fixed holdout sets
4. Structured proposals
5. Controlled promotion

If a future automated optimization layer is added, it must still operate within this scorecard and guardrail system.

## Governance Rules
1. Metric definitions must be versioned.
2. Weight changes must be documented.
3. Holdout-set changes must be documented.
4. Coverage watchlist changes must be documented.
5. No production promotion should rely solely on a subjective review.
6. Every major regression category should eventually become a metric or guardrail.
7. Label-taxonomy changes for the page classifier should be treated as schema changes and documented separately.
8. Web-page classification scope should remain distinct from non-web deterministic inputs such as email-address ingestion.

## Phased Implementation Plan

### Phase 1: Scorecard Foundation
1. Add `run_scorecard.json`.
2. Add `runtime_summary.json`.
3. Add `llm_cost_summary.json`.
4. Add classifier metrics to the scorecard.

### Phase 2: Integrity Coverage
1. Add `duplicate_audit_summary.json`.
2. Add duplicate KPIs and trend reporting.
3. Add `coverage_summary.json`.
4. Create `coverage_watchlist.csv`.

### Phase 3: Honest Evaluation
1. Define `gold_holdout`.
2. Separate `train`, `dev`, and `holdout`.
3. Add guardrails and pass/fail status.
4. Add domain-capped summaries.

### Phase 4: Optimization Loop
1. Generate Codex review bundles automatically.
2. Produce structured change recommendations.
3. Compare approved changes against previous run and holdout baseline.
4. Promote only if guardrails pass.

## Definition of Done
The evaluation framework is not complete until:
1. All four KPI families are measured.
2. Duplicate problems are measured explicitly.
3. Coverage is measured against an external watchlist or audit set.
4. Runtime and cost are reported at step level.
5. A holdout evaluation set exists.
6. Guardrails block false improvements.
7. Codex recommendations are generated from stable artifacts, not ad hoc intuition.

## Immediate Next Actions
1. Add `output/run_scorecard.json`.
2. Add `output/runtime_summary.json`.
3. Add `output/llm_cost_summary.json`.
4. Add `output/duplicate_audit_summary.json`.
5. Create `data/watchlists/coverage_watchlist.csv`.
6. Add a holdout evaluation definition.
7. Extend `comprehensive_test_report.html` to surface the scorecard and guardrail outcomes.

# PIPELINE_ARCHITECTURE.md

## Overview
- This document describes the logical architecture of the pipeline. 
- Implementation details may evolve as the system grows.
- The DanceScoop pipeline collects, processes, deduplicates, and stores social dance events in a PostgreSQL database. 
- The system aggregates events from multiple sources (websites, Facebook, Eventbrite, etc.) and transforms them into a standardized schema.

The pipeline prioritizes:

- Data quality
- Idempotent processing
- Resilient scraping
- Deterministic database updates

The pipeline is designed to run repeatedly without corrupting or duplicating data.

---

# High-Level Data Flow

External Event Sources
        │
        ▼
Scraping Layer
        │
        ▼
Raw Event Extraction
        │
        ▼
Event Normalization
        │
        ▼
Deduplication
        │
        ▼
Address Resolution
        │
        ▼
Database Update
        │
        ▼
Events Table (Production Data)

Each stage performs a clearly defined transformation on the data.

---

# Pipeline Stages

## 1. Scraping Layer

Purpose:  
Collect raw event data from multiple external sources.

Typical sources:

- Eventbrite
- Facebook events
- Dance organization websites
- Community calendars
- Search engine results

Responsibilities:

- Retrieve raw HTML or structured content
- Extract event-related content
- Capture event metadata such as:

  - event name
  - description
  - date/time
  - location text
  - URL
  - organizer/source

Key requirements:

- Scrapers must tolerate site layout changes.
- Scrapers must fail gracefully.
- Missing fields should not terminate the pipeline.

Scraping code should:

- Avoid brittle selectors.
- Log failures clearly.
- Retry transient network errors.

---

# 2. Raw Event Extraction

Purpose:  
Convert scraped content into preliminary event records.

Typical fields extracted:

- event_name
- description
- start_time
- end_time
- location_text
- source
- url

This stage does minimal transformation.

Important rules:

- Preserve original source data when possible.
- Do not aggressively clean or modify text yet.
- Store the raw event information for later processing.

---

# 3. Event Normalization

Purpose:  
Convert raw event data into a consistent format compatible with the database schema.

Normalization tasks include:

- Cleaning event names
- Standardizing dance styles
- Parsing date and time formats
- Removing obvious noise from descriptions
- Ensuring required fields exist

Normalization should:

- Be deterministic
- Avoid guesswork where possible
- Preserve useful descriptive content

Example transformations:

"Salsa Night @ XYZ!"
↓
event_name = "Salsa Night"
dance_style = "Salsa"
location_text = "XYZ"

---

# 4. Deduplication

Purpose:  
Identify and merge duplicate events that appear across multiple sources.

Duplicate events may arise when:

- Multiple websites advertise the same event
- Facebook and Eventbrite list the same event
- The same event appears in recurring listings

Deduplication uses a combination of:

- text similarity
- event name similarity
- date/time matching
- location similarity

Methods may include:

- embeddings
- clustering
- heuristic matching

Important constraints:

- Deduplication must not delete legitimate distinct events.
- Clusters should identify a canonical event record.

The canonical record should prioritize:

- records with valid locations
- records with richer descriptions
- records with reliable URLs

---

# 5. Address Resolution

Purpose:  
Convert free-text location data into structured address records.

Many scraped events contain incomplete location data such as:

"Victoria Dance Studio"

Address resolution attempts to produce:

building_name
street_number
street_name
municipality
province
postal_code
country
full_address

Resolution strategies may include:

- LLM-based parsing
- external search
- lookup against existing address records

Address resolution must:

- avoid creating duplicate address rows
- reuse existing address records when possible
- ensure canonical address formatting

---

# 6. Database Update

Purpose:  
Persist validated events and addresses into the PostgreSQL database.

Typical database tables:

events
addresses
events_history

The update stage must enforce:

- idempotency
- referential integrity
- schema compliance

Key operations include:

- inserting new events
- updating existing events
- linking events to addresses
- removing invalid records

Database writes must:

- use parameterized queries
- validate schema assumptions
- avoid destructive operations unless explicitly intended

---

# Error Handling Strategy

The pipeline is designed to be fault tolerant.

If a stage fails for an individual event:

- the error is logged
- processing continues for other events

Critical failures (e.g., database connectivity) should terminate the pipeline safely.

---

# Logging

All pipeline stages must produce structured logs.

Logs should include:

- source URL
- event identifier
- processing stage
- error messages
- timestamps

Logging is essential for diagnosing scraping issues and data anomalies.

---

# Idempotency

The pipeline must support repeated runs without corrupting data.

Key practices:

- avoid duplicate event creation
- ensure deterministic updates
- safely retry failed operations

Running the pipeline multiple times should produce the same database state.

---

# Performance Considerations

The pipeline should prioritize:

- correctness
- reliability
- maintainability

Performance optimizations should only be introduced when a clear bottleneck is identified.

Examples of acceptable optimizations:

- batching database writes
- caching address lookups
- parallelizing independent scraping tasks

Avoid premature optimization.

---

# Security Considerations

All external inputs should be treated as untrusted.

The system must:

- sanitize scraped text
- parameterize SQL queries
- protect API keys and secrets via environment variables

No credentials should ever be hardcoded.

---

# Pipeline Invariants

The following conditions should remain true across the pipeline unless a developer is intentionally changing the design.

## Data invariants

- Every persisted event should have a stable identity or matching strategy that prevents duplicate creation on repeated runs.
- The same source event should not produce multiple active database rows unless that behavior is explicitly intended.
- Event normalization should not remove meaningful source information needed later for deduplication or address resolution.
- Canonical events selected during deduplication should be traceable to their source records.
- Address resolution should prefer reuse of an existing canonical address over insertion of a near-duplicate row.
- Database writes should preserve referential integrity between events and addresses.

## Behavioral invariants

- Re-running the pipeline with the same inputs should produce the same database state.
- A failure on one event should not crash processing for unrelated events unless the failure is at a system-critical boundary.
- Logging should make it possible to identify where an event entered the pipeline, what transformations occurred, and why a failure happened.
- External-service failures should be visible in logs and handled explicitly, not silently swallowed.
- Each stage should have a clear responsibility and should not secretly perform major work that belongs to a later stage.

## Architecture invariants

- Scraping, normalization, deduplication, address resolution, and database update logic should remain conceptually separate even if helper utilities are shared.
- Existing repository utilities should be reused before new abstractions are introduced.
- Database logic should remain parameterized and schema-aware.
- New sources should plug into the existing pipeline shape rather than creating one-off side paths unless clearly justified.

When changing the code, developers and agents should verify that these invariants still hold.

---

# Common Failure Modes

The following failure patterns are common in event pipelines and should be checked whenever debugging.

## Scraping failures

- site layout changes break selectors
- anti-bot or login walls return partial or misleading content
- JavaScript-heavy pages appear empty to the scraper
- a source returns stale pages, partial pages, or localized variants that alter expected structure
- event images contain key text that is missing from plain HTML extraction

## Extraction and normalization failures

- dates are parsed incorrectly because of locale, timezone, or ambiguous formatting
- event names are over-cleaned and lose useful distinguishing information
- dance styles are inferred incorrectly from weak text cues
- descriptions include boilerplate that dominates matching logic
- missing fields are converted into misleading defaults rather than remaining explicitly null or empty

## Deduplication failures

- recurring events are collapsed into one row when they should remain separate
- similar but distinct events are merged because names or venues are too close
- duplicate events are missed because sources phrase titles or times differently
- canonical selection favors the wrong row because richer text, better address data, or better URLs are not weighted correctly
- cluster logic works on a sample set but behaves badly across the full dataset

## Address-resolution failures

- location text is too vague to resolve reliably
- LLM parsing produces a plausible but incorrect address
- building names vary slightly and create duplicate address rows
- postal codes or municipalities are missing, partial, or inconsistent
- an existing address should have been reused but a new one was inserted instead

## Database-update failures

- schema assumptions drift from the actual database
- upserts or updates overwrite better existing data with worse new data
- failed writes leave partially updated state
- sequence values or foreign keys become misaligned
- destructive cleanup removes valid future or historical events

## Operational failures

- retries create duplicates because idempotency checks are incomplete
- logs are too sparse to reconstruct what happened
- external API or model failures are handled inconsistently
- environment-specific configuration causes different behavior locally and in production
- long-running jobs fail midway and are difficult to resume safely

When debugging, check these categories systematically before making code changes.

---

# Extending the Pipeline

New sources can be added by implementing additional scraping modules.

New modules should:

- conform to the normalization schema
- return events in the standard structure
- avoid introducing schema drift

Before adding new scraping logic, developers should verify whether an existing module already handles the source.

---

# Developer Guidelines

When modifying pipeline components:

- prefer modifying existing utilities rather than duplicating logic
- maintain deterministic behavior
- update tests if behavior changes
- ensure changes preserve idempotency

Agents working in this repository should always:

- understand the full pipeline flow
- verify downstream effects of code changes
- maintain compatibility with existing database schema


## Implementation Notes

The pipeline is primarily orchestrated by:

pipeline.py

Supporting modules typically include components for:

- scraping
- normalization
- deduplication
- address processing
- database updates

These modules may evolve over time as the system grows.

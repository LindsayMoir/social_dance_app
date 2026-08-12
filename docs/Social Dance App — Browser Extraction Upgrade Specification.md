# Social Dance App — Browser Extraction Upgrade Specification

## Purpose

Update the Social Dance App web-scraping architecture to take advantage of the newer agent-oriented page representations available through Chromium/Playwright.

The goal is to improve extraction reliability—especially on difficult JavaScript-heavy sites such as Facebook and Instagram—without replacing the existing Playwright/Chromium browser infrastructure.

This should be implemented as an incremental enhancement to the existing scraping system, not as a rewrite.

---

# 1. Background

The Social Dance App performs substantial web scraping to discover social dance events.

A recurring problem is that modern websites, particularly:

- Facebook
- Instagram
- Eventbrite
- JavaScript-heavy event calendars
- React-based websites

can render information correctly for a human while making it difficult for conventional scrapers to extract that information reliably.

Problems include:

- unstable CSS classes
- deeply nested DOM structures
- dynamically generated content
- virtualized lists
- lazy loading
- iframes
- React component structures
- large amounts of irrelevant HTML
- selectors breaking when websites make small UI changes

Historically, the application has used Playwright/Chromium to render these pages and then attempted to extract information from the DOM or other page representations.

Playwright now exposes a page representation intended specifically for AI consumption.

A key capability is approximately:

```python
snapshot = await page.aria_snapshot(mode="ai")
```

This returns a compact representation derived from the browser's accessibility tree rather than requiring an AI model to interpret the complete raw HTML.

Conceptually, instead of this:

```html
<div class="x1n2onr6 x1ja2u2z ...">
    <span class="x193iq5w ...">
        <a href="/events/123456789">
            <span>Victoria Salsa Saturday</span>
        </a>
    </span>
</div>
```

the extraction system may receive something much closer to:

```yaml
- main:
  - heading "Victoria Salsa Saturday"
  - text: "Saturday, August 15"
  - text: "8:00 PM"
  - link "Victoria Salsa Saturday"
  - text: "Victoria, British Columbia"
```

This representation should generally be much easier and cheaper for an LLM to interpret.

---

# 2. Architectural Decision

DO NOT replace Playwright.

DO NOT replace Chromium.

DO NOT convert the production scraping system to Chrome DevTools MCP.

Instead, enhance the existing Playwright-based scraping architecture so that it can use an AI-oriented accessibility snapshot as one of its extraction representations.

The migration should primarily affect the extraction/read layer, not the browser-navigation layer.

Existing functionality for:

- launching Chromium
- authentication
- cookies/session handling
- navigation
- scrolling
- clicking
- pagination
- waiting for JavaScript
- opening event pages
- handling popups
- handling site-specific navigation

should remain intact unless a specific change is independently justified.

---

# 3. Important Architectural Separation

The scraping system should explicitly distinguish between:

## A. Acquisition

Acquisition means successfully getting the desired content rendered in the browser.

Responsibilities include:

- navigating to URLs
- login/session management
- authentication
- cookies
- scrolling
- clicking
- expanding content
- pagination
- waiting for dynamically loaded content
- detecting blocked pages
- dealing with lazy loading
- opening event detail pages

## B. Extraction

Extraction means understanding the content that has already been successfully rendered.

Responsibilities include:

- identifying event names
- dates
- times
- venues
- addresses
- dance styles
- organizers
- ticket URLs
- source URLs
- descriptions
- recurring-event information
- other structured event fields

These two concerns must remain separate.

The new accessibility/AI snapshot primarily improves **Extraction**.

It does NOT automatically solve acquisition problems such as Facebook refusing to display content, authentication failures, rate limiting, or lazy-loaded content that has not yet been rendered.

---

# 4. Preferred Extraction Strategy

The system should support multiple page representations.

Use the best available representation rather than assuming every site should be processed identically.

Recommended priority:

```text
1. Structured network/API data
        ↓ if unavailable or inadequate

2. Playwright AI accessibility snapshot
        ↓ if unavailable or inadequate

3. Existing deterministic DOM extraction
        ↓ if unavailable or inadequate

4. Screenshot / vision extraction
```

This ordering is guidance rather than an absolute rule.

If an existing deterministic DOM scraper for a particular site is highly reliable and inexpensive, it can remain the preferred method for that site.

The important change is to make the AI accessibility snapshot available as a first-class extraction method.

---

# 5. Structured Network Data Remains Preferred

If the browser receives clean machine-readable JSON through:

- XHR
- Fetch
- GraphQL
- embedded JSON
- public structured APIs

that data should normally be preferred over asking an LLM to reconstruct the same information from a visual page.

Example:

```json
{
    "event_name": "Friday Salsa Social",
    "start_time": "2026-08-14T20:00:00",
    "venue": "Example Dance Hall",
    "city": "Victoria"
}
```

If information of this quality is available directly, use it.

Do not replace good structured extraction with AI unnecessarily.

---

# 6. AI Accessibility Snapshot

Add support for Playwright's AI-oriented accessibility snapshot where supported by the installed Playwright version.

Conceptually:

```python
snapshot = await page.aria_snapshot(mode="ai")
```

The exact API should be verified against the Playwright version used by the project before implementation.

Do not assume the above call signature without checking the installed version and tests.

The snapshot should become an available input representation for the existing LLM extraction pipeline.

The LLM should receive the snapshot instead of the full HTML whenever the snapshot provides sufficient information.

Potential benefits:

- dramatically fewer tokens
- less irrelevant markup
- fewer dependencies on unstable CSS classes
- more human-semantic representation
- improved extraction from React applications
- better handling of links, buttons and headings
- reduced dependence on internal site implementation details
- better resilience to cosmetic website changes

---

# 7. Facebook and Instagram

Facebook and Instagram should be high-priority test cases because they are among the most difficult current sources.

However, this feature must not be treated as a way to evade Meta's automation detection.

Important distinction:

```text
Meta sees:
browser requests
navigation behaviour
sessions
cookies
timing
scrolling/clicking behaviour
HTTP requests
browser characteristics

Meta does not ordinarily receive:
whether we subsequently read the rendered DOM,
the accessibility tree,
an AI accessibility snapshot,
or a screenshot locally.
```

Generating an accessibility snapshot occurs after Chromium has rendered the page.

It is not a request to Facebook or Instagram for a special "AI version" of the page.

Therefore, switching from DOM parsing to accessibility snapshots should not inherently make the application more detectable by Meta.

At the same time:

**This does NOT make Facebook or Instagram scraping undetectable.**

Automation detection remains primarily an acquisition/browser-behaviour issue.

Do not add code intended to circumvent access controls, bot detection, authentication protections, or rate limiting as part of this change.

---

# 8. Shared Extraction Abstraction

Do not independently rewrite every scraper.

Add the new representation to the common page extraction/read layer wherever possible.

If the project currently has an abstraction such as `ReadExtract`, `PageExtractor`, or equivalent, enhance that abstraction.

Conceptual example only:

```python
class PageRepresentation:

    async def get_ai_snapshot(self, page):
        ...

    async def get_html(self, page):
        ...

    async def get_text(self, page):
        ...

    async def get_screenshot(self, page):
        ...
```

A higher-level extractor could then choose:

```python
class EventExtractor:

    async def extract(self, page):

        structured_data = await self.try_structured_network_data(page)

        if structured_data:
            return await self.extract_from_structured_data(structured_data)

        snapshot = await self.try_ai_snapshot(page)

        if snapshot and self.snapshot_is_usable(snapshot):
            return await self.extract_from_snapshot(snapshot)

        dom_result = await self.try_existing_dom_extraction(page)

        if dom_result:
            return dom_result

        return await self.try_visual_extraction(page)
```

This is architectural guidance, not mandatory class naming.

Codex should adapt the design to the existing codebase rather than introducing unnecessary new abstractions.

---

# 9. Preserve Source-Specific Navigation

Existing source-specific code should continue to be responsible for getting the browser into the correct state.

For example, a Facebook scraper may still need to:

```text
open page
→ restore session
→ dismiss popup
→ scroll
→ wait for events
→ click "See more"
→ scroll again
→ open individual event
```

Only after this should the extraction layer determine the best representation of the rendered content.

Do not assume an accessibility snapshot automatically loads information that has not yet been rendered.

---

# 10. Snapshot Storage and Debugging

The browser does not need to permanently create Markdown files for every webpage.

Do not introduce large numbers of permanent snapshot files as a normal production requirement.

However, diagnostic snapshots may be extremely useful.

Add optional debugging support so that, when enabled, the system can preserve:

- accessibility/AI snapshot
- URL
- timestamp
- scraper/source name
- extraction result
- extraction errors

This should be configurable and preferably disabled or limited in routine production operation.

The purpose is reproducibility and debugging.

If a scraper extracts an incorrect event, developers should be able to determine exactly what representation the LLM saw.

---

# 11. Do Not Assume Snapshot Equals Markdown

Terminology should be accurate.

The representation may look somewhat Markdown/YAML-like, but it should be referred to in the code and documentation as something such as:

```text
AI accessibility snapshot
ARIA snapshot
accessibility-tree snapshot
```

Do not build architecture around the assumption that Chromium automatically writes a Markdown file for every webpage.

It does not.

The browser maintains structured information about the rendered page, and Playwright can request an agent-friendly representation of it.

---

# 12. Experimental Rollout

Do not immediately convert all production scrapers.

Create an experimental implementation and compare the new extraction method against the current implementation.

Priority test sources:

1. Facebook
2. Instagram
3. Eventbrite
4. JavaScript-heavy dance/event calendars
5. Representative conventional dance websites

For each source, run existing extraction and accessibility-snapshot extraction against the same pages.

---

# 13. Metrics

Measure at least:

### Event recall

How many real events visible on the source are captured?

```text
events successfully found / events actually present
```

### Field accuracy

For captured events, evaluate accuracy of:

- title
- date
- start time
- end time
- venue
- city
- address
- dance style
- organizer
- event URL
- ticket URL

### URL accuracy

This should receive special attention because losing canonical event URLs would be unacceptable.

### Failure rate

Track:

```text
successful pages
failed pages
empty extraction
partial extraction
exceptions
blocked acquisition
```

### LLM token consumption

Compare:

```text
raw HTML tokens
vs.
AI accessibility snapshot tokens
```

### Processing time

Measure extraction latency.

### Extraction cost

Where LLM APIs are involved, estimate cost per successfully processed page/event.

---

# 14. Acceptance Criteria

The new method should not become the default merely because it is newer.

Promote it only if testing demonstrates a material benefit.

Potential success criteria:

- equal or better event recall
- equal or better field accuracy
- substantially lower token use
- fewer site-specific selectors
- reduced extraction failure rate
- no meaningful increase in acquisition failures
- correct source/event URLs retained
- easier debugging

Facebook and Instagram performance should be weighted heavily in this assessment because they are important difficult sources.

---

# 15. Fallback Behaviour

Never fail a scrape solely because the AI accessibility snapshot is unavailable.

Example:

```text
try structured data
    ↓
try AI accessibility snapshot
    ↓
try existing extraction
    ↓
try visual fallback
    ↓
record failure with diagnostic information
```

Existing successful extraction paths should remain available during the migration.

---

# 16. Logging

Add enough logging to determine which representation produced each extraction.

For example:

```text
source=facebook
url=https://...
acquisition=success
representation=aria_ai
extraction=success
events_found=6
```

or:

```text
source=facebook
acquisition=success
aria_snapshot=insufficient
fallback=dom
extraction=success
```

This will be important for comparing methods quantitatively.

---

# 17. LLM Input

Where LLM extraction is used, avoid automatically sending:

```python
await page.content()
```

if the accessibility snapshot contains everything required.

Prefer the smallest sufficiently complete representation.

The extraction prompt should make clear that the model must not invent missing fields.

Unknown values should remain NULL/None or whatever missing-value convention the Social Dance App schema uses.

---

# 18. Security / Compliance Guardrail

This change is an extraction improvement, not a bot-evasion project.

Do not introduce:

- CAPTCHA bypass
- fingerprint spoofing solely to defeat detection
- access-control circumvention
- rate-limit circumvention
- authentication bypasses

Any existing approved session/authentication behaviour should remain unchanged unless separately reviewed.

---

# 19. Codebase Investigation Required Before Modification

Before making changes, Codex should inspect the repository and identify:

1. Current Playwright version.
2. Current Chromium/browser launch architecture.
3. All common extraction abstractions.
4. Where raw HTML is currently sent to LLMs.
5. Site-specific DOM extraction.
6. Facebook extraction pipeline.
7. Instagram extraction pipeline.
8. Eventbrite extraction pipeline.
9. Existing logging and metrics.
10. Existing scraper tests.
11. Existing fixture/snapshot mechanisms.
12. Database fields related to extraction provenance or scraper runs.

Codex should adapt this specification to the actual architecture rather than assuming class names from this document.

---

# 20. Implementation Sequence

Recommended sequence:

### Phase 1 — Investigation

Inspect the repository and document the existing acquisition/extraction flow.

No major architectural changes yet.

### Phase 2 — Snapshot Prototype

Implement a small utility that can obtain the Playwright AI accessibility snapshot from an already rendered page.

Test it independently.

### Phase 3 — Diagnostic Comparison

For selected pages, capture:

```text
existing representation
AI accessibility snapshot
existing extraction result
new extraction result
```

Do not change production behaviour yet.

### Phase 4 — Facebook Test

Run controlled comparisons against representative Facebook event pages/listings.

Determine whether accessibility snapshots improve extraction from successfully rendered Facebook content.

### Phase 5 — Instagram / Eventbrite / Calendars

Repeat comparison.

### Phase 6 — Shared Extraction Integration

If results justify it, integrate accessibility snapshots into the shared extraction layer with fallback behaviour.

### Phase 7 — Gradual Source Migration

Enable the new approach source by source.

Avoid a global flag-day conversion.

---

# 21. Tests

Add unit and integration tests where practical.

At minimum test:

```text
AI snapshot can be requested successfully
snapshot contains meaningful page content
snapshot extraction produces expected event schema
fallback occurs if snapshot fails
fallback occurs if snapshot is empty
fallback occurs if snapshot lacks required event data
event URLs survive extraction
dates/times are not hallucinated
multiple events are handled correctly
iframes are handled if supported
lazy-loaded content remains an acquisition responsibility
```

Use saved fixtures where possible so tests do not depend entirely on live Facebook/Instagram behaviour.

---

# 22. Primary Design Principle

The new architecture should answer two separate questions:

```text
ACQUISITION:
Can we get the browser to successfully display the information?

EXTRACTION:
Once it is displayed, what is the most reliable representation
for an agent or deterministic parser to understand?
```

The accessibility snapshot is primarily an improvement to the second question.

This distinction should be reflected explicitly in the code and documentation.

---

# 23. Final Direction

The intended future architecture is approximately:

```text
                    SOURCE URL
                        │
                        ▼
                 PLAYWRIGHT
                  / CHROMIUM
                        │
                        │
            ┌─────────────────────┐
            │     ACQUISITION     │
            │                     │
            │ login               │
            │ cookies             │
            │ navigation          │
            │ clicking            │
            │ scrolling           │
            │ lazy loading        │
            │ pagination          │
            └──────────┬──────────┘
                       │
               rendered webpage
                       │
            ┌──────────▼──────────┐
            │     EXTRACTION      │
            │                     │
            │ 1 structured data   │
            │ 2 AI/ARIA snapshot  │
            │ 3 DOM extraction    │
            │ 4 vision fallback   │
            └──────────┬──────────┘
                       │
                       ▼
                  LLM / Parser
                       │
                       ▼
              Standard Event JSON
                       │
                       ▼
                  Validation
                       │
                       ▼
                   Postgres
```

The objective is not "use AI for every webpage."

The objective is:

> **Use the most reliable, compact and semantically useful representation of each successfully rendered webpage, while preserving deterministic extraction wherever it remains superior.**

Do not perform a wholesale scraper rewrite.

Implement this as a measured enhancement with comparative testing, fallback paths and gradual rollout.

---

## Implementation Notes (2026-08-11)

The installed Playwright version is `1.52.0`. Its supported Python API for this
feature is `page.locator("body").aria_snapshot()`; `Page.aria_snapshot(mode="ai")`
is not available in this version.

The initial implementation is deliberately opt-in through the `crawling`
configuration keys `aria_snapshot_enabled` and `aria_snapshot_debug_enabled`.
When disabled (the default), DOM text extraction remains unchanged. When enabled,
the system uses a sufficient ARIA snapshot and otherwise falls back to the
existing DOM representation. Debug mode records bounded JSON capture artifacts
under `debug/aria_snapshots`; by itself, it does not change extraction selection.

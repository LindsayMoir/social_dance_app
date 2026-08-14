# Open Issues

This file tracks design and implementation issues that need follow-up work over time.

## Discovery Depth Telemetry

Problem:
- Some sources expose event details immediately.
- Others require one or more intermediate discovery steps such as:
  - following a child event page
  - clicking a calendar day
  - clicking a local `Learn more` CTA inside a revealed panel

Example:
- `https://www.bardandbanker.com/live-music`
  - level 1: listing page
  - level 2: click the calendar/day entry
  - level 3: click the local `Learn more`
  - final result: extract from the event detail page

Design decision:
- Do not create a new classifier label yet.
- First measure and persist `discovery_depth_observed` for each scraped URL.
- Treat a meaningful discovery step as either:
  - navigation to a child/detail page
  - an interaction required to reveal event detail on the same page

Initial depth semantics:
- `1`: event details were available on the current page
- `2`: one meaningful follow/click was required
- `3`: two meaningful follow/click steps were required

Why this is worth measuring:
- It provides a stable structural signal without expanding the classifier taxonomy too early.
- It helps explain when `scraper.py` vs `rd_ext.py` is the right owner.
- It gives trendable telemetry for domains whose extraction path becomes more layered over time.

Implementation plan:
1. Persist `discovery_depth_observed` in `url_scrape_metrics`.
2. Trend average observed depth by step.
3. Add current-run reporting for:
   - domains with the highest average depth
   - success vs failure outcomes by discovery depth
4. Only consider adding it to the classifier feature set after enough telemetry has accumulated.

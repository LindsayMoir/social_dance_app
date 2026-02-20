# Design: Problem Categories Section + Week/Weekend Rules

Author: AI Assistant
Date: 2026-02-01

## Scope

1) Add a new "Problem Categories" section to the bottom of `comprehensive_test_report.html` built from `chatbot_test_results.csv`.
2) Review the LLM prompts and week/weekend definitions; propose changes to align with your requirements.

No code changes are included in this document; this is the design prior to implementation.

---

## Current Inputs and Files

- SQL-generation prompt: `prompts/contextual_sql_prompt.txt`
- Interpretation prompt: `prompts/interpretation_prompt.txt`
- Date logic helper (used by tools): `src/date_calculator.py`
- HTML report producer: `tests/validation/test_runner.py` → `_generate_html_report` (builds the page) and uses
  - Chatbot batch reporting: `tests/validation/chatbot_evaluator.py` → `generate_chatbot_report(scored_results, output_dir)`
- Results CSV consumed by the report: `output/chatbot_test_results.csv` (path resolved via `reporting.output_dir`).

---

## Observations (As-Is)

- `date_calculator.py`
  - "This week" returns Monday–Sunday of the current calendar week (inclusive), regardless of the current day.
  - "Next week" returns Monday–Sunday of the next calendar week.
  - "This weekend" returns Friday–Sunday (no explicit time end) and includes `dow_filter: [5,6,0]`.
  - "Tonight" returns the current date with `time_filter = '18:00:00'`.

- `prompts/contextual_sql_prompt.txt`
  - Already nudges the model to use the date calculator tool.
  - Currently describes weekend = Fri–Sun; does not fully encode the custom “from today forward within this week” policy.

- `comprehensive_test_report.html`
  - Already shows a "Problematic Questions" section sourced from `results['chatbot_testing']` (which reads the CSV and stats from `generate_chatbot_report`).
  - There is no roll-up of repeated issues into clearly named categories.

---

## Requested Behavior

1) Add a new section: "Problem Categories" (maximum 5 categories) that groups similar failures where `evaluation_score < 90` in `chatbot_test_results.csv`.
   - Each category should:
     - Have a human-readable name (e.g., "Week Calculation").
     - List the questions that belong to that category (or a subset if there are many), and one example row.
     - Include a brief description of the issue pattern and a suggested fix.

2) Align the week/weekend definitions with these rules:
   - Week starts on Sunday and ends on Saturday. (Note: your original message says "ends on Monday"; confirming intent: typical convention is Sunday–Saturday. If you really mean Sunday–Monday, call that out explicitly.)
   - Weekend starts on Friday at 6 PM and ends on Sunday at 11:59 PM.
   - "this week" queries should not include past days within the current week. If it’s Tuesday, return events from Tuesday through Saturday.

---

## Design: Problem Categories section

### Data Source
- `chatbot_test_results.csv` (written by `generate_chatbot_report`) contains the flattened evaluation columns, e.g.:
  - `evaluation_score`
  - `evaluation_reasoning`
  - `interpretation_score`, `interpretation_issues`, `interpretation_passed`
  - `sql_issues` (if any)
  - and other flattened columns (question, category, execution_success, etc.)

### Heuristic Category Extraction
We will scan rows with `evaluation_score < 90` and group them by a set of heuristics (in order; first match wins). We’ll keep only the top 5 categories by frequency.

Proposed categories and matching rules:

1) Week Calculation
   - Trigger if reasoning or issues contain any of:
     - `"this week"`, `"next week"`, `"Mon-Sun"`, `"Monday"`, `"Sunday"`, `"week calculation"`, `"current week"`, `"calendar week"`.
   - OR if the SQL contains explicit week logic and the interpretation mentions a week.

2) Weekend Calculation
   - Trigger if reasoning or issues mention `"weekend"`, `"Fri"`, `"Sun"`, `"Fri-Sun"` or the SQL has DOW extraction with `[5,6,0]`, but interpretation indicates different edges.

3) Tonight / Time Filter
   - Trigger if interpretation mentions `"tonight"` (or `"tomorrow night"`) and SQL is missing `start_time >= '18:00:00'`, or the interpretation/LLM reasoning flags a time filter issue.

4) Event Type Defaults and Inclusion
   - Trigger if SQL/interpretation conflicts on event types (e.g., default social dance missing, or class/live requested but not included).

5) Date Arithmetic / Interval / Relative Dates
   - Trigger if `sql_issues` or `evaluation_reasoning` mentions bad date arithmetic (e.g., `CURRENT_DATE + 7`), or invalid intervals.

6) Fallback: Other
   - If none match, group under "Other" with a concise summary from `evaluation_reasoning`.

(We will tune these over time by looking at real CSVs.)

### HTML Rendering
- Location: Below existing "Problematic Questions".
- Structure:
  - `<h3>Problem Categories</h3>`
  - For each category (max 5):
    - `<h4>{Category Name} ({count} issues)</h4>`
    - A short description and example (first matching row’s question and SQL snippet).
    - A bullet or compact list of the question texts (e.g., 5–10 items or the first N), and a line with remaining counts if many.

### Performance and Safety
- This is a light CSV scan done once at report time. No DB I/O.
- If CSV lacks the expected columns, we’ll degrade gracefully and show a small "Problem Categories" section with a single category "Other".

---

## Design: Week/Weekend rules and prompt/tool alignment

### Your target policy
- Week: Sunday–Saturday (please confirm that your earlier “ends on Monday” was a typo).
- Weekend: Friday 18:00 → Sunday 23:59.
- "This week" should not include days in the past; start from current day to Saturday.

### Proposed Implementation Plan

1) `src/date_calculator.py`
   - Change `"this week"`:
     - Compute Sunday–Saturday, then set `start_date = max(today, week_start)` and `end_date = week_end`.
     - If today is Saturday, return Saturday–Saturday.
   - Change `"next week"` to Sunday–Saturday of the following week.
   - Change `"this weekend"`:
     - Return `start_date = upcoming Friday`, `end_date = upcoming Sunday`.
     - Also return `time_filter = '18:00:00'` for Friday, and `end_time_filter = '23:59:59'` for Sunday.
       - Note: SQL consumers can translate this into `( (start_date = friday AND start_time >= '18:00') OR (start_date IN (Sat,Sun)) )`, and optionally apply an end time on Sunday.

2) `prompts/contextual_sql_prompt.txt`
   - Add explicit policy bullets:
     - Week = Sunday–Saturday.
     - Weekend = Fri 18:00–Sun 23:59.
     - For "this week": return only upcoming dates this week (do not include days already passed in the same week). If run on Tuesday, return Tuesday–Saturday.
   - Keep the instruction to always call `calculate_date_range` and use only the returned dates and optional time filters.

3) `src/main.py` and `tests/validation/chatbot_evaluator.py`
   - They already attach the date tool and sanitize SQL. After `date_calculator` changes, consumers should embed:
     - For "this week": simple `start_date >= {start} AND start_date <= {end}`.
     - For weekend: either use `dow_filter` or time boundaries returned by the tool.

### Open Questions / Confirmation Needed
- Your note says: “Week starts on Sunday and ends on Monday.”
  - Please confirm: Did you mean "ends on Saturday"?
  - If you truly want Sunday–Monday (two-day window), we’ll implement exactly that.

---

## Implementation Steps (after approval)

1) Problem Categories
   - Extend `generate_chatbot_report(scored_results, output_dir)` to:
     - Load the CSV it just wrote (or work directly from in-memory `scored_results` + flattened fields) and compute category tallies for rows with `evaluation_score < 90`.
     - Return a new `report['problem_categories']` structure with:
       ```json
       [
         {
           "name": "Week Calculation",
           "count": 12,
           "example": {"question": "Show me salsa events this week", "reason": "..."},
           "questions": ["Show me salsa events this week", "..."],
           "recommendation": "Align week boundaries and use date tool outputs"
         },
         ... up to 5 entries ...
       ]
       ```
   - Update `test_runner._generate_html_report()` to render a new section from `report['problem_categories']`.

2) Week/Weekend Rules
   - Update `date_calculator` with the new logic described above (guarded with tests/stubs where appropriate).
   - Update `contextual_sql_prompt.txt` to include week/weekend policy.
   - Optionally add a validation test suite snippet to verify:
     - “This week” on Wednesday returns Wed–Sat.
     - “This weekend” returns Fri–Sun with Friday time_filter >= 18:00:00.

3) QA
   - Rerun a subset (random_test_limit = 20) and review `comprehensive_test_report.html` for:
     - Presence of the new Problem Categories section.
     - Reduced “week calculation” issues.

---

## Rollback / Backward Compatibility
- If the policy shift (Sunday–Saturday and forward-only within week) causes regressions, we can keep both policies behind a config flag, e.g.,
  - `testing.validation.week_policy: sunday_saturday_forward_only` vs `monday_sunday_full_week`.

---

## End

This document outlines the plan. After you confirm the week boundary (Sunday–Saturday vs Sunday–Monday), I will implement the code changes accordingly.


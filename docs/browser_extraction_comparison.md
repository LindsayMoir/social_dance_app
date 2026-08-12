# Live Browser Extraction Comparison

The comparison runner navigates directly to five configured event pages for
Facebook, Instagram, and Eventbrite. For every fully rendered page it records:

- `dom_text`: the current deterministic HTML text representation;
- `aria_snapshot`: the experimental accessibility-tree representation;
- lengths, final URL, access-state classification, and any navigation or snapshot error.

The access-state fields explicitly identify login dialogs and unavailable-content
shells. Do not regard a non-empty snapshot as a successful comparison when its
access state is not `event_content_visible`.

It is read-only: it does not call the LLM, write events, update the database, or
change the configured `aria_snapshot_enabled` setting.

Facebook and Instagram deliberately reuse the pipeline's established session
clients: `FacebookEventScraper` and `ImageScraper`. With `--headful`, either
client will invoke its normal interactive login/challenge flow and persist a
refreshed session before comparison begins. This differs from the original
generic-context version, which could only attempt to load saved state.

Run it from the repository root:

```bash
python scripts/compare_browser_extraction.py --headful
```

To run one platform only, for example while refreshing an Instagram session:

```bash
python scripts/compare_browser_extraction.py --headful --platform instagram
```

`--headful` is recommended for the first run so an expired Facebook, Instagram,
or Eventbrite session/challenge is visible. The runner uses the existing saved
session state for each platform, and it never visits a login page deliberately.

Artifacts are written beneath `debug/browser_extraction_comparison/<timestamp>/`:

- `01_facebook.json` through `15_eventbrite.json` contain both complete texts;
- `summary.json` provides the per-page status and size comparison.

The URL set lives in
`data/evaluation/browser_extraction_comparison_urls.csv`. Keep five direct event
or post URLs per platform current; the runner rejects an incomplete fixture.

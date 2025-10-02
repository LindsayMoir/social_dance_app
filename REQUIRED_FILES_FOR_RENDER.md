# Required Files for Render Deployment

This document lists all files required for `pipeline.py` to run successfully on Render.

## ⚠️ CRITICAL ISSUE FOUND

**PROBLEM:** `data/other/sql_input.json` is blocked by `.gitignore` (line 15: `*.json`)

**SOLUTION:** Update `.gitignore` to allow this specific file.

## Required Files Checklist

### ✅ Configuration Files (Will be pushed to Git)

| File | Status | Purpose |
|------|--------|---------|
| `config/config.yaml` | ✅ Included | Main configuration file for all modules |
| `requirements.txt` | ✅ Included | Python dependencies |
| `.github/workflows/sync-env-to-render.yml` | ✅ Included | Auto-sync environment variables |

### ✅ Data Files - CSV (Will be pushed to Git)

**Location: `data/other/`**

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `black_list_domains.csv` | ✅ Included | scraper.py, clean_up.py | Domains to exclude from scraping |
| `calendar_urls.csv` | ✅ Included | scraper.py | Google Calendar URLs to scrape |
| `countries.csv` | ✅ Included | db.py | Country codes and names for address validation |
| `edge_cases.csv` | ✅ Included | Multiple modules | Special case handling rules |
| `emails.csv` | ✅ Included | emails.py | Email sources to process |
| `images.csv` | ✅ Included | images.py | Image URL sources |
| `keywords.csv` | ✅ Included | fb.py, clean_up.py | Search keywords for event discovery |
| `municipalities.txt` | ✅ Included | db.py | List of municipalities for address parsing |
| `nulls_addresses.csv` | ✅ Included | clean_up.py | Addresses that need fixing |
| `pdfs.csv` | ✅ Included | read_pdfs.py | PDF sources to process |

**Location: `data/urls/`**

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `gs_urls.csv` | ✅ Included | Multiple modules | Google Search result URLs to process |

### ⚠️ Data Files - JSON (BLOCKED BY .gitignore)

| File | Status | Used By | Purpose | Action Needed |
|------|--------|---------|---------|---------------|
| `data/other/sql_input.json` | ❌ BLOCKED | db.py | SQL query templates | **Fix .gitignore** |

### ✅ Prompt Files (Will be pushed to Git)

**Location: `prompts/`**

All `.txt` files in the prompts directory are required:

| File | Purpose |
|------|---------|
| `address_0_null_location_fix_prompt.txt` | Address fixing for null locations |
| `address_fix_prompt.txt` | General address fixing |
| `address_from_raw_page.txt` | Extract addresses from raw HTML |
| `address_internet_fix.txt` | Fix addresses using internet search |
| `bard_and_banker_prompt.txt` | Site-specific extraction (Bard & Banker) |
| `chatbot_instructions.txt` | Chatbot behavior instructions |
| `deb_rhymer_prompt.txt` | Site-specific extraction (Deb Rhymer) |
| `dedup_llm_address.txt` | Address deduplication logic |
| `dedup_prompt.txt` | Event deduplication logic |
| `default.txt` | Default event extraction prompt |
| `djdancingdean_prompt.txt` | Site-specific extraction (DJ Dancing Dean) |
| `event_name_location_fix.txt` | Fix event names and locations |
| `fb_prompt.txt` | Facebook event extraction |
| `fix_dup_addresses.txt` | Fix duplicate addresses |
| `images_prompt.txt` | Extract events from images |
| `interpretation_prompt.txt` | Interpret extracted data |
| `irrelevant_rows_prompt.txt` | Filter irrelevant events |
| `relevant_dance_url.txt` | Determine if URL is dance-related |
| `single_event.txt` | Extract single event from page |
| `sql_prompt.txt` | Generate SQL queries |
| `the_coda_prompt.txt` | Site-specific extraction (The Coda) |

### ❌ Runtime Files (NOT in Git, created during execution)

**Location: `checkpoint/`** - Ignored by `.gitignore` (line 6)
- `extracted_text.xlsx`
- `fb_search_keywords.csv`
- `fb_urls.csv`
- `fb_urls_cp.csv`

These are created during pipeline execution and don't need to be in Git.

**Location: `logs/`** - Ignored by `.gitignore` (line 9)
- All log files are created at runtime

**Location: `output/`** - Ignored by `.gitignore` (line 12)
- All output CSV files are created at runtime

### 🔐 Secret Files (Upload to Render Secret Files)

**DO NOT include in Git** - See `RENDER_DEPLOYMENT.md` for upload instructions

| File | Upload to Render as | Mount Path |
|------|-------------------|------------|
| `facebook_auth.json` | `facebook_auth.json` | `/etc/secrets/facebook_auth.json` |
| `eventbrite_auth.json` | `eventbrite_auth.json` | `/etc/secrets/eventbrite_auth.json` |
| `instagram_auth.json` | `instagram_auth.json` | `/etc/secrets/instagram_auth.json` |
| Gmail client secret | `desktop_client_secret.json` | `/etc/secrets/desktop_client_secret.json` |
| Gmail token | `desktop_client_secret_token.json` | `/etc/secrets/desktop_client_secret_token.json` |

### 📁 Directory Structure Required on Render

Render will automatically create these from Git:
```
social_dance_app/
├── config/
│   └── config.yaml
├── data/
│   ├── other/
│   │   ├── *.csv files
│   │   ├── *.txt files
│   │   └── sql_input.json ⚠️ (currently blocked)
│   └── urls/
│       └── *.csv files
├── prompts/
│   └── *.txt files
└── src/
    └── *.py files
```

These directories will be created at runtime:
- `checkpoint/` - Created by pipeline
- `logs/` - Created by pipeline
- `output/` - Created by pipeline
- `config/run_specific_configs/` - Created by pipeline

## Fix Required: Update .gitignore

To ensure `sql_input.json` is included in Git (and thus available on Render), update `.gitignore`:

**Change line 14-15 from:**
```gitignore
# ignore all json files
*.json
```

**To:**
```gitignore
# ignore all json files
*.json
# But include data files needed for deployment
!data/other/sql_input.json
```

## Verification Checklist

Before deploying to Render:

- [ ] **Fix .gitignore** to include `data/other/sql_input.json`
- [ ] Commit and push all data files to Git
- [ ] Upload all secret JSON files to Render Secret Files
- [ ] Configure environment variables in Render (or use GitHub Action)
- [ ] Verify `config/config.yaml` has correct paths
- [ ] Ensure all prompt files are in `prompts/` directory
- [ ] Check that `requirements.txt` includes all dependencies

## Testing Locally Before Render Deploy

```bash
# Verify all required files exist
cd /mnt/d/GitHub/social_dance_app

# Check data files
ls -la data/other/*.csv data/other/*.txt data/other/*.json
ls -la data/urls/*.csv

# Check prompt files
ls -la prompts/*.txt

# Check config
ls -la config/config.yaml

# Run pipeline locally to ensure it works
cd src
python pipeline.py
```

## What Happens on Git Push

When you `git push` to GitHub:

1. ✅ All `.py` files in `src/` → Pushed
2. ✅ All `.csv` files in `data/` → Pushed
3. ✅ All `.txt` files in `data/other/` → Pushed
4. ✅ All `.txt` files in `prompts/` → Pushed
5. ✅ `config/config.yaml` → Pushed
6. ⚠️ `data/other/sql_input.json` → **CURRENTLY BLOCKED** (need to fix)
7. ❌ `*.json` auth files → Correctly blocked (upload to Render Secret Files instead)
8. ❌ `checkpoint/`, `logs/`, `output/` → Correctly blocked (runtime directories)
9. ❌ `.env` file → Correctly blocked (use environment variables instead)

## Summary

**Total Required Files:**
- Configuration: 1 file
- Data CSV: 11 files
- Data TXT: 1 file
- Data JSON: 1 file ⚠️ (blocked, needs fix)
- Prompt files: 21 files
- Secret files: 5 files (upload separately to Render)

**Action Required:** Fix `.gitignore` to allow `data/other/sql_input.json`

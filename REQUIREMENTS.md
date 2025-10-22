# Requirements Files

This project uses split requirements files for faster deployments and clearer dependency management.

## Files:

### `requirements-web.txt` (Minimal - for web services)
Used by:
- Render backend service (FastAPI)
- Render frontend service (Streamlit)

Contains only:
- Web frameworks (FastAPI, Streamlit, Uvicorn)
- Database (SQLAlchemy, psycopg2)
- LLM providers (Mistral, OpenAI)
- Core utilities (pandas, pydantic, python-dotenv)

**Size**: ~200-300 MB
**Build time**: 2-3 minutes

### `requirements-pipeline.txt` (Full - for scraping/pipeline)
Used by:
- Local development
- Render cron jobs (pipeline execution)

Includes everything from `requirements-web.txt` PLUS:
- Playwright & Scrapy (web scraping)
- Prefect (workflow orchestration)
- PDF/image processing tools
- Text matching & ML libraries

**Size**: ~1-2 GB (includes Playwright browsers)
**Build time**: 10-15 minutes

### `requirements.txt` (Legacy - kept for reference)
The original combined requirements file.
**Status**: Can be deleted once split requirements are confirmed working.

## Why Split Requirements?

### Before (using `requirements.txt`):
- Web service deploys: **10-15 minutes** ❌
- Installing unnecessary 1+ GB of Playwright, Scrapy, ML libraries
- Web services never use these dependencies

### After (using `requirements-web.txt`):
- Web service deploys: **2-3 minutes** ✅
- Only installs what's actually needed (~200 MB)
- 5x faster deployments!

## Usage:

### Local Development:
```bash
# Install full requirements (pipeline + web)
pip install -r requirements-pipeline.txt
```

### Web Services on Render:
```bash
# Automatically uses requirements-web.txt (configured in render.yaml)
pip install -r requirements-web.txt
```

### Render Cron Jobs:
```bash
# Use full requirements for pipeline execution
pip install -r requirements-pipeline.txt
```

## Maintenance:

When adding new dependencies:

1. **Web service needs it?** → Add to `requirements-web.txt`
2. **Pipeline/scraping needs it?** → Add to `requirements-pipeline.txt`
3. **Both need it?** → Add to `requirements-web.txt` (it's included in pipeline)

## Testing:

Before committing requirements changes:

```bash
# Test web requirements
pip install -r requirements-web.txt
python src/main.py  # Should work
python -c "import streamlit"  # Should work

# Test pipeline requirements
pip install -r requirements-pipeline.txt
python src/pipeline.py  # Should work
```

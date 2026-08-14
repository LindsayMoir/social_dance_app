# Database Architecture Plan: Multi-Environment Support

## Problem Statement

We need to support three different execution modes for `pipeline.py`:

1. **Local → Local → Production**: Run locally, update local DB, then copy to Render Production
2. **Local → Render Dev → Production**: Run locally, update Render Dev DB (for testing), then copy to Production
3. **Render CRON → Render Dev → Production**: Run on Render as CRON job, update Render Dev DB, then copy to Production

Currently, the code only supports:
- Local execution → Local DB (uses `DATABASE_*` env vars)
- Render execution → Render Production DB (uses `RENDER_EXTERNAL_DB_URL`)

## Architecture Goals

1. **Environment Detection**: Automatically detect where code is running (local machine vs Render)
2. **Target Database Selection**: Allow explicit choice of which database to update
3. **Production Protection**: Never accidentally write to production during development/testing
4. **Flexible Deployment**: Support both local testing and automated CRON jobs

## Proposed Solution: Three-Tier Architecture

### Environment Variables (Already Configured in .env)

```bash
# Local Database
DATABASE_CONNECTION_STRING='postgresql://postgres:5539@localhost/social_dance_db'

# Render Dev Database
RENDER_DEV_EXTERNAL_DB_URL='postgresql://social_dance_dev_user:dXBAaGYlSuIo7jFCWIgPruqyKmSjVSol@dpg-d3fd48euk2gs73assq90-a.oregon-postgres.render.com/social_dance_dev_db'

# Render Production Database
RENDER_EXTERNAL_DB_URL='postgresql://social_dance_db_user:tbB7biYpdUCUnAQ1I0Dpw9iZyB6GQcXH@dpg-culu0r1u0jms73bgrcdg-a.oregon-postgres.render.com/social_dance_db_eimr'
```

### New Environment Variable: DATABASE_TARGET

This variable determines which database to connect to:

```bash
# Options: "local", "render_dev", "render_prod"
DATABASE_TARGET='local'          # Default: local development
DATABASE_TARGET='render_dev'     # For testing on Render Dev DB
DATABASE_TARGET='render_prod'    # For production (web service only)
```

### Execution Modes

#### Mode 1: Local Development (Default)
```bash
# In local .env
DATABASE_TARGET='local'
RENDER=false  # Not set or explicitly false

# Result: Uses DATABASE_CONNECTION_STRING
# Updates: Local PostgreSQL
# Use case: Daily development work
```

#### Mode 2: Local Testing Against Render Dev
```bash
# In local .env
DATABASE_TARGET='render_dev'
RENDER=false

# Result: Uses RENDER_DEV_EXTERNAL_DB_URL
# Updates: Render Dev database from local machine
# Use case: Testing pipeline.py against cloud database before deploying
```

#### Mode 3: Render CRON Job (Testing)
```bash
# In Render environment variables
DATABASE_TARGET='render_dev'
RENDER=true

# Result: Uses RENDER_DEV_EXTERNAL_DB_URL (or INTERNAL for speed)
# Updates: Render Dev database
# Use case: CRON job testing in staging environment
```

#### Mode 4: Production Web Service
```bash
# In Render environment variables
DATABASE_TARGET='render_prod'
RENDER=true

# Result: Uses RENDER_EXTERNAL_DB_URL
# Updates: Production database
# Use case: Web API serving requests
```

## Implementation Plan

### Step 1: Create Database Configuration Utility

**New File**: `src/db_config.py`

```python
"""
Database configuration utility for multi-environment support.

Centralizes all database connection logic and provides a single function
to get the appropriate connection string based on execution environment.
"""

import os
import logging
from typing import Tuple

def get_database_config() -> Tuple[str, str]:
    """
    Determine which database to connect to based on environment variables.

    Returns:
        Tuple[str, str]: (connection_string, environment_name)

    Environment Variables Used:
        - DATABASE_TARGET: 'local', 'render_dev', or 'render_prod'
        - RENDER: 'true' if running on Render, otherwise unset/false

    Logic:
        1. If DATABASE_TARGET is explicitly set, use that
        2. Otherwise, infer from RENDER environment variable:
           - RENDER=true → render_prod (safe default for production)
           - RENDER=false/unset → local
    """
    # Get explicit target (highest priority)
    target = os.getenv('DATABASE_TARGET', '').lower()
    is_render = os.getenv('RENDER', '').lower() == 'true'

    # If no explicit target, infer from environment
    if not target:
        target = 'render_prod' if is_render else 'local'
        logging.info(f"DATABASE_TARGET not set, inferred: {target}")

    # Map target to connection string
    connection_map = {
        'local': (
            os.getenv('DATABASE_CONNECTION_STRING'),
            'Local PostgreSQL (localhost)'
        ),
        'render_dev': (
            os.getenv('RENDER_DEV_INTERNAL_DB_URL') if is_render else os.getenv('RENDER_DEV_EXTERNAL_DB_URL'),
            'Render Development Database'
        ),
        'render_prod': (
            os.getenv('RENDER_INTERNAL_DB_URL') if is_render else os.getenv('RENDER_EXTERNAL_DB_URL'),
            'Render Production Database'
        )
    }

    if target not in connection_map:
        raise ValueError(
            f"Invalid DATABASE_TARGET: '{target}'. "
            f"Must be one of: local, render_dev, render_prod"
        )

    connection_string, env_name = connection_map[target]

    if not connection_string:
        raise ValueError(
            f"Database connection string not found for target '{target}'. "
            f"Check your environment variables."
        )

    logging.info(f"Database target: {target} → {env_name}")
    logging.info(f"Running on: {'Render' if is_render else 'Local Machine'}")

    return connection_string, env_name


def get_production_database_url() -> str:
    """
    Get the production database URL for copying data to production.

    This is always the production database, regardless of where code is running.
    Used by the final step of pipeline.py to copy data to production.

    Returns:
        str: Production database connection string
    """
    is_render = os.getenv('RENDER', '').lower() == 'true'

    if is_render:
        prod_url = os.getenv('RENDER_INTERNAL_DB_URL')
    else:
        prod_url = os.getenv('RENDER_EXTERNAL_DB_URL')

    if not prod_url:
        raise ValueError("Production database URL not configured")

    logging.info(f"Production database URL obtained for data copy")
    return prod_url
```

### Step 2: Update DatabaseHandler in db.py

**Current Code** (lines 54-68):
```python
if os.getenv("RENDER"):
    logging.info("def __init__(): Running on Render.")
    connection_string = os.getenv('RENDER_EXTERNAL_DB_URL')
    self.conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
    logging.info("def __init__(): Database connection established for Render social_dance_db.")
else:
    # Running locally
    logging.info("def __init__(): Running locally.")
    self.conn = self.get_db_connection()
    logging.info("def __init__(): Database connection established for social_dance_db.")
```

**New Code**:
```python
from db_config import get_database_config

# In __init__:
connection_string, env_name = get_database_config()
self.conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
logging.info(f"Database connection established: {env_name}")
```

### Step 3: Remove get_db_connection() Method

The `get_db_connection()` method in `db.py` is no longer needed since we now use `get_database_config()`.

### Step 4: Update All Files Using Database Connections

**Files to Update**:
1. `src/db.py` - DatabaseHandler.__init__()
2. `src/pipeline.py` - If it has direct DB connections
3. `src/clean_up.py` - If it has direct DB connections
4. `src/scraper.py` - If it has direct DB connections
5. `src/main.py` - If it has direct DB connections
6. `src/ebs.py` - If it has direct DB connections
7. `src/fb.py` - If it has direct DB connections
8. `src/dedup_llm.py` - If it has direct DB connections
9. `src/irrelevant_rows.py` - If it has direct DB connections

### Step 5: Update Production Copy Script

The final step of `pipeline.py` that copies from dev → production needs to be updated:

```python
from db_config import get_production_database_url

# When copying to production:
prod_url = get_production_database_url()
# ... perform copy operation ...
```

### Step 6: Update Render Environment Variables

**For Render Web Service (Production)**:
```bash
DATABASE_TARGET=render_prod
RENDER=true
```

**For Render CRON Job (Testing)**:
Create a separate Render service with:
```bash
DATABASE_TARGET=render_dev
RENDER=true
```

## Safety Features

1. **Explicit Target Required for Production**: Production writes require explicit `DATABASE_TARGET=render_prod`
2. **No Accidental Production Writes**: Default behavior uses dev databases
3. **Clear Logging**: Every connection logs which database it's connecting to
4. **Validation**: Raises errors if configuration is invalid or missing

## Testing Strategy

### Test 1: Local Development
```bash
cd /mnt/d/GitHub/social_dance_app/src
export DATABASE_TARGET=local
python pipeline.py
# Verify: Updates local PostgreSQL
```

### Test 2: Local → Render Dev
```bash
export DATABASE_TARGET=render_dev
python pipeline.py
# Verify: Updates Render Dev database from local machine
```

### Test 3: Render CRON → Render Dev
```bash
# Set in Render dashboard:
# DATABASE_TARGET=render_dev
# RENDER=true
# Deploy and run
# Verify: Updates Render Dev database
```

### Test 4: Production (Web Service)
```bash
# Set in Render dashboard:
# DATABASE_TARGET=render_prod
# RENDER=true
# Verify: Web service connects to production
```

## Migration Checklist

- [ ] Create `src/db_config.py` utility
- [ ] Update `src/db.py` to use `get_database_config()`
- [ ] Remove obsolete `get_db_connection()` method
- [ ] Search for direct database connections in other files
- [ ] Update all files to use centralized config
- [ ] Add `DATABASE_TARGET` to `.env` file
- [ ] Update Render environment variables
- [ ] Test all four execution modes
- [ ] Update documentation
- [ ] Create rollback plan

## Rollback Plan

If issues arise:
1. Git revert to commit before changes
2. Render environment still works (RENDER=true logic preserved)
3. Local environment uses DATABASE_CONNECTION_STRING as before

## Benefits

1. **Flexibility**: Run pipeline.py from anywhere, against any database
2. **Safety**: Explicit targeting prevents accidental production writes
3. **Clarity**: Logging shows exactly which database is being used
4. **Testability**: Easy to test against Render Dev before production deploy
5. **Maintainability**: Single source of truth for database configuration

## Future Enhancements

1. Add database connection pooling
2. Add automatic failover to backup database
3. Add read-replica support for queries
4. Add connection health checks

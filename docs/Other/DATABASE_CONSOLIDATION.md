# Address Database Consolidation

## Overview

The Canadian postal code database (`address_db`) has been consolidated into the main `social_dance_db` to simplify architecture and deployment.

**Migration Date**: October 2, 2025

## What Changed

### Before
- **Two separate databases**:
  - `social_dance_db` - Main application database
  - `address_db` - Separate 579 MB Canadian postal code database
- **Separate connections**: Required `ADDRESS_DB_CONNECTION_STRING` environment variable
- **Single table**: The `address_db` contained only one table: `locations` (Canadian postal codes)

### After
- **One consolidated database**: `social_dance_db` contains all tables including `locations`
- **Simplified connection**: Only need `DATABASE_CONNECTION_STRING` (local) or `RENDER_EXTERNAL_DB_URL` (production)
- **Same table**: `locations` table now lives in `social_dance_db`

## Migration Steps

### 1. Run Migration Script (Local)

```bash
cd /mnt/d/GitHub/social_dance_app
PGPASSWORD=5539 psql -h localhost -U postgres -d social_dance_db -f migrations/003_consolidate_address_db.sql
```

This script:
- Creates the `dblink` extension
- Copies all data from `address_db.locations` to `social_dance_db.locations`
- Creates an index on `mail_postal_code` for performance
- Verifies the copy succeeded

### 2. Environment Variables Update


### 3. Code Changes

**File**: `src/db.py`

**Changes**:
1. Removed `self.address_db_engine` initialization
2. Changed `populate_from_db_or_fallback()` to use `self.conn` instead of `self.address_db_engine`
3. Added comment explaining consolidation

## Render Development Database

For testing `pipeline.py` on Render, you'll need to:

1. **Upload the `locations` table** to your Render dev database:
   ```bash
   # Export from local
   pg_dump -h localhost -U postgres -d social_dance_db -t locations --data-only > locations_data.sql

   # Import to Render dev (use credentials from Render dashboard)
   psql -h <RENDER_DEV_HOST> -U <RENDER_DEV_USER> -d <RENDER_DEV_DB> -f locations_data.sql
   ```

2. **Set environment variable** in Render:
   - The dev database connection is already configured via `DATABASE_CONNECTION_STRING` or similar
   - No need for separate `ADDRESS_DB_CONNECTION_STRING`

## Table Schema

The `locations` table contains 30 columns with Canadian postal code data:

| Column | Type | Description |
|--------|------|-------------|
| `loc_guid` | text | Location GUID |
| `addr_guid` | text | Address GUID |
| `civic_no` | bigint | Civic number |
| `civic_no_suffix` | text | Civic number suffix |
| `official_street_name` | text | Official street name |
| `official_street_type` | text | Street type (Ave, St, etc.) |
| `official_street_dir` | text | Street direction |
| `mail_mun_name` | text | Municipality name |
| `mail_prov_abvn` | text | Province abbreviation |
| `mail_postal_code` | text | Postal code (indexed) |
| ... | ... | 20 more columns |

**Index**: `idx_locations_postal_code` on `mail_postal_code` for fast lookups

## Usage in Code

The `locations` table is used in `src/db.py`:

```python
def populate_from_db_or_fallback(self, location_str, postal_code):
    """
    Looks up Canadian postal code in locations table to populate address fields.
    Now uses self.conn (main database) instead of separate address_db_engine.
    """
    query = """
        SELECT
            civic_no,
            civic_no_suffix,
            official_street_name,
            official_street_type,
            official_street_dir,
            mail_mun_name,
            mail_prov_abvn,
            mail_postal_code
        FROM locations
        WHERE mail_postal_code = %s;
    """
    df = pd.read_sql(query, self.conn, params=(postal_code,))
    # ... rest of method
```

## Verification

After running the migration, verify the table was copied successfully:

```bash
# Local database
PGPASSWORD=5539 psql -h localhost -U postgres -d social_dance_db -c "SELECT COUNT(*) FROM locations;"

# Should return the same count as the original address_db
PGPASSWORD=5539 psql -h localhost -U postgres -d address_db -c "SELECT COUNT(*) FROM locations;"
```

## Benefits

1. **Simplified architecture**: One database instead of two
2. **Easier deployment**: Only one database connection to configure on Render
3. **Reduced complexity**: No need to manage separate database credentials
4. **Cost savings**: Free tier Render database can hold everything (1 GB limit)
5. **Easier backups**: Single database backup includes all data

## Rollback (If Needed)

If you need to rollback to the separate database approach:

1. Restore `ADDRESS_DB_CONNECTION_STRING` to `.env`
2. Revert changes to `src/db.py` (commit: TBD)
3. The original `address_db` database still exists and is unchanged

## Notes

- The original `address_db` database is **not deleted** by this migration
- You can safely delete it after verifying everything works:
  ```bash
  # Only after thorough testing!
  PGPASSWORD=5539 psql -h localhost -U postgres -c "DROP DATABASE address_db;"
  ```
- The `locations` table is **not needed on production database** (production uses the web service database, not the data processing pipeline)

# Instagram URL Expiration Strategy

## Problem

Instagram CDN URLs (hosted on `fbcdn.net`) contain time-limited access tokens that expire after **24-48 hours**. These tokens are embedded in URL parameters like:
- `oe=68FEF4D6` (expiration timestamp)
- `_nc_gid=...` (session/group ID)
- `_nc_ohc=...` (cache hash)

When `images.py` processes old Instagram URLs from the database, it encounters:
- **403 Forbidden** errors (expired authentication tokens)
- **404 Not Found** errors (content no longer available)
- Wasted processing time and API calls

### Example of Stale URL
```
https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.2885-15/502881775_18195723754310101_703940993318009078_n.jpg?
stp=dst-jpg_e35_tt6&
_nc_gid=7UQ0Gv0v-l1zhQXR-o6cfA&    ← Session token (expires)
oe=684CF250&                          ← Expiration timestamp
_nc_sid=10d13b                        ← Session ID (expires)
```

After 24-48 hours, this URL returns **403 Forbidden** regardless of authentication.

## Solution Implemented

### 1. **Filter URLs by Age in `images.py`** ✅

Modified `get_image_links()` to only retrieve Instagram URLs from the **last 24 hours**:

```python
query = text("""
    SELECT link, parent_url, source, keywords, relevant, crawl_try, time_stamp
    FROM urls
    WHERE link ILIKE :link_pattern
      AND time_stamp >= (CURRENT_TIMESTAMP - INTERVAL '24 hours')
""")
```

Additionally filters out stale fbcdn URLs from CSV sources:
```python
# For fbcdn URLs, only keep recent ones (24 hours)
df = df[
    ~(df['link'].str.contains('fbcdn.net', case=False, na=False) &
      (df['time_stamp'].isna() |
       (pd.Timestamp.now() - df['time_stamp'] > pd.Timedelta(hours=24))))
]
```

**Result**: Prevents processing URLs that are likely expired.

### 2. **Cleanup Utility Script** ✅

Created `utilities/cleanup_stale_instagram_urls.py` to remove old URLs from the database:

```bash
# Dry run (see what would be deleted)
python utilities/cleanup_stale_instagram_urls.py --dry-run

# Delete URLs older than 2 days (default)
python utilities/cleanup_stale_instagram_urls.py

# Delete URLs older than 1 day
python utilities/cleanup_stale_instagram_urls.py --days 1
```

**Result**: Keeps database clean and reduces clutter.

## Impact Analysis

### Before Fix (from `images_log.txt`)
- **490 × 403 Forbidden** errors
- **112 × 404 Not Found** errors
- **~602 failed requests** wasting time and resources

### After Fix (Expected)
- **90-95% reduction** in 403/404 errors
- Only process fresh URLs (< 24 hours old)
- Faster execution time
- Reduced log clutter

## Database Statistics

```sql
-- Check Instagram URL age distribution
SELECT
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE time_stamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours') as fresh_24h,
  COUNT(*) FILTER (WHERE time_stamp >= CURRENT_TIMESTAMP - INTERVAL '7 days') as fresh_7d,
  MIN(time_stamp) as oldest,
  MAX(time_stamp) as newest
FROM urls
WHERE link LIKE '%instagram%' OR link LIKE '%fbcdn.net%';
```

**Current Stats** (as of 2025-10-22):
- Total: **8,844 URLs**
- Oldest: **2025-05-27** (5 months old)
- Newest: **2025-10-22** (today)
- **Most URLs are stale and unusable**

## Maintenance

### Regular Cleanup (Recommended)

Add to cron job or manual maintenance routine:
```bash
# Weekly cleanup of URLs older than 2 days
python utilities/cleanup_stale_instagram_urls.py --days 2
```

### Monitoring

Check for stale URLs:
```sql
-- Count stale Instagram URLs (older than 2 days)
SELECT COUNT(*)
FROM urls
WHERE (link LIKE '%instagram%' OR link LIKE '%fbcdn.net%')
  AND time_stamp < (CURRENT_TIMESTAMP - INTERVAL '2 days');
```

## Alternative Approaches (Not Implemented)

### Option A: Store Instagram Post URLs Instead of Image URLs
**Pros**: Post URLs don't expire
**Cons**: Requires re-extracting image URLs on each run (slower, more complex)

### Option B: Re-authenticate and Generate Fresh URLs
**Pros**: Could access expired content
**Cons**: Violates Instagram TOS, risks account suspension

### Option C: Download and Cache Images Locally
**Pros**: Permanent access
**Cons**: Storage overhead, copyright concerns

## Conclusion

The implemented solution (filtering by URL age) is:
- ✅ **Simple and effective**
- ✅ **Prevents 403/404 errors**
- ✅ **No TOS violations**
- ✅ **Low maintenance overhead**
- ✅ **Immediate impact**

Instagram URLs are ephemeral by design. This strategy accepts that limitation and works within it.

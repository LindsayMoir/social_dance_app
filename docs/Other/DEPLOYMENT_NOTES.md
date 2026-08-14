# Deployment Notes - Database Configuration

## Environment Variable: DATABASE_TARGET

The codebase now auto-detects which database to use based on the environment:

### Auto-Detection Rules:
- **Local machine** (RENDER not set) → `DATABASE_TARGET='local'`
- **Render** (RENDER='true') → `DATABASE_TARGET='render_dev'`

### When to Override:

#### 1. Production Web Services on Render
The **frontend** and **backend** web services should use the production database.

**Required Action:** Set environment variable in Render Dashboard:
```
DATABASE_TARGET=render_prod
```

**Where to set:**
- Go to Render Dashboard → "backend" service → Environment
- Go to Render Dashboard → "frontend" service → Environment
- Add: `DATABASE_TARGET` = `render_prod`

#### 2. Cron Jobs on Render
Cron jobs should use the development database (default behavior).

**No action needed** - auto-detection uses `render_dev` ✅

#### 3. Local Development
Local development should use local PostgreSQL (default behavior).

**No action needed** - auto-detection uses `local` ✅
- Requires `DATABASE_CONNECTION_STRING` in `src/.env`

### Summary Table:

| Environment | RENDER | DATABASE_TARGET | Result Database | Action |
|------------|--------|-----------------|-----------------|--------|
| Local dev | not set | (auto) | local | ✅ None |
| Render cron | true | (auto) | render_dev | ✅ None |
| Render web | true | **render_prod** | render_prod | ⚠️ Set in dashboard |

### Copy Dev to Prod Pipeline Step

The `copy_dev_to_prod` step in `pipeline.py`:
- **Source**: Whatever DATABASE_TARGET points to (local or render_dev)
- **Target**: ALWAYS Render Production (render_prod)
- **Logic**: Automatically copies from your working environment to production

No need to change DATABASE_TARGET before running the copy - it will copy from wherever you're working to production.

---

## Troubleshooting

If web services are reading from the wrong database:
1. Check Render Dashboard → Service → Environment variables
2. Ensure `DATABASE_TARGET=render_prod` is set for frontend and backend
3. Restart the services after adding the variable

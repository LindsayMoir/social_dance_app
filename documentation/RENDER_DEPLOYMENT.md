# Render Deployment Guide

This guide explains how to deploy the Social Dance App to Render with automatic secret management.

## Overview

The app uses two types of secrets:
1. **Environment Variables**: API keys, database URLs, and simple configuration values
2. **Secret Files**: JSON authentication files (Facebook, Eventbrite, Gmail credentials)

## Architecture

### Local Development
- Reads `.env` file from `src/.env`
- Reads JSON auth files from current directory (e.g., `facebook_auth.json`)
- Reads Gmail credentials from paths specified in `.env`

### Render Production
- Environment variables synced automatically via GitHub Actions
- Secret files uploaded manually to Render (one-time setup)
- Code automatically uses Render paths when available

## Setup Instructions

### Step 1: Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

1. **RENDER_API_KEY**
   - Get from: Render Dashboard → Account Settings → API Keys
   - Create a new API key if needed

2. **RENDER_SERVICE_ID**
   - Get from: Your Render service URL
   - Format: `srv-xxxxxxxxxxxxxxxxxxxxx`
   - Find it in the Render service URL: `https://dashboard.render.com/web/srv-xxxxxxxxxxxxxxxxxxxxx`

### Step 2: Upload Secret Files to Render

Upload these JSON files as **Secret Files** in Render Dashboard:

| Local File | Upload to Render as | Mount Path on Render |
|------------|-------------------|---------------------|
| `facebook_auth.json` | `facebook_auth.json` | `/etc/secrets/facebook_auth.json` |
| `eventbrite_auth.json` | `eventbrite_auth.json` | `/etc/secrets/eventbrite_auth.json` |
| `instagram_auth.json` | `instagram_auth.json` | `/etc/secrets/instagram_auth.json` |
| Gmail client secret file | `desktop_client_secret.json` | `/etc/secrets/desktop_client_secret.json` |
| Gmail token file | `desktop_client_secret_token.json` | `/etc/secrets/desktop_client_secret_token.json` |

**How to upload:**
1. Go to Render Dashboard → Your Service → Environment
2. Scroll to "Secret Files" section
3. Click "Add Secret File"
4. Set filename (e.g., `facebook_auth.json`)
5. Paste the file contents
6. Click "Save"

### Step 3: Initial Environment Variable Setup

For the **first deployment**, you need to manually add environment variables to Render:

1. Go to Render Dashboard → Your Service → Environment
2. Click "Add Environment Variable"
3. Add each variable from `src/.env` (except those ending with `_PATH` that point to JSON files)

**Important:** After this initial setup, the GitHub Action will automatically sync changes.

### Step 4: Enable Automatic Sync

The GitHub Action (`.github/workflows/sync-env-to-render.yml`) automatically runs when:
- You push changes to `src/.env` on the `main` branch
- You manually trigger it from GitHub Actions tab

**Manual trigger:**
1. Go to GitHub → Actions → "Sync Environment Variables to Render"
2. Click "Run workflow"
3. Select branch and click "Run workflow"

## How the Code Works

### Secret Path Resolution

The `src/secret_paths.py` utility automatically resolves secret paths:

```python
from secret_paths import get_auth_file, get_secret_path

# For auth files (facebook_auth.json, eventbrite_auth.json, etc.)
facebook_auth = get_auth_file('facebook')
# Returns: /etc/secrets/facebook_auth.json on Render
# Returns: facebook_auth.json locally

# For custom paths
gmail_secret = get_secret_path('desktop_client_secret.json', local_path_from_env)
# Returns: /etc/secrets/desktop_client_secret.json on Render
# Returns: path from .env locally
```

### Updated Files

These files have been updated to support Render Secret Files:
- `src/fb.py` - Facebook authentication
- `src/emails.py` - Gmail API access
- `src/rd_ext.py` - General website authentication

## Environment Variables vs Secret Files

### Use Environment Variables for:
- ✅ API keys (Anthropic, OpenAI, Google, etc.)
- ✅ Database connection strings
- ✅ Simple configuration values
- ✅ URLs and endpoints

### Use Secret Files for:
- ✅ OAuth JSON credentials (Facebook, Google, etc.)
- ✅ Service account keys
- ✅ Certificate files
- ✅ Any multi-line JSON/XML configuration

## Secret Files to Upload

Based on your `.env` file, upload these as Secret Files:

1. **facebook_auth.json**
   - Facebook Playwright session storage
   - Used by: `src/fb.py`, `src/rd_ext.py`

2. **eventbrite_auth.json**
   - Eventbrite Playwright session storage
   - Used by: `src/rd_ext.py`

3. **instagram_auth.json**
   - Instagram Playwright session storage
   - Used by: `src/rd_ext.py`

4. **desktop_client_secret.json**
   - Google OAuth client secret for Gmail API
   - Currently at: `/mnt/d/OneDrive/Security/google/desktop_client_secret.json`
   - Used by: `src/emails.py`

5. **desktop_client_secret_token.json**
   - Google OAuth token for Gmail API
   - Currently at: `/mnt/d/OneDrive/Security/google/desktop_client_secret_token.json`
   - Used by: `src/emails.py`

6. **google_calendar_app_client_secret.json** (if used)
   - Google Calendar API credentials
   - Currently at: `/mnd/d/OneDrive/Security/Google/google_calendar_app_client_secret.json`

## Environment Variables to Sync

The GitHub Action syncs all variables from `src/.env` except:
- Lines starting with `#` (comments)
- Variables ending with `_PATH` that contain `.json` (handled as Secret Files)
- Empty lines

**Auto-synced variables include:**
- All API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Database credentials (DATABASE_CONNECTION_STRING, etc.)
- Service configuration (BRAVE_API_KEY, FOURSQUARE_API_KEY, etc.)

## Updating Secrets

### To Update Environment Variables:
1. Edit `src/.env` locally
2. Commit and push to `main` branch
3. GitHub Action automatically syncs to Render
4. Render auto-deploys with new values

### To Update Secret Files:
1. Go to Render Dashboard → Your Service → Environment
2. Find the Secret File in the list
3. Click "Edit"
4. Update the contents
5. Click "Save Changes"
6. Manually trigger a deploy if auto-deploy is disabled

## Troubleshooting

### Environment variables not syncing
- Check GitHub Actions logs for errors
- Verify `RENDER_API_KEY` and `RENDER_SERVICE_ID` are set correctly
- Check if the service ID matches your Render service

### Secret files not found
- Verify files are uploaded to Render Secret Files (not environment variables)
- Check file names match exactly (case-sensitive)
- Look for logs like "Using Render secret file: /etc/secrets/..."

### Code still looking in wrong location
- Check logs to see which path is being used
- Verify `RENDER=true` environment variable is set in Render
- Ensure `secret_paths.py` is imported correctly

## Security Best Practices

1. ✅ Never commit `.env` files to Git (already in `.gitignore`)
2. ✅ Never commit JSON auth files to Git (already in `.gitignore`)
3. ✅ Rotate API keys regularly
4. ✅ Use Render's Secret Files for sensitive JSON data
5. ✅ Use GitHub Secrets for automation credentials
6. ✅ Enable 2FA on GitHub, Render, and all API provider accounts

## Testing

### Test locally:
```bash
cd src
python pipeline.py
# Should read from local .env and local JSON files
```

### Test on Render:
1. Check deployment logs for "Using Render secret file: /etc/secrets/..."
2. Verify no "file not found" errors for auth files
3. Check that API calls work correctly

## Additional Resources

- [Render Secret Files Documentation](https://render.com/docs/configure-environment-variables#secret-files)
- [Render API Documentation](https://api-docs.render.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

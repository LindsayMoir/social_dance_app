# Credential Validator Design

## Overview
A pre-flight credential validation program that runs before pipeline.py to ensure Gmail, Eventbrite, and Facebook credentials are valid. This prevents mid-pipeline failures caused by expired tokens, session timeouts, CAPTCHAs, or 2FA challenges.

## Problem Statement
The pipeline.py program frequently fails mid-run due to credential issues in:
- `emails.py` (Gmail OAuth tokens)
- `ebs.py` (Eventbrite session cookies)
- `fb.py` (Facebook session cookies)

These services irregularly challenge for:
- Updated credentials
- CAPTCHA solving
- 2FA verification
- Login prompts that require human interaction

When these challenges occur in headless mode, the pipeline fails.

## Solution Architecture

### Program Name
`credential_validator.py`

### Integration Point
The validator runs **automatically before every pipeline.py execution**.

### Execution Flow
```
1. pipeline.py starts
2. Call credential_validator.py with headless=False
3. Validator quickly checks each service (~60 seconds max per service)
4. If credentials invalid, browser stays open for user to fix (no time limit)
5. Credentials validated and saved
6. credential_validator.py completes
7. pipeline.py continues with headless=True (normal operation)
```

### Critical Headless Mode Switching
- **During validation**: headless=False (browser visible for user interaction)
- **After validation**: headless=True (pipeline runs normally in headless mode)

## Technical Design

### 1. File Location
```
src/credential_validator.py
```

### 2. Function Signature
```python
def validate_credentials(headless: bool = False, check_timeout_seconds: int = 60) -> dict:
    """
    Validates credentials for Gmail, Eventbrite, and Facebook.

    Args:
        headless: Whether to run browser in headless mode (False for validation)
        check_timeout_seconds: Max time to spend checking each service (default: 60)
                              Does NOT limit user time to update credentials

    Returns:
        dict: {
            'gmail': {'valid': bool, 'error': str or None},
            'eventbrite': {'valid': bool, 'error': str or None},
            'facebook': {'valid': bool, 'error': str or None}
        }
    """
```

### 3. Validation Logic Per Service

#### Gmail (OAuth Token)
- **Import existing logic from**: `src/emails.py`
- **Method**: Reuse the OAuth flow in emails.py
- **Quick check**: Attempt to authenticate/refresh token (max 60 seconds)
- **If invalid**: Opens browser (headless=False) and waits for user to complete OAuth flow (no time limit)
- **Saves to**: Updates OAuth token files via existing emails.py mechanism

```python
def validate_gmail(headless=False, check_timeout_seconds=60):
    """
    Validates Gmail OAuth credentials.
    Uses existing emails.py OAuth flow.

    Steps:
    1. Quick check: Try to authenticate (max check_timeout_seconds)
    2. If invalid: Open browser and wait for user (no time limit)
    3. Save credentials
    """
    try:
        # Quick check: Attempt to authenticate/refresh token
        # If successful within check_timeout_seconds, return success
        # If failed or timeout, open browser for user to complete OAuth
        # Wait indefinitely for user to complete (no timeout)
        # Return validation result
    except Exception as e:
        return {'valid': False, 'error': str(e)}
```

#### Eventbrite (Session Cookie)
- **Import existing logic from**: `src/ebs.py`
- **Auth file**: `eventbrite_auth.json`
- **Quick check**: Load session and verify with test request (max 60 seconds)
- **If invalid**: Opens Playwright browser (headless=False) and waits for user to login (no time limit)
- **Saves to**: `eventbrite_auth.json`

```python
def validate_eventbrite(headless=False, check_timeout_seconds=60):
    """
    Validates Eventbrite session credentials.
    Uses existing ebs.py authentication logic.

    Steps:
    1. Quick check: Load session and test (max check_timeout_seconds)
    2. If invalid: Open browser and wait for user (no time limit)
    3. Save session to eventbrite_auth.json
    """
    try:
        # Quick check: Load eventbrite_auth.json and verify session
        # If valid within check_timeout_seconds, return success
        # If invalid, open Playwright browser (headless=False)
        # Wait indefinitely for user to login (no timeout)
        # Save new session to eventbrite_auth.json
        # Return validation result
    except Exception as e:
        return {'valid': False, 'error': str(e)}
```

#### Facebook (Session Cookie)
- **Import existing logic from**: `src/fb.py`
- **Auth file**: `facebook_auth.json`
- **Quick check**: Load session and verify with test request (max 60 seconds)
- **If invalid**: Opens Playwright browser (headless=False) and waits for user to login (no time limit)
- **Saves to**: `facebook_auth.json`

```python
def validate_facebook(headless=False, check_timeout_seconds=60):
    """
    Validates Facebook session credentials.
    Uses existing fb.py authentication logic.

    Steps:
    1. Quick check: Load session and test (max check_timeout_seconds)
    2. If invalid: Open browser and wait for user (no time limit)
    3. Save session to facebook_auth.json
    """
    try:
        # Quick check: Load facebook_auth.json and verify session
        # If valid within check_timeout_seconds, return success
        # If invalid, open Playwright browser (headless=False)
        # Wait indefinitely for user to login (no timeout)
        # Save new session to facebook_auth.json
        # Return validation result
    except Exception as e:
        return {'valid': False, 'error': str(e)}
```

### 4. Main Validation Function

```python
def validate_credentials(headless=False, check_timeout_seconds=60):
    """
    Validates all service credentials sequentially.

    Args:
        check_timeout_seconds: Max time to spend checking if credential is valid.
                              Does NOT limit user time to update credentials.
    """
    results = {}

    logging.info("Starting credential validation...")

    # Validate Gmail
    logging.info("Validating Gmail credentials...")
    results['gmail'] = validate_gmail(headless, check_timeout_seconds)
    if not results['gmail']['valid']:
        logging.error(f"Gmail validation failed: {results['gmail']['error']}")
        return results

    # Validate Eventbrite
    logging.info("Validating Eventbrite credentials...")
    results['eventbrite'] = validate_eventbrite(headless, check_timeout_seconds)
    if not results['eventbrite']['valid']:
        logging.error(f"Eventbrite validation failed: {results['eventbrite']['error']}")
        return results

    # Validate Facebook
    logging.info("Validating Facebook credentials...")
    results['facebook'] = validate_facebook(headless, check_timeout_seconds)
    if not results['facebook']['valid']:
        logging.error(f"Facebook validation failed: {results['facebook']['error']}")
        return results

    # Log summary
    logging.info("All credentials validated successfully")
    return results
```

### 5. Integration with pipeline.py

**Location**: Add to `src/pipeline.py` at the beginning of the main execution

```python
# At top of pipeline.py
from credential_validator import validate_credentials

def main():
    # STEP 0: Validate credentials with headless=False
    logging.info("=" * 50)
    logging.info("STEP 0: Validating credentials")
    logging.info("=" * 50)

    validation_results = validate_credentials(
        headless=False,
        check_timeout_seconds=60
    )

    # Check if all validations passed
    all_valid = all(r['valid'] for r in validation_results.values())
    if not all_valid:
        failed_services = [k for k, v in validation_results.items() if not v['valid']]
        logging.error(f"Credential validation failed for: {', '.join(failed_services)}")
        logging.error("Please check the credentials and try again")
        return

    logging.info("Credentials validated successfully. Continuing pipeline with headless=True")

    # STEP 1: Continue with normal pipeline (headless=True)
    # ... existing pipeline code continues here with headless=True
```

## User Interaction Flow

### Scenario 1: All Credentials Valid
```
1. Validator checks Gmail (max 60 sec) - Valid
2. Validator checks Eventbrite (max 60 sec) - Valid
3. Validator checks Facebook (max 60 sec) - Valid
4. Pipeline continues with headless=True
5. Total validation time: ~10-30 seconds
```

### Scenario 2: Facebook Session Expired
```
1. Validator checks Gmail (max 60 sec) - Valid
2. Validator checks Eventbrite (max 60 sec) - Valid
3. Validator checks Facebook (max 60 sec) - Invalid session detected
4. Facebook browser opens (headless=False)
5. User sees login prompt, enters credentials
6. User solves CAPTCHA if presented (no time limit)
7. Session saved to facebook_auth.json
8. Browser closes
9. Pipeline continues with headless=True
```

### Scenario 3: Multiple Credentials Invalid
```
1. Validator checks Gmail (max 60 sec) - Invalid
2. Gmail browser opens (headless=False)
3. User completes OAuth flow (no time limit)
4. Gmail credentials saved
5. Validator checks Eventbrite (max 60 sec) - Invalid
6. Eventbrite browser opens (headless=False)
7. User logs in and solves CAPTCHA (no time limit)
8. Eventbrite credentials saved
9. Validator checks Facebook (max 60 sec) - Invalid
10. Facebook browser opens (headless=False)
11. User logs in (no time limit)
12. Facebook credentials saved
13. Pipeline continues with headless=True
```

## Check Timeout vs User Time

### Check Timeout (60 seconds max)
- Time spent verifying if existing credentials are valid
- Making test requests to services
- Loading and parsing auth files
- If this takes longer than 60 seconds, treat as invalid and open browser

### User Time (no limit)
- Time for user to complete login
- Time to solve CAPTCHAs
- Time to complete 2FA
- Time to update credentials
- **No timeout** - browser stays open until user completes authentication

```python
# Pseudocode example
def validate_service(service_name, headless=False, check_timeout_seconds=60):
    # Phase 1: Quick check (max check_timeout_seconds)
    start_time = time.time()
    is_valid = quick_check_credentials(service_name, timeout=check_timeout_seconds)

    if is_valid:
        # Credentials are good, return immediately
        return {'valid': True, 'error': None}

    # Phase 2: Open browser for user to fix (no time limit)
    logging.info(f"{service_name} credentials invalid. Opening browser for user update...")
    browser = open_browser(headless=False)

    # Wait indefinitely for user to complete authentication
    while not authenticated(browser):
        time.sleep(1)  # Poll until user completes

    save_credentials(service_name)
    browser.close()

    return {'valid': True, 'error': None}
```

## Out of Scope

The following features are explicitly NOT included:
- ❌ Credential expiry tracking
- ❌ Email/Slack notifications
- ❌ Multi-account support (single account only)
- ❌ Automatic retry logic
- ❌ Scheduled validation runs (only runs before pipeline.py)
- ❌ Time limit for user to update credentials (user can take as long as needed)

## Logging

All validation activity will be logged to the main pipeline log:
```
INFO: Starting credential validation...
INFO: Validating Gmail credentials...
INFO: Quick check: Gmail credentials valid (12 seconds)
INFO: Validating Eventbrite credentials...
INFO: Quick check: Eventbrite session invalid (58 seconds)
INFO: Opening browser for user to update Eventbrite credentials...
INFO: User completed Eventbrite login. Credentials saved.
INFO: Validating Facebook credentials...
INFO: Quick check: Facebook credentials valid (8 seconds)
INFO: All credentials validated successfully
INFO: Credentials validated successfully. Continuing pipeline with headless=True
```

## File Changes Required

### New Files
1. `src/credential_validator.py` - Main validation program

### Modified Files
1. `src/pipeline.py` - Add credential validation step at beginning
   - Import credential_validator
   - Call validate_credentials(headless=False, check_timeout_seconds=60)
   - Check validation results
   - Continue with headless=True for normal pipeline execution

### Files Referenced (No Changes)
1. `src/emails.py` - Gmail OAuth logic (reused, not modified)
2. `src/ebs.py` - Eventbrite auth logic (reused, not modified)
3. `src/fb.py` - Facebook auth logic (reused, not modified)
4. `eventbrite_auth.json` - Updated by validator
5. `facebook_auth.json` - Updated by validator
6. Gmail OAuth token files - Updated by validator

## Testing Plan

### Manual Testing
1. Delete all auth files (`*_auth.json`)
2. Run pipeline.py
3. Verify validator opens browsers with headless=False
4. Complete login for each service (take as much time as needed)
5. Verify credentials are saved
6. Verify pipeline continues with headless=True

### Edge Cases to Test
1. All credentials valid - fast pass-through (~30 seconds total)
2. One credential invalid - user fixes it (no time pressure)
3. Multiple credentials invalid - user fixes all (no time pressure)
4. CAPTCHA challenge - user solves it (no time limit)
5. 2FA challenge - user completes it (no time limit)
6. User takes 5 minutes to update credentials - should work fine

## Implementation Priority

### Phase 1: Core Validation
1. Create `credential_validator.py` with basic structure
2. Implement Gmail validation (reuse emails.py logic)
3. Implement Eventbrite validation (reuse ebs.py logic)
4. Implement Facebook validation (reuse fb.py logic)
5. Implement quick check with check_timeout_seconds
6. Implement browser open for user update (no time limit)

### Phase 2: Integration
1. Integrate into pipeline.py
2. Add proper logging
3. Add check timeout handling
4. Add validation result checking

### Phase 3: Testing
1. Test with valid credentials (fast path)
2. Test with invalid credentials (user update path)
3. Test check timeout scenarios
4. Test CAPTCHA scenarios
5. Test user taking extended time to update

## Success Metrics

The validator is successful if:
1. ✅ Runs automatically before every pipeline.py execution
2. ✅ Opens browser with headless=False during validation
3. ✅ Spends max 60 seconds checking each service for validity
4. ✅ Allows user unlimited time to update credentials when needed
5. ✅ Saves updated credentials to `*_auth.json` files
6. ✅ Pipeline continues with headless=True after validation
7. ✅ Reduces mid-pipeline credential failures to near-zero
8. ✅ Doesn't pressure user with time limits during credential updates

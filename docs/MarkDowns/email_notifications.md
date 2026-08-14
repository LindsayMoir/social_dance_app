# Email Notifications for Validation Reports

## Overview

The validation testing framework can automatically send email notifications with test reports after each run. This allows you to receive immediate alerts about test failures and review detailed results without manually checking log files.

## Features

- **Automatic Email Sending**: Emails sent after validation tests complete
- **HTML Formatted**: Easy-to-read HTML email body with color-coded status
- **Multiple Attachments**: Includes JSON reports, CSV results, and HTML reports
- **Status Indicators**: ✓ PASS, ⚠ WARNING, ✗ FAIL with visual color coding
- **Multiple Recipients**: Support for comma-separated recipient list
- **Secure SMTP**: Uses TLS encryption for secure email transmission

## Setup

### Step 1: Configure Environment Variables

Copy `.env.example` to `.env` (if not already done):

```bash
cp .env.example .env
```

### Step 2: Add Email Configuration

Edit `.env` and add the following email settings:

```bash
# Email Notification Configuration
NOTIFICATION_EMAIL_FROM=myapp@gmail.com
NOTIFICATION_EMAIL_TO=admin@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=myapp@gmail.com
SMTP_PASSWORD=your_app_specific_password
```

### Step 3: Gmail-Specific Setup (Most Common)

If using Gmail:

1. **Enable 2-Factor Authentication** on your Google account
   - Go to: https://myaccount.google.com/security
   - Enable "2-Step Verification"

2. **Generate App Password**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the 16-character password
   - Use this as `SMTP_PASSWORD` in `.env` (NOT your regular Gmail password!)

3. **Example Gmail Configuration:**
   ```bash
   NOTIFICATION_EMAIL_FROM=socialdanceapp@gmail.com
   NOTIFICATION_EMAIL_TO=admin@example.com
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=socialdanceapp@gmail.com
   SMTP_PASSWORD=abcd efgh ijkl mnop  # App-specific password
   ```

### Step 4: Other Email Providers

**Microsoft Outlook/Office 365:**
```bash
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
```

**Yahoo Mail:**
```bash
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
```

**SendGrid:**
```bash
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USERNAME=apikey
SMTP_PASSWORD=your_sendgrid_api_key
```

## Email Content

### Subject Line Format
```
[✓ PASS] Pre-Commit Validation Report - 2026-01-23 15:30
[⚠ WARNING] Pre-Commit Validation Report - 2026-01-23 15:30
[✗ FAIL] Pre-Commit Validation Report - 2026-01-23 15:30
```

### Email Body Includes:
- **Overall Status**: Color-coded status indicator
- **Summary Statistics Table**:
  - Total tests executed
  - Execution success rate
  - Average score
  - Interpretation pass rate
  - Total failures
  - Whitelist failures
- **Timestamp**: When tests were run

### Attachments:
1. **chatbot_evaluation_report.json** - Detailed chatbot test results
2. **chatbot_test_results.csv** - All test results in CSV format
3. **scraping_validation_report.json** - Scraping failure details
4. **comprehensive_test_report.html** - Full HTML report (viewable in browser)

## Usage

### Automatic (via Pipeline)

Email notifications are sent automatically when validation tests run:

```bash
python src/pipeline.py
```

The `validation_step` in the pipeline will run tests and send email on completion.

### Manual (Standalone)

Run validation tests directly:

```bash
python tests/validation/test_runner.py
```

Email will be sent after tests complete.

### Testing Email Configuration

Test your email setup without running full validations:

```python
from src.email_notifier import EmailNotifier

notifier = EmailNotifier()
if notifier.enabled:
    print("✓ Email notifications configured correctly")
    notifier.send_validation_report(
        report_summary={
            'overall_status': 'TEST',
            'total_tests': 10,
            'execution_success_rate': 1.0,
            'average_score': 85.5
        },
        test_type="Email Configuration Test"
    )
else:
    print("✗ Email notifications not configured")
```

## Disabling Email Notifications

Email notifications are **optional** and won't block test execution if not configured.

To disable:
1. **Remove email variables** from `.env`, OR
2. **Leave them unset** - tests will run normally without sending email

Log message when disabled:
```
Email notification skipped (not configured or failed)
```

## Troubleshooting

### "Authentication failed"
- **Gmail**: Ensure you're using an app-specific password, not your regular password
- **2FA**: Enable 2-factor authentication first on Gmail
- **Less secure apps**: Don't use this option (deprecated by Google)

### "Connection refused" or "Timeout"
- Check `SMTP_HOST` and `SMTP_PORT` are correct for your provider
- Ensure firewall allows outbound SMTP connections (port 587)
- Try port 465 (SSL) instead of 587 (TLS)

### "Email not received"
- Check spam/junk folder
- Verify `NOTIFICATION_EMAIL_TO` is correct
- Check email provider's sent folder to confirm it was sent
- Review logs for "Email sent successfully" message

### Test Email Configuration
```bash
python -c "
from src.email_notifier import EmailNotifier
notifier = EmailNotifier()
print(f'Enabled: {notifier.enabled}')
print(f'From: {notifier.from_email}')
print(f'To: {notifier.to_emails}')
print(f'SMTP: {notifier.smtp_host}:{notifier.smtp_port}')
"
```

## Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore` for security
2. **Use app-specific passwords** - Not your main account password
3. **Rotate credentials regularly** - Update passwords periodically
4. **Limit recipients** - Only send to authorized personnel
5. **TLS encryption enabled** - Always use port 587 (or 465 for SSL)

## Status Determination

Email subject line status is determined by:

| Metric | PASS | WARNING | FAIL |
|--------|------|---------|------|
| Execution Success Rate | ≥ 95% | ≥ 90% | < 90% |
| Average Score | ≥ 75 | ≥ 70 | < 70 |

**Examples:**
- 98% execution rate, 85 avg score → ✓ PASS
- 92% execution rate, 72 avg score → ⚠ WARNING
- 85% execution rate, 65 avg score → ✗ FAIL

## Integration with Pipeline

The email notification runs as the final step in validation:

```
1. Scraping Validation
2. Chatbot Testing
3. Generate HTML Report
4. Send Email Notification  ← Added here
5. Return Results
```

If email sending fails, **it won't fail the entire pipeline** - tests continue normally.

## Future Enhancements

Potential additions (not yet implemented):
- Slack/Discord webhook notifications
- SMS alerts for critical failures
- Configurable email templates
- Filtered reports (only send on FAIL)
- Daily digest emails
- Email rate limiting

## Support

For issues with email notifications:
1. Check logs for detailed error messages
2. Verify all environment variables are set correctly
3. Test with a simple configuration test (see "Testing Email Configuration" above)
4. Review provider-specific SMTP documentation

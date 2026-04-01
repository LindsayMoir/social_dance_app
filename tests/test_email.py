"""
Quick test script to verify email notification configuration.
Sends a test email to verify SMTP settings are correct.
"""
import sys
sys.path.insert(0, 'src')

from email_notifier import EmailNotifier
import logging
from logging_config import setup_logging

# Setup logging
setup_logging('test_email')

def test_email_notification():
    """Test email notification system with sample data."""

    print("Initializing EmailNotifier...")
    notifier = EmailNotifier()

    if not notifier.enabled:
        print("❌ Email notifications not configured properly")
        print(f"From: {notifier.from_email}")
        print(f"To: {notifier.to_emails}")
        print(f"SMTP: {notifier.smtp_host}:{notifier.smtp_port}")
        print(f"Username: {notifier.smtp_username}")
        print(f"Password configured: {'Yes' if notifier.smtp_password else 'No'}")
        return False

    print("✓ Email configuration loaded successfully")
    print(f"  From: {notifier.from_email}")
    print(f"  To: {', '.join(notifier.to_emails)}")
    print(f"  SMTP: {notifier.smtp_host}:{notifier.smtp_port}")

    # Create test report summary
    test_summary = {
        'overall_status': 'TEST',
        'total_tests': 10,
        'execution_success_rate': 1.0,
        'average_score': 95.5,
        'interpretation_pass_rate': 1.0,
        'test_message': 'This is a test email from the Social Dance App validation system.'
    }

    print("\nSending test email...")
    success = notifier.send_validation_report(
        report_summary=test_summary,
        attachment_paths=None,  # No attachments for test
        test_type="Email Configuration Test"
    )

    if success:
        print("✓ Test email sent successfully!")
        print(f"  Check {', '.join(notifier.to_emails)} for the test message")
        return True
    else:
        print("❌ Failed to send test email")
        print("  Check logs for detailed error information")
        return False

if __name__ == "__main__":
    success = test_email_notification()
    sys.exit(0 if success else 1)

"""
Email notification system for sending test reports and alerts.

Sends email notifications with attachments for validation test results,
including chatbot evaluation reports and scraping validation reports.

Dependencies:
    - smtplib (standard library)
    - email (standard library)
    - python-dotenv for credentials
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmailNotifier:
    """
    Send email notifications with attachments for test reports.

    Configuration via environment variables:
        - NOTIFICATION_EMAIL_FROM: Sender email address
        - NOTIFICATION_EMAIL_TO: Recipient email address (comma-separated for multiple)
        - SMTP_HOST: SMTP server hostname (default: smtp.gmail.com)
        - SMTP_PORT: SMTP server port (default: 587)
        - SMTP_USERNAME: SMTP authentication username
        - SMTP_PASSWORD: SMTP authentication password (app-specific password recommended)

    Example .env file:
        NOTIFICATION_EMAIL_FROM=myapp@gmail.com
        NOTIFICATION_EMAIL_TO=admin@example.com
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USERNAME=myapp@gmail.com
        SMTP_PASSWORD=your_app_specific_password
    """

    def __init__(self):
        """Initialize EmailNotifier with configuration from environment variables."""
        self.from_email = os.getenv('NOTIFICATION_EMAIL_FROM')
        self.to_emails = os.getenv('NOTIFICATION_EMAIL_TO', '').split(',')
        self.to_emails = [email.strip() for email in self.to_emails if email.strip()]

        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')

        # Validate configuration
        if not all([self.from_email, self.to_emails, self.smtp_username, self.smtp_password]):
            logging.warning("Email notification not configured. Set environment variables: "
                          "NOTIFICATION_EMAIL_FROM, NOTIFICATION_EMAIL_TO, SMTP_USERNAME, SMTP_PASSWORD")
            self.enabled = False
        else:
            self.enabled = True
            logging.info(f"Email notifications enabled. Will send to: {', '.join(self.to_emails)}")


    def send_validation_report(
        self,
        report_summary: dict,
        attachment_paths: Optional[List[str]] = None,
        test_type: str = "Validation"
    ) -> bool:
        """
        Send validation test report via email.

        Args:
            report_summary (dict): Summary statistics from the report
            attachment_paths (Optional[List[str]]): List of file paths to attach
            test_type (str): Type of test (e.g., "Chatbot Validation", "Scraping Validation")

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logging.info("send_validation_report(): Email notifications disabled, skipping")
            return False

        try:
            # Build email subject
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            status = self._determine_status(report_summary)
            subject = f"[{status}] {test_type} Report - {timestamp}"

            # Build email body
            body = self._build_email_body(report_summary, test_type)

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject

            # Attach body as HTML
            msg.attach(MIMEText(body, 'html'))

            # Attach files if provided
            if attachment_paths:
                for file_path in attachment_paths:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            attachment = MIMEApplication(f.read())
                            attachment.add_header(
                                'Content-Disposition',
                                'attachment',
                                filename=os.path.basename(file_path)
                            )
                            msg.attach(attachment)
                        logging.info(f"send_validation_report(): Attached {os.path.basename(file_path)}")
                    else:
                        logging.warning(f"send_validation_report(): Attachment not found: {file_path}")

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()  # Enable TLS encryption
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logging.info(f"send_validation_report(): Email sent successfully to {', '.join(self.to_emails)}")
            return True

        except Exception as e:
            logging.error(f"send_validation_report(): Failed to send email: {e}")
            return False


    def _determine_status(self, report_summary: dict) -> str:
        """
        Determine overall status from report summary.

        Args:
            report_summary (dict): Report summary with test results

        Returns:
            str: Status string (PASS, WARNING, or FAIL)
        """
        # Check for common status indicators
        if 'execution_success_rate' in report_summary:
            exec_rate = report_summary['execution_success_rate']
            avg_score = report_summary.get('average_score', 0)

            if exec_rate >= 0.95 and avg_score >= 75:
                return "✓ PASS"
            elif exec_rate >= 0.90 and avg_score >= 70:
                return "⚠ WARNING"
            else:
                return "✗ FAIL"

        # Default to neutral status
        return "INFO"


    def _build_email_body(self, report_summary: dict, test_type: str) -> str:
        """
        Build HTML email body from report summary.

        Args:
            report_summary (dict): Report summary with statistics
            test_type (str): Type of test being reported

        Returns:
            str: HTML-formatted email body
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build HTML body
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .metric {{ font-weight: bold; color: #2c3e50; }}
                .value {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h2>{test_type} Report</h2>
            <p><strong>Timestamp:</strong> {timestamp}</p>

            <h3>Summary Statistics</h3>
            <table>
        """

        # Add metrics to table
        for key, value in report_summary.items():
            # Format metric name (convert snake_case to Title Case)
            metric_name = key.replace('_', ' ').title()

            # Format value based on type
            if isinstance(value, float):
                if 0 <= value <= 1:
                    # Looks like a percentage
                    formatted_value = f"{value * 100:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, dict):
                # Skip nested dicts for now (will be in attached JSON)
                continue
            else:
                formatted_value = str(value)

            # Determine value class based on metric
            value_class = self._get_value_class(key, value)

            html += f"""
                <tr>
                    <td class="metric">{metric_name}</td>
                    <td class="{value_class}">{formatted_value}</td>
                </tr>
            """

        html += """
            </table>

            <p><strong>Note:</strong> Detailed results are attached to this email.</p>

            <div class="footer">
                <p>This is an automated message from the Social Dance App validation system.</p>
                <p>Generated by Claude Code</p>
            </div>
        </body>
        </html>
        """

        return html


    def _get_value_class(self, metric_key: str, value: any) -> str:
        """
        Determine CSS class for metric value based on thresholds.

        Args:
            metric_key (str): Metric name
            value: Metric value

        Returns:
            str: CSS class name ('value', 'warning', or 'error')
        """
        # Handle percentage metrics
        if isinstance(value, float) and 0 <= value <= 1:
            if value >= 0.95:
                return "value"
            elif value >= 0.90:
                return "warning"
            else:
                return "error"

        # Handle score metrics
        if 'score' in metric_key.lower() and isinstance(value, (int, float)):
            if value >= 80:
                return "value"
            elif value >= 70:
                return "warning"
            else:
                return "error"

        # Default
        return "value"


# Convenience function for quick email sending
def send_report_email(
    report_summary: dict,
    attachment_paths: Optional[List[str]] = None,
    test_type: str = "Validation"
) -> bool:
    """
    Convenience function to send validation report email.

    Args:
        report_summary (dict): Report summary statistics
        attachment_paths (Optional[List[str]]): Files to attach
        test_type (str): Type of test being reported

    Returns:
        bool: True if email sent successfully
    """
    notifier = EmailNotifier()
    return notifier.send_validation_report(report_summary, attachment_paths, test_type)

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
import re
import smtplib
import time
from contextlib import contextmanager
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Iterator, List, Optional
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


    @contextmanager
    def _smtp_stage(self, stage: str) -> Iterator[None]:
        """Log SMTP progress without exposing protocol payloads or credentials."""
        started = time.monotonic()
        logging.info("SMTP stage=%s started host=%s port=%s", stage, self.smtp_host, self.smtp_port)
        try:
            yield
        except Exception as exc:
            # Raw SMTP replies can echo addresses or message content. Retain only
            # numeric status codes, including the RFC enhanced status if present.
            reply = getattr(exc, 'smtp_error', b'')
            if isinstance(reply, bytes):
                reply = reply.decode('ascii', errors='replace')
            enhanced = re.search(r'\b[245]\.\d{1,3}\.\d{1,3}\b', str(reply))
            logging.error(
                "SMTP stage=%s failed elapsed_seconds=%.3f error_type=%s "
                "smtp_code=%s enhanced_status=%s errno=%s",
                stage, time.monotonic() - started, type(exc).__name__,
                getattr(exc, 'smtp_code', None),
                enhanced.group() if enhanced else None, getattr(exc, 'errno', None),
            )
            raise
        logging.info("SMTP stage=%s completed elapsed_seconds=%.3f", stage, time.monotonic() - started)

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

        started = time.monotonic()
        message_accepted = False
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

            logging.info(
                "SMTP delivery starting host=%s port=%s timeout_seconds=30 recipient_count=%s",
                self.smtp_host, self.smtp_port, len(self.to_emails),
            )
            with self._smtp_stage('connect'):
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)
            try:
                with self._smtp_stage('greeting'):
                    server.ehlo_or_helo_if_needed()
                with self._smtp_stage('starttls'):
                    server.starttls()
                with self._smtp_stage('tls_greeting'):
                    server.ehlo_or_helo_if_needed()
                with self._smtp_stage('authenticate'):
                    server.login(self.smtp_username, self.smtp_password)
                with self._smtp_stage('send_message'):
                    refused = server.send_message(msg)
                    message_accepted = True
                    logging.info(
                        "SMTP message accepted by server refused_recipient_count=%s",
                        len(refused),
                    )
                    if refused:
                        logging.warning("SMTP delivery partially refused; some recipients were rejected")
                with self._smtp_stage('quit'):
                    code, reply = server.quit()
                    if code != 221:
                        raise smtplib.SMTPResponseException(code, reply)
            finally:
                server.close()

            logging.info(f"send_validation_report(): Email sent successfully to {', '.join(self.to_emails)}")
            return True

        except Exception as exc:
            logging.error(
                "send_validation_report(): Failed to send email error_type=%s "
                "message_accepted=%s elapsed_seconds=%.3f",
                type(exc).__name__, message_accepted, time.monotonic() - started,
            )
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
            exec_rate = self._safe_float(report_summary.get('execution_success_rate'))
            avg_score = self._safe_float(report_summary.get('average_score'))

            if exec_rate is None:
                return "INFO"
            if avg_score is None:
                if exec_rate >= 0.95:
                    return "⚠ WARNING"
                if exec_rate >= 0.90:
                    return "⚠ WARNING"
                return "✗ FAIL"

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

    @staticmethod
    def _safe_float(value: object) -> float | None:
        """Convert numeric-like values to float and preserve missing values as None."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


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

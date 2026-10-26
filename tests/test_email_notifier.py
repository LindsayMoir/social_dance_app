from __future__ import annotations

import os
import sys
import logging
import smtplib
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_notifier import EmailNotifier


@pytest.fixture
def smtp_notifier(monkeypatch):
    for name, value in {
        'NOTIFICATION_EMAIL_FROM': 'sender@example.test',
        'NOTIFICATION_EMAIL_TO': 'recipient@example.test',
        'SMTP_HOST': 'smtp.example.test',
        'SMTP_PORT': '587',
        'SMTP_USERNAME': 'private-user',
        'SMTP_PASSWORD': 'private-password',
    }.items():
        monkeypatch.setenv(name, value)
    notifier = EmailNotifier()
    monkeypatch.setattr(notifier, '_build_email_body', lambda *_: 'private-report-body')
    server = MagicMock()
    server.send_message.return_value = {}
    server.quit.return_value = (221, b'Bye')
    factory = MagicMock(return_value=server)
    monkeypatch.setattr('email_notifier.smtplib.SMTP', factory)
    return notifier, server, factory


def test_smtp_success_logs_stages_and_attaches_report(smtp_notifier, tmp_path, caplog) -> None:
    notifier, server, factory = smtp_notifier
    report = tmp_path / 'report.html'
    report.write_text('private-attachment-content')
    with caplog.at_level(logging.INFO):
        assert notifier.send_validation_report({}, [str(report)]) is True

    factory.assert_called_once_with('smtp.example.test', 587, timeout=30)
    assert server.ehlo_or_helo_if_needed.call_count == 2
    server.close.assert_called_once()
    message = server.send_message.call_args.args[0]
    assert message.get_payload()[-1].get_payload(decode=True) == b'private-attachment-content'
    for stage in ('connect', 'greeting', 'starttls', 'tls_greeting', 'authenticate', 'send_message', 'quit'):
        assert f'SMTP stage={stage} completed elapsed_seconds=' in caplog.text
    assert 'SMTP message accepted by server refused_recipient_count=0' in caplog.text
    assert 'private-password' not in caplog.text
    assert 'private-report-body' not in caplog.text


@pytest.mark.parametrize('stage,method,error', [
    ('connect', None, smtplib.SMTPServerDisconnected('Connection unexpectedly closed')),
    ('greeting', 'ehlo_or_helo_if_needed', smtplib.SMTPHeloError(501, b'bad greeting')),
    ('starttls', 'starttls', smtplib.SMTPNotSupportedError('TLS unavailable')),
    ('authenticate', 'login', smtplib.SMTPAuthenticationError(535, b'5.7.8 private-password private-user')),
    ('send_message', 'send_message', smtplib.SMTPDataError(552, b'5.3.4 private-report-body')),
    ('send_message', 'send_message', TimeoutError('timed out')),
    ('quit', 'quit', smtplib.SMTPServerDisconnected('Connection unexpectedly closed')),
])
def test_smtp_failure_identifies_stage_without_sensitive_payloads(
    smtp_notifier, caplog, stage, method, error,
) -> None:
    notifier, server, factory = smtp_notifier
    if method is None:
        factory.side_effect = error
    else:
        getattr(server, method).side_effect = error
    with caplog.at_level(logging.INFO):
        assert notifier.send_validation_report({}) is False

    assert f'SMTP stage={stage} failed elapsed_seconds=' in caplog.text
    assert f'error_type={type(error).__name__}' in caplog.text
    assert f'message_accepted={stage == "quit"}' in caplog.text
    if isinstance(error, smtplib.SMTPAuthenticationError):
        assert 'smtp_code=535 enhanced_status=5.7.8' in caplog.text
    for secret in ('private-user', 'private-password', 'private-report-body'):
        assert secret not in caplog.text
    if method is not None:
        server.close.assert_called_once()
    else:
        server.close.assert_not_called()


def test_tls_greeting_failure_is_identified(smtp_notifier, caplog) -> None:
    notifier, server, _ = smtp_notifier
    server.ehlo_or_helo_if_needed.side_effect = [None, smtplib.SMTPServerDisconnected('closed')]
    assert notifier.send_validation_report({}) is False
    assert 'SMTP stage=tls_greeting failed' in caplog.text
    server.login.assert_not_called()
    server.close.assert_called_once()


def test_quit_rejection_records_message_already_accepted(smtp_notifier, caplog) -> None:
    notifier, server, _ = smtp_notifier
    server.quit.return_value = (421, b'4.3.2 service closing')
    assert notifier.send_validation_report({}) is False
    assert 'SMTP stage=quit failed' in caplog.text
    assert 'smtp_code=421 enhanced_status=4.3.2' in caplog.text
    assert 'message_accepted=True' in caplog.text


def test_partial_recipient_refusal_is_visible(smtp_notifier, caplog) -> None:
    notifier, server, _ = smtp_notifier
    server.send_message.return_value = {'private-recipient': (550, b'private-response')}
    with caplog.at_level(logging.INFO):
        assert notifier.send_validation_report({}) is True
    assert 'refused_recipient_count=1' in caplog.text
    assert 'SMTP delivery partially refused' in caplog.text
    assert 'private-recipient' not in caplog.text
    assert 'private-response' not in caplog.text


def test_determine_status_handles_missing_average_score() -> None:
    notifier = EmailNotifier.__new__(EmailNotifier)

    status = notifier._determine_status(
        {
            "execution_success_rate": 1.0,
            "average_score": None,
        }
    )

    assert status == "⚠ WARNING"


def test_determine_status_handles_missing_execution_rate() -> None:
    notifier = EmailNotifier.__new__(EmailNotifier)

    status = notifier._determine_status(
        {
            "execution_success_rate": None,
            "average_score": 85.0,
        }
    )

    assert status == "INFO"

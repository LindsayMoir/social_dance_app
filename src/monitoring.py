#!/usr/bin/env python3
"""
monitoring.py - Monitoring, Metrics, and Alerting System

Provides comprehensive monitoring infrastructure for production deployment:
- Prometheus metrics collection and exposition
- Error tracking and rate calculation
- Performance SLA monitoring
- Slack/PagerDuty webhook integration
- Custom event tracking
- Metrics aggregation and reporting

Supports multiple alert channels:
- Slack webhooks for notifications
- PagerDuty for incident management
- Email for administrative notifications
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import requests


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    RESOLVED = "resolved"


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """Alert event."""
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_slack_payload(self, webhook_url: str = None) -> Dict[str, Any]:
        """Convert to Slack message payload."""
        color_map = {
            AlertSeverity.INFO: "#0099ff",
            AlertSeverity.WARNING: "#ffaa00",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.RESOLVED: "#00aa00",
        }

        return {
            "attachments": [
                {
                    "color": color_map.get(self.severity, "#cccccc"),
                    "title": self.title,
                    "text": self.message,
                    "fields": [
                        {"title": "Component", "value": self.component, "short": True},
                        {"title": "Severity", "value": self.severity.value, "short": True},
                        {"title": "Time", "value": datetime.fromtimestamp(self.timestamp).isoformat(), "short": False},
                        {"title": "Context", "value": json.dumps(self.context, indent=2), "short": False},
                    ]
                }
            ]
        }

    def to_pagerduty_payload(self) -> Dict[str, Any]:
        """Convert to PagerDuty event payload."""
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.RESOLVED: "info",
        }

        return {
            "routing_key": os.environ.get("PAGERDUTY_KEY"),
            "event_action": "trigger" if self.severity != AlertSeverity.RESOLVED else "resolve",
            "dedup_key": f"{self.component}-{self.title}",
            "payload": {
                "summary": self.title,
                "severity": severity_map.get(self.severity, "info"),
                "source": "Scraper Monitoring",
                "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
                "custom_details": {
                    "message": self.message,
                    "component": self.component,
                    "context": self.context,
                }
            }
        }


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, window_size: int = 300):
        """Initialize metrics collector.

        Args:
            window_size: Time window in seconds for rolling metrics (default: 5 minutes)
        """
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        with self.lock:
            self.metrics[name].append(MetricValue(value))

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        with self.lock:
            self.counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self.lock:
            self.gauges[name] = value

    def get_average(self, name: str, window_seconds: Optional[int] = None) -> Optional[float]:
        """Get average value for metric in time window."""
        window_seconds = window_seconds or self.window_size
        cutoff_time = time.time() - window_seconds

        with self.lock:
            if name not in self.metrics:
                return None

            values = [m.value for m in self.metrics[name] if m.timestamp >= cutoff_time]

            if not values:
                return None

            return sum(values) / len(values)

    def get_max(self, name: str, window_seconds: Optional[int] = None) -> Optional[float]:
        """Get maximum value for metric in time window."""
        window_seconds = window_seconds or self.window_size
        cutoff_time = time.time() - window_seconds

        with self.lock:
            if name not in self.metrics:
                return None

            values = [m.value for m in self.metrics[name] if m.timestamp >= cutoff_time]

            if not values:
                return None

            return max(values)

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self.lock:
            return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        with self.lock:
            return self.gauges.get(name)

    def reset_counter(self, name: str) -> None:
        """Reset counter."""
        with self.lock:
            self.counters[name] = 0

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Gauges
        lines.append("# HELP scraper_gauge_metrics Gauge metrics")
        lines.append("# TYPE scraper_gauge_metrics gauge")
        for name, value in self.gauges.items():
            lines.append(f'scraper_gauge{{{name}}} {value}')

        # Counters
        lines.append("# HELP scraper_counter_metrics Counter metrics")
        lines.append("# TYPE scraper_counter_metrics counter")
        for name, value in self.counters.items():
            lines.append(f'scraper_counter{{{name}}} {value}')

        # Rolling metrics (averages)
        lines.append("# HELP scraper_metric_average Average of metric values")
        lines.append("# TYPE scraper_metric_average gauge")
        for name in self.metrics:
            avg = self.get_average(name)
            if avg is not None:
                lines.append(f'scraper_metric_average{{metric="{name}"}} {avg:.2f}')

        return "\n".join(lines)


class AlertManager:
    """Manages alerts and integrations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager.

        Args:
            config: Configuration dict with webhook URLs and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.lock = threading.Lock()

    def register_alert_handler(self, severity: AlertSeverity, callback: Callable) -> None:
        """Register callback for specific alert severity."""
        with self.lock:
            self.alert_callbacks[severity].append(callback)

    def send_alert(self, alert: Alert) -> None:
        """Send alert through configured channels."""
        with self.lock:
            self.alert_history.append(alert)

            # Call registered handlers
            for callback in self.alert_callbacks[alert.severity]:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}", exc_info=True)

        # Send to Slack
        if self.config.get('slack_webhook'):
            self._send_slack(alert)

        # Send to PagerDuty
        if self.config.get('pagerduty_key') and alert.severity == AlertSeverity.CRITICAL:
            self._send_pagerduty(alert)

    def _send_slack(self, alert: Alert) -> None:
        """Send alert to Slack."""
        try:
            webhook_url = self.config.get('slack_webhook')
            if not webhook_url:
                return

            payload = alert.to_slack_payload()
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")

    def _send_pagerduty(self, alert: Alert) -> None:
        """Send alert to PagerDuty."""
        try:
            payload = alert.to_pagerduty_payload()
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get recent alerts."""
        with self.lock:
            return list(self.alert_history)[-count:]


class MonitoringSystem:
    """Main monitoring system orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize monitoring system."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.metrics = MetricsCollector()
        self.alerts = AlertManager(self.config)

        # Error tracking
        self.error_count = 0
        self.error_threshold = self.config.get('error_threshold', 10)
        self.error_window = deque(maxlen=100)  # Last 100 errors

        # Performance SLA tracking
        self.sla_response_time_ms = self.config.get('sla_response_time_ms', 5000)
        self.sla_availability_percent = self.config.get('sla_availability_percent', 99.5)

    def record_operation(self, operation: str, duration_ms: float, success: bool = True) -> None:
        """Record operation metrics."""
        # Record duration
        self.metrics.record_metric(f"{operation}_duration_ms", duration_ms)

        # Check SLA
        if duration_ms > self.sla_response_time_ms and success:
            alert = Alert(
                title=f"SLA Violation: {operation}",
                message=f"{operation} took {duration_ms:.0f}ms (SLA: {self.sla_response_time_ms}ms)",
                severity=AlertSeverity.WARNING,
                component=operation,
                context={"duration_ms": duration_ms, "sla_ms": self.sla_response_time_ms}
            )
            self.alerts.send_alert(alert)

        # Record success/failure
        self.metrics.increment_counter(f"{operation}_total")
        if success:
            self.metrics.increment_counter(f"{operation}_success")
        else:
            self.metrics.increment_counter(f"{operation}_error")

    def record_error(self, error: Exception, component: str, context: Dict[str, Any] = None) -> None:
        """Record error and check error rate."""
        self.error_count += 1
        self.error_window.append({
            'error': str(error),
            'component': component,
            'timestamp': time.time(),
            'context': context or {}
        })

        # Increment error counter
        self.metrics.increment_counter(f"errors_{component}")
        self.metrics.increment_counter("errors_total")

        # Check error rate
        recent_errors = len(self.error_window)
        error_rate = (recent_errors / 100) * 100  # Percentage of last 100

        if error_rate > 50:  # More than 50% errors
            alert = Alert(
                title="High Error Rate",
                message=f"{error_rate:.0f}% of recent operations failed in {component}",
                severity=AlertSeverity.CRITICAL,
                component=component,
                context={"error_rate_percent": error_rate, "total_errors": recent_errors}
            )
            self.alerts.send_alert(alert)

        # Log error
        self.logger.error(f"Error in {component}: {error}", exc_info=True)

    def set_health_status(self, component: str, healthy: bool) -> None:
        """Set health status for component."""
        value = 1.0 if healthy else 0.0
        self.metrics.set_gauge(f"health_{component}", value)

        if not healthy:
            alert = Alert(
                title=f"Component Unhealthy: {component}",
                message=f"{component} is unhealthy and requires attention",
                severity=AlertSeverity.CRITICAL,
                component=component
            )
            self.alerts.send_alert(alert)

    def get_sla_status(self) -> Dict[str, Any]:
        """Get SLA compliance status."""
        total_ops = self.metrics.get_counter("operations_total")
        total_errors = self.metrics.get_counter("errors_total")

        if total_ops == 0:
            availability = 100.0
        else:
            availability = ((total_ops - total_errors) / total_ops) * 100

        return {
            "sla_availability_target_percent": self.sla_availability_percent,
            "actual_availability_percent": round(availability, 2),
            "total_operations": total_ops,
            "total_errors": total_errors,
            "status": "PASS" if availability >= self.sla_availability_percent else "FAIL"
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "operations": {
                name: self.metrics.get_counter(name)
                for name in self.metrics.counters
                if name.endswith("_total")
            },
            "errors": {
                name: self.metrics.get_counter(name)
                for name in self.metrics.counters
                if name.startswith("errors_")
            },
            "health": {
                name: self.metrics.get_gauge(name)
                for name in self.metrics.gauges
                if name.startswith("health_")
            },
            "sla_status": self.get_sla_status(),
            "recent_alerts": [
                {
                    "title": a.title,
                    "severity": a.severity.value,
                    "component": a.component,
                    "timestamp": datetime.fromtimestamp(a.timestamp).isoformat()
                }
                for a in self.alerts.get_recent_alerts(5)
            ]
        }

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.metrics.to_prometheus_format()


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """Get or create global monitoring system."""
    global _monitoring_system

    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem(config)

    return _monitoring_system


def monitoring_context(operation: str):
    """Decorator for monitoring operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitoring_system()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_operation(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_operation(operation, duration_ms, success=False)

                # Determine component from operation name
                component = operation.split('.')[0] if '.' in operation else operation
                monitor.record_error(e, component)

                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    config = {
        'error_threshold': 10,
        'sla_response_time_ms': 1000,
        'sla_availability_percent': 99.5,
    }

    monitor = get_monitoring_system(config)

    print("=== Monitoring System Example ===\n")

    # Record successful operations
    for i in range(5):
        monitor.record_operation("scraper.extract", 500 + (i * 100), success=True)

    # Record operation that violates SLA
    monitor.record_operation("scraper.extract", 6000, success=True)

    # Record some errors
    try:
        raise ValueError("Test error")
    except Exception as e:
        monitor.record_error(e, "scraper.extract", {"url": "https://example.com"})

    # Set health status
    monitor.set_health_status("database", True)
    monitor.set_health_status("cache", False)

    # Get summary
    print("=== Metrics Summary ===")
    summary = monitor.get_metrics_summary()
    print(json.dumps(summary, indent=2))

    print("\n=== Prometheus Format ===")
    print(monitor.export_prometheus())

    print("\n✅ Monitoring system operational")

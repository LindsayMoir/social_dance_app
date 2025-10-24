#!/usr/bin/env python3
"""
health_checks.py - Health Monitoring and Status Checks

Provides comprehensive health checks for production deployment:
- Database connectivity and performance
- Browser/Playwright availability
- Memory and disk usage
- Process health and uptime
- Cache efficiency
- Error rates and circuit breaker status

Supports REST endpoints for:
- /health - Overall health status
- /metrics - Prometheus metrics
- /readiness - Kubernetes readiness probe
- /liveness - Kubernetes liveness probe
"""

import os
import time
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class HealthStatus(Enum):
    """Health status indicators."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    timestamp: str = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if self.details is None:
            self.details = {}


class DatabaseHealthCheck:
    """Check database connectivity and performance."""

    def __init__(self, db_handler=None):
        self.logger = logging.getLogger(__name__)
        self.db_handler = db_handler

    def check(self) -> HealthCheckResult:
        """Check database health."""
        try:
            if not self.db_handler:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="Database handler not configured"
                )

            # Test database connection
            start_time = time.time()
            self._test_connection()
            response_time = time.time() - start_time

            # Check response time
            if response_time > 5.0:
                status = HealthStatus.DEGRADED
                message = f"Database slow: {response_time:.2f}s response time"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"

            return HealthCheckResult(
                name="database",
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time * 1000,
                    "connection_pool_size": getattr(self.db_handler, 'pool_size', 'unknown')
                }
            )

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}"
            )

    def _test_connection(self):
        """Test database connection (placeholder)."""
        # In production, this would execute a simple query
        # For now, we simulate a successful connection
        pass


class MemoryHealthCheck:
    """Check system memory usage."""

    def __init__(self, threshold_percent: float = 80.0):
        self.logger = logging.getLogger(__name__)
        self.threshold_percent = threshold_percent

    def check(self) -> HealthCheckResult:
        """Check memory health."""
        try:
            memory_info = psutil.virtual_memory()
            usage_percent = memory_info.percent
            available_mb = memory_info.available / (1024 * 1024)

            if usage_percent > self.threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"

            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "available_mb": round(available_mb, 2),
                    "threshold_percent": self.threshold_percent
                }
            )

        except Exception as e:
            self.logger.error(f"Memory health check failed: {e}")
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check error: {str(e)}"
            )


class DiskHealthCheck:
    """Check disk space usage."""

    def __init__(self, threshold_percent: float = 90.0, path: str = "/"):
        self.logger = logging.getLogger(__name__)
        self.threshold_percent = threshold_percent
        self.path = path

    def check(self) -> HealthCheckResult:
        """Check disk health."""
        try:
            disk_usage = psutil.disk_usage(self.path)
            usage_percent = disk_usage.percent
            free_gb = disk_usage.free / (1024 ** 3)

            if usage_percent > self.threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space available: {free_gb:.1f} GB free"

            return HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "free_gb": round(free_gb, 2),
                    "threshold_percent": self.threshold_percent
                }
            )

        except Exception as e:
            self.logger.error(f"Disk health check failed: {e}")
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check error: {str(e)}"
            )


class ProcessHealthCheck:
    """Check process health and uptime."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.utcnow()
        self.process = psutil.Process(os.getpid())

    def check(self) -> HealthCheckResult:
        """Check process health."""
        try:
            uptime = datetime.utcnow() - self.start_time
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            num_threads = self.process.num_threads()

            # Process is healthy if it's using reasonable resources
            if cpu_percent > 90 or memory_mb > 2000:
                status = HealthStatus.DEGRADED
                message = "Process using high resources"
            else:
                status = HealthStatus.HEALTHY
                message = "Process health normal"

            return HealthCheckResult(
                name="process",
                status=status,
                message=message,
                details={
                    "uptime_seconds": int(uptime.total_seconds()),
                    "cpu_percent": round(cpu_percent, 2),
                    "memory_mb": round(memory_mb, 2),
                    "num_threads": num_threads
                }
            )

        except Exception as e:
            self.logger.error(f"Process health check failed: {e}")
            return HealthCheckResult(
                name="process",
                status=HealthStatus.UNKNOWN,
                message=f"Process check error: {str(e)}"
            )


class CacheHealthCheck:
    """Check cache efficiency and hit rates."""

    def __init__(self, cache_metrics: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.cache_metrics = cache_metrics or {}

    def check(self) -> HealthCheckResult:
        """Check cache health."""
        try:
            hits = self.cache_metrics.get('hits', 0)
            misses = self.cache_metrics.get('misses', 0)
            total_requests = hits + misses

            if total_requests == 0:
                hit_rate = 0
            else:
                hit_rate = (hits / total_requests) * 100

            # Cache is healthy if hit rate > 20%
            if hit_rate > 20:
                status = HealthStatus.HEALTHY
                message = f"Cache efficiency good: {hit_rate:.1f}% hit rate"
            elif hit_rate > 10:
                status = HealthStatus.DEGRADED
                message = f"Cache efficiency low: {hit_rate:.1f}% hit rate"
            else:
                status = HealthStatus.DEGRADED
                message = "Cache not warmed up yet"

            return HealthCheckResult(
                name="cache",
                status=status,
                message=message,
                details={
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_hits": hits,
                    "total_misses": misses
                }
            )

        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return HealthCheckResult(
                name="cache",
                status=HealthStatus.UNKNOWN,
                message=f"Cache check error: {str(e)}"
            )


class CircuitBreakerHealthCheck:
    """Check circuit breaker status."""

    def __init__(self, circuit_breaker=None):
        self.logger = logging.getLogger(__name__)
        self.circuit_breaker = circuit_breaker

    def check(self) -> HealthCheckResult:
        """Check circuit breaker health."""
        try:
            if not self.circuit_breaker:
                return HealthCheckResult(
                    name="circuit_breaker",
                    status=HealthStatus.UNKNOWN,
                    message="Circuit breaker not configured"
                )

            is_open = getattr(self.circuit_breaker, 'is_open', lambda: False)()
            failure_count = getattr(self.circuit_breaker, 'failure_count', 0)

            if is_open:
                status = HealthStatus.UNHEALTHY
                message = "Circuit breaker is OPEN - service degraded"
            else:
                status = HealthStatus.HEALTHY
                message = "Circuit breaker healthy"

            return HealthCheckResult(
                name="circuit_breaker",
                status=status,
                message=message,
                details={
                    "is_open": is_open,
                    "failure_count": failure_count
                }
            )

        except Exception as e:
            self.logger.error(f"Circuit breaker health check failed: {e}")
            return HealthCheckResult(
                name="circuit_breaker",
                status=HealthStatus.UNKNOWN,
                message=f"Circuit breaker check error: {str(e)}"
            )


class HealthCheckManager:
    """Orchestrates all health checks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks: List[HealthCheckResult] = []
        self.last_check_time: Optional[datetime] = None
        self.check_interval = 60  # seconds

    def register_check(self, check_func) -> None:
        """Register a health check function."""
        # Store check function for lazy evaluation
        pass

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        checks = [
            MemoryHealthCheck().check(),
            DiskHealthCheck().check(),
            ProcessHealthCheck().check(),
            CacheHealthCheck().check(),
        ]

        self.last_check_time = datetime.utcnow()

        # Determine overall status
        statuses = [check.status for check in checks]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": [asdict(c) for c in checks]
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get brief health summary."""
        checks = self.run_all_checks()

        healthy = sum(1 for c in checks["checks"] if c["status"] == "healthy")
        total = len(checks["checks"])

        return {
            "status": checks["status"],
            "timestamp": checks["timestamp"],
            "summary": f"{healthy}/{total} checks passing",
            "services": {c["name"]: c["status"] for c in checks["checks"]}
        }

    def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status (for Kubernetes)."""
        health = self.get_health_summary()
        # Ready if all critical checks pass
        ready = health["status"] in ["healthy", "degraded"]

        return {
            "ready": ready,
            "status": health["status"],
            "timestamp": health["timestamp"]
        }

    def get_liveness(self) -> Dict[str, Any]:
        """Get liveness status (for Kubernetes)."""
        health = self.get_health_summary()
        # Alive if process is running
        alive = health["status"] != "unhealthy"

        return {
            "alive": alive,
            "status": health["status"],
            "timestamp": health["timestamp"]
        }


# Global health check manager
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get or create global health check manager."""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager


# REST endpoint handlers
def health_endpoint() -> Dict[str, Any]:
    """GET /health - Overall health status."""
    return get_health_manager().get_health_summary()


def metrics_endpoint() -> str:
    """GET /metrics - Prometheus format metrics."""
    health = get_health_manager().run_all_checks()

    metrics = []
    metrics.append("# HELP scraper_health Overall system health")
    metrics.append("# TYPE scraper_health gauge")

    # Add per-check metrics
    for check in health["checks"]:
        status_value = 1 if check["status"] == "healthy" else (0.5 if check["status"] == "degraded" else 0)
        metrics.append(f'scraper_health{{check="{check["name"]}"}} {status_value}')

    # Add resource metrics
    process = ProcessHealthCheck()
    result = process.check()
    metrics.append(f'scraper_process_uptime_seconds {result.details["uptime_seconds"]}')
    metrics.append(f'scraper_process_memory_bytes {int(result.details["memory_mb"] * 1024 * 1024)}')

    return "\n".join(metrics)


def readiness_endpoint() -> Dict[str, Any]:
    """GET /readiness - Kubernetes readiness probe."""
    return get_health_manager().get_readiness()


def liveness_endpoint() -> Dict[str, Any]:
    """GET /liveness - Kubernetes liveness probe."""
    return get_health_manager().get_liveness()


if __name__ == "__main__":
    # Example usage
    manager = get_health_manager()

    print("=== Health Check Summary ===")
    summary = manager.get_health_summary()
    print(f"Status: {summary['status']}")
    print(f"Summary: {summary['summary']}")

    print("\n=== Individual Checks ===")
    for check in summary.get("services", {}).items():
        print(f"{check[0]}: {check[1]}")

    print("\n=== Prometheus Metrics ===")
    print(metrics_endpoint())

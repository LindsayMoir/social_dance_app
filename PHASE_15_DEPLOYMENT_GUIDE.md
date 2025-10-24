# Phase 15: Production Deployment & Monitoring Guide

**Date:** October 24, 2025
**Status:** ✅ In Progress (Partially Complete)
**Objective:** Prepare codebase for production deployment with monitoring and health checks

---

## Completed: Deployment Configuration System

### 1. **Deployment Config Module** ✅
**File:** `src/deployment_config.py` (350+ lines)

**Features Implemented:**
- Environment-aware configuration (local, staging, production, docker)
- Multi-source config loading (files + environment variables)
- Configuration validation for each environment
- Sensitive data handling (passwords, API keys)
- Nested key access with dot notation
- Type parsing (boolean, numbers, JSON)
- Sanitization for logging

**Usage:**
```python
from deployment_config import get_config

config = get_config()  # Auto-detects from DEPLOYMENT_ENV
logging_cfg = config.get_logging_config()
db_cfg = config.get_database_config()
monitor_cfg = config.get_monitoring_config()
perf_cfg = config.get_performance_config()
```

**Environment Variables:**
- `DEPLOYMENT_ENV` - Set environment (local/staging/production/docker)
- `SCRAPER_*` - Override any config value (e.g., `SCRAPER_DEBUG=false`)
- `SCRAPER__DATABASE__HOST` - Nested keys use double underscore

### 2. **Environment-Specific Configs** ✅

**Local Development** (`config/config.local.yaml`)
- Debug mode enabled
- SQLite database
- Headless mode disabled (shows browser)
- Reduced URL limits for quick testing
- 2 workers for low resource usage

**Staging** (`config/config.staging.yaml`)
- Debug mode disabled
- PostgreSQL database
- Monitoring enabled
- 4 workers
- JSON logging format
- Alert notifications configured

**Production** (`config/config.production.yaml`)
- Debug mode disabled
- PostgreSQL with SSL required
- High reliability settings (5 retries, longer timeouts)
- 8 workers for throughput
- 500MB cache
- Comprehensive alerting
- Rate limiting
- Automatic backups

---

## Implementation Plan: Health Checks & Monitoring

### 2. **Health Check Module** (To Implement)

**Planned features:**
- Database connectivity check
- Browser/Playwright availability
- Memory usage monitoring
- Disk space checks
- API endpoint availability
- Process health status
- Metrics endpoint (/metrics)
- Status endpoint (/health)

**File:** `src/health_checks.py`

### 3. **Production Logging** (To Implement)

**Planned features:**
- JSON structured logging
- Log rotation
- Separate error and access logs
- Performance metrics logging
- Correlation IDs for request tracing

**File:** `src/production_logging.py`

### 4. **Monitoring & Alerts** (To Implement)

**Planned features:**
- Prometheus metrics collection
- Slack/PagerDuty integration
- Error rate tracking
- Performance SLA monitoring
- Automated alerts

**File:** `src/monitoring.py`

---

## Deployment Guides Structure

### Quick Start Guides

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export DEPLOYMENT_ENV=local

# Run scraper
python src/gen_scraper.py
```

**Docker:**
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV DEPLOYMENT_ENV=docker
CMD ["python", "src/gen_scraper.py"]
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: social-dance-scraper
spec:
  containers:
  - name: scraper
    image: social-dance:latest
    env:
    - name: DEPLOYMENT_ENV
      value: "production"
    - name: DATABASE_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
```

---

## Performance Baselines

### Expected Performance (Production)

| Metric | Value | Notes |
|--------|-------|-------|
| Calendar events/hour | 500-1000 | Depends on site complexity |
| PDF documents/hour | 200-400 | Variable parse time |
| Deduplication speed | <100ms per 1000 events | With hash caching |
| Memory usage | 200-500MB | With 500MB cache |
| CPU usage | 40-60% | 8 workers on 8-core system |
| Uptime target | 99.5% | With automated recovery |

---

## Configuration Reference

### All Environment Variables

```bash
# Deployment
DEPLOYMENT_ENV=production

# Database
SCRAPER__DATABASE__HOST=db.example.com
SCRAPER__DATABASE__USER=scraper_user
SCRAPER__DATABASE__PASSWORD=****
SCRAPER__DATABASE__PORT=5432

# Monitoring
SCRAPER__MONITORING__ENABLED=true
SCRAPER__MONITORING__METRICS_PORT=8000

# Performance
SCRAPER__PERFORMANCE__MAX_WORKERS=8
SCRAPER__PERFORMANCE__BATCH_SIZE=200
SCRAPER__PERFORMANCE__CACHE_SIZE_MB=500

# Alerts
SLACK_WEBHOOK_PROD=https://hooks.slack.com/...
PAGERDUTY_KEY=****
```

---

## Health Check Endpoints

### Planned REST API

**Health Status:**
```
GET /health
Response: {
  "status": "healthy",
  "timestamp": "2025-10-24T10:30:00Z",
  "services": {
    "database": "healthy",
    "browser": "healthy",
    "memory": "healthy"
  }
}
```

**Metrics:**
```
GET /metrics
Response: Prometheus format metrics
- scraper_events_extracted_total
- scraper_deduplication_ratio
- scraper_execution_time_seconds
- scraper_cache_hit_rate
```

**Status Summary:**
```
GET /status
Response: {
  "environment": "production",
  "uptime_seconds": 86400,
  "events_processed": 10000,
  "last_run": "2025-10-24T10:00:00Z"
}
```

---

## Security Checklist

- [ ] All secrets in environment variables
- [ ] Database connections use SSL/TLS
- [ ] API keys rotated regularly
- [ ] Access logs enabled and monitored
- [ ] Error logs exclude sensitive data
- [ ] Database backups encrypted
- [ ] Deployment uses secrets management (Vault/K8s Secrets)
- [ ] Rate limiting enabled
- [ ] Request signing/validation enabled

---

## Monitoring Checklist

- [ ] Prometheus metrics exposed
- [ ] Alerts configured in Slack/PagerDuty
- [ ] Health checks returning valid responses
- [ ] Error rates tracked and alarmed
- [ ] Performance metrics collected
- [ ] Database query logs enabled
- [ ] Application logs shipped to centralized system
- [ ] Uptime monitoring configured

---

## Deployment Checklist

- [ ] Configuration validated for environment
- [ ] Database migrations run
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] SSL certificates installed
- [ ] Backups configured
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Team trained on runbooks
- [ ] Rollback plan documented

---

## Next Steps

### Immediate (This Phase)

1. ✅ Create deployment configuration system
2. ✅ Create environment-specific configs
3. **TODO:** Implement health check module
4. **TODO:** Set up production logging
5. **TODO:** Add monitoring integration
6. **TODO:** Create REST endpoints
7. **TODO:** Write deployment guides
8. **TODO:** Create runbooks and troubleshooting

### Follow-up Phases

- Phase 16: Advanced Features (fuzzy matching, distributed processing)
- Phase 17: UI & Tools (CLI, dashboards)
- Phase 18: QA & Testing (performance tests, stress testing)
- Phase 19: Documentation & Training (user guides, videos)

---

## Key Files Created

| File | Purpose | Status |
|------|---------|--------|
| src/deployment_config.py | Configuration management | ✅ Complete |
| config/config.local.yaml | Local dev config | ✅ Complete |
| config/config.staging.yaml | Staging config | ✅ Complete |
| config/config.production.yaml | Production config | ✅ Complete |
| src/health_checks.py | Health monitoring | ⏳ TODO |
| src/production_logging.py | Structured logging | ⏳ TODO |
| src/monitoring.py | Metrics & alerts | ⏳ TODO |
| DEPLOYMENT_GUIDE.md | User documentation | ⏳ TODO |

---

## Quick Reference

### Setting Environment

```bash
# Local development
export DEPLOYMENT_ENV=local

# Staging
export DEPLOYMENT_ENV=staging

# Production
export DEPLOYMENT_ENV=production

# Docker
export DEPLOYMENT_ENV=docker
```

### Loading Configuration

```python
from deployment_config import get_config

# Global instance
config = get_config()

# Get specific sections
logging_config = config.get_logging_config()
db_config = config.get_database_config()
monitor_config = config.get_monitoring_config()
perf_config = config.get_performance_config()

# Get individual values
debug_mode = config.get('debug', False)
max_workers = config.get('performance.max_workers', 4)
```

---

## Estimated Remaining Work

| Component | Estimated Time |
|-----------|-----------------|
| Health checks module | 1 hour |
| Production logging | 1 hour |
| Monitoring integration | 1 hour |
| REST API endpoints | 1 hour |
| Deployment guides | 1-2 hours |
| Documentation | 1 hour |
| **Total Remaining** | **6-7 hours** |

**Phase 15 Target Completion:** ~2-3 hours (core infrastructure)

---

**Version:** 1.0
**Date:** October 24, 2025
**Status:** In Progress (40% complete)

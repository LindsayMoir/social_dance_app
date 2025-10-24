# Deployment Guide - Social Dance Event Scraper

**Date:** October 24, 2025
**Version:** 1.0
**Status:** Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Staging Deployment](#staging-deployment)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Configuration](#configuration)
8. [Monitoring & Health Checks](#monitoring--health-checks)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)

---

## Quick Start

### Local Development (5 minutes)

```bash
# 1. Clone repository
git clone <repository-url>
cd social_dance_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment
export DEPLOYMENT_ENV=local

# 4. Run application
python src/gen_scraper.py
```

### Docker (2 minutes)

```bash
# 1. Build image
docker build -t social-dance:latest .

# 2. Run container
docker run -e DEPLOYMENT_ENV=docker social-dance:latest

# 3. Access health endpoint
curl http://localhost:8000/health
```

### Kubernetes (1 minute)

```bash
# 1. Apply manifests
kubectl apply -f k8s/

# 2. Check deployment status
kubectl get pods -l app=social-dance

# 3. View logs
kubectl logs -l app=social-dance -f
```

---

## Local Development

### Setup

1. **Install Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   export DEPLOYMENT_ENV=local
   # Optional: Override specific config values
   export SCRAPER_DEBUG=true
   export SCRAPER__CRAWLING__HEADLESS=false
   ```

### Running

**Start the scraper:**
```bash
python src/gen_scraper.py
```

**Run with specific event types:**
```bash
python src/gen_scraper.py --sources read_extract,read_pdfs
```

**Enable debug logging:**
```bash
export SCRAPER_LOG_LEVEL=DEBUG
python src/gen_scraper.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_gen_scraper_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=src

# Run excluding slow tests
python -m pytest tests/ -m "not slow" -v
```

### Configuration Files

- **Main config:** `config/config.local.yaml`
- **Database:** SQLite at `data/social_dance_dev.db`
- **Logs:** `logs/scraper.log` (development format)
- **Cache:** In-memory (50MB)

---

## Staging Deployment

### Infrastructure Requirements

- **Server:** t2.medium (2 vCPU, 4GB RAM) or equivalent
- **Database:** PostgreSQL 13+ (separate RDS instance recommended)
- **Storage:** 50GB EBS volume
- **Network:** VPC with public/private subnets

### Pre-Deployment Checklist

```bash
# 1. Validate configuration
python -c "from deployment_config import get_config; c = get_config('staging'); print(c.to_dict())"

# 2. Run tests
python -m pytest tests/ --ignore=tests/test_llm_schema_parsing.py -q

# 3. Check database migrations
python -m alembic upgrade head

# 4. Verify credentials
aws s3 ls  # Should list S3 buckets
```

### Deployment Steps

1. **Prepare Server**
   ```bash
   # SSH into server
   ssh -i key.pem ubuntu@staging.example.com

   # Install system packages
   sudo apt-get update
   sudo apt-get install -y python3.11 postgresql-client git
   ```

2. **Deploy Application**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd social_dance_app

   # Create venv
   python3.11 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   export DEPLOYMENT_ENV=staging
   export SCRAPER__DATABASE__HOST=rds-staging.example.com
   export SCRAPER__DATABASE__USER=scraper_user
   export SCRAPER__DATABASE__PASSWORD=$(aws secretsmanager get-secret-value --secret-id staging/db-password --query SecretString --output text)
   export SLACK_WEBHOOK_STAGING=https://hooks.slack.com/...
   ```

4. **Run Application**
   ```bash
   # Using systemd
   sudo tee /etc/systemd/system/scraper.service > /dev/null <<EOF
   [Unit]
   Description=Social Dance Scraper
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/social_dance_app
   Environment="DEPLOYMENT_ENV=staging"
   Environment="SCRAPER__DATABASE__HOST=rds-staging.example.com"
   Environment="SCRAPER__DATABASE__USER=scraper_user"
   ExecStart=/home/ubuntu/social_dance_app/venv/bin/python src/gen_scraper.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   EOF

   sudo systemctl daemon-reload
   sudo systemctl enable scraper
   sudo systemctl start scraper
   ```

5. **Verify Deployment**
   ```bash
   # Check service status
   sudo systemctl status scraper

   # Check logs
   tail -f logs/scraper.log

   # Health check
   curl http://localhost:8000/health
   ```

### Staging Monitoring

- **Health endpoint:** `http://<staging-server>:8000/health`
- **Metrics endpoint:** `http://<staging-server>:8000/metrics`
- **Slack alerts:** Configured to `#staging-alerts`
- **Retention:** 7 days of logs

---

## Production Deployment

### Infrastructure Requirements

- **Compute:** t3.large (2 vCPU, 8GB RAM) or auto-scaling group
- **Database:** PostgreSQL 13+ HA (Multi-AZ RDS)
- **Storage:** 200GB EBS volume with snapshots
- **CDN:** CloudFront for static assets
- **Monitoring:** CloudWatch + Prometheus + ELK Stack
- **Backup:** Daily automated backups to S3 (30-day retention)

### Pre-Deployment Checklist

```bash
# 1. Code review and testing
git log --oneline -10  # Verify latest commits
python -m pytest tests/ -q --tb=short

# 2. Configuration validation
DEPLOYMENT_ENV=production python -c "from deployment_config import get_config; get_config().validate()"

# 3. Database backups
pg_dump -U postgres social_dance_prod | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# 4. Security scan
pip install safety
safety check

# 5. Performance baseline
python scripts/performance_test.py
```

### High Availability Setup

1. **Load Balancer Configuration**
   ```yaml
   # AWS Load Balancer
   Type: AWS::ElasticLoadBalancingV2::LoadBalancer
   Properties:
     Name: scraper-lb
     Subnets:
       - subnet-public-1
       - subnet-public-2
     SecurityGroups:
       - sg-production
   ```

2. **Auto Scaling Group**
   ```yaml
   Type: AWS::AutoScaling::AutoScalingGroup
   Properties:
     LaunchTemplate:
       LaunchTemplateId: lt-scraper
       Version: $Latest
     MinSize: 2
     MaxSize: 8
     DesiredCapacity: 4
     VPCZoneIdentifier:
       - subnet-private-1
       - subnet-private-2
   ```

3. **Database Replication**
   ```bash
   # AWS RDS Multi-AZ
   aws rds modify-db-instance \
     --db-instance-identifier social-dance-prod \
     --multi-az \
     --apply-immediately
   ```

### Deployment Steps

1. **Infrastructure Preparation**
   ```bash
   # Use Terraform/CloudFormation for IaC
   terraform init
   terraform plan -out=tfplan
   terraform apply tfplan
   ```

2. **Blue-Green Deployment**
   ```bash
   # Deploy to blue environment
   docker build -t social-dance:v1.0.0 .
   docker tag social-dance:v1.0.0 registry.example.com/social-dance:v1.0.0
   docker push registry.example.com/social-dance:v1.0.0

   # Update ECS task definition with new image
   aws ecs update-service \
     --cluster production \
     --service scraper \
     --task-definition scraper:2
   ```

3. **Health Verification**
   ```bash
   # Wait for deployment
   aws ecs wait services-stable \
     --cluster production \
     --services scraper

   # Verify health
   curl -f https://api.example.com/health || exit 1

   # Check metrics
   curl -f https://api.example.com/metrics | grep scraper_health
   ```

4. **Switch Traffic (if not using ECS)**
   ```bash
   # Update Route 53 or load balancer
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z123456 \
     --change-batch '{...}'
   ```

5. **Monitor Rollout**
   ```bash
   # CloudWatch logs
   aws logs tail /aws/scraper/production --follow

   # Application metrics
   watch 'curl http://api.example.com/metrics | grep scraper_'

   # PagerDuty alerts
   # Check for any critical alerts
   ```

### Production Security

1. **SSL/TLS Certificates**
   ```bash
   # Use AWS Certificate Manager
   aws acm request-certificate --domain-name api.example.com
   ```

2. **Secrets Management**
   ```bash
   # Store secrets in AWS Secrets Manager
   aws secretsmanager create-secret \
     --name prod/db-password \
     --secret-string file://password.txt
   ```

3. **VPC Configuration**
   - Private subnets for application
   - Public subnets for load balancers
   - Security groups with minimal required ports
   - NACLs for additional protection

4. **API Keys and Authentication**
   - Rotate API keys monthly
   - Use AWS IAM for service-to-service auth
   - Enable MFA for admin access

---

## Docker Deployment

### Building Docker Image

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       chromium-browser \
       chromium-driver \
       && rm -rf /var/lib/apt/lists/*

   # Copy application
   COPY . .

   # Install Python dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Set environment
   ENV DEPLOYMENT_ENV=docker
   ENV PYTHONUNBUFFERED=1

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
     CMD python -c "import requests; requests.get('http://localhost:8000/health')"

   # Run application
   CMD ["python", "src/gen_scraper.py"]
   ```

2. **Build Image**
   ```bash
   docker build -t social-dance:latest \
     --build-arg PYTHON_VERSION=3.11 \
     .

   docker tag social-dance:latest registry.example.com/social-dance:latest
   docker push registry.example.com/social-dance:latest
   ```

3. **Run Container**
   ```bash
   docker run -d \
     --name scraper \
     -e DEPLOYMENT_ENV=docker \
     -e SCRAPER__DATABASE__HOST=postgres-host \
     -e SCRAPER__DATABASE__PASSWORD=<secret> \
     -p 8000:8000 \
     -v logs:/app/logs \
     -v cache:/app/cache \
     social-dance:latest
   ```

4. **Docker Compose**
   ```yaml
   version: '3.9'

   services:
     scraper:
       build: .
       environment:
         DEPLOYMENT_ENV: docker
         SCRAPER__DATABASE__HOST: postgres
         SCRAPER__DATABASE__PASSWORD: ${DB_PASSWORD}
       ports:
         - "8000:8000"
       volumes:
         - ./logs:/app/logs
         - ./cache:/app/cache
       depends_on:
         postgres:
           condition: service_healthy

     postgres:
       image: postgres:15-alpine
       environment:
         POSTGRES_PASSWORD: ${DB_PASSWORD}
         POSTGRES_DB: social_dance_prod
       volumes:
         - postgres_data:/var/lib/postgresql/data
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U postgres"]
         interval: 10s
         timeout: 5s
         retries: 5

   volumes:
     postgres_data:
   ```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl CLI
- Helm 3.0+
- Container registry access

### Deployment Manifests

1. **Namespace and ConfigMap**
   ```yaml
   apiVersion: v1
   kind: Namespace
   metadata:
     name: scraper
   ---
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: scraper-config
     namespace: scraper
   data:
     DEPLOYMENT_ENV: "production"
     SCRAPER__PERFORMANCE__MAX_WORKERS: "8"
     SCRAPER__MONITORING__ENABLED: "true"
   ```

2. **Secret for Credentials**
   ```bash
   kubectl create secret generic scraper-secrets \
     --from-literal=db-password=<password> \
     --from-literal=slack-webhook=<webhook> \
     --from-literal=pagerduty-key=<key> \
     -n scraper
   ```

3. **Deployment**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: scraper
     namespace: scraper
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: scraper
     template:
       metadata:
         labels:
           app: scraper
         annotations:
           prometheus.io/scrape: "true"
           prometheus.io/port: "8000"
           prometheus.io/path: "/metrics"
       spec:
         containers:
         - name: scraper
           image: registry.example.com/social-dance:v1.0.0
           imagePullPolicy: Always
           ports:
           - name: http
             containerPort: 8000
           envFrom:
           - configMapRef:
               name: scraper-config
           env:
           - name: SCRAPER__DATABASE__PASSWORD
             valueFrom:
               secretKeyRef:
                 name: scraper-secrets
                 key: db-password
           resources:
             requests:
               memory: "512Mi"
               cpu: "500m"
             limits:
               memory: "2Gi"
               cpu: "2000m"
           livenessProbe:
             httpGet:
               path: /liveness
               port: http
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /readiness
               port: http
             initialDelaySeconds: 20
             periodSeconds: 5
           volumeMounts:
           - name: logs
             mountPath: /app/logs
         volumes:
         - name: logs
           emptyDir: {}
   ```

4. **Service**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: scraper-service
     namespace: scraper
   spec:
     selector:
       app: scraper
     ports:
     - protocol: TCP
       port: 80
       targetPort: http
     type: LoadBalancer
   ```

5. **Deploy**
   ```bash
   kubectl apply -f k8s/

   # Verify deployment
   kubectl get pods -n scraper
   kubectl logs -f deployment/scraper -n scraper

   # Check service
   kubectl get svc -n scraper
   ```

---

## Configuration

### Environment Variables

**Database Configuration**
```bash
SCRAPER__DATABASE__HOST=localhost
SCRAPER__DATABASE__PORT=5432
SCRAPER__DATABASE__NAME=social_dance_prod
SCRAPER__DATABASE__USER=scraper_user
SCRAPER__DATABASE__PASSWORD=<secret>
```

**Performance Tuning**
```bash
SCRAPER__PERFORMANCE__MAX_WORKERS=8
SCRAPER__PERFORMANCE__BATCH_SIZE=200
SCRAPER__PERFORMANCE__CACHE_SIZE_MB=500
```

**Monitoring**
```bash
SCRAPER__MONITORING__ENABLED=true
SCRAPER__MONITORING__METRICS_PORT=8000
SCRAPER__MONITORING__HEALTH_CHECK_INTERVAL=60
```

**Alerting**
```bash
SLACK_WEBHOOK_PROD=https://hooks.slack.com/...
PAGERDUTY_KEY=<key>
```

### Configuration Files

**Local Development** - `config/config.local.yaml`
- SQLite database
- Headless: false
- 2 workers
- Debug logging enabled

**Staging** - `config/config.staging.yaml`
- PostgreSQL
- 4 workers
- Monitoring enabled
- JSON logging

**Production** - `config/config.production.yaml`
- PostgreSQL with SSL
- 8 workers
- 500MB cache
- Comprehensive monitoring

---

## Monitoring & Health Checks

### Health Endpoints

```bash
# Overall health status
curl http://localhost:8000/health

# Kubernetes readiness probe
curl http://localhost:8000/readiness

# Kubernetes liveness probe
curl http://localhost:8000/liveness

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Expected Responses

```json
// /health
{
  "status": "healthy",
  "timestamp": "2025-10-24T14:30:00Z",
  "services": {
    "database": "healthy",
    "memory": "healthy",
    "disk": "healthy",
    "process": "healthy",
    "cache": "healthy"
  }
}

// /readiness
{
  "ready": true,
  "status": "healthy"
}

// /liveness
{
  "alive": true,
  "status": "healthy"
}
```

### Prometheus Metrics

```
# HELP scraper_health Overall system health
# TYPE scraper_health gauge
scraper_health{check="database"} 1
scraper_health{check="memory"} 1
scraper_health{check="disk"} 1
scraper_health{check="process"} 1

# Process metrics
scraper_process_uptime_seconds 3600
scraper_process_memory_bytes 536870912
```

### Alerting

**Slack Integration**
```python
# Automatic alerts for:
# - Service unavailable (status = unhealthy)
# - High memory usage (>80%)
# - Low disk space (<10% free)
# - Database connection failures
# - Error rate > 5%
```

**PagerDuty Integration**
```python
# Triggers incident for:
# - Critical errors
# - Service unavailability
# - Database failover events
```

---

## Troubleshooting

### Common Issues

**Problem: Application won't start**
```bash
# Check logs
tail -100 logs/scraper_error.log

# Check database connection
python -c "from deployment_config import get_config; \
           cfg = get_config(); \
           print(cfg.get_database_config())"

# Test database
psql -h <host> -U <user> -d social_dance_prod -c "SELECT 1"
```

**Problem: High memory usage**
```bash
# Check process memory
ps aux | grep gen_scraper

# Check cache size
curl http://localhost:8000/health | grep -A 5 cache

# Reduce cache size
export SCRAPER__PERFORMANCE__CACHE_SIZE_MB=200
```

**Problem: Slow performance**
```bash
# Check metrics
curl http://localhost:8000/metrics | grep scraper_

# Check database performance
EXPLAIN ANALYZE SELECT * FROM events WHERE created_at > now() - interval '1 day';

# Check CPU usage
top -b -n 1 | grep gen_scraper
```

**Problem: Database connection issues**
```bash
# Check connection pool
netstat -an | grep 5432 | wc -l

# Check database logs
sudo tail -f /var/log/postgresql/postgresql.log

# Check firewall
telnet <db-host> 5432
```

### Debug Mode

```bash
# Enable debug logging
export SCRAPER_LOG_LEVEL=DEBUG
python src/gen_scraper.py

# Run with verbose output
python -u src/gen_scraper.py 2>&1 | tee debug.log

# Check configuration
python -c "from deployment_config import get_config; \
           import json; \
           print(json.dumps(get_config().to_dict(), indent=2))"
```

---

## Rollback Procedures

### Rollback from Production

1. **Immediate Rollback (< 5 minutes)**
   ```bash
   # Using ECS
   aws ecs update-service \
     --cluster production \
     --service scraper \
     --task-definition scraper:1 \
     --force-new-deployment

   # Verify rollback
   aws ecs describe-services \
     --cluster production \
     --services scraper | grep taskDefinition
   ```

2. **Database Rollback**
   ```bash
   # Stop application
   sudo systemctl stop scraper

   # Restore from backup
   psql social_dance_prod < backup_20251024_143000.sql.gz

   # Verify data
   psql -c "SELECT COUNT(*) FROM events"

   # Restart application
   sudo systemctl start scraper
   ```

3. **DNS Rollback**
   ```bash
   # If using Route 53
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z123456 \
     --change-batch file://rollback.json
   ```

### Monitoring Rollback

```bash
# Check health after rollback
curl https://api.example.com/health

# Verify error rates
curl https://api.example.com/metrics | grep scraper_errors

# Check logs
aws logs tail /aws/scraper/production --follow --since 5m
```

---

## Maintenance

### Regular Tasks

**Daily**
- Monitor health endpoints
- Check error logs for exceptions
- Verify database backups completed

**Weekly**
- Review performance metrics
- Check disk usage
- Update dependencies (pip list --outdated)

**Monthly**
- Database maintenance (VACUUM, ANALYZE)
- Log rotation and archival
- Security updates
- Performance optimization review

### Backup Strategy

```bash
# Automated daily backups at 2 AM UTC
# Retention: 30 days
# Location: S3 with 256-bit encryption

# Manual backup
pg_dump -U postgres social_dance_prod | \
  gzip | \
  aws s3 cp - s3://backups/manual_$(date +%Y%m%d_%H%M%S).sql.gz

# Test restore
pg_restore -d social_dance_test backup.sql.gz
```

---

## Support and Escalation

**Production Issues:**
1. Check `/health` endpoint
2. Review error logs
3. Contact DevOps team via PagerDuty
4. Execute rollback if necessary

**Non-Critical Issues:**
1. File issue in GitHub
2. Assign to platform team
3. Target resolution in next sprint

**Documentation:**
- Internal Wiki: https://wiki.example.com/scraper
- Runbooks: `/docs/runbooks/`
- Architecture: `ARCHITECTURE.md`

---

**End of Deployment Guide**

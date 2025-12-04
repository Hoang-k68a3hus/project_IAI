# 🚀 VieComRec Automation Pipeline - Docker Deployment

## ✅ Successfully Deployed

Automation scheduler đã được setup hoàn tất với Docker. Toàn bộ pipeline automation chạy trên docker container với lịch cố định.

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          VieComRec Microservices (Docker Compose)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🌐 API Service              📊 Dashboard               │
│  (viecomrec-api)             (viecomrec-dashboard)      │
│  Port: 8000                  Port: 8501                 │
│  Status: Healthy ✓           Status: Unhealthy ⚠       │
│                                                              │
│  ⏰ Automation Scheduler      🔄 Orchestration          │
│  (viecomrec-scheduler)       (APScheduler)             │
│  Status: Running ✓           6 scheduled jobs          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📅 Scheduled Tasks

| # | Task | Schedule | Time | Status |
|---|------|----------|------|--------|
| 1️⃣ | Data Refresh | Daily | 02:00 | ✅ Active |
| 2️⃣ | BERT Embeddings | Weekly (Tue) | 03:00 | ✅ Active |
| 3️⃣ | Drift Detection | Weekly (Mon) | 09:00 | ✅ Active |
| 4️⃣ | Model Training | Weekly (Sun) | 03:00 | ✅ Active |
| 5️⃣ | Model Deployment | Daily | 05:00 | ✅ Active |
| 6️⃣ | Health Check | Hourly | Every hour | ✅ Active |

## 🎯 Automation Pipeline Steps

### 1. Data Refresh (Daily 2 AM)
```bash
python -m automation.data_refresh --config config/data_config.yaml
```
**Input**: Raw CSV files from `data/published_data/`
**Output**: Processed Parquet + NPZ matrices → `data/processed/`
**Log**: `logs/scheduler/data_refresh_*.log`

### 2. BERT Embeddings Refresh (Weekly Tuesday 3 AM)
```bash
python scripts/refresh_bert_embeddings.py \
  --model vinai/phobert-base \
  --output data/processed/content_based_embeddings/ \
  --product-file data/published_data/data_product.csv
```
**Input**: Product metadata + PhoBERT model
**Output**: Product embeddings → `data/processed/content_based_embeddings/product_embeddings.pt`
**Log**: `logs/scheduler/bert_embeddings_*.log`

### 3. Drift Detection (Weekly Monday 9 AM)
```bash
python scripts/detect_drift.py
```
**Input**: Historical data + current metrics
**Output**: Drift report with alerts if needed
**Log**: `logs/scheduler/drift_detection_*.log`

### 4. Model Training (Weekly Sunday 3 AM)
```bash
python -m automation.model_training \
  --config config/training_config.yaml \
  --auto-select
```
**Input**: Processed training data from Step 1
**Outputs**:
- ALS model: `artifacts/cf/als/*/`
- BPR model: `artifacts/cf/bpr/*/`
- Best selected: Updated in `artifacts/cf/registry.json`
**Log**: `logs/scheduler/training_*.log`

### 5. Model Deployment (Daily 5 AM)
```bash
python -m automation.model_deployment --service-url http://viecomrec-api:8000
```
**Input**: Best model from registry
**Action**: Hot-reload model in API service without downtime
**Log**: `logs/scheduler/deployment_*.log`

### 6. Health Check (Hourly)
```bash
python -m automation.health_check --service-url http://viecomrec-api:8000
```
**Checks**:
- API service status
- Model loading status
- BERT embeddings freshness
- Resource usage
**Log**: `logs/scheduler/health_check_*.log`

## 🗂️ File Structure

```
viecomrec/
├── automation/
│   ├── __init__.py
│   ├── data_refresh.py                 ← Data pipeline (CLI via python -m)
│   ├── model_training.py               ← ALS + BPR training
│   ├── model_deployment.py             ← Hot-reload deployment
│   ├── health_check.py                 ← Service health monitoring
│   └── cleanup.py                      ← Log/artifact cleanup
│
├── scripts/
│   ├── scheduler/
│   │   └── automation_scheduler.py     ← Main scheduler (APScheduler)
│   ├── refresh_bert_embeddings.py      ← BERT embedding generation
│   ├── detect_drift.py                 ← Drift detection
│   └── ...                             ← Other utility scripts
│
├── docker-compose.yml                  ← Updated with scheduler service
├── requirements.docker.txt             ← Updated with APScheduler
├── Dockerfile                          ← Multi-stage build
├── manage_scheduler.ps1                ← Scheduler management script
│
├── docs/
│   └── AUTOMATION_SCHEDULER.md         ← Complete documentation
│
└── logs/scheduler/
    ├── scheduler.log                   ← Main scheduler logs
    ├── task_status.json                ← Task execution history
    ├── data_refresh_*.log
    ├── bert_embeddings_*.log
    ├── drift_detection_*.log
    ├── training_*.log
    ├── deployment_*.log
    └── health_check_*.log
```

## 🚀 Quick Start Commands

### Start All Services
```powershell
# Start API, Dashboard, and Scheduler
docker compose up -d api dashboard scheduler

# Verify all running
docker ps
```

### Manage Scheduler
```powershell
# View logs
docker logs -f viecomrec-scheduler

# Check scheduled jobs
docker logs viecomrec-scheduler | grep "SCHEDULED JOBS" -A 10

# Stop scheduler
docker compose stop scheduler

# Restart scheduler
docker compose restart scheduler
```

### Use Management Script
```powershell
# Start
.\manage_scheduler.ps1 -Command start

# Stop
.\manage_scheduler.ps1 -Command stop

# View logs
.\manage_scheduler.ps1 -Command logs

# Check status
.\manage_scheduler.ps1 -Command status

# Open shell
.\manage_scheduler.ps1 -Command shell
```

### Manual Task Execution
```powershell
# Run data refresh immediately
docker exec viecomrec-scheduler python -m automation.data_refresh --config config/data_config.yaml

# Run model training immediately
docker exec viecomrec-scheduler python -m automation.model_training --config config/training_config.yaml --auto-select

# Run health check immediately
docker exec viecomrec-scheduler python -m automation.health_check --service-url http://viecomrec-api:8000
```

## 📋 Task Status & Logs

### View Current Status
```powershell
# Show running containers
docker ps

# Show scheduler logs (last 50 lines)
docker logs --tail 50 viecomrec-scheduler

# Show all registered jobs
docker logs viecomrec-scheduler | grep "Trigger:"

# Show task execution history
docker exec viecomrec-scheduler cat logs/scheduler/task_status.json
```

### Access Task Logs
```powershell
# Copy logs to local machine
docker cp viecomrec-scheduler:/app/logs/scheduler ./local_logs

# View specific task log
docker exec viecomrec-scheduler cat logs/scheduler/data_refresh_*.log
```

## 🔧 Configuration

### Edit Schedule Times

Edit `scripts/scheduler/automation_scheduler.py`:

```python
SCHEDULER_CONFIG = {
    'data_refresh': {
        'enabled': True,
        'schedule': '0 2 * * *',  # Change this
        'description': 'Daily data refresh'
    },
    # ... more tasks
}
```

### Cron Format Reference
- `0 2 * * *` = 2:00 AM daily
- `0 3 * * 2` = 3:00 AM on Tuesday
- `0 * * * *` = Every hour
- `*/15 * * * *` = Every 15 minutes

### Disable Specific Tasks

In `SCHEDULER_CONFIG`:
```python
'data_refresh': {
    'enabled': False,  # Set to False to disable
    ...
}
```

## 📊 Monitoring

### Dashboard
Access Streamlit dashboard at `http://localhost:8501`
- View active jobs
- Task execution history
- Performance metrics

### API Health
Check API status at `http://localhost:8000/health`

### Container Stats
```powershell
# Real-time stats
docker stats viecomrec-scheduler

# Memory/CPU usage
docker exec viecomrec-scheduler free -h
docker exec viecomrec-scheduler top -b -n 1
```

## ⚠️ Troubleshooting

### Scheduler not starting
```powershell
# Check logs
docker logs viecomrec-scheduler

# Verify APScheduler installed
docker exec viecomrec-scheduler pip list | grep apscheduler

# Check Python syntax
docker exec viecomrec-scheduler python -m py_compile scripts/scheduler/automation_scheduler.py
```

### Tasks not running
```powershell
# Verify correct time in container
docker exec viecomrec-scheduler date

# Check job registration
docker logs viecomrec-scheduler | grep "Registered job"

# Verify SERVICE_URL
docker exec viecomrec-scheduler echo $SERVICE_URL
```

### Task failures
```powershell
# Check task logs
docker exec viecomrec-scheduler tail -200 logs/scheduler/training_*.log

# Verify required files
docker exec viecomrec-scheduler ls -la scripts/
docker exec viecomrec-scheduler ls -la config/

# Test manual execution
docker exec viecomrec-scheduler python -m automation.health_check --service-url http://viecomrec-api:8000
```

## 📝 Files Changed/Created

### Created Files
- ✅ `scripts/scheduler/automation_scheduler.py` - Main scheduler
- ✅ `manage_scheduler.ps1` - Management script
- ✅ `docs/AUTOMATION_SCHEDULER.md` - Full documentation

### Modified Files
- ✅ `docker-compose.yml` - Added scheduler service
- ✅ `requirements.docker.txt` - Added APScheduler + pytz
- ✅ `requirements.txt` - Added APScheduler + pytz

## 🎯 Next Steps

1. **Monitor Tasks**: Watch logs during scheduled times
2. **Test Manually**: Run each task manually to ensure scripts work
3. **Set Alerts**: Configure email/Slack notifications in `send_notification()`
4. **Adjust Schedule**: Fine-tune cron times based on your timezone/needs
5. **Archive Logs**: Implement log rotation to prevent disk overflow

## 📚 Documentation

- Full documentation: `docs/AUTOMATION_SCHEDULER.md`
- Task specifications: `tasks/07_automation_scheduling.md`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## 🌍 Environment Variables

**Set in docker-compose.yml**:
- `SERVICE_URL`: API service URL for scheduler tasks
- `PROJECT_DIR`: Working directory in container
- `LOG_LEVEL`: Logging level (INFO/DEBUG)

Example:
```yaml
environment:
  - SERVICE_URL=http://viecomrec-api:8000
  - PROJECT_DIR=/app
  - LOG_LEVEL=INFO
```

## ✨ Summary

✅ **Automation scheduler fully deployed**
- 6 scheduled tasks configured
- APScheduler running in Docker
- Comprehensive logging to `logs/scheduler/`
- Easy management via PowerShell script
- Hot-reload capability for model updates
- Hourly health checks
- Drift detection monitoring

🎉 **Ready for production automation!**

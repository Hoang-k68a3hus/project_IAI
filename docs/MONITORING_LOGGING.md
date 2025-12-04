# Monitoring & Logging System

Hệ thống giám sát và ghi log toàn diện cho Vietnamese Cosmetics Recommender.

## Tổng Quan

Hệ thống monitoring bao gồm:
- **Pipeline Tracking**: Theo dõi tất cả pipeline executions
- **Health Checks**: Kiểm tra sức khỏe hệ thống định kỳ
- **Drift Detection**: Phát hiện data drift hàng tuần
- **Alerting**: Gửi cảnh báo qua Email, Slack, log file
- **Scheduler**: Điều phối các jobs tự động

## Quick Start

### Chạy Health Check

```powershell
python -m automation.health_check
```

### Chạy Drift Detection

```powershell
python -m automation.drift_detection
```

### Khởi Động Scheduler

```powershell
python -m automation.scheduler
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                                │
├─────────────────────────────────────────────────────────────────┤
│  Training Logs    │  Service Logs   │  Pipeline Metrics DB      │
│  (logs/cf/)       │  (logs/service/)│  (logs/pipeline_metrics.db)│
└─────────┬─────────┴────────┬────────┴───────────┬───────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Aggregation & Analysis                         │
├─────────────────────────────────────────────────────────────────┤
│  Health Checks    │  Drift Detection  │  AlertManager           │
│  (health_check.py)│  (drift_detection)│  (alerting.py)          │
└─────────┬─────────┴────────┬──────────┴───────────┬─────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Actions                                  │
├─────────────────────────────────────────────────────────────────┤
│  Trigger Retrain  │  Rollback Model   │  Send Alerts            │
│  (model_training) │  (model_deployment)│  (Email/Slack/Log)      │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Pipeline Tracker

**Location**: `scripts/utils.py`

Theo dõi tất cả pipeline executions trong SQLite database.

#### Class: `PipelineTracker`

```python
from scripts.utils import PipelineTracker

tracker = PipelineTracker()

# Bắt đầu tracking
run_id = tracker.start_run("data_refresh", {"force": True})

try:
    # ... pipeline logic ...
    tracker.complete_run(run_id, {"records_processed": 1000})
except Exception as e:
    tracker.fail_run(run_id, str(e))
```

#### Methods

| Method | Mô tả |
|--------|-------|
| `start_run(pipeline_name, metadata)` | Bắt đầu tracking, trả về `run_id` |
| `complete_run(run_id, metadata)` | Đánh dấu hoàn thành |
| `fail_run(run_id, error_message)` | Đánh dấu thất bại |
| `cancel_run(run_id)` | Đánh dấu hủy bỏ |
| `get_run(run_id)` | Lấy thông tin run |
| `get_recent_runs(pipeline_name, limit, status)` | Query recent runs |
| `get_stats(days=7)` | Thống kê pipeline |
| `is_pipeline_running(pipeline_name)` | Kiểm tra đang chạy |
| `cleanup_stale_runs(max_running_hours=24)` | Dọn dẹp stale runs |

#### Lấy Thống Kê

```python
stats = tracker.get_stats(days=7)

# Output:
{
    'period_days': 7,
    'stats_by_pipeline': {
        'data_refresh': {
            'success': 5, 
            'failed': 1, 
            'running': 0,
            'avg_duration_seconds': 45.2,
            'success_rate': 0.833
        },
        'model_training': {...},
        'drift_detection': {...}
    }
}
```

### 2. Pipeline Lock

**Location**: `scripts/utils.py`

Ngăn chặn concurrent pipeline runs bằng file-based locks.

```python
from scripts.utils import PipelineLock

with PipelineLock("data_refresh") as lock:
    if lock.acquired:
        # Chạy pipeline an toàn
        run_pipeline()
    else:
        print("Pipeline đang chạy bởi process khác")

# Auto-cleanup stale locks (>24 hours)
```

### 3. Health Check System

**Location**: `automation/health_check.py`

Kiểm tra sức khỏe toàn diện của hệ thống.

#### Chạy Health Check

```powershell
python -m automation.health_check
```

#### Health Check Components

| Check | Function | Kiểm tra |
|-------|----------|----------|
| **Service** | `check_service_health()` | API reachable qua `/health` |
| **Data** | `check_data_health()` | Data files tồn tại |
| **Model** | `check_model_health()` | Registry và model artifacts |
| **Pipeline** | `check_pipeline_health()` | Success rate các pipelines |

#### Service Health Check

```python
def check_service_health():
    """Kiểm tra service có reachable và responding."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return {
            'healthy': response.status_code == 200,
            'latency_ms': response.elapsed.total_seconds() * 1000,
            'response': response.json()
        }
    except requests.RequestException as e:
        return {'healthy': False, 'error': str(e)}
```

#### Data Health Check

Kiểm tra các required files:

```python
required_files = [
    'data/processed/interactions.parquet',
    'data/processed/X_train_confidence.npz',
    'data/processed/user_item_mappings.json',
    'data/processed/user_metadata.pkl'
]
```

#### Model Health Check

```python
def check_model_health():
    """Kiểm tra model registry và artifacts."""
    # Check registry exists
    # Verify current_best model
    # Verify model files exist
```

#### Pipeline Health Check

```python
def check_pipeline_health():
    """Kiểm tra success rate của các pipelines."""
    # Unhealthy nếu bất kỳ pipeline nào có <50% success rate
```

### 4. Drift Detection

**Location**: `automation/drift_detection.py`

Phát hiện data drift trong rating, popularity, và interaction patterns.

#### Chạy Drift Detection

```powershell
python -m automation.drift_detection
```

#### Drift Types

##### Rating Distribution Drift

```python
def detect_rating_drift(
    historical_data: pd.DataFrame,
    new_data: pd.DataFrame,
    threshold: float = 0.1  # 10% difference threshold
) -> Dict[str, Any]:
    """
    Phát hiện thay đổi trong rating distribution.
    
    Returns:
        {
            'drift_detected': bool,
            'severity': 'low' | 'medium' | 'high',
            'max_difference': float,
            'rating_comparison': {...}
        }
    """
```

##### Popularity Shift Detection

```python
def detect_popularity_drift(
    old_popularity: pd.Series,
    new_popularity: pd.Series,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Phát hiện thay đổi trong popularity rankings.
    
    Sử dụng Jaccard similarity của top-K items.
    
    Returns:
        {
            'shift_detected': bool,
            'jaccard_similarity': float,  # 0-1
            'new_trending': list,  # Items mới vào top-K
            'dropped_out': list    # Items rời khỏi top-K
        }
    """
```

##### Interaction Volume Drift

```python
def detect_interaction_drift(
    historical_stats: Dict,
    new_stats: Dict
) -> Dict[str, Any]:
    """
    Phát hiện thay đổi trong interaction patterns.
    
    Returns:
        {
            'drift_detected': bool,
            'volume_change_pct': float,
            'user_activity_change_pct': float,
            'new_users_pct': float
        }
    """
```

#### Drift Report

Output: `reports/drift/drift_report_YYYYMMDD.json`

```json
{
    "generated_at": "2025-01-20T09:00:00",
    "rating_drift": {
        "drift_detected": false,
        "severity": "low",
        "max_difference": 0.05
    },
    "popularity_drift": {
        "shift_detected": true,
        "jaccard_similarity": 0.75,
        "new_trending": ["product_123", "product_456"]
    },
    "interaction_drift": {
        "drift_detected": false,
        "volume_change_pct": 5.2
    },
    "overall_severity": "medium",
    "recommendations": [
        "Popular items changed - update popularity baseline"
    ]
}
```

### 5. Alerting System

**Location**: `alerting.py`

Multi-channel alerting với Email, Slack, và file logging.

#### Class: `AlertManager`

```python
from alerting import AlertManager

manager = AlertManager()

# Gửi alert
manager.send_alert(
    alert_name="high_latency",
    message="Average latency exceeded 200ms",
    severity="warning",
    channels=['log', 'email']
)
```

#### Alert Severities

| Severity | Mô tả | Channels mặc định |
|----------|-------|-------------------|
| `info` | Thông tin | log |
| `warning` | Cảnh báo | log, email |
| `critical` | Nghiêm trọng | log, email, slack |

#### Configuration

**File**: `config/alerts_config.yaml`

```yaml
alerts:
  high_latency:
    threshold: 200  # ms
    severity: warning
    channels: [log, email]
  
  critical_latency:
    threshold: 500  # ms
    severity: critical
    channels: [log, email, slack]
  
  high_error_rate:
    threshold: 0.05  # 5%
    severity: critical
    channels: [log, email, slack]
  
  high_fallback_rate:
    threshold: 0.3  # 30%
    severity: warning
    channels: [log, email]

email:
  enabled: true
  smtp_server: "${SMTP_SERVER}"
  smtp_port: 587
  sender: "${ALERT_EMAIL_SENDER}"
  password: "${ALERT_EMAIL_PASSWORD}"
  recipients:
    - team@example.com

slack:
  enabled: true
  webhook_url: "${SLACK_WEBHOOK_URL}"
  channel: "#ml-alerts"

logging:
  enabled: true
  path: logs/service/alerts.log
```

#### Environment Variables

```powershell
# Email
$env:SMTP_SERVER = "smtp.gmail.com"
$env:ALERT_EMAIL_SENDER = "alerts@example.com"
$env:ALERT_EMAIL_PASSWORD = "your-app-password"

# Slack
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

#### Automatic Alert Checking

```python
# Kiểm tra metrics và trigger alerts tự động
triggered = manager.check_alert_conditions({
    'avg_latency_ms': 215.0,
    'error_rate': 0.02,
    'fallback_rate': 0.25
})

for alert in triggered:
    print(f"Alert: {alert['alert_name']} - {alert['metric']}={alert['value']}")
```

#### Alert Log Output

**File**: `logs/service/alerts.log`

```
2025-01-15 15:35:00 | WARNING | alerting | high_latency triggered: avg_latency_ms=215.3 > 200
2025-01-15 15:35:01 | INFO | alerting | Email sent to team@example.com
2025-01-15 16:00:00 | CRITICAL | alerting | high_error_rate triggered: error_rate=0.08 > 0.05
2025-01-15 16:00:01 | INFO | alerting | Slack notification sent
```

### 6. Model Deployment Monitoring

**Location**: `automation/model_deployment.py`

Deployment với hot-reload, verification, và rollback.

#### Deploy Model

```powershell
# Deploy current best từ registry
python -m automation.model_deployment

# Deploy specific model
python -m automation.model_deployment --model-id als_20250115_100000
```

#### Deployment Workflow

```
deploy_model()
├─ Get model_id từ registry
├─ Ghi nhớ previous model cho rollback
├─ Call POST /reload_model
├─ Verify deployment:
│  ├─ Check /health endpoint
│  ├─ Test recommendations
│  └─ Verify model_id matches
├─ On failure:
│  ├─ Rollback to previous model
│  └─ Send alert
└─ Record deployment history
```

#### Deployment History

**File**: `logs/deployment_history.json`

```json
{
    "deployments": [
        {
            "timestamp": "2025-01-15T10:00:00",
            "model_id": "als_20250115_100000",
            "previous_model_id": "als_20250110_090000",
            "status": "success",
            "verification": {
                "health_check": true,
                "model_id_match": true,
                "recommendations_work": true
            },
            "rollback_triggered": false
        }
    ]
}
```

### 7. Data Refresh Pipeline

**Location**: `automation/data_refresh.py`

Data refresh với incremental updates.

#### Chạy Data Refresh

```powershell
# Normal refresh (auto-detect incremental vs full)
python -m automation.data_refresh

# Force full refresh
python -m automation.data_refresh --force-full
```

#### Refresh Strategy

| Điều kiện | Strategy |
|-----------|----------|
| ≤100 new interactions | Incremental update |
| >100 new interactions | Full pipeline |
| `--force-full` flag | Full pipeline |

#### Incremental Update

```python
def incremental_update(new_data, existing_data_path):
    """
    Incremental update cho small data changes.
    
    1. Load existing data
    2. Score new comments (AI)
    3. Merge và deduplicate
    4. Save updated data
    """
```

### 8. Scheduler

**Location**: `automation/scheduler.py`

APScheduler-based job scheduling.

#### Khởi Động Scheduler

```powershell
# Foreground
python -m automation.scheduler

# Via manage script
.\manage_scheduler.ps1 -Action start
```

#### Job Schedule

| Job | Schedule | Mô tả |
|-----|----------|-------|
| `data_refresh` | Daily 2:00 AM | Refresh data pipeline |
| `bert_embeddings` | Tuesday 3:00 AM | Update BERT embeddings |
| `drift_detection` | Monday 9:00 AM | Weekly drift check |
| `model_training` | Sunday 3:00 AM | Weekly model retraining |
| `model_deployment` | Daily 5:00 AM | Deploy best model |
| `health_check` | Hourly :00 | Health monitoring |

#### Configuration

**File**: `config/scheduler_config.json`

```json
{
    "timezone": "Asia/Ho_Chi_Minh",
    "jobs": {
        "data_refresh": {
            "enabled": true,
            "cron": "0 2 * * *",
            "timeout_minutes": 30
        },
        "health_check": {
            "enabled": true,
            "cron": "0 * * * *",
            "timeout_minutes": 5
        }
    }
}
```

#### Scheduler Logs

**File**: `logs/scheduler/scheduler.log`

```
2025-01-15 02:00:00 | INFO | Starting job: data_refresh
2025-01-15 02:01:30 | INFO | Job completed: data_refresh | duration=90s | status=success
2025-01-15 03:00:00 | INFO | Starting job: health_check
2025-01-15 03:00:05 | INFO | Job completed: health_check | duration=5s | status=success
```

## Databases

### Pipeline Metrics DB

**Path**: `logs/pipeline_metrics.db`

```sql
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'running', 'success', 'failed', 'cancelled'
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_seconds REAL,
    error_message TEXT,
    metadata TEXT  -- JSON
);
```

### Training Metrics DB

**Path**: `logs/training_metrics.db`

```sql
CREATE TABLE training_runs (
    run_id TEXT PRIMARY KEY,
    model_type TEXT,  -- 'als' or 'bpr'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,
    hyperparameters TEXT,  -- JSON
    recall_at_10 REAL,
    ndcg_at_10 REAL,
    baseline_recall_at_10 REAL,
    improvement_pct REAL,
    model_id TEXT
);
```

### Service Metrics DB

**Path**: `logs/service_metrics.db`

```sql
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP,
    user_id INTEGER,
    topk INTEGER,
    latency_ms REAL,
    fallback BOOLEAN,
    error TEXT,
    model_id TEXT
);
```

## Log Files

| File | Path | Mô tả |
|------|------|-------|
| ALS Training | `logs/cf/als.log` | ALS training logs |
| BPR Training | `logs/cf/bpr.log` | BPR training logs |
| Service | `logs/service/recommender.log` | API request logs |
| Alerts | `logs/service/alerts.log` | Triggered alerts |
| Scheduler | `logs/scheduler/scheduler.log` | Scheduler activity |
| Registry | `logs/registry_audit.log` | Registry operations |

## Command Reference

### Manual Pipeline Runs

```powershell
# Data refresh
python -m automation.data_refresh
python -m automation.data_refresh --force-full

# Drift detection
python -m automation.drift_detection

# Model training
python -m automation.model_training
python -m automation.model_training --model-types als bpr

# Model deployment
python -m automation.model_deployment
python -m automation.model_deployment --model-id <model_id>

# Health check
python -m automation.health_check

# BERT embeddings
python -m automation.bert_embeddings
```

### Scheduler Management

```powershell
# Start scheduler
python -m automation.scheduler
.\manage_scheduler.ps1 -Action start

# Stop scheduler
.\manage_scheduler.ps1 -Action stop

# Check status
.\manage_scheduler.ps1 -Action status
```

### Query Pipeline Stats

```powershell
# Via Python
python -c "from scripts.utils import PipelineTracker; import json; print(json.dumps(PipelineTracker().get_stats(), indent=2))"
```

## Troubleshooting

### Pipeline Stuck in "Running"

```python
from scripts.utils import PipelineTracker

tracker = PipelineTracker()
# Cleanup runs stuck >24 hours
tracker.cleanup_stale_runs(max_running_hours=24)
```

### Lock Not Released

```powershell
# Check lock files
Get-ChildItem logs/locks/

# Remove stale lock (>24 hours old)
Remove-Item logs/locks/data_refresh.lock
```

### Health Check Failing

1. **Service Health**: Kiểm tra API có đang chạy
   ```powershell
   curl http://localhost:8000/health
   ```

2. **Data Health**: Kiểm tra data files
   ```powershell
   Test-Path data/processed/interactions.parquet
   ```

3. **Model Health**: Kiểm tra registry
   ```powershell
   Get-Content artifacts/cf/registry.json | ConvertFrom-Json
   ```

### Alert Not Sending

1. Kiểm tra environment variables đã set
2. Kiểm tra `config/alerts_config.yaml` có đúng format
3. Xem logs: `logs/service/alerts.log`

## Best Practices

### 1. Pipeline Development

```python
from scripts.utils import PipelineTracker, PipelineLock

def my_pipeline():
    tracker = PipelineTracker()
    
    with PipelineLock("my_pipeline") as lock:
        if not lock.acquired:
            logger.warning("Pipeline already running")
            return
        
        run_id = tracker.start_run("my_pipeline", {"version": "1.0"})
        
        try:
            # Pipeline logic here
            result = do_work()
            tracker.complete_run(run_id, {"result": result})
        except Exception as e:
            tracker.fail_run(run_id, str(e))
            raise
```

### 2. Alert Integration

```python
from scripts.utils import send_pipeline_alert

def my_pipeline():
    try:
        # ... pipeline logic ...
        send_pipeline_alert("my_pipeline", "success", "Completed successfully")
    except Exception as e:
        send_pipeline_alert("my_pipeline", "failed", str(e), severity="critical")
        raise
```

### 3. Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# Structured logs
logger.info(f"Processing started | records={count}, batch_size={batch}")
logger.warning(f"Slow query | query_time={duration:.2f}s, threshold=1.0s")
logger.error(f"Failed to process | error={str(e)}, user_id={uid}")
```

## Dependencies

```txt
# requirements_monitoring.txt
apscheduler>=3.10.0  # Job scheduling
requests>=2.28.0     # HTTP calls
pyyaml>=6.0          # Config files
```

## Related Documentation

- [Model Registry](./MODEL_REGISTRY.md)
- [Serving Layer](./SERVING_LAYER.md)
- [Training Guide](./TRAINING_GUIDE.md)
- [Data Processing Guide](./DATA_PROCESSING_GUIDE.md)

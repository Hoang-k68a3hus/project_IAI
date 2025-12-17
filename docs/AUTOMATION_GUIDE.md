# VieComRec Automation & Scheduling Guide

## Tổng Quan

Hệ thống automation của VieComRec sử dụng APScheduler để tự động hóa toàn bộ ML pipeline bao gồm: data refresh, model training, deployment, health monitoring, drift detection và cleanup.

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APScheduler (BlockingScheduler)                  │
│                        Timezone: Asia/Ho_Chi_Minh                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  Daily Jobs   │         │  Weekly Jobs  │         │ Hourly Jobs   │
├───────────────┤         ├───────────────┤         ├───────────────┤
│ • Data Refresh│         │ • BERT Embed  │         │ • Health Check│
│   (2:00 AM)   │         │   (Tue 3AM)   │         │   (every :00) │
│ • Deployment  │         │ • Drift Detect│         └───────────────┘
│   (5:00 AM)   │         │   (Mon 9AM)   │
└───────────────┘         │ • Model Train │
                          │   (Sun 3AM)   │
                          └───────────────┘
```

## Cấu Trúc Module

```
automation/
├── __init__.py
├── scheduler.py          # Main scheduler orchestration
├── data_refresh.py       # Data pipeline (incremental + full)
├── model_training.py     # ALS/BPR training với BERT init
├── model_deployment.py   # Model deployment với rollback
├── health_check.py       # System health monitoring
├── drift_detection.py    # Data drift detection
├── bert_embeddings.py    # PhoBERT embeddings generation
└── cleanup.py            # Log và artifact cleanup

scripts/
└── utils.py              # PipelineTracker, PipelineLock, retry

config/
└── scheduler_config.json # Scheduler configuration
```

---

## Quick Start

### Khởi Động Scheduler

```powershell
# Chạy scheduler (blocking mode)
python -m automation.scheduler

# Chạy background (PowerShell)
.\manage_scheduler.ps1 start -Background

# Kiểm tra status
.\manage_scheduler.ps1 status

# Xem logs
.\manage_scheduler.ps1 logs
```

### Chạy Từng Pipeline Riêng Lẻ

```powershell
# Data Refresh
python -m automation.data_refresh [--force] [--force-full] [--dry-run]

# Model Training
python -m automation.model_training [--model als|bpr|both] [--auto-select] [--warmstart]

# Model Deployment
python -m automation.model_deployment [--model-id ID] [--rollback] [--dry-run]

# Health Check
python -m automation.health_check [--component all|service|data|models|pipelines] [--alert]

# Drift Detection
python -m automation.drift_detection [--update-baseline]

# BERT Embeddings
python -m automation.bert_embeddings [--force]

# Cleanup
python -m automation.cleanup [--type logs|checkpoints|all] [--dry-run]
```

---

## Chi Tiết Các Pipeline

### 1. Data Refresh Pipeline

**File**: `automation/data_refresh.py`  
**Schedule**: Daily 2:00 AM

#### Chức Năng
- Merge staging data từ web ingestion
- Hỗ trợ incremental update (< 100 new interactions)
- Full pipeline cho thay đổi lớn
- AI-powered comment quality scoring

#### Luồng Xử Lý

```
Raw Data (staging/) → Merge → Decision:
                              ├─ < 100 new → Incremental Update
                              │              ├─ Update matrices (LIL format)
                              │              ├─ AI scoring (ViSoBERT)
                              │              └─ Track newly trainable users
                              │
                              └─ >= 100 new → Full Pipeline
                                             ├─ Load & validate
                                             ├─ Feature engineering
                                             ├─ User segmentation
                                             ├─ ID mapping
                                             ├─ Temporal split
                                             └─ Build matrices
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--force` | Force refresh ngay cả khi không có data mới |
| `--force-full` | Force chạy full pipeline (bỏ qua incremental) |
| `--dry-run` | Chỉ show kế hoạch, không thực hiện |
| `--skip-merge` | Bỏ qua bước merge staging data |

#### Output Files
- `data/processed/interactions.parquet`
- `data/processed/X_train_confidence.npz`
- `data/processed/X_train_binary.npz`
- `data/processed/user_item_mappings.json`
- `data/processed/user_metadata.pkl`
- `data/processed/user_pos_train.pkl`
- `data/processed/data_stats.json`

---

### 2. Model Training Pipeline

**File**: `automation/model_training.py`  
**Schedule**: Weekly Sunday 3:00 AM

#### Chức Năng
- Train ALS với confidence-weighted matrix
- Train BPR với hard negative sampling
- BERT initialization cho cold-start items
- Early stopping (BPR)
- Warm-start từ previous model
- Checkpointing cho crash recovery

#### Features

##### ALS Training
```python
# Configuration
TRAINING_CONFIG["als"] = {
    "factors": 64,
    "regularization": 0.1,      # Higher due to sparsity
    "iterations": 15,
    "alpha": 5,
    "use_gpu": False,
    "use_bert_init": True,
    "bert_init_cold_threshold": 5,  # Items với < 5 interactions
}
```

##### BPR Training
```python
TRAINING_CONFIG["bpr"] = {
    "factors": 64,
    "learning_rate": 0.05,
    "regularization": 0.0001,
    "epochs": 50,
    "neg_sample_ratio": 0.3,    # 30% hard negatives
}

TRAINING_CONFIG["early_stopping"] = {
    "enabled": True,
    "patience": 5,
    "min_delta": 0.001,
}
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--model als\|bpr\|both` | Chọn model để train |
| `--auto-select` | Tự động chọn model tốt nhất |
| `--warmstart` | Warm-start từ model trước |
| `--skip-eval` | Bỏ qua evaluation |
| `--force` | Force train ngay cả khi không có data mới |

#### Output Structure
```
artifacts/cf/
├── als/
│   └── als_20250130_v1/
│       ├── U.npy               # User factors (26K x 64)
│       ├── V.npy               # Item factors (2.2K x 64)
│       ├── params.json         # Hyperparameters
│       ├── metrics.json        # Evaluation metrics
│       └── metadata.json       # Training info
├── bpr/
│   └── bpr_20250130_v1/
│       └── ...
└── registry.json               # Model registry
```

---

### 3. Model Deployment Pipeline

**File**: `automation/model_deployment.py`  
**Schedule**: Daily 5:00 AM

#### Chức Năng
- Deploy model từ registry
- Rollback support
- Verify deployment qua API
- Record deployment history
- Handle offline service gracefully

#### Deployment Flow

```
Load Registry → Determine Model → Check Service Health:
                                   ├─ Online → Trigger Reload → Verify → Record
                                   └─ Offline → Update Registry → Record (pending)
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--model-id ID` | Deploy specific model |
| `--rollback` | Rollback về model trước |
| `--dry-run` | Chỉ show kế hoạch |

#### Rollback Example
```powershell
# Rollback to previous model
python -m automation.model_deployment --rollback

# Deploy specific model
python -m automation.model_deployment --model-id als_20250129_v1
```

---

### 4. BERT Embeddings Refresh

**File**: `automation/bert_embeddings.py`  
**Schedule**: Weekly Tuesday 3:00 AM

#### Chức Năng
- Generate PhoBERT embeddings cho products
- Skip nếu embeddings còn fresh (< 7 days)
- Save dưới dạng PyTorch tensor

#### Text Processing
```python
# Combine fields with [SEP] token
text = f"{product_name} [SEP] {description} [SEP] {feature}"

# Generate embedding using CLS token
embedding = model(**inputs).last_hidden_state[:, 0, :]  # (768-dim)
```

#### Output
```
data/processed/content_based_embeddings/
└── product_embeddings.pt
    ├── product_ids: List[str]
    ├── embeddings: Tensor (N x 768)
    ├── model: "vinai/phobert-base"
    └── created_at: ISO timestamp
```

---

### 5. Health Check System

**File**: `automation/health_check.py`  
**Schedule**: Hourly at :00

#### Components Checked

| Component | Checks |
|-----------|--------|
| **Service** | API reachable, model loaded |
| **Data** | Files exist, freshness (< 7 days), embeddings |
| **Models** | Registry valid, current_best exists, performance |
| **Pipelines** | Success rate (> 50%), stale runs cleanup |

#### Status Levels
- **healthy**: All checks passed
- **warning**: Non-critical issues
- **critical**: Service down or data missing

#### CLI Options

| Option | Description |
|--------|-------------|
| `--component all\|service\|data\|models\|pipelines` | Check specific component |
| `--json` | Output as JSON |
| `--alert` | Send alerts on failures |

#### Example Output
```json
{
  "timestamp": "2025-01-30T10:00:00",
  "overall_status": "healthy",
  "components": {
    "service": {"status": "healthy", "checks": [...]},
    "data": {"status": "healthy", "checks": [...]},
    "models": {"status": "healthy", "checks": [...]},
    "pipelines": {"status": "healthy", "checks": [...]}
  }
}
```

---

### 6. Drift Detection Pipeline

**File**: `automation/drift_detection.py`  
**Schedule**: Weekly Monday 9:00 AM

#### Drift Types Detected

| Type | Method | Threshold |
|------|--------|-----------|
| Rating Distribution | Total absolute difference | > 0.1 |
| Popularity Shift | Jaccard similarity (top 20 items) | < 0.8 |
| Interaction Rate | Percentage change | > 30% |

#### Baseline Management
```powershell
# First run creates baseline
python -m automation.drift_detection

# Update baseline manually
python -m automation.drift_detection --update-baseline
```

#### Output
```
reports/drift/
└── drift_report_20250130.json
    ├── rating_distribution: {drift_detected, total_difference}
    ├── popularity_distribution: {jaccard_similarity, new_items, dropped_items}
    └── interaction_rate: {change_rate}
```

---

### 7. Cleanup Pipeline

**File**: `automation/cleanup.py`  
**Schedule**: Monthly (manual or scheduled)

#### Cleanup Targets

| Target | Retention Policy |
|--------|------------------|
| Logs | 30 days |
| Checkpoints | Keep last 3 per model type |
| Model Artifacts | Keep last 5 + deployed model |

#### CLI Options

| Option | Description |
|--------|-------------|
| `--type logs\|checkpoints\|models\|all` | Cleanup specific type |
| `--dry-run` | Preview what would be deleted |

#### Dry Run Example
```powershell
python -m automation.cleanup --dry-run --type all
# Output:
# {
#   "status": "success",
#   "dry_run": true,
#   "results": [
#     {"type": "logs", "deleted_count": 45, "deleted_size_mb": 12.5},
#     {"type": "checkpoints", "deleted_count": 6, "deleted_size_mb": 120.0},
#     {"type": "models", "deleted_count": 2, "deleted_size_mb": 85.0}
#   ],
#   "total_freed_mb": 217.5
# }
```

---

## Utility Components

### PipelineTracker

SQLite-based tracking cho pipeline runs.

```python
from scripts.utils import PipelineTracker

tracker = PipelineTracker()

# Start run
run_id = tracker.start_run("data_refresh", {"force": True})

# Complete run
tracker.complete_run(run_id, {"status": "success", "records": 1000})

# Or fail run
tracker.fail_run(run_id, "Connection timeout")

# Get stats
stats = tracker.get_stats(days=7)
# {'days': 7, 'stats_by_pipeline': {'data_refresh': {'total': 7, 'completed': 6, 'failed': 1, 'success_rate': 0.857}}}
```

### PipelineLock

File-based lock để ngăn concurrent runs.

```python
from scripts.utils import PipelineLock

with PipelineLock("data_refresh", timeout=3600) as lock:
    if lock.acquired:
        run_pipeline()
    else:
        print("Pipeline already running")
```

Features:
- Stale lock detection (auto-cleanup sau timeout)
- Process ID tracking
- Safe cleanup on exit

### Retry Decorator

Exponential backoff retry.

```python
from scripts.utils import retry

@retry(max_attempts=3, delay=60, backoff=2)
def unreliable_operation():
    # May fail sometimes
    pass

# Attempts at: 0s, 60s, 120s (total 3 attempts)
```

---

## Configuration

### scheduler_config.json

```json
{
  "timezone": "Asia/Ho_Chi_Minh",
  "jobs": {
    "data_refresh": {
      "enabled": true,
      "description": "Daily data refresh from raw CSV files",
      "schedule": {"hour": 2, "minute": 0},
      "module": "automation.data_refresh",
      "args": []
    },
    "bert_embeddings": {
      "enabled": true,
      "schedule": {"day_of_week": "tue", "hour": 3, "minute": 0},
      "module": "automation.bert_embeddings"
    },
    "drift_detection": {
      "enabled": true,
      "schedule": {"day_of_week": "mon", "hour": 9, "minute": 0},
      "module": "automation.drift_detection"
    },
    "model_training": {
      "enabled": true,
      "schedule": {"day_of_week": "sun", "hour": 3, "minute": 0},
      "module": "automation.model_training",
      "args": ["--auto-select"]
    },
    "model_deployment": {
      "enabled": true,
      "schedule": {"hour": 5, "minute": 0},
      "module": "automation.model_deployment"
    },
    "health_check": {
      "enabled": true,
      "schedule": {"minute": 0},
      "module": "automation.health_check"
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_DIR` | Auto-detect | Project root directory |
| `SERVICE_URL` | `http://localhost:8000` | Recommendation service URL |
| `TZ` | `UTC` | Timezone for scheduler |

---

## Monitoring & Alerting

### Task Status Tracking

Status được lưu tại `logs/scheduler/task_status.json`:

```json
{
  "data_refresh": {
    "status": "success",
    "timestamp": "2025-01-30T02:15:00",
    "exit_code": 0,
    "log_file": "logs/scheduler/data_refresh_20250130_020000.log"
  },
  "model_training": {
    "status": "failed",
    "timestamp": "2025-01-26T03:45:00",
    "exit_code": 1,
    "error": "CUDA out of memory"
  }
}
```

### Pipeline Metrics Database

SQLite database tại `logs/pipeline_metrics.db`:

```sql
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT DEFAULT 'running',
    config TEXT,       -- JSON
    result TEXT,       -- JSON
    error_message TEXT
);
```

### Alert Integration

Alerts được gửi qua `alerting.py`:

```python
from scripts.utils import send_pipeline_alert

send_pipeline_alert(
    pipeline_name="model_training",
    severity="error",      # info, warning, error
    message="Training failed: CUDA OOM",
    details={"model": "als", "epoch": 10}
)
```

---

## Troubleshooting

### Common Issues

#### 1. Pipeline Already Running
```
{"status": "skipped", "message": "Already running"}
```
**Solution**: Check lock file tại `logs/locks/{pipeline_name}.lock`. Delete nếu stale.

#### 2. Service Offline During Deployment
```
{"status": "pending", "message": "Service offline, will deploy on startup"}
```
**Solution**: Registry được cập nhật, model sẽ load khi service restart.

#### 3. Incremental Update Fallback
```
Falling back to full pipeline: Matrix dimension mismatch
```
**Solution**: New users/items detected, full pipeline chạy tự động.

#### 4. BERT Embeddings Skipped
```
{"status": "skipped", "message": "Fresh (3 days old)"}
```
**Solution**: Dùng `--force` để regenerate.

### Log Locations

| Log Type | Location |
|----------|----------|
| Scheduler | `logs/scheduler/scheduler.log` |
| Task runs | `logs/scheduler/{task}_{timestamp}.log` |
| Pipeline metrics | `logs/pipeline_metrics.db` |
| Locks | `logs/locks/*.lock` |

### Manual Recovery

```powershell
# Clear all locks
Remove-Item logs/locks/*.lock

# Reset pipeline tracker
Remove-Item logs/pipeline_metrics.db

# Force full data refresh
python -m automation.data_refresh --force-full

# Rollback model
python -m automation.model_deployment --rollback
```

---

## Best Practices

### 1. Production Deployment

```powershell
# Set timezone
$env:TZ = "Asia/Ho_Chi_Minh"

# Use service manager (systemd on Linux)
# Or Windows Task Scheduler
.\manage_scheduler.ps1 start -Background
```

### 2. Pre-Deployment Checklist

- [ ] Data files tại `data/published_data/`
- [ ] Service running tại `SERVICE_URL`
- [ ] Sufficient disk space for artifacts
- [ ] Database connections available

### 3. Monitoring Setup

- [ ] Configure alert recipients
- [ ] Set up log rotation (handled by cleanup pipeline)
- [ ] Monitor `pipeline_metrics.db` for success rates
- [ ] Review drift reports weekly

### 4. Disaster Recovery

1. **Data corruption**: Re-run `python -m automation.data_refresh --force-full`
2. **Model performance drop**: `python -m automation.model_deployment --rollback`
3. **Service unresponsive**: Check health logs, restart service

---

## Dependencies

```txt
# Core
apscheduler>=3.10.0
pytz>=2023.3
requests>=2.28.0

# ML
implicit>=0.7.0
scipy>=1.10.0
numpy>=1.24.0

# BERT
torch>=2.0.0
transformers>=4.30.0

# Data
pandas>=2.0.0
pyarrow>=12.0.0
```

---


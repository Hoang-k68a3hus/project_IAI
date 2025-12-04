# Task 06: Monitoring & Logging

## Mục Tiêu

Xây dựng hệ thống monitoring và logging toàn diện để theo dõi training performance, service health, data quality, và trigger retraining khi cần. System phải hỗ trợ debugging, alerting, và continuous improvement.

## Implementation Status

### ✅ Đã Hoàn Thành

| Component | Module | Mô tả |
|-----------|--------|-------|
| **Pipeline Tracker** | `scripts/utils.py` | SQLite-based tracking với `PipelineTracker` class |
| **Drift Detection** | `automation/drift_detection.py` | Rating, popularity, interaction drift detection |
| **Health Check** | `automation/health_check.py` | Service, data, model, pipeline health checks |
| **Alerting System** | `alerting.py` | Email, Slack, logging alerts với `AlertManager` |
| **Scheduler** | `automation/scheduler.py` | APScheduler-based job scheduling |
| **Data Refresh** | `automation/data_refresh.py` | Incremental + full pipeline với tracking |
| **Model Training** | `automation/model_training.py` | Auto-register, baseline comparison |
| **Model Deployment** | `automation/model_deployment.py` | Hot-reload, rollback, verification |

### 📁 Key Files & Databases

| File/DB | Path | Purpose |
|---------|------|---------|
| `pipeline_metrics.db` | `logs/pipeline_metrics.db` | Pipeline runs tracking |
| `service_metrics.db` | `logs/service_metrics.db` | Service request metrics |
| `training_metrics.db` | `logs/training_metrics.db` | Training runs và iterations |
| `registry_audit.log` | `logs/registry_audit.log` | Registry operations audit |
| `deployment_history.json` | `logs/deployment_history.json` | Deployment records |
| `alerts.log` | `logs/service/alerts.log` | All triggered alerts |
| `scheduler.log` | `logs/scheduler/scheduler.log` | Scheduler activity |

## Monitoring Architecture

```
Data Sources
    ↓
├─ Training Logs (logs/cf/als.log, logs/cf/bpr.log)
├─ Service Logs (logs/service/recommender.log)
├─ Registry Audit Logs (logs/registry_audit.log)
├─ Pipeline Metrics DB (logs/pipeline_metrics.db)
├─ Service Metrics DB (logs/service_metrics.db)
└─ Data Quality Checks
    ↓
Aggregation & Analysis (automation/)
    ↓
├─ Health Checks (automation/health_check.py)
├─ Drift Detection (automation/drift_detection.py)
├─ AlertManager (alerting.py)
└─ Registry Monitoring
    ↓
Actions
    ↓
├─ Trigger Retrain (automation/model_training.py)
├─ Rollback Model (automation/model_deployment.py --rollback)
├─ Model Hot Reload (POST /reload_model)
└─ Send Alerts (Email/Slack/Log)
```

## Component 1: Pipeline Tracking (Implemented)

### Class: `PipelineTracker`

Location: `scripts/utils.py`

Tracks all pipeline executions in SQLite database với status, duration, và metadata.

#### Database Schema: `logs/pipeline_metrics.db`

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

CREATE INDEX idx_pipeline_name_status ON pipeline_runs(pipeline_name, status);
CREATE INDEX idx_started_at ON pipeline_runs(started_at);
```

#### Usage

```python
from scripts.utils import PipelineTracker

tracker = PipelineTracker()

# Start tracking
run_id = tracker.start_run("data_refresh", {"force": True})

try:
    # ... pipeline logic ...
    tracker.complete_run(run_id, {"records_processed": 1000})
except Exception as e:
    tracker.fail_run(run_id, str(e))
```

#### Methods

| Method | Description |
|--------|-------------|
| `start_run(pipeline_name, metadata)` | Start tracking, returns `run_id` |
| `complete_run(run_id, metadata)` | Mark as success |
| `fail_run(run_id, error_message)` | Mark as failed |
| `cancel_run(run_id)` | Mark as cancelled |
| `get_run(run_id)` | Get run details |
| `get_recent_runs(pipeline_name, limit, status)` | Query recent runs |
| `get_stats(days=7)` | Pipeline statistics |
| `is_pipeline_running(pipeline_name)` | Check concurrent runs |
| `cleanup_stale_runs(max_running_hours=24)` | Cleanup stale runs |

#### Statistics Output

```python
stats = tracker.get_stats(days=7)

# Returns:
{
    'period_days': 7,
    'stats_by_pipeline': {
        'data_refresh': {
            'success': 5, 'failed': 1, 'running': 0,
            'avg_duration_seconds': 45.2,
            'success_rate': 0.833
        },
        'model_training': {...},
        'drift_detection': {...}
    }
}
```

### Class: `PipelineLock`

Prevents concurrent pipeline runs using file-based locks.

```python
from scripts.utils import PipelineLock

with PipelineLock("data_refresh") as lock:
    if lock.acquired:
        # Run pipeline safely
        pass
    else:
        print("Pipeline already running")

# Auto-cleanup stale locks (>24 hours)
```

### Dataclass: `PipelineRun`

```python
@dataclass
class PipelineRun:
    run_id: str
    pipeline_name: str
    status: str  # 'running', 'success', 'failed', 'cancelled'
    started_at: str
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

## Component 2: Training Monitoring (Implemented)

### Module: `automation/model_training.py`

Training pipeline với advanced features: BERT initialization, warm-start, early stopping, checkpointing.

#### Main Function: `train_models()`

```python
def train_models(
    model_types: List[str] = None,  # ['als', 'bpr'] or None for both
    force_retrain: bool = False,
    auto_deploy: bool = True
) -> Dict[str, Any]
```

**Features:**
- **BERT Initialization**: Projects PhoBERT embeddings (768-dim) → item factors (64-dim) via SVD
- **Warm-start**: Loads previous model factors as initialization
- **Early Stopping**: Stops if no improvement for `patience` epochs (default: 5)
- **Checkpointing**: Saves checkpoints every N epochs to `checkpoints/{model_type}/`
- **Baseline Comparison**: Auto-computes popularity baseline for improvement %
- **Auto-registration**: Registers to Model Registry if passes baseline

#### Log File Structure

**File: `logs/cf/{als|bpr}.log`**
```
2025-01-15 10:30:00 | INFO | ========== ALS Training Started ==========
2025-01-15 10:30:00 | INFO | Config: factors=64, regularization=0.05, iterations=15
2025-01-15 10:30:00 | INFO | Training data: 26,000 users × 2,200 items
2025-01-15 10:30:15 | INFO | Iteration 1/15 | loss=125.34
2025-01-15 10:30:30 | INFO | Iteration 2/15 | loss=98.12
...
2025-01-15 10:32:00 | INFO | Training completed in 45.2s
2025-01-15 10:32:05 | INFO | Evaluation: recall@10=0.234, ndcg@10=0.189
2025-01-15 10:32:05 | INFO | Baseline (popularity): recall@10=0.145
2025-01-15 10:32:05 | INFO | Improvement: +61.4%
2025-01-15 10:32:10 | INFO | Registered to registry: als_20250115_103000
```

#### Training Workflow

```
train_models()
├─ Check PipelineLock("model_training")
├─ Load training config from config/als_config.yaml or bpr_config.yaml
├─ Load data matrices from data/processed/
├─ For each model_type:
│  ├─ Initialize factors (BERT init or random)
│  ├─ Run training with checkpointing
│  ├─ Early stopping on validation NDCG@10
│  ├─ Evaluate vs popularity baseline
│  ├─ Register to Model Registry (if beats baseline)
│  └─ Auto-deploy (if auto_deploy=True)
└─ Send completion alert (success/failure)
```

#### Config: `config/als_config.yaml`

```yaml
model:
  factors: 64
  regularization: 0.05
  iterations: 15
  alpha: 10

training:
  use_gpu: false
  bert_init: true
  warm_start: false
  checkpoint_every: 5
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.001

evaluation:
  k_values: [10, 20]
  metrics: [recall, ndcg, coverage]
```

### Training Metrics Database

#### Schema: `logs/training_metrics.db` (SQLite)

##### Table: `training_runs`
```sql
CREATE TABLE training_runs (
    run_id TEXT PRIMARY KEY,
    model_type TEXT,  -- 'als' or 'bpr'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,  -- 'running', 'completed', 'failed'
    hyperparameters TEXT,  -- JSON
    recall_at_10 REAL,
    ndcg_at_10 REAL,
    coverage REAL,
    baseline_recall_at_10 REAL,
    improvement_pct REAL,
    training_time_seconds REAL,
    data_version TEXT,
    git_commit TEXT,
    artifacts_path TEXT,
    model_id TEXT,  -- Registry model_id
    registered_at TIMESTAMP,
    registry_status TEXT  -- 'active', 'archived', 'failed'
);
```

##### Table: `iteration_metrics`
```sql
CREATE TABLE iteration_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INT,
    timestamp TIMESTAMP,
    loss REAL,
    validation_recall REAL,
    validation_ndcg REAL,
    wall_time_seconds REAL,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);
```

## Component 3: Health Check System (Implemented)

### Module: `automation/health_check.py`

Comprehensive health checking cho all system components.

#### Main Function: `run_health_check()`

```python
def run_health_check() -> Dict[str, Any]:
    """
    Run comprehensive health check across all system components.
    
    Returns:
        dict: Health status for all components
    """
```

#### Health Check Components

| Check | Function | What It Checks |
|-------|----------|----------------|
| **Service** | `check_service_health()` | API reachability via `/health` endpoint |
| **Data** | `check_data_health()` | Required data files exist in `data/processed/` |
| **Model** | `check_model_health()` | Registry + model artifacts integrity |
| **Pipeline** | `check_pipeline_health()` | Recent pipeline success rates |

#### Implementation Details

```python
def check_service_health():
    """Check if service is reachable và responding."""
    try:
        response = requests.get(
            f"http://localhost:8000/health",
            timeout=5
        )
        return {
            'healthy': response.status_code == 200,
            'latency_ms': response.elapsed.total_seconds() * 1000,
            'response': response.json()
        }
    except requests.RequestException as e:
        return {'healthy': False, 'error': str(e)}

def check_data_health():
    """Check required data files."""
    required_files = [
        'data/processed/interactions.parquet',
        'data/processed/X_train_confidence.npz',
        'data/processed/user_item_mappings.json',
        'data/processed/user_metadata.pkl'
    ]
    
    results = {}
    for filepath in required_files:
        path = Path(filepath)
        results[filepath] = {
            'exists': path.exists(),
            'size_mb': path.stat().st_size / (1024*1024) if path.exists() else 0,
            'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
        }
    
    return {
        'healthy': all(r['exists'] for r in results.values()),
        'files': results
    }

def check_model_health():
    """Check model registry và artifacts."""
    registry_path = Path('artifacts/cf/registry.json')
    
    if not registry_path.exists():
        return {'healthy': False, 'error': 'Registry not found'}
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    current_best = registry.get('current_best')
    models = registry.get('models', {})
    
    # Verify current best model files exist
    if current_best:
        model_info = models.get(current_best, {})
        model_path = Path(model_info.get('path', ''))
        
        return {
            'healthy': model_path.exists(),
            'current_best': current_best,
            'total_models': len(models),
            'model_exists': model_path.exists()
        }
    
    return {'healthy': False, 'error': 'No current best model'}

def check_pipeline_health():
    """Check recent pipeline run status."""
    from scripts.utils import PipelineTracker
    
    tracker = PipelineTracker()
    stats = tracker.get_stats(days=7)
    
    # Unhealthy if any pipeline has <50% success rate
    unhealthy_pipelines = []
    for pipeline, metrics in stats.get('stats_by_pipeline', {}).items():
        if metrics.get('success_rate', 1.0) < 0.5:
            unhealthy_pipelines.append(pipeline)
    
    return {
        'healthy': len(unhealthy_pipelines) == 0,
        'unhealthy_pipelines': unhealthy_pipelines,
        'stats': stats
    }
```

#### Health Check Workflow

```
run_health_check()
├─ check_service_health()  → API reachable?
├─ check_data_health()     → Data files present?
├─ check_model_health()    → Registry & model valid?
├─ check_pipeline_health() → Pipelines running ok?
├─ Aggregate results
├─ Log to health_check.log
└─ Send alert if any component unhealthy
```

#### Scheduled Health Check

Run hourly via scheduler:

```python
# In automation/scheduler.py
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.health_check"]),
    CronTrigger(minute=0),  # Every hour at :00
    id="health_check",
    name="Hourly Health Check"
)
```

## Component 4: Data Drift Detection (Implemented)

### Module: `automation/drift_detection.py`

Comprehensive drift detection cho rating, popularity, và interaction patterns.

#### Main Function: `detect_drift()`

```python
def detect_drift(
    output_dir: str = 'reports/drift'
) -> Dict[str, Any]:
    """
    Run comprehensive drift detection pipeline.
    
    Returns:
        dict: Drift detection results with severity levels
    """
```

#### Drift Detection Functions

##### 1. Rating Distribution Drift

```python
def detect_rating_drift(
    historical_data: pd.DataFrame,
    new_data: pd.DataFrame,
    threshold: float = 0.1  # 10% difference threshold
) -> Dict[str, Any]:
    """
    Detect changes in rating distribution.
    
    Compares rating distributions between historical and new data.
    Uses absolute difference in rating percentages.
    
    Returns:
        dict: {
            'drift_detected': bool,
            'severity': str,  # 'low', 'medium', 'high'
            'max_difference': float,
            'rating_comparison': {...}
        }
    """
    historical_dist = historical_data['rating'].value_counts(normalize=True)
    new_dist = new_data['rating'].value_counts(normalize=True)
    
    differences = {}
    for rating in range(1, 6):
        hist_pct = historical_dist.get(rating, 0)
        new_pct = new_dist.get(rating, 0)
        differences[rating] = abs(hist_pct - new_pct)
    
    max_diff = max(differences.values())
    
    return {
        'drift_detected': max_diff > threshold,
        'severity': 'high' if max_diff > 0.2 else 'medium' if max_diff > 0.1 else 'low',
        'max_difference': max_diff,
        'rating_comparison': {
            'historical': dict(historical_dist),
            'new': dict(new_dist),
            'differences': differences
        }
    }
```

##### 2. Popularity Shift Detection

```python
def detect_popularity_drift(
    old_popularity: pd.Series,  # product_id → interaction_count
    new_popularity: pd.Series,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Detect shifts in item popularity rankings.
    
    Uses Jaccard similarity of top-K items.
    
    Returns:
        dict: {
            'shift_detected': bool,
            'jaccard_similarity': float,  # 0-1, higher = more stable
            'new_trending': list,  # New items entering top-K
            'dropped_out': list   # Items leaving top-K
        }
    """
    old_top_k = set(old_popularity.nlargest(top_k).index)
    new_top_k = set(new_popularity.nlargest(top_k).index)
    
    jaccard = len(old_top_k & new_top_k) / len(old_top_k | new_top_k)
    
    return {
        'shift_detected': jaccard < 0.8,  # <80% overlap = drift
        'jaccard_similarity': jaccard,
        'new_trending': list(new_top_k - old_top_k),
        'dropped_out': list(old_top_k - new_top_k),
        'stable_count': len(old_top_k & new_top_k)
    }
```

##### 3. Interaction Volume Drift

```python
def detect_interaction_drift(
    historical_stats: Dict,
    new_stats: Dict
) -> Dict[str, Any]:
    """
    Detect changes in interaction patterns.
    
    Compares interaction counts, user activity, etc.
    
    Returns:
        dict: {
            'drift_detected': bool,
            'volume_change_pct': float,
            'user_activity_change_pct': float,
            'new_users_pct': float
        }
    """
    volume_change = (new_stats['total_interactions'] - historical_stats['total_interactions']) / historical_stats['total_interactions']
    
    user_activity_change = (new_stats['avg_interactions_per_user'] - historical_stats['avg_interactions_per_user']) / historical_stats['avg_interactions_per_user']
    
    return {
        'drift_detected': abs(volume_change) > 0.2 or abs(user_activity_change) > 0.2,
        'volume_change_pct': volume_change * 100,
        'user_activity_change_pct': user_activity_change * 100,
        'new_users_pct': new_stats.get('new_users_pct', 0)
    }
```

#### Drift Report Generation

```python
def generate_drift_report(
    rating_drift: Dict,
    popularity_drift: Dict,
    interaction_drift: Dict,
    output_path: str
) -> str:
    """
    Generate comprehensive drift analysis report.
    
    Outputs JSON report with all drift metrics and recommendations.
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'rating_drift': rating_drift,
        'popularity_drift': popularity_drift,
        'interaction_drift': interaction_drift,
        'overall_severity': max(
            rating_drift.get('severity', 'low'),
            'high' if popularity_drift['shift_detected'] else 'low',
            'high' if interaction_drift['drift_detected'] else 'low',
            key=lambda x: {'low': 0, 'medium': 1, 'high': 2}[x]
        ),
        'recommendations': []
    }
    
    if rating_drift['drift_detected']:
        report['recommendations'].append('Rating distribution changed - consider retraining')
    if popularity_drift['shift_detected']:
        report['recommendations'].append('Popular items changed - update popularity baseline')
    if interaction_drift['drift_detected']:
        report['recommendations'].append('Interaction patterns changed - review data pipeline')
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path
```

#### Drift Detection Workflow

```
detect_drift()
├─ Load historical data (training data)
├─ Load new data (last 7 days from staging)
├─ detect_rating_drift()
├─ detect_popularity_drift()
├─ detect_interaction_drift()
├─ generate_drift_report()
├─ Track in PipelineTracker
├─ Send alert if high severity drift
└─ Return comprehensive results
```

#### Scheduled Drift Detection

Run weekly via scheduler:

```python
# In automation/scheduler.py
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.drift_detection"]),
    CronTrigger(day_of_week='mon', hour=9),  # Monday 9:00 AM
    id="drift_detection",
    name="Weekly Drift Detection"
)
```

#### Output Reports

Location: `reports/drift/`

```
reports/drift/
├── drift_report_20250120.json
├── drift_report_20250127.json
└── drift_summary.md
```

## Component 5: Alerting System (Implemented)

### Module: `alerting.py`

Multi-channel alerting system với Email, Slack, và file logging support.

#### Class: `AlertManager`

```python
class AlertManager:
    """
    Manages alerts across multiple channels.
    
    Channels:
    - Email (via SMTP)
    - Slack (via webhook)
    - File logging
    
    Args:
        config_path: Path to alerts_config.yaml
    """
    
    def __init__(self, config_path: str = 'config/alerts_config.yaml'):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger('alerting')
    
    def send_alert(
        self,
        alert_name: str,
        message: str,
        severity: str = 'info',  # 'info', 'warning', 'critical'
        channels: List[str] = None  # ['email', 'slack', 'log']
    ) -> Dict[str, bool]:
        """
        Send alert via configured channels.
        
        Returns:
            dict: {channel: success_bool} for each channel
        """
```

#### Configuration: `config/alerts_config.yaml`

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
  
  data_drift:
    threshold: true
    severity: warning
    channels: [log, email]
  
  model_degradation:
    threshold: 0.1  # 10% drop
    severity: critical
    channels: [log, email, slack]

email:
  enabled: true
  smtp_server: "${SMTP_SERVER}"
  smtp_port: 587
  sender: "${ALERT_EMAIL_SENDER}"
  password: "${ALERT_EMAIL_PASSWORD}"
  recipients:
    - team@example.com
    - oncall@example.com

slack:
  enabled: true
  webhook_url: "${SLACK_WEBHOOK_URL}"
  channel: "#ml-alerts"

logging:
  enabled: true
  path: logs/service/alerts.log
  format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
```

#### Alert Methods

##### Send Email Alert

```python
def _send_email(
    self,
    subject: str,
    message: str,
    recipients: List[str] = None
) -> bool:
    """
    Send email alert via SMTP.
    
    Uses TLS encryption.
    Supports HTML body.
    """
    try:
        msg = MIMEMultipart()
        msg['Subject'] = f"[{severity.upper()}] {subject}"
        msg['From'] = self.config['email']['sender']
        msg['To'] = ', '.join(recipients or self.config['email']['recipients'])
        
        msg.attach(MIMEText(message, 'html'))
        
        with smtplib.SMTP(
            self.config['email']['smtp_server'],
            self.config['email']['smtp_port']
        ) as server:
            server.starttls()
            server.login(
                self.config['email']['sender'],
                self.config['email']['password']
            )
            server.send_message(msg)
        
        return True
    except Exception as e:
        self.logger.error(f"Email failed: {e}")
        return False
```

##### Send Slack Alert

```python
def _send_slack(
    self,
    message: str,
    severity: str = 'info'
) -> bool:
    """
    Send Slack alert via webhook.
    
    Formats message with severity emoji.
    """
    emoji = {
        'info': ':information_source:',
        'warning': ':warning:',
        'critical': ':rotating_light:'
    }
    
    payload = {
        'text': f"{emoji.get(severity, '')} *{severity.upper()}*\n{message}",
        'channel': self.config['slack'].get('channel', '#alerts')
    }
    
    try:
        response = requests.post(
            self.config['slack']['webhook_url'],
            json=payload,
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        self.logger.error(f"Slack failed: {e}")
        return False
```

#### Automatic Alert Condition Checking

```python
def check_alert_conditions(
    self,
    metrics: Dict[str, float]
) -> List[Dict]:
    """
    Check metrics against configured thresholds.
    
    Args:
        metrics: {
            'avg_latency_ms': 150.0,
            'error_rate': 0.02,
            'fallback_rate': 0.25,
            ...
        }
    
    Returns:
        List of triggered alerts with details
    """
    triggered = []
    
    for alert_name, config in self.config['alerts'].items():
        metric_name = alert_name.replace('high_', '').replace('critical_', '')
        
        if metric_name in metrics:
            value = metrics[metric_name]
            threshold = config['threshold']
            
            if value > threshold:
                triggered.append({
                    'alert_name': alert_name,
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'severity': config['severity']
                })
    
    return triggered
```

#### Default Alerts

Predefined trong code:

```python
DEFAULT_ALERTS = {
    'high_latency': {
        'metric': 'avg_latency_ms',
        'threshold': 200,
        'severity': 'warning'
    },
    'critical_latency': {
        'metric': 'avg_latency_ms', 
        'threshold': 500,
        'severity': 'critical'
    },
    'high_error_rate': {
        'metric': 'error_rate',
        'threshold': 0.05,
        'severity': 'critical'
    },
    'high_fallback_rate': {
        'metric': 'fallback_rate',
        'threshold': 0.3,
        'severity': 'warning'
    }
}
```

#### Pipeline Alert Integration

`scripts/utils.py` provides helper function:

```python
def send_pipeline_alert(
    pipeline_name: str,
    status: str,
    message: str,
    severity: str = 'info'
):
    """
    Send alert for pipeline status changes.
    
    Integrates with AlertManager.
    """
    try:
        from alerting import AlertManager
        
        manager = AlertManager()
        manager.send_alert(
            alert_name=f"pipeline_{pipeline_name}_{status}",
            message=f"Pipeline: {pipeline_name}\nStatus: {status}\n{message}",
            severity=severity
        )
    except Exception as e:
        logger.warning(f"Failed to send pipeline alert: {e}")
```

#### Alert Log Output

**File: `logs/service/alerts.log`**

```
2025-01-15 15:35:00 | WARNING | alerting | high_latency triggered: avg_latency_ms=215.3 > 200
2025-01-15 15:35:01 | INFO | alerting | Email sent to team@example.com
2025-01-15 16:00:00 | CRITICAL | alerting | high_error_rate triggered: error_rate=0.08 > 0.05
2025-01-15 16:00:01 | INFO | alerting | Slack notification sent
2025-01-15 16:00:02 | INFO | alerting | Email sent to team@example.com, oncall@example.com
```
```

## Component 6: Model Deployment Monitoring (Implemented)

### Module: `automation/model_deployment.py`

Model deployment với hot-reload, rollback, và verification.

#### Main Function: `deploy_model()`

```python
def deploy_model(
    model_id: str = None,  # None = deploy current_best from registry
    service_url: str = 'http://localhost:8000',
    verify: bool = True,
    rollback_on_failure: bool = True
) -> Dict[str, Any]:
    """
    Deploy model to production service.
    
    Features:
    - Hot-reload via POST /reload_model
    - Deployment verification
    - Automatic rollback on failure
    - Deployment history tracking
    
    Returns:
        dict: Deployment result with status, verification results
    """
```

#### Deployment Workflow

```
deploy_model()
├─ Get model_id from registry (if not specified)
├─ Record previous model for rollback
├─ Call POST /reload_model with new model_id
├─ Wait for reload completion
├─ Verify deployment (if verify=True):
│  ├─ Check /health endpoint
│  ├─ Test recommendation for sample users
│  └─ Verify model_id matches
├─ On failure + rollback_on_failure:
│  ├─ Call POST /reload_model with previous model
│  └─ Verify rollback
├─ Record deployment in history
└─ Send alert on failure
```

#### Deployment Verification

```python
def verify_deployment(
    service_url: str,
    expected_model_id: str,
    test_users: List[int] = None
) -> Dict[str, Any]:
    """
    Verify model deployment is working correctly.
    
    Checks:
    1. Service health endpoint responds
    2. Model ID matches expected
    3. Recommendations work for test users
    """
    results = {
        'health_check': False,
        'model_id_match': False,
        'recommendations_work': False,
        'errors': []
    }
    
    # Check health
    try:
        resp = requests.get(f"{service_url}/health", timeout=5)
        results['health_check'] = resp.status_code == 200
        
        # Check model ID
        health_data = resp.json()
        results['model_id_match'] = health_data.get('model_id') == expected_model_id
    except Exception as e:
        results['errors'].append(f"Health check failed: {e}")
    
    # Test recommendations
    test_users = test_users or [12345, 67890, 11111]
    for user_id in test_users:
        try:
            resp = requests.post(
                f"{service_url}/recommend",
                json={'user_id': user_id, 'topk': 5},
                timeout=10
            )
            if resp.status_code == 200:
                results['recommendations_work'] = True
                break
        except Exception as e:
            results['errors'].append(f"Recommendation test failed: {e}")
    
    results['success'] = all([
        results['health_check'],
        results['model_id_match'],
        results['recommendations_work']
    ])
    
    return results
```

#### Rollback Function

```python
def rollback_deployment(
    service_url: str,
    previous_model_id: str
) -> bool:
    """
    Rollback to previous model version.
    
    Called automatically on deployment failure if rollback_on_failure=True.
    """
    logger.warning(f"Rolling back to {previous_model_id}")
    
    try:
        resp = requests.post(
            f"{service_url}/reload_model",
            json={'model_id': previous_model_id},
            timeout=30
        )
        
        if resp.status_code == 200:
            # Verify rollback
            verification = verify_deployment(service_url, previous_model_id)
            return verification['success']
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
    
    return False
```

#### Deployment History

**File: `logs/deployment_history.json`**

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
    },
    {
      "timestamp": "2025-01-16T10:00:00",
      "model_id": "bpr_20250116_100000",
      "previous_model_id": "als_20250115_100000",
      "status": "failed",
      "verification": {
        "health_check": true,
        "model_id_match": true,
        "recommendations_work": false
      },
      "rollback_triggered": true,
      "rollback_status": "success"
    }
  ]
}
```

#### Scheduled Deployment

Run daily via scheduler:

```python
# In automation/scheduler.py
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.model_deployment"]),
    CronTrigger(hour=5),  # 5:00 AM daily
    id="model_deployment",
    name="Daily Model Deployment Check"
)
```
```

## Component 7: Data Refresh Pipeline (Implemented)

### Module: `automation/data_refresh.py`

Data refresh với incremental updates và AI-powered quality scoring.

#### Main Function: `refresh_data()`

```python
def refresh_data(
    force_full: bool = False,
    staging_dir: str = 'data/staging'
) -> Dict[str, Any]:
    """
    Run data refresh pipeline.
    
    Features:
    - Incremental updates for small changes (≤100 new interactions)
    - Full pipeline for large changes
    - AI-powered comment quality scoring
    - Automatic data validation
    
    Returns:
        dict: Refresh results with statistics
    """
```

#### Data Refresh Workflow

```
refresh_data()
├─ Check PipelineLock("data_refresh")
├─ Check staging directory for new data
├─ Count new interactions
├─ Decision: Incremental vs Full
│  ├─ If new_count ≤ 100: Incremental update
│  │  ├─ Merge new data with existing
│  │  ├─ Re-score comments (AI)
│  │  └─ Update metadata
│  └─ If new_count > 100 or force_full: Full pipeline
│     ├─ Run complete Task 01 pipeline
│     ├─ Rebuild all matrices
│     └─ Update all artifacts
├─ Validate output data
├─ Track in PipelineTracker
├─ Archive staging data
└─ Send completion alert
```

#### Incremental Update

```python
def incremental_update(
    new_data: pd.DataFrame,
    existing_data_path: str = 'data/processed/interactions.parquet'
) -> Dict[str, Any]:
    """
    Perform incremental data update.
    
    For small data changes (≤100 interactions):
    - Merge new data with existing
    - Apply AI quality scoring to new comments
    - Update statistics and metadata
    """
    existing = pd.read_parquet(existing_data_path)
    
    # Score new comments using AI
    new_data['comment_quality'] = score_comments_with_ai(
        new_data['comment'].tolist()
    )
    
    # Merge and deduplicate
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=['user_id', 'product_id', 'cmt_date'],
        keep='last'
    )
    
    # Save
    combined.to_parquet(existing_data_path)
    
    return {
        'type': 'incremental',
        'new_records': len(new_data),
        'total_records': len(combined)
    }
```

#### AI Comment Quality Scoring

```python
def score_comments_with_ai(comments: List[str]) -> List[float]:
    """
    Score comment quality using AI model.
    
    Uses sentiment analysis and quality heuristics.
    Returns scores in [0, 1] range.
    """
    scores = []
    
    for comment in comments:
        if not comment or pd.isna(comment):
            scores.append(0.0)
            continue
        
        # Positive keywords (Vietnamese)
        positive_keywords = [
            'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 
            'sáng da', 'dưỡng ẩm', 'tốt', 'đẹp'
        ]
        
        # Negative keywords
        negative_keywords = [
            'không thích', 'dở', 'tệ', 'kém', 'giả'
        ]
        
        score = 0.5  # Neutral baseline
        
        for kw in positive_keywords:
            if kw in comment.lower():
                score += 0.1
        
        for kw in negative_keywords:
            if kw in comment.lower():
                score -= 0.1
        
        scores.append(max(0.0, min(1.0, score)))
    
    return scores
```

#### Scheduled Data Refresh

Run daily via scheduler:

```python
# In automation/scheduler.py
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.data_refresh"]),
    CronTrigger(hour=2),  # 2:00 AM daily
    id="data_refresh",
    name="Daily Data Refresh"
)
```

## Component 8: BERT Embeddings Monitoring

### Module: `automation/bert_embeddings.py`

Generate và monitor PhoBERT embeddings cho content-based recommendations.

#### Embedding Generation

```python
def generate_embeddings(
    products_path: str = 'data/published_data/data_product.csv',
    output_path: str = 'data/content_based_embeddings/product_embeddings.pt',
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Generate PhoBERT embeddings for all products.
    
    Returns:
        dict: Generation statistics
    """
```

#### Embedding Freshness Tracking

```python
def check_embedding_freshness(
    embeddings_path: str = 'data/content_based_embeddings/product_embeddings.pt'
) -> Dict[str, Any]:
    """
    Check embedding age và alert nếu stale.
    
    Returns:
        dict: {
            'age_days': int,
            'is_stale': bool,  # >30 days
            'num_embeddings': int,
            'embedding_dim': int
        }
    """
    metadata = torch.load(embeddings_path)
    created_at = datetime.fromisoformat(metadata.get('created_at', '2020-01-01'))
    age_days = (datetime.now() - created_at).days
    
    return {
        'age_days': age_days,
        'is_stale': age_days > 30,
        'num_embeddings': len(metadata.get('embeddings', [])),
        'embedding_dim': 768  # PhoBERT base
    }
```

#### Scheduled BERT Update

Run weekly via scheduler:

```python
# In automation/scheduler.py
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.bert_embeddings"]),
    CronTrigger(day_of_week='tue', hour=3),  # Tuesday 3:00 AM
    id="bert_embeddings",
    name="Weekly BERT Embeddings Update"
)
```
```

## Component 9: Scheduler Configuration (Implemented)

### Module: `automation/scheduler.py`

APScheduler-based job scheduling với cron triggers.

#### Scheduler Setup

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
import sys

scheduler = BlockingScheduler(timezone='Asia/Ho_Chi_Minh')
python = sys.executable
```

#### Scheduled Jobs

| Job ID | Schedule | Module | Description |
|--------|----------|--------|-------------|
| `data_refresh` | Daily 2:00 AM | `automation.data_refresh` | Refresh data pipeline |
| `bert_embeddings` | Tuesday 3:00 AM | `automation.bert_embeddings` | Update BERT embeddings |
| `drift_detection` | Monday 9:00 AM | `automation.drift_detection` | Weekly drift check |
| `model_training` | Sunday 3:00 AM | `automation.model_training` | Weekly model retraining |
| `model_deployment` | Daily 5:00 AM | `automation.model_deployment` | Deploy best model |
| `health_check` | Hourly :00 | `automation.health_check` | Health monitoring |

#### Job Configuration

```python
# Data Refresh - Daily at 2:00 AM
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.data_refresh"]),
    CronTrigger(hour=2, minute=0),
    id="data_refresh",
    name="Daily Data Refresh",
    replace_existing=True
)

# BERT Embeddings - Weekly on Tuesday at 3:00 AM
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.bert_embeddings"]),
    CronTrigger(day_of_week='tue', hour=3, minute=0),
    id="bert_embeddings",
    name="Weekly BERT Embeddings Update",
    replace_existing=True
)

# Drift Detection - Weekly on Monday at 9:00 AM
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.drift_detection"]),
    CronTrigger(day_of_week='mon', hour=9, minute=0),
    id="drift_detection",
    name="Weekly Drift Detection",
    replace_existing=True
)

# Model Training - Weekly on Sunday at 3:00 AM
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.model_training"]),
    CronTrigger(day_of_week='sun', hour=3, minute=0),
    id="model_training",
    name="Weekly Model Training",
    replace_existing=True
)

# Model Deployment - Daily at 5:00 AM
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.model_deployment"]),
    CronTrigger(hour=5, minute=0),
    id="model_deployment",
    name="Daily Model Deployment Check",
    replace_existing=True
)

# Health Check - Every hour at :00
scheduler.add_job(
    lambda: subprocess.run([python, "-m", "automation.health_check"]),
    CronTrigger(minute=0),
    id="health_check",
    name="Hourly Health Check",
    replace_existing=True
)
```

#### Running the Scheduler

```powershell
# Start scheduler
python -m automation.scheduler

# Or via manage script
.\manage_scheduler.ps1 -Action start
```

#### Scheduler Logging

**File: `logs/scheduler/scheduler.log`**

```
2025-01-15 02:00:00 | INFO | Starting job: data_refresh
2025-01-15 02:01:30 | INFO | Job completed: data_refresh | duration=90s | status=success
2025-01-15 03:00:00 | INFO | Starting job: health_check
2025-01-15 03:00:05 | INFO | Job completed: health_check | duration=5s | status=success
2025-01-15 05:00:00 | INFO | Starting job: model_deployment
2025-01-15 05:00:45 | INFO | Job completed: model_deployment | duration=45s | status=success
```

#### Configuration File: `config/scheduler_config.json`

```json
{
  "timezone": "Asia/Ho_Chi_Minh",
  "jobs": {
    "data_refresh": {
      "enabled": true,
      "cron": "0 2 * * *",
      "timeout_minutes": 30
    },
    "bert_embeddings": {
      "enabled": true,
      "cron": "0 3 * * 2",
      "timeout_minutes": 60
    },
    "drift_detection": {
      "enabled": true,
      "cron": "0 9 * * 1",
      "timeout_minutes": 15
    },
    "model_training": {
      "enabled": true,
      "cron": "0 3 * * 0",
      "timeout_minutes": 120
    },
    "model_deployment": {
      "enabled": true,
      "cron": "0 5 * * *",
      "timeout_minutes": 10
    },
    "health_check": {
      "enabled": true,
      "cron": "0 * * * *",
      "timeout_minutes": 5
    }
  }
}
```

## Dependencies

```python
# requirements_monitoring.txt
apscheduler>=3.10.0  # Job scheduling
requests>=2.28.0  # HTTP calls, Slack webhooks
pyyaml>=6.0  # Config files

# Database
sqlite3  # Built-in

# Alerting
smtplib  # Built-in (email)

# BERT monitoring
torch>=1.13.0
transformers>=4.30.0

# Optional: Dashboard
streamlit>=1.20.0
plotly>=5.13.0
```

## Timeline Estimate

- **Pipeline tracking**: ✅ Implemented
- **Training monitoring**: ✅ Implemented
- **Health check system**: ✅ Implemented
- **Drift detection**: ✅ Implemented
- **Alerting system**: ✅ Implemented
- **Model deployment monitoring**: ✅ Implemented
- **Data refresh pipeline**: ✅ Implemented
- **BERT embeddings monitoring**: ✅ Implemented
- **Scheduler**: ✅ Implemented

**Remaining work**:
- Dashboard visualization: 2 days
- Integration testing: 1 day
- Documentation: 1 day
- **Total remaining**: ~4 days

## Component 10: Registry Integration Summary

### Key Integrations

1. **Training → Registry**: Auto-register models sau training completion
2. **Registry → Service**: Model hot-reload via `/reload_model` endpoint
3. **Registry Health**: Continuous monitoring trong `check_model_health()`
4. **Model Changes**: Alert khi deployment fails hoặc rollback triggers
5. **Audit Trail**: All operations logged to `logs/registry_audit.log`

### Command Reference

```powershell
# Run individual modules
python -m automation.data_refresh
python -m automation.drift_detection
python -m automation.model_training
python -m automation.model_deployment
python -m automation.health_check
python -m automation.bert_embeddings

# Start scheduler (background)
python -m automation.scheduler

# Check pipeline stats
python -c "from scripts.utils import PipelineTracker; print(PipelineTracker().get_stats())"
```

## Success Criteria

- [x] Pipeline runs tracked với `PipelineTracker` class
- [x] Training runs logged và auto-registered
- [x] Health checks run hourly
- [x] Drift detection runs weekly
- [x] Alerting system với Email, Slack, logging
- [x] Model deployment với verification và rollback
- [x] Data refresh với incremental updates
- [x] BERT embedding freshness tracked
- [x] Scheduler orchestrates all jobs
- [ ] Dashboard visualizes all metrics (TODO)
- [x] Logs retained với proper rotation
- [x] File-based locking prevents concurrent runs

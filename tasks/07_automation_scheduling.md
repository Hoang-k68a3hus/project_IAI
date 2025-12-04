# Task 07: Automation & Scheduling

## Mục Tiêu

Xây dựng hệ thống tự động hóa cho toàn bộ ML pipeline: data refresh, model training, evaluation, deployment, và monitoring. Hệ thống phải chạy ổn định, có error handling, và dễ dàng maintain.

## Trạng Thái: ✅ IMPLEMENTED

Hệ thống automation đã được implement đầy đủ với các module sau:
- `automation/scheduler.py` - APScheduler orchestration
- `automation/data_refresh.py` - Data refresh pipeline với incremental update
- `automation/model_training.py` - ALS/BPR training với BERT init, warm-start, early stopping
- `automation/model_deployment.py` - Model deployment với rollback
- `automation/health_check.py` - Health check cho service, data, models, pipelines
- `automation/drift_detection.py` - Drift detection pipeline
- `automation/bert_embeddings.py` - BERT embeddings refresh
- `automation/cleanup.py` - Log và artifact cleanup
- `scripts/utils.py` - PipelineTracker, PipelineLock, retry decorator

## Automation Architecture

```
APScheduler (BlockingScheduler)
    ↓
├─ Daily 2:00 AM: Data Refresh
│   - Merge staging data (from web ingestion)
│   - Incremental update (if < 100 new interactions)
│   - Full pipeline (if many changes or new users/items)
│   - AI-powered comment quality scoring
│   - Track newly trainable users
│
├─ Weekly Tuesday 3:00 AM: BERT Embeddings Refresh
│   - Load products from data_product.csv
│   - Generate PhoBERT embeddings (vinai/phobert-base)
│   - Save to product_embeddings.pt
│   - Track freshness for model training
│
├─ Weekly Monday 9:00 AM: Drift Detection
│   - Rating distribution drift
│   - Popularity shift (Jaccard similarity)
│   - Interaction rate drift
│   - Generate drift report
│   - Alert if significant drift
│
├─ Weekly Sunday 3:00 AM: Model Training
│   - Train ALS with confidence-weighted matrix
│   - Train BPR with early stopping
│   - BERT initialization for cold items
│   - Warm-start from previous model (optional)
│   - Popularity baseline comparison
│   - Auto-register best model
│
├─ Daily 5:00 AM: Model Deployment
│   - Check registry for updates
│   - Trigger service reload
│   - Verify deployment
│   - Record deployment history
│   - Support rollback
│
├─ Hourly :00: Health Checks
│   - Service health (API reachable, model loaded)
│   - Data health (files exist, freshness, integrity)
│   - Model health (registry, performance metrics)
│   - Pipeline health (success rate, stale runs)
│   - Send alerts for failures
│
└─ Monthly: Cleanup
    - Old logs (30 days)
    - Old checkpoints (7 days)
    - Old model versions (keep 5 per type)
    - Vacuum SQLite databases
    - Remove empty directories
```

## Component 1: Scheduler & Orchestration

### Main Scheduler

#### File: `automation/scheduler.py`
```python
"""
VieComRec Automation Scheduler.

This module orchestrates scheduled tasks for the recommendation system.
All tasks are modules within the automation package.

Usage:
    python -m automation.scheduler
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# Configuration
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).parent.parent))
LOG_DIR = PROJECT_DIR / "logs" / "scheduler"
SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")
TIMEZONE = pytz.timezone(os.environ.get("TZ", "UTC"))

# Scheduler config (also in config/scheduler_config.json)
SCHEDULER_CONFIG = {
    "data_refresh": {
        "enabled": True,
        "description": "Daily data refresh from raw CSV files",
        "schedule": {"hour": 2, "minute": 0},  # 2:00 AM daily
        "module": "automation.data_refresh",
        "args": [],
    },
    "bert_embeddings": {
        "enabled": True,
        "description": "Weekly BERT embeddings refresh",
        "schedule": {"day_of_week": "tue", "hour": 3, "minute": 0},  # Tuesday 3:00 AM
        "module": "automation.bert_embeddings",
        "args": [],
    },
    "drift_detection": {
        "enabled": True,
        "description": "Weekly drift detection monitoring",
        "schedule": {"day_of_week": "mon", "hour": 9, "minute": 0},  # Monday 9:00 AM
        "module": "automation.drift_detection",
        "args": [],
    },
    "model_training": {
        "enabled": True,
        "description": "Weekly model training (ALS + BPR)",
        "schedule": {"day_of_week": "sun", "hour": 3, "minute": 0},  # Sunday 3:00 AM
        "module": "automation.model_training",
        "args": ["--auto-select"],
    },
    "model_deployment": {
        "enabled": True,
        "description": "Daily model deployment check",
        "schedule": {"hour": 5, "minute": 0},  # 5:00 AM daily
        "module": "automation.model_deployment",
        "args": [],
    },
    "health_check": {
        "enabled": True,
        "description": "Hourly health check",
        "schedule": {"minute": 0},  # Every hour at :00
        "module": "automation.health_check",
        "args": [],
    },
}


def update_task_status(task_name: str, status: str, exit_code: Optional[int] = None, 
                       log_file: Optional[str] = None, error: Optional[str] = None) -> None:
    """Update task status in JSON file for monitoring."""
    status_file = LOG_DIR / "task_status.json"
    
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            all_status = json.load(f)
    else:
        all_status = {}
    
    all_status[task_name] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "exit_code": exit_code,
        "log_file": log_file,
        "error": error,
    }
    
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(all_status, f, indent=2, ensure_ascii=False)


def run_task(task_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a scheduled task via subprocess."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{task_name}_{timestamp}.log"
    
    logger.info("=" * 60)
    logger.info(f"TASK: {config['description']}")
    logger.info("=" * 60)
    
    result = {"task": task_name, "status": "running", "timestamp": datetime.now().isoformat()}
    
    try:
        cmd = [sys.executable, "-m", config["module"]] + config.get("args", [])
        
        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.run(
                cmd, cwd=str(PROJECT_DIR), stdout=f, stderr=subprocess.STDOUT,
                timeout=3600, env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
            )
        
        result["exit_code"] = process.returncode
        result["log_file"] = str(log_file)
        result["status"] = "success" if process.returncode == 0 else "failed"
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Task timed out after 1 hour"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    update_task_status(task_name, result["status"], result.get("exit_code"), 
                       result.get("log_file"), result.get("error"))
    return result


def create_scheduler() -> BlockingScheduler:
    """Create and configure the scheduler with all jobs."""
    scheduler = BlockingScheduler(timezone=TIMEZONE)
    
    for task_name, config in SCHEDULER_CONFIG.items():
        if not config.get("enabled", True):
            logger.info(f"Skipping disabled task: {task_name}")
            continue
        
        trigger = CronTrigger(**config["schedule"], timezone=TIMEZONE)
        scheduler.add_job(
            lambda tn=task_name, cfg=config: run_task(tn, cfg),
            trigger=trigger, id=task_name, name=config.get("description", task_name),
            replace_existing=True,
        )
        logger.info(f"✓ Registered job: {task_name}")
    
    return scheduler


def main() -> None:
    """Main entry point for the scheduler."""
    logger.info("=" * 80)
    logger.info("VIECOMREC AUTOMATION SCHEDULER STARTING")
    logger.info(f"Project Directory: {PROJECT_DIR}")
    logger.info(f"Service URL: {SERVICE_URL}")
    logger.info("=" * 80)
    
    scheduler = create_scheduler()
    
    # Print scheduled jobs
    for job in scheduler.get_jobs():
        logger.info(f"  [{job.id}] - Trigger: {job.trigger}")
    
    try:
        logger.info("✓ Scheduler started successfully")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
```

## Component 2: Data Refresh Pipeline

### File: `automation/data_refresh.py`
```python
"""
Data Refresh Pipeline with Incremental Update Support.

Usage:
    python -m automation.data_refresh [--force] [--force-full] [--dry-run] [--skip-merge]

Key Features:
- Merge staging data (from web ingestion)
- Incremental update for small changes (< 100 new interactions)
- Full pipeline for larger changes or new users/items
- AI-powered comment quality scoring via FeatureEngineer
- Tracks newly trainable users (≥2 interactions)
"""

# Configuration
DATA_CONFIG = {
    "raw_data_dir": PROJECT_ROOT / "data" / "published_data",
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "staging_dir": PROJECT_ROOT / "data" / "staging",
    "raw_files": ["data_reviews_purchase.csv", "data_product.csv", "data_product_attribute.csv"],
    "output_files": [
        "interactions.parquet", "all_quality_scores_cache.parquet",
        "X_train_confidence.npz", "X_train_binary.npz", "user_item_mappings.json",
        "user_metadata.pkl", "user_pos_train.pkl", "user_hard_neg_train.pkl", "data_stats.json",
    ],
}

INCREMENTAL_THRESHOLD = 100  # Use incremental if <= 100 new interactions


def merge_staging_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Merge staging data (from web ingestion) into raw data.
    
    Process:
    1. Read new interactions from staging/new_interactions.csv
    2. Backup existing raw data
    3. Merge and deduplicate
    4. Archive staging file
    
    Returns new_interactions_df for incremental processing.
    """
    staging_file = DATA_CONFIG["staging_dir"] / "new_interactions.csv"
    raw_file = DATA_CONFIG["raw_data_dir"] / "data_reviews_purchase.csv"
    
    if not staging_file.exists():
        return {"merged": False, "new_interactions_df": None}
    
    staging_df = pd.read_csv(staging_file, encoding='utf-8')
    raw_df = pd.read_csv(raw_file, encoding='utf-8')
    
    # Map columns and merge
    staging_mapped = pd.DataFrame({
        'user_id': staging_df['user_id'],
        'product_id': staging_df['product_id'],
        'rating': staging_df['rating'].astype(float),
        'comment': staging_df['comment'].fillna(''),
        'cmt_date': pd.to_datetime(staging_df['timestamp'], format='ISO8601', utc=True)
                     .dt.tz_convert(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Backup and merge
    backup_file = raw_dir / f"data_reviews_purchase_backup_{timestamp}.csv"
    shutil.copy(raw_file, backup_file)
    
    merged_df = pd.concat([raw_df, staging_mapped], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['user_id', 'product_id', 'cmt_date'], keep='last')
    merged_df.to_csv(raw_file, index=False, encoding='utf-8')
    
    # Archive staging
    shutil.move(staging_file, staging_dir / "archived" / f"interactions_{timestamp}.csv")
    
    return {"merged": True, "new_interactions_df": staging_mapped, "new_interactions_count": len(staging_mapped)}


def run_incremental_update(new_interactions: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run incremental update - only process new interactions and update matrices.
    
    Features:
    - AI-powered comment quality scoring (ViSoBERT via FeatureEngineer)
    - Support for users becoming trainable (≥2 interactions)
    - Proper confidence score calculation matching full pipeline
    - Appends to existing parquet and matrices
    
    Falls back to full pipeline if:
    - Too many new users/items
    - Matrix dimension mismatch
    """
    # Load existing data
    existing_df = pd.read_parquet(output_dir / "interactions.parquet")
    X_train_conf = sp.load_npz(output_dir / "X_train_confidence.npz")
    
    # Initialize FeatureEngineer for AI-powered scoring
    feature_engineer = FeatureEngineer(
        positive_threshold=4.0, hard_negative_threshold=3.0,
        use_ai_sentiment=True, batch_size=32
    )
    
    # Process each interaction
    for _, row in new_interactions.iterrows():
        # Compute comment quality using AI
        comment_quality = feature_engineer.compute_comment_quality_score(row['comment'])
        confidence = row['rating'] + comment_quality
        
        # Update matrices (LIL format for efficiency)
        X_conf_lil[u_idx, i_idx] = confidence
        if is_positive:
            X_bin_lil[u_idx, i_idx] = 1.0
            user_pos_train[u_idx].add(i_idx)
    
    # Save updated matrices
    sp.save_npz(output_dir / "X_train_confidence.npz", X_train_conf)
    sp.save_npz(output_dir / "X_train_binary.npz", X_train_bin)
    
    return {"success": True, "mode": "incremental", "updates_applied": len(updates)}


@retry(max_attempts=3, backoff_factor=2.0)
def run_data_pipeline(logger: logging.Logger, force_full: bool = False) -> Dict[str, Any]:
    """
    Run the full Task 01 data processing pipeline.
    
    Steps:
    1. Load and validate interactions (with cached quality scores optimization)
    2. Create explicit features (is_positive, is_hard_negative)
    3. User segmentation (trainable vs cold-start)
    4. Create ID mappings
    5. Temporal split (leave-one-out)
    6. Create LOCAL indices for trainable users
    7. Build matrices with LOCAL indices
    8. Save all artifacts
    """
    processor = DataProcessor(base_path=str(raw_dir))
    
    # Load cached comment_quality scores (avoid recomputing AI sentiment)
    cache_file_path = output_dir / "all_quality_scores_cache.parquet"
    cached_scores = None
    if cache_file_path.exists():
        cached_df = pd.read_parquet(cache_file_path)
        cached_scores = cached_df.set_index('_key')['comment_quality'].to_dict()
    
    # Step 1: Load and validate
    df_clean, validation_stats = processor.load_and_validate_interactions(
        apply_deduplication=True, detect_outliers=True,
        compute_quality_scores=True, cached_quality_scores=cached_scores
    )
    
    # Save ALL quality scores cache (before user filtering)
    cache_df = df_clean[['user_id', 'product_id', 'comment_quality']].copy()
    cache_df.to_parquet(cache_file_path, index=False)
    
    # Steps 2-8: Feature engineering, filtering, mapping, splitting, building...
    # ... (see full implementation in automation/data_refresh.py)
    
    return {"success": True, "mode": "full", "stats": stats}


def refresh_data(force=False, force_full=False, dry_run=False, skip_merge=False):
    """
    Main data refresh function with dual-mode support.
    
    Modes:
    - INCREMENTAL: Fast update when few new interactions (< 100)
    - FULL: Complete reprocessing when many changes or force_full=True
    
    Also triggers full pipeline if pending_trainable_users > 50.
    """
    with PipelineLock("data_refresh") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already running"}
        
        run_id = tracker.start_run("data_refresh", {"force": force, "force_full": force_full})
        
        # Step 0: Merge staging data
        merge_result = merge_staging_data(logger)
        new_interactions_df = merge_result.get("new_interactions_df")
        
        # Decide incremental vs full
        use_incremental = (
            not force_full
            and new_interactions_df is not None
            and len(new_interactions_df) <= INCREMENTAL_THRESHOLD
        )
        
        if use_incremental:
            result = run_incremental_update(new_interactions_df, logger)
            if result is None:  # Fallback to full
                result = run_data_pipeline(logger)
        else:
            result = run_data_pipeline(logger)
        
        tracker.complete_run(run_id, {"status": "success", "mode": result["mode"]})
        send_pipeline_alert("data_refresh", "success", f"Mode: {result['mode']}")
        
        return result
```

## Component 3: Model Training Pipeline

### File: `automation/model_training.py`
```python
"""
Model Training Pipeline with Advanced Features.

Usage:
    python -m automation.model_training [--model als|bpr|both] [--auto-select] 
                                         [--skip-eval] [--force] [--warmstart]

Key Features:
- ALS training with confidence-weighted matrix factorization
- BPR training with hard negative sampling
- BERT initialization for cold-start items (optional)
- Checkpointing for crash recovery
- Popularity baseline comparison
- Incremental retraining (warm start from previous model)
- Early stopping for BPR when validation metric plateaus
"""

TRAINING_CONFIG = {
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    "checkpoints_dir": PROJECT_ROOT / "checkpoints",
    "registry_path": PROJECT_ROOT / "artifacts" / "cf" / "registry.json",
    "bert_embeddings_path": PROJECT_ROOT / "data" / "processed" / "content_based_embeddings" / "product_embeddings.pt",
    
    # ALS hyperparameters
    "als": {
        "factors": 64,
        "regularization": 0.1,  # Higher due to sparsity
        "iterations": 15,
        "alpha": 5,
        "use_gpu": False,
        "use_bert_init": False,
        "bert_init_cold_threshold": 5,  # Items with < 5 interactions
    },
    # BPR hyperparameters
    "bpr": {
        "factors": 64,
        "learning_rate": 0.05,
        "regularization": 0.0001,
        "epochs": 50,
        "neg_sample_ratio": 0.3,  # 30% hard negatives
    },
    # Evaluation
    "eval_k_values": [5, 10, 20],
    "primary_metric": "recall",
    "primary_k": 10,
    
    # Early stopping
    "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.001},
    
    # Checkpointing
    "checkpoint_every_n_iters": 5,
    "keep_n_checkpoints": 3,
    
    # Incremental training
    "incremental": {"enabled": True, "warmstart": True, "warmstart_iters": 5},
}


@retry(max_attempts=2, backoff_factor=2.0)
def train_als_model(data: Dict, logger, warmstart=False, use_bert_init=True):
    """
    Train ALS model using implicit library.
    
    Features:
    - Warm-start from previous model
    - BERT initialization for cold-start items (project 768-dim to 64-dim via SVD)
    - Checkpointing during training
    """
    from implicit.als import AlternatingLeastSquares
    
    config = TRAINING_CONFIG["als"]
    iterations = config["warmstart_iters"] if warmstart else config["iterations"]
    
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        iterations=iterations,
        alpha=config["alpha"],
        use_gpu=config["use_gpu"],
        random_state=42,
    )
    
    X_train = data["X_confidence"]  # (users, items) matrix
    
    # === WARM-START ===
    if warmstart:
        prev_model = load_previous_model("als", logger)
        if prev_model:
            model.user_factors = prev_model["U"].copy()
            model.item_factors = prev_model["V"].copy()
            logger.info(f"Warm-start from: {prev_model['model_id']}")
    
    # === BERT INIT for cold items ===
    if use_bert_init:
        cold_mask = get_cold_items(X_train, threshold=config["bert_init_cold_threshold"])
        bert_embeddings = load_bert_embeddings(logger)
        if bert_embeddings is not None and cold_mask.sum() > 0:
            projected = project_bert_to_factors(bert_embeddings, config["factors"], logger)
            model.item_factors[cold_mask] = projected[cold_mask]
            logger.info(f"BERT-initialized {cold_mask.sum()} cold items")
    
    # Train
    start_time = datetime.now()
    model.fit(X_train, show_progress=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save checkpoint
    save_checkpoint("als", iterations, model.user_factors, model.item_factors, logger=logger)
    
    return {
        "model": model, "U": model.user_factors, "V": model.item_factors,
        "model_type": "als", "training_time": training_time,
        "warmstart": warmstart, "bert_initialized": cold_mask.sum() if use_bert_init else 0,
    }


@retry(max_attempts=2, backoff_factor=2.0)
def train_bpr_model(data: Dict, logger, warmstart=False):
    """
    Train BPR model with early stopping.
    
    Features:
    - Early stopping based on validation Recall@10
    - Warm-start from previous model
    - Hard negative sampling (30% hard + 70% random)
    """
    from implicit.bpr import BayesianPersonalizedRanking
    
    config = TRAINING_CONFIG["bpr"]
    early_stop_cfg = TRAINING_CONFIG["early_stopping"]
    
    model = BayesianPersonalizedRanking(
        factors=config["factors"],
        learning_rate=config["learning_rate"],
        regularization=config["regularization"],
        iterations=config["epochs"],
        random_state=42,
    )
    
    X_train = data["X_binary"]
    
    # Validation split for early stopping
    if early_stop_cfg["enabled"] and not warmstart:
        train_sets, val_sets = create_validation_split(data["user_pos_train"], val_ratio=0.1)
        
        best_val_metric = 0.0
        patience_counter = 0
        model.iterations = 1  # Train epoch-by-epoch
        
        for epoch in range(config["epochs"]):
            model.fit(X_train, show_progress=False)
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                val_recall = compute_validation_recall(model, train_sets, val_sets)
                logger.info(f"Epoch {epoch + 1} - Val Recall@10: {val_recall:.4f}")
                
                if val_recall > best_val_metric + early_stop_cfg["min_delta"]:
                    best_val_metric = val_recall
                    best_factors = (model.user_factors.copy(), model.item_factors.copy())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_cfg["patience"]:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    model.user_factors, model.item_factors = best_factors
                    break
    else:
        model.fit(X_train, show_progress=True)
    
    return {"model": model, "U": model.user_factors, "V": model.item_factors, "model_type": "bpr"}


def compute_popularity_baseline(data: Dict, logger) -> Dict[str, Any]:
    """Compute popularity baseline for comparison."""
    X_train = data["X_binary"]
    item_popularity = np.array(X_train.sum(axis=0)).flatten()
    top_popular_items = np.argsort(item_popularity)[-20:][::-1]
    
    # Evaluate against test set
    metrics = evaluate_predictions(top_popular_items, data["user_pos_test"])
    logger.info(f"Popularity Baseline - Recall@10: {metrics['recall@10']:.4f}")
    
    return metrics


def train_models(model_types=None, auto_select=True, warmstart=False, logger=None):
    """Main training pipeline."""
    if model_types is None:
        model_types = ["als", "bpr"]
    
    with PipelineLock("model_training") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already running"}
        
        run_id = tracker.start_run("model_training", {"models": model_types, "warmstart": warmstart})
        
        data = load_training_data(logger)
        
        # Popularity baseline
        baseline_metrics = compute_popularity_baseline(data, logger)
        
        trained_models = []
        for model_type in model_types:
            if model_type == "als":
                result = train_als_model(data, logger, warmstart=warmstart, use_bert_init=True)
            else:
                result = train_bpr_model(data, logger, warmstart=warmstart)
            
            metrics = evaluate_model(result, data, logger)
            
            # Compare with baseline
            improvement = (metrics["recall@10"] - baseline_metrics["recall@10"]) / baseline_metrics["recall@10"] * 100
            metrics["baseline_improvement_pct"] = improvement
            
            save_result = save_model(result, metrics, data, logger)
            trained_models.append({**save_result, "metrics": metrics})
        
        # Auto-select best
        if auto_select:
            best = max(trained_models, key=lambda m: m["metrics"]["recall@10"])
            register_model(best["model_id"], best["model_type"], best["metrics"], Path(best["output_dir"]), logger)
        
        tracker.complete_run(run_id, {"status": "success", "selected_model": best["model_id"]})
        return {"status": "success", "models": trained_models, "selected_model": best["model_id"]}
```

## Component 4: Model Deployment Pipeline

### File: `automation/model_deployment.py`
```python
"""
Model Deployment Pipeline with Rollback Support.

Usage:
    python -m automation.model_deployment [--model-id MODEL_ID] [--rollback] [--dry-run]

Key Features:
- Deploy current_best from registry
- Rollback to previous model
- Verify deployment via /model_info endpoint
- Record deployment history
- Handle offline service gracefully
"""

DEPLOY_CONFIG = {
    "registry_path": PROJECT_ROOT / "artifacts" / "cf" / "registry.json",
    "service_url": os.environ.get("SERVICE_URL", "http://localhost:8000"),
    "health_check_timeout": 30,
    "reload_timeout": 60,
    "deployment_history_path": PROJECT_ROOT / "logs" / "deployment_history.json",
}


@retry(max_attempts=3, backoff_factor=2.0)
def check_service_health(logger) -> Dict[str, Any]:
    """Check if the recommendation service is healthy."""
    url = f"{DEPLOY_CONFIG['service_url']}/health"
    try:
        response = requests.get(url, timeout=DEPLOY_CONFIG["health_check_timeout"])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Connection refused"}


@retry(max_attempts=3, backoff_factor=2.0)
def trigger_model_reload(model_id: Optional[str], logger) -> Dict[str, Any]:
    """Trigger model reload on the service."""
    url = f"{DEPLOY_CONFIG['service_url']}/reload_model"
    payload = {"model_id": model_id} if model_id else {}
    response = requests.post(url, json=payload, timeout=DEPLOY_CONFIG["reload_timeout"])
    response.raise_for_status()
    return response.json()


def verify_deployment(expected_model_id: str, logger) -> bool:
    """Verify the correct model is loaded via /model_info."""
    url = f"{DEPLOY_CONFIG['service_url']}/model_info"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    loaded_model = response.json().get("model_id")
    return loaded_model == expected_model_id


def get_previous_model(registry: Dict) -> Optional[str]:
    """Get the previous active model for rollback."""
    current_best = registry.get("current_best")
    if isinstance(current_best, dict):
        current_best = current_best.get("model_id")
    
    models = registry.get("models", {})
    sorted_models = sorted(models.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)
    
    for model_id, _ in sorted_models:
        if model_id != current_best:
            return model_id
    return None


def record_deployment(model_id: str, status: str, metadata: Optional[Dict] = None):
    """Record deployment in history file (keep last 100)."""
    history_path = DEPLOY_CONFIG["deployment_history_path"]
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {"deployments": []}
    
    history["deployments"].append({
        "model_id": model_id,
        "deployed_at": datetime.now().isoformat(),
        "status": status,
        "git_commit": get_git_commit(),
        "metadata": metadata or {},
    })
    
    history["deployments"] = history["deployments"][-100:]
    
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def deploy_model(model_id=None, rollback=False, dry_run=False, logger=None):
    """
    Main deployment function.
    
    Process:
    1. Load registry and determine model to deploy
    2. Check service health
    3. Trigger reload (or just update registry if offline)
    4. Verify deployment
    5. Record history
    """
    with PipelineLock("model_deployment") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already in progress"}
        
        run_id = tracker.start_run("model_deployment", {"model_id": model_id, "rollback": rollback})
        
        registry = load_registry()
        
        # Determine model to deploy
        if rollback:
            model_id = get_previous_model(registry)
            if not model_id:
                raise ValueError("No previous model available for rollback")
        elif not model_id:
            current_best = registry.get("current_best")
            model_id = current_best["model_id"] if isinstance(current_best, dict) else current_best
        
        # Check service health
        health = check_service_health(logger)
        
        if health.get("status") == "offline":
            # Just update registry, deployment on next startup
            update_registry_active_status(model_id, logger)
            record_deployment(model_id, "pending_restart")
            return {"status": "pending", "message": "Service offline, will deploy on startup"}
        
        if dry_run:
            return {"status": "dry_run", "message": f"Would deploy {model_id}"}
        
        # Trigger reload
        reload_result = trigger_model_reload(model_id, logger)
        
        # Verify
        if not verify_deployment(model_id, logger):
            raise RuntimeError("Deployment verification failed")
        
        # Update registry and record
        update_registry_active_status(model_id, logger)
        record_deployment(model_id, "success", {"reload_response": reload_result})
        
        tracker.complete_run(run_id, {"status": "success", "model_id": model_id})
        send_pipeline_alert("model_deployment", "success", f"Deployed: {model_id}")
        
        return {"status": "success", "model_id": model_id}
```

## Component 5: BERT Embeddings Refresh

### File: `automation/bert_embeddings.py`
```python
"""
BERT Embeddings Refresh Pipeline.

Usage:
    python -m automation.bert_embeddings [--force]

Key Features:
- Generate PhoBERT embeddings for all products
- Skip if embeddings are fresh (< 7 days old)
- Save as PyTorch tensor format
"""

BERT_CONFIG = {
    "model_name": "vinai/phobert-base",
    "product_file": PROJECT_ROOT / "data" / "published_data" / "data_product.csv",
    "output_dir": PROJECT_ROOT / "data" / "processed" / "content_based_embeddings",
    "output_file": "product_embeddings.pt",
    "batch_size": 32,
    "max_length": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_product_texts(logger) -> Dict[str, str]:
    """Load and combine product text fields."""
    df = pd.read_csv(BERT_CONFIG["product_file"], encoding="utf-8")
    
    texts = {}
    for _, row in df.iterrows():
        product_id = str(row.get("product_id", row.name))
        
        # Combine: name + description + features with [SEP]
        text_parts = []
        if pd.notna(row.get("product_name")):
            text_parts.append(str(row["product_name"]))
        if pd.notna(row.get("processed_description")):
            text_parts.append(str(row["processed_description"]))
        if pd.notna(row.get("feature")):
            text_parts.append(str(row["feature"]))
        
        texts[product_id] = " [SEP] ".join(text_parts)
    
    return texts


def generate_embeddings(texts: Dict[str, str], logger) -> Dict[str, torch.Tensor]:
    """Generate PhoBERT embeddings using CLS token."""
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
    model = AutoModel.from_pretrained(BERT_CONFIG["model_name"])
    model = model.to(BERT_CONFIG["device"])
    model.eval()
    
    product_ids = list(texts.keys())
    product_texts = [texts[pid] for pid in product_ids]
    
    embeddings = {}
    with torch.no_grad():
        for i in range(0, len(product_texts), BERT_CONFIG["batch_size"]):
            batch_texts = product_texts[i:i + BERT_CONFIG["batch_size"]]
            batch_ids = product_ids[i:i + BERT_CONFIG["batch_size"]]
            
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=BERT_CONFIG["max_length"], return_tensors="pt"
            )
            inputs = {k: v.to(BERT_CONFIG["device"]) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # CLS token
            
            for pid, emb in zip(batch_ids, batch_embeddings):
                embeddings[pid] = emb
    
    return embeddings


def save_embeddings(embeddings: Dict[str, torch.Tensor], logger) -> Path:
    """Save embeddings to PyTorch file."""
    output_dir = BERT_CONFIG["output_dir"]
    output_file = output_dir / BERT_CONFIG["output_file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    product_ids = list(embeddings.keys())
    embedding_tensor = torch.stack([embeddings[pid] for pid in product_ids])
    
    torch.save({
        "product_ids": product_ids,
        "embeddings": embedding_tensor,
        "model": BERT_CONFIG["model_name"],
        "created_at": datetime.now().isoformat(),
        "shape": embedding_tensor.shape,
    }, output_file)
    
    return output_file


def refresh_bert_embeddings(force=False, logger=None):
    """Main BERT embeddings refresh pipeline."""
    with PipelineLock("bert_embeddings") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already running"}
        
        run_id = tracker.start_run("bert_embeddings", {"force": force})
        
        # Check freshness (skip if < 7 days old)
        output_file = BERT_CONFIG["output_dir"] / BERT_CONFIG["output_file"]
        if output_file.exists() and not force:
            mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            if age_days < 7:
                return {"status": "skipped", "message": f"Fresh ({age_days} days old)"}
        
        # Generate embeddings
        texts = load_product_texts(logger)
        embeddings = generate_embeddings(texts, logger)
        output_path = save_embeddings(embeddings, logger)
        
        tracker.complete_run(run_id, {"status": "success", "num_products": len(embeddings)})
        send_pipeline_alert("bert_embeddings", "success", f"Generated {len(embeddings)} embeddings")
        
        return {"status": "success", "num_products": len(embeddings), "output_file": str(output_path)}
```

## Component 6: Health Check System

### File: `automation/health_check.py`
```python
"""
System Health Check Pipeline.

Usage:
    python -m automation.health_check [--component all|service|data|models|pipelines] 
                                       [--json] [--alert]

Key Features:
- Service health (API reachable, model loaded)
- Data health (files exist, freshness, integrity)
- Model health (registry valid, performance metrics)
- Pipeline health (success rate, stale runs)
- Alert on failures
"""

HEALTH_CONFIG = {
    "service_url": os.environ.get("SERVICE_URL", "http://localhost:8000"),
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    "registry_path": PROJECT_ROOT / "artifacts" / "cf" / "registry.json",
    "embeddings_path": PROJECT_ROOT / "data" / "processed" / "content_based_embeddings" / "product_embeddings.pt",
    # Thresholds
    "max_data_age_days": 7,
    "max_model_age_days": 30,
    "min_recall_threshold": 0.10,
    "service_timeout": 10,
}


def check_service_health(logger) -> Dict[str, Any]:
    """Check recommendation service health."""
    result = {"component": "service", "status": "unknown", "checks": []}
    
    url = f"{HEALTH_CONFIG['service_url']}/health"
    
    try:
        response = requests.get(url, timeout=HEALTH_CONFIG["service_timeout"])
        response.raise_for_status()
        health_data = response.json()
        
        result["status"] = "healthy"
        result["checks"].append({"name": "api_reachable", "passed": True})
        
        # Check model loaded
        model_loaded = health_data.get("model_loaded") or bool(health_data.get("model_id"))
        result["checks"].append({
            "name": "model_loaded",
            "passed": model_loaded,
            "message": f"Model {health_data.get('model_id', 'unknown')}" if model_loaded else "No model"
        })
        
    except requests.exceptions.ConnectionError:
        result["status"] = "offline"
        result["checks"].append({"name": "api_reachable", "passed": False, "message": "Not running"})
    
    return result


def check_data_health(logger) -> Dict[str, Any]:
    """Check processed data health."""
    result = {"component": "data", "status": "healthy", "checks": []}
    
    required_files = [
        "interactions.parquet", "X_train_confidence.npz", "X_train_binary.npz",
        "user_item_mappings.json", "user_metadata.pkl", "user_pos_train.pkl", "data_stats.json",
    ]
    
    missing_files = []
    file_ages = {}
    
    for filename in required_files:
        file_path = HEALTH_CONFIG["processed_dir"] / filename
        if file_path.exists():
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            file_ages[filename] = (datetime.now() - mtime).days
        else:
            missing_files.append(filename)
    
    if missing_files:
        result["status"] = "critical"
        result["checks"].append({"name": "files_exist", "passed": False, "message": f"Missing: {missing_files}"})
    else:
        result["checks"].append({"name": "files_exist", "passed": True})
    
    # Check freshness
    if file_ages:
        max_age = max(file_ages.values())
        is_stale = max_age > HEALTH_CONFIG["max_data_age_days"]
        if is_stale:
            result["status"] = "warning"
        result["checks"].append({
            "name": "data_freshness",
            "passed": not is_stale,
            "message": f"Age: {max_age} days (threshold: {HEALTH_CONFIG['max_data_age_days']})"
        })
    
    # Check embeddings
    embeddings_exist = HEALTH_CONFIG["embeddings_path"].exists()
    result["checks"].append({
        "name": "embeddings_exist",
        "passed": embeddings_exist,
        "message": "PhoBERT embeddings available" if embeddings_exist else "Not found"
    })
    
    return result


def check_model_health(logger) -> Dict[str, Any]:
    """Check model artifacts health."""
    result = {"component": "models", "status": "healthy", "checks": []}
    
    registry_path = HEALTH_CONFIG["registry_path"]
    
    if not registry_path.exists():
        result["status"] = "critical"
        result["checks"].append({"name": "registry_exists", "passed": False})
        return result
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    # Check current_best
    current_best = registry.get("current_best")
    if isinstance(current_best, dict):
        current_best = current_best.get("model_id")
    
    if not current_best:
        result["status"] = "critical"
        result["checks"].append({"name": "current_best", "passed": False})
        return result
    
    models = registry.get("models", {})
    best_model = models.get(current_best) if isinstance(models, dict) else None
    
    if not best_model:
        result["status"] = "critical"
        result["checks"].append({"name": "current_best", "passed": False, "message": "Not found in registry"})
        return result
    
    result["checks"].append({"name": "current_best", "passed": True, "message": current_best})
    
    # Check model files exist
    model_path = Path(best_model.get("path", ""))
    result["checks"].append({
        "name": "model_files",
        "passed": model_path.exists(),
        "message": str(model_path) if model_path.exists() else "Path not found"
    })
    
    # Check performance
    metrics = best_model.get("metrics", {})
    recall_10 = metrics.get("recall@10", 0.0)
    is_performing = recall_10 >= HEALTH_CONFIG["min_recall_threshold"]
    result["checks"].append({
        "name": "model_performance",
        "passed": is_performing,
        "message": f"Recall@10: {recall_10:.4f} (threshold: {HEALTH_CONFIG['min_recall_threshold']})"
    })
    
    return result


def check_pipeline_health(logger) -> Dict[str, Any]:
    """Check pipeline execution health."""
    result = {"component": "pipelines", "status": "healthy", "checks": []}
    
    tracker = PipelineTracker()
    stats = tracker.get_stats(days=7)
    
    for pipeline_name, pipeline_stats in stats.get("stats_by_pipeline", {}).items():
        success_rate = pipeline_stats.get("success_rate")
        if success_rate is not None and success_rate < 0.5:
            result["status"] = "warning"
            result["checks"].append({
                "name": f"{pipeline_name}_failures",
                "passed": False,
                "message": f"{success_rate:.0%} success rate"
            })
    
    # Cleanup stale runs
    stale_count = tracker.cleanup_stale_runs(max_running_hours=24)
    if stale_count > 0:
        result["checks"].append({"name": "stale_runs", "passed": False, "message": f"Cleaned {stale_count}"})
    
    return result


def run_health_check(components=None, send_alerts=False, logger=None):
    """Run comprehensive health check."""
    if components is None:
        components = ["service", "data", "models", "pipelines"]
    
    check_functions = {
        "service": check_service_health,
        "data": check_data_health,
        "models": check_model_health,
        "pipelines": check_pipeline_health,
    }
    
    result = {"timestamp": datetime.now().isoformat(), "overall_status": "healthy", "components": {}}
    
    for component in components:
        if component in check_functions:
            check_result = check_functions[component](logger)
            result["components"][component] = check_result
            
            # Update overall status
            if check_result["status"] in ("critical", "error"):
                result["overall_status"] = "critical"
            elif check_result["status"] == "warning" and result["overall_status"] == "healthy":
                result["overall_status"] = "warning"
    
    # Send alerts
    if send_alerts and result["overall_status"] in ("warning", "critical"):
        failed = [n for n, d in result["components"].items() if d["status"] in ("warning", "critical")]
        send_pipeline_alert("health_check", result["overall_status"], f"Issues: {failed}", 
                           severity="error" if result["overall_status"] == "critical" else "warning")
    
    return result
```

## Component 7: Drift Detection

### File: `automation/drift_detection.py`
```python
"""
Drift Detection Pipeline.

Usage:
    python -m automation.drift_detection [--update-baseline]

Key Features:
- Rating distribution drift
- Popularity shift (Jaccard similarity)
- Interaction rate drift
- Generate drift report
- Create baseline on first run
"""

DRIFT_CONFIG = {
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "reports_dir": PROJECT_ROOT / "reports" / "drift",
    # Thresholds
    "rating_dist_threshold": 0.1,
    "popularity_shift_threshold": 0.2,
    "interaction_rate_threshold": 0.3,
}


def detect_rating_drift(current: Dict, baseline: Dict, logger) -> Dict[str, Any]:
    """Detect drift in rating distribution."""
    current_dist = current.get("rating_distribution", {})
    baseline_dist = baseline.get("rating_distribution", {})
    
    total_diff = 0.0
    for rating in ["1", "2", "3", "4", "5"]:
        diff = abs(current_dist.get(rating, 0) - baseline_dist.get(rating, 0))
        total_diff += diff
    
    drift_detected = total_diff > DRIFT_CONFIG["rating_dist_threshold"]
    return {"metric": "rating_distribution", "drift_detected": drift_detected, "total_difference": total_diff}


def detect_popularity_drift(current: Dict, baseline: Dict, logger) -> Dict[str, Any]:
    """Detect drift in item popularity using Jaccard similarity."""
    current_top = set(current.get("top_items", [])[:20])
    baseline_top = set(baseline.get("top_items", [])[:20])
    
    intersection = len(current_top & baseline_top)
    union = len(current_top | baseline_top)
    similarity = intersection / union if union > 0 else 1.0
    shift = 1.0 - similarity
    
    drift_detected = shift > DRIFT_CONFIG["popularity_shift_threshold"]
    return {
        "metric": "popularity_distribution",
        "drift_detected": drift_detected,
        "jaccard_similarity": similarity,
        "shift": shift,
        "new_items": list(current_top - baseline_top),
        "dropped_items": list(baseline_top - current_top),
    }


def detect_interaction_drift(current: Dict, baseline: Dict, logger) -> Dict[str, Any]:
    """Detect drift in interaction patterns."""
    current_rate = current.get("avg_interactions_per_user", 0)
    baseline_rate = baseline.get("avg_interactions_per_user", 0)
    
    if baseline_rate == 0:
        return {"metric": "interaction_rate", "status": "no_baseline"}
    
    change_rate = abs(current_rate - baseline_rate) / baseline_rate
    drift_detected = change_rate > DRIFT_CONFIG["interaction_rate_threshold"]
    
    return {"metric": "interaction_rate", "drift_detected": drift_detected, "change_rate": change_rate}


def detect_drift(update_baseline=False, logger=None):
    """Main drift detection pipeline."""
    with PipelineLock("drift_detection") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already running"}
        
        run_id = tracker.start_run("drift_detection", {"update_baseline": update_baseline})
        
        current_stats = load_current_stats(logger)
        baseline_stats = load_baseline_stats(logger)
        
        if baseline_stats is None:
            save_baseline_stats(current_stats, logger)
            return {"status": "baseline_created", "message": "Initial baseline created"}
        
        # Run all drift detections
        drift_results = [
            detect_rating_drift(current_stats, baseline_stats, logger),
            detect_popularity_drift(current_stats, baseline_stats, logger),
            detect_interaction_drift(current_stats, baseline_stats, logger),
        ]
        
        any_drift = any(r.get("drift_detected", False) for r in drift_results)
        
        # Generate report
        report_path = generate_drift_report(drift_results, logger)
        
        if update_baseline:
            save_baseline_stats(current_stats, logger)
        
        if any_drift:
            send_pipeline_alert("drift_detection", "warning", "Data drift detected")
        
        tracker.complete_run(run_id, {"status": "success", "drift_detected": any_drift})
        return {"status": "success", "drift_detected": any_drift, "report_file": str(report_path)}
```

## Component 8: Cleanup Pipeline

### File: `automation/cleanup.py`
```python
"""
Cleanup Pipeline for old logs, checkpoints, and artifacts.

Usage:
    python -m automation.cleanup [--dry-run] [--type logs|checkpoints|all]

Key Features:
- Log cleanup (configurable retention days)
- Checkpoint cleanup (keep recent N)
- Model artifact cleanup (keep deployed + recent)
- Dry-run mode for preview
"""

CLEANUP_CONFIG = {
    "logs_dir": PROJECT_ROOT / "logs",
    "checkpoints_dir": PROJECT_ROOT / "checkpoints",
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    # Retention settings
    "log_retention_days": 30,
    "checkpoint_keep_count": 3,
    "model_keep_count": 5,
}


def cleanup_old_logs(retention_days: int, dry_run: bool = False, logger=None) -> Dict:
    """Clean up log files older than retention period."""
    logs_dir = CLEANUP_CONFIG["logs_dir"]
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted_files = []
    deleted_size = 0
    
    for log_file in logs_dir.rglob("*.log"):
        if log_file.stat().st_mtime < cutoff.timestamp():
            deleted_size += log_file.stat().st_size
            if not dry_run:
                log_file.unlink()
            deleted_files.append(str(log_file))
    
    return {
        "type": "logs",
        "dry_run": dry_run,
        "deleted_count": len(deleted_files),
        "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
        "cutoff_date": cutoff.isoformat(),
    }


def cleanup_old_checkpoints(keep_count: int, dry_run: bool = False, logger=None) -> Dict:
    """Clean up old training checkpoints, keeping most recent ones."""
    checkpoints_dir = CLEANUP_CONFIG["checkpoints_dir"]
    deleted_files = []
    deleted_size = 0
    
    for model_type in ["als", "bpr"]:
        model_dir = checkpoints_dir / model_type
        if not model_dir.exists():
            continue
        
        # Get checkpoint files sorted by modification time (newest first)
        checkpoints = sorted(
            model_dir.glob("checkpoint_*.npy"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Delete old checkpoints beyond keep count
        for ckpt in checkpoints[keep_count:]:
            deleted_size += ckpt.stat().st_size
            if not dry_run:
                ckpt.unlink()
            deleted_files.append(str(ckpt))
    
    return {
        "type": "checkpoints",
        "dry_run": dry_run,
        "deleted_count": len(deleted_files),
        "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
        "keep_count": keep_count,
    }


def cleanup_old_models(keep_count: int, dry_run: bool = False, logger=None) -> Dict:
    """Clean up old model artifacts, preserving deployed and recent models."""
    artifacts_dir = CLEANUP_CONFIG["artifacts_dir"]
    registry_path = artifacts_dir / "registry.json"
    
    # Load registry to know which model is deployed
    deployed_model_id = None
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
            deployed_model_id = registry.get("current_best")
    
    deleted_dirs = []
    deleted_size = 0
    
    for model_type in ["als", "bpr"]:
        model_dir = artifacts_dir / model_type
        if not model_dir.exists():
            continue
        
        # Get model versions sorted by creation time
        versions = sorted(
            [d for d in model_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        
        for version_dir in versions[keep_count:]:
            # Never delete deployed model
            if deployed_model_id and version_dir.name == deployed_model_id:
                continue
            
            dir_size = sum(f.stat().st_size for f in version_dir.rglob("*") if f.is_file())
            deleted_size += dir_size
            
            if not dry_run:
                shutil.rmtree(version_dir)
            deleted_dirs.append(str(version_dir))
    
    return {
        "type": "models",
        "dry_run": dry_run,
        "deleted_count": len(deleted_dirs),
        "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
        "preserved_deployed": deployed_model_id,
    }


def run_cleanup(cleanup_type: str = "all", dry_run: bool = False, logger=None) -> Dict:
    """Main cleanup pipeline."""
    with PipelineLock("cleanup") as lock:
        if not lock.acquired:
            return {"status": "skipped", "message": "Already running"}
        
        run_id = tracker.start_run("cleanup", {"type": cleanup_type, "dry_run": dry_run})
        results = []
        
        if cleanup_type in ["all", "logs"]:
            results.append(cleanup_old_logs(CLEANUP_CONFIG["log_retention_days"], dry_run, logger))
        
        if cleanup_type in ["all", "checkpoints"]:
            results.append(cleanup_old_checkpoints(CLEANUP_CONFIG["checkpoint_keep_count"], dry_run, logger))
        
        if cleanup_type in ["all", "models"]:
            results.append(cleanup_old_models(CLEANUP_CONFIG["model_keep_count"], dry_run, logger))
        
        total_deleted_mb = sum(r.get("deleted_size_mb", 0) for r in results)
        tracker.complete_run(run_id, {"status": "success", "total_freed_mb": total_deleted_mb})
        
        return {"status": "success", "dry_run": dry_run, "results": results, "total_freed_mb": total_deleted_mb}
```

---

## Component 9: Utility Functions

### File: `scripts/utils.py`
```python
"""
Common utilities for automation pipelines.

Key Components:
- retry decorator with exponential backoff
- PipelineTracker for SQLite metrics storage
- PipelineLock for preventing concurrent runs
- alert helpers
"""

# ========== Retry Decorator ==========
def retry(max_attempts=3, delay=60, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}")
                    logger.info(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
            return None
        return wrapper
    return decorator


# ========== Pipeline Tracker ==========
class PipelineTracker:
    """Track pipeline runs in SQLite database."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or PROJECT_ROOT / "logs" / "pipeline_metrics.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize database with runs table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_name TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    config TEXT,
                    result TEXT,
                    error_message TEXT
                )
            """)
    
    def start_run(self, pipeline_name: str, config: Dict = None) -> str:
        """Record pipeline run start, return run_id."""
        run_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, pipeline_name, start_time, status, config) VALUES (?, ?, ?, ?, ?)",
                (run_id, pipeline_name, datetime.now().isoformat(), "running", json.dumps(config or {}))
            )
        return run_id
    
    def complete_run(self, run_id: str, result: Dict):
        """Mark run as completed with result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pipeline_runs SET end_time=?, status=?, result=? WHERE run_id=?",
                (datetime.now().isoformat(), "completed", json.dumps(result), run_id)
            )
    
    def fail_run(self, run_id: str, error_message: str):
        """Mark run as failed with error."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pipeline_runs SET end_time=?, status=?, error_message=? WHERE run_id=?",
                (datetime.now().isoformat(), "failed", error_message, run_id)
            )
    
    def get_stats(self, days: int = 7) -> Dict:
        """Get pipeline statistics for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pipeline_name, status, COUNT(*) as count
                FROM pipeline_runs
                WHERE start_time > ?
                GROUP BY pipeline_name, status
            """, (cutoff.isoformat(),))
            
            stats_by_pipeline = {}
            for row in cursor:
                pipeline, status, count = row
                if pipeline not in stats_by_pipeline:
                    stats_by_pipeline[pipeline] = {"total": 0, "completed": 0, "failed": 0}
                stats_by_pipeline[pipeline][status] = count
                stats_by_pipeline[pipeline]["total"] += count
            
            # Calculate success rates
            for stats in stats_by_pipeline.values():
                if stats["total"] > 0:
                    stats["success_rate"] = stats.get("completed", 0) / stats["total"]
            
            return {"days": days, "stats_by_pipeline": stats_by_pipeline}
    
    def cleanup_stale_runs(self, max_running_hours: int = 24) -> int:
        """Clean up runs stuck in 'running' state."""
        cutoff = datetime.now() - timedelta(hours=max_running_hours)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE pipeline_runs SET status='failed', error_message='Stale run cleaned up' "
                "WHERE status='running' AND start_time < ?",
                (cutoff.isoformat(),)
            )
            return cursor.rowcount


# ========== Pipeline Lock ==========
class PipelineLock:
    """
    File-based lock to prevent concurrent pipeline runs.
    
    Usage:
        with PipelineLock("data_refresh") as lock:
            if lock.acquired:
                run_pipeline()
            else:
                print("Already running")
    """
    
    def __init__(self, pipeline_name: str, timeout: int = 3600):
        self.lock_dir = PROJECT_ROOT / "logs" / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_dir / f"{pipeline_name}.lock"
        self.timeout = timeout
        self.acquired = False
    
    def __enter__(self):
        # Check for stale lock
        if self.lock_file.exists():
            lock_time = datetime.fromtimestamp(self.lock_file.stat().st_mtime)
            if (datetime.now() - lock_time).total_seconds() > self.timeout:
                self.lock_file.unlink()  # Remove stale lock
        
        if not self.lock_file.exists():
            self.lock_file.write_text(str(os.getpid()))
            self.acquired = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired and self.lock_file.exists():
            self.lock_file.unlink()
        return False


# ========== Alert Helper ==========
def send_pipeline_alert(pipeline_name: str, severity: str, message: str, details: Dict = None):
    """Send alert for pipeline events (integrates with alerting.py)."""
    from alerting import send_alert
    
    full_message = f"[{pipeline_name.upper()}] {message}"
    if details:
        full_message += f"\nDetails: {json.dumps(details, indent=2)}"
    
    send_alert(
        subject=f"Pipeline Alert: {pipeline_name}",
        message=full_message,
        severity=severity
    )
```

---

## Component 10: Configuration & CLI

### Config File: `config/scheduler_config.json`
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
      "description": "Weekly BERT embeddings refresh",
      "schedule": {"day_of_week": "tue", "hour": 3, "minute": 0},
      "module": "automation.bert_embeddings",
      "args": []
    },
    "drift_detection": {
      "enabled": true,
      "description": "Weekly drift detection monitoring",
      "schedule": {"day_of_week": "mon", "hour": 9, "minute": 0},
      "module": "automation.drift_detection",
      "args": []
    },
    "model_training": {
      "enabled": true,
      "description": "Weekly model training (ALS + BPR)",
      "schedule": {"day_of_week": "sun", "hour": 3, "minute": 0},
      "module": "automation.model_training",
      "args": ["--auto-select"]
    },
    "model_deployment": {
      "enabled": true,
      "description": "Daily model deployment check",
      "schedule": {"hour": 5, "minute": 0},
      "module": "automation.model_deployment",
      "args": []
    },
    "health_check": {
      "enabled": true,
      "description": "Hourly health check",
      "schedule": {"minute": 0},
      "module": "automation.health_check",
      "args": []
    }
  }
}
```

### Management Script: `manage_scheduler.ps1`
```powershell
# PowerShell script for managing the VieComRec scheduler

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "logs", "run")]
    [string]$Action = "status",
    
    [Parameter()]
    [string]$Task = "",
    
    [switch]$Background
)

$ProjectDir = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectDir "logs\scheduler"

function Start-Scheduler {
    Write-Host "Starting VieComRec Scheduler..." -ForegroundColor Green
    
    if ($Background) {
        Start-Process -FilePath "python" -ArgumentList "-m automation.scheduler" `
            -WorkingDirectory $ProjectDir -WindowStyle Hidden
        Write-Host "Scheduler started in background" -ForegroundColor Green
    } else {
        python -m automation.scheduler
    }
}

function Stop-Scheduler {
    Write-Host "Stopping scheduler..." -ForegroundColor Yellow
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*automation.scheduler*" }
    
    if ($processes) {
        $processes | Stop-Process -Force
        Write-Host "Scheduler stopped" -ForegroundColor Green
    } else {
        Write-Host "Scheduler not running" -ForegroundColor Gray
    }
}

function Get-SchedulerStatus {
    $statusFile = Join-Path $LogDir "task_status.json"
    
    if (Test-Path $statusFile) {
        $status = Get-Content $statusFile | ConvertFrom-Json
        Write-Host "`nTask Status:" -ForegroundColor Cyan
        $status.PSObject.Properties | ForEach-Object {
            $taskName = $_.Name
            $taskStatus = $_.Value.status
            $timestamp = $_.Value.timestamp
            
            $color = switch ($taskStatus) {
                "success" { "Green" }
                "failed" { "Red" }
                "running" { "Yellow" }
                default { "Gray" }
            }
            
            Write-Host "  $taskName: $taskStatus ($timestamp)" -ForegroundColor $color
        }
    } else {
        Write-Host "No status file found. Scheduler may not have run yet." -ForegroundColor Gray
    }
}

function Run-Task {
    if (-not $Task) {
        Write-Host "Please specify a task with -Task <name>" -ForegroundColor Red
        return
    }
    
    Write-Host "Running task: $Task" -ForegroundColor Green
    python -m automation.$Task
}

# Main
switch ($Action) {
    "start" { Start-Scheduler }
    "stop" { Stop-Scheduler }
    "status" { Get-SchedulerStatus }
    "logs" { Get-Content (Join-Path $LogDir "scheduler.log") -Tail 50 }
    "run" { Run-Task }
}
```

---

## Full Automation Checklist

### Daily Tasks
- [x] Data refresh (2:00 AM) - `automation.data_refresh`
- [x] Deploy model updates (5:00 AM) - `automation.model_deployment`
- [x] Health checks (hourly :00) - `automation.health_check`

### Weekly Tasks
- [x] BERT embeddings refresh (Tuesday 3:00 AM) - `automation.bert_embeddings`
- [x] Drift detection (Monday 9:00 AM) - `automation.drift_detection`
- [x] Model retraining (Sunday 3:00 AM) - `automation.model_training`

### Monthly Tasks
- [x] Log cleanup (30-day retention)
- [x] Checkpoint cleanup (keep last 3)
- [x] Model artifact cleanup (keep last 5 + deployed)
- [ ] Database vacuum (SQLite maintenance)

---

## Dependencies

```python
# requirements_automation.txt (included in requirements.txt)
apscheduler>=3.10.0
requests>=2.28.0
pytz>=2023.3
torch>=2.0.0
transformers>=4.30.0
```

---

## CLI Quick Reference

```powershell
# Start scheduler (blocking)
python -m automation.scheduler

# Start scheduler (background - PowerShell)
.\manage_scheduler.ps1 start -Background

# Run individual pipelines
python -m automation.data_refresh [--force] [--force-full] [--dry-run]
python -m automation.model_training [--model als|bpr|both] [--auto-select]
python -m automation.model_deployment [--model-id ID] [--rollback]
python -m automation.health_check [--component all|service|data|models] [--alert]
python -m automation.drift_detection [--update-baseline]
python -m automation.bert_embeddings [--force]
python -m automation.cleanup [--type logs|checkpoints|all] [--dry-run]

# Check scheduler status
.\manage_scheduler.ps1 status

# View logs
.\manage_scheduler.ps1 logs
```

---

## Success Criteria

- [x] APScheduler với cron triggers hoạt động
- [x] Data refresh với incremental update support
- [x] Model training với BERT init, warm-start, early stopping
- [x] Model deployment với rollback support
- [x] BERT embeddings refresh cho content-based
- [x] Health checks toàn diện (service, data, models, pipelines)
- [x] Drift detection với baseline comparison
- [x] Cleanup pipeline với dry-run mode
- [x] PipelineLock ngăn concurrent runs
- [x] PipelineTracker lưu metrics vào SQLite
- [x] Retry decorator với exponential backoff
- [x] Alert integration (send_pipeline_alert)
- [x] Task status tracking (JSON file)
- [x] PowerShell management script

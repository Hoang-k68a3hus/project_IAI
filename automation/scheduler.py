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

# =============================================================================
# Configuration
# =============================================================================

# Project directory (in Docker: /app, local: project root)
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).parent.parent))
LOG_DIR = PROJECT_DIR / "logs" / "scheduler"
SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")
TIMEZONE = pytz.timezone(os.environ.get("TZ", "UTC"))

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# All tasks use automation modules (python -m automation.xxx)
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


# =============================================================================
# Logging Setup
# =============================================================================

def setup_scheduler_logging() -> logging.Logger:
    """Set up logging for the scheduler."""
    logger = logging.getLogger("AutomationScheduler")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = LOG_DIR / "scheduler.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)

    return logger


logger = setup_scheduler_logging()


# =============================================================================
# Task Status Tracking
# =============================================================================

def update_task_status(
    task_name: str,
    status: str,
    exit_code: Optional[int] = None,
    log_file: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update task status in JSON file."""
    status_file = LOG_DIR / "task_status.json"

    # Load existing status
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            all_status = json.load(f)
    else:
        all_status = {}

    # Update task status
    all_status[task_name] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "exit_code": exit_code,
        "log_file": log_file,
        "error": error,
    }

    # Save
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(all_status, f, indent=2, ensure_ascii=False)


# =============================================================================
# Task Execution
# =============================================================================

def run_task(task_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a scheduled task.

    Args:
        task_name: Name of the task
        config: Task configuration

    Returns:
        Task result dict
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{task_name}_{timestamp}.log"

    logger.info("=" * 60)
    logger.info(f"TASK: {config['description']}")
    logger.info("=" * 60)
    logger.info(f"Starting task: {task_name}")

    result: Dict[str, Any] = {
        "task": task_name,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # All tasks use python -m automation.xxx
        cmd = [
            sys.executable,
            "-m",
            config["module"],
        ] + config.get("args", [])

        # Execute command
        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
                env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
            )

        exit_code = process.returncode
        result["exit_code"] = exit_code
        result["log_file"] = str(log_file)

        if exit_code == 0:
            result["status"] = "success"
            logger.info(f"✓ Task completed: {task_name}")
        else:
            result["status"] = "failed"
            logger.error(f"✗ Task failed: {task_name} (exit code: {exit_code})")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Task timed out after 1 hour"
        logger.error(f"✗ Task timed out: {task_name}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"✗ Task error: {task_name} - {e}")

    # Update status file
    update_task_status(
        task_name,
        result["status"],
        result.get("exit_code"),
        result.get("log_file"),
        result.get("error"),
    )

    return result


def create_task_wrapper(task_name: str, config: Dict[str, Any]) -> Callable[[], None]:
    """Create a wrapper function for a task."""
    def wrapper() -> None:
        try:
            result = run_task(task_name, config)
            if result["status"] != "success":
                logger.warning(f"Task {task_name} failed: {result}")
        except Exception as e:
            logger.error(f"Unhandled error in task {task_name}: {e}")

    return wrapper


# =============================================================================
# Scheduler Setup
# =============================================================================

def create_scheduler() -> BlockingScheduler:
    """Create and configure the scheduler."""
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    for task_name, config in SCHEDULER_CONFIG.items():
        if not config.get("enabled", True):
            logger.info(f"Skipping disabled task: {task_name}")
            continue

        # Create trigger from schedule config
        schedule = config["schedule"]
        trigger = CronTrigger(**schedule, timezone=TIMEZONE)

        # Add job
        scheduler.add_job(
            create_task_wrapper(task_name, config),
            trigger=trigger,
            id=task_name,
            name=config.get("description", task_name),
            replace_existing=True,
        )

        logger.info(f"✓ Registered job: {task_name}")

    return scheduler


def print_scheduled_jobs(scheduler: BlockingScheduler) -> None:
    """Print all scheduled jobs."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("SCHEDULED JOBS:")
    logger.info("=" * 80)

    for job in scheduler.get_jobs():
        logger.info(f"  [{job.id}] - Trigger: {job.trigger}")

    logger.info("=" * 80)
    logger.info("")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the scheduler."""
    logger.info("=" * 80)
    logger.info("VIECOMREC AUTOMATION SCHEDULER STARTING")
    logger.info(f"Project Directory: {PROJECT_DIR}")
    logger.info(f"Service URL: {SERVICE_URL}")
    logger.info(f"Log Directory: {LOG_DIR}")
    logger.info("=" * 80)

    # Create scheduler
    scheduler = create_scheduler()

    # Print registered jobs
    print_scheduled_jobs(scheduler)

    try:
        logger.info("✓ Scheduler started successfully")
        logger.info("Press Ctrl+C to stop the scheduler")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise


if __name__ == "__main__":
    main()

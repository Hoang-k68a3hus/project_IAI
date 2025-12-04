"""
Scheduler API Endpoints for VieComRec.

This module provides REST API endpoints for managing the automation scheduler.

Endpoints:
- GET /scheduler/status: Get scheduler status and jobs
- GET /scheduler/jobs: List all scheduled jobs
- POST /scheduler/jobs/{job_id}/run: Trigger a job manually
- POST /scheduler/jobs/{job_id}/enable: Enable a job
- POST /scheduler/jobs/{job_id}/disable: Disable a job
- PUT /scheduler/jobs/{job_id}/schedule: Update job schedule
- GET /scheduler/logs: Get scheduler logs
- GET /scheduler/logs/{task_name}: Get logs for a specific task
- GET /scheduler/history: Get task execution history

Usage:
    Include this router in the main API app:
    from service.scheduler_api import scheduler_router
    app.include_router(scheduler_router, prefix="/scheduler", tags=["Scheduler"])
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", project_root))
LOG_DIR = PROJECT_DIR / "logs" / "scheduler"
SCHEDULER_CONFIG_PATH = PROJECT_DIR / "config" / "scheduler_config.json"

# Default scheduler config (if file doesn't exist)
DEFAULT_SCHEDULER_CONFIG = {
    "data_refresh": {
        "enabled": True,
        "description": "Daily data refresh from raw CSV files",
        "schedule": {"hour": 2, "minute": 0},
        "module": "automation.data_refresh",
        "args": [],
    },
    "bert_embeddings": {
        "enabled": True,
        "description": "Weekly BERT embeddings refresh",
        "schedule": {"day_of_week": "tue", "hour": 3, "minute": 0},
        "module": "automation.bert_embeddings",
        "args": [],
    },
    "drift_detection": {
        "enabled": True,
        "description": "Weekly drift detection monitoring",
        "schedule": {"day_of_week": "mon", "hour": 9, "minute": 0},
        "module": "automation.drift_detection",
        "args": [],
    },
    "model_training": {
        "enabled": True,
        "description": "Weekly model training (ALS + BPR)",
        "schedule": {"day_of_week": "sun", "hour": 3, "minute": 0},
        "module": "automation.model_training",
        "args": ["--auto-select"],
    },
    "model_deployment": {
        "enabled": True,
        "description": "Daily model deployment check",
        "schedule": {"hour": 5, "minute": 0},
        "module": "automation.model_deployment",
        "args": [],
    },
    "health_check": {
        "enabled": True,
        "description": "Hourly health check",
        "schedule": {"minute": 0},
        "module": "automation.health_check",
        "args": [],
    },
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler_api")

# ============================================================================
# Request/Response Models
# ============================================================================


class ScheduleConfig(BaseModel):
    """Cron-style schedule configuration."""
    minute: Optional[int] = Field(None, ge=0, le=59, description="Minute (0-59)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour (0-23)")
    day_of_week: Optional[str] = Field(
        None,
        description="Day of week: mon, tue, wed, thu, fri, sat, sun"
    )
    day: Optional[int] = Field(None, ge=1, le=31, description="Day of month (1-31)")


class JobInfo(BaseModel):
    """Information about a scheduled job."""
    job_id: str
    description: str
    enabled: bool
    schedule: Dict[str, Any]
    module: str
    args: List[str]
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for listing all jobs."""
    jobs: List[JobInfo]
    total: int
    scheduler_running: bool


class JobStatusResponse(BaseModel):
    """Response for job status."""
    job_id: str
    status: str
    enabled: bool
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None
    last_log_file: Optional[str] = None


class UpdateScheduleRequest(BaseModel):
    """Request to update job schedule."""
    schedule: ScheduleConfig


class UpdateScheduleResponse(BaseModel):
    """Response for updating job schedule."""
    job_id: str
    status: str
    old_schedule: Dict[str, Any]
    new_schedule: Dict[str, Any]
    message: str


class RunJobResponse(BaseModel):
    """Response for manually running a job."""
    job_id: str
    status: str
    message: str
    log_file: Optional[str] = None


class TaskLogEntry(BaseModel):
    """A single log entry."""
    timestamp: str
    level: str
    message: str


class TaskLogsResponse(BaseModel):
    """Response for task logs."""
    task_name: str
    logs: List[str]
    log_file: Optional[str] = None
    total_lines: int


class SchedulerStatusResponse(BaseModel):
    """Scheduler status response."""
    running: bool
    uptime: Optional[str] = None
    total_jobs: int
    enabled_jobs: int
    disabled_jobs: int
    last_health_check: Optional[str] = None
    next_scheduled_task: Optional[Dict[str, Any]] = None


class TaskHistoryEntry(BaseModel):
    """A single task execution history entry."""
    task_name: str
    status: str
    timestamp: str
    exit_code: Optional[int] = None
    log_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class TaskHistoryResponse(BaseModel):
    """Response for task execution history."""
    history: List[TaskHistoryEntry]
    total: int
    page: int
    page_size: int


# ============================================================================
# Helper Functions
# ============================================================================


def load_scheduler_config() -> Dict[str, Any]:
    """Load scheduler configuration from file or return default."""
    if SCHEDULER_CONFIG_PATH.exists():
        try:
            with open(SCHEDULER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading scheduler config: {e}. Using default.")
    return DEFAULT_SCHEDULER_CONFIG.copy()


def save_scheduler_config(config: Dict[str, Any]) -> None:
    """Save scheduler configuration to file."""
    SCHEDULER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULER_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_task_status() -> Dict[str, Any]:
    """Load task status from file."""
    status_file = LOG_DIR / "task_status.json"
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading task status: {e}")
    return {}


def get_latest_log_file(task_name: str) -> Optional[Path]:
    """Get the most recent log file for a task."""
    if not LOG_DIR.exists():
        return None
    
    log_files = list(LOG_DIR.glob(f"{task_name}_*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=lambda f: f.stat().st_mtime)


def format_schedule(schedule: Dict[str, Any]) -> str:
    """Format schedule dict as human-readable string."""
    parts = []
    
    if "minute" in schedule:
        parts.append(f"minute={schedule['minute']}")
    if "hour" in schedule:
        parts.append(f"hour={schedule['hour']}")
    if "day_of_week" in schedule:
        parts.append(f"day={schedule['day_of_week']}")
    if "day" in schedule:
        parts.append(f"day_of_month={schedule['day']}")
    
    return ", ".join(parts) if parts else "every minute"


def check_scheduler_running() -> bool:
    """Check if the scheduler process is running."""
    try:
        # Check if scheduler.py process exists
        result = subprocess.run(
            ["pgrep", "-f", "automation.scheduler"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        # On Windows or if pgrep not available
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True
            )
            return "automation.scheduler" in result.stdout or "scheduler.py" in result.stdout
        except Exception:
            return False


# ============================================================================
# API Router
# ============================================================================

scheduler_router = APIRouter()


@scheduler_router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status():
    """
    Get scheduler status and overview.
    
    Returns:
        SchedulerStatusResponse with scheduler state information
    """
    config = load_scheduler_config()
    task_status = load_task_status()
    
    enabled_count = sum(1 for job in config.values() if job.get("enabled", True))
    disabled_count = len(config) - enabled_count
    
    # Check if scheduler is running
    is_running = check_scheduler_running()
    
    # Get last health check status
    last_health = task_status.get("health_check", {})
    last_health_time = last_health.get("timestamp")
    
    # Find next scheduled task (simplified - actual timing from APScheduler)
    next_task = None
    for job_id, job_config in config.items():
        if job_config.get("enabled", True):
            next_task = {
                "job_id": job_id,
                "description": job_config.get("description", ""),
                "schedule": format_schedule(job_config.get("schedule", {}))
            }
            break
    
    return SchedulerStatusResponse(
        running=is_running,
        uptime=None,  # Would need scheduler process start time
        total_jobs=len(config),
        enabled_jobs=enabled_count,
        disabled_jobs=disabled_count,
        last_health_check=last_health_time,
        next_scheduled_task=next_task
    )


@scheduler_router.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """
    List all scheduled jobs.
    
    Returns:
        JobListResponse with list of all jobs and their status
    """
    config = load_scheduler_config()
    task_status = load_task_status()
    is_running = check_scheduler_running()
    
    jobs = []
    for job_id, job_config in config.items():
        status = task_status.get(job_id, {})
        
        jobs.append(JobInfo(
            job_id=job_id,
            description=job_config.get("description", ""),
            enabled=job_config.get("enabled", True),
            schedule=job_config.get("schedule", {}),
            module=job_config.get("module", ""),
            args=job_config.get("args", []),
            last_run=status.get("timestamp"),
            last_status=status.get("status")
        ))
    
    return JobListResponse(
        jobs=jobs,
        total=len(jobs),
        scheduler_running=is_running
    )


@scheduler_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a specific job.
    
    Args:
        job_id: The job identifier
    
    Returns:
        JobStatusResponse with job details and status
    """
    config = load_scheduler_config()
    task_status = load_task_status()
    
    if job_id not in config:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job_config = config[job_id]
    status = task_status.get(job_id, {})
    
    # Get latest log file
    log_file = get_latest_log_file(job_id)
    
    return JobStatusResponse(
        job_id=job_id,
        status="enabled" if job_config.get("enabled", True) else "disabled",
        enabled=job_config.get("enabled", True),
        last_run=status.get("timestamp"),
        last_status=status.get("status"),
        last_log_file=str(log_file) if log_file else None
    )


@scheduler_router.post("/jobs/{job_id}/run", response_model=RunJobResponse)
async def run_job_manually(
    job_id: str,
    background_tasks: BackgroundTasks
):
    """
    Trigger a job to run immediately.
    
    This runs the job in the background and returns immediately.
    Check job status or logs for execution results.
    
    Args:
        job_id: The job identifier
    
    Returns:
        RunJobResponse with job execution status
    """
    config = load_scheduler_config()
    
    if job_id not in config:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job_config = config[job_id]
    module = job_config.get("module", "")
    args = job_config.get("args", [])
    
    if not module:
        raise HTTPException(status_code=400, detail=f"Job has no module configured: {job_id}")
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{job_id}_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run job in background
    def execute_job():
        try:
            cmd = [sys.executable, "-m", module] + args
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Manual run triggered at {datetime.now().isoformat()} ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("=" * 60 + "\n\n")
                
                process = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_DIR),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)}
                )
                
                f.write(f"\n\n=== Task completed with exit code {process.returncode} ===\n")
            
            # Update task status
            status_file = LOG_DIR / "task_status.json"
            status = {}
            if status_file.exists():
                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
            
            status[job_id] = {
                "status": "success" if process.returncode == 0 else "failed",
                "timestamp": datetime.now().isoformat(),
                "exit_code": process.returncode,
                "log_file": str(log_file),
                "manual_trigger": True
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
                
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id} timed out")
        except Exception as e:
            logger.error(f"Error running job {job_id}: {e}")
    
    background_tasks.add_task(execute_job)
    
    logger.info(f"Manually triggered job: {job_id}")
    
    return RunJobResponse(
        job_id=job_id,
        status="started",
        message=f"Job '{job_id}' has been triggered. Check logs for progress.",
        log_file=str(log_file)
    )


@scheduler_router.post("/jobs/{job_id}/enable", response_model=JobStatusResponse)
async def enable_job(job_id: str):
    """
    Enable a scheduled job.
    
    Args:
        job_id: The job identifier
    
    Returns:
        JobStatusResponse with updated job status
    """
    config = load_scheduler_config()
    
    if job_id not in config:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    config[job_id]["enabled"] = True
    save_scheduler_config(config)
    
    logger.info(f"Enabled job: {job_id}")
    
    task_status = load_task_status()
    status = task_status.get(job_id, {})
    
    return JobStatusResponse(
        job_id=job_id,
        status="enabled",
        enabled=True,
        last_run=status.get("timestamp"),
        last_status=status.get("status")
    )


@scheduler_router.post("/jobs/{job_id}/disable", response_model=JobStatusResponse)
async def disable_job(job_id: str):
    """
    Disable a scheduled job.
    
    Args:
        job_id: The job identifier
    
    Returns:
        JobStatusResponse with updated job status
    """
    config = load_scheduler_config()
    
    if job_id not in config:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    config[job_id]["enabled"] = False
    save_scheduler_config(config)
    
    logger.info(f"Disabled job: {job_id}")
    
    task_status = load_task_status()
    status = task_status.get(job_id, {})
    
    return JobStatusResponse(
        job_id=job_id,
        status="disabled",
        enabled=False,
        last_run=status.get("timestamp"),
        last_status=status.get("status")
    )


@scheduler_router.put("/jobs/{job_id}/schedule", response_model=UpdateScheduleResponse)
async def update_job_schedule(job_id: str, request: UpdateScheduleRequest):
    """
    Update the schedule for a job.
    
    Args:
        job_id: The job identifier
        request: UpdateScheduleRequest with new schedule
    
    Returns:
        UpdateScheduleResponse with old and new schedule
    
    Example:
        PUT /scheduler/jobs/data_refresh/schedule
        {
            "schedule": {
                "hour": 3,
                "minute": 30
            }
        }
    """
    config = load_scheduler_config()
    
    if job_id not in config:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    old_schedule = config[job_id].get("schedule", {}).copy()
    
    # Update schedule with new values
    new_schedule = {}
    if request.schedule.minute is not None:
        new_schedule["minute"] = request.schedule.minute
    if request.schedule.hour is not None:
        new_schedule["hour"] = request.schedule.hour
    if request.schedule.day_of_week is not None:
        new_schedule["day_of_week"] = request.schedule.day_of_week
    if request.schedule.day is not None:
        new_schedule["day"] = request.schedule.day
    
    if not new_schedule:
        raise HTTPException(
            status_code=400, 
            detail="At least one schedule parameter must be provided"
        )
    
    config[job_id]["schedule"] = new_schedule
    save_scheduler_config(config)
    
    logger.info(f"Updated schedule for job {job_id}: {old_schedule} -> {new_schedule}")
    
    return UpdateScheduleResponse(
        job_id=job_id,
        status="updated",
        old_schedule=old_schedule,
        new_schedule=new_schedule,
        message=(
            f"Schedule updated. Restart scheduler for changes to take effect. "
            f"New schedule: {format_schedule(new_schedule)}"
        )
    )


@scheduler_router.get("/logs", response_model=TaskLogsResponse)
async def get_scheduler_logs(
    lines: int = Query(default=100, ge=1, le=1000, description="Number of lines to return")
):
    """
    Get main scheduler logs.
    
    Args:
        lines: Number of log lines to return
    
    Returns:
        TaskLogsResponse with scheduler logs
    """
    log_file = LOG_DIR / "scheduler.log"
    
    if not log_file.exists():
        return TaskLogsResponse(
            task_name="scheduler",
            logs=[],
            log_file=None,
            total_lines=0
        )
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Get last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        log_lines = [line.strip() for line in log_lines]
        
        return TaskLogsResponse(
            task_name="scheduler",
            logs=log_lines,
            log_file=str(log_file),
            total_lines=len(all_lines)
        )
    except Exception as e:
        logger.error(f"Error reading scheduler logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")


@scheduler_router.get("/logs/{task_name}", response_model=TaskLogsResponse)
async def get_task_logs(
    task_name: str,
    lines: int = Query(default=100, ge=1, le=1000, description="Number of lines to return")
):
    """
    Get logs for a specific task.
    
    Args:
        task_name: The task name (e.g., 'data_refresh', 'model_training')
        lines: Number of log lines to return
    
    Returns:
        TaskLogsResponse with task logs
    """
    config = load_scheduler_config()
    
    if task_name not in config:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_name}")
    
    log_file = get_latest_log_file(task_name)
    
    if log_file is None or not log_file.exists():
        return TaskLogsResponse(
            task_name=task_name,
            logs=[],
            log_file=None,
            total_lines=0
        )
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Get last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        log_lines = [line.strip() for line in log_lines]
        
        return TaskLogsResponse(
            task_name=task_name,
            logs=log_lines,
            log_file=str(log_file),
            total_lines=len(all_lines)
        )
    except Exception as e:
        logger.error(f"Error reading task logs for {task_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")


@scheduler_router.get("/history", response_model=TaskHistoryResponse)
async def get_task_history(
    task_name: Optional[str] = Query(default=None, description="Filter by task name"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page")
):
    """
    Get task execution history.
    
    Args:
        task_name: Optional filter by task name
        page: Page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        TaskHistoryResponse with execution history
    """
    task_status = load_task_status()
    
    # Build history list
    history = []
    for job_id, status in task_status.items():
        if task_name and job_id != task_name:
            continue
        
        history.append(TaskHistoryEntry(
            task_name=job_id,
            status=status.get("status", "unknown"),
            timestamp=status.get("timestamp", ""),
            exit_code=status.get("exit_code"),
            log_file=status.get("log_file"),
            duration_seconds=status.get("duration"),
            error=status.get("error")
        ))
    
    # Sort by timestamp (most recent first)
    history.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Paginate
    total = len(history)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = history[start:end]
    
    return TaskHistoryResponse(
        history=paginated,
        total=total,
        page=page,
        page_size=page_size
    )


@scheduler_router.post("/reload")
async def reload_scheduler_config():
    """
    Signal the scheduler to reload its configuration.
    
    Note: This requires the scheduler to support hot-reloading.
    Currently, you may need to restart the scheduler process.
    
    Returns:
        Status message
    """
    config = load_scheduler_config()
    
    # Save config to ensure file is up-to-date
    save_scheduler_config(config)
    
    logger.info("Scheduler configuration saved. Restart scheduler for changes to take effect.")
    
    return {
        "status": "config_saved",
        "message": (
            "Configuration saved. Please restart the scheduler for changes to take effect. "
            "Use: docker compose restart scheduler (Docker) or "
            "python -m automation.scheduler (local)"
        ),
        "total_jobs": len(config),
        "enabled_jobs": sum(1 for job in config.values() if job.get("enabled", True))
    }

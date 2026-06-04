"""Shared utilities for automation pipelines.

The automation modules intentionally depend on a small interface here instead
of each module carrying its own retry, logging, locking, and run tracking code.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, cast


F = TypeVar("F", bound=Callable[..., Any])

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
AUTOMATION_LOG_DIR = PROJECT_DIR / "logs" / "automation"
RUNS_PATH = AUTOMATION_LOG_DIR / "pipeline_runs.jsonl"
ALERTS_PATH = AUTOMATION_LOG_DIR / "alerts.jsonl"
LOCK_DIR = AUTOMATION_LOG_DIR / "locks"


def _utc_now() -> datetime:
    return datetime.utcnow()


def _iso_now() -> str:
    return _utc_now().isoformat(timespec="seconds") + "Z"


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def setup_logging(
    name: str,
    level: Union[int, str] = logging.INFO,
    console: bool = True,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Create a logger with a stable file under logs/automation."""

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    target_dir = Path(log_dir) if log_dir is not None else AUTOMATION_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(cast(int, level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_viecomrec_utils_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(target_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(cast(int, level))
    setattr(file_handler, "_viecomrec_utils_handler", True)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(cast(int, level))
        setattr(console_handler, "_viecomrec_utils_handler", True)
        logger.addHandler(console_handler)

    return logger


def retry(
    func: Optional[F] = None,
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    backoff_factor: Optional[float] = None,
    exceptions: Union[type[BaseException], Tuple[type[BaseException], ...]] = Exception,
) -> Union[F, Callable[[F], F]]:
    """Retry a function with exponential backoff.

    Supports both ``@retry`` and ``@retry(max_attempts=3, backoff_factor=2.0)``.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    factor = backoff_factor if backoff_factor is not None else backoff

    def decorator(inner: F) -> F:
        @functools.wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sleep_seconds = delay
            last_error: Optional[BaseException] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return inner(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt >= max_attempts:
                        raise
                    logging.getLogger(inner.__module__).warning(
                        "Retrying %s after attempt %d/%d failed: %s",
                        inner.__name__,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    time.sleep(sleep_seconds)
                    sleep_seconds *= factor

            if last_error is not None:
                raise last_error
            raise RuntimeError("retry wrapper reached unreachable state")

        return cast(F, wrapper)

    if func is not None:
        return decorator(func)
    return decorator


class PipelineLock:
    """Simple non-blocking file lock for scheduled pipeline jobs."""

    def __init__(
        self,
        name: str,
        timeout: float = 24 * 60 * 60,
        lock_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.name = name
        self.timeout = timeout
        self.lock_dir = Path(lock_dir) if lock_dir is not None else LOCK_DIR
        self.path = self.lock_dir / f"{name}.lock"
        self.acquired = False
        self._fd: Optional[int] = None

    def __enter__(self) -> "PipelineLock":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._remove_stale_lock()

        try:
            self._fd = os.open(
                str(self.path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            payload = {
                "name": self.name,
                "pid": os.getpid(),
                "created_at": _iso_now(),
            }
            os.write(
                self._fd,
                json.dumps(payload, default=_json_default).encode("utf-8"),
            )
            self.acquired = True
        except FileExistsError:
            self.acquired = False

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            finally:
                self.acquired = False

    def _remove_stale_lock(self) -> None:
        if not self.path.exists() or self.timeout <= 0:
            return

        lock_pid = self._read_lock_pid()
        if lock_pid is not None and not self._is_pid_running(lock_pid):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return

        age_seconds = time.time() - self.path.stat().st_mtime
        if age_seconds > self.timeout:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _read_lock_pid(self) -> Optional[int]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = payload.get("pid")
            return int(pid) if pid is not None else None
        except Exception:
            return None

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        if pid <= 0:
            return False

        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                return True

            output = result.stdout.strip().lower()
            return bool(output) and "no tasks are running" not in output

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class PipelineTracker:
    """Append-only JSONL tracker for pipeline run lifecycle events."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path is not None else RUNS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        pipeline_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        record = {
            "event": "start",
            "run_id": run_id,
            "pipeline": pipeline_name,
            "pipeline_name": pipeline_name,
            "status": "running",
            "pid": os.getpid(),
            "started_at": _iso_now(),
            "timestamp": _iso_now(),
            "metadata": metadata or {},
        }
        _append_jsonl(self.path, record)
        return run_id

    def complete_run(
        self,
        run_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        result = result or {}
        status = str(result.get("status") or "success")
        _append_jsonl(
            self.path,
            {
                "event": "complete",
                "run_id": run_id,
                "status": status,
                "finished_at": _iso_now(),
                "timestamp": _iso_now(),
                "result": result,
            },
        )

    def fail_run(self, run_id: str, error: Union[str, BaseException]) -> None:
        _append_jsonl(
            self.path,
            {
                "event": "fail",
                "run_id": run_id,
                "status": "failed",
                "finished_at": _iso_now(),
                "timestamp": _iso_now(),
                "error": str(error),
            },
        )

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        cutoff = _utc_now() - timedelta(days=days)
        latest = self._latest_runs(cutoff=cutoff)
        stats_by_pipeline: Dict[str, Dict[str, Any]] = {}

        for run in latest.values():
            pipeline = str(run.get("pipeline") or run.get("pipeline_name") or "unknown")
            status = str(run.get("status") or "unknown").lower()
            pipeline_stats = stats_by_pipeline.setdefault(
                pipeline,
                {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "running": 0,
                    "stale": 0,
                    "skipped": 0,
                    "other": 0,
                    "success_rate": None,
                },
            )

            pipeline_stats["total"] += 1
            if status in {"success", "completed", "complete", "dry_run", "partial", "baseline_created"}:
                pipeline_stats["success"] += 1
            elif status in {"failed", "failure", "error"}:
                pipeline_stats["failed"] += 1
            elif status == "running":
                pipeline_stats["running"] += 1
            elif status == "stale":
                pipeline_stats["stale"] += 1
                pipeline_stats["failed"] += 1
            elif status == "skipped":
                pipeline_stats["skipped"] += 1
            else:
                pipeline_stats["other"] += 1

        for pipeline_stats in stats_by_pipeline.values():
            total_finished = (
                pipeline_stats["success"]
                + pipeline_stats["failed"]
                + pipeline_stats["skipped"]
                + pipeline_stats["other"]
            )
            if total_finished > 0:
                pipeline_stats["success_rate"] = (
                    pipeline_stats["success"] / total_finished
                )

        return {
            "days": days,
            "total_runs": len(latest),
            "stats_by_pipeline": stats_by_pipeline,
        }

    def cleanup_stale_runs(self, max_running_hours: float = 24) -> int:
        cutoff = _utc_now() - timedelta(hours=max_running_hours)
        latest = self._latest_runs()
        stale_runs: List[Tuple[str, str]] = []

        for run_id, run in latest.items():
            if str(run.get("status", "")).lower() != "running":
                continue

            stale_reason: Optional[str] = None
            started_at = _parse_ts(cast(Optional[str], run.get("started_at")))
            if started_at is not None and started_at < cutoff:
                stale_reason = f"Run exceeded {max_running_hours} hours"
            else:
                pid = run.get("pid")
                if pid is not None and not PipelineLock._is_pid_running(int(pid)):
                    stale_reason = f"Run process {pid} is no longer active"

            if stale_reason is None:
                pipeline = str(run.get("pipeline") or run.get("pipeline_name") or "")
                lock_path = LOCK_DIR / f"{pipeline}.lock" if pipeline else None
                if lock_path is not None and not lock_path.exists():
                    stale_reason = "Run has no active pipeline lock"

            if stale_reason is not None:
                stale_runs.append((run_id, stale_reason))

        for run_id, stale_reason in stale_runs:
            _append_jsonl(
                self.path,
                {
                    "event": "stale",
                    "run_id": run_id,
                    "status": "stale",
                    "finished_at": _iso_now(),
                    "timestamp": _iso_now(),
                    "error": stale_reason,
                },
            )

        return len(stale_runs)

    def _latest_runs(
        self,
        cutoff: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}

        if not self.path.exists():
            return latest

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                run_id = record.get("run_id")
                if not run_id:
                    continue

                timestamp = _parse_ts(
                    record.get("timestamp")
                    or record.get("started_at")
                    or record.get("finished_at")
                )
                if cutoff is not None and timestamp is not None and timestamp < cutoff:
                    continue

                current = latest.get(run_id)
                if current is None:
                    latest[run_id] = record
                    continue

                if record.get("pipeline") is None and current.get("pipeline") is not None:
                    record["pipeline"] = current.get("pipeline")
                if record.get("pipeline_name") is None and current.get("pipeline_name") is not None:
                    record["pipeline_name"] = current.get("pipeline_name")
                if record.get("started_at") is None and current.get("started_at") is not None:
                    record["started_at"] = current.get("started_at")
                if record.get("metadata") is None and current.get("metadata") is not None:
                    record["metadata"] = current.get("metadata")

                latest[run_id] = record

        return latest


def send_pipeline_alert(
    pipeline: str,
    status: str,
    message: str,
    *,
    severity: str = "info",
    metadata: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Record a pipeline alert locally.

    This implementation deliberately does not require network access. External
    alert integrations can tail ``logs/automation/alerts.jsonl``.
    """

    payload = {
        "timestamp": _iso_now(),
        "pipeline": pipeline,
        "status": status,
        "severity": severity,
        "message": message,
        "metadata": metadata or {},
    }
    payload.update(extra)
    _append_jsonl(ALERTS_PATH, payload)
    logging.getLogger(pipeline).log(
        logging.ERROR if severity in {"error", "critical"} else logging.INFO,
        "Pipeline alert [%s/%s]: %s",
        status,
        severity,
        message,
    )
    return payload


def get_git_commit(repo_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Return the current git commit hash, or None outside a git checkout."""

    cwd = Path(repo_path) if repo_path is not None else PROJECT_DIR
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def compute_data_hash(
    root_or_files: Union[str, Path, Sequence[Union[str, Path]]],
    files: Optional[Iterable[Union[str, Path]]] = None,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute a deterministic SHA-256 hash for files or a directory subset."""

    paths = _resolve_hash_paths(root_or_files, files)
    digest = hashlib.sha256()

    for label, path in paths:
        digest.update(label.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")

        if not path.exists():
            digest.update(b"MISSING")
            digest.update(b"\0")
            continue

        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")

    return digest.hexdigest()


def _resolve_hash_paths(
    root_or_files: Union[str, Path, Sequence[Union[str, Path]]],
    files: Optional[Iterable[Union[str, Path]]],
) -> List[Tuple[str, Path]]:
    if files is not None:
        root = Path(cast(Union[str, Path], root_or_files))
        return sorted(
            ((str(file), root / Path(file)) for file in files),
            key=lambda item: item[0],
        )

    if isinstance(root_or_files, (str, Path)):
        root = Path(root_or_files)
        if root.is_dir():
            return sorted(
                (
                    (str(path.relative_to(root)), path)
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                key=lambda item: item[0],
            )
        return [(root.name, root)]

    return sorted(
        ((str(Path(path)), Path(path)) for path in root_or_files),
        key=lambda item: item[0],
    )


__all__ = [
    "PipelineLock",
    "PipelineTracker",
    "compute_data_hash",
    "get_git_commit",
    "retry",
    "send_pipeline_alert",
    "setup_logging",
]

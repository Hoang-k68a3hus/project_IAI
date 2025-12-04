"""
Log and Artifact Cleanup Pipeline.

This module contains the implementation of the cleanup pipeline and can be
run via `python -m automation.cleanup`.
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (  # type: ignore
    PipelineTracker,
    PipelineLock,
    setup_logging,
    send_pipeline_alert,
)


# =============================================================================
# Configuration
# =============================================================================

CLEANUP_CONFIG = {
    "log_dirs": [
        PROJECT_ROOT / "logs" / "pipelines",
        PROJECT_ROOT / "logs" / "cf",
        PROJECT_ROOT / "logs" / "service",
    ],
    "checkpoint_dirs": [
        PROJECT_ROOT / "checkpoints" / "als",
        PROJECT_ROOT / "checkpoints" / "bpr",
    ],
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    "registry_path": PROJECT_ROOT / "artifacts" / "cf" / "registry.json",
    "db_paths": [
        PROJECT_ROOT / "logs" / "training_metrics.db",
        PROJECT_ROOT / "logs" / "service_metrics.db",
        PROJECT_ROOT / "logs" / "pipeline_metrics.db",
    ],
    # Defaults
    "default_log_retention_days": 30,
    "default_checkpoint_retention_days": 7,
    "default_keep_models": 5,
    # Patterns
    "log_patterns": ["*.log", "*.log.*"],
    "checkpoint_patterns": ["*.npy", "*.json", "*.pkl"],
}


# =============================================================================
# Cleanup Functions
# =============================================================================

def cleanup_old_logs(
    retention_days: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Clean up old log files.

    Returns:
        Cleanup statistics
    """
    result: Dict[str, Any] = {
        "type": "logs",
        "files_found": 0,
        "files_deleted": 0,
        "bytes_freed": 0,
        "errors": [],
    }

    cutoff = datetime.now() - timedelta(days=retention_days)

    for log_dir in CLEANUP_CONFIG["log_dirs"]:
        if not log_dir.exists():
            continue

        logger.info("Scanning %s...", log_dir)

        for pattern in CLEANUP_CONFIG["log_patterns"]:
            for file_path in log_dir.glob(pattern):
                if not file_path.is_file():
                    continue

                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if mtime < cutoff:
                    result["files_found"] += 1
                    file_size = file_path.stat().st_size

                    if dry_run:
                        logger.info(
                            "  Would delete: %s (%.1f KB)",
                            file_path,
                            file_size / 1024,
                        )
                    else:
                        try:
                            file_path.unlink()
                            result["files_deleted"] += 1
                            result["bytes_freed"] += file_size
                            logger.info("  Deleted: %s", file_path)
                        except Exception as e:
                            result["errors"].append(f"{file_path}: {e}")
                            logger.error("  Failed to delete %s: %s", file_path, e)

    return result


def cleanup_checkpoints(
    retention_days: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Clean up old checkpoint files.

    Returns:
        Cleanup statistics
    """
    result: Dict[str, Any] = {
        "type": "checkpoints",
        "dirs_found": 0,
        "dirs_deleted": 0,
        "bytes_freed": 0,
        "errors": [],
    }

    cutoff = datetime.now() - timedelta(days=retention_days)

    for checkpoint_dir in CLEANUP_CONFIG["checkpoint_dirs"]:
        if not checkpoint_dir.exists():
            continue

        logger.info("Scanning %s...", checkpoint_dir)

        # Look for timestamped subdirectories
        for subdir in checkpoint_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Check directory age
            mtime = datetime.fromtimestamp(subdir.stat().st_mtime)

            if mtime < cutoff:
                result["dirs_found"] += 1

                # Calculate size
                dir_size = sum(
                    f.stat().st_size
                    for f in subdir.rglob("*")
                    if f.is_file()
                )

                if dry_run:
                    logger.info(
                        "  Would delete: %s (%.1f MB)",
                        subdir,
                        dir_size / 1024 / 1024,
                    )
                else:
                    try:
                        shutil.rmtree(subdir)
                        result["dirs_deleted"] += 1
                        result["bytes_freed"] += dir_size
                        logger.info("  Deleted: %s", subdir)
                    except Exception as e:
                        result["errors"].append(f"{subdir}: {e}")
                        logger.error("  Failed to delete %s: %s", subdir, e)

    return result


def cleanup_old_models(
    keep_count: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Clean up old model versions, keeping the most recent N.

    Returns:
        Cleanup statistics
    """
    result: Dict[str, Any] = {
        "type": "models",
        "models_found": 0,
        "models_deleted": 0,
        "bytes_freed": 0,
        "kept_models": [],
        "errors": [],
    }

    registry_path = CLEANUP_CONFIG["registry_path"]

    if not registry_path.exists():
        logger.warning("Registry not found, skipping model cleanup")
        return result

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except Exception as e:
        result["errors"].append(f"Failed to load registry: {e}")
        return result

    models_data = registry.get("models", {})
    current_best_data = registry.get("current_best")

    # Get current_best model_id
    if isinstance(current_best_data, dict):
        current_best = current_best_data.get("model_id")
    else:
        current_best = current_best_data

    # Convert to list of tuples for sorting
    if isinstance(models_data, dict):
        models_list = [
            (model_id, model_info)
            for model_id, model_info in models_data.items()
        ]
    else:
        # Old list format
        models_list = [(m.get("model_id"), m) for m in models_data]

    # Sort by creation date (newest first)
    sorted_models = sorted(
        models_list,
        key=lambda x: x[1].get("created_at", x[1].get("registered_at", "")),
        reverse=True,
    )

    # Group by model type
    models_by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for model_id, model_info in sorted_models:
        model_type = model_info.get("model_type", "unknown")
        models_by_type.setdefault(model_type, []).append((model_id, model_info))

    # For each type, keep the newest N models
    models_to_delete: List[Tuple[str, Dict[str, Any]]] = []
    models_to_keep: Dict[str, Dict[str, Any]] = {}

    for model_type, type_models in models_by_type.items():
        logger.info("\nProcessing %s models (%d total)...", model_type, len(type_models))

        for i, (model_id, model_info) in enumerate(type_models):
            # Always keep current_best
            if model_id == current_best:
                models_to_keep[model_id] = model_info
                logger.info("  Keeping (current_best): %s", model_id)
                continue

            # Keep the newest N for each type
            if i < keep_count:
                models_to_keep[model_id] = model_info
                logger.info("  Keeping (recent): %s", model_id)
            else:
                models_to_delete.append((model_id, model_info))
                result["models_found"] += 1

    result["kept_models"] = list(models_to_keep.keys())

    # Delete old models
    for model_id, model_info in models_to_delete:
        model_path = Path(model_info.get("path", ""))

        # Handle relative paths
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        if not model_path.exists():
            logger.info("  Already deleted: %s", model_id)
            continue

        # Calculate size
        dir_size = sum(
            f.stat().st_size
            for f in model_path.rglob("*")
            if f.is_file()
        )

        if dry_run:
            logger.info(
                "  Would delete: %s (%.1f MB)",
                model_id,
                dir_size / 1024 / 1024,
            )
        else:
            try:
                shutil.rmtree(model_path)
                result["models_deleted"] += 1
                result["bytes_freed"] += dir_size
                logger.info("  Deleted: %s", model_id)
            except Exception as e:
                result["errors"].append(f"{model_id}: {e}")
                logger.error("  Failed to delete %s: %s", model_id, e)

    # Update registry (remove deleted models)
    if not dry_run and result["models_deleted"] > 0:
        registry["models"] = models_to_keep
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info(
            "\nUpdated registry: %d models remaining",
            len(models_to_keep),
        )

    return result


def vacuum_databases(
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Vacuum SQLite databases to reclaim space.

    Returns:
        Cleanup statistics
    """
    import sqlite3

    result: Dict[str, Any] = {
        "type": "databases",
        "dbs_processed": 0,
        "bytes_freed": 0,
        "errors": [],
    }

    for db_path in CLEANUP_CONFIG["db_paths"]:
        if not db_path.exists():
            continue

        original_size = db_path.stat().st_size

        if dry_run:
            logger.info(
                "Would vacuum: %s (%.1f KB)",
                db_path,
                original_size / 1024,
            )
        else:
            try:
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")
                conn.close()

                new_size = db_path.stat().st_size
                freed = original_size - new_size

                result["dbs_processed"] += 1
                result["bytes_freed"] += max(0, freed)

                logger.info(
                    "Vacuumed: %s (freed %.1f KB)",
                    db_path,
                    freed / 1024,
                )

            except Exception as e:
                result["errors"].append(f"{db_path}: {e}")
                logger.error("Failed to vacuum %s: %s", db_path, e)

    return result


def cleanup_empty_dirs(
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Remove empty directories.

    Returns:
        Cleanup statistics
    """
    result: Dict[str, Any] = {
        "type": "empty_dirs",
        "dirs_found": 0,
        "dirs_deleted": 0,
        "errors": [],
    }

    dirs_to_check = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "artifacts",
    ]

    for base_dir in dirs_to_check:
        if not base_dir.exists():
            continue

        # Walk bottom-up to delete empty subdirectories first
        for dirpath in sorted(base_dir.rglob("*"), reverse=True):
            if not dirpath.is_dir():
                continue

            # Check if empty
            try:
                contents = list(dirpath.iterdir())
                if len(contents) == 0:
                    result["dirs_found"] += 1

                    if dry_run:
                        logger.info("Would remove empty: %s", dirpath)
                    else:
                        dirpath.rmdir()
                        result["dirs_deleted"] += 1
                        logger.info("Removed empty: %s", dirpath)
            except Exception as e:
                result["errors"].append(f"{dirpath}: {e}")

    return result


# =============================================================================
# Main Cleanup Pipeline
# =============================================================================

def run_cleanup(
    log_retention_days: int = 30,
    checkpoint_retention_days: int = 7,
    keep_models: int = 5,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive cleanup.

    Args:
        log_retention_days: Keep logs newer than this
        checkpoint_retention_days: Keep checkpoints newer than this
        keep_models: Number of old model versions to keep
        dry_run: Show what would be deleted without deleting
        logger: Logger instance

    Returns:
        Cleanup results
    """
    if logger is None:
        logger = setup_logging("cleanup")

    tracker = PipelineTracker()

    result: Dict[str, Any] = {
        "pipeline": "cleanup",
        "started_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "config": {
            "log_retention_days": log_retention_days,
            "checkpoint_retention_days": checkpoint_retention_days,
            "keep_models": keep_models,
        },
        "results": {},
    }

    with PipelineLock("cleanup") as lock:
        if not lock.acquired:
            msg = "Cleanup already running"
            logger.warning(msg)
            result["status"] = "skipped"
            result["message"] = msg
            return result

        run_id = tracker.start_run("cleanup", result["config"])

        try:
            total_bytes_freed = 0
            total_errors: List[str] = []

            # Clean logs
            logger.info("\n%s", "=" * 60)
            logger.info("Cleaning old logs...")
            logs_result = cleanup_old_logs(log_retention_days, dry_run, logger)
            result["results"]["logs"] = logs_result
            total_bytes_freed += logs_result["bytes_freed"]
            total_errors.extend(logs_result.get("errors", []))

            # Clean checkpoints
            logger.info("\n%s", "=" * 60)
            logger.info("Cleaning old checkpoints...")
            checkpoints_result = cleanup_checkpoints(
                checkpoint_retention_days,
                dry_run,
                logger,
            )
            result["results"]["checkpoints"] = checkpoints_result
            total_bytes_freed += checkpoints_result["bytes_freed"]
            total_errors.extend(checkpoints_result.get("errors", []))

            # Clean old models
            logger.info("\n%s", "=" * 60)
            logger.info("Cleaning old model versions...")
            models_result = cleanup_old_models(keep_models, dry_run, logger)
            result["results"]["models"] = models_result
            total_bytes_freed += models_result["bytes_freed"]
            total_errors.extend(models_result.get("errors", []))

            # Vacuum databases
            logger.info("\n%s", "=" * 60)
            logger.info("Vacuuming databases...")
            db_result = vacuum_databases(dry_run, logger)
            result["results"]["databases"] = db_result
            total_bytes_freed += db_result["bytes_freed"]
            total_errors.extend(db_result.get("errors", []))

            # Clean empty directories
            logger.info("\n%s", "=" * 60)
            logger.info("Cleaning empty directories...")
            dirs_result = cleanup_empty_dirs(dry_run, logger)
            result["results"]["empty_dirs"] = dirs_result
            total_errors.extend(dirs_result.get("errors", []))

            # Summary
            result["total_bytes_freed"] = total_bytes_freed
            result["total_errors"] = len(total_errors)
            result["errors"] = total_errors
            result["status"] = "success" if len(total_errors) == 0 else "partial"
            result["finished_at"] = datetime.now().isoformat()

            tracker.complete_run(
                run_id,
                {
                    "status": result["status"],
                    "bytes_freed": total_bytes_freed,
                    "errors": len(total_errors),
                },
            )

            logger.info(
                "\nCleanup %s: %.2f MB",
                "would free" if dry_run else "freed",
                total_bytes_freed / 1024 / 1024,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Cleanup failed: %s", error_msg)

            result["status"] = "failed"
            result["error"] = error_msg

            tracker.fail_run(run_id, error_msg)

            raise

    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point for cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up old logs and artifacts",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=30,
        help="Keep logs newer than N days (default: 30)",
    )
    parser.add_argument(
        "--checkpoint-days",
        type=int,
        default=7,
        help="Keep checkpoints newer than N days (default: 7)",
    )
    parser.add_argument(
        "--keep-models",
        "-k",
        type=int,
        default=5,
        help="Number of old model versions to keep per type (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("cleanup", level=level)

    try:
        result = run_cleanup(
            log_retention_days=args.days,
            checkpoint_retention_days=args.checkpoint_days,
            keep_models=args.keep_models,
            dry_run=args.dry_run,
            logger=logger,
        )

        print(f"\n{'=' * 60}")
        print(f"Cleanup Result: {result['status'].upper()}")
        print(f"{'=' * 60}")

        if args.dry_run:
            print("  (Dry run - no changes made)")

        print(
            "\n  Space %s: %.2f MB"
            % (
                "would be freed" if args.dry_run else "freed",
                result["total_bytes_freed"] / 1024 / 1024,
            )
        )

        for category, data in result.get("results", {}).items():
            if isinstance(data, Dict):
                deleted = (
                    data.get("files_deleted", 0)
                    or data.get("dirs_deleted", 0)
                    or data.get("models_deleted", 0)
                )
                found = (
                    data.get("files_found", 0)
                    or data.get("dirs_found", 0)
                    or data.get("models_found", 0)
                )
                if found > 0 or deleted > 0:
                    print(f"  {category}: {deleted}/{found} items")

        if result.get("total_errors", 0) > 0:
            print(f"\n  Errors: {result['total_errors']}")

        sys.exit(0 if result["status"] == "success" else 1)

    except Exception as e:  # pragma: no cover - CLI guard
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()



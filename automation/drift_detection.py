"""
Drift Detection Pipeline.

This module detects data drift and model drift.
Run via: python -m automation.drift_detection
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    PipelineTracker,
    PipelineLock,
    setup_logging,
    send_pipeline_alert,
    get_git_commit,
)


# =============================================================================
# Configuration
# =============================================================================

DRIFT_CONFIG = {
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    "reports_dir": PROJECT_ROOT / "reports" / "drift",
    # Thresholds
    "rating_dist_threshold": 0.1,  # Max change in rating distribution
    "popularity_shift_threshold": 0.2,  # Max shift in top items
    "interaction_rate_threshold": 0.3,  # Max change in interaction rate
}


# =============================================================================
# Drift Detection Functions
# =============================================================================

def load_current_stats(logger: logging.Logger) -> Dict[str, Any]:
    """Load current data statistics."""
    stats_file = DRIFT_CONFIG["processed_dir"] / "data_stats.json"

    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_stats(logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Load baseline statistics for comparison."""
    reports_dir = DRIFT_CONFIG["reports_dir"]
    baseline_file = reports_dir / "baseline_stats.json"

    if not baseline_file.exists():
        logger.warning("No baseline stats found, will create new baseline")
        return None

    with open(baseline_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline_stats(stats: Dict[str, Any], logger: logging.Logger) -> None:
    """Save current stats as baseline."""
    reports_dir = DRIFT_CONFIG["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = reports_dir / "baseline_stats.json"
    stats["baseline_created_at"] = datetime.now().isoformat()

    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved baseline stats to {baseline_file}")


def detect_rating_drift(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Detect drift in rating distribution."""
    result = {
        "metric": "rating_distribution",
        "drift_detected": False,
        "details": {},
    }

    # Compare rating distributions
    current_dist = current.get("rating_distribution", {})
    baseline_dist = baseline.get("rating_distribution", {})

    if not current_dist or not baseline_dist:
        result["status"] = "no_data"
        return result

    # Calculate total difference
    total_diff = 0.0
    for rating in ["1", "2", "3", "4", "5"]:
        curr_pct = current_dist.get(rating, 0)
        base_pct = baseline_dist.get(rating, 0)
        diff = abs(curr_pct - base_pct)
        total_diff += diff
        result["details"][f"rating_{rating}_diff"] = diff

    result["total_difference"] = total_diff
    result["threshold"] = DRIFT_CONFIG["rating_dist_threshold"]

    if total_diff > DRIFT_CONFIG["rating_dist_threshold"]:
        result["drift_detected"] = True
        logger.warning(
            f"Rating distribution drift detected: {total_diff:.3f} > "
            f"{DRIFT_CONFIG['rating_dist_threshold']}"
        )
    else:
        logger.info(f"Rating distribution stable: {total_diff:.3f}")

    return result


def detect_popularity_drift(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Detect drift in item popularity."""
    result = {
        "metric": "popularity_distribution",
        "drift_detected": False,
        "details": {},
    }

    # Compare top items
    current_top = set(current.get("top_items", [])[:20])
    baseline_top = set(baseline.get("top_items", [])[:20])

    if not current_top or not baseline_top:
        result["status"] = "no_data"
        return result

    # Calculate Jaccard similarity
    intersection = len(current_top & baseline_top)
    union = len(current_top | baseline_top)
    similarity = intersection / union if union > 0 else 1.0
    shift = 1.0 - similarity

    result["jaccard_similarity"] = similarity
    result["shift"] = shift
    result["threshold"] = DRIFT_CONFIG["popularity_shift_threshold"]
    result["details"]["new_items"] = list(current_top - baseline_top)
    result["details"]["dropped_items"] = list(baseline_top - current_top)

    if shift > DRIFT_CONFIG["popularity_shift_threshold"]:
        result["drift_detected"] = True
        logger.warning(
            f"Popularity drift detected: shift={shift:.3f} > "
            f"{DRIFT_CONFIG['popularity_shift_threshold']}"
        )
    else:
        logger.info(f"Popularity stable: shift={shift:.3f}")

    return result


def detect_interaction_drift(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Detect drift in interaction patterns."""
    result = {
        "metric": "interaction_rate",
        "drift_detected": False,
        "details": {},
    }

    # Compare interaction rates - handle nested structure
    current_rate = (
        current.get("avg_interactions_per_user") or
        current.get("trainable_users", {}).get("avg_interactions_per_trainable_user", 0)
    )
    baseline_rate = (
        baseline.get("avg_interactions_per_user") or
        baseline.get("trainable_users", {}).get("avg_interactions_per_trainable_user", 0)
    )

    if baseline_rate == 0:
        result["status"] = "no_baseline"
        return result

    change_rate = abs(current_rate - baseline_rate) / baseline_rate

    result["current_rate"] = current_rate
    result["baseline_rate"] = baseline_rate
    result["change_rate"] = change_rate
    result["threshold"] = DRIFT_CONFIG["interaction_rate_threshold"]

    if change_rate > DRIFT_CONFIG["interaction_rate_threshold"]:
        result["drift_detected"] = True
        logger.warning(
            f"Interaction rate drift detected: change={change_rate:.3f} > "
            f"{DRIFT_CONFIG['interaction_rate_threshold']}"
        )
    else:
        logger.info(f"Interaction rate stable: change={change_rate:.3f}")

    return result


def generate_drift_report(
    drift_results: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Path:
    """Generate drift detection report."""
    reports_dir = DRIFT_CONFIG["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"drift_report_{timestamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "drift_detected": any(r.get("drift_detected", False) for r in drift_results),
        "results": drift_results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved drift report to {report_file}")
    return report_file


# =============================================================================
# Main Pipeline
# =============================================================================

def detect_drift(
    update_baseline: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Main drift detection pipeline.

    Args:
        update_baseline: Update baseline with current stats
        logger: Logger instance

    Returns:
        Pipeline result
    """
    if logger is None:
        logger = setup_logging("drift_detection")

    tracker = PipelineTracker()
    result: Dict[str, Any] = {
        "pipeline": "drift_detection",
        "started_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }

    with PipelineLock("drift_detection") as lock:
        if not lock.acquired:
            msg = "Drift detection already running"
            logger.warning(msg)
            result["status"] = "skipped"
            result["message"] = msg
            return result

        run_id = tracker.start_run(
            "drift_detection",
            {"update_baseline": update_baseline},
        )

        try:
            # Load current stats
            logger.info("Loading current statistics...")
            current_stats = load_current_stats(logger)

            # Load baseline
            baseline_stats = load_baseline_stats(logger)

            if baseline_stats is None:
                # No baseline - create one
                logger.info("Creating initial baseline...")
                save_baseline_stats(current_stats, logger)
                result["status"] = "baseline_created"
                result["message"] = "Initial baseline created, no comparison available"
                tracker.complete_run(run_id, {"status": "baseline_created"})
                return result

            # Run drift detection
            logger.info("Running drift detection...")
            drift_results = []

            # Rating distribution drift
            rating_drift = detect_rating_drift(current_stats, baseline_stats, logger)
            drift_results.append(rating_drift)

            # Popularity drift
            popularity_drift = detect_popularity_drift(
                current_stats, baseline_stats, logger
            )
            drift_results.append(popularity_drift)

            # Interaction rate drift
            interaction_drift = detect_interaction_drift(
                current_stats, baseline_stats, logger
            )
            drift_results.append(interaction_drift)

            # Generate report
            report_path = generate_drift_report(drift_results, logger)

            # Check if any drift detected
            any_drift = any(r.get("drift_detected", False) for r in drift_results)

            # Update baseline if requested
            if update_baseline:
                save_baseline_stats(current_stats, logger)

            # Success
            result["status"] = "success"
            result["finished_at"] = datetime.now().isoformat()
            result["drift_detected"] = any_drift
            result["drift_results"] = drift_results
            result["report_file"] = str(report_path)

            tracker.complete_run(
                run_id,
                {
                    "status": "success",
                    "drift_detected": any_drift,
                },
            )

            if any_drift:
                logger.warning("⚠ Drift detected! Check report for details.")
                send_pipeline_alert(
                    "drift_detection",
                    "warning",
                    "Data drift detected - review recommended",
                    severity="warning",
                )
            else:
                logger.info("✓ No significant drift detected")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Drift detection failed: {error_msg}")

            result["status"] = "failed"
            result["error"] = error_msg

            tracker.fail_run(run_id, error_msg)

            send_pipeline_alert(
                "drift_detection",
                "failed",
                f"Drift detection failed: {error_msg}",
                severity="error",
            )

            raise

    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect data and model drift",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current stats",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("drift_detection", level=level)

    try:
        result = detect_drift(
            update_baseline=args.update_baseline,
            logger=logger,
        )

        print(f"\n{'=' * 60}")
        print(f"Drift Detection: {result['status'].upper()}")
        print(f"{'=' * 60}")

        if result["status"] == "success":
            drift_status = "⚠ YES" if result["drift_detected"] else "✓ NO"
            print(f"  Drift Detected: {drift_status}")
            print(f"  Report: {result.get('report_file', 'N/A')}")

            for dr in result.get("drift_results", []):
                status = "⚠" if dr.get("drift_detected") else "✓"
                print(f"  {status} {dr['metric']}")

        elif result.get("message"):
            print(f"  Message: {result['message']}")

        sys.exit(0 if result["status"] in ("success", "skipped", "baseline_created") else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

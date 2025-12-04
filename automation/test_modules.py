"""
Test script to verify automation modules work correctly.

Run: python -m automation.test_modules
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all automation modules can be imported."""
    print("=" * 60)
    print("Testing Automation Module Imports")
    print("=" * 60)
    
    modules = [
        ("automation.scheduler", "main"),
        ("automation.data_refresh", "main"),
        ("automation.model_training", "main"),
        ("automation.model_deployment", "main"),
        ("automation.health_check", "main"),
        ("automation.bert_embeddings", "main"),
        ("automation.drift_detection", "main"),
        ("automation.cleanup", "main"),
    ]
    
    all_passed = True
    
    for module_name, func_name in modules:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name, None)
            if func:
                print(f"  ✓ {module_name}.{func_name}()")
            else:
                print(f"  ⚠ {module_name} - no {func_name}() function")
        except Exception as e:
            print(f"  ✗ {module_name} - {type(e).__name__}: {e}")
            all_passed = False
    
    return all_passed


def test_utils_imports():
    """Test that scripts.utils can be imported."""
    print("\n" + "=" * 60)
    print("Testing scripts.utils Imports")
    print("=" * 60)
    
    try:
        from scripts.utils import (
            retry,
            PipelineTracker,
            PipelineLock,
            setup_logging,
            send_pipeline_alert,
            get_git_commit,
            compute_data_hash,
        )
        print("  ✓ All required utilities imported successfully")
        
        # Test setup_logging
        logger = setup_logging("test", console=False)
        print(f"  ✓ setup_logging() works - logger: {logger.name}")
        
        # Test get_git_commit
        commit = get_git_commit()
        print(f"  ✓ get_git_commit() works - commit: {commit[:8] if commit else 'None'}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {type(e).__name__}: {e}")
        return False


def test_scheduler_config():
    """Test scheduler configuration is valid."""
    print("\n" + "=" * 60)
    print("Testing Scheduler Configuration")
    print("=" * 60)
    
    try:
        from automation.scheduler import SCHEDULER_CONFIG, PROJECT_DIR, LOG_DIR
        
        print(f"  Project Dir: {PROJECT_DIR}")
        print(f"  Log Dir: {LOG_DIR}")
        print(f"  Tasks configured: {len(SCHEDULER_CONFIG)}")
        
        for task_name, config in SCHEDULER_CONFIG.items():
            enabled = "✓" if config.get("enabled") else "○"
            module = config.get("module", "N/A")
            schedule = config.get("schedule", {})
            print(f"    {enabled} {task_name}: {module}")
            print(f"        Schedule: {schedule}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config check failed: {type(e).__name__}: {e}")
        return False


def test_apscheduler():
    """Test APScheduler can be imported and configured."""
    print("\n" + "=" * 60)
    print("Testing APScheduler")
    print("=" * 60)
    
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        import pytz
        
        print("  ✓ APScheduler imported successfully")
        
        # Test creating a scheduler
        scheduler = BlockingScheduler(timezone=pytz.UTC)
        print("  ✓ BlockingScheduler created")
        
        # Test creating a trigger
        trigger = CronTrigger(hour=2, minute=0, timezone=pytz.UTC)
        print(f"  ✓ CronTrigger created: {trigger}")
        
        return True
    except ImportError as e:
        print(f"  ✗ APScheduler not installed: {e}")
        print("  → Run: pip install apscheduler pytz")
        return False
    except Exception as e:
        print(f"  ✗ APScheduler test failed: {type(e).__name__}: {e}")
        return False


def test_torch():
    """Test PyTorch for BERT embeddings."""
    print("\n" + "=" * 60)
    print("Testing PyTorch (for BERT embeddings)")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            print("  ✓ Transformers library available")
        except ImportError:
            print("  ⚠ Transformers not installed (needed for BERT embeddings)")
            print("  → Run: pip install transformers")
        
        return True
    except ImportError as e:
        print(f"  ⚠ PyTorch not installed: {e}")
        print("  → BERT embeddings task may fail")
        return True  # Not critical for other tasks


def test_data_files():
    """Test that required data files exist."""
    print("\n" + "=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    required_files = [
        ("data/published_data/data_reviews_purchase.csv", True),
        ("data/published_data/data_product.csv", True),
        ("data/processed/user_item_mappings.json", False),
        ("data/processed/X_train_confidence.npz", False),
        ("artifacts/cf/registry.json", False),
        ("config/serving_config.yaml", False),
    ]
    
    all_ok = True
    for rel_path, required in required_files:
        full_path = PROJECT_ROOT / rel_path
        exists = full_path.exists()
        
        if exists:
            print(f"  ✓ {rel_path}")
        elif required:
            print(f"  ✗ {rel_path} (REQUIRED - MISSING)")
            all_ok = False
        else:
            print(f"  ○ {rel_path} (optional - not found)")
    
    return all_ok


def test_run_simple_task():
    """Test running a simple task (health_check in dry mode)."""
    print("\n" + "=" * 60)
    print("Testing Task Execution (health_check)")
    print("=" * 60)
    
    try:
        from automation.health_check import check_data_health
        from scripts.utils import setup_logging
        
        logger = setup_logging("test_health", console=False)
        
        # Run data health check
        result = check_data_health(logger)
        
        print(f"  ✓ health_check.check_data_health() executed")
        print(f"    Status: {result.get('status', 'unknown')}")
        print(f"    Checks: {len(result.get('checks', []))}")
        
        for check in result.get("checks", [])[:3]:
            icon = "✓" if check.get("passed") else "✗"
            print(f"      {icon} {check.get('name')}: {check.get('message', '')[:50]}")
        
        return True
    except Exception as e:
        print(f"  ✗ Task execution failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VIECOMREC AUTOMATION MODULE TESTS")
    print("=" * 60 + "\n")
    
    results = {
        "imports": test_imports(),
        "utils": test_utils_imports(),
        "scheduler_config": test_scheduler_config(),
        "apscheduler": test_apscheduler(),
        "torch": test_torch(),
        "data_files": test_data_files(),
        "task_execution": test_run_simple_task(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        icon = "✓" if result else "✗"
        print(f"  {icon} {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Automation modules are ready.")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

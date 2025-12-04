"""
Automation package for VieComRec.

This package groups all long-running automation tasks such as:
- Data refresh
- Model training
- Model deployment
- System health checks
- BERT embeddings refresh
- Drift detection
- Log and artifact cleanup
- Scheduler

Each task exposes a `main()` function suitable for CLI entry points.
Run any task via: python -m automation.<task_name>
"""

__all__ = [
    "data_refresh",
    "model_training",
    "model_deployment",
    "health_check",
    "bert_embeddings",
    "drift_detection",
    "cleanup",
    "scheduler",
]



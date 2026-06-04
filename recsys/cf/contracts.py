"""
Shared data contracts for VieComRec CF pipelines.

This module is intentionally dependency-light so training, registry, and
serving layers can all use the same schema normalization and artifact checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import os
import tempfile

import numpy as np


REGISTRY_VERSION = "2.0"

MODEL_STATUS_ACTIVE = "active"
MODEL_STATUS_ARCHIVED = "archived"
MODEL_STATUS_FAILED = "failed"
MODEL_STATUSES = {
    MODEL_STATUS_ACTIVE,
    MODEL_STATUS_ARCHIVED,
    MODEL_STATUS_FAILED,
}

PREFERRED_CONTENT_EMBEDDING_DIM = 1024
SUPPORTED_CONTENT_EMBEDDING_DIMS = (768, 1024)


@dataclass(frozen=True)
class ModelArtifactLayout:
    """Canonical model artifact layout: U is users, V is items."""

    model_type: str
    model_dir: Path
    user_factors_file: Path
    item_factors_file: Path
    params_file: Path
    metadata_file: Path


def now_iso() -> str:
    """Return local ISO timestamp for registry metadata."""
    return datetime.now().isoformat()


def create_empty_registry() -> Dict[str, Any]:
    """Create the canonical registry document."""
    return {
        "current_best": None,
        "models": {},
        "bert_embeddings": {},
        "metadata": {
            "registry_version": REGISTRY_VERSION,
            "last_updated": now_iso(),
            "num_models": 0,
            "num_embeddings": 0,
            "selection_criteria": "recall@10",
        },
    }


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON to avoid partially-written registry files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _infer_version(model_id: str, model_type: str, entry: Dict[str, Any]) -> str:
    if entry.get("version"):
        return str(entry["version"])

    prefix = f"{model_type}_"
    if model_id.startswith(prefix):
        return model_id[len(prefix):]

    return model_id


def get_current_best_model_id(registry: Dict[str, Any]) -> Optional[str]:
    """Return current best model_id from either legacy or canonical format."""
    current_best = registry.get("current_best")
    if isinstance(current_best, dict):
        value = current_best.get("model_id")
        return str(value) if value else None
    if current_best:
        return str(current_best)
    return None


def normalize_model_entry(
    model_id: str,
    entry: Dict[str, Any],
    current_best_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one model registry entry while preserving extra fields."""
    normalized = dict(entry)
    model_type = str(normalized.get("model_type", "unknown"))
    created_at = (
        normalized.get("created_at")
        or normalized.get("registered_at")
        or normalized.get("timestamp")
        or now_iso()
    )

    status = normalized.get("status")
    if status not in MODEL_STATUSES:
        status = MODEL_STATUS_ACTIVE

    normalized["model_id"] = str(normalized.get("model_id") or model_id)
    normalized["model_type"] = model_type
    normalized["version"] = _infer_version(normalized["model_id"], model_type, normalized)
    normalized["created_at"] = created_at
    normalized["status"] = status
    normalized["metrics"] = normalized.get("metrics") or {}
    normalized["path"] = str(normalized.get("path", ""))
    normalized["is_active"] = normalized["model_id"] == current_best_id

    return normalized


def normalize_registry(registry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert legacy registry shapes into the canonical dict format.

    Canonical shape:
        current_best: dict or None
        models: dict[model_id, model_entry]
        model_entry.status: active|archived|failed

    Legacy shapes accepted:
        current_best: string
        models: list[entry]
        model_entry.is_active: bool
    """
    if not registry:
        return create_empty_registry()

    normalized = dict(registry)
    models_raw = normalized.get("models") or {}
    current_best_id = get_current_best_model_id(normalized)

    if current_best_id is None:
        if isinstance(models_raw, dict):
            for mid, entry in models_raw.items():
                if isinstance(entry, dict) and entry.get("is_active"):
                    current_best_id = str(entry.get("model_id") or mid)
                    break
        elif isinstance(models_raw, list):
            for entry in models_raw:
                if isinstance(entry, dict) and entry.get("is_active"):
                    current_best_id = str(entry.get("model_id"))
                    break

    models: Dict[str, Dict[str, Any]] = {}
    if isinstance(models_raw, dict):
        iterable = models_raw.items()
    elif isinstance(models_raw, list):
        iterable = (
            (str(entry.get("model_id", f"model_{idx}")), entry)
            for idx, entry in enumerate(models_raw)
            if isinstance(entry, dict)
        )
    else:
        iterable = []

    for raw_id, entry in iterable:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model_id") or raw_id)
        models[model_id] = normalize_model_entry(model_id, entry, current_best_id)

    normalized["models"] = models

    metadata = dict(normalized.get("metadata") or {})
    metadata["registry_version"] = REGISTRY_VERSION
    metadata["num_models"] = len(models)
    metadata.setdefault("num_embeddings", len(normalized.get("bert_embeddings", {})))
    metadata.setdefault("selection_criteria", "recall@10")
    metadata.setdefault("last_updated", now_iso())
    normalized["metadata"] = metadata
    normalized.setdefault("bert_embeddings", {})

    if current_best_id and current_best_id in models:
        best_entry = models[current_best_id]
        current_best = normalized.get("current_best")
        best_payload = dict(current_best) if isinstance(current_best, dict) else {}
        best_payload.update({
            "model_id": current_best_id,
            "model_type": best_entry.get("model_type"),
            "version": best_entry.get("version"),
            "path": best_entry.get("path"),
        })
        best_payload.setdefault("selected_at", best_entry.get("created_at"))
        normalized["current_best"] = best_payload
    else:
        normalized["current_best"] = None

    return normalized


def artifact_layout(model_dir: Path, model_type: str) -> ModelArtifactLayout:
    """Return canonical artifact paths for a model type."""
    model_dir = Path(model_dir)
    return ModelArtifactLayout(
        model_type=model_type,
        model_dir=model_dir,
        user_factors_file=model_dir / f"{model_type}_U.npy",
        item_factors_file=model_dir / f"{model_type}_V.npy",
        params_file=model_dir / f"{model_type}_params.json",
        metadata_file=model_dir / f"{model_type}_metadata.json",
    )


def validate_factor_matrices(
    U: np.ndarray,
    V: np.ndarray,
    expected_num_users: Optional[int] = None,
    expected_num_items: Optional[int] = None,
) -> None:
    """Validate canonical CF matrix layout: U=users, V=items."""
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError(f"U and V must be 2D arrays, got U={U.shape}, V={V.shape}")

    if U.shape[1] != V.shape[1]:
        raise ValueError(f"U/V factor dimension mismatch: U={U.shape}, V={V.shape}")

    if expected_num_users and U.shape[0] != expected_num_users:
        if expected_num_items and U.shape[0] == expected_num_items and V.shape[0] == expected_num_users:
            raise ValueError(
                "Model factors appear swapped. Contract requires "
                f"U=(users,factors), V=(items,factors); got U={U.shape}, V={V.shape}, "
                f"expected users={expected_num_users}, items={expected_num_items}."
            )
        raise ValueError(
            f"User factor row count mismatch: got U={U.shape}, expected users={expected_num_users}"
        )

    if expected_num_items and V.shape[0] != expected_num_items:
        raise ValueError(
            f"Item factor row count mismatch: got V={V.shape}, expected items={expected_num_items}"
        )


def load_model_artifacts(
    model_path: Path,
    model_type: str,
    expected_num_users: Optional[int] = None,
    expected_num_items: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Load canonical model artifacts and validate matrix layout."""
    layout = artifact_layout(model_path, model_type)

    required = [
        layout.user_factors_file,
        layout.item_factors_file,
        layout.params_file,
        layout.metadata_file,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {missing}")

    U = np.load(layout.user_factors_file)
    V = np.load(layout.item_factors_file)

    with open(layout.params_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    with open(layout.metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_num_users = expected_num_users or metadata.get("num_users") or params.get("num_users")
    expected_num_items = expected_num_items or metadata.get("num_items") or params.get("num_items")
    validate_factor_matrices(U, V, expected_num_users, expected_num_items)

    return U, V, params, metadata


def validate_content_embedding_dim(dim: int) -> None:
    """Validate supported content embedding dimensions."""
    if dim not in SUPPORTED_CONTENT_EMBEDDING_DIMS:
        raise ValueError(
            f"Unsupported content embedding dimension: {dim}. "
            f"Supported dimensions: {SUPPORTED_CONTENT_EMBEDDING_DIMS}"
        )

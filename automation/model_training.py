"""
Model Training Pipeline.

This module contains the implementation of the ALS/BPR training pipeline
and exposes a CLI via `python -m automation.model_training`.

Key Features:
- ALS training with confidence-weighted matrix factorization
- BPR training with hard negative sampling
- BERT initialization for cold-start items (optional, using PhoBERT embeddings)
- Checkpointing for crash recovery
- Popularity baseline comparison
- Incremental retraining support (warm start from previous model)
- Early stopping for BPR when validation metric plateaus
- Proper train/validation split for hyperparameter tuning
"""

import os
import sys
import json
import argparse
import logging
import shutil
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import scipy.sparse as sp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (  # type: ignore
    retry,
    PipelineTracker,
    PipelineLock,
    setup_logging,
    send_pipeline_alert,
    get_git_commit,
)


# =============================================================================
# Configuration
# =============================================================================

TRAINING_CONFIG = {
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "artifacts_dir": PROJECT_ROOT / "artifacts" / "cf",
    "checkpoints_dir": PROJECT_ROOT / "checkpoints",
    "registry_path": PROJECT_ROOT / "artifacts" / "cf" / "registry.json",
    "bert_embeddings_path": PROJECT_ROOT / "data" / "processed" / "content_based_embeddings" / "product_embeddings.pt",
    
    # ALS hyperparameters (tuned for sparse data with ≥2 interactions threshold)
    "als": {
        "factors": 64,
        "regularization": 0.1,  # Higher due to sparsity
        "iterations": 15,
        "alpha": 5,  # Confidence weight (adjusted for 1-6 range)
        "use_gpu": False,
        "calculate_training_loss": True,
        "use_bert_init": False,  # Only for cold-start items
        "bert_init_cold_threshold": 5,  # Items with < 5 interactions are "cold"
    },
    # BPR hyperparameters
    "bpr": {
        "factors": 64,
        "learning_rate": 0.05,
        "regularization": 0.0001,
        "epochs": 50,
        "neg_sample_ratio": 0.3,  # 30% hard negatives
        "use_bert_init": False,
        "bert_init_cold_threshold": 5,
    },
    # Evaluation settings
    "eval_k_values": [5, 10, 20],
    "primary_metric": "recall",  # For model selection
    "primary_k": 10,  # Recall@10
    
    # Validation split (for early stopping)
    "validation_ratio": 0.1,  # 10% of training data for validation
    
    # Early stopping (for BPR)
    "early_stopping": {
        "enabled": True,
        "patience": 5,  # Stop if no improvement for 5 epochs
        "min_delta": 0.001,  # Minimum improvement to count as progress
    },
    
    # Checkpointing
    "checkpoint_every_n_iters": 5,  # Save checkpoint every 5 iterations
    "keep_n_checkpoints": 3,  # Keep last 3 checkpoints
    
    # Incremental training
    "incremental": {
        "enabled": True,
        "warmstart": True,  # Warm start from previous model
        "warmstart_iters": 5,  # Fewer iterations for incremental updates
        "trigger_threshold": 100,  # Number of new interactions to trigger incremental
    },
    
    # Popularity baseline
    "compute_baseline": True,  # Compare with popularity baseline
}


# =============================================================================
# Training Functions
# =============================================================================

def load_training_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Load all required training data.

    Returns:
        Dict with loaded data arrays and mappings
    """
    processed_dir = TRAINING_CONFIG["processed_dir"]

    # Load sparse matrices
    X_confidence = sp.load_npz(processed_dir / "X_train_confidence.npz")
    X_binary = sp.load_npz(processed_dir / "X_train_binary.npz")

    # Load mappings
    with open(processed_dir / "user_item_mappings.json", "r") as f:
        mappings = json.load(f)

    # Load user sets
    with open(processed_dir / "user_pos_train.pkl", "rb") as f:
        user_pos_train = pickle.load(f)

    with open(processed_dir / "user_hard_neg_train.pkl", "rb") as f:
        user_hard_neg_train = pickle.load(f)

    # Load stats
    with open(processed_dir / "data_stats.json", "r") as f:
        data_stats = json.load(f)

    logger.info(
        "Loaded training data: "
        f"{X_confidence.shape[0]} users x {X_confidence.shape[1]} items"
    )

    return {
        "X_confidence": X_confidence,
        "X_binary": X_binary,
        "mappings": mappings,
        "user_pos_train": user_pos_train,
        "user_hard_neg_train": user_hard_neg_train,
        "data_stats": data_stats,
        "data_hash": mappings.get("data_hash"),
    }


def load_bert_embeddings(logger: logging.Logger) -> Optional[np.ndarray]:
    """
    Load PhoBERT embeddings for item initialization.
    
    Returns:
        np.ndarray of shape (num_items, 768) or None if not available
    """
    import torch
    
    bert_path = TRAINING_CONFIG["bert_embeddings_path"]
    if not bert_path.exists():
        logger.warning("BERT embeddings not found at %s", bert_path)
        return None
    
    try:
        embeddings = torch.load(bert_path, map_location='cpu')
        if isinstance(embeddings, dict):
            # Handle dict format with product_id keys
            embeddings = embeddings.get('embeddings', embeddings)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        logger.info("Loaded BERT embeddings: shape=%s", embeddings.shape)
        return embeddings
    except Exception as e:
        logger.warning("Failed to load BERT embeddings: %s", e)
        return None


def project_bert_to_factors(
    bert_embeddings: np.ndarray,
    num_factors: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Project BERT embeddings (768-dim) to model factors (e.g., 64-dim) using SVD.
    
    Args:
        bert_embeddings: (num_items, 768) PhoBERT embeddings
        num_factors: Target dimensionality (e.g., 64)
        logger: Logger instance
    
    Returns:
        np.ndarray of shape (num_items, num_factors)
    """
    from sklearn.decomposition import TruncatedSVD
    
    logger.info("Projecting BERT embeddings from %d to %d dimensions...", 
                bert_embeddings.shape[1], num_factors)
    
    svd = TruncatedSVD(n_components=num_factors, random_state=42)
    projected = svd.fit_transform(bert_embeddings)
    
    # Normalize to match typical ALS/BPR initialization scale
    projected = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
    projected *= 0.1  # Scale down to typical init range
    
    logger.info("Projected BERT embeddings: variance explained = %.2f%%",
                svd.explained_variance_ratio_.sum() * 100)
    
    return projected.astype(np.float32)


def get_cold_items(X_train: sp.csr_matrix, threshold: int = 5) -> np.ndarray:
    """
    Identify cold-start items with fewer than threshold interactions.
    
    Args:
        X_train: Training matrix (users x items)
        threshold: Minimum interactions to be considered "warm"
    
    Returns:
        Boolean array indicating cold items
    """
    item_counts = np.array(X_train.sum(axis=0)).flatten()
    cold_mask = item_counts < threshold
    return cold_mask


def create_validation_split(
    user_pos_train: Dict[int, set],
    val_ratio: float = 0.1,
    min_train_items: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[int, set], Dict[int, set]]:
    """
    Create train/validation split from user positive sets.
    
    For each user with ≥2 positives, hold out val_ratio as validation.
    
    Args:
        user_pos_train: Dict mapping user_idx to set of positive item indices
        val_ratio: Fraction to hold out for validation
        min_train_items: Minimum items to keep in training per user
    
    Returns:
        Tuple of (train_sets, val_sets)
    """
    train_sets = {}
    val_sets = {}
    
    for u_idx, items in user_pos_train.items():
        items_list = list(items)
        n_items = len(items_list)
        
        if n_items <= min_train_items:
            # Keep all in training
            train_sets[u_idx] = set(items_list)
            val_sets[u_idx] = set()
        else:
            # Split
            n_val = max(1, int(n_items * val_ratio))
            n_train = n_items - n_val
            
            # Random shuffle for fairness
            np.random.shuffle(items_list)
            
            train_sets[u_idx] = set(items_list[:n_train])
            val_sets[u_idx] = set(items_list[n_train:])
    
    if logger:
        total_train = sum(len(v) for v in train_sets.values())
        total_val = sum(len(v) for v in val_sets.values())
        logger.info("Validation split: %d train, %d validation (%.1f%%)",
                    total_train, total_val, 100 * total_val / (total_train + total_val))
    
    return train_sets, val_sets


def compute_popularity_baseline(
    data: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Compute popularity baseline metrics for comparison.
    
    Strategy: Recommend most popular items (by interaction count) to all users.
    
    Returns:
        Dict with baseline metrics
    """
    X_train = data["X_binary"]
    user_pos_train = data["user_pos_train"]
    
    # Compute item popularity (number of interactions)
    item_popularity = np.array(X_train.sum(axis=0)).flatten()
    
    # Get top-K popular items
    max_k = max(TRAINING_CONFIG["eval_k_values"])
    top_popular_items = np.argsort(item_popularity)[-max_k:][::-1]
    
    logger.info("Computing popularity baseline...")
    
    # Load test data
    processed_dir = TRAINING_CONFIG["processed_dir"]
    try:
        with open(processed_dir / "user_pos_test.pkl", "rb") as f:
            user_pos_test = pickle.load(f)
    except FileNotFoundError:
        # Fallback: use holdout from train
        user_pos_test = {}
        for u_idx, items in user_pos_train.items():
            if len(items) >= 2:
                items_list = list(items)
                user_pos_test[u_idx] = {items_list[-1]}  # Last item as test
    
    # Evaluate baseline
    k_values = TRAINING_CONFIG["eval_k_values"]
    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    
    for u_idx, test_items in user_pos_test.items():
        if not test_items:
            continue
        
        # Filter out training items from popular items
        train_items = user_pos_train.get(u_idx, set())
        preds = [i for i in top_popular_items if i not in train_items]
        
        for k in k_values:
            preds_k = set(preds[:k])
            
            # Recall@K
            hits = len(preds_k & test_items)
            recall = hits / min(len(test_items), k) if test_items else 0.0
            metrics[f"recall@{k}"].append(recall)
            
            # NDCG@K
            dcg = 0.0
            for i, item in enumerate(preds[:k]):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg@{k}"].append(ndcg)
    
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    avg_metrics["model_type"] = "popularity_baseline"
    
    logger.info("Popularity Baseline:")
    logger.info("  Recall@10: %.4f", avg_metrics["recall@10"])
    logger.info("  NDCG@10: %.4f", avg_metrics["ndcg@10"])
    
    return avg_metrics


def save_checkpoint(
    model_type: str,
    iteration: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Save training checkpoint for crash recovery.
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = TRAINING_CONFIG["checkpoints_dir"] / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration:04d}.npz"
    
    np.savez(
        checkpoint_path,
        user_factors=user_factors,
        item_factors=item_factors,
        iteration=iteration,
        timestamp=datetime.now().isoformat(),
        metrics=json.dumps(metrics or {})
    )
    
    if logger:
        logger.info("Saved checkpoint: %s", checkpoint_path.name)
    
    # Cleanup old checkpoints
    keep_n = TRAINING_CONFIG["keep_n_checkpoints"]
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_iter_*.npz"))
    if len(checkpoints) > keep_n:
        for old_ckpt in checkpoints[:-keep_n]:
            old_ckpt.unlink()
            if logger:
                logger.debug("Removed old checkpoint: %s", old_ckpt.name)
    
    return checkpoint_path


def load_latest_checkpoint(
    model_type: str,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Load the latest checkpoint for incremental training.
    
    Returns:
        Dict with checkpoint data or None if not found
    """
    checkpoint_dir = TRAINING_CONFIG["checkpoints_dir"] / model_type
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_iter_*.npz"))
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    logger.info("Loading checkpoint: %s", latest.name)
    
    data = np.load(latest, allow_pickle=True)
    return {
        "user_factors": data["user_factors"],
        "item_factors": data["item_factors"],
        "iteration": int(data["iteration"]),
        "timestamp": str(data["timestamp"]),
        "metrics": json.loads(str(data["metrics"])),
    }


def load_previous_model(
    model_type: str,
    logger: logging.Logger
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load the most recent trained model for warm-start.
    
    Returns:
        Dict with U and V matrices or None
    """
    registry_path = TRAINING_CONFIG["registry_path"]
    if not registry_path.exists():
        return None
    
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    # Find latest model of this type
    models_of_type = [
        m for m in registry.get("models", [])
        if m.get("model_type") == model_type
    ]
    
    if not models_of_type:
        return None
    
    latest = max(models_of_type, key=lambda m: m.get("registered_at", ""))
    model_path = Path(latest["path"])
    
    if not model_path.exists():
        logger.warning("Previous model path not found: %s", model_path)
        return None
    
    try:
        U = np.load(model_path / f"{model_type}_U.npy")
        V = np.load(model_path / f"{model_type}_V.npy")
        logger.info("Loaded previous model for warm-start: %s", latest["model_id"])
        return {"U": U, "V": V, "model_id": latest["model_id"]}
    except Exception as e:
        logger.warning("Failed to load previous model: %s", e)
        return None


@retry(max_attempts=2, backoff_factor=2.0)  # type: ignore[misc]
def train_als_model(
    data: Dict[str, Any],
    logger: logging.Logger,
    warmstart: bool = False,
    use_bert_init: bool = True,
) -> Dict[str, Any]:
    """
    Train ALS model using implicit library.
    
    Supports:
    - Warm-start from previous model (incremental training)
    - BERT initialization for cold-start items
    - Checkpointing during training

    Returns:
        Dict with model, embeddings, and training info
    """
    from implicit.als import AlternatingLeastSquares

    config = TRAINING_CONFIG["als"]
    incremental_cfg = TRAINING_CONFIG["incremental"]
    
    # Determine training mode
    is_warmstart = warmstart and incremental_cfg["enabled"] and incremental_cfg["warmstart"]
    iterations = incremental_cfg["warmstart_iters"] if is_warmstart else config["iterations"]

    logger.info("Training ALS model...")
    logger.info(
        "  factors=%s, reg=%s, iter=%s, alpha=%s, warmstart=%s",
        config["factors"],
        config["regularization"],
        iterations,
        config["alpha"],
        is_warmstart,
    )

    # Initialize model
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        iterations=iterations,
        alpha=config["alpha"],
        use_gpu=config["use_gpu"],
        calculate_training_loss=config["calculate_training_loss"],
        random_state=42,
    )

    # Train on confidence matrix (item x user format for implicit)
    # IMPORTANT: Implicit 0.7+ expects (users, items) matrix, NOT (items, users)
    # Old code transposed, but new implicit library expects user-item directly
    X_train = data["X_confidence"]  # Keep as (users, items) - no transpose needed
    num_users, num_items = X_train.shape

    # === WARM-START: Load previous model embeddings ===
    prev_model = None
    if is_warmstart:
        prev_model = load_previous_model("als", logger)
        if prev_model is not None:
            prev_U, prev_V = prev_model["U"], prev_model["V"]
            
            # Initialize with previous factors (handle dimension mismatch)
            if prev_U.shape[1] == config["factors"]:
                # Copy existing user factors
                model.user_factors = np.zeros((num_users, config["factors"]), dtype=np.float32)
                copy_users = min(prev_U.shape[0], num_users)
                model.user_factors[:copy_users] = prev_U[:copy_users]
                
                # Initialize new users with mean
                if num_users > prev_U.shape[0]:
                    model.user_factors[prev_U.shape[0]:] = prev_U.mean(axis=0)
                    logger.info("  Initialized %d new users from mean", 
                               num_users - prev_U.shape[0])
                
                # Copy existing item factors
                model.item_factors = np.zeros((num_items, config["factors"]), dtype=np.float32)
                copy_items = min(prev_V.shape[0], num_items)
                model.item_factors[:copy_items] = prev_V[:copy_items]
                
                logger.info("  Warm-start: loaded %d users, %d items from previous model",
                           copy_users, copy_items)
            else:
                logger.warning("  Factor dimension mismatch, falling back to cold start")
                prev_model = None

    # === BERT INIT: Initialize cold items with projected BERT embeddings ===
    bert_initialized = False
    if use_bert_init and config.get("bert_init_cold_threshold", 5) > 0:
        cold_threshold = config["bert_init_cold_threshold"]
        cold_mask = get_cold_items(data["X_confidence"], threshold=cold_threshold)
        n_cold = cold_mask.sum()
        
        if n_cold > 0:
            bert_embeddings = load_bert_embeddings(logger)
            if bert_embeddings is not None and len(bert_embeddings) == num_items:
                projected = project_bert_to_factors(bert_embeddings, config["factors"], logger)
                
                # Only apply to cold items (unless warm-start already set them)
                if not hasattr(model, 'item_factors') or model.item_factors is None:
                    model.item_factors = np.zeros((num_items, config["factors"]), dtype=np.float32)
                
                # Initialize cold items with projected BERT
                model.item_factors[cold_mask] = projected[cold_mask]
                bert_initialized = True
                logger.info("  BERT-initialized %d cold items (threshold=%d)",
                           n_cold, cold_threshold)

    start_time = datetime.now()
    model.fit(X_train, show_progress=True)
    training_time = (datetime.now() - start_time).total_seconds()

    logger.info("ALS training completed in %.1fs", training_time)

    # Save checkpoint
    save_checkpoint(
        model_type="als",
        iteration=iterations,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        metrics={"training_time": training_time},
        logger=logger
    )

    # Extract embeddings
    U = model.user_factors  # (num_users, factors)
    V = model.item_factors  # (num_items, factors)

    return {
        "model": model,
        "U": U,
        "V": V,
        "model_type": "als",
        "training_time": training_time,
        "config": config,
        "loss": getattr(model, "loss", None),
        "warmstart": is_warmstart and prev_model is not None,
        "bert_initialized": bert_initialized,
    }


@retry(max_attempts=2, backoff_factor=2.0)  # type: ignore[misc]
def train_bpr_model(
    data: Dict[str, Any],
    logger: logging.Logger,
    warmstart: bool = False,
) -> Dict[str, Any]:
    """
    Train BPR model using implicit library.
    
    Supports:
    - Warm-start from previous model
    - Early stopping based on validation Recall@K
    - Checkpointing during training

    Returns:
        Dict with model, embeddings, and training info
    """
    from implicit.bpr import BayesianPersonalizedRanking

    config = TRAINING_CONFIG["bpr"]
    early_stop_cfg = TRAINING_CONFIG["early_stopping"]
    incremental_cfg = TRAINING_CONFIG["incremental"]
    val_ratio = TRAINING_CONFIG["validation_ratio"]

    # Determine training mode
    is_warmstart = warmstart and incremental_cfg["enabled"] and incremental_cfg["warmstart"]
    epochs = incremental_cfg["warmstart_iters"] if is_warmstart else config["epochs"]

    logger.info("Training BPR model...")
    logger.info(
        "  factors=%s, lr=%s, reg=%s, epochs=%s, warmstart=%s",
        config["factors"],
        config["learning_rate"],
        config["regularization"],
        epochs,
        is_warmstart,
    )

    # === VALIDATION SPLIT for early stopping ===
    user_pos_train = data["user_pos_train"]
    train_sets, val_sets = None, None
    
    if early_stop_cfg["enabled"] and not is_warmstart:
        train_sets, val_sets = create_validation_split(
            user_pos_train, val_ratio=val_ratio, logger=logger
        )
    
    # === Create modified training matrix if using validation ===
    # IMPORTANT: Implicit 0.7+ expects (users, items) matrix, NOT (items, users)
    X_train = data["X_binary"]  # Keep as (users, items) - no transpose needed
    num_users, num_items = X_train.shape

    # Initialize model
    model = BayesianPersonalizedRanking(
        factors=config["factors"],
        learning_rate=config["learning_rate"],
        regularization=config["regularization"],
        iterations=epochs,
        random_state=42,
        verify_negative_samples=True,
    )

    # === WARM-START: Load previous model embeddings ===
    prev_model = None
    if is_warmstart:
        prev_model = load_previous_model("bpr", logger)
        if prev_model is not None:
            prev_U, prev_V = prev_model["U"], prev_model["V"]
            
            if prev_U.shape[1] == config["factors"]:
                model.user_factors = np.zeros((num_users, config["factors"]), dtype=np.float32)
                copy_users = min(prev_U.shape[0], num_users)
                model.user_factors[:copy_users] = prev_U[:copy_users]
                
                if num_users > prev_U.shape[0]:
                    model.user_factors[prev_U.shape[0]:] = prev_U.mean(axis=0)
                
                model.item_factors = np.zeros((num_items, config["factors"]), dtype=np.float32)
                copy_items = min(prev_V.shape[0], num_items)
                model.item_factors[:copy_items] = prev_V[:copy_items]
                
                logger.info("  Warm-start: loaded %d users, %d items from previous model",
                           copy_users, copy_items)
            else:
                logger.warning("  Factor dimension mismatch, falling back to cold start")
                prev_model = None

    start_time = datetime.now()
    
    # === TRAINING WITH EARLY STOPPING ===
    if early_stop_cfg["enabled"] and val_sets and not is_warmstart:
        logger.info("  Early stopping enabled: patience=%d, min_delta=%.4f",
                   early_stop_cfg["patience"], early_stop_cfg["min_delta"])
        
        best_val_metric = 0.0
        patience_counter = 0
        best_factors = None
        
        # Train epoch-by-epoch for early stopping
        model.iterations = 1  # Single epoch at a time
        
        for epoch in range(epochs):
            model.fit(X_train, show_progress=False)
            
            # Evaluate on validation set every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                U = model.user_factors
                V = model.item_factors
                
                # Quick validation Recall@10
                val_recall = 0.0
                n_val_users = 0
                
                for u_idx, val_items in val_sets.items():
                    if not val_items or u_idx >= U.shape[0]:
                        continue
                    
                    scores = U[u_idx] @ V.T
                    train_items = train_sets.get(u_idx, set())
                    if train_items:
                        scores[list(train_items)] = -np.inf
                    
                    top_10 = set(np.argsort(scores)[-10:])
                    hits = len(top_10 & val_items)
                    val_recall += hits / min(len(val_items), 10)
                    n_val_users += 1
                
                val_recall /= max(n_val_users, 1)
                logger.info("  Epoch %d/%d - Val Recall@10: %.4f", 
                           epoch + 1, epochs, val_recall)
                
                # Check improvement
                if val_recall > best_val_metric + early_stop_cfg["min_delta"]:
                    best_val_metric = val_recall
                    best_factors = (U.copy(), V.copy())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stop check
                if patience_counter >= early_stop_cfg["patience"]:
                    logger.info("  Early stopping at epoch %d (no improvement for %d checks)",
                               epoch + 1, early_stop_cfg["patience"])
                    break
                
                # Checkpoint
                save_checkpoint(
                    model_type="bpr",
                    iteration=epoch + 1,
                    user_factors=U,
                    item_factors=V,
                    metrics={"val_recall@10": val_recall},
                    logger=logger
                )
        
        # Restore best factors if we found any
        if best_factors is not None:
            model.user_factors, model.item_factors = best_factors
            logger.info("  Restored best model (val_recall@10=%.4f)", best_val_metric)
    else:
        # Standard training without early stopping
        model.fit(X_train, show_progress=True)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info("BPR training completed in %.1fs", training_time)

    # Save final checkpoint
    save_checkpoint(
        model_type="bpr",
        iteration=epochs,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        metrics={"training_time": training_time},
        logger=logger
    )

    # Extract embeddings
    U = model.user_factors
    V = model.item_factors

    return {
        "model": model,
        "U": U,
        "V": V,
        "model_type": "bpr",
        "training_time": training_time,
        "config": config,
        "warmstart": is_warmstart and prev_model is not None,
        "early_stopped": early_stop_cfg["enabled"] and val_sets is not None,
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(
    model_result: Dict[str, Any],
    data: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Evaluate trained model using Recall@K and NDCG@K.

    Returns:
        Dict with evaluation metrics
    """
    import numpy as np

    model_type = model_result["model_type"]
    U = model_result["U"]
    V = model_result["V"]
    user_pos_train = data["user_pos_train"]

    logger.info("Evaluating %s model...", model_type.upper())

    # Load test data
    processed_dir = TRAINING_CONFIG["processed_dir"]

    try:
        import pickle

        with open(processed_dir / "user_pos_test.pkl", "rb") as f:
            user_pos_test = pickle.load(f)
    except FileNotFoundError:
        logger.warning("Test data not found, using 20%% holdout from train")
        # Fallback: use last 20% of each user's positives as test
        user_pos_test = {}
        for u_idx, items in user_pos_train.items():
            if len(items) >= 2:
                items_list = list(items)
                split_idx = max(1, int(len(items_list) * 0.8))
                user_pos_test[u_idx] = set(items_list[split_idx:])

    k_values = TRAINING_CONFIG["eval_k_values"]

    # Compute predictions for test users
    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    test_users = list(user_pos_test.keys())

    for u_idx in test_users:
        if u_idx >= U.shape[0]:
            continue

        # Compute scores
        scores = U[u_idx] @ V.T

        # Filter out training items
        train_items = user_pos_train.get(u_idx, set())
        if train_items:
            scores[list(train_items)] = -np.inf

        # Get top-K predictions
        max_k = max(k_values)
        top_k_items = np.argsort(scores)[-max_k:][::-1]

        # Ground truth
        test_items = user_pos_test[u_idx]

        for k in k_values:
            preds_k = set(top_k_items[:k])

            # Recall@K
            hits = len(preds_k & test_items)
            recall = hits / min(len(test_items), k) if test_items else 0.0
            metrics[f"recall@{k}"].append(recall)

            # NDCG@K
            dcg = 0.0
            for i, item in enumerate(top_k_items[:k]):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)

            idcg = sum(
                1.0 / np.log2(i + 2) for i in range(min(len(test_items), k))
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg@{k}"].append(ndcg)

    # Average metrics
    avg_metrics: Dict[str, Any] = {
        k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()
    }
    avg_metrics["num_test_users"] = len(test_users)

    logger.info("  Recall@10: %.4f", avg_metrics["recall@10"])
    logger.info("  NDCG@10: %.4f", avg_metrics["ndcg@10"])

    return avg_metrics


# =============================================================================
# Model Saving & Registration
# =============================================================================

def save_model(
    model_result: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Save model artifacts to disk.

    Returns:
        Dict with save paths and model_id
    """
    import numpy as np

    model_type = model_result["model_type"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{model_type}_{timestamp}"

    # Create output directory
    output_dir = TRAINING_CONFIG["artifacts_dir"] / model_type / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving %s model to %s", model_type.upper(), output_dir)

    # Save embeddings
    np.save(output_dir / f"{model_type}_U.npy", model_result["U"])
    np.save(output_dir / f"{model_type}_V.npy", model_result["V"])

    # Save parameters
    params = {
        "model_type": model_type,
        "config": model_result["config"],
        "training_time": model_result["training_time"],
        "num_users": model_result["U"].shape[0],
        "num_items": model_result["V"].shape[0],
        "factors": model_result["U"].shape[1],
    }

    with open(output_dir / f"{model_type}_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save metrics
    with open(output_dir / f"{model_type}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save metadata
    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "created_at": datetime.now().isoformat(),
        "data_hash": data["data_hash"],
        "git_commit": get_git_commit(),
        "score_range": {
            "min": 0.0,
            "max": float(model_result["U"].shape[1]),  # Approximate max score
        },
    }

    with open(output_dir / f"{model_type}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model_id": model_id,
        "output_dir": str(output_dir),
        "files": [
            f"{model_type}_U.npy",
            f"{model_type}_V.npy",
            f"{model_type}_params.json",
            f"{model_type}_metrics.json",
            f"{model_type}_metadata.json",
        ],
    }


def register_model(
    model_id: str,
    model_type: str,
    metrics: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Register model in the registry.

    Returns:
        True if registered as best model
    """
    registry_path = TRAINING_CONFIG["registry_path"]

    # Load or create registry
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": [], "current_best": None}

    # Create model entry
    entry = {
        "model_id": model_id,
        "model_type": model_type,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "path": str(output_dir),
        "is_active": False,
    }

    # Check if this is the best model
    is_best = False
    primary_metric = (
        f"{TRAINING_CONFIG['primary_metric']}@{TRAINING_CONFIG['primary_k']}"
    )
    current_score = metrics.get(primary_metric, 0.0)

    # Handle both dict and list formats for models
    models_dict = registry.get("models", {})
    if isinstance(models_dict, list):
        # Convert list to dict (old format)
        models_dict = {m.get("model_id", f"model_{i}"): m for i, m in enumerate(models_dict)}
        registry["models"] = models_dict

    if registry.get("current_best"):
        # Find current best's score
        best_model = models_dict.get(registry["current_best"])
        if best_model:
            best_metrics = best_model.get("metrics", {})
            best_score = best_metrics.get(primary_metric, 0.0)
            if current_score > best_score:
                is_best = True
                logger.info(
                    "New best model! %s: %.4f > %.4f",
                    primary_metric,
                    current_score,
                    best_score,
                )
        else:
            is_best = True
    else:
        is_best = True

    if is_best:
        # Deactivate previous best
        for model_key in models_dict:
            models_dict[model_key]["is_active"] = False

        entry["is_active"] = True
        registry["current_best"] = model_id

    # Add new model entry
    models_dict[model_id] = entry

    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Registered model: %s (is_best=%s)", model_id, is_best)

    return is_best


# =============================================================================
# Main Pipeline
# =============================================================================

def train_models(
    model_types: List[str] = None,
    auto_select: bool = True,
    skip_eval: bool = False,
    force: bool = False,
    warmstart: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Main training pipeline with advanced features.

    Args:
        model_types: List of models to train ('als', 'bpr')
        auto_select: Automatically register best model
        skip_eval: Skip evaluation
        force: Force training even if recent model exists
        warmstart: Use warm-start from previous model (incremental training)
        logger: Logger instance

    Returns:
        Dict with training results
    """
    if model_types is None:
        model_types = ["als", "bpr"]

    if logger is None:
        logger = setup_logging("model_training")

    tracker = PipelineTracker()
    result: Dict[str, Any] = {
        "pipeline": "model_training",
        "started_at": datetime.now().isoformat(),
        "models": {},
        "warmstart": warmstart,
    }

    with PipelineLock("model_training") as lock:
        if not lock.acquired:
            msg = "Model training already running"
            logger.warning(msg)
            result["status"] = "skipped"
            result["message"] = msg
            return result

        run_id = tracker.start_run("model_training", {
            "models": model_types,
            "warmstart": warmstart,
        })

        try:
            # Load data
            logger.info("Loading training data...")
            data = load_training_data(logger)
            result["data_hash"] = data["data_hash"]

            # === POPULARITY BASELINE ===
            baseline_metrics = None
            if TRAINING_CONFIG.get("compute_baseline", True) and not skip_eval:
                logger.info("\n%s", "=" * 60)
                logger.info("Computing popularity baseline...")
                logger.info("%s", "=" * 60)
                baseline_metrics = compute_popularity_baseline(data, logger)
                result["baseline_metrics"] = baseline_metrics

            trained_models: List[Dict[str, Any]] = []

            # Train each model type
            for model_type in model_types:
                logger.info("\n%s", "=" * 60)
                logger.info("Training %s model%s", model_type.upper(),
                           " (warmstart)" if warmstart else "")
                logger.info("%s", "=" * 60)

                try:
                    # Train with new features
                    if model_type == "als":
                        model_result = train_als_model(
                            data, logger,
                            warmstart=warmstart,
                            use_bert_init=True
                        )
                    elif model_type == "bpr":
                        model_result = train_bpr_model(
                            data, logger,
                            warmstart=warmstart
                        )
                    else:
                        logger.error("Unknown model type: %s", model_type)
                        continue

                    # Evaluate
                    if not skip_eval:
                        metrics = evaluate_model(model_result, data, logger)
                        
                        # === COMPARE WITH BASELINE ===
                        if baseline_metrics:
                            baseline_recall = baseline_metrics.get("recall@10", 0)
                            model_recall = metrics.get("recall@10", 0)
                            
                            if baseline_recall > 0:
                                improvement = (model_recall - baseline_recall) / baseline_recall * 100
                                metrics["baseline_improvement_pct"] = improvement
                                logger.info("  Improvement over baseline: %.1f%%", improvement)
                                
                                if model_recall <= baseline_recall:
                                    logger.warning(
                                        "  WARNING: Model not beating popularity baseline!"
                                    )
                    else:
                        metrics = {}

                    # Save
                    save_result = save_model(
                        model_result,
                        metrics,
                        data,
                        logger,
                    )

                    trained_models.append(
                        {
                            "model_id": save_result["model_id"],
                            "model_type": model_type,
                            "metrics": metrics,
                            "output_dir": save_result["output_dir"],
                            "training_time": model_result["training_time"],
                            "warmstart": model_result.get("warmstart", False),
                            "bert_initialized": model_result.get("bert_initialized", False),
                        }
                    )

                    result["models"][model_type] = {
                        "model_id": save_result["model_id"],
                        "metrics": metrics,
                        "training_time": model_result["training_time"],
                        "warmstart": model_result.get("warmstart", False),
                        "bert_initialized": model_result.get("bert_initialized", False),
                    }

                except Exception as e:
                    logger.error("Failed to train %s: %s", model_type, e)
                    result["models"][model_type] = {
                        "status": "failed",
                        "error": str(e),
                    }

            # Auto-select best model
            if auto_select and trained_models:
                logger.info("\n%s", "=" * 60)
                logger.info("Selecting best model...")

                primary_metric = (
                    f"{TRAINING_CONFIG['primary_metric']}@"
                    f"{TRAINING_CONFIG['primary_k']}"
                )
                best_model = max(
                    trained_models,
                    key=lambda m: m["metrics"].get(primary_metric, 0.0),
                )

                is_best = register_model(
                    best_model["model_id"],
                    best_model["model_type"],
                    best_model["metrics"],
                    Path(best_model["output_dir"]),
                    logger,
                )

                result["selected_model"] = best_model["model_id"]
                result["is_new_best"] = is_best

            # Success
            result["status"] = "success"
            result["finished_at"] = datetime.now().isoformat()

            tracker.complete_run(
                run_id,
                {
                    "status": "success",
                    "models_trained": len(trained_models),
                    "selected_model": result.get("selected_model"),
                },
            )

            logger.info("\nModel training completed successfully!")

            # Send alert
            send_pipeline_alert(
                "model_training",
                "success",
                f"Trained {len(trained_models)} models. "
                f"Best: {result.get('selected_model')}",
                severity="info",
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Training pipeline failed: %s", error_msg)

            result["status"] = "failed"
            result["error"] = error_msg

            tracker.fail_run(run_id, error_msg)

            send_pipeline_alert(
                "model_training",
                "failed",
                f"Training failed: {error_msg}",
                severity="error",
            )

            raise

    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train recommendation models",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["als", "bpr", "both"],
        default="both",
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        default=True,
        help="Automatically register best model",
    )
    parser.add_argument(
        "--no-auto-select",
        action="store_false",
        dest="auto_select",
        help="Do not auto-register best model",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force training even if recent model exists",
    )
    parser.add_argument(
        "--warmstart",
        "-w",
        action="store_true",
        help="Use warm-start from previous model (incremental training)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Determine models to train
    if args.model == "both":
        model_types = ["als", "bpr"]
    else:
        model_types = [args.model]

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("model_training", level=level)

    try:
        result = train_models(
            model_types=model_types,
            auto_select=args.auto_select,
            skip_eval=args.skip_eval,
            force=args.force,
            warmstart=args.warmstart,
            logger=logger,
        )

        print(f"\n{'=' * 60}")
        print(f"Training Result: {result['status'].upper()}")
        print(f"{'=' * 60}")
        
        # Show baseline metrics if computed
        if "baseline_metrics" in result:
            baseline = result["baseline_metrics"]
            print("\nPopularity Baseline:")
            print(f"  Recall@10: {baseline.get('recall@10', 0):.4f}")
            print(f"  NDCG@10: {baseline.get('ndcg@10', 0):.4f}")

        for model_type, model_info in result.get("models", {}).items():
            print(f"\n{model_type.upper()}:")
            if "model_id" in model_info:
                print(f"  Model ID: {model_info['model_id']}")
                print(f"  Training Time: {model_info['training_time']:.1f}s")
                
                if model_info.get("warmstart"):
                    print("  Mode: Warm-start (incremental)")
                if model_info.get("bert_initialized"):
                    print("  BERT Init: Yes (cold items)")
                
                if model_info.get("metrics"):
                    metrics = model_info['metrics']
                    print(
                        "  Recall@10: "
                        f"{metrics.get('recall@10', 0):.4f}"
                    )
                    print(
                        "  NDCG@10: "
                        f"{metrics.get('ndcg@10', 0):.4f}"
                    )
                    if "baseline_improvement_pct" in metrics:
                        print(
                            f"  vs Baseline: +{metrics['baseline_improvement_pct']:.1f}%"
                        )
            else:
                print(f"  Status: {model_info.get('status', 'unknown')}")
                if "error" in model_info:
                    print(f"  Error: {model_info['error']}")

        if "selected_model" in result:
            print(f"\nSelected Best Model: {result['selected_model']}")

        sys.exit(0 if result["status"] == "success" else 1)

    except Exception as e:  # pragma: no cover - CLI guard
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()


"""
Data Refresh Pipeline.

This module contains the full implementation of the data refresh pipeline
and exposes a CLI via `python -m automation.data_refresh`.

Key responsibilities:
1. Merge staging data (from web ingestion) into raw data
2. Run data processing pipeline (validate, engineer features, split, build matrices)
3. Save processed artifacts for training
"""

import os
import sys
import json
import shutil
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (  # type: ignore
    retry,
    PipelineTracker,
    PipelineLock,
    setup_logging,
    compute_data_hash,
    send_pipeline_alert,
    get_git_commit,
)


# =============================================================================
# Configuration
# =============================================================================

DATA_CONFIG = {
    "raw_data_dir": PROJECT_ROOT / "data" / "published_data",
    "processed_dir": PROJECT_ROOT / "data" / "processed",
    "staging_dir": PROJECT_ROOT / "data" / "staging",
    "raw_files": [
        "data_reviews_purchase.csv",
        "data_product.csv",
        "data_product_attribute.csv",
    ],
    "output_files": [
        "interactions.parquet",
        "all_quality_scores_cache.parquet",  # Cache for ALL interactions' quality scores
        "X_train_confidence.npz",
        "X_train_binary.npz",
        "user_item_mappings.json",
        "user_metadata.pkl",
        "user_pos_train.pkl",
        "user_hard_neg_train.pkl",
        "data_stats.json",
    ],
}


# =============================================================================
# Staging Data Merge
# =============================================================================

def merge_staging_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Merge staging data (from web ingestion) into raw data.
    
    Returns:
        Dict with merge statistics and the new interactions DataFrame
    """
    staging_dir = DATA_CONFIG["staging_dir"]
    raw_dir = DATA_CONFIG["raw_data_dir"]
    staging_file = staging_dir / "new_interactions.csv"
    raw_file = raw_dir / "data_reviews_purchase.csv"
    
    result = {
        "staging_file_exists": staging_file.exists(),
        "new_interactions_count": 0,
        "merged": False,
        "new_interactions_df": None  # Return the new data for incremental processing
    }
    
    if not staging_file.exists():
        logger.info("No staging data found, skipping merge")
        return result
    
    # Read staging data
    try:
        staging_df = pd.read_csv(staging_file, encoding='utf-8')
        result["new_interactions_count"] = len(staging_df)
        
        if len(staging_df) == 0:
            logger.info("Staging file is empty, skipping merge")
            return result
            
        logger.info(f"Found {len(staging_df)} new interactions in staging")
    except Exception as e:
        logger.warning(f"Could not read staging file: {e}")
        return result
    
    # Read existing raw data
    try:
        raw_df = pd.read_csv(raw_file, encoding='utf-8')
        logger.info(f"Loaded {len(raw_df)} existing interactions from raw data")
    except Exception as e:
        logger.error(f"Could not read raw data file: {e}")
        raise
    
    # Map staging columns to raw data format
    # Staging: interaction_id, user_id, product_id, rating, comment, timestamp, interaction_type, quantity, ingested_at
    # Raw: user_id, product_id, rating, comment, cmt_date, [other columns...]
    
    # Parse timestamp with ISO8601 format (handles 'Z' suffix)
    timestamps = pd.to_datetime(staging_df['timestamp'], format='ISO8601', utc=True)
    # Convert to local timezone and format
    timestamps = timestamps.dt.tz_convert(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    staging_mapped = pd.DataFrame({
        'user_id': staging_df['user_id'],
        'product_id': staging_df['product_id'],
        'rating': staging_df['rating'].astype(float),
        'comment': staging_df['comment'].fillna(''),
        'cmt_date': timestamps
    })
    
    # Store the new interactions for incremental processing
    result["new_interactions_df"] = staging_mapped.copy()
    
    # Add any missing columns from raw_df
    for col in raw_df.columns:
        if col not in staging_mapped.columns:
            if col == 'processed_comment':
                staging_mapped[col] = staging_mapped['comment']
            else:
                staging_mapped[col] = None
    
    # Ensure column order matches
    staging_mapped = staging_mapped[raw_df.columns]
    
    # Backup raw file before merge
    backup_file = raw_dir / f"data_reviews_purchase_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy(raw_file, backup_file)
    logger.info(f"Backed up raw data to {backup_file}")
    
    # Merge data
    merged_df = pd.concat([raw_df, staging_mapped], ignore_index=True)
    
    # Remove duplicates (same user_id, product_id, cmt_date)
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['user_id', 'product_id', 'cmt_date'], keep='last')
    after_dedup = len(merged_df)
    
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup} duplicate entries")
    
    # Save merged data
    merged_df.to_csv(raw_file, index=False, encoding='utf-8')
    logger.info(f"Saved merged data with {len(merged_df)} total interactions")
    
    # Archive staging file
    archive_dir = staging_dir / "archived"
    archive_dir.mkdir(exist_ok=True)
    archive_file = archive_dir / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.move(staging_file, archive_file)
    logger.info(f"Archived staging file to {archive_file}")
    
    result["merged"] = True
    result["total_after_merge"] = len(merged_df)
    result["duplicates_removed"] = before_dedup - after_dedup
    
    return result


# =============================================================================
# Data Refresh Logic
# =============================================================================

def check_data_changed(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if raw data has changed since last processing.

    Returns:
        Dict with 'changed' flag and hash values
    """
    raw_dir = DATA_CONFIG["raw_data_dir"]
    processed_dir = DATA_CONFIG["processed_dir"]

    # Compute current raw data hash
    current_hash = compute_data_hash(raw_dir, DATA_CONFIG["raw_files"])
    logger.info(f"Current raw data hash: {current_hash}")

    # Check previous hash
    mappings_file = processed_dir / "user_item_mappings.json"
    previous_hash = None

    if mappings_file.exists():
        try:
            with open(mappings_file, "r") as f:
                mappings = json.load(f)
                # Try both locations: root level and metadata
                previous_hash = mappings.get("data_hash")
                if previous_hash is None and "metadata" in mappings:
                    previous_hash = mappings["metadata"].get("data_hash")
                logger.info(f"Previous data hash: {previous_hash}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"Could not read previous hash: {e}")

    changed = current_hash != previous_hash

    return {
        "changed": changed,
        "current_hash": current_hash,
        "previous_hash": previous_hash,
    }


def run_incremental_update(
    new_interactions: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run incremental update - only process new interactions and update matrices.
    
    This is MUCH faster than full pipeline when only a few new interactions.
    Now includes:
    - AI-powered comment quality scoring (ViSoBERT via FeatureEngineer)
    - Support for users becoming trainable (≥2 interactions)
    - Proper confidence score calculation matching full pipeline
    
    Args:
        new_interactions: DataFrame with new interactions from staging
        logger: Logger instance
        
    Returns:
        Dict with update results, or None to signal fallback to full pipeline
    """
    import pickle
    from scipy import sparse as sp
    
    output_dir = DATA_CONFIG["processed_dir"]
    start_time = datetime.now()
    
    logger.info(f"Running INCREMENTAL update for {len(new_interactions)} new interactions...")
    
    # Load existing processed data
    logger.info("Loading existing processed data...")
    
    # Load existing interactions
    existing_parquet = output_dir / "interactions.parquet"
    if not existing_parquet.exists():
        logger.warning("No existing processed data, falling back to full pipeline")
        return None  # Signal to run full pipeline
    
    existing_df = pd.read_parquet(existing_parquet)
    logger.info(f"Loaded {len(existing_df)} existing interactions")
    
    # Load mappings
    mappings_file = output_dir / "user_item_mappings.json"
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)
    
    user_to_idx = {int(k): v for k, v in mappings['user_to_idx'].items()}
    item_to_idx = {int(k): v for k, v in mappings['item_to_idx'].items()}
    idx_to_user = {v: int(k) for k, v in mappings['user_to_idx'].items()}
    idx_to_item = {v: int(k) for k, v in mappings['item_to_idx'].items()}
    
    # Get metadata (can be at root or in 'metadata' key)
    metadata = mappings.get('metadata', mappings)
    positive_threshold = metadata.get('positive_threshold', 4.0)
    hard_negative_threshold = metadata.get('hard_negative_threshold', 3.0)
    
    # Load existing matrices
    X_train_conf = sp.load_npz(output_dir / "X_train_confidence.npz")
    X_train_bin = sp.load_npz(output_dir / "X_train_binary.npz")
    
    # Load user sets
    with open(output_dir / "user_pos_train.pkl", 'rb') as f:
        user_pos_train = pickle.load(f)
    with open(output_dir / "user_hard_neg_train.pkl", 'rb') as f:
        user_hard_neg_train = pickle.load(f)
    
    # Get actual matrix dimensions
    matrix_num_users, matrix_num_items = X_train_conf.shape
    logger.info(f"Matrix dimensions: {matrix_num_users} users x {matrix_num_items} items")
    
    # Initialize FeatureEngineer for AI-powered comment quality scoring
    logger.info("Initializing FeatureEngineer for AI-powered scoring...")
    try:
        from recsys.cf.data.processing.feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer(
            positive_threshold=positive_threshold,
            hard_negative_threshold=hard_negative_threshold,
            use_ai_sentiment=True,
            batch_size=32,  # Smaller batch for few interactions
            enable_fake_review_checks=True
        )
        use_ai_scoring = feature_engineer.model is not None
        if use_ai_scoring:
            logger.info("✓ AI sentiment model loaded for quality scoring")
        else:
            logger.info("⚠ AI model not available, using keyword-based fallback")
    except Exception as e:
        logger.warning(f"Failed to initialize FeatureEngineer: {e}")
        logger.info("Using simple confidence scoring (rating + 0.5)")
        feature_engineer = None
        use_ai_scoring = False
    
    # Count existing interactions per user (for trainability check)
    user_interaction_counts = existing_df.groupby('user_id').size().to_dict()
    user_positive_counts = existing_df[existing_df['is_positive'] == 1].groupby('user_id').size().to_dict()
    
    # Process new interactions
    logger.info("Processing new interactions...")
    
    new_users = set()  # Completely new users (not in any mapping)
    new_items = set()  # Completely new items
    newly_trainable_users = set()  # Users who JUST became trainable with this batch
    out_of_bounds = []  # Users in mapping but not in matrix (cold-start)
    updates = []
    skipped_still_coldstart = []  # Users still not trainable
    
    for _, row in new_interactions.iterrows():
        user_id = int(row['user_id'])
        product_id = int(row['product_id'])
        rating = float(row['rating'])
        comment = str(row.get('comment', '') or '')
        
        # Check if item exists
        if product_id not in item_to_idx:
            new_items.add(product_id)
            continue  # Skip new items (need full reindex to add item columns)
        
        i_idx = item_to_idx[product_id]
        if i_idx >= matrix_num_items:
            out_of_bounds.append(('item', product_id, i_idx))
            continue
        
        # Check user status
        if user_id not in user_to_idx:
            # Completely new user - check if they can become trainable
            prev_count = user_interaction_counts.get(user_id, 0)
            prev_positive = user_positive_counts.get(user_id, 0)
            new_count = prev_count + 1
            new_positive = prev_positive + (1 if rating >= positive_threshold else 0)
            
            # Check trainability: ≥2 interactions AND ≥1 positive
            if new_count >= 2 and new_positive >= 1:
                newly_trainable_users.add(user_id)
            else:
                skipped_still_coldstart.append(user_id)
            
            new_users.add(user_id)
            continue  # Need full reindex to add user rows
        
        # User exists in mapping - check if in matrix
        u_idx = user_to_idx[user_id]
        if u_idx >= matrix_num_users:
            # User in mapping but not in matrix (was cold-start, now maybe trainable?)
            prev_count = user_interaction_counts.get(user_id, 0)
            prev_positive = user_positive_counts.get(user_id, 0)
            new_count = prev_count + 1
            new_positive = prev_positive + (1 if rating >= positive_threshold else 0)
            
            if new_count >= 2 and new_positive >= 1:
                newly_trainable_users.add(user_id)
            
            out_of_bounds.append(('user', user_id, u_idx))
            continue
        
        # Compute comment quality using AI if available
        if feature_engineer is not None and use_ai_scoring:
            comment_quality = feature_engineer.compute_comment_quality_score(comment)
        else:
            # Simple fallback
            comment_quality = 0.5 if comment and len(comment) > 5 else 0.0
        
        confidence = rating + comment_quality
        is_positive = rating >= positive_threshold
        is_hard_negative = rating <= hard_negative_threshold
        
        updates.append({
            'user_id': user_id,
            'product_id': product_id,
            'u_idx': u_idx,
            'i_idx': i_idx,
            'rating': rating,
            'comment': comment,
            'comment_quality': comment_quality,
            'confidence': confidence,
            'is_positive': is_positive,
            'is_hard_negative': is_hard_negative
        })
    
    # Log newly trainable users (need full pipeline to actually add them)
    if newly_trainable_users:
        logger.info(f"Found {len(newly_trainable_users)} users who just became trainable!")
        logger.info("These users will be included in the next full pipeline run.")
    
    if skipped_still_coldstart:
        logger.info(f"Skipped {len(skipped_still_coldstart)} users still in cold-start (<2 interactions)")
    
    if new_users or new_items:
        logger.warning(f"Found {len(new_users)} new users, {len(new_items)} new items - need full reindex")
    
    if out_of_bounds:
        logger.warning(f"Found {len(out_of_bounds)} interactions out of matrix bounds (cold-start users)")
    
    if len(new_users) > 10 or len(new_items) > 5:
        logger.info("Too many new users/items, falling back to full pipeline")
        return None  # Signal to run full pipeline
    
    if not updates:
        logger.info("No valid updates for trainable users, skipping matrix update")
        
        # Still track newly trainable users even if no matrix updates
        if newly_trainable_users:
            stats_file = output_dir / "data_stats.json"
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                stats = {}
            
            pending_trainable = set(stats.get('pending_trainable_users', []))
            pending_trainable.update(newly_trainable_users)
            stats['pending_trainable_users'] = list(pending_trainable)
            stats['pending_trainable_count'] = len(pending_trainable)
            stats['last_incremental_update'] = datetime.now().isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"  - Tracked {len(newly_trainable_users)} newly trainable users for next full pipeline")
        
        # Still count as success - data was merged to raw file
        return {
            "success": True,
            "mode": "incremental",
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "updates_applied": 0,
            "ai_scoring_used": use_ai_scoring,
            "new_users_skipped": len(new_users),
            "new_items_skipped": len(new_items),
            "newly_trainable_users": len(newly_trainable_users),
            "out_of_bounds_skipped": len(out_of_bounds),
            "message": "New interactions are for cold-start users, matrices unchanged"
        }
    
    logger.info(f"Applying {len(updates)} updates to matrices...")
    
    # Convert to LIL format for efficient updates
    X_conf_lil = X_train_conf.tolil()
    X_bin_lil = X_train_bin.tolil()
    
    for upd in updates:
        u_idx = upd['u_idx']
        i_idx = upd['i_idx']
        
        # Check if this is a duplicate (user already interacted with this item)
        # If existing value > 0, this is a repeat interaction
        existing_conf = X_train_conf[u_idx, i_idx]
        if existing_conf > 0:
            # Keep the higher confidence (user may rate again with different sentiment)
            # Or could use: max(existing, new) or average
            upd['is_duplicate'] = True
            upd['confidence'] = max(existing_conf, upd['confidence'])
        else:
            upd['is_duplicate'] = False
        
        # Update confidence matrix
        X_conf_lil[u_idx, i_idx] = upd['confidence']
        
        # Update binary matrix (only positives)
        if upd['is_positive']:
            X_bin_lil[u_idx, i_idx] = 1.0
            
            # Update user positive sets
            if u_idx not in user_pos_train:
                user_pos_train[u_idx] = set()
            user_pos_train[u_idx].add(i_idx)
        else:
            # Handle hard negatives (rating <= 3)
            if upd['rating'] <= 3.0:
                if u_idx not in user_hard_neg_train:
                    user_hard_neg_train[u_idx] = {'explicit': set(), 'implicit': set()}
                elif isinstance(user_hard_neg_train[u_idx], set):
                    # Convert old format to new format
                    user_hard_neg_train[u_idx] = {'explicit': user_hard_neg_train[u_idx], 'implicit': set()}
                user_hard_neg_train[u_idx]['explicit'].add(i_idx)
    
    # Convert back to CSR
    X_train_conf = X_conf_lil.tocsr()
    X_train_bin = X_bin_lil.tocsr()
    
    # Count duplicates
    duplicates = sum(1 for u in updates if u.get('is_duplicate', False))
    if duplicates:
        logger.info(f"  - Duplicate interactions updated: {duplicates}")
    
    # Append new interactions to parquet (for consistency with full pipeline)
    logger.info("Appending new interactions to parquet...")
    try:
        # Build new rows with required columns directly from updates
        new_rows = []
        for upd in updates:
            new_rows.append({
                'user_id': upd['user_id'],
                'product_id': upd['product_id'],
                'rating': upd['rating'],
                'comment': upd['comment'],
                'cmt_date': pd.Timestamp.now(),  # Use current time for incremental updates
                'comment_quality': upd['comment_quality'],
                'confidence_score': upd['confidence'],
                'is_positive': int(upd['is_positive']),
                'is_hard_negative': int(upd['is_hard_negative']),
                'is_trainable_user': True,  # Only trainable users get here
                'u_idx': upd['u_idx'],
                'i_idx': upd['i_idx'],
            })
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Drop duplicates keeping last (newest interaction)
            combined_df = combined_df.drop_duplicates(
                subset=['user_id', 'product_id'], 
                keep='last'
            )
            combined_df.to_parquet(existing_parquet, index=False)
            logger.info(f"  - Parquet updated: {len(existing_df)} → {len(combined_df)} interactions")
    except Exception as e:
        logger.warning(f"Failed to update parquet (non-critical): {e}")
    
    # Save updated matrices
    logger.info("Saving updated matrices...")
    sp.save_npz(output_dir / "X_train_confidence.npz", X_train_conf)
    sp.save_npz(output_dir / "X_train_binary.npz", X_train_bin)
    
    # Save updated user sets
    with open(output_dir / "user_pos_train.pkl", 'wb') as f:
        pickle.dump(user_pos_train, f)
    
    # Save updated hard negative sets
    with open(output_dir / "user_hard_neg_train.pkl", 'wb') as f:
        pickle.dump(user_hard_neg_train, f)
    
    # Update stats
    stats_file = output_dir / "data_stats.json"
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        stats = {}
    
    # Update with safe defaults - ensure all required fields exist for verification
    stats['num_interactions'] = stats.get('num_interactions', X_train_conf.nnz) + len(updates)
    stats['num_users'] = matrix_num_users  # Required for verify_outputs()
    stats['num_items'] = matrix_num_items  # Required for verify_outputs()
    stats['last_incremental_update'] = datetime.now().isoformat()
    stats['incremental_updates_count'] = stats.get('incremental_updates_count', 0) + 1
    stats['ai_scoring_used'] = use_ai_scoring  # Track if AI was used
    
    # Track newly trainable users for next full pipeline
    if newly_trainable_users:
        pending_trainable = set(stats.get('pending_trainable_users', []))
        pending_trainable.update(newly_trainable_users)
        stats['pending_trainable_users'] = list(pending_trainable)
        stats['pending_trainable_count'] = len(pending_trainable)
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Incremental update completed in {duration:.1f}s")
    logger.info(f"  - Updates applied: {len(updates)}")
    logger.info(f"  - AI scoring used: {use_ai_scoring}")
    logger.info(f"  - Skipped (new users): {len(new_users)}")
    logger.info(f"  - Skipped (new items): {len(new_items)}")
    logger.info(f"  - Newly trainable users (pending): {len(newly_trainable_users)}")
    
    return {
        "success": True,
        "mode": "incremental",
        "duration_seconds": duration,
        "updates_applied": len(updates),
        "ai_scoring_used": use_ai_scoring,
        "new_users_skipped": len(new_users),
        "new_items_skipped": len(new_items),
        "newly_trainable_users": len(newly_trainable_users),
        "out_of_bounds_skipped": len(out_of_bounds),
        "duplicates_updated": duplicates,
    }


@retry(max_attempts=3, backoff_factor=2.0)  # type: ignore[misc]
def run_data_pipeline(logger: logging.Logger, force_full: bool = False) -> Dict[str, Any]:
    """
    Run the Task 01 data processing pipeline.
    
    Args:
        logger: Logger instance
        force_full: If True, always run full pipeline (skip incremental)

    Returns:
        Dict with processing results
    """
    from recsys.cf.data import DataProcessor
    import pickle
    from scipy import sparse as sp
    
    raw_dir = DATA_CONFIG["raw_data_dir"]
    output_dir = DATA_CONFIG["processed_dir"]

    logger.info("Initializing DataProcessor...")
    processor = DataProcessor(
        base_path=str(raw_dir),
    )

    logger.info("Running FULL data processing pipeline...")
    start_time = datetime.now()
    
    # === OPTIMIZATION: Load cached comment_quality scores from previous run ===
    # Use dedicated cache file with ALL interactions (not just trainable users)
    cache_file_path = output_dir / "all_quality_scores_cache.parquet"
    cached_scores = None
    
    if cache_file_path.exists():
        try:
            cached_df = pd.read_parquet(cache_file_path)
            if 'comment_quality' in cached_df.columns:
                # Create lookup by (user_id, product_id) -> comment_quality
                cached_df['_key'] = (
                    cached_df['user_id'].astype(str) + '_' + 
                    cached_df['product_id'].astype(str)
                )
                cached_scores = cached_df.set_index('_key')['comment_quality'].to_dict()
                logger.info(f"  ⚡ Loaded {len(cached_scores):,} cached comment_quality scores from ALL interactions cache")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to load cached scores: {e}")
            cached_scores = None
    else:
        # Fallback: Try old interactions.parquet (only trainable users, less coverage)
        fallback_path = output_dir / "interactions.parquet"
        if fallback_path.exists():
            try:
                cached_df = pd.read_parquet(fallback_path)
                if 'comment_quality' in cached_df.columns:
                    cached_df['_key'] = (
                        cached_df['user_id'].astype(str) + '_' + 
                        cached_df['product_id'].astype(str)
                    )
                    cached_scores = cached_df.set_index('_key')['comment_quality'].to_dict()
                    logger.info(f"  ⚡ Loaded {len(cached_scores):,} cached scores (fallback to trainable users only)")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load fallback cached scores: {e}")
                cached_scores = None

    # Step 1: Load and validate interactions with quality scores
    logger.info("Step 1: Loading and validating interactions...")
    df_clean, validation_stats = processor.load_and_validate_interactions(
        apply_deduplication=True,
        detect_outliers=True,
        compute_quality_scores=True,
        cached_quality_scores=cached_scores  # Pass cached scores for optimization
    )
    logger.info(f"Loaded {len(df_clean)} clean interactions")
    
    # === SAVE ALL QUALITY SCORES CACHE (before user filtering) ===
    # This saves ALL 338K interactions' quality scores, not just trainable users
    # Critical for future runs to avoid recomputing AI sentiment on ALL comments
    cache_cols = ['user_id', 'product_id', 'comment_quality']
    if all(col in df_clean.columns for col in cache_cols):
        cache_df = df_clean[cache_cols].copy()
        cache_file_path = output_dir / "all_quality_scores_cache.parquet"
        cache_df.to_parquet(cache_file_path, index=False)
        logger.info(f"  💾 Saved {len(cache_df):,} quality scores to cache for future runs")
    
    # Step 2.1-2.2: Create explicit features (is_positive, is_hard_negative)
    # This is REQUIRED before user segmentation (Step 2.3)
    logger.info("Step 2.1-2.2: Creating explicit feedback features...")
    df_clean = processor.feature_engineer.create_explicit_features(df_clean)
    
    # Step 2.3: User segmentation (trainable vs cold-start)
    logger.info("Step 2.3: Segmenting users...")
    df_segmented, segment_stats = processor.apply_complete_filtering(df_clean)
    logger.info(f"Trainable users: {segment_stats.get('final_trainable_users', 'N/A')}")
    
    # Step 3: Create ID mappings
    logger.info("Step 3: Creating ID mappings...")
    processor.create_id_mappings(df_segmented)
    df_mapped = processor.apply_id_mappings(df_segmented)
    
    # Get mapping stats
    mapping_stats = processor.get_mapping_stats()
    num_users = mapping_stats['num_users']
    num_items = mapping_stats['num_items']
    logger.info(f"Mapped {num_users} users, {num_items} items")
    
    # Step 4: Temporal split (leave-one-out) - ONLY for trainable users
    logger.info("Step 4: Performing temporal split...")
    
    # IMPORTANT: Only split trainable users for CF training
    # Cold-start users are handled separately via content-based fallback
    trainable_df = df_mapped[df_mapped['is_trainable_user'] == True].copy()
    logger.info(f"Splitting {len(trainable_df)} trainable interactions from {trainable_df['u_idx'].nunique()} users")
    
    train_df, test_df, val_df = processor.temporal_split(
        trainable_df,
        method='leave_one_out',
        use_validation=False
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} interactions")
    
    # === CRITICAL: Create LOCAL indices for trainable users ===
    # The training matrix should only contain trainable users (26K) not all users (294K)
    # This creates a compact matrix for efficient ALS/BPR training
    logger.info("Step 4.5: Creating LOCAL indices for trainable users...")
    
    # Get unique trainable user indices (GLOBAL u_idx)
    trainable_u_idx = sorted(train_df['u_idx'].unique())
    num_trainable_users = len(trainable_u_idx)
    
    # Create mapping: GLOBAL u_idx -> LOCAL u_idx_cf (0 to N-1)
    u_idx_to_u_idx_cf = {int(u_idx): cf_idx for cf_idx, u_idx in enumerate(trainable_u_idx)}
    u_idx_cf_to_u_idx = {cf_idx: int(u_idx) for cf_idx, u_idx in enumerate(trainable_u_idx)}
    
    logger.info(f"Created LOCAL indices: {num_trainable_users} trainable users (0 to {num_trainable_users-1})")
    
    # Apply LOCAL indices to train_df and test_df
    train_df = train_df.copy()
    train_df['u_idx_cf'] = train_df['u_idx'].map(u_idx_to_u_idx_cf)
    
    test_df = test_df.copy()
    test_df['u_idx_cf'] = test_df['u_idx'].map(u_idx_to_u_idx_cf)
    # Some test users might not be in train (edge case), drop them
    test_df = test_df.dropna(subset=['u_idx_cf'])
    test_df['u_idx_cf'] = test_df['u_idx_cf'].astype(int)
    
    logger.info(f"Applied LOCAL indices: train={len(train_df)}, test={len(test_df)}")
    
    # Step 5: Build matrices with LOCAL indices
    logger.info("Step 5: Building matrices with LOCAL indices...")
    
    # Use LOCAL u_idx_cf for matrix row indices
    # Confidence matrix for ALS
    X_train_conf = processor.build_confidence_matrix(
        train_df, num_trainable_users, num_items, 
        user_col='u_idx_cf',  # Use LOCAL indices
        value_col='confidence_score'
    )
    
    # Binary matrix for BPR (optional)
    X_train_bin = processor.build_binary_matrix(
        train_df, num_trainable_users, num_items, 
        user_col='u_idx_cf',  # Use LOCAL indices
        positive_only=True
    )
    
    logger.info(f"Matrix shape: {X_train_conf.shape} (trainable users x items)")
    
    # Build user positive sets with LOCAL indices
    user_pos_train = processor.build_user_positive_sets(train_df, user_col='u_idx_cf')
    
    # Build user hard negative sets with LOCAL indices
    user_hard_neg_train = processor.build_user_hard_negative_sets(train_df, user_col='u_idx_cf')
    
    # Step 6: Save artifacts
    logger.info("Step 6: Saving artifacts...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save interactions parquet
    df_mapped.to_parquet(output_dir / "interactions.parquet", index=False)
    
    # Save sparse matrices
    sp.save_npz(output_dir / "X_train_confidence.npz", X_train_conf)
    sp.save_npz(output_dir / "X_train_binary.npz", X_train_bin)
    
    # Save mappings
    processor.save_id_mappings(
        str(output_dir / "user_item_mappings.json"),
        df_clean
    )
    
    # === CRITICAL: Save trainable user mapping ===
    # This maps LOCAL indices (matrix rows) to GLOBAL indices (for serving lookup)
    trainable_mapping = {
        'u_idx_to_u_idx_cf': {str(k): v for k, v in u_idx_to_u_idx_cf.items()},
        'u_idx_cf_to_u_idx': {str(k): v for k, v in u_idx_cf_to_u_idx.items()},
        'num_trainable_users': num_trainable_users,
        'note': 'u_idx is original index from Step 3; u_idx_cf is contiguous [0, N-1] for CF matrix rows'
    }
    with open(output_dir / "trainable_user_mapping.json", 'w') as f:
        json.dump(trainable_mapping, f, indent=2)
    logger.info(f"Saved trainable_user_mapping.json: {num_trainable_users} trainable users")
    
    # Save user sets (with LOCAL indices)
    with open(output_dir / "user_pos_train.pkl", 'wb') as f:
        pickle.dump(user_pos_train, f)
    
    with open(output_dir / "user_hard_neg_train.pkl", 'wb') as f:
        pickle.dump(user_hard_neg_train, f)
    
    # Save user_pos_test for evaluation (with LOCAL indices)
    user_pos_test = processor.build_user_positive_sets(test_df, user_col='u_idx_cf')
    with open(output_dir / "user_pos_test.pkl", 'wb') as f:
        pickle.dump(user_pos_test, f)
    logger.info(f"Saved user_pos_test: {len(user_pos_test)} users (LOCAL indices)")
    
    # Also save with GLOBAL u_idx for backward compatibility
    user_pos_train_global = processor.build_user_positive_sets(train_df, user_col='u_idx')
    user_pos_test_global = processor.build_user_positive_sets(test_df, user_col='u_idx')
    with open(output_dir / "user_pos_train_u_idx.pkl", 'wb') as f:
        pickle.dump(user_pos_train_global, f)
    with open(output_dir / "user_pos_test_u_idx.pkl", 'wb') as f:
        pickle.dump(user_pos_test_global, f)
    logger.info(f"Saved global u_idx versions for backward compatibility")
    
    # Save user metadata
    user_metadata = {
        'trainable_users': list(processor.get_trainable_user_set(df_segmented)),
        'cold_start_users': list(processor.get_cold_start_user_set(df_segmented))
    }
    with open(output_dir / "user_metadata.pkl", 'wb') as f:
        pickle.dump(user_metadata, f)
    
    # Save stats
    duration = (datetime.now() - start_time).total_seconds()
    
    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": len(df_mapped),
        "num_train": len(train_df),
        "num_test": len(test_df),
        "trainable_users": segment_stats.get('final_trainable_users', 0),
        "processing_duration_seconds": duration,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        # Clear pending trainable users after full pipeline
        "pending_trainable_users": [],
        "pending_trainable_count": 0,
    }
    
    with open(output_dir / "data_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Data pipeline completed in {duration:.1f}s")

    return {
        "success": True,
        "duration_seconds": duration,
        "stats": stats,
        "files_created": DATA_CONFIG["output_files"],
    }


def verify_outputs(logger: logging.Logger) -> Dict[str, Any]:
    """
    Verify all expected output files exist and are valid.

    Returns:
        Dict with verification results
    """
    processed_dir = DATA_CONFIG["processed_dir"]
    missing_files = []
    file_sizes = {}

    for filename in DATA_CONFIG["output_files"]:
        file_path = processed_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            file_sizes[filename] = file_path.stat().st_size

    # Verify data_stats.json has required fields
    stats_valid = False
    stats_file = processed_dir / "data_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
                required_fields = ["num_users", "num_items", "num_interactions"]
                stats_valid = all(field in stats for field in required_fields)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"Could not validate stats file: {e}")

    return {
        "success": len(missing_files) == 0 and stats_valid,
        "missing_files": missing_files,
        "file_sizes": file_sizes,
        "stats_valid": stats_valid,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

# Threshold for incremental vs full refresh
INCREMENTAL_THRESHOLD = 100  # Use incremental if <= 100 new interactions


def refresh_data(
    force: bool = False,
    force_full: bool = False,
    dry_run: bool = False,
    skip_merge: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Main data refresh function.
    
    Supports two modes:
    - INCREMENTAL: Fast update when few new interactions (< 100)
    - FULL: Complete reprocessing when many changes or force_full=True

    Args:
        force: Force refresh even if data unchanged
        force_full: Force full pipeline (skip incremental optimization)
        dry_run: Check for changes without processing
        skip_merge: Skip merging staging data
        logger: Logger instance

    Returns:
        Dict with refresh results
    """
    if logger is None:
        logger = setup_logging("data_refresh")

    tracker = PipelineTracker()
    result: Dict[str, Any] = {
        "pipeline": "data_refresh",
        "started_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }

    # Check for concurrent runs
    with PipelineLock("data_refresh") as lock:
        if not lock.acquired:
            msg = "Data refresh already running"
            logger.warning(msg)
            result["status"] = "skipped"
            result["message"] = msg
            return result

        # Start tracking
        run_id = tracker.start_run(
            "data_refresh", {"force": force, "force_full": force_full, "dry_run": dry_run, "skip_merge": skip_merge}
        )

        try:
            # Step 0: Merge staging data (from web ingestion)
            new_interactions_df = None
            if not skip_merge:
                logger.info("Step 0: Merging staging data...")
                merge_result = merge_staging_data(logger)
                result["merge_result"] = merge_result
                
                if merge_result["merged"]:
                    logger.info(f"Merged {merge_result['new_interactions_count']} new interactions")
                    new_interactions_df = merge_result.get("new_interactions_df")
                    force = True  # Force reprocessing if new data merged

            # Step 1: Check if data changed
            logger.info("Checking for data changes...")

            change_check = check_data_changed(logger)
            result["change_check"] = change_check

            if not change_check["changed"] and not force:
                msg = "Raw data unchanged, skipping refresh"
                logger.info(msg)
                result["status"] = "skipped"
                result["message"] = msg
                tracker.complete_run(
                    run_id,
                    {"status": "skipped", "reason": "data_unchanged"},
                )
                return result

            if dry_run:
                msg = "Dry run - would process data"
                logger.info(msg)
                result["status"] = "dry_run"
                result["message"] = msg
                tracker.complete_run(run_id, {"status": "dry_run"})
                return result

            # Step 2: Decide between incremental or full pipeline
            # Check for pending trainable users that need full reindex
            pending_trainable_count = 0
            stats_file = DATA_CONFIG["processed_dir"] / "data_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        pending_trainable_count = stats.get('pending_trainable_count', 0)
                except Exception:
                    pass
            
            # Force full pipeline if too many pending trainable users (>50)
            # These users need proper matrix indexing
            PENDING_TRAINABLE_THRESHOLD = 50
            force_full_due_to_pending = pending_trainable_count >= PENDING_TRAINABLE_THRESHOLD
            
            if force_full_due_to_pending:
                logger.info(f"Found {pending_trainable_count} pending trainable users - forcing full pipeline")
            
            use_incremental = (
                not force_full
                and not force_full_due_to_pending
                and new_interactions_df is not None
                and len(new_interactions_df) <= INCREMENTAL_THRESHOLD
                and change_check["previous_hash"] is not None  # Has existing processed data
            )
            
            if use_incremental:
                logger.info(f"Using INCREMENTAL mode for {len(new_interactions_df)} new interactions")
                pipeline_result = run_incremental_update(new_interactions_df, logger)
                
                if pipeline_result is None:
                    # Incremental failed, fall back to full
                    logger.info("Incremental update not possible, falling back to full pipeline")
                    use_incremental = False
            
            if not use_incremental:
                logger.info("Using FULL pipeline mode")
                pipeline_result = run_data_pipeline(logger, force_full=True)
            
            result["pipeline_result"] = pipeline_result
            result["mode"] = "incremental" if use_incremental and pipeline_result else "full"

            # Step 3: Verify outputs
            logger.info("Verifying outputs...")
            verify_result = verify_outputs(logger)
            result["verification"] = verify_result

            if not verify_result["success"]:
                raise RuntimeError(f"Output verification failed: {verify_result}")

            # Success
            result["status"] = "success"
            result["finished_at"] = datetime.now().isoformat()

            tracker.complete_run(
                run_id,
                {
                    "status": "success",
                    "mode": result["mode"],
                    "data_hash": change_check["current_hash"],
                    "files_created": len(DATA_CONFIG["output_files"]),
                },
            )

            logger.info("Data refresh completed successfully!")

            # Send success alert
            send_pipeline_alert(
                "data_refresh",
                "success",
                f"Data refresh completed. Hash: {change_check['current_hash'][:8]}",
                severity="info",
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Data refresh failed: {error_msg}")

            result["status"] = "failed"
            result["error"] = error_msg

            tracker.fail_run(run_id, error_msg)

            # Send failure alert
            send_pipeline_alert(
                "data_refresh",
                "failed",
                f"Data refresh failed: {error_msg}",
                severity="error",
            )

            raise

    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point for data refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh processed data from raw sources"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force refresh even if data unchanged",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Force full pipeline (skip incremental optimization)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Check for changes without processing",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging staging data from web ingestion",
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
    logger = setup_logging("data_refresh", level=level)

    try:
        result = refresh_data(
            force=args.force,
            force_full=args.force_full,
            dry_run=args.dry_run,
            skip_merge=args.skip_merge,
            logger=logger,
        )

        print(f"\n{'=' * 60}")
        print(f"Data Refresh Result: {result['status'].upper()}")
        print(f"{'=' * 60}")

        if result["status"] == "success":
            mode = result.get("mode", "unknown")
            print(f"  Mode: {mode.upper()}")
            print(f"  Data hash: {result['change_check']['current_hash'][:16]}...")
            if "pipeline_result" in result and result["pipeline_result"]:
                duration = result['pipeline_result'].get('duration_seconds', 0)
                print(f"  Duration: {duration:.1f}s")
                if mode == "incremental":
                    print(f"  Updates applied: {result['pipeline_result'].get('updates_applied', 0)}")
            if "merge_result" in result and result["merge_result"]["merged"]:
                print(f"  New interactions merged: {result['merge_result']['new_interactions_count']}")
        elif result.get("message"):
            print(f"  Message: {result['message']}")

        sys.exit(0 if result["status"] in ("success", "skipped", "dry_run") else 1)

    except Exception as e:  # pragma: no cover - CLI guard
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()



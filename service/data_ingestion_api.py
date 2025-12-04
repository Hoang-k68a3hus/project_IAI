"""
Data Ingestion API for Web Integration.

This module provides REST API endpoints for receiving new user interactions
(purchases, reviews) from the web application and staging them for processing.

Endpoints:
- POST /ingest/review: Submit a new review
- POST /ingest/purchase: Submit a new purchase
- POST /ingest/batch: Submit multiple interactions
- GET /ingest/stats: Get ingestion statistics
- GET /ingest/pending: Get pending data count

The ingested data is staged in data/staging/ and will be processed
by the scheduled data_refresh job.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from threading import Lock
import csv

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STAGING_DIR = DATA_DIR / "staging"
STAGING_FILE = STAGING_DIR / "new_interactions.csv"
STATS_FILE = STAGING_DIR / "ingestion_stats.json"

# Ensure staging directory exists
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safe file writing
_file_lock = Lock()


# ============================================================================
# Request/Response Models
# ============================================================================

class ReviewRequest(BaseModel):
    """Single review submission."""
    user_id: int = Field(..., ge=0, description="User ID")
    product_id: int = Field(..., ge=0, description="Product ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating (1-5)")
    comment: Optional[str] = Field(default="", description="Review comment (Vietnamese)")
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now().isoformat()


class PurchaseRequest(BaseModel):
    """Single purchase submission (implicit feedback)."""
    user_id: int = Field(..., ge=0, description="User ID")
    product_id: int = Field(..., ge=0, description="Product ID")
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")
    quantity: int = Field(default=1, ge=1, description="Purchase quantity")
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now().isoformat()


class BatchInteractionRequest(BaseModel):
    """Batch submission of multiple interactions."""
    reviews: Optional[List[ReviewRequest]] = Field(default=[], description="List of reviews")
    purchases: Optional[List[PurchaseRequest]] = Field(default=[], description="List of purchases")


class InteractionResponse(BaseModel):
    """Response for single interaction submission."""
    status: str
    interaction_id: str
    message: str
    timestamp: str


class BatchResponse(BaseModel):
    """Response for batch submission."""
    status: str
    total_received: int
    reviews_count: int
    purchases_count: int
    message: str
    timestamp: str


class IngestionStats(BaseModel):
    """Ingestion statistics."""
    total_pending: int
    reviews_pending: int
    purchases_pending: int
    last_ingestion: Optional[str]
    last_processed: Optional[str]
    today_count: int
    staging_file_size_kb: float


class PendingDataResponse(BaseModel):
    """Pending data summary."""
    pending_count: int
    pending_reviews: int
    pending_purchases: int
    oldest_entry: Optional[str]
    newest_entry: Optional[str]


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_interaction_id() -> str:
    """Generate unique interaction ID."""
    return f"int_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _load_stats() -> Dict[str, Any]:
    """Load ingestion statistics."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "total_ingested": 0,
        "reviews_ingested": 0,
        "purchases_ingested": 0,
        "last_ingestion": None,
        "last_processed": None,
        "daily_counts": {}
    }


def _save_stats(stats: Dict[str, Any]):
    """Save ingestion statistics."""
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def _append_to_staging(row: Dict[str, Any]):
    """Append a row to the staging CSV file."""
    with _file_lock:
        file_exists = STAGING_FILE.exists()
        
        with open(STAGING_FILE, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'interaction_id', 'user_id', 'product_id', 'rating', 
                'comment', 'timestamp', 'interaction_type', 'quantity',
                'ingested_at'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)


def _count_pending() -> Dict[str, int]:
    """Count pending interactions in staging file."""
    if not STAGING_FILE.exists():
        return {"total": 0, "reviews": 0, "purchases": 0}
    
    reviews = 0
    purchases = 0
    
    try:
        with open(STAGING_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('interaction_type') == 'review':
                    reviews += 1
                elif row.get('interaction_type') == 'purchase':
                    purchases += 1
    except:
        pass
    
    return {"total": reviews + purchases, "reviews": reviews, "purchases": purchases}


def _get_pending_time_range() -> Dict[str, Optional[str]]:
    """Get oldest and newest pending entry timestamps."""
    if not STAGING_FILE.exists():
        return {"oldest": None, "newest": None}
    
    oldest = None
    newest = None
    
    try:
        with open(STAGING_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get('ingested_at')
                if ts:
                    if oldest is None or ts < oldest:
                        oldest = ts
                    if newest is None or ts > newest:
                        newest = ts
    except:
        pass
    
    return {"oldest": oldest, "newest": newest}


# ============================================================================
# Router
# ============================================================================

ingestion_router = APIRouter(tags=["Data Ingestion"])


@ingestion_router.post("/review", response_model=InteractionResponse)
async def submit_review(request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    Submit a new product review.
    
    The review will be staged for processing by the next scheduled data_refresh job.
    
    Args:
        request: Review data including user_id, product_id, rating, and optional comment
    
    Returns:
        Confirmation with interaction_id for tracking
    """
    interaction_id = _generate_interaction_id()
    now = datetime.now().isoformat()
    
    row = {
        'interaction_id': interaction_id,
        'user_id': request.user_id,
        'product_id': request.product_id,
        'rating': request.rating,
        'comment': request.comment or "",
        'timestamp': request.timestamp,
        'interaction_type': 'review',
        'quantity': 1,
        'ingested_at': now
    }
    
    try:
        _append_to_staging(row)
        
        # Update stats in background
        background_tasks.add_task(_update_stats, 'review')
        
        logger.info(f"Review ingested: user={request.user_id}, product={request.product_id}, rating={request.rating}")
        
        return InteractionResponse(
            status="accepted",
            interaction_id=interaction_id,
            message="Review staged for processing. Will be included in next data refresh.",
            timestamp=now
        )
    except Exception as e:
        logger.error(f"Failed to ingest review: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stage review: {str(e)}")


@ingestion_router.post("/purchase", response_model=InteractionResponse)
async def submit_purchase(request: PurchaseRequest, background_tasks: BackgroundTasks):
    """
    Submit a new purchase (implicit positive feedback).
    
    Purchases are treated as strong positive signals (equivalent to 5-star rating)
    for the recommendation model.
    
    Args:
        request: Purchase data including user_id, product_id, and optional quantity
    
    Returns:
        Confirmation with interaction_id for tracking
    """
    interaction_id = _generate_interaction_id()
    now = datetime.now().isoformat()
    
    row = {
        'interaction_id': interaction_id,
        'user_id': request.user_id,
        'product_id': request.product_id,
        'rating': 5.0,  # Implicit positive rating for purchase
        'comment': "",
        'timestamp': request.timestamp,
        'interaction_type': 'purchase',
        'quantity': request.quantity,
        'ingested_at': now
    }
    
    try:
        _append_to_staging(row)
        
        # Update stats in background
        background_tasks.add_task(_update_stats, 'purchase')
        
        logger.info(f"Purchase ingested: user={request.user_id}, product={request.product_id}, qty={request.quantity}")
        
        return InteractionResponse(
            status="accepted",
            interaction_id=interaction_id,
            message="Purchase staged for processing. Will be included in next data refresh.",
            timestamp=now
        )
    except Exception as e:
        logger.error(f"Failed to ingest purchase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stage purchase: {str(e)}")


@ingestion_router.post("/batch", response_model=BatchResponse)
async def submit_batch(request: BatchInteractionRequest, background_tasks: BackgroundTasks):
    """
    Submit multiple interactions in a single request.
    
    Useful for syncing historical data or batch updates from the web application.
    
    Args:
        request: Batch containing lists of reviews and/or purchases
    
    Returns:
        Summary of ingested interactions
    """
    now = datetime.now().isoformat()
    reviews_count = 0
    purchases_count = 0
    
    try:
        # Process reviews
        for review in (request.reviews or []):
            interaction_id = _generate_interaction_id()
            row = {
                'interaction_id': interaction_id,
                'user_id': review.user_id,
                'product_id': review.product_id,
                'rating': review.rating,
                'comment': review.comment or "",
                'timestamp': review.timestamp,
                'interaction_type': 'review',
                'quantity': 1,
                'ingested_at': now
            }
            _append_to_staging(row)
            reviews_count += 1
        
        # Process purchases
        for purchase in (request.purchases or []):
            interaction_id = _generate_interaction_id()
            row = {
                'interaction_id': interaction_id,
                'user_id': purchase.user_id,
                'product_id': purchase.product_id,
                'rating': 5.0,
                'comment': "",
                'timestamp': purchase.timestamp,
                'interaction_type': 'purchase',
                'quantity': purchase.quantity,
                'ingested_at': now
            }
            _append_to_staging(row)
            purchases_count += 1
        
        # Update stats in background
        background_tasks.add_task(_update_batch_stats, reviews_count, purchases_count)
        
        total = reviews_count + purchases_count
        logger.info(f"Batch ingested: {reviews_count} reviews, {purchases_count} purchases")
        
        return BatchResponse(
            status="accepted",
            total_received=total,
            reviews_count=reviews_count,
            purchases_count=purchases_count,
            message=f"Batch of {total} interactions staged for processing.",
            timestamp=now
        )
    except Exception as e:
        logger.error(f"Failed to ingest batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stage batch: {str(e)}")


@ingestion_router.get("/stats", response_model=IngestionStats)
async def get_ingestion_stats():
    """
    Get ingestion statistics.
    
    Returns counts of pending interactions and processing history.
    """
    stats = _load_stats()
    pending = _count_pending()
    
    file_size_kb = 0.0
    if STAGING_FILE.exists():
        file_size_kb = STAGING_FILE.stat().st_size / 1024
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = stats.get("daily_counts", {}).get(today, 0)
    
    return IngestionStats(
        total_pending=pending["total"],
        reviews_pending=pending["reviews"],
        purchases_pending=pending["purchases"],
        last_ingestion=stats.get("last_ingestion"),
        last_processed=stats.get("last_processed"),
        today_count=today_count,
        staging_file_size_kb=round(file_size_kb, 2)
    )


@ingestion_router.get("/pending", response_model=PendingDataResponse)
async def get_pending_data():
    """
    Get pending data summary.
    
    Shows count and time range of interactions waiting to be processed.
    """
    pending = _count_pending()
    time_range = _get_pending_time_range()
    
    return PendingDataResponse(
        pending_count=pending["total"],
        pending_reviews=pending["reviews"],
        pending_purchases=pending["purchases"],
        oldest_entry=time_range["oldest"],
        newest_entry=time_range["newest"]
    )


@ingestion_router.delete("/clear", response_model=Dict[str, Any])
async def clear_staging():
    """
    Clear all pending (staging) data.
    
    WARNING: This will delete all unprocessed interactions!
    Use with caution, typically only for testing or error recovery.
    """
    try:
        pending = _count_pending()
        
        if STAGING_FILE.exists():
            STAGING_FILE.unlink()
        
        logger.warning(f"Staging cleared: {pending['total']} interactions removed")
        
        return {
            "status": "cleared",
            "removed_count": pending["total"],
            "reviews_removed": pending["reviews"],
            "purchases_removed": pending["purchases"],
            "message": "All staging data has been cleared."
        }
    except Exception as e:
        logger.error(f"Failed to clear staging: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Background Tasks
# ============================================================================

def _update_stats(interaction_type: str):
    """Update ingestion statistics."""
    try:
        stats = _load_stats()
        stats["total_ingested"] = stats.get("total_ingested", 0) + 1
        stats["last_ingestion"] = datetime.now().isoformat()
        
        if interaction_type == "review":
            stats["reviews_ingested"] = stats.get("reviews_ingested", 0) + 1
        elif interaction_type == "purchase":
            stats["purchases_ingested"] = stats.get("purchases_ingested", 0) + 1
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily = stats.get("daily_counts", {})
        daily[today] = daily.get(today, 0) + 1
        stats["daily_counts"] = daily
        
        _save_stats(stats)
    except Exception as e:
        logger.error(f"Failed to update stats: {e}")


def _update_batch_stats(reviews_count: int, purchases_count: int):
    """Update ingestion statistics for batch."""
    try:
        stats = _load_stats()
        total = reviews_count + purchases_count
        
        stats["total_ingested"] = stats.get("total_ingested", 0) + total
        stats["reviews_ingested"] = stats.get("reviews_ingested", 0) + reviews_count
        stats["purchases_ingested"] = stats.get("purchases_ingested", 0) + purchases_count
        stats["last_ingestion"] = datetime.now().isoformat()
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily = stats.get("daily_counts", {})
        daily[today] = daily.get(today, 0) + total
        stats["daily_counts"] = daily
        
        _save_stats(stats)
    except Exception as e:
        logger.error(f"Failed to update batch stats: {e}")

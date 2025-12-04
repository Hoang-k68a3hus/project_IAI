"""
FastAPI Recommendation Service.

This module provides REST API endpoints for the CF recommendation service.

Endpoints:
- GET /health: Health check
- POST /recommend: Single user recommendation
- POST /batch_recommend: Batch recommendation
- POST /similar_items: Similar items
- POST /reload_model: Hot-reload model
- POST /search: Semantic search for products
- POST /search/similar: Find similar products by product ID
- POST /search/profile: Search based on user profile

Usage:
    uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import time
import os
import sys
import asyncio
import numpy as np
from dataclasses import asdict, is_dataclass

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from service.recommender import CFRecommender, get_loader
from service.recommender.cache import get_cache_manager, async_warmup

# Import Scheduler API router
try:
    from service.scheduler_api import scheduler_router
    SCHEDULER_API_AVAILABLE = True
except ImportError as e:
    SCHEDULER_API_AVAILABLE = False
    scheduler_router = None

# Import Data Ingestion API router
try:
    from service.data_ingestion_api import ingestion_router
    INGESTION_API_AVAILABLE = True
except ImportError as e:
    INGESTION_API_AVAILABLE = False
    ingestion_router = None

# ============================================================================
# Logging Setup (must be before evaluation imports)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cf_service")

# ============================================================================
# Security Configuration
# ============================================================================

# Environment variables for security
ENV = os.getenv("ENV", "development")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000"
).split(",")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Import evaluation modules
try:
    from recsys.cf.evaluation import (
        ModelEvaluator, PopularityBaseline, RandomBaseline,
        ModelComparator, StatisticalTester, BootstrapEstimator,
        DiversityMetric, NoveltyMetric, SemanticAlignmentMetric,
        ColdStartCoverageMetric, HybridMetricCollection,
        evaluate_model, compare_models
    )
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evaluation modules not available: {e}")
    EVALUATION_AVAILABLE = False

# Import Smart Search service (lazy initialization)
search_service = None


def get_search_service():
    """Lazy initialization of search service."""
    global search_service
    if search_service is None:
        try:
            from service.search import SmartSearchService
            search_service = SmartSearchService()
            search_service.initialize()
            logger.info("Smart Search service initialized")
        except Exception as e:
            logger.warning(f"Could not initialize search service: {e}")
    return search_service



# ============================================================================
# Global State
# ============================================================================

recommender: Optional[CFRecommender] = None
cache_manager = None  # Cache manager instance
service_metrics_db = None  # Lazy initialization


def get_service_metrics_db():
    """Get or create service metrics database."""
    global service_metrics_db
    if service_metrics_db is None:
        try:
            from recsys.cf.logging_utils import ServiceMetricsDB
            service_metrics_db = ServiceMetricsDB()
        except Exception as e:
            logger.warning(f"Could not initialize metrics DB: {e}")
    return service_metrics_db


# ============================================================================
# Lifespan Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global recommender, cache_manager
    
    logger.info("Starting CF Recommendation Service...")
    
    try:
        recommender = CFRecommender(auto_load=True)
        model_info = recommender.get_model_info()
        
        if model_info.get('model_id'):
            logger.info(f"Loaded model: {model_info.get('model_id')}")
            logger.info(
                f"Users: {model_info.get('num_users')}, "
                f"Items: {model_info.get('num_items')}, "
                f"Trainable: {model_info.get('trainable_users')}"
            )
        else:
            logger.warning(
                "⚠️  SERVICE RUNNING IN EMPTY MODE - No CF model loaded! "
                "Please mount data volumes or upload model artifacts, then call POST /reload_model"
            )
        
        # Initialize cache manager
        cache_manager = get_cache_manager()
        
        # Initialize metrics DB
        get_service_metrics_db()
        logger.info("Service metrics database initialized")
        
        # Warm up caches for cold-start optimization (~91% traffic)
        logger.info("Warming up caches for cold-start path...")
        warmup_stats = await async_warmup(cache_manager)
        logger.info(
            f"Cache warmup complete: {warmup_stats.get('popular_items', 0)} popular items, "
            f"{warmup_stats.get('popular_similarities', 0)} similarities precomputed, "
            f"duration={warmup_stats.get('warmup_duration_ms', 0):.1f}ms"
        )
        
        # Warm up search service (pre-load PhoBERT model to avoid 30s latency on first query)
        logger.info("Warming up search service (PhoBERT model)...")
        try:
            from service.search import get_search_service
            search_service = get_search_service()
            # Trigger model loading with a simple query
            import time
            warmup_start = time.perf_counter()
            _ = search_service.search("kem dưỡng da", topk=1)
            warmup_elapsed = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Search service warmup complete: PhoBERT loaded in {warmup_elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"Search service warmup failed (non-critical): {e}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize recommender: {e}")
        logger.warning(
            "⚠️  SERVICE RUNNING IN EMPTY MODE - Initialization failed! "
            "Please check data volumes and model artifacts, then restart or call POST /reload_model"
        )
        # Don't raise - allow service to start in empty mode
        recommender = CFRecommender(auto_load=False)  # Empty recommender
    
    # Start background health aggregation task
    aggregation_task = asyncio.create_task(periodic_health_aggregation())
    
    yield
    
    # Cleanup
    aggregation_task.cancel()
    logger.info("Shutting down CF Recommendation Service...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CF Recommendation Service",
    description="Collaborative Filtering recommendation API for Vietnamese cosmetics",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include Scheduler API router
if SCHEDULER_API_AVAILABLE and scheduler_router is not None:
    app.include_router(scheduler_router, prefix="/scheduler", tags=["Scheduler"])
    logger.info("Scheduler API endpoints enabled at /scheduler/*")

# Include Data Ingestion API router
if INGESTION_API_AVAILABLE and ingestion_router is not None:
    app.include_router(ingestion_router, prefix="/ingest", tags=["Data Ingestion"])
    logger.info("Data Ingestion API endpoints enabled at /ingest/*")

# CORS middleware - Security: Only allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ENV == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Request ID for tracing
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    response.headers["X-Request-ID"] = request_id
    
    return response

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent DoS attacks."""
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    
    if request.headers.get("content-length"):
        try:
            size = int(request.headers["content-length"])
            if size > MAX_REQUEST_SIZE:
                logger.warning(f"Request too large: {size} bytes from {get_remote_address(request)}")
                return Response(
                    content='{"detail": "Request body too large. Maximum size is 10MB."}',
                    status_code=413,
                    media_type="application/json"
                )
        except (ValueError, TypeError):
            pass
    
    return await call_next(request)


# ============================================================================
# Security Helper Functions
# ============================================================================

def sanitize_error_message(error: Exception, endpoint: str = "") -> str:
    """
    Sanitize error messages for production.
    
    In production, don't expose internal error details.
    In development, show full error for debugging.
    """
    if ENV == "production":
        # Generic error messages in production
        if "not initialized" in str(error).lower():
            return "Service temporarily unavailable"
        elif "not found" in str(error).lower() or "invalid" in str(error).lower():
            return "Invalid request parameters"
        else:
            return "An error occurred processing your request"
    else:
        # Full error in development
        return str(error)


# ============================================================================
# Background Tasks
# ============================================================================

async def periodic_health_aggregation():
    """Periodically aggregate health metrics."""
    while True:
        try:
            await asyncio.sleep(60)  # Every minute
            db = get_service_metrics_db()
            if db and recommender:
                db.aggregate_health_metrics(recommender.model_id)
                logger.debug("Health metrics aggregated")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health aggregation error: {e}")


def log_request_metrics(
    user_id: int,
    topk: int,
    latency_ms: float,
    num_recommendations: int,
    fallback: bool,
    fallback_method: Optional[str] = None,
    rerank_enabled: bool = False,
    error: Optional[str] = None
):
    """Log request metrics to database (in background)."""
    try:
        db = get_service_metrics_db()
        if db:
            db.log_request(
                user_id=user_id,
                topk=topk,
                latency_ms=latency_ms,
                num_recommendations=num_recommendations,
                fallback=fallback,
                model_id=recommender.model_id if recommender else None,
                fallback_method=fallback_method,
                rerank_enabled=rerank_enabled,
                error=error
            )
    except Exception as e:
        logger.warning(f"Failed to log request metrics: {e}")


# ============================================================================
# Request/Response Models
# ============================================================================

class APIBaseModel(BaseModel):
    """Base model with JSON encoders for numpy types."""
    model_config = ConfigDict(
        json_encoders={
            np.integer: int,
            np.floating: float,
            np.ndarray: lambda v: v.tolist(),
            np.bool_: bool,
        }
    )


def _sanitize_numpy_types(value: Any) -> Any:
    """
    Recursively convert numpy/scalar-like values into native Python types.
    Ensures downstream JSON serialization never raises Pydantic errors.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_numpy_types(v) for v in value.tolist()]
    if is_dataclass(value):
        return _sanitize_numpy_types(asdict(value))
    if isinstance(value, dict):
        return {k: _sanitize_numpy_types(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_numpy_types(v) for v in value]
    return value


class RecommendRequest(APIBaseModel):
    """Single user recommendation request."""
    user_id: int = Field(..., ge=0, description="User ID (must be non-negative)")
    topk: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    exclude_seen: bool = Field(default=True, description="Exclude items user has interacted with")
    filter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Attribute filters, e.g., {'brand': 'Innisfree'}"
    )
    rerank: bool = Field(default=False, description="Apply hybrid reranking")
    rerank_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reranking weights: {cf, popularity, quality, content}"
    )


class RecommendResponse(APIBaseModel):
    """Recommendation response."""
    user_id: int
    recommendations: List[Dict[str, Any]]
    count: int
    is_fallback: bool
    fallback_method: Optional[str]
    latency_ms: float
    model_id: Optional[str]


class BatchRequest(APIBaseModel):
    """Batch recommendation request."""
    user_ids: List[int] = Field(..., description="List of user IDs")
    topk: int = Field(default=10, ge=1, le=100)
    exclude_seen: bool = Field(default=True)


class BatchResponse(APIBaseModel):
    """Batch recommendation response."""
    results: Dict[int, Dict[str, Any]]
    num_users: int
    total_latency_ms: float
    cf_users: int
    fallback_users: int


class SimilarItemsRequest(APIBaseModel):
    """Similar items request."""
    product_id: int = Field(..., description="Query product ID")
    topk: int = Field(default=10, ge=1, le=50)
    use_cf: bool = Field(default=True, description="Use CF embeddings (else PhoBERT)")


class SimilarItemsResponse(APIBaseModel):
    """Similar items response."""
    product_id: int
    similar_items: List[Dict[str, Any]]
    count: int
    method: str


class HealthResponse(APIBaseModel):
    """Health check response."""
    status: str
    model_id: Optional[str]
    model_type: Optional[str]
    num_users: int
    num_items: int
    trainable_users: int
    timestamp: str
    empty_mode: bool = False  # True if no model loaded


class ReloadResponse(APIBaseModel):
    """Model reload response."""
    status: str
    previous_model_id: Optional[str]
    new_model_id: Optional[str]
    reloaded: bool


# ============================================================================
# Smart Search Request/Response Models
# ============================================================================

class SearchRequest(APIBaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query in Vietnamese")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Attribute filters: {brand, category, min_price, max_price}"
    )
    rerank: bool = Field(default=True, description="Apply hybrid reranking")


class SearchSimilarRequest(APIBaseModel):
    """Similar products search request."""
    product_id: int = Field(..., description="Product ID to find similar products")
    topk: int = Field(default=10, ge=1, le=50, description="Number of similar products")
    exclude_self: bool = Field(default=True, description="Exclude the query product from results")


class SearchByProfileRequest(APIBaseModel):
    """Search based on user profile/history."""
    product_history: List[int] = Field(..., min_length=1, description="List of product IDs user has interacted with")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    exclude_history: bool = Field(default=True, description="Exclude products in history from results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Attribute filters")


class SearchResultItem(APIBaseModel):
    """Single search result item."""
    rank: int
    product_id: int
    product_name: str
    brand: Optional[str]
    category: Optional[str]
    price: Optional[float]
    avg_rating: Optional[float]
    num_sold: Optional[int]
    semantic_score: float
    final_score: float


class SearchResponse(APIBaseModel):
    """Search response."""
    query: str
    results: List[SearchResultItem]
    count: int
    method: str
    latency_ms: float
    available_filters: Optional[Dict[str, Any]] = None


# ============================================================================
# Evaluation Request/Response Models
# ============================================================================

class EvaluateModelRequest(APIBaseModel):
    """Model evaluation request."""
    k_values: List[int] = Field(default=[10, 20], description="K values for @K metrics")
    test_data: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Test data: {'user_id': [item_ids]}. If None, uses internal test split."
    )
    user_pos_train: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Training positive items per user: {'user_id': [item_ids]}. If None, uses internal data."
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to compute. If None, computes all standard metrics."
    )


class EvaluateModelResponse(APIBaseModel):
    """Model evaluation response."""
    model_id: str
    model_type: str
    results: Dict[str, float]
    evaluation_time_ms: float
    num_test_users: int
    k_values: List[int]


class CompareModelsRequest(APIBaseModel):
    """Model comparison request."""
    baseline_names: List[str] = Field(
        default=["popularity", "random"],
        description="Baselines to compare: 'popularity', 'random'"
    )
    k_values: List[int] = Field(default=[10, 20])
    test_data: Optional[Dict[str, List[int]]] = None
    user_pos_train: Optional[Dict[str, List[int]]] = None


class CompareModelsResponse(APIBaseModel):
    """Model comparison response."""
    model_id: str
    model_results: Dict[str, float]
    baseline_results: Dict[str, Dict[str, float]]
    improvements: Dict[str, Dict[str, Any]]
    comparison_time_ms: float


class StatisticalTestRequest(APIBaseModel):
    """Statistical test request."""
    model1_scores: List[float] = Field(..., description="Scores from model 1")
    model2_scores: List[float] = Field(..., description="Scores from model 2")
    test_type: str = Field(
        default="paired_t_test",
        description="Test type: 'paired_t_test', 'wilcoxon_test'"
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.1)


class StatisticalTestResponse(APIBaseModel):
    """Statistical test response."""
    test_type: str
    p_value: float
    significant: bool
    effect_size: Optional[Dict[str, Any]] = None
    confidence_interval: Optional[Dict[str, float]] = None


class ComputeMetricsRequest(APIBaseModel):
    """Compute specific metrics request."""
    predictions: List[List[int]] = Field(..., description="Predictions per user")
    ground_truth: List[List[int]] = Field(..., description="Ground truth per user (will be converted to sets)")
    metric: str = Field(..., description="Metric name: 'recall', 'ndcg', 'precision', 'mrr', 'map'")
    k: int = Field(default=10, ge=1, le=100)


class ComputeMetricsResponse(APIBaseModel):
    """Compute metrics response."""
    metric: str
    k: int
    value: float
    per_user: List[float]
    mean: float
    std: float
    min: float
    max: float


class HybridMetricsRequest(APIBaseModel):
    """Hybrid metrics request."""
    recommendations: Dict[int, List[int]] = Field(
        ...,
        description="Recommendations per user: {user_id: [item_ids]}"
    )
    item_embeddings_path: Optional[str] = Field(
        default=None,
        description="Path to item embeddings .npy file. If None, uses model embeddings."
    )
    metrics: List[str] = Field(
        default=["diversity", "novelty", "cold_start_coverage"],
        description="Metrics to compute: 'diversity', 'novelty', 'alignment', 'cold_start_coverage', 'serendipity'"
    )
    k_values: List[int] = Field(default=[10, 20])
    cold_threshold: int = Field(default=5, description="Threshold for cold-start items")


class HybridMetricsResponse(APIBaseModel):
    """Hybrid metrics response."""
    results: Dict[str, Any]
    computation_time_ms: float
    num_users: int


class GenerateReportRequest(APIBaseModel):
    """Generate evaluation report request."""
    model_results: Dict[str, float]
    baseline_results: Optional[Dict[str, Dict[str, float]]] = None
    format: str = Field(default="markdown", description="Report format: 'markdown', 'json', 'csv'")
    include_statistics: bool = Field(default=True, description="Include statistical tests")


class GenerateReportResponse(APIBaseModel):
    """Generate report response."""
    report_path: str
    format: str
    file_size_bytes: int
    download_url: str


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and model status.
    
    Returns:
        Health status with model information
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    model_info = recommender.get_model_info()
    
    # Determine status based on whether model is loaded
    is_empty_mode = model_info.get('model_id') is None
    status = "degraded" if is_empty_mode else "healthy"
    
    payload = _sanitize_numpy_types(
        {
            "status": status,
            "model_id": model_info.get('model_id'),
            "model_type": model_info.get('model_type'),
            "num_users": model_info.get('num_users', 0),
            "num_items": model_info.get('num_items', 0),
            "trainable_users": model_info.get('trainable_users', 0),
            "timestamp": datetime.now().isoformat(),
            "empty_mode": is_empty_mode,
        }
    )

    return HealthResponse(**payload)


@app.post("/recommend", response_model=RecommendResponse)
@limiter.limit("60/minute")  # 60 requests per minute per IP
async def recommend(request: Request, recommend_request: RecommendRequest):
    """
    Get recommendations for a single user.
    
    Args:
        request: FastAPI Request object
        recommend_request: RecommendRequest with user_id, topk, filters, etc.
    
    Returns:
        RecommendResponse with recommendations
    
    Example:
        POST /recommend
        {
            "user_id": 12345,
            "topk": 10,
            "exclude_seen": true,
            "filter_params": {"brand": "Innisfree"}
        }
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        result = recommender.recommend(
            user_id=recommend_request.user_id,
            topk=recommend_request.topk,
            exclude_seen=recommend_request.exclude_seen,
            filter_params=recommend_request.filter_params
        )
        
        # Apply reranking if requested
        if recommend_request.rerank and result.recommendations:
            from service.recommender.rerank import rerank_with_signals
            
            result.recommendations = rerank_with_signals(
                recommendations=result.recommendations,
                user_id=recommend_request.user_id,
                weights=recommend_request.rerank_weights,
                score_range=recommender.score_range
            )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Log request to console
        logger.info(
            f"user_id={recommend_request.user_id}, topk={recommend_request.topk}, "
            f"count={result.count}, fallback={result.is_fallback}, "
            f"latency={latency:.1f}ms"
        )
        
        # Log to metrics DB (background)
        log_request_metrics(
            user_id=recommend_request.user_id,
            topk=recommend_request.topk,
            latency_ms=latency,
            num_recommendations=result.count,
            fallback=result.is_fallback,
            fallback_method=result.fallback_method,
            rerank_enabled=recommend_request.rerank
        )
        
        response_payload = _sanitize_numpy_types({
            "user_id": result.user_id,
            "recommendations": result.recommendations,
            "count": result.count,
            "is_fallback": result.is_fallback,
            "fallback_method": result.fallback_method,
            "latency_ms": latency,
            "model_id": result.model_id,
        })

        return RecommendResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error for user {recommend_request.user_id}: {e}", exc_info=True)
        # Log error to metrics
        log_request_metrics(
            user_id=recommend_request.user_id,
            topk=recommend_request.topk,
            latency_ms=0,
            num_recommendations=0,
            fallback=False,
            error=str(e)
        )
        error_detail = sanitize_error_message(e, "recommend")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/batch_recommend", response_model=BatchResponse)
@limiter.limit("20/minute")  # Lower limit for batch operations
async def batch_recommend(request: Request, batch_request: BatchRequest):
    """
    Get recommendations for multiple users.
    
    Args:
        request: BatchRequest with list of user_ids
    
    Returns:
        BatchResponse with results for all users
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate batch size
    if len(batch_request.user_ids) > 1000:
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large. Maximum 1000 users per batch."
        )
    
    try:
        start_time = time.perf_counter()
        
        results = recommender.batch_recommend(
            user_ids=batch_request.user_ids,
            topk=batch_request.topk,
            exclude_seen=batch_request.exclude_seen
        )
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Convert results to dict format
        results_dict = {}
        cf_count = 0
        fallback_count = 0
        
        for uid, result in results.items():
            results_dict[uid] = {
                'recommendations': result.recommendations,
                'count': result.count,
                'is_fallback': result.is_fallback,
                'fallback_method': result.fallback_method,
            }
            
            if result.is_fallback:
                fallback_count += 1
            else:
                cf_count += 1
        
        logger.info(
            f"Batch recommendation: {len(batch_request.user_ids)} users, "
            f"cf={cf_count}, fallback={fallback_count}, "
            f"latency={total_latency:.1f}ms"
        )
        
        response_payload = _sanitize_numpy_types({
            "results": results_dict,
            "num_users": len(batch_request.user_ids),
            "total_latency_ms": total_latency,
            "cf_users": cf_count,
            "fallback_users": fallback_count,
        })

        return BatchResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch recommendation error: {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "batch_recommend")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/similar_items", response_model=SimilarItemsResponse)
async def similar_items(request: SimilarItemsRequest):
    """
    Find similar items to a given product.
    
    Args:
        request: SimilarItemsRequest with product_id
    
    Returns:
        SimilarItemsResponse with similar items
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        similar = recommender.similar_items(
            product_id=request.product_id,
            topk=request.topk,
            use_cf=request.use_cf
        )
        
        method = "cf_embeddings" if request.use_cf else "phobert"
        
        response_payload = _sanitize_numpy_types({
            "product_id": request.product_id,
            "similar_items": similar,
            "count": len(similar),
            "method": method,
        })

        return SimilarItemsResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar items error for product {request.product_id}: {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "similar_items")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/reload_model", response_model=ReloadResponse)
async def reload_model():
    """
    Hot-reload model from registry.
    
    Checks if a new best model is available and reloads if so.
    
    Returns:
        ReloadResponse with reload status
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        previous_model_id = recommender.model_id
        reloaded = recommender.reload_model()
        new_model_id = recommender.model_id
        
        status = "reloaded" if reloaded else "no_update"
        
        logger.info(
            f"Model reload: {status}, "
            f"previous={previous_model_id}, new={new_model_id}"
        )
        
        return ReloadResponse(
            status=status,
            previous_model_id=previous_model_id,
            new_model_id=new_model_id,
            reloaded=reloaded
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model reload error: {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "reload_model")
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/model_info")
async def model_info():
    """Get detailed model information."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return _sanitize_numpy_types(recommender.get_model_info())


@app.get("/stats")
async def service_stats():
    """Get service statistics."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    loader = recommender.loader
    
    # Get cache stats
    cache_stats = {}
    if cache_manager is not None:
        cache_stats = cache_manager.get_stats()
    
    stats_payload = {
        "model_id": recommender.model_id,
        "total_users": loader.mappings['metadata']['num_users'] if loader.mappings else 0,
        "trainable_users": len(loader.trainable_user_set or set()),
        "cold_start_users": (
            loader.mappings['metadata']['num_users'] - len(loader.trainable_user_set or set())
            if loader.mappings else 0
        ),
        "trainable_percentage": (
            len(loader.trainable_user_set or set()) / 
            max(1, loader.mappings['metadata']['num_users']) * 100
            if loader.mappings else 0
        ),
        "num_items": loader.mappings['metadata']['num_items'] if loader.mappings else 0,
        "popular_items_cached": len(loader.top_k_popular_items or []),
        "user_histories_cached": len(loader.user_history_cache or {}),
        "cache": cache_stats
    }
    
    return _sanitize_numpy_types(stats_payload)


@app.get("/cache_stats")
async def cache_stats():
    """Get detailed cache statistics."""
    if cache_manager is None:
        return {"status": "not_initialized"}
    
    return _sanitize_numpy_types(cache_manager.get_stats())


@app.post("/cache_warmup")
async def trigger_warmup(force: bool = False):
    """Trigger cache warmup."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    stats = await async_warmup(cache_manager)
    return _sanitize_numpy_types(stats)


@app.post("/cache_clear")
async def clear_cache():
    """Clear all caches."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    cache_manager.clear_all()
    return {"status": "cleared"}


# ============================================================================
# Smart Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse)
@limiter.limit("100/minute")  # Higher limit for search
async def search_products(request: Request, search_request: SearchRequest):
    """
    Semantic search for products using Vietnamese query.
    
    This endpoint uses PhoBERT embeddings to find products semantically
    similar to the search query. Supports Vietnamese text with automatic
    abbreviation expansion (e.g., "srm" → "sữa rửa mặt").
    
    Args:
        request: FastAPI Request object
        search_request: SearchRequest with query, topk, filters, rerank
    
    Returns:
        SearchResponse with ranked results
    
    Example:
        POST /search
        {
            "query": "kem dưỡng ẩm cho da khô",
            "topk": 10,
            "filters": {"brand": "Innisfree"},
            "rerank": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search(
            query=search_request.query,
            topk=search_request.topk,
            filters=search_request.filters,
            rerank=search_request.rerank
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Search: query='{search_request.query[:50]}...', "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=search_request.query,
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error for query '{search_request.query}': {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "search")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_products(request: SearchSimilarRequest):
    """
    Find products similar to a given product.
    
    Uses PhoBERT semantic embeddings to find products with similar
    descriptions, ingredients, and features.
    
    Args:
        request: SearchSimilarRequest with product_id, topk
    
    Returns:
        SearchResponse with similar products
    
    Example:
        POST /search/similar
        {
            "product_id": 12345,
            "topk": 10,
            "exclude_self": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search_similar(
            product_id=request.product_id,
            topk=request.topk,
            exclude_self=request.exclude_self
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Similar search: product_id={request.product_id}, "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=f"similar_to:{request.product_id}",
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar search error for product {request.product_id}: {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "search_similar")
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/search/profile", response_model=SearchResponse)
async def search_by_profile(request: SearchByProfileRequest):
    """
    Search based on user profile/history.
    
    Computes an average embedding from the user's product history
    and finds products semantically similar to their interests.
    
    Args:
        request: SearchByProfileRequest with product_history, topk
    
    Returns:
        SearchResponse with personalized recommendations
    
    Example:
        POST /search/profile
        {
            "product_history": [123, 456, 789],
            "topk": 10,
            "exclude_history": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search_by_user_profile(
            user_history=request.product_history,
            topk=request.topk,
            exclude_history=request.exclude_history,
            filters=request.filters
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Profile search: history_size={len(request.product_history)}, "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=f"profile:{len(request.product_history)}_products",
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile search error: {e}", exc_info=True)
        error_detail = sanitize_error_message(e, "search_profile")
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/search/filters")
async def get_search_filters():
    """
    Get available filter options for search.
    
    Returns:
        Dict with available brands, categories, and price range
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        return _sanitize_numpy_types(service.get_available_filters())
    except Exception as e:
        logger.error(f"Get filters error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/stats")
async def get_search_stats():
    """
    Get search service statistics.
    
    Returns:
        Search performance metrics and cache statistics
    """
    service = get_search_service()
    if service is None:
        return {"status": "not_initialized"}
    
    return _sanitize_numpy_types(service.get_stats())


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@app.post("/evaluate/model", response_model=EvaluateModelResponse)
async def evaluate_model_endpoint(request: EvaluateModelRequest):
    """
    Evaluate model performance on test data.
    
    Computes standard metrics (Recall@K, NDCG@K, Precision@K, MRR, MAP@K)
    for the current model.
    
    Args:
        request: EvaluateModelRequest with k_values, test_data, etc.
    
    Returns:
        EvaluateModelResponse with evaluation results
    
    Example:
        POST /evaluate/model
        {
            "k_values": [10, 20],
            "metrics": ["recall", "ndcg"]
        }
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        start_time = time.perf_counter()
        
        # Get model embeddings from current_model dict
        loader = recommender.loader
        if loader.current_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model embeddings not available. Model may not be loaded."
            )
        
        U = loader.current_model['U']
        V = loader.current_model['V']
        
        # Prepare test data
        if request.test_data is None or request.user_pos_train is None:
            # Use internal test split if available
            # For now, return error - user should provide test data
            raise HTTPException(
                status_code=400,
                detail="test_data and user_pos_train are required. Use internal evaluation scripts for automatic test split."
            )
        
        # Convert string keys to int and lists to sets
        test_data = {
            int(uid): set(items) 
            for uid, items in request.test_data.items()
        }
        user_pos_train = {
            int(uid): set(items)
            for uid, items in request.user_pos_train.items()
        }
        
        # Validate user IDs exist in model
        num_users = U.shape[0]
        num_items = V.shape[0]
        invalid_users = [uid for uid in test_data.keys() if uid >= num_users or uid < 0]
        if invalid_users:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid user IDs: {invalid_users}. Model has {num_users} users (0-{num_users-1})"
            )
        
        # Validate item IDs
        all_test_items = set()
        for items in test_data.values():
            all_test_items.update(items)
        for items in user_pos_train.values():
            all_test_items.update(items)
        invalid_items = [item for item in all_test_items if item >= num_items or item < 0]
        if invalid_items:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid item IDs: {list(invalid_items)[:10]}. Model has {num_items} items (0-{num_items-1})"
            )
        
        # Create evaluator
        evaluator = ModelEvaluator(U, V, k_values=request.k_values)
        
        # Evaluate
        results = evaluator.evaluate(
            test_data=test_data,
            user_pos_train=user_pos_train
        )
        
        # Filter metrics if requested
        if request.metrics:
            filtered_results = {}
            for metric_name in request.metrics:
                for k in request.k_values:
                    key = f"{metric_name}@{k}"
                    if key in results:
                        filtered_results[key] = results[key]
            results = filtered_results
        
        evaluation_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Model evaluation: model_id={recommender.model_id}, "
            f"k_values={request.k_values}, time={evaluation_time:.1f}ms"
        )
        
        # Get model_type from loader
        model_type = "unknown"
        if recommender.loader.current_model:
            model_type = recommender.loader.current_model.get('model_type', 'unknown')
        
        return EvaluateModelResponse(
            model_id=recommender.model_id or "unknown",
            model_type=model_type,
            results=results,
            evaluation_time_ms=evaluation_time,
            num_test_users=len(test_data),
            k_values=request.k_values
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/compare", response_model=CompareModelsResponse)
async def compare_models_endpoint(request: CompareModelsRequest):
    """
    Compare model with baselines.
    
    Evaluates the current model and specified baselines, then computes
    improvement percentages.
    
    Args:
        request: CompareModelsRequest with baseline names, k_values, etc.
    
    Returns:
        CompareModelsResponse with comparison results
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        start_time = time.perf_counter()
        
        # Get model embeddings from current_model dict
        loader = recommender.loader
        if loader.current_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model embeddings not available"
            )
        
        U = loader.current_model['U']
        V = loader.current_model['V']
        
        if request.test_data is None or request.user_pos_train is None:
            raise HTTPException(
                status_code=400,
                detail="test_data and user_pos_train are required"
            )
        
        # Convert string keys to int and lists to sets
        test_data = {
            int(uid): set(items) 
            for uid, items in request.test_data.items()
        }
        user_pos_train = {
            int(uid): set(items)
            for uid, items in request.user_pos_train.items()
        }
        
        # Evaluate model
        model_evaluator = ModelEvaluator(U, V, k_values=request.k_values)
        model_results = model_evaluator.evaluate(
            test_data=test_data,
            user_pos_train=user_pos_train
        )
        
        # Evaluate baselines
        baseline_results = {}
        loader_data = loader.mappings if loader.mappings else {}
        num_items = loader_data.get('metadata', {}).get('num_items', V.shape[0])
        
        # Get item popularity if needed
        item_popularity = None
        if "popularity" in request.baseline_names:
            try:
                # Try to get popularity from loader
                if hasattr(loader, 'item_popularity'):
                    item_popularity = loader.item_popularity
                else:
                    # Compute from user histories if available
                    if hasattr(loader, 'user_history_cache'):
                        item_counts = {}
                        for history in loader.user_history_cache.values():
                            for item in history:
                                item_counts[item] = item_counts.get(item, 0) + 1
                        item_popularity = np.array([item_counts.get(i, 0) for i in range(num_items)])
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Cannot compute popularity baseline: item popularity not available"
                        )
            except Exception as e:
                logger.warning(f"Could not get item popularity: {e}")
        
        for baseline_name in request.baseline_names:
            try:
                if baseline_name == "popularity" and item_popularity is not None:
                    baseline = PopularityBaseline(item_popularity, num_items)
                    baseline_results[baseline_name] = baseline.evaluate(
                        test_data=test_data,
                        user_pos_train=user_pos_train,
                        k_values=request.k_values
                    )
                elif baseline_name == "random":
                    baseline = RandomBaseline(num_items)
                    baseline_results[baseline_name] = baseline.evaluate(
                        test_data=test_data,
                        user_pos_train=user_pos_train,
                        k_values=request.k_values
                    )
                else:
                    logger.warning(f"Unknown baseline: {baseline_name}")
            except Exception as e:
                logger.error(f"Error evaluating baseline {baseline_name}: {e}")
                baseline_results[baseline_name] = {}
        
        # Compute improvements
        improvements = {}
        comparator = ModelComparator()
        comparator.add_model_results("current_model", model_results)
        
        for baseline_name, baseline_res in baseline_results.items():
            comparator.add_baseline_results(baseline_name, baseline_res)
            
            # Compute improvements for each metric
            baseline_improvements = {}
            for k in request.k_values:
                for metric in ["recall", "ndcg", "precision"]:
                    metric_key = f"{metric}@{k}"
                    if metric_key in model_results and metric_key in baseline_res:
                        try:
                            improvement = comparator.compute_improvement(
                                "current_model", baseline_name, metric_key
                            )
                            baseline_improvements[metric_key] = {
                                "model_value": improvement["model_value"],
                                "baseline_value": improvement["baseline_value"],
                                "absolute_improvement": improvement["absolute_improvement"],
                                "relative_percent": improvement["relative_percent"]
                            }
                        except Exception as e:
                            logger.warning(f"Could not compute improvement for {metric_key}: {e}")
            
            improvements[baseline_name] = baseline_improvements
        
        comparison_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Model comparison: model_id={recommender.model_id}, "
            f"baselines={request.baseline_names}, time={comparison_time:.1f}ms"
        )
        
        response_payload = _sanitize_numpy_types({
            "model_id": recommender.model_id or "unknown",
            "model_results": model_results,
            "baseline_results": baseline_results,
            "improvements": improvements,
            "comparison_time_ms": comparison_time,
        })
        
        return CompareModelsResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/statistical_test", response_model=StatisticalTestResponse)
async def statistical_test_endpoint(request: StatisticalTestRequest):
    """
    Perform statistical significance testing.
    
    Compares two sets of scores using paired t-test or Wilcoxon test.
    
    Args:
        request: StatisticalTestRequest with scores and test type
    
    Returns:
        StatisticalTestResponse with test results
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    try:
        import numpy as np
        
        model1_scores = np.array(request.model1_scores)
        model2_scores = np.array(request.model2_scores)
        
        if len(model1_scores) != len(model2_scores):
            raise HTTPException(
                status_code=400,
                detail="model1_scores and model2_scores must have same length"
            )
        
        tester = StatisticalTester(significance_level=request.significance_level)
        
        # Perform test
        if request.test_type == "paired_t_test":
            result = tester.paired_t_test(model1_scores, model2_scores)
        elif request.test_type == "wilcoxon_test":
            result = tester.wilcoxon_test(model1_scores, model2_scores)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown test type: {request.test_type}"
            )
        
        # Compute effect size
        effect_size = tester.cohens_d(model1_scores, model2_scores)
        
        # Compute confidence interval
        ci = tester.confidence_interval(model1_scores - model2_scores)
        
        logger.info(
            f"Statistical test: type={request.test_type}, "
            f"p_value={result['p_value']:.4f}, significant={result['significant']}"
        )
        
        response_payload = _sanitize_numpy_types({
            "test_type": request.test_type,
            "p_value": result['p_value'],
            "significant": result['significant'],
            "effect_size": effect_size,
            "confidence_interval": {
                "lower": ci["lower"],
                "upper": ci["upper"],
            },
        })
        
        return StatisticalTestResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistical test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/metrics", response_model=ComputeMetricsResponse)
async def compute_metrics_endpoint(request: ComputeMetricsRequest):
    """
    Compute specific metrics for given predictions and ground truth.
    
    Args:
        request: ComputeMetricsRequest with predictions, ground_truth, metric, k
    
    Returns:
        ComputeMetricsResponse with metric values
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    try:
        from recsys.cf.evaluation.metrics import MetricFactory
        import numpy as np
        
        if len(request.predictions) != len(request.ground_truth):
            raise HTTPException(
                status_code=400,
                detail="predictions and ground_truth must have same length"
            )
        
        # Create metric
        metric_name = request.metric.lower()
        if metric_name == "mrr":
            metric = MetricFactory.create("mrr")
        else:
            metric = MetricFactory.create(metric_name, k=request.k)
        
        # Compute per-user
        per_user_scores = []
        for preds, gt_list in zip(request.predictions, request.ground_truth):
            gt = set(gt_list)  # Convert to set
            if metric_name == "mrr":
                score = metric.compute(preds, gt)
            else:
                score = metric.compute(preds, gt, k=request.k)
            per_user_scores.append(score)
        
        per_user_scores = np.array(per_user_scores)
        
        logger.info(
            f"Computed {metric_name}@{request.k}: "
            f"mean={np.mean(per_user_scores):.4f}, "
            f"std={np.std(per_user_scores):.4f}"
        )
        
        response_payload = _sanitize_numpy_types({
            "metric": metric_name,
            "k": request.k,
            "value": float(np.mean(per_user_scores)),
            "per_user": per_user_scores.tolist(),
            "mean": float(np.mean(per_user_scores)),
            "std": float(np.std(per_user_scores)),
            "min": float(np.min(per_user_scores)),
            "max": float(np.max(per_user_scores)),
        })
        
        return ComputeMetricsResponse(**response_payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compute metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/hybrid", response_model=HybridMetricsResponse)
async def hybrid_metrics_endpoint(request: HybridMetricsRequest):
    """
    Compute hybrid metrics (diversity, novelty, etc.).
    
    Args:
        request: HybridMetricsRequest with recommendations, embeddings path, etc.
    
    Returns:
        HybridMetricsResponse with hybrid metric results
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    try:
        import numpy as np
        
        start_time = time.perf_counter()
        
        # Convert string keys to int and validate
        try:
            recommendations = {
                int(uid): items
                for uid, items in request.recommendations.items()
            }
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid recommendations format: {e}. Expected {{'user_id': [item_ids]}}"
            )
        
        if len(recommendations) == 0:
            raise HTTPException(
                status_code=400,
                detail="recommendations cannot be empty"
            )
        
        # Load item embeddings
        if request.item_embeddings_path:
            item_embeddings = np.load(request.item_embeddings_path)
        else:
            # Try to use model embeddings from current_model dict
            if recommender is None:
                raise HTTPException(
                    status_code=400,
                    detail="item_embeddings_path required when recommender not available"
                )
            loader = recommender.loader
            if loader.current_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="Model embeddings not available. Provide item_embeddings_path."
                )
            item_embeddings = loader.current_model['V']
        
        results = {}
        
        # Compute diversity
        if "diversity" in request.metrics:
            diversity_metric = DiversityMetric()
            diversity_batch = diversity_metric.compute_batch(
                recommendations, item_embeddings
            )
            results["diversity"] = diversity_batch
        
        # Compute novelty (requires item popularity)
        if "novelty" in request.metrics:
            # Try to get item popularity
            if recommender and hasattr(recommender.loader, 'item_popularity'):
                item_popularity = recommender.loader.item_popularity
            else:
                # Estimate from recommendations (not ideal but works)
                item_counts = {}
                for recs in recommendations.values():
                    for item in recs:
                        item_counts[item] = item_counts.get(item, 0) + 1
                item_popularity = np.array([
                    item_counts.get(i, 1) for i in range(len(item_embeddings))
                ])
            
            num_users = len(recommendations)
            novelty_results = {}
            for k in request.k_values:
                novelty_metric = NoveltyMetric(k=k)
                novelties = []
                for recs in recommendations.values():
                    nov = novelty_metric.compute(recs, item_popularity, num_users)
                    novelties.append(nov)
                novelty_results[f"novelty@{k}"] = {
                    "mean": float(np.mean(novelties)),
                    "std": float(np.std(novelties))
                }
            results["novelty"] = novelty_results
        
        # Compute cold-start coverage
        if "cold_start_coverage" in request.metrics:
            # Get item counts
            if recommender and hasattr(recommender.loader, 'item_popularity'):
                item_counts = recommender.loader.item_popularity
            else:
                item_counts = np.array([
                    sum(1 for recs in recommendations.values() if i in recs)
                    for i in range(len(item_embeddings))
                ])
            
            cold_metric = ColdStartCoverageMetric(cold_threshold=request.cold_threshold)
            cold_coverage = cold_metric.compute(recommendations, item_counts)
            cold_stats = cold_metric.get_cold_item_stats(item_counts, request.cold_threshold)
            results["cold_start_coverage"] = {
                "coverage": float(cold_coverage),
                "stats": cold_stats
            }
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Hybrid metrics computed: metrics={request.metrics}, "
            f"time={computation_time:.1f}ms"
        )
        
        return HybridMetricsResponse(
            results=results,
            computation_time_ms=computation_time,
            num_users=len(recommendations)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/report", response_model=GenerateReportResponse)
async def generate_report_endpoint(request: GenerateReportRequest):
    """
    Generate evaluation report in specified format.
    
    Args:
        request: GenerateReportRequest with results and format
    
    Returns:
        GenerateReportResponse with report path and download URL
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation modules not available")
    
    try:
        from recsys.cf.evaluation.comparison import ReportGenerator
        from pathlib import Path
        
        report_gen = ReportGenerator()
        
        # Prepare report data
        report_data = {
            "models": {
                "current_model": request.model_results
            }
        }
        
        if request.baseline_results:
            report_data["baselines"] = request.baseline_results
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        if request.format == "markdown":
            report_path = reports_dir / f"evaluation_report_{timestamp}.md"
            # Write markdown manually (ReportGenerator needs DataFrame which we don't have)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Evaluation Report\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write("## Model Results\n\n")
                for model_name, metrics in report_data.get("models", {}).items():
                    f.write(f"### {model_name}\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for metric, value in sorted(metrics.items()):
                        f.write(f"| {metric} | {value:.4f} |\n")
                    f.write("\n")
                if "baselines" in report_data:
                    f.write("## Baseline Results\n\n")
                    for baseline_name, metrics in report_data["baselines"].items():
                        f.write(f"### {baseline_name}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for metric, value in sorted(metrics.items()):
                            f.write(f"| {metric} | {value:.4f} |\n")
                        f.write("\n")
        elif request.format == "json":
            report_path = reports_dir / f"evaluation_report_{timestamp}.json"
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
        elif request.format == "csv":
            report_path = reports_dir / f"evaluation_report_{timestamp}.csv"
            try:
                import pandas as pd
                # Convert to DataFrame
                rows = []
                for model_name, metrics in report_data.get("models", {}).items():
                    for metric, value in metrics.items():
                        rows.append({"model": model_name, "metric": metric, "value": value})
                for baseline_name, metrics in report_data.get("baselines", {}).items():
                    for metric, value in metrics.items():
                        rows.append({"model": baseline_name, "metric": metric, "value": value})
                df = pd.DataFrame(rows)
                df.to_csv(report_path, index=False)
            except ImportError:
                # Fallback: write CSV manually
                logger.warning("pandas not available, writing CSV manually")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("model,metric,value\n")
                    for model_name, metrics in report_data.get("models", {}).items():
                        for metric, value in metrics.items():
                            f.write(f"{model_name},{metric},{value}\n")
                    for baseline_name, metrics in report_data.get("baselines", {}).items():
                        for metric, value in metrics.items():
                            f.write(f"{baseline_name},{metric},{value}\n")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown format: {request.format}"
            )
        
        file_size = report_path.stat().st_size
        download_url = f"/reports/{report_path.name}"
        
        logger.info(
            f"Report generated: {report_path.name}, "
            f"format={request.format}, size={file_size} bytes"
        )
        
        return GenerateReportResponse(
            report_path=str(report_path),
            format=request.format,
            file_size_bytes=file_size,
            download_url=download_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

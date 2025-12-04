# Serving Layer Documentation

## Tổng Quan

Serving Layer là thành phần phục vụ recommendations trong production cho hệ thống Vietnamese Cosmetics Recommender. Layer này xử lý ~300K users, 2.2K products với chiến lược **user segmentation** để tối ưu performance.

### Thống Kê Traffic

| User Type | Số lượng | Tỷ lệ | Phương pháp |
|-----------|----------|-------|-------------|
| Trainable (≥2 interactions) | ~26,000 | 8.6% | CF (ALS/BPR) + Reranking |
| Cold-start (<2 interactions) | ~274,000 | 91.4% | Content-based + Popularity |

---

## Quick Start

### 1. Khởi động Service

```powershell
# Development mode (auto-reload)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload

# Production mode (4 workers)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-compose up -d
```

### 2. Test Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# Single user recommendation
curl -X POST http://localhost:8000/recommend `
  -H "Content-Type: application/json" `
  -d '{"user_id": 12345, "topk": 10}'

# Batch recommendation
curl -X POST http://localhost:8000/batch_recommend `
  -H "Content-Type: application/json" `
  -d '{"user_ids": [123, 456, 789], "topk": 10}'

# Search products
curl -X POST http://localhost:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query": "kem dưỡng ẩm cho da khô", "topk": 10}'
```

---

## Kiến Trúc

```
service/
├── recommender/
│   ├── __init__.py          # Package exports
│   ├── loader.py            # CFModelLoader (singleton)
│   ├── recommender.py       # CFRecommender (main engine)
│   ├── fallback.py          # FallbackRecommender (cold-start)
│   ├── phobert_loader.py    # PhoBERTEmbeddingLoader
│   ├── rerank.py            # HybridReranker
│   ├── filters.py           # Attribute filtering & boosting
│   └── cache.py             # CacheManager
├── search/                  # Smart Search (Task 09)
├── api.py                   # FastAPI REST API
├── scheduler_api.py         # Scheduler endpoints
├── data_ingestion_api.py    # Data ingestion endpoints
└── config/
    ├── serving_config.yaml
    └── rerank_config.yaml
```

### Serving Flow

```
User Request (user_id, topk)
         ↓
    Load user_metadata
         ↓
    Check is_trainable_user?
         ↓
    ┌────┴────┐
    │         │
   YES        NO
    ↓          ↓
CF Path    Fallback Path
    ↓          ↓
U[u] @ V.T  PhoBERT similarity
    ↓          ↓
Filter seen  + Popularity
    ↓          ↓
Hybrid Rerank  Content Rerank
    ↓          ↓
    └────┬────┘
         ↓
   Return Top-K
```

---

## Components

### 1. CFModelLoader

Singleton class load models, mappings, và metadata.

```python
from service.recommender import get_loader, reset_loader

# Lấy singleton instance
loader = get_loader()

# Load model cụ thể
model = loader.load_model(model_id='als_v2_20250116')

# Load model mới nhất từ registry
model = loader.load_model()  # model_id=None

# Graceful empty mode (không throw error nếu không có model)
model = loader.load_model(raise_if_missing=False)

# Lấy thông tin model
info = loader.get_model_info()
# Returns: {
#     'model_id': 'als_v2_20250116',
#     'model_type': 'als',
#     'num_users': 26000,
#     'num_items': 2200,
#     'trainable_users': 26000,
#     'empty_mode': False
# }

# Hot-reload khi registry update
loader.check_and_reload()

# Reset singleton (testing)
reset_loader()
```

### 2. CFRecommender

Main recommendation engine.

```python
from service.recommender import CFRecommender

# Khởi tạo với auto-load
recommender = CFRecommender(auto_load=True)

# Single user recommendation
result = recommender.recommend(
    user_id=12345,
    topk=10,
    exclude_seen=True,
    filter_params={'brand': 'Innisfree'}
)

# Result structure
# RecommendationResult(
#     user_id=12345,
#     recommendations=[
#         {'rank': 1, 'product_id': 100, 'score': 0.95, 'product_name': '...', ...},
#         {'rank': 2, 'product_id': 200, 'score': 0.92, ...},
#         ...
#     ],
#     count=10,
#     is_fallback=False,
#     fallback_method=None,
#     model_id='als_v2_20250116'
# )

# Batch recommendation
results = recommender.batch_recommend(
    user_ids=[123, 456, 789],
    topk=10,
    exclude_seen=True
)
# Returns: Dict[int, RecommendationResult]

# Similar items (CF embeddings)
similar = recommender.similar_items(
    product_id=100,
    topk=10,
    use_cf=True  # False = PhoBERT
)

# Enable/disable reranking
recommender.set_reranking(enabled=True)

# Update reranking weights
recommender.update_rerank_weights({
    'cf': 0.4,
    'content': 0.3,
    'popularity': 0.2,
    'quality': 0.1
})

# Hot-reload model
reloaded = recommender.reload_model()
```

### 3. FallbackRecommender

Cold-start recommendations cho users có <2 interactions.

```python
from service.recommender import FallbackRecommender

fallback = FallbackRecommender()

# Item similarity (content-based)
recs = fallback.item_similarity_fallback(
    user_history=[100, 200],  # Products user đã mua
    topk=10,
    exclude_history=True
)

# Popularity fallback
recs = fallback.popularity_fallback(topk=10)

# Hybrid fallback (recommended)
recs = fallback.hybrid_fallback(
    user_history=[100, 200],
    topk=10,
    content_weight=0.6,
    popularity_weight=0.3,
    quality_weight=0.1
)
```

### 4. PhoBERTEmbeddingLoader

Load và cache PhoBERT embeddings cho content-based recommendations.

```python
from service.recommender import get_phobert_loader, reset_phobert_loader

phobert = get_phobert_loader()

# Load embeddings với fallback paths
phobert.load_embeddings(
    path='data/processed/content_based_embeddings/product_embeddings.pt',
    fallback_paths=[
        'data/published_data/content_based_embeddings/phobert_description_feature.pt'
    ]
)

# Check loaded
if phobert.is_loaded():
    print(f"Loaded {phobert.num_products} products, dim={phobert.embedding_dim}")

# Get embedding cho 1 product
embedding = phobert.get_embedding(product_id=100)

# Compute user profile từ history
profile = phobert.compute_user_profile(
    user_history=[100, 200, 300],
    strategy='weighted_mean'  # 'mean', 'weighted_mean', 'recency'
)

# Compute similarity
sim = phobert.compute_similarity(
    query_embedding=profile,
    top_k=10
)

# Get pre-computed similar items
similar = phobert.get_precomputed_similar(
    product_id=100,
    topk=10
)
```

### 5. HybridReranker

Two-stage reranking với multiple signals.

```python
from service.recommender import HybridReranker, get_reranker, RerankerConfig

# Lấy singleton
reranker = get_reranker()

# Hoặc custom config
config = RerankerConfig(
    weights_trainable={'cf': 0.3, 'content': 0.4, 'popularity': 0.2, 'quality': 0.1},
    weights_cold_start={'content': 0.6, 'popularity': 0.3, 'quality': 0.1},
    diversity_enabled=True,
    diversity_penalty=0.1,
    diversity_threshold=0.85,
    candidate_multiplier=5,
    user_profile_strategy='weighted_mean'
)
reranker = HybridReranker(config)

# Rerank recommendations
reranked = reranker.rerank(
    recommendations=candidates,
    user_id=12345,
    is_trainable=True,
    user_history=[100, 200, 300]
)

# Convenience functions
from service.recommender import rerank_with_signals, rerank_cold_start

# Quick rerank với custom weights
reranked = rerank_with_signals(
    recommendations=recs,
    user_id=12345,
    weights={'cf': 0.4, 'content': 0.4, 'popularity': 0.2}
)

# Cold-start specific rerank
reranked = rerank_cold_start(
    recommendations=recs,
    user_history=[100, 200]
)

# Update config
reranker.update_config(weights_trainable={'cf': 0.5, 'content': 0.3, ...})

# Clear cache
reranker.clear_cache()
```

### 6. CacheManager

LRU caching cho cold-start optimization.

```python
from service.recommender import get_cache_manager, reset_cache_manager, async_warmup
from service.recommender.cache import CacheConfig, LRUCache

# Lấy singleton
cache = get_cache_manager()

# Custom config
config = CacheConfig(
    user_profile_max_size=50000,
    user_profile_ttl=3600,
    item_similarity_max_size=10000,
    item_similarity_ttl=7200,
    fallback_max_size=5000,
    fallback_ttl=1800,
    warmup_num_popular_items=200,
    precompute_popular_similarities=True
)

# Warmup caches (sync)
stats = cache.warmup(force=False, include_similarities=True)
# Returns: {
#     'status': 'warmup_complete',
#     'popular_items': 200,
#     'popular_similarities': 200,
#     'warmup_duration_ms': 1500.0
# }

# Async warmup (for FastAPI lifespan)
stats = await async_warmup(cache)

# Get cached data
popular = cache.get_popular_items()  # List[int]
popular_enriched = cache.get_popular_items_enriched()  # List[Dict]
profile = cache.get_user_profile(user_id)  # Optional[np.ndarray]

# Set cache
cache.set_user_profile(user_id, profile)

# Compute and cache
profile = cache.compute_and_cache_user_profile(
    user_id=12345,
    user_history=[100, 200],
    phobert_loader=phobert
)

# Invalidation
cache.invalidate_user(user_id)
cache.invalidate_item(product_id)
cache.on_model_update()  # Clear item caches, re-warm
cache.clear_all()

# Stats
stats = cache.get_stats()
# Returns: {
#     'warmed_up': True,
#     'caches': {
#         'user_profile': {'size': 100, 'hit_rate': 0.85, ...},
#         'item_similarity': {'size': 200, 'hit_rate': 0.90, ...},
#         'fallback': {'size': 50, 'hit_rate': 0.75, ...}
#     },
#     'precomputed': {
#         'popular_items': 200,
#         'popular_items_enriched': 200,
#         'popular_similarities': 200
#     }
# }
```

### 7. Filters

Attribute filtering và boosting.

```python
from service.recommender.filters import (
    apply_filters,
    filter_by_brand,
    filter_by_skin_type,
    filter_by_price_range,
    get_valid_item_indices,
    exclude_products,
    boost_by_attributes,
    boost_by_user_preferences,
    infer_user_preferences,
    filter_and_boost
)

# Apply multiple filters
filtered = apply_filters(
    recommendations=recs,
    filter_params={
        'brand': 'Innisfree',
        'skin_type': ['oily', 'acne'],
        'price_min': 100000,
        'price_max': 500000,
        'min_rating': 4.0
    }
)

# Individual filters
filtered = filter_by_brand(recs, brand_filter='Innisfree', metadata=item_metadata)
filtered = filter_by_skin_type(recs, skin_types=['oily'], metadata=item_metadata)
filtered = filter_by_price_range(recs, price_min=100000, price_max=500000, metadata=item_metadata)

# Pre-filtering (more efficient for scoring)
valid_indices = get_valid_item_indices(
    filter_params={'brand': 'Innisfree'},
    metadata=item_metadata,
    idx_to_item=mappings['idx_to_item']
)

# Exclude specific products
filtered = exclude_products(recs, exclude_ids={100, 200, 300})

# Boosting
boosted = boost_by_attributes(
    recommendations=recs,
    boost_config={
        'brand': {'Innisfree': 1.15, 'CeraVe': 1.12},
        'skin_type': {'oily': 1.1}
    },
    metadata=item_metadata
)

# Infer user preferences từ history
prefs = infer_user_preferences(
    user_history=[100, 200, 300],
    metadata=item_metadata
)
# Returns: {
#     'preferred_brands': ['Innisfree', 'Cetaphil'],
#     'skin_types': ['oily', 'acne'],
#     'price_range': {'min': 150000, 'max': 500000}
# }

# Boost by user preferences
boosted = boost_by_user_preferences(recs, user_preferences=prefs, metadata=item_metadata)

# Combined filter + boost
result = filter_and_boost(
    recommendations=recs,
    filter_params={'brand': 'Innisfree'},
    boost_config={'brand': {'Innisfree': 1.15}},
    metadata=item_metadata
)
```

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/health` | Health check + model status | - |
| POST | `/recommend` | Single user recommendation | 60/min |
| POST | `/batch_recommend` | Batch recommendation (max 1000) | 20/min |
| POST | `/similar_items` | Similar items | - |
| POST | `/reload_model` | Hot-reload model | - |
| GET | `/model_info` | Model details | - |
| GET | `/stats` | Service statistics | - |

### Cache Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/cache_stats` | Cache statistics |
| POST | `/cache_warmup` | Trigger warmup |
| POST | `/cache_clear` | Clear all caches |

### Search Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/search` | Semantic search | 100/min |
| POST | `/search/similar` | Similar products | - |
| POST | `/search/profile` | Profile-based search | - |
| GET | `/search/filters` | Available filters | - |
| GET | `/search/stats` | Search statistics | - |

### Evaluation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/evaluate/model` | Evaluate model |
| POST | `/evaluate/compare` | Compare with baselines |
| POST | `/evaluate/statistical_test` | Statistical tests |
| POST | `/evaluate/metrics` | Compute metrics |
| POST | `/evaluate/hybrid` | Hybrid metrics |
| POST | `/evaluate/report` | Generate report |

### API Examples

#### Recommendation Request

```json
POST /recommend
{
    "user_id": 12345,
    "topk": 10,
    "exclude_seen": true,
    "filter_params": {
        "brand": "Innisfree",
        "skin_type": ["oily", "acne"],
        "price_max": 500000
    },
    "rerank": true,
    "rerank_weights": {
        "cf": 0.4,
        "content": 0.3,
        "popularity": 0.2,
        "quality": 0.1
    }
}
```

#### Recommendation Response

```json
{
    "user_id": 12345,
    "recommendations": [
        {
            "rank": 1,
            "product_id": 100,
            "score": 0.95,
            "product_name": "Innisfree Green Tea Seed Serum",
            "brand": "Innisfree",
            "category": "Serum",
            "price": 350000,
            "avg_rating": 4.8,
            "num_sold": 15000
        },
        ...
    ],
    "count": 10,
    "is_fallback": false,
    "fallback_method": null,
    "latency_ms": 45.2,
    "model_id": "als_v2_20250116"
}
```

#### Search Request

```json
POST /search
{
    "query": "kem dưỡng ẩm cho da khô nhạy cảm",
    "topk": 10,
    "filters": {
        "brand": "CeraVe",
        "min_price": 100000,
        "max_price": 500000
    },
    "rerank": true
}
```

---

## Configuration

### serving_config.yaml

```yaml
model:
  registry_path: "artifacts/cf/registry.json"
  default_model_type: "bert_als"
  auto_reload: true
  reload_check_interval_seconds: 300

serving:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout_seconds: 30

recommendation:
  default_topk: 10
  max_topk: 100
  exclude_seen: true

user_segmentation:
  trainable_threshold: 2
  positive_rating_threshold: 4.0

fallback:
  default_strategy: "hybrid"
  hybrid_weights:
    content: 0.6
    popularity: 0.3
    quality: 0.1

caching:
  user_history:
    enabled: true
    preload: true
  popular_items:
    top_k: 50
    refresh_interval_seconds: 3600
```

### rerank_config.yaml

```yaml
reranking:
  enabled: true
  candidate_multiplier: 5
  
  weights_trainable:
    cf: 0.30
    content: 0.40
    popularity: 0.20
    quality: 0.10
  
  weights_cold_start:
    content: 0.60
    popularity: 0.30
    quality: 0.10
  
  diversity:
    enabled: true
    penalty: 0.10
    threshold: 0.85
  
  user_profile_strategy: weighted_mean
  
  normalization:
    cf:
      min: 0.0
      max: 1.5
    content:
      min: -1.0
      max: 1.0
    popularity:
      p01: 0.0
      p99: 6.0
    quality:
      min: 1.0
      max: 5.0
```

---

## Security

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/recommend")
@limiter.limit("60/minute")
async def recommend(request: Request, ...):
    ...

@app.post("/batch_recommend")
@limiter.limit("20/minute")
async def batch_recommend(request: Request, ...):
    ...
```

### Security Headers

Tự động thêm vào mọi response:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Request-ID: req_1234567890
```

### Request Size Limit

Maximum request body: 10MB

### CORS

```python
# Development
allow_origins = ["*"]

# Production
allow_origins = os.getenv("ALLOWED_ORIGINS", "...").split(",")
```

### Error Sanitization

```python
# Production: Generic error messages
"Service temporarily unavailable"
"Invalid request parameters"

# Development: Full error details
str(error)
```

---

## Performance

### Latency Targets

| Path | Target | Actual |
|------|--------|--------|
| CF recommendation | <100ms | ~45ms |
| With reranking | <200ms | ~120ms |
| Cold-start fallback | <200ms | ~150ms |
| Search | <300ms | ~200ms |

### Throughput

- Single worker: ~100 req/s
- 4 workers: ~350 req/s
- With caching: ~500 req/s

### Optimizations

1. **Cache Warmup**: Pre-compute popular items, similarities
2. **LRU Caching**: User profiles, item similarities
3. **Batch Inference**: Matrix multiplication for batch users
4. **Pre-filtering**: Filter before scoring for efficiency
5. **Singleton Loaders**: Share model state across requests

---

## Monitoring

### Logs

```python
# Request logging
logger.info(
    f"user_id={user_id}, topk={topk}, "
    f"count={count}, fallback={is_fallback}, "
    f"latency={latency:.1f}ms"
)
```

### Metrics

- Request count by endpoint
- Latency percentiles (p50, p95, p99)
- CF vs Fallback ratio
- Cache hit rates
- Error rate

### Health Check Response

```json
{
    "status": "healthy",  // or "degraded"
    "model_id": "als_v2_20250116",
    "model_type": "als",
    "num_users": 300000,
    "num_items": 2200,
    "trainable_users": 26000,
    "timestamp": "2025-11-30T10:00:00",
    "empty_mode": false
}
```

### Service Stats Response

```json
{
    "model_id": "als_v2_20250116",
    "total_users": 300000,
    "trainable_users": 26000,
    "cold_start_users": 274000,
    "trainable_percentage": 8.67,
    "num_items": 2200,
    "popular_items_cached": 200,
    "user_histories_cached": 26000,
    "cache": {
        "warmed_up": true,
        "caches": {...}
    }
}
```

---

## Docker Deployment

### docker-compose.yaml

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
    environment:
      - ENV=production
      - ALLOWED_ORIGINS=https://your-domain.com
    command: uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## Troubleshooting

### Common Issues

#### 1. "User not in mappings"
- **Cause**: Cold-start user
- **Solution**: Automatically triggers fallback, not an error

#### 2. "Empty mode" warning
- **Cause**: Model artifacts not found
- **Solution**: 
  1. Check `artifacts/cf/registry.json` exists
  2. Mount volumes correctly in Docker
  3. Call `POST /reload_model` after uploading artifacts

#### 3. Matrix shape mismatch
- **Cause**: Model/mappings version mismatch
- **Solution**: Re-train model with latest data

#### 4. Slow first request
- **Cause**: Cold cache, PhoBERT loading
- **Solution**: Wait for startup warmup to complete

#### 5. High latency
- **Cause**: Large batch, no caching
- **Solution**:
  1. Reduce batch size
  2. Enable cache warmup
  3. Increase workers

### Debug Commands

```powershell
# Check service health
curl http://localhost:8000/health

# Check model info
curl http://localhost:8000/model_info

# Check cache stats
curl http://localhost:8000/cache_stats

# Force cache warmup
curl -X POST http://localhost:8000/cache_warmup?force=true

# Reload model
curl -X POST http://localhost:8000/reload_model
```

---

## Best Practices

1. **Always warmup on startup**: Giảm latency cho cold-start path (~91% traffic)

2. **Use batch endpoints**: Hiệu quả hơn nhiều so với single requests

3. **Enable reranking**: Cải thiện diversity và relevance

4. **Monitor fallback rate**: Nếu >95%, check model quality

5. **Invalidate cache khi update model**: `cache.on_model_update()`

6. **Use pre-filtering**: `get_valid_item_indices()` trước khi scoring

7. **Handle empty mode gracefully**: Service vẫn hoạt động với fallback

8. **Log latency metrics**: Để detect performance issues sớm

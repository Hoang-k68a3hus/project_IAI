# Task 05: Serving Layer

## Mục Tiêu

Xây dựng service layer để serve recommendations trong production, bao gồm model loading, recommendation generation, **user segmentation routing**, cold-start handling, filtering logic, và optional API endpoints. Service phải đảm bảo latency thấp, reliability cao, và dễ dàng integration.

## 🔄 Updated Serving Strategy (November 2025)

### Context: High Sparsity Data (~1.23 interactions/user, Updated ≥2 Threshold)
- **Trainable users** (≥2 interactions): ~26,000 users (~8.6% of total) → Serve with CF (ALS/BPR) + reranking
- **Cold-start users** (1 interaction or new): ~274,000 users (~91.4%) → Serve with content-based + popularity
- **Routing decision**: Load `user_metadata.pkl` to check `is_trainable_user` flag
- **Key insight**: 90%+ traffic will use content-based; CF is for the minority with ≥2 interactions

### Serving Flow by User Type:

```
User Request
    ↓
Check user_metadata
    ↓
├─ Trainable User? (≥2 interactions + ≥1 positive)
│  ├─ CF Recommender (ALS/BPR)
│  ├─ Generate Top-K candidates
│  ├─ Hybrid Reranking (CF + Content + Popularity)
│  └─ Return personalized results (~8.6% of traffic)
│
└─ Cold-Start User? (1 interaction or new user)
   ├─ Skip CF (no reliable user embedding)
   ├─ Item-Item Similarity (PhoBERT)
   │  └─ Find similar products to user's purchase history
   ├─ Mix with Popularity (Top sellers)
   └─ Return content-based + popular results (~91.4% of traffic)
```

**Benefits**:
- Don't waste CF computation on users with insufficient data
- Content-based provides better recommendations for sparse users than weak CF embeddings
- Clear separation of concerns for monitoring and A/B testing
- **Traffic optimization**: ~91.4% traffic uses fast content-based path; only ~8.6% uses CF

## Architecture Overview

```
service/
├── recommender/
│   ├── __init__.py          # Package exports (with reset_* functions)
│   ├── loader.py            # CFModelLoader (singleton, graceful empty mode)
│   ├── recommender.py       # CFRecommender (main engine with swapped matrix handling)
│   ├── fallback.py          # FallbackRecommender (cold-start with LRU caching)
│   ├── phobert_loader.py    # PhoBERTEmbeddingLoader (singleton with fallback paths)
│   ├── rerank.py            # HybridReranker (hybrid reranking with global normalization)
│   ├── filters.py           # Attribute filtering & boosting
│   └── cache.py             # CacheManager (LRU caching & warm-up)
├── search/                  # Smart Search (Task 09)
│   ├── query_encoder.py
│   ├── search_index.py
│   ├── smart_search.py
│   └── test_search_features.py
├── api.py                   # FastAPI REST API (with rate limiting, security headers)
├── dashboard.py             # Monitoring dashboard
├── scheduler_api.py         # Scheduler API router
├── data_ingestion_api.py    # Data ingestion API router
└── config/
    ├── serving_config.yaml
    ├── rerank_config.yaml
    └── search_config.yaml
```

### Package Exports

```python
from service.recommender import (
    # Loaders
    CFModelLoader, get_loader, reset_loader,
    PhoBERTEmbeddingLoader, get_phobert_loader, reset_phobert_loader,
    
    # Core
    CFRecommender, RecommendationResult,
    FallbackRecommender,
    
    # Hybrid Reranking
    HybridReranker, get_reranker, RerankerConfig, RerankedResult,
    rerank_with_signals, rerank_cold_start, diversify_recommendations,
    
    # Filtering
    apply_filters, filter_by_brand, filter_by_skin_type, filter_by_price_range,
    boost_by_attributes, boost_by_user_preferences,
    infer_user_preferences, filter_and_boost,
    
    # Caching
    CacheManager, CacheConfig, LRUCache,
    get_cache_manager, reset_cache_manager, async_warmup
)
```

## Component 1: Model Loader

### Module: `service/recommender/loader.py`

#### Class: `CFModelLoader`

##### Purpose
Singleton class for loading CF models, mappings, metadata, and trainable user routing information. Handles hot-reload when registry updates. Supports **graceful empty mode** when model artifacts are not available.

##### Initialization

```python
from service.recommender.loader import CFModelLoader, get_loader, reset_loader

# Option 1: Direct instantiation (singleton pattern)
loader = CFModelLoader(
    registry_path='artifacts/cf/registry.json',
    data_dir='data/processed',
    published_dir='data/published_data',
    auto_load=False  # Set True to auto-load on init
)

# Option 2: Singleton getter (recommended)
loader = get_loader()  # Returns singleton instance

# Option 3: Reset singleton (for testing)
reset_loader()  # Resets singleton instance
```

##### Attributes

```python
class CFModelLoader:
    # Cached state
    current_model: Optional[Dict[str, Any]]  # Loaded model dict
    current_model_id: Optional[str]
    mappings: Optional[Dict[str, Any]]  # User/item ID mappings
    trainable_user_mapping: Optional[Dict[int, int]]  # u_idx -> u_idx_cf
    trainable_user_set: Optional[Set[int]]  # Set of trainable u_idx
    item_metadata: Optional[pd.DataFrame]  # Product metadata
    user_history_cache: Optional[Dict[int, Set[int]]]  # user_id -> {product_ids}
    top_k_popular_items: Optional[List[int]]  # Pre-computed popular items
    data_stats: Optional[Dict[str, Any]]  # Data statistics
```

##### Method 1: `load_model(model_id=None, raise_if_missing=True)`

```python
model = loader.load_model(model_id=None, raise_if_missing=True)

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_type': 'als',
#     'U': np.ndarray (num_trainable_users, factors),
#     'V': np.ndarray (num_items, factors),
#     'params': dict,
#     'metadata': dict,
#     'score_range': {'min': 0.0, 'max': 1.5, 'p01': ..., 'p99': ...},
#     'loaded_at': '2025-01-16T14:30:00'
# }
# Or None if raise_if_missing=False and model not found

# Features:
# - Auto-detects current_best if model_id=None
# - Handles U/V matrix swap detection (from Colab training)
# - Loads score_range for normalization
# - Caches model in memory
# - Graceful empty mode with raise_if_missing=False
```

##### Method 2: `load_mappings(data_version=None, raise_if_missing=True)`

```python
mappings = loader.load_mappings(data_version=None, raise_if_missing=True)

# Returns:
# {
#     'user_to_idx': {user_id: u_idx},
#     'idx_to_user': {u_idx: user_id},
#     'item_to_idx': {product_id: i_idx},
#     'idx_to_item': {i_idx: product_id},
#     'metadata': {
#         'num_users': 300000,
#         'num_items': 2244,
#         'num_trainable_users': 26000,
#         'data_hash': 'abc123...'
#     }
# }
# Or None if raise_if_missing=False and mappings not found

# Also automatically loads:
# - trainable_user_mapping (u_idx -> u_idx_cf)
# - top_k_popular_items
# - data_stats
```

##### Method 3: `load_item_metadata(raise_if_missing=True)`

```python
metadata = loader.load_item_metadata(raise_if_missing=True)

# Returns: pd.DataFrame with product info
# Tries enriched_products.parquet first, falls back to raw CSVs
# Columns: product_id, product_name, brand, price, avg_star, num_sold_time, ...
# Or None if raise_if_missing=False and metadata not found
```

##### Method 4: `load_user_histories()`

```python
histories = loader.load_user_histories()

# Returns: Dict[int, Set[int]] - {user_id: {product_ids}}
# IMPORTANT: Only loads TRAIN split to avoid data leakage
# Handles case where 'split' column doesn't exist (old format)
# Cached in memory for fast seen-item filtering
```

##### Method 5: `is_trainable_user(user_id)`

```python
is_trainable = loader.is_trainable_user(user_id=12345)

# Returns: bool
# True if user has ≥2 interactions AND ≥1 positive rating
# Used for routing: CF vs content-based
```

##### Method 6: `get_cf_user_index(user_id)`

```python
u_idx_cf = loader.get_cf_user_index(user_id=12345)

# Returns: int or None
# CF matrix row index (u_idx_cf) for trainable users
# None if user is not trainable
```

##### Method 7: `get_user_history(user_id)`

```python
history = loader.get_user_history(user_id=12345)

# Returns: Set[int] - Set of product_ids user has interacted with
# Uses cached user_history_cache (train split only)
```

##### Method 8: `reload_if_updated()`

```python
reloaded = loader.reload_if_updated()

# Returns: bool - True if model was reloaded
# Checks registry for new current_best and reloads if changed
```

##### Method 9: `get_model_info()`

```python
info = loader.get_model_info()

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_type': 'als',
#     'num_users': 26000,
#     'num_items': 2244,
#     'factors': 128,
#     'loaded_at': '2025-01-16T14:30:00',
#     'score_range': {...},
#     'empty_mode': False  # True if no model loaded
# }
```

##### Method 10: `get_popular_items(topk=50)`

```python
popular_indices = loader.get_popular_items(topk=50)

# Returns: List[int] - Top-K popular item indices
# Uses pre-computed top_k_popular_items from data processing
```

## Component 2: Core Recommender

### Module: `service/recommender/recommender.py`

#### Class: `CFRecommender`

##### Purpose
Main recommendation engine with user segmentation routing, CF scoring, hybrid reranking, and fallback handling. Supports **empty mode** when no CF model is loaded.

##### Initialization

```python
from service.recommender import CFRecommender

recommender = CFRecommender(
    loader=None,  # Uses get_loader() singleton if None
    phobert_loader=None,  # Lazy-loaded if None
    auto_load=True,  # Auto-load models and data on init
    enable_reranking=True,  # Enable hybrid reranking by default
    rerank_config_path=None  # Path to rerank config YAML
)
```

##### Data Class: `RecommendationResult`

```python
@dataclass
class RecommendationResult:
    user_id: int
    recommendations: List[Dict[str, Any]]  # Enriched recommendations
    count: int
    is_fallback: bool  # True if used fallback (cold-start)
    fallback_method: Optional[str]  # 'popularity', 'item_similarity', 'hybrid', 'no_model'
    latency_ms: float
    model_id: Optional[str]  # CF model ID (None for fallback)
```

##### Method 1: `recommend()`

```python
result = recommender.recommend(
    user_id=12345,
    topk=10,
    exclude_seen=True,
    filter_params={'brand': 'Innisfree'},  # Optional
    normalize_scores=False,  # Normalize CF scores to [0, 1]
    rerank=None  # Override default reranking (None = use default)
)

# Returns: RecommendationResult

# Workflow:
# 1. Check if running in empty mode (no model)
#    - If empty mode → return empty result with fallback_method='no_model'
# 2. Check if user is trainable (is_trainable_user)
# 3. If trainable:
#    - Get CF user index (get_cf_user_index)
#    - Handle swapped U/V matrices (from Colab training)
#    - Compute CF scores: U[u_idx_cf] @ V.T (or V[u_idx_cf] @ U.T if swapped)
#    - Exclude seen items
#    - Apply attribute filters
#    - Get top-K candidates (5x if reranking)
#    - Apply hybrid reranking if enabled
# 4. If cold-start:
#    - Use FallbackRecommender
#    - Strategy: 'hybrid' (content + popularity)
#    - Apply reranking to fallback results
```

##### Method 2: `batch_recommend()`

```python
results = recommender.batch_recommend(
    user_ids=[12345, 67890, 11111],
    topk=10,
    exclude_seen=True
)

# Returns: Dict[int, RecommendationResult]
# Uses vectorized CF scoring for efficiency
# Separates trainable vs cold-start users
# Batch matrix multiplication: U[u_indices] @ V.T
```

##### Method 3: `similar_items()`

```python
similar = recommender.similar_items(
    product_id=123,
    topk=10,
    use_cf=True  # If False, uses PhoBERT embeddings
)

# Returns: List[Dict[str, Any]] - Similar items with metadata
# If use_cf=True: Uses V @ V.T for CF-based similarity
# If use_cf=False: Uses PhoBERT embeddings for content similarity
```

##### Method 4: `reload_model()`

```python
reloaded = recommender.reload_model()

# Returns: bool - True if model was reloaded
# Checks registry for updates and reloads if current_best changed
# Updates U, V, model_id, score_range
```

##### Method 5: `get_model_info()`

```python
info = recommender.get_model_info()

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_type': 'als',
#     'num_users': 26000,
#     'num_items': 2244,
#     'factors': 128,
#     'trainable_users': 26000,
#     'reranking_enabled': True,
#     ...
# }
```

##### Method 6: `set_reranking(enabled: bool)`

```python
recommender.set_reranking(enabled=True)

# Enable or disable hybrid reranking
```

##### Method 7: `update_rerank_weights()`

```python
recommender.update_rerank_weights(
    weights_trainable={'cf': 0.35, 'content': 0.35, 'popularity': 0.20, 'quality': 0.10},
    weights_cold_start={'content': 0.65, 'popularity': 0.25, 'quality': 0.10}
)

# Dynamically update reranking weights
```

##### Properties

```python
# Lazy-loaded fallback recommender
fallback = recommender.fallback  # Returns FallbackRecommender instance

# Lazy-loaded hybrid reranker
reranker = recommender.reranker  # Returns HybridReranker instance
```

##### Method 2: `_get_user_history(user_id)`
```python
def _get_user_history(self, user_id):
    """
    Retrieve items user đã interact.
    
    Args:
        user_id: Original user ID
    
    Returns:
        set: Set of product_ids
    """
    cached = self.user_history_cache.get(user_id)
    return set(cached) if cached else set()
```

##### Method 3: `_apply_filters(filter_params)`
```python
def _apply_filters(self, filter_params):
    """
    Apply attribute filters.
    
    Args:
        filter_params: Dict như {'brand': 'Innisfree', 'skin_type': 'oily'}
    
    Returns:
        np.array: Indices of valid items
    """
    mask = pd.Series([True] * len(self.item_metadata))
    
    for key, value in filter_params.items():
        if key in self.item_metadata.columns:
            mask &= self.item_metadata[key] == value
    
    valid_product_ids = self.item_metadata[mask]['product_id'].values
    valid_indices = [self.mappings['item_to_idx'][str(pid)] for pid in valid_product_ids]
    
    return np.array(valid_indices)
```

##### Method 4: `batch_recommend(user_ids, topk=10, exclude_seen=True)`
```python
def batch_recommend(self, user_ids, topk=10, exclude_seen=True):
    """
    Batch recommendation cho nhiều users (efficient).
    
    Args:
        user_ids: List of user IDs
        topk: Number of recommendations per user
        exclude_seen: Filter seen items
    
    Returns:
        dict: {user_id: [recommendations]}
    """
    results = {}
    
    # Separate cold-start users
    known_users = [uid for uid in user_ids if uid in self.mappings['user_to_idx']]
    cold_users = [uid for uid in user_ids if uid not in self.mappings['user_to_idx']]
    
    # Batch scoring cho known users
    if known_users:
        u_indices = [self.mappings['user_to_idx'][uid] for uid in known_users]
        scores_batch = self.U[u_indices] @ self.V.T  # (len(known_users), num_items)
        
        for i, uid in enumerate(known_users):
            scores = scores_batch[i]
            
            if exclude_seen:
                seen = self._get_user_history(uid)
                seen_indices = [self.mappings['item_to_idx'][str(pid)] for pid in seen]
                scores[seen_indices] = -np.inf
            
            top_k_indices = np.argsort(scores)[::-1][:topk]
            product_ids = [self.mappings['idx_to_item'][str(i)] for i in top_k_indices]
            
            # Enrich (simplified)
            results[uid] = [{'product_id': pid, 'score': float(scores[i])} for pid, i in zip(product_ids, top_k_indices)]
    
    # Fallback cho cold-start
    for uid in cold_users:
        results[uid] = self._fallback_recommendations(topk)
    
    return results
```

## Component 3: Cold-Start Fallback

### Module: `service/recommender/fallback.py`

#### Class: `FallbackRecommender`

##### Purpose
Handles cold-start users (~91.4% of traffic) with content-based and popularity-based recommendations. Optimized with LRU caching for low latency.

##### Initialization

```python
from service.recommender.fallback import FallbackRecommender

fallback = FallbackRecommender(
    cf_loader=loader,  # CFModelLoader instance
    phobert_loader=None,  # Lazy-loaded if None
    default_content_weight=0.7,
    default_popularity_weight=0.3,
    enable_cache=True  # Enable LRU caching
)
```

##### Strategy Overview
For users with <2 interactions (cold-start), skip CF and use:
1. **Item-Item Similarity** (PhoBERT embeddings)
2. **Popularity** (Top-selling products)
3. **Hybrid** (Weighted combination of both)

##### Method 1: `recommend()`

```python
recs = fallback.recommend(
    user_id=12345,  # Optional, to fetch history
    user_history=[100, 200, 300],  # Overrides user_id lookup
    topk=10,
    strategy='hybrid',  # 'popularity', 'item_similarity', or 'hybrid'
    exclude_ids={400, 500},  # Product IDs to exclude
    filter_params={'brand': 'Innisfree'}  # Optional filters
)

# Returns: List[Dict[str, Any]] - Recommendations with metadata
```

##### Method 2: `fallback_popularity()`

```python
recs = fallback.fallback_popularity(
    topk=10,
    exclude_ids=None,
    filter_params=None
)

# Returns: List of popular products
# Uses pre-computed top_k_popular_items from loader
# Optimized: Uses cached enriched popular items if available
```

##### Method 3: `fallback_item_similarity()`

```python
recs = fallback.fallback_item_similarity(
    user_history=[100, 200, 300],
    topk=10,
    exclude_ids=None,
    filter_params=None
)

# Returns: List of content-similar products
# Uses PhoBERTEmbeddingLoader.compute_user_profile()
# Caches user profiles for performance
# Falls back to popularity if PhoBERT unavailable
```

##### Method 4: `hybrid_fallback()`

```python
recs = fallback.hybrid_fallback(
    user_history=[100, 200, 300],
    topk=10,
    content_weight=0.7,  # Weight for content similarity
    popularity_weight=0.3,  # Weight for popularity
    exclude_ids=None,
    filter_params=None
)

# Returns: List of hybrid recommendations
# Combines content similarity and popularity scores
# Weighted combination: final_score = content_weight * content + popularity_weight * pop
```

## Component 4: PhoBERT Embedding Loader

### Module: `service/recommender/phobert_loader.py`

#### Class: `PhoBERTEmbeddingLoader`

##### Purpose
Singleton class for loading and using PhoBERT product embeddings for content-based recommendations. Pre-normalizes embeddings for fast cosine similarity. Supports **fallback paths** for finding embeddings.

##### Initialization

```python
from service.recommender.phobert_loader import PhoBERTEmbeddingLoader, get_phobert_loader, reset_phobert_loader

# Option 1: Direct instantiation (singleton)
phobert = PhoBERTEmbeddingLoader(
    embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt',
    fallback_paths=[  # Fallback locations if primary not found
        'data/published_data/content_based_embeddings/product_embeddings.pt',
        'data/published_data/content_based_embeddings/phobert_description_feature.pt'
    ],
    auto_load=True
)

# Option 2: Singleton getter (recommended)
phobert = get_phobert_loader()

# Option 3: Reset singleton (for testing)
reset_phobert_loader()
```

##### Method 1: `get_embedding(product_id)`

```python
emb = phobert.get_embedding(product_id=123)

# Returns: np.ndarray (768,) or (1024,) - Raw embedding
# None if product not found
```

##### Method 2: `get_embedding_normalized(product_id)`

```python
emb_norm = phobert.get_embedding_normalized(product_id=123)

# Returns: L2-normalized embedding for fast cosine similarity
```

##### Method 3: `is_loaded()`

```python
is_loaded = phobert.is_loaded()

# Returns: bool - True if embeddings are loaded
```

##### Method 4: `compute_user_profile()`

```python
profile = phobert.compute_user_profile(
    user_history_items=[100, 200, 300],
    weights=[1.0, 1.5, 1.0],  # Optional weights (e.g., ratings)
    strategy='weighted_mean'  # 'mean', 'weighted_mean', or 'max'
)

# Returns: np.ndarray (768,) - User profile embedding
# Aggregates history items into single embedding
```

##### Method 5: `compute_similarity()`

```python
similarities = phobert.compute_similarity(
    query_embedding=query_emb,
    candidate_indices=[0, 1, 2]  # Optional, all items if None
)

# Returns: np.ndarray of similarity scores
```

##### Method 6: `find_similar_items()`

```python
similar = phobert.find_similar_items(
    product_id=123,
    topk=10,
    exclude_self=True,
    exclude_ids={400, 500}
)

# Returns: List[Tuple[int, float]] - [(product_id, similarity_score), ...]
# Uses pre-normalized embeddings for fast cosine similarity
```

##### Method 7: `find_similar_to_profile()`

```python
similar = phobert.find_similar_to_profile(
    user_profile=profile_embedding,
    topk=10,
    exclude_ids={100, 200, 300}
)

# Returns: List[Tuple[int, float]] - Similar items to user profile
```

##### Method 8: `precompute_item_similarity()`

```python
phobert.precompute_item_similarity(max_items=3000)

# Precomputes V @ V.T similarity matrix for small catalogs
# Speeds up repeated similar item queries
```

##### Method 9: `get_precomputed_similar()`

```python
similar = phobert.get_precomputed_similar(
    product_id=123,
    topk=10,
    exclude_ids={400, 500}
)

# Returns: List[Tuple[int, float]]
# Uses precomputed matrix if available, falls back to compute
```

##### Properties

```python
# Embedding dimension
dim = phobert.embedding_dim  # Returns 768 or 1024

# Number of products
num_products = phobert.num_products  # Returns count of products with embeddings
```

## Component 5: Hybrid Reranking

### Module: `service/recommender/rerank.py`

#### Class: `HybridReranker`

##### Purpose
Hybrid reranker combining CF, content (PhoBERT), popularity, and quality signals. Uses global normalization for consistent scoring across requests. Supports BERT-based diversity penalty.

##### Initialization

```python
from service.recommender.rerank import HybridReranker, get_reranker, RerankerConfig

# Option 1: Direct instantiation
reranker = HybridReranker(
    phobert_loader=phobert_loader,
    item_metadata=item_metadata,
    config=None,  # Uses default RerankerConfig if None
    config_path='service/config/rerank_config.yaml'  # Optional YAML config
)

# Option 2: Singleton getter (recommended)
reranker = get_reranker(
    phobert_loader=phobert_loader,
    item_metadata=item_metadata,
    config_path='service/config/rerank_config.yaml'
)
```

##### Data Class: `RerankerConfig`

```python
@dataclass
class RerankerConfig:
    # Weights for trainable users (≥2 interactions)
    weights_trainable: Dict[str, float] = {
        'cf': 0.30,         # SECONDARY - Collaborative signal
        'content': 0.40,    # PRIMARY - PhoBERT semantic similarity
        'popularity': 0.20, # TERTIARY - Trending items
        'quality': 0.10     # BONUS - High-rated products
    }
    
    # Weights for cold-start users (<2 interactions)
    weights_cold_start: Dict[str, float] = {
        'content': 0.60,    # DOMINANT - Only reliable signal
        'popularity': 0.30, # Social proof
        'quality': 0.10     # Bonus
    }
    
    # Diversity settings
    diversity_enabled: bool = True
    diversity_penalty: float = 0.1
    diversity_threshold: float = 0.85  # BERT similarity threshold
    
    # User profile strategy
    user_profile_strategy: str = 'weighted_mean'  # mean, weighted_mean, recency
    
    # Candidate multiplier
    candidate_multiplier: int = 5  # Generate N * topk candidates
    
    # Normalization ranges (global, not local)
    cf_score_min: float = 0.0
    cf_score_max: float = 1.5
    content_score_min: float = -1.0
    content_score_max: float = 1.0
    quality_min: float = 1.0
    quality_max: float = 5.0
    popularity_p01: float = 0.0  # From data_stats.json
    popularity_p99: float = 6.0  # From data_stats.json
```

##### Data Class: `RerankedResult`

```python
@dataclass
class RerankedResult:
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    diversity_score: float
    weights_used: Dict[str, float]
    num_candidates: int
    num_output: int
```

##### Method 1: `rerank()`

```python
result = reranker.rerank(
    cf_recommendations=cf_recs,  # List from CFRecommender
    user_id=12345,
    user_history=[100, 200, 300],
    topk=10,
    is_cold_start=False  # True for cold-start users
)

# Returns: RerankedResult
# Workflow:
# 1. Compute signals: CF, content, popularity, quality
# 2. Normalize signals using global ranges (not local min/max)
# 3. Combine with weights (trainable vs cold-start)
# 4. Apply diversity penalty if enabled
# 5. Re-sort and return top-K
```

##### Method 2: `rerank_cold_start()`

```python
result = reranker.rerank_cold_start(
    recommendations=fallback_recs,
    user_history=[100, 200],
    topk=10
)

# Returns: RerankedResult
# Uses weights_cold_start (no CF signal)
```

##### Method 3: `update_config()`

```python
reranker.update_config(
    weights_trainable={'cf': 0.35, 'content': 0.35, 'popularity': 0.20, 'quality': 0.10},
    weights_cold_start={'content': 0.65, 'popularity': 0.25, 'quality': 0.10},
    diversity_enabled=True,
    diversity_penalty=0.1,
    diversity_threshold=0.85
)

# Dynamically update reranking configuration
```

##### Method 4: `clear_cache()`

```python
reranker.clear_cache()

# Clear cached popularity and quality scores
```

##### Convenience Functions

```python
from service.recommender.rerank import (
    rerank_with_signals,       # Legacy function for single-pass reranking
    rerank_cold_start,         # Legacy function for cold-start reranking
    diversify_recommendations, # Diversity-only function (category-based)
    min_max_normalize,         # Helper: min-max normalization
    robust_normalize           # Helper: percentile-based normalization
)
```

## Component 6: Cache Manager

### Module: `service/recommender/cache.py`

#### Class: `LRUCache`

##### Purpose
Thread-safe LRU cache with TTL support.

```python
from service.recommender.cache import LRUCache

cache = LRUCache(
    max_size=10000,
    ttl_seconds=3600,  # 1 hour
    name="user_profile"
)

# Basic operations
cache.put(key, value)
value = cache.get(key)
cache.delete(key)
cache.clear()

# Statistics
stats = cache.stats()  # hit_rate, hits, misses, evictions
```

#### Class: `CacheConfig`

```python
@dataclass
class CacheConfig:
    # User profile cache (BERT aggregation)
    user_profile_max_size: int = 50000
    user_profile_ttl_seconds: float = 3600  # 1 hour
    
    # Item similarity cache (top-K similar items)
    item_similarity_max_size: int = 5000
    item_similarity_ttl_seconds: float = 86400  # 24 hours
    
    # Fallback recommendation cache
    fallback_max_size: int = 10000
    fallback_ttl_seconds: float = 1800  # 30 min
    
    # Popular items - refreshed less frequently
    popular_items_ttl_seconds: float = 3600  # 1 hour
    
    # Pre-warm settings
    warmup_num_popular_items: int = 200
    warmup_num_user_profiles: int = 1000
    
    # Cold-start optimization
    precompute_popular_similarities: bool = True
```

#### Class: `CacheManager`

##### Purpose
LRU caching and warm-up strategies for optimizing cold-start path latency (~91% of traffic).

##### Features
- LRU caches for user profiles, item similarities, fallback results
- Pre-computation of popular items and their similarities
- Warm-up strategies for cold-start recommendations
- Cache invalidation hooks for model updates

##### Initialization

```python
from service.recommender.cache import CacheManager, get_cache_manager, CacheConfig, reset_cache_manager

# Option 1: Direct instantiation
cache = CacheManager(
    config=CacheConfig(
        user_profile_max_size=50000,
        item_similarity_max_size=5000,
        fallback_max_size=10000,
        user_profile_ttl_seconds=3600,
        warmup_num_popular_items=200,
        precompute_popular_similarities=True
    )
)

# Option 2: Singleton getter (recommended)
cache = get_cache_manager()

# Option 3: Reset singleton (for testing)
reset_cache_manager()
```

##### Method: `warmup()`

```python
stats = cache.warmup(force=False, include_similarities=True)

# Pre-computes:
# - Popular items with enriched metadata
# - Popular item similarities (PhoBERT-based)
# - Common user profiles
# Returns warmup statistics

# Returns:
# {
#     'status': 'warmup_complete',
#     'skipped': False,
#     'popular_items': 200,
#     'popular_items_enriched': 200,
#     'popular_similarities': 200,
#     'warmup_duration_ms': 1500.0
# }
```

##### Method: `get_popular_items()`

```python
popular_items = cache.get_popular_items()

# Returns: List[int] - Popular item indices
```

##### Method: `get_popular_items_enriched()`

```python
popular = cache.get_popular_items_enriched()

# Returns: List[Dict] - Pre-computed popular items with metadata
# Used by FallbackRecommender for fast popularity fallback
```

##### Method: `get_user_profile(user_id)`

```python
profile = cache.get_user_profile(user_id)

# Returns: Cached user profile embedding or None
# Used by FallbackRecommender to avoid recomputing profiles
```

##### Method: `set_user_profile(user_id, profile)`

```python
cache.set_user_profile(user_id, profile)

# Cache user BERT profile
```

##### Method: `compute_and_cache_user_profile()`

```python
profile = cache.compute_and_cache_user_profile(
    user_id=12345,
    user_history=[100, 200, 300],
    phobert_loader=phobert_loader
)

# Compute user profile and cache it
# Returns: User profile embedding or None
```

##### Method: `invalidate_user(user_id)`

```python
cache.invalidate_user(user_id)

# Invalidate all caches for a user (profile, fallback)
```

##### Method: `invalidate_item(product_id)`

```python
cache.invalidate_item(product_id)

# Invalidate caches for an item
# Also clears popular items if item is popular
```

##### Method: `clear_all()`

```python
cache.clear_all()

# Clear all caches
```

##### Method: `on_model_update()`

```python
cache.on_model_update()

# Handle model update - clear relevant caches
# Clears item similarities, keeps user profiles
# Re-warms popular items
```

##### Method: `get_stats()`

```python
stats = cache.get_stats()

# Returns:
# {
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

##### Async Warmup

```python
from service.recommender.cache import async_warmup

# For FastAPI lifespan
stats = await async_warmup(cache)

# Runs warmup in thread pool to not block event loop
```

## Component 7: API Layer (FastAPI)

### Module: `service/api.py`

#### FastAPI Application

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from service.recommender import CFRecommender, get_loader
from service.recommender.cache import get_cache_manager, async_warmup

# Security Configuration
ENV = os.getenv("ENV", "development")  # "production" or "development"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "...").split(",")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="CF Recommendation Service",
    description="Collaborative Filtering recommendation API for Vietnamese cosmetics",
    version="1.0.0",
    lifespan=lifespan  # Startup/shutdown handlers
)

# Initialize rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include Scheduler API router (optional)
if SCHEDULER_API_AVAILABLE:
    app.include_router(scheduler_router, prefix="/scheduler", tags=["Scheduler"])

# Include Data Ingestion API router (optional)
if INGESTION_API_AVAILABLE:
    app.include_router(ingestion_router, prefix="/ingest", tags=["Data Ingestion"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ENV == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
```

#### Security Middleware

```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Request-ID"] = request_id  # For tracing
    
    return response

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent DoS attacks."""
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    # Returns 413 if exceeded
    ...
```

#### Lifespan Handler

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global recommender, cache_manager
    
    # 1. Initialize recommender (supports empty mode)
    recommender = CFRecommender(auto_load=True)
    model_info = recommender.get_model_info()
    
    if model_info.get('model_id'):
        logger.info(f"Loaded model: {model_info.get('model_id')}")
    else:
        logger.warning("⚠️ SERVICE RUNNING IN EMPTY MODE - No CF model loaded!")
    
    # 2. Initialize cache manager
    cache_manager = get_cache_manager()
    
    # 3. Initialize metrics DB
    get_service_metrics_db()
    
    # 4. Warm up caches for cold-start optimization (~91% traffic)
    warmup_stats = await async_warmup(cache_manager)
    
    # 5. Warm up search service (PhoBERT model)
    try:
        search_service = get_search_service()
        _ = search_service.search("kem dưỡng da", topk=1)  # Trigger model loading
    except Exception as e:
        logger.warning(f"Search service warmup failed (non-critical): {e}")
    
    # 6. Start background health aggregation task
    aggregation_task = asyncio.create_task(periodic_health_aggregation())
    
    yield
    
    # Cleanup
    aggregation_task.cancel()
```

#### Endpoints

##### 1. Health Check

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and model status.
    
    Returns:
        HealthResponse with model information including empty_mode flag
    """
    model_info = recommender.get_model_info()
    is_empty_mode = model_info.get('model_id') is None
    status = "degraded" if is_empty_mode else "healthy"
    
    return HealthResponse(
        status=status,
        model_id=model_info.get('model_id'),
        model_type=model_info.get('model_type'),
        num_users=model_info.get('num_users', 0),
        num_items=model_info.get('num_items', 0),
        trainable_users=model_info.get('trainable_users', 0),
        timestamp=datetime.now().isoformat(),
        empty_mode=is_empty_mode,  # New field
    )
```

##### 2. Single User Recommendation

```python
@app.post("/recommend", response_model=RecommendResponse)
@limiter.limit("60/minute")  # Rate limiting: 60 requests per minute per IP
async def recommend(request: Request, recommend_request: RecommendRequest):
    """
    Get recommendations for a single user.
    
    Request:
        {
            "user_id": 12345,
            "topk": 10,
            "exclude_seen": true,
            "filter_params": {"brand": "Innisfree"},
            "rerank": false,
            "rerank_weights": {...}  # Optional override
        }
    
    Returns:
        RecommendResponse with recommendations and metadata
    """
```

##### 3. Batch Recommendation

```python
@app.post("/batch_recommend", response_model=BatchResponse)
@limiter.limit("20/minute")  # Lower limit for batch operations
async def batch_recommend(request: Request, batch_request: BatchRequest):
    """
    Get recommendations for multiple users.
    Maximum batch size: 1000 users
    
    Returns:
        BatchResponse with results for all users
        Includes: cf_users, fallback_users counts
    """
```

##### 4. Similar Items

```python
@app.post("/similar_items", response_model=SimilarItemsResponse)
async def similar_items(request: SimilarItemsRequest):
    """
    Find similar items to a given product.
    
    Request:
        {
            "product_id": 123,
            "topk": 10,
            "use_cf": true  # If false, uses PhoBERT
        }
    """
```

##### 5. Model Reload

```python
@app.post("/reload_model", response_model=ReloadResponse)
async def reload_model():
    """
    Hot-reload model from registry.
    
    Returns:
        ReloadResponse with reload status
    """
```

##### 6. Model Info

```python
@app.get("/model_info")
async def model_info():
    """Get detailed model information."""
```

##### 7. Service Statistics

```python
@app.get("/stats")
async def service_stats():
    """
    Get service statistics.
    
    Returns:
        model_id, total_users, trainable_users, cold_start_users,
        trainable_percentage, num_items, popular_items_cached,
        user_histories_cached, cache stats
    """
```

##### 8. Cache Management

```python
@app.get("/cache_stats")
async def cache_stats():
    """Get detailed cache statistics."""

@app.post("/cache_warmup")
async def trigger_warmup(force: bool = False):
    """Trigger cache warmup."""

@app.post("/cache_clear")
async def clear_cache():
    """Clear all caches."""
```

##### 9. Smart Search Endpoints (Task 09)

```python
@app.post("/search", response_model=SearchResponse)
@limiter.limit("100/minute")  # Higher limit for search
async def search_products(request: SearchRequest):
    """Semantic search for products using Vietnamese query."""

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_products(request: SearchSimilarRequest):
    """Find products similar to a given product."""

@app.post("/search/profile", response_model=SearchResponse)
async def search_by_profile(request: SearchByProfileRequest):
    """Search based on user profile/history."""

@app.get("/search/filters")
async def get_search_filters():
    """Get available filter options for search."""

@app.get("/search/stats")
async def get_search_stats():
    """Get search service statistics."""
```

##### 10. Evaluation Endpoints

```python
@app.post("/evaluate/model", response_model=EvaluateModelResponse)
async def evaluate_model_endpoint(request: EvaluateModelRequest):
    """Evaluate model performance on test data."""

@app.post("/evaluate/compare", response_model=CompareModelsResponse)
async def compare_models_endpoint(request: CompareModelsRequest):
    """Compare model with baselines (popularity, random)."""

@app.post("/evaluate/statistical_test", response_model=StatisticalTestResponse)
async def statistical_test_endpoint(request: StatisticalTestRequest):
    """Perform statistical significance testing."""

@app.post("/evaluate/metrics", response_model=ComputeMetricsResponse)
async def compute_metrics_endpoint(request: ComputeMetricsRequest):
    """Compute specific metrics for given predictions."""

@app.post("/evaluate/hybrid", response_model=HybridMetricsResponse)
async def hybrid_metrics_endpoint(request: HybridMetricsRequest):
    """Compute hybrid metrics (diversity, novelty, etc.)."""

@app.post("/evaluate/report", response_model=GenerateReportResponse)
async def generate_report_endpoint(request: GenerateReportRequest):
    """Generate evaluation report in markdown/json/csv format."""
```

##### 11. Scheduler Endpoints (if enabled)

```python
# Available at /scheduler/* prefix
# Managed by service/scheduler_api.py router
```

##### 12. Data Ingestion Endpoints (if enabled)

```python
# Available at /ingest/* prefix
# Managed by service/data_ingestion_api.py router
```

#### Response Models

```python
class HealthResponse(APIBaseModel):
    status: str  # "healthy" or "degraded"
    model_id: Optional[str]
    model_type: Optional[str]
    num_users: int
    num_items: int
    trainable_users: int
    timestamp: str
    empty_mode: bool = False  # True if no model loaded

class RecommendResponse(APIBaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    count: int
    is_fallback: bool
    fallback_method: Optional[str]
    latency_ms: float
    model_id: Optional[str]
```

#### Numpy Type Handling

```python
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
```

#### Error Sanitization for Production

```python
def sanitize_error_message(error: Exception, endpoint: str = "") -> str:
    """
    Sanitize error messages for production.
    Don't expose internal error details in production.
    """
    if ENV == "production":
        if "not initialized" in str(error).lower():
            return "Service temporarily unavailable"
        elif "not found" in str(error).lower():
            return "Invalid request parameters"
        else:
            return "An error occurred processing your request"
    else:
        return str(error)  # Full error in development
```

## Configuration

### File: `service/config/serving_config.yaml`

```yaml
# Model Configuration
model:
  registry_path: "artifacts/cf/registry.json"
  default_model_type: "bert_als"  # bert_als, als, bpr
  auto_reload: true
  reload_check_interval_seconds: 300

# Serving Configuration  
serving:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout_seconds: 30
  max_batch_size: 100
  
# Default recommendation parameters
recommendation:
  default_topk: 10
  max_topk: 100
  exclude_seen: true
  
# User segmentation thresholds
user_segmentation:
  trainable_threshold: 2
  positive_rating_threshold: 4.0

# Fallback Configuration (for cold-start users)
fallback:
  default_strategy: "hybrid"  # popularity, item_similarity, hybrid
  
  popularity:
    top_k_items: 50
    boost_new_items: true
    new_item_days: 30
    
  item_similarity:
    use_phobert: true
    embedding_dim: 768
    precompute_similarity: true
    similarity_topk: 100
    
  hybrid_weights:
    content: 0.6
    popularity: 0.3
    quality: 0.1

# Reranking Configuration
reranking:
  enabled: true
  weights:
    cf: 0.3
    content: 0.4
    popularity: 0.2
    quality: 0.1
  diversify: true
  diversity_category_weight: 0.5
  diversity_window: 3

# Filtering Configuration
filtering:
  enabled: true
  available_filters:
    - brand
    - skin_type
    - price_range
    - category
  
# Caching Configuration
caching:
  user_history:
    enabled: true
    preload: true
  popular_items:
    enabled: true
    top_k: 50
    refresh_interval_seconds: 3600
  item_similarity:
    enabled: true
    precompute_at_startup: false
    max_cache_items: 1000

# API Configuration
api:
  cors_origins: ["*"]
  rate_limit: 1000
  enable_docs: true
  
# Logging Configuration
logging:
  level: "INFO"
  log_requests: true
  log_latency: true
  files:
    api_log: "logs/service/api.log"
    error_log: "logs/service/error.log"
    
# Monitoring Configuration
monitoring:
  enabled: true
  metrics_endpoint: "/metrics"
  health_endpoint: "/health"
  metrics:
    - request_count
    - request_latency
    - cf_vs_fallback_ratio
    - recommendation_count
    - error_rate

# Paths (relative to project root)
paths:
  data_dir: "data/processed"
  artifacts_dir: "artifacts/cf"
  logs_dir: "logs/service"
  user_mappings: "data/processed/user_item_mappings.json"
  trainable_user_mapping: "data/processed/trainable_user_mapping.json"
  item_metadata: "data/processed/item_metadata.pkl"
  phobert_embeddings: "data/processed/content_based_embeddings/product_embeddings.pt"
  phobert_embeddings_published: "data/published_data/content_based_embeddings/phobert_description_feature.pt"
```

### File: `service/config/rerank_config.yaml`

```yaml
reranking:
  enabled: true
  
  # Candidate multiplier: Generate N * topk candidates for reranking
  candidate_multiplier: 5
  
  # Weights for trainable users (≥2 interactions)
  # Evaluated: Recall@10=0.106, Diversity=0.198 (+6.9% vs pure CF)
  weights_trainable:
    cf: 0.30          # SECONDARY - Collaborative filtering score
    content: 0.40     # PRIMARY - PhoBERT semantic similarity
    popularity: 0.20  # TERTIARY - Trending/popular items
    quality: 0.10     # BONUS - High-rated products
  
  # Weights for cold-start users (<2 interactions)
  weights_cold_start:
    content: 0.60     # DOMINANT - Only reliable signal
    popularity: 0.30  # Social proof
    quality: 0.10     # Bonus for well-reviewed products
  
  # Diversity settings (BERT-based intra-list diversity)
  diversity:
    enabled: true
    penalty: 0.10     # Penalty factor for similar items (0.0-1.0)
    threshold: 0.85   # BERT similarity threshold
  
  # User profile computation strategy
  # Options: 'mean', 'weighted_mean', 'recency'
  user_profile_strategy: weighted_mean
  
  # Normalization ranges (pre-computed from training data)
  normalization:
    cf:
      min: 0.0
      max: 1.5        # ALS/BPR scores typically in [0, 1.5]
    content:
      min: -1.0       # Cosine similarity range
      max: 1.0
    popularity:
      p01: 0.0
      p99: 6.0        # Log-transformed popularity (from data_stats.json)
    quality:
      min: 1.0        # Rating range
      max: 5.0

# Attribute Boosting Configuration
attribute_boost:
  brand:
    Innisfree: 1.15
    Cetaphil: 1.10
    CeraVe: 1.12
    "La Roche-Posay": 1.12
    Bioderma: 1.10
    "The Ordinary": 1.08
  skin_type_standardized:
    oily: 1.10
    acne: 1.10
    sensitive: 1.08
    dry: 1.08
    combination: 1.05

# Category Diversity Settings
category_diversity:
  max_per_brand: 3
  max_per_type: 4
  enabled: true

# Fallback Settings
fallback:
  default_strategy: hybrid
  content_weight: 0.6
  popularity_weight: 0.3
  quality_weight: 0.1

# Performance Settings
performance:
  precompute_item_similarity: false
  user_profile_cache_size: 1000
  cache_ttl: 3600
```

## Logging & Monitoring

### Request Logging

The API automatically logs all requests to:
- **Console**: Structured logging with user_id, topk, latency, fallback status
- **Metrics DB**: SQLite database (`logs/service_metrics.db`) for aggregation

```python
# Automatic logging in API endpoints
logger.info(
    f"user_id={request.user_id}, topk={request.topk}, "
    f"count={result.count}, fallback={result.is_fallback}, "
    f"latency={latency:.1f}ms"
)

# Background logging to metrics DB
log_request_metrics(
    user_id=request.user_id,
    topk=request.topk,
    latency_ms=latency,
    num_recommendations=result.count,
    fallback=result.is_fallback,
    fallback_method=result.fallback_method,
    rerank_enabled=request.rerank,
    error=None
)
```

### Metrics to Track
- **Latency**: p50, p95, p99 response time (logged per request)
- **Throughput**: Requests per second (aggregated hourly)
- **Fallback rate**: % requests using fallback (~91.4% expected)
- **Error rate**: % failed requests
- **Cache hit rate**: From CacheManager stats
- **Reranking overhead**: Latency difference with/without reranking

### Background Tasks

```python
# Periodic health aggregation (every minute)
async def periodic_health_aggregation():
    # Aggregates metrics from requests table
    # Updates service_health table
    # Runs in background task
```

## Performance Optimizations

### 1. Singleton Pattern
- **CFModelLoader**: Single instance shared across requests
- **PhoBERTEmbeddingLoader**: Single instance, embeddings loaded once
- **HybridReranker**: Single instance, config cached
- **CacheManager**: Single instance, shared LRU caches

### 2. Pre-normalized Embeddings
- PhoBERT embeddings pre-normalized on load
- Fast cosine similarity: `embeddings_norm @ query_norm` (no per-request normalization)

### 3. Pre-computed Popular Items
- Top-K popular items loaded once from `top_k_popular_items.json`
- Cached enriched popular items in CacheManager
- Fast fallback for truly new users

### 4. User History Caching
- User histories loaded once (train split only)
- Cached in `CFModelLoader.user_history_cache`
- Fast seen-item filtering

### 5. Batch Inference
- `batch_recommend()` uses vectorized CF scoring
- Batch matrix multiplication: `U[u_indices] @ V.T`
- Amortizes overhead across multiple users

### 6. LRU Caching (CacheManager)
- User profiles cached (avoid recomputing from history)
- Popular items enriched and cached
- Similarity results cached for repeated queries

### 7. Lazy Loading
- PhoBERT loader initialized only when needed
- Fallback recommender created on first cold-start request
- Reranker initialized only when reranking enabled

### 8. Hot Reload
- Model reload without service restart
- Registry checked periodically
- Seamless transition to new best model

## Deployment

### Docker Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/ service/
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Run Locally
```bash
# Install dependencies
pip install fastapi uvicorn

# Start service
uvicorn service.api:app --reload --port 8000
```

### Test Endpoint
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "topk": 10,
    "exclude_seen": true
  }'
```

## Component 8: Attribute Filtering

### Module: `service/recommender/filters.py`

#### Function: `apply_filters()`

```python
from service.recommender.filters import apply_filters

filtered = apply_filters(
    recommendations=recs,
    filter_params={
        'brand': 'Innisfree',
        'skin_type': ['oily', 'acne'],  # List support
        'price_min': 100000,
        'price_max': 500000,
        'min_rating': 4.0
    }
)

# Returns: Filtered recommendations with updated ranks
```

#### Function: `filter_by_brand()`

```python
from service.recommender.filters import filter_by_brand

filtered = filter_by_brand(
    recommendations=recs,
    brand_filter='Innisfree',  # Can be str or list
    metadata=item_metadata
)
```

#### Function: `filter_by_skin_type()`

```python
from service.recommender.filters import filter_by_skin_type

filtered = filter_by_skin_type(
    recommendations=recs,
    skin_types=['oily', 'acne'],  # List of skin types
    metadata=item_metadata
)
```

#### Function: `filter_by_price_range()`

```python
from service.recommender.filters import filter_by_price_range

filtered = filter_by_price_range(
    recommendations=recs,
    price_min=100000,
    price_max=500000,
    metadata=item_metadata
)
```

#### Function: `get_valid_item_indices()`

```python
from service.recommender.filters import get_valid_item_indices

valid_indices = get_valid_item_indices(
    filter_params={'brand': 'Innisfree'},
    metadata=item_metadata,
    idx_to_item=idx_to_item  # From mappings
)

# Returns: Set[int] - Set of valid item indices matching filters
# Used for pre-filtering during scoring (more efficient)
```

#### Function: `exclude_products()`

```python
from service.recommender.filters import exclude_products

filtered = exclude_products(
    recommendations=recs,
    exclude_ids={100, 200, 300}  # Product IDs to exclude
)

# Returns: Filtered recommendations without excluded products
```

#### Function: `boost_by_attributes()`

```python
from service.recommender.filters import boost_by_attributes

boosted = boost_by_attributes(
    recommendations=recs,
    boost_config={
        'brand': {'Innisfree': 1.2, 'Cetaphil': 1.1},
        'skin_type': {'oily': 1.1}
    },
    metadata=item_metadata
)

# Returns: Recommendations with boosted scores
```

#### Function: `boost_by_user_preferences()`

```python
from service.recommender.filters import boost_by_user_preferences

boosted = boost_by_user_preferences(
    recommendations=recs,
    user_preferences={
        'preferred_brands': ['Innisfree', 'CeraVe'],
        'skin_type': 'oily'
    },
    metadata=item_metadata
)

# Returns: Recommendations boosted by user preferences
```

#### Function: `infer_user_preferences()`

```python
from service.recommender.filters import infer_user_preferences

prefs = infer_user_preferences(
    user_history=[100, 200, 300],  # Product IDs
    metadata=item_metadata
)

# Returns: Dict with inferred brand, skin_type preferences
# {
#     'preferred_brands': ['Innisfree', 'Cetaphil'],
#     'skin_types': ['oily', 'acne'],
#     'price_range': {'min': 150000, 'max': 500000}
# }
# Used for personalized boosting
```

#### Function: `filter_and_boost()`

```python
from service.recommender.filters import filter_and_boost

result = filter_and_boost(
    recommendations=recs,
    filter_params={'brand': 'Innisfree', 'min_rating': 4.0},
    boost_config={'brand': {'Innisfree': 1.15}},
    metadata=item_metadata
)

# Combined filter + boost in one call
# Returns: Filtered and boosted recommendations
```

## Timeline Estimate

- **Loader + Recommender**: 2 days
- **Fallback logic**: 0.5 day
- **API endpoints**: 1 day
- **BERT integration + Reranking**: 2 days
- **Testing**: 1 day
- **Deployment setup**: 0.5 day
- **Total**: ~7 days

## Integration Points

### Task 01 (Data Layer)
- Uses `user_item_mappings.json` for ID mappings
- Uses `trainable_user_mapping.json` for routing
- Uses `top_k_popular_items.json` for fallback
- Uses `enriched_products.parquet` for metadata
- Uses `interactions.parquet` (train split) for user histories

### Task 02 (Training Pipelines)
- Loads models from registry (Task 04)
- Uses `score_range` from model metadata for normalization

### Task 03 (Evaluation Metrics)
- Uses evaluation metrics for model comparison
- Can integrate with evaluation pipeline

### Task 04 (Model Registry)
- Uses `ModelLoader` from registry for model loading
- Hot-reload when registry updates

### Task 06 (Monitoring)
- Logs requests to `ServiceMetricsDB`
- Tracks latency, fallback rate, error rate

### Task 08 (Hybrid Reranking)
- Uses `HybridReranker` for weighted signal combination
- Uses `RerankerConfig` for weight management

### Task 09 (Smart Search)
- Integrated search endpoints in API
- Shares `PhoBERTEmbeddingLoader` with recommender

## Success Criteria

- [x] Load model từ registry (<1 second)
- [x] Generate recommendations (<100ms per user CF-only)
- [x] Two-stage reranking (<200ms with BERT)
- [x] BERT embeddings loaded and cached
- [x] Cold-start fallback works (popularity, content, hybrid)
- [x] API endpoints functional (recommend, batch, similar, search)
- [x] Hot-reload model without downtime
- [x] Logging tracks latency, fallback rate, rerank metrics
- [x] Cache manager with warm-up for cold-start optimization
- [x] User segmentation routing (trainable vs cold-start)
- [x] Thread-safe singleton loaders
- [x] Docker deployment ready
- [x] Rate limiting and security headers
- [x] Graceful empty mode when model not available
- [x] Search service with PhoBERT warmup

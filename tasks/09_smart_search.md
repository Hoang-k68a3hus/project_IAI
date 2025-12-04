# Task 09: Smart Search - Tích Hợp Tìm Kiếm Ngữ Nghĩa ✅ ĐÃ HOÀN THÀNH

## Mục Tiêu

Tích hợp tính năng **Smart Search** (tìm kiếm thông minh) vào hệ thống recommendation service, sử dụng PhoBERT embeddings đã được tạo ra từ các task trước. Tính năng này cho phép người dùng tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên (tiếng Việt) thay vì chỉ dựa trên keyword matching truyền thống.

---

## ✅ Trạng Thái Triển Khai (Tháng 11/2025)

**Các thành phần đã hoàn thành**:

| Thành phần | File | Trạng thái |
|------------|------|------------|
| QueryEncoder | `service/search/query_encoder.py` | ✅ Hoàn thành |
| SearchIndex | `service/search/search_index.py` | ✅ Hoàn thành |
| SmartSearchService | `service/search/smart_search.py` | ✅ Hoàn thành |
| API Endpoints | `service/api.py` | ✅ Hoàn thành |
| Test Script | `service/search/test_search_features.py` | ✅ Hoàn thành |

---

## 📊 Phụ Thuộc Dữ Liệu

**Embeddings đã có sẵn từ các tasks trước**:
- **Product Embeddings**: `data/processed/content_based_embeddings/product_embeddings.pt`
  - Chứa BERT embeddings cho ~2,200 products
  - Dimension: 768 (PhoBERT-base)
  - Pre-normalized vectors cho fast cosine similarity
- **PhoBERTEmbeddingLoader** (Task 05/08): Singleton class đã implement loading và similarity computation

**Lợi thế so với keyword search truyền thống**:
- Hiểu ngữ nghĩa tiếng Việt (synonyms, paraphrases)
- Xử lý viết tắt và biến thể từ vựng (srm → sữa rửa mặt, kcn → kem chống nắng)
- Tìm kiếm theo intent/concept, không chỉ exact match

---

## 🎯 Các Trường Hợp Sử Dụng

### Use Case 1: Tìm Kiếm Sản Phẩm
```
User: "tìm kem dưỡng da cho da dầu mụn"
→ Semantic search: tìm products có embeddings gần với query embedding
→ Return: kem trị mụn, gel kiểm soát dầu, serum BHA, etc.
```

### Use Case 2: Tìm Sản Phẩm Tương Tự
```
User: "sản phẩm tương tự [product_id=123]"
→ Item-item similarity từ PhoBERT embeddings
→ Return: top-K similar products
```

### Use Case 3: Tìm Theo Hồ Sơ Người Dùng
```
User với lịch sử: [product_1, product_2, product_3]
→ Tính user profile embedding từ lịch sử
→ Return: products tương tự về semantic với sở thích
```

### Use Case 4: Hybrid Search (Thuộc Tính + Ngữ Nghĩa)
```
User: "kem chống nắng cho da nhạy cảm"
Filter: brand='Innisfree', max_price=500000
→ Filter theo thuộc tính trước
→ Rank theo embedding similarity
→ Return: filtered & ranked products
```

---

## 🏗️ Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Search Service                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  QueryEncoder   │  │   SearchIndex   │  │ PhoBERT      │ │
│  │  ─────────────  │  │   ───────────   │  │ Loader       │ │
│  │  • Singleton    │  │  • Exact search │  │ ───────────  │ │
│  │  • Vietnamese   │  │  • FAISS ANN    │  │ • Embeddings │ │
│  │    preprocessing│  │  • Metadata     │  │ • Similarity │ │
│  │  • LRU Cache    │  │    filtering    │  │ • Profile    │ │
│  │  • Batch encode │  │  • Thread-safe  │  │   compute    │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
│           └────────────────────┼───────────────────┘         │
│                                │                             │
│                    ┌───────────▼───────────┐                 │
│                    │  SmartSearchService   │                 │
│                    │  ─────────────────    │                 │
│                    │  • search()           │                 │
│                    │  • search_similar()   │                 │
│                    │  • search_by_profile()│                 │
│                    │  • Multi-signal rerank│                 │
│                    └───────────┬───────────┘                 │
│                                │                             │
└────────────────────────────────┼─────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      API Endpoints      │
                    │  ────────────────────   │
                    │  POST /search           │
                    │  POST /search/similar   │
                    │  POST /search/profile   │
                    │  GET  /search/filters   │
                    │  GET  /search/stats     │
                    └─────────────────────────┘
```

---

## Thành Phần 1: QueryEncoder ✅

### Module: `service/search/query_encoder.py`

**Mô tả**: Singleton encoder để chuyển đổi text queries thành embeddings sử dụng PhoBERT.

#### Tính năng chính:
1. **Singleton pattern**: Thread-safe, chỉ load model một lần
2. **Vietnamese preprocessing**: Mở rộng viết tắt, chuẩn hóa text
3. **LRU Cache**: Cache embeddings để tăng tốc queries lặp lại
4. **Batch encoding**: Encode nhiều queries cùng lúc hiệu quả

```python
class QueryEncoder:
    """
    Encode text queries to embeddings using PhoBERT.
    
    Features:
    - Lazy loading of PhoBERT model
    - Query embedding caching (LRU)
    - Batch encoding for efficiency
    - Vietnamese text preprocessing with abbreviation expansion
    
    Example:
        >>> encoder = QueryEncoder()
        >>> emb = encoder.encode("kem dưỡng da cho da dầu")
        >>> embeddings = encoder.encode_batch(["query1", "query2"])
    """
    
    _instance: Optional['QueryEncoder'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        max_length: int = 256,
        cache_size: int = 1000,
        device: str = "cpu",
        abbreviations: Optional[Dict[str, str]] = None
    ):
        """
        Initialize QueryEncoder.
        
        Args:
            model_name: HuggingFace model name for PhoBERT
            max_length: Maximum sequence length for tokenization
            cache_size: Size of LRU cache for query embeddings
            device: Device for model inference ('cpu' or 'cuda')
            abbreviations: Additional abbreviation mappings
        """
```

#### Vietnamese Abbreviations Mapping:

```python
VIETNAMESE_ABBREVIATIONS = {
    # Product abbreviations
    'sp': 'sản phẩm',
    'kem dc': 'kem dưỡng da',
    'kem dd': 'kem dưỡng da',
    'srm': 'sữa rửa mặt',
    'tbc': 'tẩy bong chết',
    'tdc': 'tẩy da chết',
    'kcn': 'kem chống nắng',
    'cn': 'chống nắng',
    'nc': 'nước',
    'nht': 'nước hoa hồng',
    
    # Skin type abbreviations
    'dn': 'da nhờn',
    'dk': 'da khô',
    'dh': 'da hỗn hợp',
    'dnc': 'da nhạy cảm',
    'dm': 'da mụn',
    
    # Common abbreviations
    'ko': 'không',
    'dc': 'được',
    'vs': 'với',
    'cx': 'cũng',
    'ntn': 'như thế nào',
}
```

#### Phương thức quan trọng:

##### `preprocess_query()` - Tiền xử lý tiếng Việt

```python
def preprocess_query(self, query: str) -> str:
    """
    Preprocess Vietnamese query text.
    
    Steps:
    1. Lowercase and strip whitespace
    2. Expand abbreviations
    3. Normalize whitespace
    4. Remove special characters (keep Vietnamese)
    
    Args:
        query: Raw query text
    
    Returns:
        Preprocessed query string
    """
```

##### `encode()` - Encode một query

```python
def encode(
    self,
    query: str,
    normalize: bool = True,
    use_cache: bool = True
) -> np.ndarray:
    """
    Encode a single query to embedding.
    
    Args:
        query: Text query (Vietnamese)
        normalize: L2 normalize the embedding for cosine similarity
        use_cache: Use LRU cache for repeated queries
    
    Returns:
        np.ndarray of shape (embedding_dim,)
    """
```

##### `encode_batch()` - Encode nhiều queries

```python
def encode_batch(
    self,
    queries: List[str],
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Encode multiple queries efficiently with batching.
    
    Args:
        queries: List of text queries
        normalize: L2 normalize embeddings
        batch_size: Batch size for encoding
        show_progress: Show progress bar (requires tqdm)
    
    Returns:
        np.ndarray of shape (num_queries, embedding_dim)
    """
```

#### LRU Cache Implementation:

```python
class LRUCache:
    """Simple LRU cache for query embeddings."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache, moving to end if found."""
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Put item in cache, evicting oldest if at capacity."""
    
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {'size': len(self.cache), 'capacity': self.capacity}
```

---

## Thành Phần 2: SearchIndex ✅

### Module: `service/search/search_index.py`

**Mô tả**: Index cho tìm kiếm similarity nhanh, hỗ trợ exact search và FAISS ANN.

#### Tính năng chính:
1. **Exact cosine similarity search**: Cho catalog nhỏ (<5K products)
2. **FAISS ANN search**: Cho catalog lớn, hỗ trợ flat, ivf, hnsw
3. **Metadata filtering**: Lọc theo brand, category, price range
4. **Thread-safe operations**: An toàn cho multi-threading

```python
class SearchIndex:
    """
    Search index for semantic product search.
    
    Features:
    - Exact cosine similarity search (for small catalogs)
    - FAISS ANN search (optional, for large catalogs >5K items)
    - Metadata filtering (brand, category, price range)
    - Thread-safe operations
    - Integration with PhoBERTEmbeddingLoader
    
    Example:
        >>> index = SearchIndex()
        >>> index.build_index()
        >>> results = index.search(query_embedding, topk=10)
        >>> results = index.search_with_filter(query_embedding, filters={'brand': 'Innisfree'})
    """
    
    def __init__(
        self,
        phobert_loader=None,
        product_metadata=None,
        use_faiss: bool = False,
        faiss_index_type: str = "flat",
        auto_build: bool = False
    ):
        """
        Initialize SearchIndex.
        
        Args:
            phobert_loader: PhoBERTEmbeddingLoader instance
            product_metadata: DataFrame with product info
            use_faiss: Use FAISS for ANN search (faster for large catalogs)
            faiss_index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            auto_build: Automatically build index on init
        """
```

#### Phương thức quan trọng:

##### `build_index()` - Xây dựng index

```python
def build_index(self) -> None:
    """
    Build search index from embeddings.
    
    Loads embeddings from PhoBERTEmbeddingLoader and builds
    necessary indices for fast similarity search.
    """
```

##### `_build_faiss_index()` - Xây dựng FAISS index

```python
def _build_faiss_index(self) -> None:
    """
    Build FAISS index for approximate nearest neighbor search.
    
    Supported index types:
    - flat: Exact search (brute force) - good for <10K items
    - ivf: IVF index - good for 10K-1M items
    - hnsw: HNSW - fast approximate search
    """
```

##### `_build_metadata_indices()` - Xây dựng inverted indices

```python
def _build_metadata_indices(self) -> None:
    """
    Build inverted indices for metadata filtering.
    
    Creates:
    - brand_index: Dict[str, Set[int]] - brand → product_ids
    - category_index: Dict[str, Set[int]] - category → product_ids
    - price_data: Dict[int, float] - product_id → price
    """
```

##### `search()` - Tìm kiếm cơ bản

```python
def search(
    self,
    query_embedding: np.ndarray,
    topk: int = 10,
    exclude_ids: Optional[Set[int]] = None
) -> List[Tuple[int, float]]:
    """
    Search for similar products.
    
    Args:
        query_embedding: Query embedding vector (should be normalized)
        topk: Number of results to return
        exclude_ids: Product IDs to exclude from results
    
    Returns:
        List of (product_id, similarity_score) tuples
    """
```

##### `search_with_filter()` - Tìm kiếm với bộ lọc

```python
def search_with_filter(
    self,
    query_embedding: np.ndarray,
    topk: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    exclude_ids: Optional[Set[int]] = None
) -> List[Tuple[int, float]]:
    """
    Search with metadata filtering.
    
    Args:
        query_embedding: Query embedding
        topk: Number of results
        filters: Metadata filters:
            - 'brand': Brand name (string, case-insensitive)
            - 'category': Category/type name (string, case-insensitive)
            - 'min_price': Minimum price (float)
            - 'max_price': Maximum price (float)
        exclude_ids: IDs to exclude
    
    Returns:
        Filtered and ranked results as list of (product_id, score) tuples
    """
```

---

## Thành Phần 3: SmartSearchService ✅

### Module: `service/search/smart_search.py`

**Mô tả**: Service chính cho tìm kiếm sản phẩm ngữ nghĩa với multi-signal reranking.

#### Tính năng chính:
1. **Text-to-product search**: Tìm kiếm bằng text tiếng Việt
2. **Item-to-item similarity**: Tìm sản phẩm tương tự
3. **User profile search**: Tìm kiếm dựa trên lịch sử người dùng
4. **Multi-signal reranking**: Kết hợp semantic, popularity, quality, recency

```python
class SmartSearchService:
    """
    Smart Search Service for semantic product discovery.
    
    Features:
    - Text-to-product semantic search using PhoBERT
    - Item-to-item similarity search
    - User profile-based recommendations
    - Hybrid search with attribute filters
    - Multi-signal reranking (semantic, popularity, quality)
    
    Example:
        >>> service = SmartSearchService()
        >>> results = service.search("kem dưỡng da cho da dầu", topk=10)
        >>> similar = service.search_similar(product_id=123, topk=10)
    """
```

#### Cấu hình mặc định:

```python
DEFAULT_CONFIG = {
    'default_topk': 10,
    'max_topk': 100,
    'min_semantic_score': 0.25,  # Minimum score to include in results
    'enable_rerank': True,
    'candidate_multiplier': 3,   # Fetch 3x candidates for reranking
    
    # Reranking weights
    'rerank_weights': {
        'semantic': 0.50,
        'popularity': 0.25,
        'quality': 0.15,
        'recency': 0.10
    },
    
    # User profile config
    'user_profile': {
        'strategy': 'weighted_mean',  # 'mean', 'weighted_mean', 'max'
        'max_history_items': 50       # Limit history items for profile
    }
}
```

#### Data Classes:

```python
@dataclass
class SearchResult:
    """Single search result."""
    product_id: int
    product_name: str
    semantic_score: float
    final_score: float
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    avg_rating: Optional[float] = None
    num_sold: Optional[int] = None
    signals: Optional[Dict[str, float]] = None
    rank: int = 0


@dataclass
class SearchResponse:
    """Search response container."""
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float
    method: str  # 'semantic', 'hybrid', 'similar_items', 'user_profile', 'popular'
    filters_applied: Optional[Dict[str, Any]] = None
```

#### Phương thức quan trọng:

##### `search()` - Tìm kiếm ngữ nghĩa

```python
def search(
    self,
    query: str,
    topk: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    exclude_ids: Optional[Set[int]] = None,
    rerank: bool = True
) -> SearchResponse:
    """
    Semantic search for products.
    
    Args:
        query: Text query in Vietnamese (e.g., "kem dưỡng da cho da dầu")
        topk: Number of results to return
        filters: Attribute filters:
            - 'brand': Brand name (string)
            - 'category': Category name (string)
            - 'min_price': Minimum price (float)
            - 'max_price': Maximum price (float)
        exclude_ids: Product IDs to exclude from results
        rerank: Apply multi-signal reranking
    
    Returns:
        SearchResponse with ranked results
    
    Example:
        >>> results = service.search("kem dưỡng ẩm cho da khô", topk=10)
        >>> results = service.search("sữa rửa mặt", filters={'brand': 'innisfree'})
    """
```

##### `search_similar()` - Tìm sản phẩm tương tự

```python
def search_similar(
    self,
    product_id: int,
    topk: int = 10,
    exclude_self: bool = True,
    exclude_ids: Optional[Set[int]] = None
) -> SearchResponse:
    """
    Find products similar to a given product.
    
    Uses PhoBERT embeddings for semantic similarity.
    
    Args:
        product_id: Source product ID
        topk: Number of similar products to return
        exclude_self: Exclude source product from results
        exclude_ids: Additional IDs to exclude
    
    Returns:
        SearchResponse with similar products
    """
```

##### `search_by_user_profile()` - Tìm theo hồ sơ người dùng

```python
def search_by_user_profile(
    self,
    user_history: List[int],
    topk: int = 10,
    exclude_history: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    weights: Optional[List[float]] = None
) -> SearchResponse:
    """
    Search products similar to user's interaction history.
    
    Computes a user profile embedding from history and finds similar products.
    Useful for cold-start personalization based on browsing history.
    
    Args:
        user_history: List of product IDs user has interacted with
        topk: Number of results
        exclude_history: Exclude products from history in results
        filters: Attribute filters
        weights: Optional weights for each history item (e.g., recency, rating)
    
    Returns:
        SearchResponse with personalized recommendations
    """
```

##### `_rerank_results()` - Multi-signal reranking

```python
def _rerank_results(
    self,
    raw_results: List[Tuple[int, float]],
    topk: int
) -> List[SearchResult]:
    """
    Rerank results using multiple signals.
    
    Signals:
    - semantic: Embedding similarity (from search)
    - popularity: num_sold_time or popularity_score
    - quality: avg_rating or quality_score
    - recency: Product freshness (placeholder)
    
    Công thức:
        final_score = Σ weight_i × signal_i
    
    Mặc định:
        semantic=0.50, popularity=0.25, quality=0.15, recency=0.10
    """
```

---

## Thành Phần 4: API Endpoints ✅

### Module: `service/api.py`

#### Request/Response Models:

```python
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
```

#### Endpoints:

##### `POST /search` - Tìm kiếm sản phẩm

```python
@app.post("/search", response_model=SearchResponse)
async def search_products(request: Request, search_request: SearchRequest):
    """
    Smart semantic search for products.
    
    Uses PhoBERT embeddings for Vietnamese semantic search.
    Supports attribute filtering and multi-signal reranking.
    
    Example:
        POST /search
        {
            "query": "kem dưỡng da cho da dầu",
            "topk": 10,
            "filters": {"brand": "innisfree"},
            "rerank": true
        }
    """
```

##### `POST /search/similar` - Tìm sản phẩm tương tự

```python
@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_products(request: SimilarSearchRequest):
    """
    Find products similar to a given product.
    
    Uses PhoBERT embeddings for item-item similarity.
    
    Example:
        POST /search/similar
        {
            "product_id": 123,
            "topk": 10,
            "exclude_self": true
        }
    """
```

##### `POST /search/profile` - Tìm theo hồ sơ người dùng

```python
@app.post("/search/profile", response_model=SearchResponse)
async def search_by_profile(request: SearchByProfileRequest):
    """
    Search products based on user browsing/interaction history.
    
    Useful for cold-start personalization.
    
    Example:
        POST /search/profile
        {
            "product_history": [123, 456, 789],
            "topk": 10,
            "exclude_history": true
        }
    """
```

##### `GET /search/filters` - Lấy danh sách bộ lọc

```python
@app.get("/search/filters")
async def get_search_filters():
    """
    Get available filter options.
    
    Returns:
        - brands: List of available brand names
        - categories: List of available category names
        - price_range: (min_price, max_price)
    """
```

##### `GET /search/stats` - Thống kê search service

```python
@app.get("/search/stats")
async def get_search_stats():
    """
    Get search service statistics.
    
    Returns:
        - total_searches: Total number of searches
        - avg_latency_ms: Average search latency
        - errors: Number of errors
        - index stats: Index statistics
        - encoder stats: Encoder statistics
    """
```

---

## Thành Phần 5: Test Script ✅

### Module: `service/search/test_search_features.py`

```python
"""
Test script để verify các tính năng của module search hoạt động đúng.

Chạy script này để kiểm tra:
- QueryEncoder encoding
- SearchIndex search functionality
- SmartSearchService các tính năng tìm kiếm

Usage:
    python service/search/test_search_features.py
"""

def test_query_encoder():
    """Test QueryEncoder functionality."""
    # Test 1: Preprocessing với Vietnamese text
    # Test 2: Encoding (nếu model có sẵn)
    # Test 3: Cache functionality

def test_search_index():
    """Test SearchIndex functionality."""
    # Test 1: Initialization
    # Test 2: Build index
    # Test 3: Available filters

def test_smart_search_service():
    """Test SmartSearchService functionality."""
    # Test 1: Initialization
    # Test 2: Text search
    # Test 3: Similar items search
    # Test 4: Service stats
```

---

## Cấu Trúc Thư Mục

```
service/
├─ search/
│  ├─ __init__.py              # Module exports
│  ├─ query_encoder.py         # QueryEncoder class
│  ├─ search_index.py          # SearchIndex class
│  ├─ smart_search.py          # SmartSearchService class
│  └─ test_search_features.py  # Test script
├─ api.py                      # API endpoints (updated)
└─ recommender/
   └─ phobert_loader.py        # Shared PhoBERT embeddings
```

---

## Hướng Dẫn Sử Dụng

### 1. Tìm kiếm sản phẩm bằng text

```python
from service.search import get_search_service

service = get_search_service()

# Tìm kiếm đơn giản
results = service.search("kem dưỡng da cho da dầu", topk=10)
print(f"Tìm thấy: {results.count} sản phẩm")
print(f"Latency: {results.latency_ms:.2f}ms")

for item in results.results:
    print(f"  {item.rank}. {item.product_name}")
    print(f"     Score: {item.final_score:.3f}")
    print(f"     Brand: {item.brand}")

# Tìm kiếm với bộ lọc
results = service.search(
    "sữa rửa mặt",
    topk=10,
    filters={
        'brand': 'innisfree',
        'max_price': 300000
    }
)
```

### 2. Tìm sản phẩm tương tự

```python
# Tìm sản phẩm tương tự với product_id=123
similar = service.search_similar(
    product_id=123,
    topk=10,
    exclude_self=True
)

for item in similar.results:
    print(f"  {item.product_name}: {item.semantic_score:.3f}")
```

### 3. Tìm kiếm theo lịch sử người dùng

```python
# Người dùng đã xem các sản phẩm này
user_history = [101, 102, 103, 104]

# Tìm sản phẩm phù hợp với sở thích
recommendations = service.search_by_user_profile(
    user_history=user_history,
    topk=10,
    exclude_history=True
)

print(f"Gợi ý cho người dùng: {recommendations.count} sản phẩm")
```

### 4. Sử dụng API

```bash
# Tìm kiếm
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "kem dưỡng da", "topk": 10}'

# Tìm sản phẩm tương tự
curl -X POST http://localhost:8000/search/similar \
  -H "Content-Type: application/json" \
  -d '{"product_id": 123, "topk": 10}'

# Lấy danh sách bộ lọc
curl http://localhost:8000/search/filters

# Xem thống kê
curl http://localhost:8000/search/stats
```

---

## Tích Hợp Liên Task

| Task | Điểm tích hợp | Trạng thái |
|------|---------------|------------|
| Task 05 (Serving) | Chia sẻ `PhoBERTEmbeddingLoader` singleton | ✅ |
| Task 08 (Hybrid Reranking) | Reuse reranking logic và normalization | ✅ |
| Task 06 (Monitoring) | Log search queries và latencies | ✅ |
| Task 01 (Data Layer) | Sử dụng product metadata enriched | ✅ |

---

## Mục Tiêu Hiệu Năng

| Metric | Mục tiêu | Ghi chú |
|--------|----------|---------|
| Latency P50 | <100ms | Median response |
| Latency P95 | <300ms | 95% requests |
| Latency P99 | <500ms | SLA target |
| Cache hit rate | >70% | Cho repeated queries |
| Min semantic score | 0.25 | Threshold để include |

---

## Tiêu Chí Thành Công ✅ ĐẠT ĐƯỢC

- [x] QueryEncoder với singleton pattern và LRU cache
- [x] Vietnamese preprocessing với abbreviation expansion
- [x] SearchIndex hỗ trợ exact và FAISS search
- [x] Metadata filtering (brand, category, price)
- [x] SmartSearchService với multi-signal reranking
- [x] Text search, similar items, user profile search
- [x] API endpoints đầy đủ với Pydantic models
- [x] Thread-safe operations
- [x] Test script cho tất cả components
- [x] Tích hợp với PhoBERTEmbeddingLoader có sẵn

---

## Mở Rộng Tương Lai

1. **Query Understanding**:
   - Intent classification (browse, compare, specific search)
   - Query expansion with synonyms
   - Spell correction for Vietnamese

2. **Personalization**:
   - Learn from search click history
   - User preference weighting
   - Session-based personalization

3. **Advanced Ranking**:
   - Learning-to-rank models
   - A/B testing framework
   - Dynamic weight adjustment

4. **Scalability**:
   - Distributed FAISS index
   - Redis caching layer
   - Async query processing

---

**Created**: 2025-11-26  
**Updated**: 2025-11-30  
**Status**: ✅ Hoàn thành  
**Priority**: High

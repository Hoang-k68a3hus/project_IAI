# Hướng Dẫn Smart Search

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Kiến Trúc Tổng Quan](#kiến-trúc-tổng-quan)
3. [Cài Đặt & Cấu Hình](#cài-đặt--cấu-hình)
4. [Các Thành Phần Chính](#các-thành-phần-chính)
5. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
6. [API Reference](#api-reference)
7. [Xử Lý Tiếng Việt](#xử-lý-tiếng-việt)
8. [Tối Ưu Hiệu Năng](#tối-ưu-hiệu-năng)
9. [Xử Lý Sự Cố](#xử-lý-sự-cố)

---

## Giới Thiệu

### Smart Search là gì?

Smart Search là tính năng tìm kiếm ngữ nghĩa (semantic search) cho phép người dùng tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên tiếng Việt. Thay vì chỉ khớp từ khóa (keyword matching), hệ thống hiểu được ý nghĩa của câu truy vấn.

### So sánh với Keyword Search

| Tính năng | Keyword Search | Smart Search |
|-----------|----------------|--------------|
| "kem dưỡng da" | Chỉ tìm có từ "kem dưỡng da" | Tìm cả moisturizer, lotion, cream |
| "srm cho da nhờn" | Không hiểu viết tắt | Hiểu srm = sữa rửa mặt |
| Typo/lỗi chính tả | Không tìm được | Vẫn tìm được sản phẩm liên quan |
| Đồng nghĩa | Không hiểu | Hiểu "làm sạch" ≈ "rửa mặt" |

### Các tính năng chính

```
┌─────────────────────────────────────────────────────────────┐
│                      SMART SEARCH                           │
├─────────────────────────────────────────────────────────────┤
│  🔍 Tìm kiếm ngữ nghĩa    │  Hiểu ý nghĩa, không chỉ từ khóa│
│  📦 Sản phẩm tương tự     │  Tìm items giống với item đã chọn│
│  👤 Theo hồ sơ người dùng │  Gợi ý dựa trên lịch sử xem     │
│  🏷️ Bộ lọc thuộc tính     │  Brand, category, price range   │
│  ⚡ Reranking đa tín hiệu │  Semantic + Popularity + Quality│
└─────────────────────────────────────────────────────────────┘
```

---

## Kiến Trúc Tổng Quan

### Luồng Xử Lý Tìm Kiếm

```
                    ┌──────────────────┐
                    │   User Query     │
                    │ "kem dưỡng da"   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  QueryEncoder    │
                    │  ─────────────   │
                    │  1. Preprocessing│
                    │  2. Expand abbr  │
                    │  3. Encode BERT  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Query Embedding │
                    │  (768 dimensions)│
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Có bộ lọc?       │         │  Không có lọc     │
    │  (brand, price)   │         │                   │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │ search_with_filter│         │ search (all)      │
    │ Lọc trước, rank   │         │ Similarity search │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Multi-Signal    │
                    │  Reranking       │
                    │  ─────────────   │
                    │  semantic: 50%   │
                    │  popularity: 25% │
                    │  quality: 15%    │
                    │  recency: 10%    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Top-K Results   │
                    │  với metadata    │
                    └──────────────────┘
```

### Các thành phần

```
service/search/
├── __init__.py              # Module exports
├── query_encoder.py         # Encode text → embedding
├── search_index.py          # Similarity search index
├── smart_search.py          # Main service class
└── test_search_features.py  # Test script
```

---

## Cài Đặt & Cấu Hình

### Yêu cầu packages

```bash
# Core packages
pip install torch transformers sentence-transformers
pip install numpy scipy

# Optional: FAISS for large catalogs
pip install faiss-cpu
# hoặc với GPU
pip install faiss-gpu
```

### Cấu hình mặc định

```python
# Trong smart_search.py
DEFAULT_CONFIG = {
    'default_topk': 10,
    'max_topk': 100,
    'min_semantic_score': 0.25,  # Ngưỡng tối thiểu để hiển thị
    'enable_rerank': True,
    'candidate_multiplier': 3,   # Lấy 3x ứng viên cho reranking
    
    # Trọng số reranking
    'rerank_weights': {
        'semantic': 0.50,      # Độ tương đồng ngữ nghĩa
        'popularity': 0.25,    # Độ phổ biến (num_sold)
        'quality': 0.15,       # Chất lượng (avg_rating)
        'recency': 0.10        # Độ mới (placeholder)
    },
    
    # Cấu hình user profile
    'user_profile': {
        'strategy': 'weighted_mean',  # 'mean', 'weighted_mean', 'max'
        'max_history_items': 50       # Giới hạn lịch sử
    }
}
```

### Các file dữ liệu cần thiết

```
data/
├── processed/
│   ├── content_based_embeddings/
│   │   └── product_embeddings.pt    # Vietnamese Embedding (~2.2K products)
│   └── product_attributes_enriched.parquet  # Product metadata
│
└── published_data/
    └── data_product.csv             # Fallback metadata
```

---

## Các Thành Phần Chính

### 1. QueryEncoder

**Chức năng**: Chuyển đổi text query thành embedding vector sử dụng model `AITeamVN/Vietnamese_Embedding`.

```python
from service.search.query_encoder import QueryEncoder, get_query_encoder

# Lấy singleton instance
encoder = get_query_encoder()

# Encode một query
embedding = encoder.encode("kem dưỡng da cho da dầu")
print(f"Shape: {embedding.shape}")  # (768,)

# Encode nhiều queries cùng lúc
queries = ["sữa rửa mặt", "kem chống nắng", "serum vitamin C"]
embeddings = encoder.encode_batch(queries)
print(f"Shape: {embeddings.shape}")  # (3, 768)

# Xem preprocessing
processed = encoder.preprocess_query("srm cho dn")
print(processed)  # "sữa rửa mặt cho da nhờn"

# Xem statistics
stats = encoder.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

#### Viết tắt được hỗ trợ:

| Viết tắt | Mở rộng | Viết tắt | Mở rộng |
|----------|---------|----------|---------|
| sp | sản phẩm | dn | da nhờn |
| srm | sữa rửa mặt | dk | da khô |
| kcn | kem chống nắng | dh | da hỗn hợp |
| tdc | tẩy da chết | dnc | da nhạy cảm |
| nht | nước hoa hồng | dm | da mụn |
| ko | không | dc | được |

### 2. SearchIndex

**Chức năng**: Quản lý index để tìm kiếm similarity nhanh.

```python
from service.search.search_index import SearchIndex, get_search_index

# Lấy singleton instance
index = get_search_index()

# Build index (tự động khi cần)
index.build_index()
print(f"Indexed products: {index.num_products}")

# Tìm kiếm cơ bản
query_embedding = encoder.encode("kem dưỡng da")
results = index.search(query_embedding, topk=10)
for product_id, score in results:
    print(f"Product {product_id}: {score:.3f}")

# Tìm kiếm với bộ lọc
results = index.search_with_filter(
    query_embedding,
    topk=10,
    filters={
        'brand': 'innisfree',
        'category': 'kem dưỡng',
        'min_price': 100000,
        'max_price': 500000
    }
)

# Xem bộ lọc có sẵn
print(f"Brands: {index.get_available_brands()[:5]}")
print(f"Categories: {index.get_available_categories()[:5]}")
print(f"Price range: {index.get_price_range()}")
```

#### Các loại index FAISS:

| Loại | Catalog size | Tốc độ | Độ chính xác |
|------|--------------|--------|--------------|
| `flat` | <10K | Chậm nhất | 100% (exact) |
| `ivf` | 10K-1M | Nhanh | ~95% |
| `hnsw` | 10K-10M | Rất nhanh | ~90% |

### 3. SmartSearchService

**Chức năng**: Service chính tích hợp tất cả components.

```python
from service.search import get_search_service

# Lấy singleton instance
service = get_search_service()

# Tìm kiếm text
results = service.search("kem dưỡng da cho da dầu", topk=10)

# Tìm sản phẩm tương tự
similar = service.search_similar(product_id=123, topk=10)

# Tìm theo hồ sơ người dùng
profile_results = service.search_by_user_profile(
    user_history=[101, 102, 103],
    topk=10
)

# Xem statistics
stats = service.get_stats()
print(f"Total searches: {stats['total_searches']}")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
```

---

## Hướng Dẫn Sử Dụng

### Trường hợp 1: Tìm kiếm sản phẩm đơn giản

```python
from service.search import get_search_service

service = get_search_service()

# Tìm kiếm
results = service.search("kem dưỡng ẩm cho da khô", topk=10)

# Kiểm tra kết quả
print(f"Tìm thấy: {results.count} sản phẩm")
print(f"Phương thức: {results.method}")
print(f"Latency: {results.latency_ms:.2f}ms")

# Hiển thị kết quả
for item in results.results:
    print(f"\n{item.rank}. {item.product_name}")
    print(f"   Brand: {item.brand}")
    print(f"   Price: {item.price:,.0f}đ" if item.price else "   Price: N/A")
    print(f"   Rating: {item.avg_rating:.1f}⭐" if item.avg_rating else "   Rating: N/A")
    print(f"   Semantic score: {item.semantic_score:.3f}")
    print(f"   Final score: {item.final_score:.3f}")
```

### Trường hợp 2: Tìm kiếm với bộ lọc

```python
# Lọc theo brand
results = service.search(
    "sữa rửa mặt",
    topk=10,
    filters={'brand': 'innisfree'}
)

# Lọc theo nhiều tiêu chí
results = service.search(
    "serum vitamin c",
    topk=10,
    filters={
        'brand': 'some by mi',
        'category': 'serum',
        'min_price': 200000,
        'max_price': 600000
    }
)

# Xem bộ lọc đã áp dụng
print(f"Filters: {results.filters_applied}")
```

### Trường hợp 3: Tìm sản phẩm tương tự

```python
# Tìm sản phẩm tương tự với product_id=123
similar = service.search_similar(
    product_id=123,
    topk=10,
    exclude_self=True  # Không bao gồm sản phẩm gốc
)

print(f"Sản phẩm tương tự với product #{123}:")
for item in similar.results:
    print(f"  - {item.product_name} ({item.semantic_score:.3f})")
```

### Trường hợp 4: Tìm theo lịch sử người dùng

```python
# Người dùng đã xem các sản phẩm này
user_history = [101, 102, 103, 104, 105]

# Tìm sản phẩm phù hợp với sở thích
recommendations = service.search_by_user_profile(
    user_history=user_history,
    topk=10,
    exclude_history=True,  # Không gợi ý sản phẩm đã xem
    filters={'category': 'kem dưỡng'}  # Có thể kết hợp filter
)

print(f"Gợi ý cho người dùng:")
for item in recommendations.results:
    print(f"  {item.rank}. {item.product_name}")
```

### Trường hợp 5: Tắt reranking

```python
# Chỉ dùng semantic score thuần (không rerank)
results = service.search(
    "nước hoa hồng",
    topk=10,
    rerank=False  # Tắt multi-signal reranking
)

# Kết quả sắp xếp theo semantic_score thay vì final_score
for item in results.results:
    print(f"{item.product_name}: {item.semantic_score:.3f}")
```

---

## API Reference

### Endpoints

#### `POST /search` - Tìm kiếm sản phẩm

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kem dưỡng da cho da dầu",
    "topk": 10,
    "filters": {"brand": "innisfree"},
    "rerank": true
  }'
```

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | ✅ | - | Query tiếng Việt (1-500 chars) |
| topk | int | ❌ | 10 | Số kết quả (1-100) |
| filters | object | ❌ | null | Bộ lọc thuộc tính |
| rerank | bool | ❌ | true | Áp dụng multi-signal reranking |

**Response**:
```json
{
  "query": "kem dưỡng da cho da dầu",
  "results": [
    {
      "rank": 1,
      "product_id": 123,
      "product_name": "Innisfree Green Tea Seed Cream",
      "brand": "innisfree",
      "category": "kem dưỡng",
      "price": 450000,
      "avg_rating": 4.5,
      "num_sold": 1500,
      "semantic_score": 0.85,
      "final_score": 0.78
    }
  ],
  "count": 10,
  "method": "hybrid",
  "latency_ms": 45.23,
  "available_filters": null
}
```

#### `POST /search/similar` - Tìm sản phẩm tương tự

```bash
curl -X POST http://localhost:8000/search/similar \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 123,
    "topk": 10,
    "exclude_self": true
  }'
```

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| product_id | int | ✅ | - | ID sản phẩm gốc |
| topk | int | ❌ | 10 | Số kết quả (1-50) |
| exclude_self | bool | ❌ | true | Loại trừ sản phẩm gốc |

#### `POST /search/profile` - Tìm theo hồ sơ người dùng

```bash
curl -X POST http://localhost:8000/search/profile \
  -H "Content-Type: application/json" \
  -d '{
    "product_history": [101, 102, 103],
    "topk": 10,
    "exclude_history": true,
    "filters": {"category": "serum"}
  }'
```

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| product_history | list[int] | ✅ | - | Danh sách product_id đã xem |
| topk | int | ❌ | 10 | Số kết quả (1-100) |
| exclude_history | bool | ❌ | true | Loại trừ sản phẩm trong lịch sử |
| filters | object | ❌ | null | Bộ lọc thuộc tính |

#### `GET /search/filters` - Lấy bộ lọc có sẵn

```bash
curl http://localhost:8000/search/filters
```

**Response**:
```json
{
  "brands": ["innisfree", "the face shop", "some by mi", ...],
  "categories": ["kem dưỡng", "sữa rửa mặt", "serum", ...],
  "price_range": [15000, 2500000]
}
```

#### `GET /search/stats` - Thống kê search

```bash
curl http://localhost:8000/search/stats
```

**Response**:
```json
{
  "total_searches": 1250,
  "similar_searches": 320,
  "profile_searches": 85,
  "avg_latency_ms": 42.5,
  "errors": 3,
  "index": {
    "num_products": 2200,
    "num_brands": 150,
    "num_categories": 25
  },
  "encoder": {
    "cache_hit_rate": 0.72,
    "queries_encoded": 1500
  }
}
```

---

## Xử Lý Tiếng Việt

### Preprocessing Pipeline

```
Input: "srm cho dn ko gây mụn"
         │
         ▼
┌─────────────────────────────┐
│ 1. Lowercase & Strip        │
│    "srm cho dn ko gây mụn"  │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2. Expand Abbreviations     │
│    "sữa rửa mặt cho da nhờn │
│     không gây mụn"          │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3. Normalize Whitespace     │
│    Remove extra spaces      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 4. Remove Special Chars     │
│    Keep Vietnamese chars    │
└─────────────────────────────┘
         │
         ▼
Output: "sữa rửa mặt cho da nhờn không gây mụn"
```

### Thêm viết tắt mới

```python
from service.search.query_encoder import QueryEncoder

# Thêm abbreviations khi khởi tạo
encoder = QueryEncoder(
    abbreviations={
        'bb': 'kem nền bb cream',
        'cc': 'kem nền cc cream',
        'vc': 'vitamin c'
    }
)

# Hoặc update sau
encoder.abbreviations['newabbr'] = 'new full form'
```

### Unicode và Diacritics

Hệ thống giữ nguyên các ký tự tiếng Việt:
- Dấu: á, à, ả, ã, ạ, ă, â, ...
- Chữ đặc biệt: đ, ơ, ư, ...
- Unicode ranges: `\u00C0-\u024F`, `\u1E00-\u1EFF`

---

## Tối Ưu Hiệu Năng

### Mục tiêu latency

| Metric | Mục tiêu | Ghi chú |
|--------|----------|---------|
| P50 | <100ms | Median response |
| P95 | <300ms | 95% requests |
| P99 | <500ms | SLA target |

### Các kỹ thuật tối ưu

#### 1. LRU Cache cho Query Embeddings

```python
# Mặc định cache 1000 queries gần nhất
encoder = QueryEncoder(cache_size=1000)

# Xem cache stats
stats = encoder.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Cache size: {stats['cache']['size']}/{stats['cache']['capacity']}")
```

#### 2. Pre-normalized Embeddings

```python
# Vietnamese Embedding đã được normalize sẵn
# Cosine similarity = dot product (nhanh hơn)
similarity = embeddings_norm @ query_embedding
```

#### 3. FAISS cho Large Catalogs

```python
from service.search.search_index import SearchIndex

# Enable FAISS cho catalog lớn
index = SearchIndex(
    use_faiss=True,
    faiss_index_type="hnsw"  # Nhanh nhất
)
```

#### 4. Metadata Inverted Index

```python
# Filter được thực hiện trước search
# Giảm số candidates cần tính similarity

results = index.search_with_filter(
    query_embedding,
    filters={'brand': 'innisfree'}  # Chỉ search trong ~100 products
)
```

#### 5. Candidate Multiplier

```python
# Lấy 3x ứng viên, rerank, lấy top-K
# Tăng chất lượng kết quả

config = {
    'candidate_multiplier': 3  # 30 candidates → rerank → 10 results
}
```

### Monitoring Performance

```python
# Xem latency trung bình
stats = service.get_stats()
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")

# Xem latency từng request
results = service.search("kem dưỡng da", topk=10)
print(f"This request: {results.latency_ms:.2f}ms")

# Xem encoder stats
encoder_stats = service.query_encoder.get_stats()
print(f"Avg encoding time: {encoder_stats['avg_encoding_time_ms']:.2f}ms")
```

---

## Xử Lý Sự Cố

### Lỗi thường gặp

#### 1. "Model not loaded"

```
RuntimeError: Vietnamese Embedding model not found
```

**Nguyên nhân**: Model chưa được download.

**Giải pháp**:
```python
# Model sẽ tự động download khi encode lần đầu
# Hoặc download trước:
from transformers import AutoTokenizer, AutoModel

# Model: AITeamVN/Vietnamese_Embedding
tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding")
model = AutoModel.from_pretrained("AITeamVN/Vietnamese_Embedding")
```

#### 2. "Product embeddings not found"

```
FileNotFoundError: product_embeddings.pt not found
```

**Nguyên nhân**: File embeddings chưa được tạo.

**Giải pháp**:
```bash
# Tạo embeddings với Vietnamese Embedding model
python scripts/generate_bert_embeddings.py

# Hoặc kiểm tra đường dẫn
ls data/processed/content_based_embeddings/
```

#### 3. "No results found"

**Nguyên nhân**: 
- Query quá cụ thể
- Bộ lọc quá chặt
- Ngưỡng semantic score quá cao

**Giải pháp**:
```python
# Giảm ngưỡng min_semantic_score
service.config['min_semantic_score'] = 0.15  # Mặc định 0.25

# Bỏ bớt filters
results = service.search("query", filters=None)

# Tăng topk
results = service.search("query", topk=50)
```

#### 4. "Slow search latency (>500ms)"

**Nguyên nhân**:
- Vietnamese Embedding model chưa được cache
- FAISS chưa build
- Quá nhiều candidates

**Giải pháp**:
```python
# Warm up encoder (load Vietnamese Embedding model trước)
encoder = get_query_encoder()
encoder.encode("warm up query")

# Giảm candidate_multiplier
service.config['candidate_multiplier'] = 2  # Mặc định 3

# Enable FAISS
index = SearchIndex(use_faiss=True, faiss_index_type="hnsw")
```

#### 5. "Viết tắt không được mở rộng"

**Nguyên nhân**: Viết tắt không có trong dictionary.

**Giải pháp**:
```python
# Thêm viết tắt mới
encoder.abbreviations['myabbr'] = 'my full form'

# Hoặc update file query_encoder.py
VIETNAMESE_ABBREVIATIONS = {
    ...
    'myabbr': 'my full form',
}
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger('service.search').setLevel(logging.DEBUG)

# Xem preprocessing
encoder = get_query_encoder()
print(f"Original: 'srm cho dn'")
print(f"Processed: '{encoder.preprocess_query('srm cho dn')}'")

# Xem raw search results
results = service.search("kem dưỡng", topk=10, rerank=False)
for item in results.results:
    print(f"{item.product_id}: semantic={item.semantic_score:.3f}")
```

### Health Check

```python
def check_smart_search_health():
    """Kiểm tra sức khỏe hệ thống Smart Search."""
    
    issues = []
    
    # 1. Kiểm tra QueryEncoder
    try:
        from service.search.query_encoder import get_query_encoder
        encoder = get_query_encoder()
        test_emb = encoder.encode("test query")
        if test_emb is None or len(test_emb) == 0:
            issues.append("QueryEncoder: Empty embedding")
    except Exception as e:
        issues.append(f"QueryEncoder: {e}")
    
    # 2. Kiểm tra SearchIndex
    try:
        from service.search.search_index import get_search_index
        index = get_search_index()
        if not index.is_initialized:
            index.build_index()
        if index.num_products == 0:
            issues.append("SearchIndex: No products indexed")
    except Exception as e:
        issues.append(f"SearchIndex: {e}")
    
    # 3. Kiểm tra SmartSearchService
    try:
        from service.search import get_search_service
        service = get_search_service()
        results = service.search("test", topk=1)
        # Có thể không có kết quả nhưng không được lỗi
    except Exception as e:
        issues.append(f"SmartSearchService: {e}")
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues
    }

# Chạy health check
health = check_smart_search_health()
print(f"Healthy: {health['healthy']}")
for issue in health['issues']:
    print(f"  ⚠ {issue}")
```

---

## Tài Liệu Liên Quan

- [Task 09: Smart Search Spec](../tasks/09_smart_search.md)
- [Hybrid Reranking Guide](./HYBRID_RERANKING_GUIDE.md)
- [PhoBERT Loader (Task 08)](../tasks/08_hybrid_reranking.md)
- [API Documentation](../service/api.py)

---

## Changelog

| Phiên bản | Ngày | Thay đổi |
|-----------|------|----------|
| 1.0.0 | 2025-11-30 | Phiên bản đầu tiên |

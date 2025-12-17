# Hướng Dẫn Hybrid Reranking

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Kiến Trúc Tổng Quan](#kiến-trúc-tổng-quan)
3. [Cài Đặt & Cấu Hình](#cài-đặt--cấu-hình)
4. [Các Thành Phần Chính](#các-thành-phần-chính)
5. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
6. [API Reference](#api-reference)
7. [Đánh Giá & Metrics](#đánh-giá--metrics)
8. [Tối Ưu Hiệu Năng](#tối-ưu-hiệu-năng)
9. [Xử Lý Sự Cố](#xử-lý-sự-cố)

---

## Giới Thiệu

### Hybrid Reranking là gì?

Hybrid Reranking là kỹ thuật kết hợp nhiều nguồn tín hiệu (signals) để xếp hạng lại (rerank) danh sách gợi ý ban đầu từ Collaborative Filtering. Mục tiêu là tạo ra gợi ý **đa dạng hơn**, **cá nhân hóa hơn**, và **xử lý cold-start tốt hơn**.

### Tại sao cần Hybrid Reranking?

Với dữ liệu mỹ phẩm Việt Nam của chúng ta:

| Thách thức | Con số | Giải pháp |
|------------|--------|-----------|
| Dữ liệu thưa | ~1.23 tương tác/người dùng | Kết hợp content signal |
| Người dùng cold-start | 91.4% (274K users) | Fallback content + popularity |
| Rating lệch | 95% rating 5 sao | Sử dụng thêm comment quality |

### Các tín hiệu được sử dụng

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID RERANKING                         │
├─────────────────────────────────────────────────────────────┤
│  CF Score (30%)      │  Điểm từ ALS/BPR (U @ V.T)          │
│  Content Score (40%) │  Cosine similarity PhoBERT          │
│  Popularity (20%)    │  Số lượt bán (num_sold_time)        │
│  Quality (10%)       │  Rating trung bình (avg_star)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Kiến Trúc Tổng Quan

### Luồng Xử Lý Gợi Ý

```
                    ┌──────────────────┐
                    │   Request API    │
                    │  (user_id, topk) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  CFRecommender   │
                    │   .recommend()   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  is_trainable?    │         │  is_cold_start?   │
    │   (≥2 tương tác)  │         │   (<2 tương tác)  │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │   CF Scoring      │         │ FallbackRecommender│
    │   U[idx] @ V.T    │         │ (content+popularity)│
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  HybridReranker   │         │  rerank_cold_start │
    │    .rerank()      │         │     (weights)      │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │   Top-K Results  │
                    │  + diversity     │
                    └──────────────────┘
```

### Phân Bổ Lưu Lượng

```
Tổng người dùng: ~300,000
    │
    ├── Trainable users (8.6%, ~26K)
    │   └── CF + Hybrid Reranking
    │       Weights: cf=0.30, content=0.40, popularity=0.20, quality=0.10
    │
    └── Cold-start users (91.4%, ~274K)
        └── Fallback + Rerank Cold-Start
            Weights: content=0.60, popularity=0.30, quality=0.10
```

---

## Cài Đặt & Cấu Hình

### Yêu Cầu Hệ Thống

```bash
# Python packages
torch>=1.9.0
transformers>=4.0.0
numpy>=1.20.0
scipy>=1.7.0
pyyaml>=5.4.0
```

### Cấu Hình Chính

File: `config/serving_config.yaml`

```yaml
# ═══════════════════════════════════════════════════════════
# CẤU HÌNH HYBRID RERANKING
# ═══════════════════════════════════════════════════════════

reranking:
  enabled: true
  
  # Trọng số cho người dùng có dữ liệu CF
  weights_trainable:
    cf: 0.30              # Tín hiệu cộng tác
    content: 0.40         # Độ tương đồng nội dung (PhoBERT)
    popularity: 0.20      # Sản phẩm trending
    quality: 0.10         # Sản phẩm đánh giá cao
  
  # Trọng số cho người dùng mới
  weights_cold_start:
    content: 0.60         # Nội dung là tín hiệu chính
    popularity: 0.30      # Bằng chứng xã hội
    quality: 0.10         # Thưởng chất lượng
  
  # Cài đặt đa dạng
  diversity:
    enabled: true
    penalty: 0.1          # Mức phạt (0.0 - 1.0)
    threshold: 0.85       # Ngưỡng similarity để phạt
  
  # Mở rộng ứng viên
  candidate_multiplier: 5  # Tạo 5x ứng viên

# ═══════════════════════════════════════════════════════════
# CẤU HÌNH PHOBERT
# ═══════════════════════════════════════════════════════════

phobert:
  embeddings_path: "data/processed/content_based_embeddings/product_embeddings.pt"
  precompute_similarity_matrix: true
  max_items_for_precompute: 3000
  user_profile_strategy: "weighted_mean"

# ═══════════════════════════════════════════════════════════
# CẤU HÌNH FALLBACK
# ═══════════════════════════════════════════════════════════

fallback:
  default_strategy: "hybrid"
  content_weight: 0.7
  popularity_weight: 0.3
  enable_cache: true
```

### Các File Dữ Liệu Cần Thiết

```
data/
├── processed/
│   ├── content_based_embeddings/
│   │   └── product_embeddings.pt    # PhoBERT embeddings (~2.2K sản phẩm)
│   ├── data_stats.json              # Phạm vi chuẩn hóa
│   └── user_item_mappings.json      # ID mappings
│
└── published_data/
    └── content_based_embeddings/
        ├── product_embeddings.pt    # Backup embeddings
        └── phobert_description_feature.pt
```

---

## Các Thành Phần Chính

### 1. HybridReranker

**File**: `service/recommender/rerank.py`

Bộ reranker chính kết hợp nhiều tín hiệu.

```python
from service.recommender.rerank import HybridReranker, RerankerConfig

# Khởi tạo với cấu hình mặc định
config = RerankerConfig()
reranker = HybridReranker(
    phobert_loader=phobert_loader,
    item_metadata=item_metadata,
    config=config
)

# Rerank gợi ý CF
result = reranker.rerank(
    cf_recommendations=cf_recs,
    user_id=12345,
    user_history=[101, 102, 103],
    topk=10
)

print(f"Đa dạng: {result.diversity_score:.3f}")
print(f"Latency: {result.latency_ms:.1f}ms")
```

#### Các tham số quan trọng:

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `weights_trainable` | Trọng số cho user có CF | cf=0.3, content=0.4, pop=0.2, quality=0.1 |
| `weights_cold_start` | Trọng số cho user mới | content=0.6, pop=0.3, quality=0.1 |
| `diversity_penalty` | Mức phạt đa dạng | 0.1 |
| `diversity_threshold` | Ngưỡng similarity | 0.85 |
| `candidate_multiplier` | Hệ số mở rộng | 5 |

### 2. PhoBERTEmbeddingLoader

**File**: `service/recommender/phobert_loader.py`

Singleton loader cho PhoBERT embeddings.

```python
from service.recommender.phobert_loader import get_phobert_loader

# Lấy instance (singleton - thread-safe)
loader = get_phobert_loader()

# Lấy embedding của sản phẩm
embedding = loader.get_embedding(product_id=123)
print(f"Kích thước: {embedding.shape}")  # (768,)

# Tính profile người dùng từ lịch sử
user_profile = loader.compute_user_profile(
    user_history_items=[101, 102, 103],
    strategy='weighted_mean'
)

# Tìm sản phẩm tương tự
similar_items = loader.find_similar_items(
    product_id=123,
    topk=10,
    exclude_self=True
)
for pid, score in similar_items:
    print(f"  Sản phẩm {pid}: similarity={score:.3f}")
```

#### Các chiến lược tạo user profile:

| Chiến lược | Mô tả | Khi nào dùng |
|------------|-------|--------------|
| `mean` | Trung bình cộng | Mọi item đều quan trọng như nhau |
| `weighted_mean` | Trung bình có trọng số | Có rating/recency weights |
| `max` | Max pooling | Muốn highlight đặc điểm nổi bật |

### 3. FallbackRecommender

**File**: `service/recommender/fallback.py`

Xử lý gợi ý cho người dùng cold-start (91.4% traffic).

```python
from service.recommender.fallback import FallbackRecommender

fallback = FallbackRecommender(
    model_loader=model_loader,
    phobert_loader=phobert_loader
)

# Fallback dựa trên popularity
recs = fallback.popularity_fallback(topk=10)

# Fallback dựa trên content similarity
recs = fallback.item_similarity_fallback(
    user_history=[101, 102],
    topk=10
)

# Fallback hybrid (khuyến nghị)
recs = fallback.hybrid_fallback(
    user_history=[101, 102],
    topk=10,
    content_weight=0.7,
    popularity_weight=0.3
)
```

#### So sánh các chiến lược fallback:

| Chiến lược | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| `popularity` | Nhanh, an toàn | Không cá nhân hóa |
| `item_similarity` | Cá nhân hóa | Cần lịch sử user |
| `hybrid` | Cân bằng | Phức tạp hơn |

### 4. CFRecommender

**File**: `service/recommender/recommender.py`

Engine gợi ý chính tích hợp tất cả thành phần.

```python
from service.recommender import CFRecommender

recommender = CFRecommender()

# Gợi ý cơ bản
result = recommender.recommend(user_id=12345, topk=10)

# Với bộ lọc thuộc tính
result = recommender.recommend(
    user_id=12345,
    topk=10,
    filter_params={'brand': 'Innisfree', 'skin_type': 'oily'}
)

# Kiểm tra kết quả
print(f"Là fallback: {result.is_fallback}")
print(f"Model: {result.model_id}")
print(f"Số gợi ý: {len(result.recommendations)}")
```

---

## Hướng Dẫn Sử Dụng

### Trường hợp 1: Gợi ý cho người dùng thường

```python
from service.recommender import CFRecommender

recommender = CFRecommender()

# Người dùng có ≥2 tương tác
result = recommender.recommend(user_id=12345, topk=10)

# Kiểm tra loại gợi ý
if not result.is_fallback:
    print("✓ Sử dụng CF + Hybrid Reranking")
    print(f"  Model: {result.model_id}")
    
# Xem chi tiết từng gợi ý
for rec in result.recommendations:
    print(f"\nSản phẩm {rec['product_id']}:")
    print(f"  Điểm cuối: {rec['final_score']:.3f}")
    print(f"  Tín hiệu:")
    print(f"    - CF: {rec['signals']['cf']:.3f}")
    print(f"    - Content: {rec['signals']['content']:.3f}")
    print(f"    - Popularity: {rec['signals']['popularity']:.3f}")
    print(f"    - Quality: {rec['signals']['quality']:.3f}")
```

### Trường hợp 2: Gợi ý cho người dùng mới

```python
# Người dùng mới (<2 tương tác)
result = recommender.recommend(user_id=999999, topk=10)

if result.is_fallback:
    print("✓ Sử dụng Fallback (content + popularity)")
    print(f"  Phương thức: {result.fallback_method}")

# Gợi ý vẫn có tín hiệu (nhưng cf=0)
for rec in result.recommendations:
    print(f"Sản phẩm {rec['product_id']}: {rec['score']:.3f}")
```

### Trường hợp 3: Điều chỉnh trọng số động

```python
# Tăng trọng số content, giảm CF
recommender.update_rerank_weights(
    weights_trainable={
        'cf': 0.20,        # Giảm từ 0.30
        'content': 0.50,   # Tăng từ 0.40
        'popularity': 0.20,
        'quality': 0.10
    }
)

# Tắt reranking hoàn toàn (chỉ dùng CF thuần)
recommender.set_reranking(enabled=False)

# Bật lại
recommender.set_reranking(enabled=True)
```

### Trường hợp 4: Lọc theo thuộc tính

```python
# Chỉ gợi ý sản phẩm của brand cụ thể
result = recommender.recommend(
    user_id=12345,
    topk=10,
    filter_params={'brand': 'Innisfree'}
)

# Lọc theo loại da
result = recommender.recommend(
    user_id=12345,
    topk=10,
    filter_params={'skin_type': ['oily', 'combination']}
)

# Kết hợp nhiều bộ lọc
result = recommender.recommend(
    user_id=12345,
    topk=10,
    filter_params={
        'brand': 'The Face Shop',
        'skin_type': 'sensitive',
        'price_range': (100000, 500000)
    }
)
```

### Trường hợp 5: Batch recommendation

```python
# Gợi ý cho nhiều người dùng cùng lúc
user_ids = [12345, 12346, 12347, 12348]

results = recommender.batch_recommend(
    user_ids=user_ids,
    topk=10
)

for user_id, result in zip(user_ids, results):
    print(f"\nUser {user_id}:")
    print(f"  Fallback: {result.is_fallback}")
    print(f"  Top gợi ý: {[r['product_id'] for r in result.recommendations[:3]]}")
```

---

## API Reference

### HybridReranker

```python
class HybridReranker:
    """Bộ reranker hybrid kết hợp nhiều tín hiệu."""
    
    def __init__(
        self,
        phobert_loader: PhoBERTEmbeddingLoader,
        item_metadata: Dict[int, Dict],
        config: Optional[RerankerConfig] = None
    ):
        """
        Khởi tạo HybridReranker.
        
        Args:
            phobert_loader: Loader cho PhoBERT embeddings
            item_metadata: Dict product_id -> metadata
            config: Cấu hình reranker (None = mặc định)
        """
    
    def rerank(
        self,
        cf_recommendations: List[Dict[str, Any]],
        user_id: Optional[int] = None,
        user_history: Optional[List[int]] = None,
        topk: Optional[int] = None,
        is_cold_start: bool = False
    ) -> RerankedResult:
        """
        Rerank danh sách gợi ý CF.
        
        Args:
            cf_recommendations: Gợi ý từ CF model
            user_id: ID người dùng (để log)
            user_history: Lịch sử tương tác
            topk: Số kết quả trả về
            is_cold_start: True nếu là user mới
        
        Returns:
            RerankedResult với gợi ý đã rerank
        """
    
    def update_weights(
        self,
        weights_trainable: Optional[Dict[str, float]] = None,
        weights_cold_start: Optional[Dict[str, float]] = None
    ) -> None:
        """Cập nhật trọng số động."""
```

### PhoBERTEmbeddingLoader

```python
class PhoBERTEmbeddingLoader:
    """Singleton loader cho PhoBERT embeddings."""
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """Lấy embedding của sản phẩm."""
    
    def compute_user_profile(
        self,
        user_history_items: List[int],
        weights: Optional[List[float]] = None,
        strategy: str = 'weighted_mean'
    ) -> Optional[np.ndarray]:
        """Tính profile người dùng từ lịch sử."""
    
    def find_similar_items(
        self,
        product_id: int,
        topk: int = 10,
        exclude_self: bool = True,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Tìm sản phẩm tương tự."""
    
    def compute_similarity(
        self,
        product_id_1: int,
        product_id_2: int
    ) -> float:
        """Tính similarity giữa 2 sản phẩm."""

# Hàm helper để lấy singleton
def get_phobert_loader() -> PhoBERTEmbeddingLoader:
    """Lấy instance singleton của PhoBERTEmbeddingLoader."""
```

### FallbackRecommender

```python
class FallbackRecommender:
    """Các chiến lược fallback cho cold-start users."""
    
    def popularity_fallback(
        self,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback dựa trên popularity."""
    
    def item_similarity_fallback(
        self,
        user_history: List[int],
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback dựa trên content similarity."""
    
    def hybrid_fallback(
        self,
        user_history: List[int],
        topk: int = 10,
        content_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback hybrid (content + popularity)."""
```

### CFRecommender

```python
class CFRecommender:
    """Engine gợi ý chính."""
    
    def recommend(
        self,
        user_id: int,
        topk: int = 10,
        exclude_seen: bool = True,
        filter_params: Optional[Dict[str, Any]] = None,
        normalize_scores: bool = False,
        rerank: Optional[bool] = None
    ) -> RecommendationResult:
        """Tạo gợi ý cho người dùng."""
    
    def batch_recommend(
        self,
        user_ids: List[int],
        topk: int = 10,
        **kwargs
    ) -> List[RecommendationResult]:
        """Tạo gợi ý cho nhiều người dùng."""
    
    def set_reranking(self, enabled: bool) -> None:
        """Bật/tắt reranking."""
    
    def update_rerank_weights(
        self,
        weights_trainable: Optional[Dict[str, float]] = None,
        weights_cold_start: Optional[Dict[str, float]] = None
    ) -> None:
        """Cập nhật trọng số reranking."""
```

---

## Đánh Giá & Metrics

### Các Metric Hybrid

**File**: `recsys/cf/evaluation/hybrid_metrics.py`

#### 1. DiversityMetric - Đo đa dạng

```python
from recsys.cf.evaluation.hybrid_metrics import DiversityMetric, compute_diversity_bert

# Sử dụng class
diversity_metric = DiversityMetric()
score = diversity_metric.compute(
    recommendations=rec_ids,
    item_embeddings=embeddings
)

# Hoặc hàm tiện ích
diversity = compute_diversity_bert(
    recommendations=rec_ids,
    bert_embeddings=phobert.embeddings_norm,
    item_to_idx=phobert.product_id_to_idx
)

print(f"Diversity: {diversity:.3f}")
# Giá trị 0.3: Sản phẩm khá giống nhau
# Giá trị 0.6: Sản phẩm khá đa dạng
```

#### 2. NoveltyMetric - Đo độ mới lạ

```python
from recsys.cf.evaluation.hybrid_metrics import NoveltyMetric

novelty_metric = NoveltyMetric()
score = novelty_metric.compute(
    recommendations=rec_ids,
    item_popularity=item_popularity,
    num_users=300000
)

print(f"Novelty: {score:.3f}")
# Cao = gợi ý sản phẩm long-tail (ít phổ biến)
# Thấp = gợi ý sản phẩm phổ biến
```

#### 3. SemanticAlignmentMetric - Đo độ phù hợp

```python
from recsys.cf.evaluation.hybrid_metrics import SemanticAlignmentMetric

alignment_metric = SemanticAlignmentMetric()
score = alignment_metric.compute(
    user_profile_emb=user_profile,
    recommendations=rec_ids,
    item_embeddings=embeddings
)

print(f"Semantic Alignment: {score:.3f}")
# Cao = gợi ý CF khớp với sở thích nội dung
```

#### 4. ColdStartCoverageMetric - Đo độ phủ cold-start

```python
from recsys.cf.evaluation.hybrid_metrics import ColdStartCoverageMetric

coverage_metric = ColdStartCoverageMetric()
score = coverage_metric.compute(
    all_recommendations=all_recs,
    item_counts=item_counts,
    cold_threshold=5
)

print(f"Cold-Start Coverage: {score:.3f}")
# Cao = hệ thống expose được nhiều sản phẩm mới
```

### Đánh Giá Toàn Diện

```python
from recsys.cf.evaluation.hybrid_metrics import HybridMetricCollection

# Tạo collection với các K values
collection = HybridMetricCollection(k_values=[10, 20, 50])

# Đánh giá tất cả metrics
results = collection.evaluate_all(
    all_recommendations=all_recs,      # Dict user_id -> [product_ids]
    item_embeddings=embeddings,         # PhoBERT embeddings
    item_popularity=item_popularity,    # Dict product_id -> popularity
    item_counts=item_counts,            # Dict product_id -> count
    num_users=300000
)

# In kết quả
for metric_name, values in results.items():
    print(f"\n{metric_name}:")
    for k, value in values.items():
        print(f"  @{k}: {value:.4f}")
```

**Kết quả mẫu**:
```
Diversity:
  @10: 0.5234
  @20: 0.5678
  @50: 0.6012

Novelty:
  @10: 4.2345
  @20: 4.5678
  @50: 5.1234

SemanticAlignment:
  @10: 0.7234
  @20: 0.6987
  @50: 0.6543

ColdStartCoverage:
  @10: 0.1234
  @20: 0.2345
  @50: 0.4567
```

---

## Tối Ưu Hiệu Năng

### Mục Tiêu Latency

| Metric | Mục tiêu | Ghi chú |
|--------|----------|---------|
| P50 | <50ms | Median response |
| P90 | <100ms | 90% requests |
| P95 | <150ms | 95% requests |
| P99 | <200ms | SLA target |

### Các Kỹ Thuật Tối Ưu

#### 1. Pre-compute Similarity Matrix

```python
# Cấu hình trong serving_config.yaml
phobert:
  precompute_similarity_matrix: true
  max_items_for_precompute: 3000
```

Với catalog <3000 sản phẩm, pre-compute toàn bộ item-item similarity matrix để tránh tính toán runtime.

#### 2. Pre-normalize Embeddings

```python
# PhoBERTEmbeddingLoader tự động normalize
# Cosine similarity = dot product (vì đã normalize)
similarity = embeddings_norm @ query_emb  # Nhanh hơn
```

#### 3. Cache User History

```python
# Cấu hình trong serving_config.yaml
fallback:
  enable_cache: true
```

Cache lịch sử người dùng để tránh query database mỗi request.

#### 4. Batch Processing

```python
# Xử lý nhiều users cùng lúc
results = recommender.batch_recommend(
    user_ids=[1, 2, 3, 4, 5],
    topk=10
)
```

#### 5. Candidate Multiplier Hợp Lý

```python
# Đừng để quá cao (tốn compute)
# Đừng để quá thấp (mất đa dạng)
reranking:
  candidate_multiplier: 5  # 5x là hợp lý
```

### Monitoring Performance

```python
import time

# Đo latency
start = time.perf_counter()
result = recommender.recommend(user_id=12345, topk=10)
latency = (time.perf_counter() - start) * 1000

print(f"Latency: {latency:.1f}ms")

# Kiểm tra từ kết quả rerank
if hasattr(result, 'rerank_latency_ms'):
    print(f"Rerank latency: {result.rerank_latency_ms:.1f}ms")
```

---

## Xử Lý Sự Cố

### Lỗi thường gặp

#### 1. "PhoBERT embeddings not found"

```
FileNotFoundError: product_embeddings.pt not found
```

**Nguyên nhân**: File embeddings chưa được tạo hoặc sai đường dẫn.

**Giải pháp**:
```bash
# Kiểm tra file tồn tại
ls data/processed/content_based_embeddings/

# Nếu chưa có, tạo embeddings
python scripts/generate_bert_embeddings.py
```

#### 2. "User not in mappings"

```
KeyError: User 999999 not in user_to_idx mappings
```

**Nguyên nhân**: User mới (cold-start) không có trong CF model.

**Giải pháp**: Đây là hành vi đúng - hệ thống sẽ tự động dùng fallback.

```python
# Kiểm tra trong code
if result.is_fallback:
    # User cold-start, đã dùng fallback
    pass
```

#### 3. "Memory error loading embeddings"

```
MemoryError: Unable to allocate array
```

**Nguyên nhân**: Embeddings quá lớn cho RAM.

**Giải pháp**:
```python
# Tắt pre-compute similarity matrix
phobert:
  precompute_similarity_matrix: false

# Hoặc giảm max items
phobert:
  max_items_for_precompute: 1000
```

#### 4. "Diversity score = 0"

**Nguyên nhân**: Tất cả sản phẩm trong gợi ý giống hệt nhau.

**Giải pháp**:
```yaml
# Tăng diversity penalty
diversity:
  penalty: 0.2  # Tăng từ 0.1
  threshold: 0.80  # Giảm từ 0.85
```

#### 5. "Slow reranking latency (>200ms)"

**Nguyên nhân**: Quá nhiều ứng viên hoặc tính toán không hiệu quả.

**Giải pháp**:
```yaml
# Giảm candidate multiplier
reranking:
  candidate_multiplier: 3  # Giảm từ 5

# Bật pre-compute
phobert:
  precompute_similarity_matrix: true
```

### Debug Mode

```python
import logging

# Bật debug logging
logging.getLogger('service.recommender').setLevel(logging.DEBUG)

# Xem chi tiết tín hiệu
result = recommender.recommend(user_id=12345, topk=10)

for rec in result.recommendations:
    signals = rec.get('signals', {})
    print(f"Product {rec['product_id']}:")
    print(f"  CF: {signals.get('cf', 'N/A')}")
    print(f"  Content: {signals.get('content', 'N/A')}")
    print(f"  Popularity: {signals.get('popularity', 'N/A')}")
    print(f"  Quality: {signals.get('quality', 'N/A')}")
```

### Health Check

```python
def check_hybrid_reranking_health():
    """Kiểm tra sức khỏe hệ thống hybrid reranking."""
    
    issues = []
    
    # 1. Kiểm tra PhoBERT loader
    try:
        loader = get_phobert_loader()
        if loader.num_products == 0:
            issues.append("PhoBERT: No products loaded")
    except Exception as e:
        issues.append(f"PhoBERT: {e}")
    
    # 2. Kiểm tra CFRecommender
    try:
        recommender = CFRecommender()
        result = recommender.recommend(user_id=1, topk=1)
        if len(result.recommendations) == 0:
            issues.append("CFRecommender: Empty recommendations")
    except Exception as e:
        issues.append(f"CFRecommender: {e}")
    
    # 3. Kiểm tra config
    import yaml
    try:
        with open('config/serving_config.yaml') as f:
            config = yaml.safe_load(f)
        if not config.get('reranking', {}).get('enabled'):
            issues.append("Config: Reranking disabled")
    except Exception as e:
        issues.append(f"Config: {e}")
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues
    }
```

---

## Tài Liệu Liên Quan

- [Task 08: Hybrid Reranking Spec](../tasks/08_hybrid_reranking.md)
- [Training Guide](./TRAINING_GUIDE.md)
- [Data Processing Guide](./DATA_PROCESSING_GUIDE.md)
- [Automation Guide](./AUTOMATION_GUIDE.md)

---

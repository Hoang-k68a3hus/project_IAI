# Task 08: Hybrid Reranking & Tích Hợp ✅ ĐÃ HOÀN THÀNH

## Mục Tiêu

Kết hợp Collaborative Filtering (ALS/BPR) với PhoBERT embeddings và thuộc tính sản phẩm để tạo hệ thống gợi ý hybrid. Mục tiêu là tăng **đa dạng (diversity)**, **cá nhân hóa (personalization)**, và xử lý **người dùng mới (cold-start)** tốt hơn bằng cách kết hợp nhiều tín hiệu (signals).

---

## ✅ Trạng Thái Triển Khai (Tháng 1/2025)

**Các thành phần đã hoàn thành**:

| Thành phần | File | Trạng thái |
|------------|------|------------|
| HybridReranker | `service/recommender/rerank.py` | ✅ Hoàn thành |
| PhoBERTEmbeddingLoader | `service/recommender/phobert_loader.py` | ✅ Hoàn thành |
| FallbackRecommender | `service/recommender/fallback.py` | ✅ Hoàn thành |
| Tích hợp CFRecommender | `service/recommender/recommender.py` | ✅ Hoàn thành |
| Hybrid Metrics | `recsys/cf/evaluation/hybrid_metrics.py` | ✅ Hoàn thành |
| Cấu hình Serving | `config/serving_config.yaml` | ✅ Hoàn thành |

---

## 📊 Phụ Thuộc Dữ Liệu

**Các file đầu vào**:
- **BERT Embeddings**: `data/processed/content_based_embeddings/product_embeddings.pt`
- **Đường dẫn dự phòng**: 
  - `data/published_data/content_based_embeddings/product_embeddings.pt`
  - `data/published_data/content_based_embeddings/phobert_description_feature.pt`
- **Metadata sản phẩm**: Từ `CFModelLoader` (popularity_score, avg_star, brand, v.v.)
- **Thống kê toàn cục**: `data/processed/data_stats.json` (phạm vi chuẩn hóa)

---

## 🎯 Chiến Lược Hybrid Ưu Tiên Nội Dung

### Bối Cảnh
- **Dữ liệu thưa (Data sparsity)**: ~1.23 tương tác/người dùng → CF có tín hiệu cộng tác hạn chế
- **Người dùng trainable**: ~26,000 (8.6%) có ≥2 tương tác
- **Người dùng cold-start**: ~274,000 (91.4%) → sử dụng fallback dựa trên nội dung

### Phân Bổ Trọng Số Đã Triển Khai

```yaml
# Người dùng Trainable (≥2 tương tác, ~8.6% lưu lượng)
weights_trainable:
  cf: 0.30         # PHỤ - Tín hiệu cộng tác
  content: 0.40    # CHÍNH - Độ tương đồng ngữ nghĩa PhoBERT  
  popularity: 0.20 # BỔ SUNG - Sản phẩm trending
  quality: 0.10    # THƯỞNG - Sản phẩm đánh giá cao

# Người dùng Cold-Start (<2 tương tác, ~91.4% lưu lượng)
weights_cold_start:
  content: 0.60    # CHỦ ĐẠO - Tín hiệu duy nhất đáng tin cậy
  popularity: 0.30 # Bằng chứng xã hội
  quality: 0.10    # Thưởng
```

**Lý do**:
- **Người dùng trainable**: Content (40%) đáng tin cậy nhất dù có ≥2 tương tác; CF (30%) vẫn có giá trị với BERT init + regularization cao (λ=0.1)
- **Người dùng cold-start** (đa số): Content (60%) chủ đạo; CF không sử dụng được
- Popularity (20-30%): Bằng chứng xã hội, xử lý sản phẩm trending
- Quality (10%): Thưởng cho sản phẩm được đánh giá cao

---

## Tổng Quan Kiến Trúc

```
Yêu cầu người dùng
    ↓
CFRecommender.recommend()
    ↓
├─ is_trainable_user? ──→ True: Tính điểm CF (U[u_idx] @ V.T)
│                        False: FallbackRecommender (content + popularity)
    ↓
Top-K Ứng viên (K × candidate_multiplier = 5x)
    ↓
HybridReranker.rerank() / rerank_cold_start()
    ↓
├─ _compute_signals(): CF, Content, Popularity, Quality
├─ _normalize_signals(): Chuẩn hóa toàn cục (KHÔNG phải cục bộ theo request)
├─ _combine_scores(): Kết hợp trọng số
└─ _apply_diversity_penalty(): Phạt đa dạng kiểu MMR dựa trên BERT
    ↓
Top-K cuối cùng với điểm đa dạng
```

**Các điểm tích hợp chính**:
- **PhoBERTEmbeddingLoader** (`service/recommender/phobert_loader.py`): Singleton với embeddings đã chuẩn hóa sẵn
- **HybridReranker** (`service/recommender/rerank.py`): Kết hợp tín hiệu có trọng số với đa dạng
- **FallbackRecommender** (`service/recommender/fallback.py`): Xử lý 91.4% lưu lượng cold-start
- **Hybrid Metrics** (`recsys/cf/evaluation/hybrid_metrics.py`): DiversityMetric, NoveltyMetric, SemanticAlignmentMetric

---

## Thành Phần 1: RerankerConfig & RerankedResult ✅

### Module: `service/recommender/rerank.py`

**Mô tả**: Các dataclass cấu hình cho HybridReranker.

```python
@dataclass
class RerankerConfig:
    """
    Cấu hình cho HybridReranker.
    
    Attributes:
        weights_trainable: Trọng số cho người dùng trainable (≥2 tương tác)
        weights_cold_start: Trọng số cho người dùng cold-start
        diversity_enabled: Bật/tắt phạt đa dạng
        diversity_penalty: Mức phạt cho sản phẩm tương tự (0.0-1.0)
        diversity_threshold: Ngưỡng BERT similarity để áp dụng phạt
        user_profile_strategy: Chiến lược tạo profile (mean, weighted_mean, recency)
        candidate_multiplier: Hệ số nhân ứng viên cho reranking
        
        # Phạm vi chuẩn hóa (toàn cục, không phải cục bộ theo request)
        cf_score_min/max: Phạm vi điểm CF (thường [0, 1.5] sau U@V.T)
        content_score_min/max: Phạm vi cosine similarity [-1, 1]
        quality_min/max: Phạm vi rating [1, 5]
        popularity_p01/p99: Phân vị từ data_stats.json
    """
    
    weights_trainable: Dict[str, float] = field(default_factory=lambda: {
        'cf': 0.30, 'content': 0.40, 'popularity': 0.20, 'quality': 0.10
    })
    weights_cold_start: Dict[str, float] = field(default_factory=lambda: {
        'content': 0.60, 'popularity': 0.30, 'quality': 0.10
    })
    diversity_enabled: bool = True
    diversity_penalty: float = 0.1
    diversity_threshold: float = 0.85
    user_profile_strategy: str = 'weighted_mean'
    candidate_multiplier: int = 5


@dataclass
class RerankedResult:
    """
    Kết quả của thao tác reranking.
    
    Attributes:
        recommendations: Danh sách gợi ý đã rerank
        latency_ms: Thời gian xử lý (ms)
        diversity_score: Điểm đa dạng [0, 1]
        weights_used: Trọng số đã sử dụng
        num_candidates: Số ứng viên đầu vào
        num_output: Số kết quả đầu ra
    """
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    diversity_score: float
    weights_used: Dict[str, float]
    num_candidates: int
    num_output: int
```

---

## Thành Phần 2: HybridReranker ✅

### Module: `service/recommender/rerank.py`

**Mô tả**: Bộ reranker hybrid kết hợp các tín hiệu CF, content, popularity, quality.

#### Tính năng chính:
1. **Kết hợp trọng số nhiều tín hiệu**: CF, content similarity, popularity, quality
2. **Chuẩn hóa toàn cục**: Sử dụng phạm vi cố định thay vì min-max cục bộ
3. **Phạt đa dạng**: Giảm điểm sản phẩm quá giống nhau dựa trên BERT similarity
4. **Xử lý cold-start**: Trọng số riêng cho người dùng mới
5. **Cập nhật động**: Thay đổi trọng số mà không cần restart

```python
class HybridReranker:
    """
    Bộ reranker hybrid kết hợp tín hiệu CF, content, popularity, quality.
    
    Sử dụng PhoBERTEmbeddingLoader để tính độ tương đồng nội dung
    và áp dụng chuẩn hóa toàn cục để đảm bảo điểm số nhất quán.
    
    Ví dụ:
        >>> reranker = HybridReranker(phobert_loader, item_metadata)
        >>> result = reranker.rerank(cf_recs, user_id, user_history)
    """
```

#### Phương thức quan trọng:

##### `_normalize_global()` - Chuẩn hóa toàn cục

```python
def _normalize_global(self, values: Dict[int, float], signal_type: str) -> Dict[int, float]:
    """
    Chuẩn hóa giá trị sử dụng phạm vi toàn cục (không phải cục bộ theo request).
    
    QUAN TRỌNG: Đảm bảo chuẩn hóa nhất quán giữa các request khác nhau.
    
    Vấn đề với chuẩn hóa cục bộ:
    - User A: [0.91, ..., 0.99] → chuẩn hóa thành [0.0, ..., 1.0]
    - User B: [0.11, ..., 0.19] → chuẩn hóa thành [0.0, ..., 1.0]
    - Cả hai trông như nhau dù điểm User A cao hơn nhiều!
    
    Giải pháp với chuẩn hóa toàn cục:
    - User A: [0.91/1.5, ..., 0.99/1.5] = [0.61, ..., 0.66]
    - User B: [0.11/1.5, ..., 0.19/1.5] = [0.07, ..., 0.13]
    - Giờ phản ánh đúng sự khác biệt chất lượng!
    
    Args:
        values: Dict product_id -> giá trị thô
        signal_type: 'cf', 'content', 'popularity', 'quality'
    
    Returns:
        Dict product_id -> giá trị chuẩn hóa trong [0, 1]
    """
```

##### `_compute_signals()` - Tính toán tín hiệu

```python
def _compute_signals(
    self,
    candidate_ids: List[int],
    cf_scores: Dict[int, float],
    user_history: Optional[List[int]] = None
) -> Dict[str, Dict[int, float]]:
    """
    Tính toán tất cả tín hiệu cho ứng viên.
    
    Các tín hiệu được tính:
    1. CF: Điểm từ U @ V.T (đã có sẵn)
    2. Content: Cosine similarity giữa user profile và item embedding
    3. Popularity: Điểm phổ biến từ metadata
    4. Quality: Điểm chất lượng (avg_star) từ metadata
    
    Args:
        candidate_ids: Danh sách ID sản phẩm ứng viên
        cf_scores: Dict product_id -> điểm CF
        user_history: Lịch sử tương tác của người dùng
    
    Returns:
        Dict tên_tín_hiệu -> {product_id: điểm}
    """
```

##### `_apply_diversity_penalty()` - Áp dụng phạt đa dạng

```python
def _apply_diversity_penalty(
    self,
    scores: Dict[int, float],
    candidate_ids: List[int]
) -> Tuple[Dict[int, float], float]:
    """
    Áp dụng phạt đa dạng để giảm sản phẩm tương tự trong ranking.
    
    Sử dụng phạt kiểu MMR (Maximal Marginal Relevance) dựa trên BERT similarity.
    
    Thuật toán:
    1. Sắp xếp ứng viên theo điểm giảm dần
    2. Với mỗi sản phẩm, tính max similarity với các sản phẩm đã chọn
    3. Nếu max_sim > threshold: giảm điểm theo công thức phạt
    
    Công thức phạt:
        new_score = old_score * (1 - penalty * (max_sim - threshold) / (1 - threshold))
    
    Args:
        scores: Dict product_id -> điểm
        candidate_ids: Danh sách ID ứng viên đã sắp xếp
    
    Returns:
        Tuple (điểm_đã_phạt, điểm_đa_dạng)
    """
```

##### `rerank()` - Rerank gợi ý

```python
def rerank(
    self,
    cf_recommendations: List[Dict[str, Any]],
    user_id: Optional[int] = None,
    user_history: Optional[List[int]] = None,
    topk: Optional[int] = None,
    is_cold_start: bool = False
) -> RerankedResult:
    """
    Rerank gợi ý CF với các tín hiệu hybrid.
    
    Quy trình:
    1. Trích xuất ID ứng viên và điểm CF
    2. Chọn trọng số dựa trên loại người dùng
    3. Tính toán tất cả tín hiệu
    4. Chuẩn hóa tín hiệu sử dụng phạm vi toàn cục
    5. Kết hợp điểm với trọng số
    6. Áp dụng phạt đa dạng
    7. Sắp xếp và cập nhật rank
    
    Args:
        cf_recommendations: Danh sách dict gợi ý từ CFRecommender
        user_id: ID người dùng để log
        user_history: Lịch sử tương tác để tính content similarity
        topk: Số sản phẩm trả về (None = tất cả)
        is_cold_start: Người dùng cold-start (sử dụng trọng số khác)
    
    Returns:
        RerankedResult với gợi ý đã rerank và metadata
    """
```

---

## Thành Phần 3: PhoBERTEmbeddingLoader ✅

### Module: `service/recommender/phobert_loader.py`

**Mô tả**: Singleton loader để load và cache PhoBERT embeddings cho gợi ý dựa trên nội dung.

#### Tính năng chính:
1. **Singleton pattern**: Thread-safe, chỉ load một lần
2. **Pre-normalized embeddings**: Embeddings đã chuẩn hóa L2 sẵn cho cosine similarity nhanh
3. **Tính user profile**: Từ lịch sử tương tác (mean, weighted_mean, max)
4. **Tìm sản phẩm tương tự**: Dựa trên cosine similarity
5. **Pre-compute similarity matrix**: Cho catalog nhỏ (<3000 sản phẩm)

```python
class PhoBERTEmbeddingLoader:
    """
    Load và cache PhoBERT embeddings cho gợi ý dựa trên nội dung.
    
    Tính năng:
    - Singleton pattern (thread-safe)
    - Load embeddings từ file PyTorch .pt
    - Pre-normalize embeddings cho cosine similarity nhanh
    - Tính user profile từ lịch sử (mean, weighted_mean, max)
    - Tìm sản phẩm tương tự hiệu quả
    - Pre-compute item-item similarity matrix cho catalog nhỏ
    
    Ví dụ:
        >>> loader = PhoBERTEmbeddingLoader()
        >>> emb = loader.get_embedding(123)
        >>> similar = loader.find_similar_items(123, topk=10)
    """
```

#### Phương thức quan trọng:

##### `compute_user_profile()` - Tính profile người dùng

```python
def compute_user_profile(
    self,
    user_history_items: List[int],
    weights: Optional[List[float]] = None,
    strategy: str = 'weighted_mean'
) -> Optional[np.ndarray]:
    """
    Tính embedding profile người dùng từ lịch sử tương tác.
    
    Các chiến lược tổng hợp:
    - 'mean': Trung bình cộng đơn giản của embeddings
    - 'weighted_mean': Trung bình có trọng số (weights = ratings hoặc recency)
    - 'max': Max pooling (lấy max theo từng chiều)
    
    Args:
        user_history_items: Danh sách product_id đã tương tác
        weights: Trọng số tùy chọn cho mỗi item (rating, recency)
        strategy: Chiến lược tổng hợp ('mean', 'weighted_mean', 'max')
    
    Returns:
        np.array shape (embedding_dim,) đại diện profile người dùng
    """
```

##### `find_similar_items()` - Tìm sản phẩm tương tự

```python
def find_similar_items(
    self,
    product_id: int,
    topk: int = 10,
    exclude_self: bool = True,
    exclude_ids: Optional[Set[int]] = None
) -> List[Tuple[int, float]]:
    """
    Tìm top-K sản phẩm tương tự với sản phẩm cho trước.
    
    Sử dụng cosine similarity trên embeddings đã chuẩn hóa:
        similarity = embeddings_norm @ query_emb
    
    Args:
        product_id: ID sản phẩm truy vấn
        topk: Số sản phẩm tương tự cần tìm
        exclude_self: Loại trừ sản phẩm truy vấn
        exclude_ids: Các ID sản phẩm cần loại trừ thêm
    
    Returns:
        Danh sách (product_id, similarity_score) tuples
    """
```

---

## Thành Phần 4: FallbackRecommender ✅

### Module: `service/recommender/fallback.py`

**Mô tả**: Xử lý ~91.4% lưu lượng cold-start với chiến lược hybrid content + popularity.

#### Các chiến lược fallback:
1. **Popularity-based**: Trả về sản phẩm bán chạy nhất
2. **Item-similarity**: Dựa trên PhoBERT content similarity
3. **Hybrid**: Kết hợp content similarity và popularity

```python
class FallbackRecommender:
    """
    Các chiến lược gợi ý fallback cho người dùng cold-start.
    
    Cung cấp gợi ý dựa trên nội dung và độ phổ biến
    cho người dùng không có đủ dữ liệu CF.
    
    Các chiến lược:
    1. Popularity-based: Trả về sản phẩm bán chạy nhất
    2. Item-similarity: Content-based sử dụng PhoBERT embeddings
    3. Hybrid: Kết hợp content similarity và popularity
    
    Ví dụ:
        >>> fallback = FallbackRecommender(loader, phobert_loader)
        >>> recs = fallback.recommend(user_history, topk=10)
    """
```

#### Phương thức quan trọng:

##### `hybrid_fallback()` - Fallback hybrid

```python
def hybrid_fallback(
    self,
    user_history: List[int],
    topk: int = 10,
    content_weight: Optional[float] = None,
    popularity_weight: Optional[float] = None,
    exclude_ids: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """
    Fallback hybrid kết hợp content similarity và popularity.
    
    Công thức điểm cuối:
        final_score = content_weight * content_score + popularity_weight * pop_score
    
    Mặc định: content_weight=0.7, popularity_weight=0.3
    
    Quy trình:
    1. Lấy gợi ý content-based (2x topk để có đủ sau filter)
    2. Lấy điểm popularity cho từng sản phẩm
    3. Tính điểm kết hợp
    4. Sắp xếp và trả về top-K
    
    Args:
        user_history: Lịch sử mua hàng của người dùng
        topk: Số gợi ý cần trả về
        content_weight: Trọng số content (mặc định 0.7)
        popularity_weight: Trọng số popularity (mặc định 0.3)
        exclude_ids: Các ID sản phẩm cần loại trừ
    
    Returns:
        Danh sách dict gợi ý với metadata
    """
```

---

## Thành Phần 5: Tích Hợp CFRecommender ✅

### Module: `service/recommender/recommender.py`

**Mô tả**: Engine gợi ý chính với CF scoring, reranking, và xử lý fallback.

#### Logic routing:
- **Người dùng trainable** (≥2 tương tác): CF scoring → HybridReranker
- **Người dùng cold-start** (<2 tương tác): FallbackRecommender → rerank_cold_start

```python
class CFRecommender:
    """
    Engine gợi ý chính với CF scoring, reranking, và xử lý fallback.
    
    Tính năng:
    - Routing theo phân khúc người dùng (CF vs content-based)
    - Tính điểm CF sử dụng U @ V.T
    - Hybrid reranking với content, popularity, quality signals
    - Lọc sản phẩm đã xem, lọc theo thuộc tính
    - Fallback cold-start sang content-based + popularity
    
    Ví dụ:
        >>> recommender = CFRecommender()
        >>> result = recommender.recommend(user_id=12345, topk=10)
    """
```

#### Phương thức quan trọng:

##### `recommend()` - Tạo gợi ý

```python
def recommend(
    self,
    user_id: int,
    topk: int = 10,
    exclude_seen: bool = True,
    filter_params: Optional[Dict[str, Any]] = None,
    normalize_scores: bool = False,
    rerank: Optional[bool] = None
) -> RecommendationResult:
    """
    Tạo top-K gợi ý cho người dùng.
    
    Logic routing:
    1. Kiểm tra is_trainable_user
    2. Nếu trainable: CF scoring → HybridReranker
    3. Nếu cold-start: FallbackRecommender → rerank_cold_start
    
    Args:
        user_id: ID người dùng gốc (int)
        topk: Số gợi ý (mặc định 10)
        exclude_seen: Loại trừ sản phẩm đã tương tác
        filter_params: Bộ lọc thuộc tính (ví dụ: {'brand': 'Innisfree'})
        normalize_scores: Chuẩn hóa điểm CF về [0, 1]
        rerank: Ghi đè cài đặt reranking mặc định
    
    Returns:
        RecommendationResult với gợi ý và metadata
    """
```

##### Các phương thức tiện ích:

```python
def set_reranking(self, enabled: bool) -> None:
    """Bật hoặc tắt hybrid reranking."""

def update_rerank_weights(
    self,
    weights_trainable: Optional[Dict[str, float]] = None,
    weights_cold_start: Optional[Dict[str, float]] = None
) -> None:
    """Cập nhật trọng số reranking động mà không cần restart."""
```

---

## Thành Phần 6: Hybrid Metrics ✅

### Module: `recsys/cf/evaluation/hybrid_metrics.py`

**Mô tả**: Các metric đánh giá cho hệ thống gợi ý hybrid.

#### Các metric đã triển khai:

##### 1. DiversityMetric - Đo đa dạng trong danh sách

```python
class DiversityMetric(HybridMetric):
    """
    Metric đo độ đa dạng trong danh sách (Intra-List Diversity).
    
    Công thức:
        Diversity = 1 - (1/K(K-1)) * ΣΣ similarity(i, j) với i ≠ j
    
    Giải thích:
        - Diversity = 0.3: Sản phẩm khá giống nhau (avg similarity = 0.7)
        - Diversity = 0.6: Sản phẩm khá đa dạng (avg similarity = 0.4)
        - Cao hơn = đa dạng hơn = tốt hơn
    
    Ví dụ:
        >>> diversity = DiversityMetric()
        >>> score = diversity.compute(recommendations, bert_embeddings)
    """
```

##### 2. NoveltyMetric - Đo độ mới lạ

```python
class NoveltyMetric(HybridMetric):
    """
    Metric đo độ mới lạ: sản phẩm ít phổ biến/bất ngờ.
    
    Công thức:
        Novelty@K = (1/K) * Σ log2(num_users / item_popularity_i)
    
    Giải thích:
        - Novelty cao: Gợi ý sản phẩm long-tail (ít phổ biến)
        - Novelty thấp: Gợi ý sản phẩm phổ biến
        - Đánh đổi với accuracy
    """
```

##### 3. SemanticAlignmentMetric - Đo độ phù hợp ngữ nghĩa

```python
class SemanticAlignmentMetric(HybridMetric):
    """
    Điểm căn chỉnh ngữ nghĩa: gợi ý CF khớp với sở thích nội dung của người dùng.
    
    Công thức:
        Alignment = (1/K) * Σ cosine_similarity(user_profile_emb, item_emb_i)
    
    Giải thích:
        - Alignment cao: Gợi ý CF khớp với sở thích nội dung
        - Hữu ích để xác nhận BERT-initialized embeddings
    """
```

##### 4. ColdStartCoverageMetric - Đo độ phủ cold-start

```python
class ColdStartCoverageMetric(HybridMetric):
    """
    Độ phủ Cold-Start: phần trăm sản phẩm cold-start được gợi ý.
    
    Công thức:
        ColdStartCoverage = |Sản phẩm cold unique trong tất cả gợi ý| / |Tổng sản phẩm cold|
    
    Giải thích:
        - Coverage cao: Hệ thống có thể expose sản phẩm mới
        - Quan trọng cho độ tươi mới của catalog
    """
```

##### 5. HybridMetricCollection - Bộ sưu tập metric

```python
class HybridMetricCollection:
    """
    Bộ sưu tập các metric hybrid để đánh giá toàn diện.
    
    Ví dụ:
        >>> collection = HybridMetricCollection(k_values=[10, 20])
        >>> results = collection.evaluate_all(
        ...     all_recommendations=all_recs,
        ...     item_embeddings=embeddings,
        ...     item_popularity=popularity,
        ...     item_counts=counts
        ... )
    """
```

#### Các hàm tiện ích:

```python
# Tính đa dạng sử dụng BERT embeddings
compute_diversity_bert(recommendations, bert_embeddings, item_to_idx=None) -> float

# Tính độ căn chỉnh ngữ nghĩa
compute_semantic_alignment(user_profile_emb, recommendations, item_embeddings, item_to_idx=None) -> float

# Tính độ phủ cold-start
compute_cold_start_coverage(all_recommendations, item_counts, cold_threshold=5) -> float
```

---

## Thành Phần 7: Cấu Hình Serving ✅

### File: `config/serving_config.yaml`

```yaml
# Cấu hình Reranking
reranking:
  enabled: true
  
  # Trọng số cho người dùng trainable (có CF)
  weights_trainable:
    cf: 0.30              # Tín hiệu CF (cộng tác)
    content: 0.40         # Độ tương đồng PhoBERT
    popularity: 0.20      # Sản phẩm trending
    quality: 0.10         # Sản phẩm đánh giá cao
  
  # Trọng số cho người dùng cold-start (không có CF)
  weights_cold_start:
    content: 0.60         # Tín hiệu CHỦ ĐẠO
    popularity: 0.30      # Bằng chứng xã hội
    quality: 0.10         # Thưởng
  
  # Cài đặt đa dạng
  diversity:
    enabled: true
    penalty: 0.1          # Phạt cho sản phẩm giống
    threshold: 0.85       # Ngưỡng BERT similarity
  
  # Mở rộng ứng viên
  candidate_multiplier: 5  # Tạo 5x ứng viên cho reranking

# Cấu hình PhoBERT
phobert:
  embeddings_path: "data/processed/content_based_embeddings/product_embeddings.pt"
  precompute_similarity_matrix: true
  max_items_for_precompute: 3000
  user_profile_strategy: "weighted_mean"

# Cấu hình Fallback
fallback:
  default_strategy: "hybrid"
  content_weight: 0.7
  popularity_weight: 0.3
  enable_cache: true

# Mục tiêu hiệu năng
targets:
  latency:
    p50_ms: 50            # Median latency
    p90_ms: 100           # Phân vị 90
    p95_ms: 150           # Phân vị 95
    p99_ms: 200           # Phân vị 99 (SLA)
  cache:
    hit_rate_target: 0.70 # Mục tiêu 70% cache hit
```

---

## Các Trường Hợp Sử Dụng

### 1. Gợi ý cho người dùng Trainable

```python
from service.recommender import CFRecommender

recommender = CFRecommender()

# Người dùng trainable → CF + HybridReranker
result = recommender.recommend(user_id=12345, topk=10)
print(f"Là fallback: {result.is_fallback}")  # False
print(f"Model: {result.model_id}")           # 'als_20250115_v1'

# Mỗi gợi ý có chi tiết tín hiệu
for rec in result.recommendations:
    print(f"Sản phẩm {rec['product_id']}: final={rec['final_score']:.3f}")
    print(f"  CF: {rec['signals']['cf']:.3f}")
    print(f"  Content: {rec['signals']['content']:.3f}")
    print(f"  Popularity: {rec['signals']['popularity']:.3f}")
```

### 2. Gợi ý cho người dùng Cold-Start

```python
# Người dùng cold-start → FallbackRecommender + rerank_cold_start
result = recommender.recommend(user_id=999999, topk=10)
print(f"Là fallback: {result.is_fallback}")  # True
print(f"Phương thức fallback: {result.fallback_method}")  # 'hybrid'

# Gợi ý vẫn có chi tiết tín hiệu (nhưng cf=0)
for rec in result.recommendations:
    print(f"Sản phẩm {rec['product_id']}: score={rec['score']:.3f}")
```

### 3. Điều chỉnh trọng số động

```python
# Tăng trọng số content, giảm CF
recommender.update_rerank_weights(
    weights_trainable={'cf': 0.20, 'content': 0.50, 'popularity': 0.20, 'quality': 0.10}
)

# Tắt reranking hoàn toàn
recommender.set_reranking(enabled=False)
```

### 4. Đánh giá đa dạng

```python
from recsys.cf.evaluation.hybrid_metrics import HybridMetricCollection, compute_diversity_bert
from service.recommender.phobert_loader import get_phobert_loader

phobert = get_phobert_loader()

# Đa dạng cho một danh sách
recs_ids = [rec['product_id'] for rec in result.recommendations]
diversity = compute_diversity_bert(recs_ids, phobert.embeddings_norm, phobert.product_id_to_idx)
print(f"Đa dạng: {diversity:.3f}")

# Đánh giá đầy đủ
collection = HybridMetricCollection(k_values=[10, 20])
all_recs = {user_id: [rec['product_id'] for rec in result.recommendations]}
metrics = collection.evaluate_all(
    all_recommendations=all_recs,
    item_embeddings=phobert.embeddings_norm,
    item_popularity=item_popularity,
    item_counts=item_counts,
    num_users=300000
)
print(metrics)
```

---

## Tóm Tắt Tích Hợp Liên Task

| Task | Điểm tích hợp | Trạng thái |
|------|---------------|------------|
| Task 01 (Data Layer) | `data_stats.json` cho phạm vi chuẩn hóa | ✅ |
| Task 03 (Evaluation) | `HybridMetricCollection` cho diversity/alignment | ✅ |
| Task 05 (Serving) | Tích hợp `PhoBERTEmbeddingLoader` | ✅ |
| Task 06 (Monitoring) | Theo dõi latency, metric đa dạng | ✅ |
| Task 07 (Automation) | Tự động refresh BERT embeddings | ✅ |

---

## Tiêu Chí Thành Công ✅ ĐẠT ĐƯỢC

- [x] HybridReranker tích hợp với PhoBERTEmbeddingLoader
- [x] FallbackRecommender xử lý 91.4% lưu lượng cold-start
- [x] Chuẩn hóa toàn cục (không phải cục bộ theo request)
- [x] Phạt đa dạng sử dụng ngưỡng BERT similarity
- [x] Cập nhật trọng số động không cần restart
- [x] Hybrid metrics (DiversityMetric, NoveltyMetric, SemanticAlignmentMetric, ColdStartCoverageMetric)
- [x] Cấu hình serving trong `config/serving_config.yaml`
- [x] Tích hợp CFRecommender với logic routing

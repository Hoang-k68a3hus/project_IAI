# Hướng Dẫn Training: ALS & BPR Models

> **Hướng dẫn thực hành training các mô hình Collaborative Filtering**  
> Cập nhật: 2025-01-16

## 📋 Tổng Quan

Hướng dẫn này sẽ đưa bạn qua quá trình training các mô hình ALS và BPR cho hệ thống gợi ý mỹ phẩm Việt Nam.

**Yêu cầu:**
- Task 01 đã hoàn thành (dữ liệu đã xử lý trong `data/processed/`)
- Python 3.10+ với dependencies đã cài đặt
- Tối thiểu ~4GB RAM

---

## 🚀 Bắt Đầu Nhanh

### Train Mô Hình ALS (5 phút)

```python
from recsys.cf.model.als import (
    ALSMatrixPreparer,
    ALSModelInitializer,
    ALSTrainer,
    ALSEvaluator,
    save_als_complete
)

# Bước 1: Load dữ liệu
preparer = ALSMatrixPreparer(base_path='data/processed')
data = preparer.prepare_complete_als_data()

# Bước 2: Khởi tạo mô hình (dùng preset sparse_data)
initializer = ALSModelInitializer(preset='sparse_data')
model = initializer.initialize_model()

# Bước 3: Training
trainer = ALSTrainer(model=model)
result = trainer.fit(data.X_train_implicit)

# Bước 4: Đánh giá
evaluator = ALSEvaluator(
    user_factors=model.user_factors,
    item_factors=model.item_factors,
    user_to_idx=data.mappings['user_to_idx'],
    idx_to_user=data.mappings['idx_to_user'],
    item_to_idx=data.mappings['item_to_idx'],
    idx_to_item=data.mappings['idx_to_item'],
    user_pos_train=data.user_pos_train,
    user_pos_test=data.user_pos_test
)
results = evaluator.evaluate(k_values=[10, 20], compare_baseline=True)
results.print_summary()

# Bước 5: Lưu artifacts
artifacts = save_als_complete(
    user_embeddings=model.user_factors,
    item_embeddings=model.item_factors,
    params=initializer.config,
    metrics=results.metrics,
    validation_user_indices=list(data.user_pos_test.keys())[:1000],
    data_version_hash=data.metadata.get('data_hash', 'unknown'),
    output_dir='artifacts/cf/als'
)
```

### Train Mô Hình BPR (20-30 phút)

```python
from recsys.cf.model.bpr import (
    BPRDataLoader,
    TripletSampler,
    HardNegativeMixer,
    BPRTrainer,
    save_bpr_complete
)
import numpy as np

# Bước 1: Load dữ liệu
loader = BPRDataLoader(base_path='data/processed')
data = loader.load_all()

# Bước 2: Khởi tạo embeddings
rng = np.random.default_rng(42)
U = rng.normal(0, 0.01, (data.num_users, 64)).astype(np.float32)
V = rng.normal(0, 0.01, (data.num_items, 64)).astype(np.float32)

# Bước 3: Cấu hình sampler
mixer = HardNegativeMixer(
    hard_neg_sets=data.hard_neg_sets,
    hard_ratio=0.3
)

# Bước 4: Training
trainer = BPRTrainer(
    U=U,
    V=V,
    learning_rate=0.05,
    regularization=0.0001
)
history = trainer.fit(
    positive_pairs=data.positive_pairs,
    user_pos_sets=data.user_pos_sets,
    hard_neg_sets=data.hard_neg_sets,
    num_items=data.num_items,
    epochs=50,
    samples_per_epoch=5
)

# Bước 5: Lấy embeddings cuối cùng
U, V = trainer.get_embeddings()

# Bước 6: Lưu artifacts
artifacts = save_bpr_complete(
    user_embeddings=U,
    item_embeddings=V,
    params={'factors': 64, 'lr': 0.05, 'reg': 0.0001, 'hard_ratio': 0.3},
    metrics={'best_epoch': history.get_best_epoch('recall@10')},
    training_history=history,
    data_version_hash='unknown',
    output_dir='artifacts/cf/bpr'
)
```

---

## 📊 So Sánh Mô Hình

| Khía cạnh | ALS | BPR |
|-----------|-----|-----|
| **Thời gian train** | 1-2 phút | 20-30 phút |
| **Bộ nhớ sử dụng** | ~2GB | ~1GB |
| **Phù hợp cho** | Lặp nhanh | Chất lượng ranking |
| **Điểm mạnh** | Nhanh, hỗ trợ GPU | Hard negative mining |
| **Điểm yếu** | Point-wise loss | Training chậm hơn |

---

## ⚙️ Cấu Hình Presets

### ALS Presets

| Preset | factors | regularization | alpha | Trường hợp sử dụng |
|--------|---------|----------------|-------|--------------------|
| `default` | 64 | 0.01 | 10 | Sử dụng chung |
| `sparse_data` | 64 | 0.10 | 5 | **Khuyến nghị** cho dataset này |
| `high_quality` | 128 | 0.05 | 10 | Embeddings phong phú hơn |
| `fast` | 32 | 0.01 | 10 | Test nhanh |
| `normalized` | 64 | 0.01 | 40 | Cho confidence đã chuẩn hóa |

**Tại sao nên dùng `sparse_data`?**
- Dữ liệu của chúng ta có mật độ matrix ~0.11%
- Regularization cao hơn (λ=0.1) ngăn overfitting
- Alpha thấp hơn (5) bù đắp cho confidence range 1-6

### Cấu Hình BPR

**Cơ bản (SGD):**
```python
trainer = BPRTrainer(
    U=U, V=V,
    learning_rate=0.05,
    regularization=0.0001,
    lr_decay=0.9,
    lr_decay_every=10
)
```

**Nâng cao (AdamW + Dropout):**
```python
from recsys.cf.model.bpr import (
    AdvancedBPRTrainer,
    OptimizerConfig,
    TrainingConfig,
    OptimizerType
)

trainer = AdvancedBPRTrainer(
    U=U, V=V,
    optimizer_config=OptimizerConfig(
        optimizer_type=OptimizerType.ADAMW,
        learning_rate=0.01,
        weight_decay=0.01
    ),
    training_config=TrainingConfig(
        dropout_rate=0.1,
        gradient_clip=1.0
    )
)
```

---

## 🎯 Hard Negative Sampling

### Tại Sao Quan Trọng
- Random negatives quá dễ → mô hình không học được sở thích chi tiết
- Hard negatives buộc mô hình phân biệt các items tương tự

### Chiến Lược (30% hard + 70% random)

```python
mixer = HardNegativeMixer(
    hard_neg_sets=data.hard_neg_sets,
    hard_ratio=0.3  # 30% from hard negatives
)
```

**Nguồn hard negatives:**
1. **Explicit**: Items user đánh giá ≤3 sao (không thích rõ ràng)
2. **Implicit**: Top-50 items phổ biến user không mua (từ chối ngầm)

### Theo Dõi Thống Kê Sampling

```python
# Sau khi training
stats = mixer.get_stats()
print(f"Hard samples: {stats['hard_count']} ({stats['hard_ratio']:.1%})")
print(f"Random samples: {stats['random_count']} ({stats['random_ratio']:.1%})")
print(f"Fallbacks: {stats['fallback_count']}")
```

---

## 📈 Theo Dõi Training

### Tiến Trình ALS

```python
trainer = ALSTrainer(
    model=model,
    track_memory=True,
    checkpoint_interval=5
)
result = trainer.fit(X_train)

print(f"Training time: {result.training_time:.1f}s")
print(f"Peak memory: {result.memory_usage['peak_mb']:.1f} MB")
```

### Tiến Trình BPR

```python
# Training history theo dõi:
# - losses: BPR loss mỗi epoch
# - val_metrics: Validation Recall@K, NDCG@K
# - learning_rates: Lịch LR
# - durations: Thời gian mỗi epoch

history = trainer.fit(...)

# Vẽ learning curve
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Đồ thị Loss
axes[0].plot(history.epochs, history.losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('BPR Loss')
axes[0].set_title('Training Loss')

# Đồ thị Recall
if 'recall@10' in history.val_metrics:
    axes[1].plot(history.epochs, history.val_metrics['recall@10'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Recall@10')
    axes[1].set_title('Validation Recall')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🔧 Xử Lý Sự Cố

### Lỗi Bộ Nhớ

**ALS hết bộ nhớ (OOM):**
```python
# Cách 1: Giảm factors
initializer = ALSModelInitializer(preset='fast')  # factors=32

# Cách 2: Bật GPU (cần cupy)
initializer = ALSModelInitializer(config={
    'factors': 64,
    'use_gpu': True
})
```

**BPR hết bộ nhớ (OOM):**
```python
# Giảm samples mỗi epoch
trainer.fit(
    ...,
    samples_per_epoch=3  # Thay vì 5
)
```

### Training Không Ổn Định

**Loss dao động (BPR):**
```python
# Giảm learning rate
trainer = BPRTrainer(
    U=U, V=V,
    learning_rate=0.01,  # Giảm từ 0.05
    lr_decay=0.95,       # Decay mạnh hơn
    lr_decay_every=5
)
```

**NaN trong embeddings:**
```python
# Tăng regularization
trainer = BPRTrainer(
    U=U, V=V,
    regularization=0.001  # Tăng từ 0.0001
)

# Hoặc dùng gradient clipping
from recsys.cf.model.bpr import AdvancedBPRTrainer, TrainingConfig

trainer = AdvancedBPRTrainer(
    U=U, V=V,
    training_config=TrainingConfig(gradient_clip=1.0)
)
```

### Metrics Thấp

**Recall@10 dưới baseline:**
1. Kiểm tra test set chỉ có positive (rating ≥4)
2. Xác nhận seen item filtering hoạt động
3. Thử factors cao hơn (128) hoặc regularization thấp hơn

```python
# Debug: Kiểm tra cấu hình evaluation
print(f"Test users: {len(user_pos_test)}")
print(f"Trung bình test items mỗi user: {np.mean([len(v) for v in user_pos_test.values()]):.1f}")
print(f"Train items đã lọc: {sum(len(v) for v in user_pos_train.values())}")
```

---

## 🧪 Tuning Hyperparameter

### Ví Dụ Grid Search

```python
from itertools import product
import pandas as pd

# ALS grid
param_grid = {
    'factors': [32, 64, 128],
    'regularization': [0.01, 0.05, 0.1],
    'alpha': [5, 10, 20]
}

results = []
for factors, reg, alpha in product(*param_grid.values()):
    # Training
    initializer = ALSModelInitializer(config={
        'factors': factors,
        'regularization': reg,
        'alpha': alpha,
        'iterations': 15
    })
    model = initializer.initialize_model()
    trainer = ALSTrainer(model=model)
    trainer.fit(data.X_train_implicit)
    
    # Đánh giá
    evaluator = ALSEvaluator(...)
    metrics = evaluator.evaluate(k_values=[10])
    
    results.append({
        'factors': factors,
        'regularization': reg,
        'alpha': alpha,
        'recall@10': metrics.metrics['recall@10']
    })

# Tìm config tốt nhất
df = pd.DataFrame(results)
best = df.loc[df['recall@10'].idxmax()]
print(f"Config tốt nhất: {best.to_dict()}")
```

---

## 📦 Lưu & Load

### Lưu Toàn Bộ Artifacts

```python
# ALS
from recsys.cf.model.als import save_als_complete

artifacts = save_als_complete(
    user_embeddings=model.user_factors,
    item_embeddings=model.item_factors,
    params={'factors': 64, 'regularization': 0.1, 'alpha': 10},
    metrics={'recall@10': 0.234, 'ndcg@10': 0.189},
    validation_user_indices=list(user_pos_test.keys())[:1000],
    data_version_hash='abc123',
    output_dir='artifacts/cf/als'
)

# Các file được tạo:
# - artifacts/cf/als/als_U.npy
# - artifacts/cf/als/als_V.npy
# - artifacts/cf/als/als_params.json
# - artifacts/cf/als/als_metrics.json
# - artifacts/cf/als/als_metadata.json (bao gồm score_range)
```

### Load Cho Serving

```python
import numpy as np
import json

# Load embeddings
U = np.load('artifacts/cf/als/als_U.npy')
V = np.load('artifacts/cf/als/als_V.npy')

# Load metadata
with open('artifacts/cf/als/als_metadata.json') as f:
    metadata = json.load(f)

# Lấy score range cho normalization (Task 08)
score_range = metadata['score_range']
print(f"Score range: [{score_range['p01']:.3f}, {score_range['p99']:.3f}]")
```

---

## 🔗 Tích Hợp Với Task 08

### Score Range Cho Hybrid Reranking

Cả ALS và BPR đều lưu `score_range` trong metadata cho Task 08 hybrid reranking:

```python
# Trong quá trình training/saving
artifacts = save_als_complete(
    ...,
    validation_user_indices=[10, 25, 42, ...],  # Quan trọng!
    ...
)

# Score range được tính bằng U @ V.T trên validation users
# Cung cấp p01, p99 percentiles cho normalization ổn định
```

### Sử Dụng Trong Task 08

```python
def normalize_cf_scores(scores, score_range):
    """Chuẩn hóa CF scores về [0, 1] dùng p01-p99 range."""
    p01, p99 = score_range['p01'], score_range['p99']
    normalized = (scores - p01) / (p99 - p01)
    return np.clip(normalized, 0, 1)
```

---

## 📚 Tài Liệu Liên Quan

- [Task 02: Training Pipelines](../tasks/02_training_pipelines.md) - Đặc tả kỹ thuật đầy đủ
- [Hướng Dẫn Xử Lý Dữ Liệu](DATA_PROCESSING_GUIDE.md) - Outputs của Task 01
- [API Reference](API_REFERENCE.md) - Các endpoints phục vụ

---

---

## 📏 Đánh Giá Mô Hình (Evaluation)

### Tổng Quan Metrics

| Metric | Công Thức | Mô Tả |
|--------|-----------|-------|
| **Recall@K** | `\|Top-K ∩ Test\| / \|Test\|` | Tỷ lệ items test tìm thấy trong top-K |
| **NDCG@K** | `DCG@K / IDCG@K` | Chất lượng ranking (items ở top được reward) |
| **Precision@K** | `\|Top-K ∩ Test\| / K` | Độ chính xác của top-K |
| **Coverage** | `\|Unique Recs\| / \|All Items\|` | Đa dạng: % items được recommend |

### Đánh Giá Nhanh Với ALSEvaluator

```python
from recsys.cf.model.als import ALSEvaluator

# Khởi tạo evaluator
evaluator = ALSEvaluator(
    user_factors=model.user_factors,
    item_factors=model.item_factors,
    user_to_idx=data.mappings['user_to_idx'],
    idx_to_user=data.mappings['idx_to_user'],
    item_to_idx=data.mappings['item_to_idx'],
    idx_to_item=data.mappings['idx_to_item'],
    user_pos_train=data.user_pos_train,
    user_pos_test=data.user_pos_test
)

# Chạy evaluation với so sánh baseline
results = evaluator.evaluate(
    k_values=[10, 20],          # K values cần đánh giá
    filter_seen=True,            # Lọc items đã thấy trong train
    compare_baseline=True,       # So sánh với popularity baseline
    baseline_source='train',     # Nguồn popularity: 'train' hoặc 'metadata'
    model_type='als'
)

# In kết quả dạng bảng
results.print_summary()
```

**Output mẫu:**
```
======================================================================
EVALUATION RESULTS: ALS
======================================================================

Test Users: 26234
K Values: [10, 20]
Evaluation Time: 45.23s

----------------------------------------------------------------------
Metric               Model        Baseline     Improvement    
----------------------------------------------------------------------
recall@10            0.2453       0.1421       +72.6%         
ndcg@10              0.1892       0.1024       +84.8%         
recall@20            0.3124       0.2013       +55.2%         
ndcg@20              0.2215       0.1342       +65.1%         
======================================================================
```

### Đánh Giá Đơn Giản (Không Cần Full Mappings)

```python
from recsys.cf.model.als import quick_evaluate

metrics = quick_evaluate(
    user_factors=U,
    item_factors=V,
    user_pos_test=user_pos_test,
    user_pos_train=user_pos_train,
    k_values=[10, 20],
    model_type='als'
)

print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

### So Sánh Với Popularity Baseline

```python
from recsys.cf.model.als import PopularityBaseline

# Khởi tạo baseline từ training data
baseline = PopularityBaseline()
baseline.fit_from_train(
    user_pos_train=user_pos_train,
    num_items=2231
)

# Hoặc từ product metadata (num_sold_time)
# baseline.fit_from_metadata(
#     product_df=product_df,
#     item_to_idx=item_to_idx,
#     popularity_col='num_sold_time'
# )

# Đánh giá baseline
baseline_metrics = baseline.evaluate(
    user_pos_test=user_pos_test,
    user_pos_train=user_pos_train,
    k_values=[10, 20],
    filter_seen=True
)

print(f"Baseline Recall@10: {baseline_metrics['recall@10']:.4f}")
```

**Expected Performance:**
| Mô Hình | Recall@10 | NDCG@10 | Coverage |
|---------|-----------|---------|----------|
| **Popularity Baseline** | 0.12-0.15 | 0.08-0.10 | <0.05 |
| **ALS** | >0.20 | >0.16 | 0.25-0.35 |
| **BPR** | >0.22 | >0.18 | 0.28-0.38 |

### Đánh Giá Hybrid (CF + BERT Reranking)

```bash
# So sánh pure CF vs hybrid reranking
python scripts/evaluate_hybrid.py --num-users 500 --topk 10 20

# Chỉ đánh giá pure CF
python scripts/evaluate_hybrid.py --cf-only --num-users 500

# Lưu kết quả
python scripts/evaluate_hybrid.py --output reports/hybrid_eval.json
```

**Các metrics hybrid bổ sung:**
| Metric | Mô Tả |
|--------|-------|
| **Diversity** | 1 - avg pairwise similarity (BERT) trong top-K |
| **Semantic Alignment** | Cosine similarity giữa user profile và recommendations |
| **Brand Coverage** | % brands đa dạng trong recommendations |

```python
# Sử dụng HybridEvaluator trực tiếp
from scripts.evaluate_hybrid import HybridEvaluator, compare_cf_vs_hybrid

# So sánh CF vs Hybrid
comparison = compare_cf_vs_hybrid(
    cf_recommender=recommender,
    test_data=test_data,
    phobert_loader=phobert_loader,
    metadata=product_df,
    num_users=200,
    k_values=[5, 10, 20]
)

print(f"Diversity improvement: {comparison['summary']['diversity_improvement']:+.1f}%")
print(f"Recall@10 improvement: {comparison['summary']['recall@10_improvement']:+.1f}%")
```

### Lưu Kết Quả Evaluation

```python
from pathlib import Path

# Lưu kết quả evaluation
results.save(Path('artifacts/cf/als/als_eval_results.json'))

# Hoặc thủ công
import json

eval_output = {
    'model_type': 'als',
    'metrics': {
        'recall@10': 0.2453,
        'ndcg@10': 0.1892,
        'coverage': 0.312
    },
    'baseline': {
        'recall@10': 0.1421,
        'ndcg@10': 0.1024
    },
    'improvement': {
        'recall@10': '+72.6%',
        'ndcg@10': '+84.8%'
    },
    'num_test_users': 26234,
    'evaluation_time_seconds': 45.23
}

with open('reports/als_eval.json', 'w', encoding='utf-8') as f:
    json.dump(eval_output, f, indent=2, ensure_ascii=False)
```

### Phân Tích Per-User (Advanced)

```python
import numpy as np
from collections import defaultdict

# Thu thập metrics theo từng user
per_user_metrics = defaultdict(list)

for user_idx, ground_truth in user_pos_test.items():
    scores = model.user_factors[user_idx] @ model.item_factors.T
    
    # Lọc seen items
    if user_idx in user_pos_train:
        scores[list(user_pos_train[user_idx])] = -np.inf
    
    # Top-K predictions
    top_k = np.argpartition(scores, -10)[-10:]
    predictions = top_k[np.argsort(scores[top_k])[::-1]]
    
    # Recall@10
    hits = len(set(predictions) & ground_truth)
    recall = hits / len(ground_truth)
    per_user_metrics['recall@10'].append(recall)
    per_user_metrics['user_idx'].append(user_idx)

# Phân tích distribution
recalls = np.array(per_user_metrics['recall@10'])
print(f"Recall@10 Mean: {recalls.mean():.4f}")
print(f"Recall@10 Std: {recalls.std():.4f}")
print(f"Recall@10 Median: {np.median(recalls):.4f}")
print(f"Users với Recall=0: {(recalls == 0).sum()} ({(recalls == 0).mean()*100:.1f}%)")
```

### Stratification Theo User Activity

```python
# Phân nhóm users theo số interactions
def stratified_evaluation(user_pos_train, user_pos_test, metrics):
    """Đánh giá theo activity level."""
    
    # Phân nhóm
    low_activity = []     # 2-5 interactions
    medium_activity = []  # 6-15 interactions
    high_activity = []    # >15 interactions
    
    user_recalls = dict(zip(
        per_user_metrics['user_idx'], 
        per_user_metrics['recall@10']
    ))
    
    for user_idx, train_items in user_pos_train.items():
        if user_idx not in user_recalls:
            continue
        
        recall = user_recalls[user_idx]
        num_train = len(train_items)
        
        if num_train <= 5:
            low_activity.append(recall)
        elif num_train <= 15:
            medium_activity.append(recall)
        else:
            high_activity.append(recall)
    
    print(f"Low activity (2-5): Recall@10 = {np.mean(low_activity):.4f} ({len(low_activity)} users)")
    print(f"Medium activity (6-15): Recall@10 = {np.mean(medium_activity):.4f} ({len(medium_activity)} users)")
    print(f"High activity (>15): Recall@10 = {np.mean(high_activity):.4f} ({len(high_activity)} users)")

stratified_evaluation(user_pos_train, user_pos_test, per_user_metrics)
```

**Expected Output:**
```
Low activity (2-5): Recall@10 = 0.1823 (18234 users)
Medium activity (6-15): Recall@10 = 0.2891 (6521 users)
High activity (>15): Recall@10 = 0.3542 (1479 users)
```

### Kiểm Tra Statistical Significance

```python
from scipy import stats

# Paired t-test: CF vs Baseline
cf_recalls = per_user_cf['recall@10']
baseline_recalls = per_user_baseline['recall@10']

t_stat, p_value = stats.ttest_rel(cf_recalls, baseline_recalls)

print(f"Paired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.2e}")

if p_value < 0.05:
    print("  → Significant improvement (p < 0.05)")
else:
    print("  → Not significant")

# Effect size (Cohen's d)
mean_diff = np.mean(cf_recalls) - np.mean(baseline_recalls)
pooled_std = np.sqrt((np.std(cf_recalls)**2 + np.std(baseline_recalls)**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print("  → Small effect")
elif abs(cohens_d) < 0.5:
    print("  → Medium effect")
else:
    print("  → Large effect")
```

### Debug: Inspect Recommendations

```python
import pandas as pd

def inspect_user_recommendations(
    user_id: str,
    model,
    mappings,
    products_df,
    user_pos_train,
    k: int = 10
):
    """Debug recommendations cho 1 user cụ thể."""
    
    # Map user_id → u_idx
    if user_id not in mappings['user_to_idx']:
        print(f"User {user_id} không tìm thấy")
        return
    
    u_idx = mappings['user_to_idx'][user_id]
    
    # Compute scores
    scores = model.user_factors[u_idx] @ model.item_factors.T
    
    # Get seen items
    seen_items = user_pos_train.get(u_idx, set())
    print(f"User {user_id} (u_idx={u_idx}) đã thấy {len(seen_items)} items")
    
    # Filter seen
    scores[list(seen_items)] = -np.inf
    
    # Top-K
    top_k = np.argsort(scores)[::-1][:k]
    
    # Map to product_ids
    product_ids = [mappings['idx_to_item'][i] for i in top_k]
    
    # Get product info
    recs = products_df[products_df['product_id'].isin(product_ids)]
    recs = recs[['product_id', 'product_name', 'brand', 'avg_star', 'num_sold_time']]
    
    print(f"\nTop-{k} Recommendations:")
    for rank, (idx, row) in enumerate(recs.iterrows(), 1):
        print(f"  {rank}. {row['product_name'][:50]}...")
        print(f"     Brand: {row['brand']}, Rating: {row['avg_star']:.1f}, Sold: {row['num_sold_time']}")
    
    return recs

# Usage
# inspect_user_recommendations('12345', model, mappings, products_df, user_pos_train)
```

---

## 🗄️ Model Registry & Versioning

### Tổng Quan Registry

Model Registry quản lý tất cả các phiên bản models đã train, theo dõi performance, và tự động chọn "best model" cho production serving.

**Cấu trúc thư mục:**
```
artifacts/cf/
├── als/
│   ├── v1_20250115_103000/      # Version 1
│   │   ├── als_U.npy
│   │   ├── als_V.npy
│   │   ├── als_params.json
│   │   └── als_metadata.json
│   └── v2_20250116_141500/      # Version 2
├── bpr/
│   └── v1_20250115_120000/
├── registry.json                 # CF models registry
└── bert_registry.json           # BERT embeddings registry
```

### Đăng Ký Model Mới

```python
from recsys.cf.registry import ModelRegistry

registry = ModelRegistry(registry_path='artifacts/cf/registry.json')

# Đăng ký model sau khi train
model_id = registry.register_model(
    artifacts_path='artifacts/cf/als/v2_20250116_141500',
    model_type='als',
    hyperparameters={
        'factors': 128,
        'regularization': 0.01,
        'iterations': 20,
        'alpha': 60
    },
    metrics={
        'recall@10': 0.245,
        'ndcg@10': 0.195,
        'coverage': 0.310
    },
    training_info={
        'training_time_seconds': 102.8,
        'num_users': 26000,
        'num_items': 2200
    },
    data_version='abc123...',     # Hash từ Task 01
    git_commit='ghi789...',       # Auto-detect nếu None
    baseline_comparison={
        'baseline_type': 'popularity',
        'improvement_ndcg@10': 0.912  # 91.2% improvement
    }
)

print(f"Registered model: {model_id}")
# Output: Registered model: als_v2_20250116_141500
```

### Chọn Best Model Tự Động

```python
# Chọn model tốt nhất theo metric
best = registry.select_best_model(
    metric='ndcg@10',              # Metric để so sánh
    min_improvement=0.1,           # Tối thiểu 10% improvement so với baseline
    model_type=None                # None = tất cả types (als, bpr)
)

print(f"Best model: {best['model_id']}")
print(f"NDCG@10: {best['value']:.4f}")

# Output:
# Best model: als_v2_20250116_141500
# NDCG@10: 0.1950
```

### Liệt Kê & So Sánh Models

```python
import pandas as pd

# Liệt kê tất cả models
df = registry.list_models()

# Lọc theo loại và status
als_models = registry.list_models(
    model_type='als',
    status='active',
    sort_by='ndcg@10',
    ascending=False
)

print(als_models[['model_id', 'ndcg@10', 'recall@10', 'training_time']])
```

**Output mẫu:**
```
                model_id  ndcg@10  recall@10  training_time
0  als_v2_20250116_141500   0.195      0.245          102.8
1  als_v1_20250115_103000   0.189      0.234           45.2
```

```python
# So sánh chi tiết các models
comparison = registry.compare_models(
    model_ids=['als_v1_20250115_103000', 'als_v2_20250116_141500', 'bpr_v1_20250115_120000'],
    metrics=['recall@10', 'ndcg@10', 'coverage']
)

print(comparison)
```

### Load Model Cho Serving

```python
from recsys.cf.registry import ModelLoader, get_loader

# Cách 1: Singleton pattern (khuyến nghị cho serving)
loader = get_loader()

# Load current best model
U, V, metadata = loader.load_current_best()
print(f"Loaded model: {metadata['model_id']}")
print(f"Embeddings shape: U={U.shape}, V={V.shape}")

# Cách 2: Load model cụ thể
U, V, metadata = loader.load_model('als_v2_20250116_141500')

# Quick access embeddings
U, V = loader.get_embeddings()
```

### Hot-Reload Model (Không Downtime)

```python
# Kiểm tra và reload nếu có model mới
model_changed = loader.reload_model()

if model_changed:
    print("Model updated!")
    U, V, metadata = loader.load_current_best()
else:
    print("Model unchanged")

# Xem thống kê loader
stats = loader.get_stats()
print(f"Total loads: {stats['total_loads']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Last reload: {stats['last_reload_at']}")
```

### Archive & Delete Models

```python
# Archive model cũ (không xóa files)
success = registry.archive_model('als_v1_20250115_103000')
# Model archived → không xuất hiện trong best model selection

# Xóa model (cẩn thận!)
success = registry.delete_model(
    model_id='als_v1_20250115_103000',
    delete_files=False  # True để xóa cả files trên disk
)
```

**Lưu ý:**
- Không thể archive/delete model đang là `current_best`
- Tất cả thao tác được ghi vào `logs/registry_audit.log`

### Registry Statistics

```python
stats = registry.get_registry_stats()

print(f"Total models: {stats['total_models']}")
print(f"Active models: {stats['active_models']}")
print(f"Archived models: {stats['archived_models']}")
print(f"By type: {stats['by_type']}")  # {'als': 2, 'bpr': 1}
print(f"Current best: {stats['current_best']}")
```

### BERT Embeddings Registry (Riêng Biệt)

```python
from recsys.cf.registry import BERTEmbeddingsRegistry, get_bert_registry

bert_registry = get_bert_registry()

# Đăng ký BERT embeddings
version = bert_registry.register_embeddings(
    embedding_path='data/processed/content_based_embeddings',
    model_name='vinai/phobert-base',
    num_items=2244,
    embedding_dim=768,
    generation_config={
        'batch_size': 32,
        'max_length': 256
    },
    text_fields_used=['product_name', 'description', 'ingredients']
)

# Load embeddings
from recsys.cf.registry import load_bert_embeddings

embeddings, metadata = load_bert_embeddings()  # Current best
# hoặc
embeddings, metadata = load_bert_embeddings(version='bert_20250115_103000')
```

### Utility Functions

```python
from recsys.cf.registry.utils import (
    generate_version_id,
    compute_data_version,
    get_git_commit,
    backup_registry,
    restore_registry
)

# Tạo version ID
version = generate_version_id(prefix='v')  # → 'v_20250116_141500'

# Compute data version (hash)
data_version = compute_data_version([
    'data/processed/interactions.parquet',
    'data/processed/user_item_mappings.json'
])

# Get git commit
commit = get_git_commit()  # → 'def4567890abcdef...'

# Backup registry
backup_path = backup_registry('artifacts/cf/registry.json')

# Restore từ backup
restore_registry(
    backup_path='artifacts/cf/registry_backup_20250116_120000.json',
    registry_path='artifacts/cf/registry.json',
    create_current_backup=True
)
```

### Tích Hợp Với Training Pipeline

```python
from recsys.cf.registry import ModelRegistry
from recsys.cf.registry.utils import compute_data_version, get_git_commit

# Sau khi train xong
def register_trained_model(output_path, model_type, params, metrics, elapsed_time):
    """Đăng ký model vào registry sau khi train."""
    
    registry = ModelRegistry()
    
    # Compute data version
    data_version = compute_data_version([
        'data/processed/interactions.parquet',
        'data/processed/user_item_mappings.json'
    ])
    
    # Register
    model_id = registry.register_model(
        artifacts_path=output_path,
        model_type=model_type,
        hyperparameters=params,
        metrics=metrics,
        training_info={
            'training_time_seconds': elapsed_time,
            'num_users': metrics.get('num_users', 0),
            'num_items': metrics.get('num_items', 0)
        },
        data_version=data_version,
        git_commit=get_git_commit()
    )
    
    return model_id

# Usage trong training script
model_id = register_trained_model(
    output_path='artifacts/cf/als/v2_20250116_141500',
    model_type='als',
    params={'factors': 128, 'regularization': 0.01},
    metrics={'recall@10': 0.245, 'ndcg@10': 0.195},
    elapsed_time=102.8
)

# Auto-select nếu metrics tốt hơn
best = registry.select_best_model(metric='ndcg@10', min_improvement=0.05)
if best and best['model_id'] == model_id:
    print(f"🎉 New best model selected: {model_id}")
```

### Registry Schema (registry.json)

```json
{
  "current_best": {
    "model_id": "als_v2_20250116_141500",
    "model_type": "als",
    "version": "v2_20250116_141500",
    "path": "artifacts/cf/als/v2_20250116_141500",
    "selection_metric": "ndcg@10",
    "selection_value": 0.195,
    "selected_at": "2025-01-16T14:30:00"
  },
  
  "models": {
    "als_v2_20250116_141500": {
      "model_type": "als",
      "version": "v2_20250116_141500",
      "path": "artifacts/cf/als/v2_20250116_141500",
      "created_at": "2025-01-16T14:15:00",
      "data_version": "abc123...",
      "git_commit": "ghi789...",
      "hyperparameters": {
        "factors": 128,
        "regularization": 0.01,
        "iterations": 20,
        "alpha": 60
      },
      "metrics": {
        "recall@10": 0.245,
        "ndcg@10": 0.195,
        "coverage": 0.310
      },
      "baseline_comparison": {
        "baseline_type": "popularity",
        "improvement_ndcg@10": 0.912
      },
      "status": "active"
    }
  },
  
  "metadata": {
    "registry_version": "1.0",
    "last_updated": "2025-01-16T14:30:00",
    "num_models": 3,
    "selection_criteria": "ndcg@10"
  }
}
```

### Scripts Tiện Ích

```bash
# Update registry với model mới và auto-select
python scripts/update_registry.py \
  --model-path artifacts/cf/als/v2_20250116_141500 \
  --auto-select

# Cleanup models cũ (giữ 5 gần nhất)
python scripts/cleanup_old_models.py \
  --keep-last 5 \
  --archive-old  # Archive thay vì delete
```

### Audit Trail

Tất cả thao tác được ghi vào `logs/registry_audit.log`:

```
2025-01-15 10:30:00 | REGISTER | als_v1_20250115_103000 | ndcg@10=0.189
2025-01-16 14:15:00 | REGISTER | als_v2_20250116_141500 | ndcg@10=0.195
2025-01-16 14:30:00 | SELECT_BEST | als_v2_20250116_141500 | improvement=+3.2%
2025-01-17 09:00:00 | ARCHIVE | als_v1_20250115_103000 | reason=superseded
```

---

## ✅ Checklist

Trước khi deploy mô hình đã train:

- [ ] Recall@10 vượt popularity baseline ≥20%
- [ ] Score range đã tính và lưu trong metadata
- [ ] Training hoàn thành không có NaN/Inf
- [ ] Checkpoint đã lưu để fine-tuning nếu cần
- [ ] Metrics đã log để so sánh
- [ ] Evaluation report đã lưu (`artifacts/cf/{model}/eval_results.json`)
- [ ] Statistical significance test đã pass (p < 0.05)
- [ ] Model đã đăng ký vào Registry
- [ ] Best model đã được select (nếu metrics cải thiện)
- [ ] Audit log đã ghi nhận thao tác

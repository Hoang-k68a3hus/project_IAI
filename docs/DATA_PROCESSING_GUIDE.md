# Hướng Dẫn Xử Lý Dữ Liệu (Data Processing Guide)

> **Module**: `recsys/cf/data/`  
> **Version**: 1.0 (January 2025)  
> **Status**: ✅ Production Ready

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Kiến Trúc Module](#2-kiến-trúc-module)
3. [Cài Đặt & Yêu Cầu](#3-cài-đặt--yêu-cầu)
4. [Hướng Dẫn Sử Dụng](#4-hướng-dẫn-sử-dụng)
5. [Chi Tiết Các Bước Xử Lý](#5-chi-tiết-các-bước-xử-lý)
6. [Output Artifacts](#6-output-artifacts)
7. [Cấu Hình Nâng Cao](#7-cấu-hình-nâng-cao)
8. [Xử Lý Lỗi & Debug](#8-xử-lý-lỗi--debug)
9. [FAQ](#9-faq)

---

## 1. Tổng Quan

### 1.1 Data Processing Pipeline là gì?

Data Processing Pipeline là bộ công cụ xử lý dữ liệu thô (raw data) từ các file CSV thành các định dạng tối ưu cho việc training mô hình Collaborative Filtering (ALS, BPR). Pipeline này giải quyết các thách thức đặc thù của dataset:

| Thách Thức | Giải Pháp |
|------------|-----------|
| **Sparse Data** (~1.23 interactions/user) | Phân loại user thành trainable vs cold-start |
| **Rating Skew** (95% là 5 sao) | AI Sentiment Analysis tạo confidence scores |
| **Vietnamese Text** | ViSoBERT model cho sentiment tiếng Việt |
| **Large Scale** (369K interactions) | Vectorized operations, GPU batch processing |

### 1.2 Input & Output

```
📁 INPUT (data/published_data/)
├── data_reviews_purchase.csv    # 369K reviews
├── data_product.csv             # 2,244 products
├── data_product_attribute.csv   # Product attributes
└── data_shop.csv               # Shop metadata

        ⬇️ Data Processing Pipeline ⬇️

📁 OUTPUT (data/processed/)
├── interactions.parquet         # Processed interactions
├── user_item_mappings.json      # ID mappings
├── X_train_confidence.npz       # ALS training matrix
├── user_pos_train.pkl           # Positive item sets
├── user_hard_neg_train.pkl      # Hard negative sets
├── user_metadata.pkl            # User segmentation
├── item_popularity.npy          # Popularity scores
├── top_k_popular_items.json     # Top-50 popular items
├── data_stats.json              # Statistics
└── versions.json                # Version tracking
```

---

## 2. Kiến Trúc Module

### 2.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       DataProcessor                              │
│  (Main Orchestrator - recsys/cf/data/data.py)                   │
├─────────────────────────────────────────────────────────────────┤
│  + load_and_validate_interactions()                             │
│  + compute_comment_quality()                                     │
│  + segment_users()                                               │
│  + create_id_mappings()                                          │
│  + temporal_split()                                              │
│  + build_confidence_matrix()                                     │
│  + save_all_artifacts()                                          │
│  + create_data_version()                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  DataReader   │   │  DataAuditor  │   │FeatureEngineer│
│  (read_data)  │   │ (audit_data)  │   │(feature_eng)  │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ read_csv()    │   │ validate()    │   │ compute_      │
│ UTF-8 support │   │ deduplicate() │   │ sentiment()   │
│               │   │ detect_       │   │ fake_review   │
│               │   │ outliers()    │   │ detection()   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  UserFilter   │   │   IDMapper    │   │TemporalSplit  │
│(user_filter)  │   │ (id_mapping)  │   │(temporal_split│
├───────────────┤   ├───────────────┤   ├───────────────┤
│ segment_      │   │ create_       │   │ leave_one_out │
│ users()       │   │ mappings()    │   │ vectorized    │
│ trainable vs  │   │ bidirectional │   │ implicit_neg  │
│ cold-start    │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ MatrixBuilder │   │  DataSaver    │   │VersionRegistry│
│(matrix_const) │   │ (data_saver)  │   │(version_reg)  │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ build_csr()   │   │ save_parquet()│   │ create_       │
│ user_sets()   │   │ save_json()   │   │ version()     │
│ hard_negs()   │   │ save_npz()    │   │ compare()     │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 2.2 7-Step Pipeline Flow

```
Step 1: Load & Validate    ──► Step 2: Feature Engineering ──► Step 3: ID Mapping
   │                              │                               │
   ├─ Read CSV (UTF-8)           ├─ AI Sentiment (ViSoBERT)      ├─ user_to_idx
   ├─ Validate ratings           ├─ Fake review detection        ├─ item_to_idx
   ├─ Drop missing timestamps    ├─ Emoji sentiment              └─ Contiguous IDs
   └─ Deduplicate                └─ confidence_score [1-6]
                                          │
                                          ▼
Step 4: Temporal Split    ◄─────────── User Segmentation
   │                                      │
   ├─ Leave-one-out                      ├─ Trainable: ≥2 interactions
   ├─ Positive-only test                 ├─ Cold-start: <2 interactions
   ├─ Vectorized (10-100x faster)        └─ is_trainable_user flag
   └─ Implicit negatives (50/user)
        │
        ▼
Step 5: Matrix Construction ──► Step 6: Save Artifacts ──► Step 7: Versioning
   │                               │                          │
   ├─ X_train_confidence.npz      ├─ Parquet                  ├─ data_hash
   ├─ user_pos_train.pkl          ├─ JSON                     ├─ git_commit
   ├─ user_hard_neg_train.pkl     ├─ NPZ (sparse)             └─ versions.json
   └─ item_popularity.npy         └─ Pickle
```

---

## 3. Cài Đặt & Yêu Cầu

### 3.1 Dependencies

```bash
# Core dependencies
pip install pandas>=1.5.0 numpy>=1.23.0 scipy>=1.9.0 pyarrow>=10.0.0

# AI Sentiment (ViSoBERT)
pip install torch>=1.13.0 transformers>=4.25.0 sentencepiece>=0.1.96

# Optional: GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | None (CPU fallback) | NVIDIA 8GB+ VRAM |
| Storage | 5 GB | 10 GB |
| CPU | 4 cores | 8+ cores |

### 3.3 Verify Installation

```python
# Kiểm tra installation
from recsys.cf.data import DataProcessor

# Kiểm tra GPU (optional)
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 4. Hướng Dẫn Sử Dụng

### 4.1 Quick Start (5 phút)

```python
from recsys.cf.data import DataProcessor

# Khởi tạo processor với config mặc định
processor = DataProcessor(
    base_path="data/published_data",
    output_path="data/processed"
)

# Chạy pipeline đầy đủ
df_clean, _ = processor.load_and_validate_interactions()
df_enriched, _ = processor.compute_comment_quality(df_clean)
df_segmented, _ = processor.segment_users(df_enriched)
df_mapped, mappings = processor.create_id_mappings(df_segmented)
df_split, stats = processor.temporal_split(df_mapped)

# Build matrices và save
# ... (xem Full Pipeline Example bên dưới)
```

### 4.2 Full Pipeline Example

```python
from recsys.cf.data import DataProcessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# KHỞI TẠO PROCESSOR
# ═══════════════════════════════════════════════════════════════
processor = DataProcessor(
    # Paths
    base_path="data/published_data",
    output_path="data/processed",
    
    # Thresholds
    positive_threshold=4.0,      # rating >= 4 → positive
    hard_negative_threshold=3.0, # rating <= 3 → hard negative
    
    # Comment quality
    no_comment_quality=0.5,      # Default cho missing comments
    
    # User filtering
    min_user_interactions=2,     # Minimum để trainable
    min_user_positives=1,        # Ít nhất 1 positive
)

# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD & VALIDATE
# ═══════════════════════════════════════════════════════════════
logger.info("Step 1: Loading and validating data...")
df_clean, quality_report = processor.load_and_validate_interactions()

print(f"""
📊 Data Quality Report:
   - Total rows: {quality_report['total_rows']:,}
   - Valid rows: {quality_report['valid_rows']:,}
   - Duplicates removed: {quality_report.get('duplicates_removed', 0):,}
   - Invalid ratings: {quality_report.get('invalid_ratings', 0):,}
""")

# ═══════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING (AI Sentiment)
# ═══════════════════════════════════════════════════════════════
logger.info("Step 2: Computing comment quality with AI sentiment...")
df_enriched, quality_stats = processor.compute_comment_quality(
    df_clean,
    comment_column='processed_comment'
)

print(f"""
🤖 AI Sentiment Analysis:
   - Mean quality: {quality_stats['mean_quality']:.3f}
   - Std quality: {quality_stats['std_quality']:.3f}
   - Range: [{quality_stats['min_quality']:.2f}, {quality_stats['max_quality']:.2f}]
""")

# ═══════════════════════════════════════════════════════════════
# STEP 2.3: USER SEGMENTATION
# ═══════════════════════════════════════════════════════════════
logger.info("Step 2.3: Segmenting users (trainable vs cold-start)...")
df_segmented, segment_stats = processor.segment_users(df_enriched)

print(f"""
👥 User Segmentation:
   - Trainable users: {segment_stats['trainable_count']:,} ({segment_stats['trainable_pct']:.1f}%)
   - Cold-start users: {segment_stats['cold_start_count']:,} ({segment_stats['cold_start_pct']:.1f}%)
""")

# ═══════════════════════════════════════════════════════════════
# STEP 3: ID MAPPING
# ═══════════════════════════════════════════════════════════════
logger.info("Step 3: Creating ID mappings...")
df_mapped, mappings = processor.create_id_mappings(df_segmented)

num_users = len(mappings['user_to_idx'])
num_items = len(mappings['item_to_idx'])
print(f"🔢 Mappings: {num_users:,} users × {num_items:,} items")

# ═══════════════════════════════════════════════════════════════
# STEP 4: TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════
logger.info("Step 4: Performing temporal split...")
df_split, split_stats = processor.temporal_split(
    df_mapped,
    method='leave_one_out',
    use_validation=False
)

print(f"""
📅 Temporal Split:
   - Train: {split_stats['train_size']:,} interactions
   - Test: {split_stats['test_size']:,} interactions
   - Sparsity: {split_stats.get('sparsity', 0):.4f}
""")

# ═══════════════════════════════════════════════════════════════
# STEP 5: MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
logger.info("Step 5: Building matrices...")

# Filter train data
df_train = df_split[df_split['split'] == 'train']

# Build CSR matrix for ALS (confidence scores)
X_confidence = processor.build_confidence_matrix(
    df_train, num_users, num_items,
    value_col='confidence_score'
)

# Build user positive sets
user_pos_sets = processor.build_user_positive_sets(df_train)

# Build item popularity
item_popularity = processor.build_item_popularity(df_train, num_items)

# Get top-K popular items
top_k_popular = processor.get_top_k_popular_items(df_train, k=50)

# Build hard negative sets
user_hard_neg_sets = processor.build_user_hard_negative_sets(
    df_train, top_k_popular
)

# Build user metadata
user_metadata = processor.build_user_metadata(df_split)

print(f"""
📦 Matrices Built:
   - X_confidence: {X_confidence.shape} (nnz: {X_confidence.nnz:,})
   - user_pos_sets: {len(user_pos_sets):,} users
   - user_hard_neg_sets: {len(user_hard_neg_sets):,} users
   - item_popularity: {len(item_popularity):,} items
""")

# ═══════════════════════════════════════════════════════════════
# STEP 6: SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════
logger.info("Step 6: Saving artifacts...")

# Save all at once
saved_files = processor.save_all_artifacts(
    interactions_df=df_split,
    mappings=mappings,
    X_confidence=X_confidence,
    user_pos_sets=user_pos_sets,
    user_hard_neg_sets=user_hard_neg_sets,
    user_metadata=user_metadata,
    item_popularity=item_popularity,
    top_k_popular=top_k_popular,
    stats=split_stats
)

print(f"💾 Saved {len(saved_files)} artifacts")

# ═══════════════════════════════════════════════════════════════
# STEP 7: VERSIONING
# ═══════════════════════════════════════════════════════════════
logger.info("Step 7: Creating data version...")

data_hash = processor.compute_data_hash()
version_id = processor.create_data_version(
    data_hash=data_hash,
    filters={
        'min_user_interactions': 2,
        'min_user_positives': 1,
        'positive_threshold': 4.0
    },
    files=list(saved_files.keys()),
    stats=split_stats
)

print(f"📌 Version created: {version_id}")
print("\n✅ Pipeline completed successfully!")
```

### 4.3 Chạy từ Script

```powershell
# Chạy pipeline hoàn chỉnh
python scripts/run_task01_complete.py

# Hoặc với custom config
python scripts/run_task01_complete.py --config config/data_config.yaml
```

---

## 5. Chi Tiết Các Bước Xử Lý

### 5.1 Step 1: Data Validation

**Mục đích**: Đảm bảo data quality trước khi xử lý

**Các kiểm tra thực hiện**:

| Kiểm tra | Hành động | Lý do |
|----------|-----------|-------|
| Missing `user_id`, `product_id` | Drop row | Cannot process without IDs |
| Missing `rating` | Drop row | Core feature required |
| Missing `cmt_date` (NaT) | Drop row | Tránh data leakage |
| Rating ngoài [1.0, 5.0] | Drop row | Invalid data |
| Duplicate (user, product) | Keep latest | Lấy interaction mới nhất |

**Code Example**:
```python
from recsys.cf.data import DataAuditor

auditor = DataAuditor(rating_min=1.0, rating_max=5.0)
df_valid, report = auditor.validate(df_raw)
df_dedup, dedup_stats = auditor.deduplicate(df_valid, strategy='keep_latest')
outliers = auditor.detect_outliers(df_dedup)
```

### 5.2 Step 2: Feature Engineering (AI Sentiment)

**Mục đích**: Tạo `comment_quality` và `confidence_score` để phân biệt quality của reviews

**Model**: `5CD-AI/Vietnamese-Sentiment-visobert`
- Trained trên 120K Vietnamese e-commerce reviews
- 3 classes: NEGATIVE, POSITIVE, NEUTRAL
- GPU batch processing (batch_size=64)

**Fake Review Detection Heuristics**:

| Heuristic | Bonus/Penalty | Điều kiện |
|-----------|---------------|-----------|
| Long review | +0.1 | >25 words |
| Short review | -0.1 | <4 words |
| Positive keywords | +0.15 | "thấm nhanh", "hiệu quả"... |
| Negative keywords | -0.15 | "kém", "dở", "fake"... |
| Positive emojis | +0.1 | 😍❤️👍✨🌟💯🔥 |
| Negative emojis | -0.1 | 😢😭💔👎😡 |
| Rating-sentiment mismatch | -0.2 | High rating + negative text |
| Repetition (spam) | -0.15 | Low character diversity |

**Output Columns**:
- `comment_quality`: [0.0, 1.0] - Chất lượng review
- `confidence_score`: [1.0, 6.0] - rating + comment_quality

**Code Example**:
```python
from recsys.cf.data import FeatureEngineer

engineer = FeatureEngineer(
    model_name="5CD-AI/Vietnamese-Sentiment-visobert",
    batch_size=64,
    no_comment_quality=0.5,
    enable_fake_review_checks=True
)

df_enriched, stats = engineer.compute_confidence_scores(
    df_clean, 
    comment_column='processed_comment'
)
```

### 5.3 Step 2.3: User Segmentation

**Mục đích**: Phân loại users để quyết định serving strategy

**Tiêu chí phân loại**:

| Loại | Điều kiện | % Users | Serving Strategy |
|------|-----------|---------|------------------|
| **Trainable** | ≥2 interactions AND ≥1 positive | ~8.6% | CF (ALS/BPR) |
| **Cold-start** | <2 interactions OR 0 positives | ~91.4% | Content-based + Popularity |

**Special Case**: User có đúng 2 interactions nhưng cả 2 đều negative (rating <4) → Force cold-start

**Code Example**:
```python
from recsys.cf.data import UserFilter

user_filter = UserFilter(
    min_interactions=2,
    min_positives=1,
    positive_threshold=4.0
)

df_segmented, stats = user_filter.segment_users(df_enriched)
# df_segmented có cột 'is_trainable_user' (True/False)
```

### 5.4 Step 3: ID Mapping

**Mục đích**: Chuyển đổi original IDs sang contiguous indices cho sparse matrix

**Tại sao cần mapping?**:
- Original `user_id`: sparse (gaps, range 1-304708)
- Matrix cần: contiguous (0 to num_users-1)
- Mapping cho phép O(1) lookup

**Output Structure**:
```json
{
  "user_to_idx": {"12345": 0, "67890": 1, ...},
  "idx_to_user": {"0": "12345", "1": "67890", ...},
  "item_to_idx": {"101": 0, "102": 1, ...},
  "idx_to_item": {"0": "101", "1": "102", ...}
}
```

### 5.5 Step 4: Temporal Split

**Mục đích**: Chia train/test theo thời gian, đảm bảo no data leakage

**Method**: Leave-One-Out
- **Test**: Latest POSITIVE interaction per user
- **Train**: Tất cả interactions còn lại
- **Validation** (optional): 2nd latest positive

**Key Features**:
- **Positive-only test**: Chỉ đo khả năng recommend items user thích
- **Vectorized**: 10-100x faster than iterative approach
- **Implicit negatives**: Sample 50 popular items user chưa interact

**Code Example**:
```python
from recsys.cf.data import TemporalSplitter

splitter = TemporalSplitter(
    positive_threshold=4.0,
    implicit_negative_per_user=50,
    implicit_negative_strategy='popular'
)

df_split, stats = splitter.split(df_mapped, method='leave_one_out')
# df_split có cột 'split' = 'train' | 'test' | 'val'
```

### 5.6 Step 5: Matrix Construction

**Mục đích**: Build sparse matrices và auxiliary structures cho training

**Output Artifacts**:

| File | Type | Shape | Usage |
|------|------|-------|-------|
| `X_train_confidence.npz` | CSR matrix | (users, items) | ALS training |
| `user_pos_train.pkl` | Dict[int, Set] | - | Negative sampling |
| `user_hard_neg_train.pkl` | Dict[int, Dict] | - | Hard negative mining |
| `item_popularity.npy` | ndarray | (items,) | Popularity baseline |

**Code Example**:
```python
from recsys.cf.data import MatrixBuilder

builder = MatrixBuilder()

X_conf = builder.build_confidence_matrix(df_train, num_users, num_items)
user_pos = builder.build_user_positive_sets(df_train)
top_k = builder.get_top_k_popular_items(df_train, k=50)
user_neg = builder.build_user_hard_negative_sets(df_train, top_k)
```

### 5.7 Step 6 & 7: Save & Version

**Formats Used**:
- **Parquet**: Interactions (10x faster, 50% smaller)
- **JSON**: Mappings, stats (human-readable)
- **NPZ**: Sparse matrices (scipy format)
- **Pickle**: Python objects (sets, dicts)

**Versioning**:
- Mỗi version có `data_hash` (MD5 của raw CSVs)
- Track `git_commit` cho reproducibility
- `is_stale()` check để trigger retraining

---

## 6. Output Artifacts

### 6.1 File Summary

| File | Size (approx) | Format | Description |
|------|---------------|--------|-------------|
| `interactions.parquet` | ~50 MB | Parquet | Full processed data |
| `user_item_mappings.json` | ~5 MB | JSON | ID mappings |
| `X_train_confidence.npz` | ~10 MB | NPZ | ALS matrix |
| `user_pos_train.pkl` | ~2 MB | Pickle | Positive sets |
| `user_hard_neg_train.pkl` | ~5 MB | Pickle | Hard negatives |
| `user_metadata.pkl` | ~1 MB | Pickle | User segmentation |
| `item_popularity.npy` | ~20 KB | NumPy | Popularity scores |
| `top_k_popular_items.json` | ~1 KB | JSON | Top-50 items |
| `data_stats.json` | ~5 KB | JSON | Statistics |
| `versions.json` | ~2 KB | JSON | Version history |

### 6.2 Loading Artifacts

```python
import pandas as pd
import numpy as np
import json
import pickle
from scipy.sparse import load_npz

# Load interactions
df = pd.read_parquet("data/processed/interactions.parquet")

# Load mappings
with open("data/processed/user_item_mappings.json") as f:
    mappings = json.load(f)

# Load confidence matrix
X_conf = load_npz("data/processed/X_train_confidence.npz")

# Load user sets
with open("data/processed/user_pos_train.pkl", "rb") as f:
    user_pos = pickle.load(f)

# Load popularity
popularity = np.load("data/processed/item_popularity.npy")
```

---

## 7. Cấu Hình Nâng Cao

### 7.1 Full Configuration Options

```python
processor = DataProcessor(
    # ═══ Data Paths ═══
    base_path="data/published_data",
    output_path="data/processed",
    
    # ═══ Validation ═══
    rating_min=1.0,
    rating_max=5.0,
    drop_missing_timestamps=True,  # CRITICAL: avoid data leakage
    
    # ═══ Thresholds ═══
    positive_threshold=4.0,      # rating >= 4 → positive
    hard_negative_threshold=3.0, # rating <= 3 → hard negative
    
    # ═══ Comment Quality (FeatureEngineer) ═══
    no_comment_quality=0.5,      # Default for missing comments
    sentiment_model="5CD-AI/Vietnamese-Sentiment-visobert",
    batch_size=64,               # GPU batch size
    enable_fake_review_checks=True,
    
    # ═══ User Filtering ═══
    min_user_interactions=2,     # Min total interactions
    min_user_positives=1,        # Min positive interactions
    min_item_positives=5,        # Min positives per item
    
    # ═══ Temporal Split ═══
    include_negative_holdout=True,
    implicit_negative_per_user=50,
    implicit_negative_strategy='popular',  # or 'random'
    
    # ═══ Versioning ═══
    versions_file="versions.json",
    max_versions_kept=10
)
```

### 7.2 Disable AI Sentiment (CPU-only mode)

```python
# Nếu không có GPU, có thể disable AI sentiment
processor = DataProcessor(
    # ... other config
    sentiment_model=None,  # Disable ViSoBERT
    enable_fake_review_checks=True  # Still use heuristics
)
```

### 7.3 Custom Keyword Dictionaries

```python
from recsys.cf.data import FeatureEngineer

engineer = FeatureEngineer(
    # Custom positive keywords
    positive_keywords={
        'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 'sáng da',
        'tốt', 'ưng', 'recommend', 'mua lại', 'hàng auth'
    },
    # Custom negative keywords
    negative_keywords={
        'kém', 'dở', 'thất vọng', 'fake', 'giả', 'tệ',
        'không hiệu quả', 'hàng nhái', 'lừa đảo'
    },
    # Custom emoji mappings
    positive_emojis={'😍', '❤️', '👍', '✨', '🌟', '💯'},
    negative_emojis={'😢', '😭', '💔', '👎', '😡'}
)
```

---

## 8. Xử Lý Lỗi & Debug

### 8.1 Common Errors

| Error | Nguyên nhân | Giải pháp |
|-------|-------------|-----------|
| `UnicodeDecodeError` | CSV không phải UTF-8 | Ensure `encoding='utf-8'` |
| `KeyError: 'processed_comment'` | Thiếu column | Check CSV schema |
| `CUDA out of memory` | GPU VRAM không đủ | Giảm `batch_size` |
| `Empty DataFrame after filtering` | Threshold quá strict | Giảm `min_user_interactions` |
| `Matrix shape mismatch` | ID mapping sai | Verify mappings |

### 8.2 Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cf/data_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Processor sẽ log chi tiết từng step
processor = DataProcessor(...)
```

### 8.3 Validation Checks

```python
# Sau khi chạy pipeline, validate outputs
def validate_outputs(output_path="data/processed"):
    import pandas as pd
    from scipy.sparse import load_npz
    import json
    
    # 1. Check interactions
    df = pd.read_parquet(f"{output_path}/interactions.parquet")
    assert df['rating'].between(1, 5).all(), "Invalid ratings!"
    assert df['confidence_score'].between(1, 6).all(), "Invalid confidence!"
    assert not df['cmt_date'].isna().any(), "NaT timestamps found!"
    
    # 2. Check mappings alignment
    with open(f"{output_path}/user_item_mappings.json") as f:
        mappings = json.load(f)
    
    X = load_npz(f"{output_path}/X_train_confidence.npz")
    assert X.shape[0] <= len(mappings['user_to_idx']), "User mismatch!"
    assert X.shape[1] == len(mappings['item_to_idx']), "Item mismatch!"
    
    # 3. Check test set is positive-only
    df_test = df[df['split'] == 'test']
    assert (df_test['rating'] >= 4).all(), "Test contains negatives!"
    
    print("✅ All validations passed!")

validate_outputs()
```

---

## 9. FAQ

### Q1: Tại sao không dùng rating trực tiếp làm confidence?

**A**: Vì 95% ratings là 5 sao, không có discriminative power. Confidence score = rating + comment_quality cho phép phân biệt:
- 5 sao + review chi tiết → confidence 5.8-6.0 (genuine)
- 5 sao + review ngắn/spam → confidence 5.0-5.3 (suspicious)

### Q2: Tại sao threshold trainable là ≥2 interactions?

**A**: Trade-off giữa data hunger và statistical viability:
- ≥3: Chỉ ~15K users (~5%), quá ít
- ≥2: ~26K users (~8.6%), đủ lớn với BERT initialization
- ≥1: Tất cả users nhưng không có collaborative signal

### Q3: Cold-start users (91%) được serve như thế nào?

**A**: Content-based + Popularity:
1. PhoBERT item-item similarity (nếu có lịch sử)
2. Popularity baseline (Top-50 popular items)
3. Hybrid reranking với weights: content=0.6, popularity=0.3, quality=0.1

### Q4: Implicit negatives dùng làm gì?

**A**: Để đánh giá model công bằng hơn:
- Test chỉ có 1 positive per user
- Sample 50 popular items user chưa mua làm negatives
- Tính Recall@K, NDCG@K trên set này

### Q5: Làm sao biết khi nào cần retrain?

**A**: Dùng VersionRegistry:
```python
if processor.is_data_version_stale(current_version, max_age_hours=168):  # 1 week
    print("Data is stale, trigger retraining!")
```

Hoặc monitor drift metrics trong `data_stats.json`.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial release with all 7 steps |

---

## Tài Liệu Liên Quan

- [Task 01: Data Layer Specification](../tasks/01_data_layer.md)
- [Task 02: ALS/BPR Training](../tasks/02_cf_training.md)
- [Task 05: Serving Layer](../tasks/05_serving.md)
- [API Reference](./API_REFERENCE.md)

---

*Last updated: January 2025*

# Task 02: Training Pipelines (ALS & BPR)

> **Status**: ✅ **FULLY IMPLEMENTED** (January 2025)  
> **Last Updated**: 2025-01-16

## Mục Tiêu

Xây dựng hai pipelines huấn luyện Collaborative Filtering song song: ALS (Alternating Least Squares) và BPR (Bayesian Personalized Ranking). Mỗi pipeline sẽ train, evaluate, và persist artifacts độc lập, sau đó được so sánh để chọn best model.

---

## 📦 Implementation Summary

### Module Locations
```
recsys/cf/model/
├── als/                      # ALS 7-step pipeline
│   ├── __init__.py          # Package exports với convenience functions
│   ├── pre_data.py          # Step 1: ALSMatrixPreparer
│   ├── model_init.py        # Step 2: ALSModelInitializer (5 presets)
│   ├── trainer.py           # Step 3: ALSTrainer với checkpointing
│   ├── embeddings.py        # Step 4: EmbeddingExtractor
│   ├── recommender.py       # Step 5: ALSRecommender
│   ├── evaluation.py        # Step 6: ALSEvaluator, PopularityBaseline
│   └── artifact_saver.py    # Step 7: save_als_complete, ScoreRange
├── bpr/                      # BPR 7-step pipeline với advanced features
│   ├── __init__.py          # Package exports (basic + advanced)
│   ├── pre_data.py          # Step 1: BPRDataLoader
│   ├── sampler.py           # Step 2: TripletSampler, HardNegativeMixer
│   ├── model_init.py        # Step 3: BPRModelInitializer
│   ├── trainer.py           # Step 4: BPRTrainer với SGD
│   ├── advanced_sampler.py  # Advanced: ContextualNegativeSampler
│   ├── advanced_trainer.py  # Advanced: AdvancedBPRTrainer, AdamW
│   └── artifact_saver.py    # Step 7: save_bpr_complete, BPRArtifacts
└── bert_enhanced_als.py      # BERT-enhanced ALS wrapper
```

### Key Classes & Quick Usage
```python
# === ALS Pipeline ===
from recsys.cf.model.als import (
    ALSMatrixPreparer,       # Step 1: Load & prepare matrices
    ALSModelInitializer,     # Step 2: Initialize model with presets
    ALSTrainer,              # Step 3: Train with checkpointing
    EmbeddingExtractor,      # Step 4: Extract U, V matrices
    ALSRecommender,          # Step 5: Generate recommendations
    ALSEvaluator,            # Step 6: Evaluate with metrics
    save_als_complete,       # Step 7: Save all artifacts
    ScoreRange               # Score stats for Task 08
)

# === BPR Pipeline (Basic) ===
from recsys.cf.model.bpr import (
    BPRDataLoader,           # Step 1: Load positive pairs + hard negs
    TripletSampler,          # Step 2: Sample (u, i+, j-) triplets
    HardNegativeMixer,       # Step 2: 30% hard + 70% random mixing
    BPRTrainer,              # Step 4: SGD training with early stopping
    save_bpr_complete        # Step 7: Save all artifacts
)

# === BPR Pipeline (Advanced) ===
from recsys.cf.model.bpr import (
    AdvancedTripletSampler,      # Contextual + sentiment-aware sampling
    ContextualNegativeSampler,   # BERT similarity-based negatives
    SentimentContrastSampler,    # Sentiment-contrasted negatives
    AdvancedBPRTrainer,          # AdamW, dropout, LR scheduling
    OptimizerConfig,             # Configure optimizer (SGD/AdamW)
    TrainingConfig,              # Configure training loop
    EmbeddingDropout             # Regularization via dropout
)
```

---

## 🔄 Updated Training Strategy (January 2025)

### Data Context: High Sparsity + Rating Skew
- **Sparsity**: ~1.23 interactions/user → Most users are one-time buyers
- **Rating Skew**: ~95% are 5-star → Loss of discriminative power
- **Trainable Users**: ≥2 interactions (~26K users, 8.6% of total)
- **Challenge**: Traditional CF struggles with minimal user overlap

### Key Strategic Decisions:

1. **User Segmentation** (≥2 threshold)
   - Train CF only on **trainable users** (≥2 interactions, ≥1 positive)
   - Skip cold-start users (<2 interactions) → serve with content-based
   - Result: ~26K users trainable, ~274K users cold-start

2. **ALS: Sentiment-Enhanced Confidence**
   - Input: Confidence matrix (`rating + comment_quality`, range 1.0-6.0)
   - Distinguishes "genuine 5-star" from "bare 5-star"
   - Lower alpha scaling (5-10) due to higher confidence range
   - **5 Presets**: default, normalized, high_quality, fast, sparse_data

3. **BPR: Dual Hard Negative Mining**
   - Explicit: Low ratings (≤3) when available
   - Implicit: Top-50 popular items user didn't buy
   - Sampling: 30% hard negatives (merged explicit+implicit) + 70% random unseen
   - **Advanced**: Contextual negatives (BERT similarity), sentiment-contrasted

4. **Test Set: Trainable Users Only**
   - Evaluate CF models only on trainable users
   - Cold-start users evaluated separately on content-based metrics
   - Fair comparison of CF effectiveness

---

## Pipeline Overview

```
Preprocessed Data (từ Task 01)
    ↓
├─ ALS Pipeline ─────────────────┐
│  Step 1: ALSMatrixPreparer     │
│  Step 2: ALSModelInitializer   │
│  Step 3: ALSTrainer            │
│  Step 4: EmbeddingExtractor    │
│  Step 5: ALSRecommender        │
│  Step 6: ALSEvaluator          │
│  Step 7: save_als_complete     │
│                                │
└─ BPR Pipeline ─────────────────┤
   Step 1: BPRDataLoader         │
   Step 2: TripletSampler        │
   Step 3: BPRModelInitializer   │
   Step 4: BPRTrainer            │
   Step 5: (Recommender)         │
   Step 6: (Evaluator)           │
   Step 7: save_bpr_complete     │
                                 ↓
              Compare Metrics (NDCG@10)
                                 ↓
              Update Registry (best model)
```

---

## ALS Pipeline

### Step 1: Matrix Preparation

#### Module: `recsys/cf/model/als/pre_data.py`

#### Class: `ALSMatrixPreparer`

Loads processed data from Task 01 and prepares for ALS training.

```python
@dataclass
class ALSPreparedData:
    """Container for all prepared ALS data."""
    X_train_implicit: csr_matrix    # Transposed (items × users) for implicit lib
    X_train_confidence: csr_matrix  # Original (users × items)
    X_train_binary: csr_matrix      # Binary preference matrix
    mappings: Dict[str, Any]        # ID mappings
    user_pos_train: Dict[int, Set[int]]  # User positive sets
    metadata: Dict[str, Any]        # Data stats
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `load_processed_data()` | Load all Task 01 outputs (NPZ, PKL, JSON) |
| `prepare_confidence_matrix()` | Get CSR matrix with confidence scores |
| `derive_binary_preference(threshold)` | Create binary P matrix |
| `prepare_for_implicit()` | Transpose to item-user format |
| `get_alpha_recommendations()` | Suggest alpha based on confidence range |
| `prepare_complete_als_data()` | **One-line orchestrator** |

**Usage:**
```python
from recsys.cf.model.als import ALSMatrixPreparer

preparer = ALSMatrixPreparer(base_path='data/processed')
data = preparer.prepare_complete_als_data()

X_train = data.X_train_implicit    # Ready for implicit library
mappings = data.mappings
user_pos_train = data.user_pos_train

# Check recommended alpha
alpha = preparer.get_alpha_recommendations()
# Returns: {'recommended': 10, 'range': (5, 15), 'reason': 'confidence 1.0-6.0'}
```

#### Confidence Scaling Strategy
- **Raw Range**: [1.0, 6.0] → alpha = 5-10 (lower due to higher range)
- **Normalized Range**: [0.0, 1.0] → alpha = 20-40 (standard scaling)
- **Logic**: `C[u,i] = 1 + alpha * confidence_score[u,i]`

---

### Step 2: Model Initialization

#### Module: `recsys/cf/model/als/model_init.py`

#### Class: `ALSModelInitializer`

Initialize ALS model with presets or custom configuration.

**Available Presets:**

| Preset | factors | regularization | alpha | Description |
|--------|---------|----------------|-------|-------------|
| `default` | 64 | 0.01 | 10 | For sentiment-enhanced confidence (1-6) |
| `normalized` | 64 | 0.01 | 40 | For normalized confidence (0-1) |
| `high_quality` | 128 | 0.05 | 10 | More expressive embeddings |
| `fast` | 32 | 0.01 | 10 | Quick training, less accurate |
| `sparse_data` | 64 | 0.10 | 5 | For ≥2 threshold, high sparsity |

**Usage:**
```python
from recsys.cf.model.als import ALSModelInitializer

# Method 1: Using preset
initializer = ALSModelInitializer(preset='sparse_data')
model = initializer.initialize_model()

# Method 2: Custom configuration
initializer = ALSModelInitializer(config={
    'factors': 64,
    'regularization': 0.1,
    'iterations': 15,
    'alpha': 10,
    'random_state': 42,
    'use_gpu': False
})
model = initializer.initialize_model()

# Method 3: Data-driven recommendations
recommended = initializer.recommend_config_for_data(
    num_users=26000,
    num_items=2231,
    nnz=65000,
    confidence_range=(1.0, 6.0)
)
# Returns: {'factors': 64, 'regularization': 0.1, 'alpha': 10, ...}
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `initialize_model()` | Create implicit.ALS model |
| `get_alpha_recommendation(confidence_range)` | Suggest alpha value |
| `recommend_config_for_data(num_users, num_items, nnz)` | Full config recommendation |
| `update_config(**kwargs)` | Update configuration in-place |

---

### Step 3: Training

#### Module: `recsys/cf/model/als/trainer.py`

#### Class: `ALSTrainer`

Train ALS model with checkpointing and progress tracking.

```python
@dataclass
class TrainingResult:
    """Container for training results."""
    training_time: float
    iterations: int
    loss_history: List[float]       # If available
    memory_usage: Dict[str, float]  # Peak memory stats
    checkpoint_paths: List[Path]    # Saved checkpoints
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `fit(X_train)` | Train model on transposed CSR matrix |
| `get_embeddings()` | Extract U, V after training |
| `save_embeddings(output_dir)` | Save U.npy, V.npy |
| `resume_from_checkpoint(path)` | Resume interrupted training |

**Usage:**
```python
from recsys.cf.model.als import ALSTrainer

trainer = ALSTrainer(
    model=model,
    checkpoint_dir=Path('checkpoints/als'),
    checkpoint_interval=5,   # Save every 5 iterations
    track_memory=True,
    enable_validation=False
)

# Train
result = trainer.fit(X_train_implicit)

print(f"Training time: {result.training_time:.1f}s")
print(f"Peak memory: {result.memory_usage['peak_mb']:.1f} MB")

# Get embeddings
U, V = trainer.get_embeddings()
```

**Features:**
- Progress bar với tqdm
- Memory usage monitoring
- Checkpointing every N iterations
- Optional validation callback

---

### Step 4: Extract Embeddings

#### Module: `recsys/cf/model/als/embeddings.py`

#### Class: `EmbeddingExtractor`

Extract and optionally normalize embeddings.

**Usage:**
```python
from recsys.cf.model.als import EmbeddingExtractor, extract_embeddings

# Method 1: Using class
extractor = EmbeddingExtractor(model, normalize=True)
U, V = extractor.get_embeddings()

print(f"U shape: {U.shape}")  # (num_users, factors)
print(f"V shape: {V.shape}")  # (num_items, factors)
print(f"Normalized: {extractor.is_normalized}")

# Check quality
quality = extractor.compute_embedding_quality_score()
print(f"User embedding variance: {quality['user_variance']:.4f}")
print(f"Item embedding variance: {quality['item_variance']:.4f}")

# Method 2: Quick function
U, V = extract_embeddings(model, normalize=True)
```

**Normalization:**
- L2 normalize rows for cosine similarity
- Useful when combining with content-based embeddings

---

### Step 5: Recommendation Generation

#### Module: `recsys/cf/model/als/recommender.py`

#### Class: `ALSRecommender`

Generate recommendations with automatic ID mapping and seen item filtering.

```python
@dataclass
class RecommendationResult:
    """Single user recommendation result."""
    user_id: Any                # Original user_id
    user_idx: int               # Internal u_idx
    item_ids: List[Any]         # Recommended product_ids
    item_indices: List[int]     # Internal i_idx values
    scores: List[float]         # Recommendation scores
    filtered_count: int         # Number of seen items filtered
```

**Usage:**
```python
from recsys.cf.model.als import ALSRecommender

recommender = ALSRecommender(
    user_factors=U,
    item_factors=V,
    user_to_idx=mappings['user_to_idx'],
    idx_to_user=mappings['idx_to_user'],
    item_to_idx=mappings['item_to_idx'],
    idx_to_item=mappings['idx_to_item'],
    user_pos_train=user_pos_train
)

# Single user
result = recommender.recommend(user_id='user_12345', k=10, filter_seen=True)
print(f"Top items: {result.item_ids[:5]}")
print(f"Scores: {result.scores[:5]}")
print(f"Filtered {result.filtered_count} seen items")

# Batch recommendations (efficient)
results = recommender.recommend_batch(user_ids=test_users, k=10)
for r in results[:3]:
    print(f"User {r.user_id}: {r.item_ids[:3]}")
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `recommend(user_id, k, filter_seen)` | Single user recommendations |
| `recommend_batch(user_ids, k)` | Batch recommendations (vectorized) |
| `recommend_for_idx(u_idx, k)` | Using internal index directly |
| `get_user_scores(user_id)` | Get all item scores for user |

---

### Step 6: Evaluation

#### Module: `recsys/cf/model/als/evaluation.py`

#### Classes: `ALSEvaluator`, `PopularityBaseline`

Evaluate model with Recall@K, NDCG@K and compare to popularity baseline.

```python
@dataclass
class EvaluationResult:
    """Evaluation results container."""
    metrics: Dict[str, float]           # {'recall@10': 0.234, ...}
    baseline_metrics: Dict[str, float]  # Popularity baseline
    improvement: Dict[str, str]         # {'recall@10': '+45.2%', ...}
    per_user_metrics: pd.DataFrame      # Per-user breakdown (optional)
```

**Usage:**
```python
from recsys.cf.model.als import ALSEvaluator, PopularityBaseline

# Create evaluator
evaluator = ALSEvaluator(
    user_factors=U,
    item_factors=V,
    user_to_idx=mappings['user_to_idx'],
    idx_to_user=mappings['idx_to_user'],
    item_to_idx=mappings['item_to_idx'],
    idx_to_item=mappings['idx_to_item'],
    user_pos_train=user_pos_train,
    user_pos_test=user_pos_test
)

# Evaluate with baseline comparison
results = evaluator.evaluate(k_values=[10, 20], compare_baseline=True)

# Print summary
results.print_summary()
# Output:
# === ALS Evaluation Results ===
# Recall@10: 0.234 (baseline: 0.145, +61.4%)
# NDCG@10:   0.189 (baseline: 0.102, +85.3%)
# Recall@20: 0.312 (baseline: 0.198, +57.6%)

# Access metrics
print(f"Recall@10: {results.metrics['recall@10']:.3f}")
print(f"Improvement: {results.improvement['recall@10']}")
```

**Metrics:**
- **Recall@K**: % of test items in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Popularity Baseline**: Top-K popular items (by `num_sold_time`)

---

### Step 7: Save Artifacts

#### Module: `recsys/cf/model/als/artifact_saver.py`

#### Key Components:

```python
@dataclass
class ScoreRange:
    """CF score statistics for Task 08 normalization."""
    method: str      # 'validation_set' or 'random_sample'
    min: float
    max: float
    mean: float
    std: float
    p01: float       # 1st percentile (robust min)
    p99: float       # 99th percentile (robust max)
    num_samples: int

@dataclass
class ALSArtifacts:
    """Container for all saved artifacts."""
    model_type: str
    output_dir: Path
    embeddings_path: Path
    params_path: Path
    metrics_path: Path
    metadata_path: Path
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]  # Includes score_range
```

**Main Function: `save_als_complete()`**

One-line orchestrator to save all artifacts with score range computation.

**Usage:**
```python
from recsys.cf.model.als import save_als_complete, compute_score_range

# Complete save with score range
artifacts = save_als_complete(
    user_embeddings=U,
    item_embeddings=V,
    params={
        'factors': 64,
        'regularization': 0.1,
        'iterations': 15,
        'alpha': 10,
        'random_seed': 42
    },
    metrics={
        'recall@10': 0.234,
        'ndcg@10': 0.189,
        'recall@20': 0.312,
        'ndcg@20': 0.221
    },
    validation_user_indices=[10, 25, 42, ...],  # For score range
    user_pos_train=user_pos_train,              # Exclude seen items
    data_version_hash='abc123def456',
    output_dir='artifacts/cf/als'
)

print(artifacts.summary())
# Files saved:
#   artifacts/cf/als/als_U.npy
#   artifacts/cf/als/als_V.npy
#   artifacts/cf/als/als_params.json
#   artifacts/cf/als/als_metrics.json
#   artifacts/cf/als/als_metadata.json

# Score range for Task 08
score_range = artifacts.metadata['score_range']
print(f"Score range: [{score_range['p01']:.3f}, {score_range['p99']:.3f}]")
```

**Saved Files:**

| File | Content |
|------|---------|
| `als_U.npy` | User embeddings (num_users, factors) |
| `als_V.npy` | Item embeddings (num_items, factors) |
| `als_params.json` | Training hyperparameters |
| `als_metrics.json` | Evaluation metrics |
| `als_metadata.json` | Metadata with score_range |
| `als_model.pkl` | Serialized model (optional) |

**Metadata Structure:**
```json
{
  "timestamp": "2025-01-16T10:30:00",
  "data_version_hash": "abc123def456",
  "git_commit": "a1b2c3d4",
  "system_info": {
    "platform": "Windows-10",
    "python_version": "3.10.12",
    "memory_gb": 32.0
  },
  "score_range": {
    "method": "validation_set",
    "min": 0.0,
    "max": 1.48,
    "mean": 0.32,
    "std": 0.21,
    "p01": 0.01,
    "p99": 1.12,
    "num_samples": 50000
  }
}
```

---

## BPR Pipeline

### Step 1: Data Preparation

#### Module: `recsys/cf/model/bpr/pre_data.py`

#### Class: `BPRDataLoader`

Load positive pairs, user sets, and hard negatives from Task 01 outputs.

```python
@dataclass
class BPRData:
    """Container for BPR training data."""
    positive_pairs: np.ndarray      # Shape (N, 2) with [u_idx, i_idx]
    user_pos_sets: Dict[int, Set[int]]
    hard_neg_sets: Dict[int, Set[int]]  # Merged explicit + implicit
    num_users: int
    num_items: int
    mappings: Dict[str, Any]
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `load_mappings()` | Load user_item_mappings.json |
| `load_user_pos_sets()` | Load user_pos_train.pkl |
| `load_hard_neg_sets()` | Load user_hard_neg_train.pkl (merged structure) |
| `build_positive_pairs_from_sets()` | Convert sets to (u, i) pairs |
| `load_all()` | **One-line orchestrator** |
| `validate_data()` | Check data integrity |

**Usage:**
```python
from recsys.cf.model.bpr import BPRDataLoader

loader = BPRDataLoader(base_path='data/processed')
data = loader.load_all()

print(f"Positive pairs: {data.positive_pairs.shape}")  # (N, 2)
print(f"Users with hard negs: {len(data.hard_neg_sets)}")
print(f"Dimensions: {data.num_users} users × {data.num_items} items")

# Validate
validation = loader.validate_data()
print(f"Valid: {validation['is_valid']}")
```

**Hard Negative Structure (from Task 01):**
```python
# Original format in user_hard_neg_train.pkl:
{
    u_idx: {
        "explicit": {i1, i2, ...},  # Items rated ≤3
        "implicit": {i3, i4, ...}   # Top-50 popular not bought
    }
}

# After load_hard_neg_sets() - merged for sampling:
{
    u_idx: {i1, i2, i3, i4, ...}  # Combined set
}
```

---

### Step 2: Negative Sampling

#### Module: `recsys/cf/model/bpr/sampler.py`

#### Classes: `TripletSampler`, `HardNegativeMixer`

Sample (user, positive_item, negative_item) triplets with hard negative mixing.

**`HardNegativeMixer`** - Controls negative sampling strategy:

```python
class HardNegativeMixer:
    """Mix hard negatives with random negatives."""
    
    def __init__(
        self,
        hard_neg_sets: Dict[int, Set[int]],
        hard_ratio: float = 0.3,  # 30% hard, 70% random
        random_seed: int = 42
    ):
        ...
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `sample_negative(user_idx, positive_set, num_items)` | Sample one negative |
| `sample_negatives_batch(user_indices, user_pos_sets, num_items)` | Batch sampling |
| `get_stats()` | Sampling statistics (hard vs random counts) |

**`TripletSampler`** - Generate training triplets:

```python
class TripletSampler:
    """Sample triplets for BPR training."""
    
    def __init__(
        self,
        positive_pairs: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        hard_neg_mixer: HardNegativeMixer,
        num_items: int,
        samples_per_epoch: int = 5,  # Multiplier
        random_seed: int = 42
    ):
        ...
```

**Usage:**
```python
from recsys.cf.model.bpr import TripletSampler, HardNegativeMixer

# Initialize mixer
mixer = HardNegativeMixer(
    hard_neg_sets=data.hard_neg_sets,
    hard_ratio=0.3,  # 30% hard + 70% random
    random_seed=42
)

# Initialize sampler
sampler = TripletSampler(
    positive_pairs=data.positive_pairs,
    user_pos_sets=data.user_pos_sets,
    hard_neg_mixer=mixer,
    num_items=data.num_items,
    samples_per_epoch=5,  # 5× num_positives per epoch
    random_seed=42
)

# Sample one epoch
triplets = sampler.sample_epoch()
print(f"Triplets: {triplets.shape}")  # (N*5, 3) = [u, i_pos, i_neg]

# Check sampling stats
stats = mixer.get_stats()
print(f"Hard negatives: {stats['hard_count']} ({stats['hard_ratio']:.1%})")
print(f"Random negatives: {stats['random_count']} ({stats['random_ratio']:.1%})")
```

**Sampling Logic:**
```
For each (user, positive_item) pair:
  1. Roll dice: p < 0.3 → try hard negative
  2. If hard_neg_sets[user] exists and not empty:
     - Sample from hard_neg_sets[user] (excluding positives)
  3. Else fallback to random:
     - Sample uniform from [0, num_items) \ positive_set
```

---

### Step 2 (Advanced): Contextual Negative Sampling

#### Module: `recsys/cf/model/bpr/advanced_sampler.py`

#### Classes: `ContextualNegativeSampler`, `SentimentContrastSampler`, `SamplingStrategy`

Advanced sampling strategies using BERT similarity and sentiment contrast.

```python
@dataclass
class SamplingStrategy:
    """Configuration for negative sampling ratios."""
    hard_ratio: float = 0.25           # Explicit hard negatives
    contextual_ratio: float = 0.20     # BERT similar but not bought
    sentiment_contrast_ratio: float = 0.15  # Opposite sentiment
    popular_ratio: float = 0.10        # Cold-start popular items
    # random_ratio auto-computed as remainder (0.30)

@dataclass
class DynamicSamplingConfig:
    """Curriculum learning for sampling difficulty."""
    enable_dynamic: bool = True
    warmup_epochs: int = 5
    difficulty_schedule: str = 'linear'  # 'linear', 'cosine', 'exponential'
    initial_difficulty: float = 0.3
    final_difficulty: float = 0.8
```

**`ContextualNegativeSampler`:**
```python
class ContextualNegativeSampler:
    """Sample negatives based on BERT/PhoBERT similarity."""
    
    def __init__(
        self,
        item_embeddings: np.ndarray,  # BERT embeddings (num_items, 768)
        item_categories: Dict[int, str] = None,
        item_attributes: Dict[int, Dict] = None,
        top_k_similar: int = 50,
        random_seed: int = 42
    ):
        # Pre-computes item similarity for efficient sampling
        ...
```

**Usage:**
```python
from recsys.cf.model.bpr import (
    AdvancedTripletSampler,
    ContextualNegativeSampler,
    SamplingStrategy,
    DynamicSamplingConfig
)

# Load BERT embeddings
import torch
bert_data = torch.load('data/processed/content_based_embeddings/product_embeddings.pt')
item_embeddings = bert_data['embeddings'].numpy()

# Create contextual sampler
contextual_sampler = ContextualNegativeSampler(
    item_embeddings=item_embeddings,
    top_k_similar=50
)

# Configure strategy
strategy = SamplingStrategy(
    hard_ratio=0.25,
    contextual_ratio=0.20,
    sentiment_contrast_ratio=0.15,
    popular_ratio=0.10
    # random_ratio = 0.30 (auto-computed)
)

# Create advanced sampler
advanced_sampler = AdvancedTripletSampler(
    positive_pairs=data.positive_pairs,
    user_pos_sets=data.user_pos_sets,
    hard_neg_sets=data.hard_neg_sets,
    contextual_sampler=contextual_sampler,
    strategy=strategy,
    dynamic_config=DynamicSamplingConfig(enable_dynamic=True)
)

# Sample with curriculum learning
triplets = advanced_sampler.sample_epoch(epoch=10, total_epochs=50)
```

---

### Step 3: Model Initialization

#### Module: `recsys/cf/model/bpr/model_init.py`

#### Class: `BPRModelInitializer`

Initialize BPR embeddings with optional BERT initialization.

**Usage:**
```python
from recsys.cf.model.bpr import BPRModelInitializer

initializer = BPRModelInitializer(
    num_users=26000,
    num_items=2231,
    factors=64,
    random_seed=42
)

# Method 1: Random initialization
U, V = initializer.initialize_embeddings()
# U, V ~ Normal(0, 0.01)

# Method 2: With BERT initialization for items
U, V = initializer.initialize_embeddings(
    bert_embeddings=item_embeddings,  # (num_items, 768)
    item_mapping=mappings['item_to_idx'],
    projection_method='svd'  # Project 768 → 64 dims
)
```

---

### Step 4: Training

#### Module: `recsys/cf/model/bpr/trainer.py`

#### Class: `BPRTrainer`

Basic BPR trainer with SGD, early stopping, and checkpointing.

```python
@dataclass
class TrainingHistory:
    """Track training progress."""
    epochs: List[int]
    losses: List[float]
    val_metrics: Dict[str, List[float]]  # {'recall@10': [...], ...}
    learning_rates: List[float]
    durations: List[float]
    
    def get_best_epoch(self, metric: str) -> int:
        """Find epoch with best validation metric."""
        ...
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `fit(positive_pairs, user_pos_sets, hard_neg_sets, epochs)` | Train model |
| `_sgd_update_batch(users, pos_items, neg_items)` | Batch SGD update |
| `_compute_validation_metrics(user_pos_test)` | Validation evaluation |
| `get_embeddings()` | Return final U, V |
| `save_checkpoint(path)` | Save training state |
| `load_checkpoint(path)` | Resume training |

**BPR Loss:**
```
L = -log(sigmoid(x_uij)) + reg * (||U[u]||² + ||V[i]||² + ||V[j]||²)

where x_uij = U[u] · V[i] - U[u] · V[j]  (preference difference)
```

**SGD Update Rules:**
```python
# For each triplet (u, i_pos, j_neg):
sigmoid_term = 1 - sigmoid(x_uij)

U[u] += lr * (sigmoid_term * (V[i] - V[j]) - reg * U[u])
V[i] += lr * (sigmoid_term * U[u] - reg * V[i])
V[j] += lr * (-sigmoid_term * U[u] - reg * V[j])
```

**Usage:**
```python
from recsys.cf.model.bpr import BPRTrainer

trainer = BPRTrainer(
    U=U,
    V=V,
    learning_rate=0.05,
    regularization=0.0001,
    lr_decay=0.9,
    lr_decay_every=10,
    checkpoint_dir=Path('checkpoints/bpr'),
    checkpoint_interval=5,
    random_seed=42
)

# Train with early stopping
history = trainer.fit(
    positive_pairs=data.positive_pairs,
    user_pos_sets=data.user_pos_sets,
    hard_neg_sets=data.hard_neg_sets,
    num_items=data.num_items,
    epochs=50,
    samples_per_epoch=5,
    early_stopping_patience=10,
    user_pos_test=user_pos_test  # For validation
)

# Check results
print(f"Best epoch: {history.get_best_epoch('recall@10')}")
print(f"Final loss: {history.losses[-1]:.4f}")
print(f"Training time: {sum(history.durations):.1f}s")

# Get trained embeddings
U, V = trainer.get_embeddings()
```

---

### Step 4 (Advanced): Advanced BPR Trainer

#### Module: `recsys/cf/model/bpr/advanced_trainer.py`

#### Classes: `AdvancedBPRTrainer`, `OptimizerConfig`, `TrainingConfig`, `EmbeddingDropout`

Advanced trainer with AdamW, learning rate scheduling, and gradient clipping.

```python
class OptimizerType(Enum):
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMW = "adamw"

class SchedulerType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    WARMUP_COSINE = "warmup_cosine"

@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    user_weight_decay: float = None  # Separate L2 for users
    item_weight_decay: float = None  # Separate L2 for items
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    scheduler_type: SchedulerType = SchedulerType.WARMUP_COSINE
    warmup_epochs: int = 5
    min_lr: float = 1e-6

@dataclass
class TrainingConfig:
    factors: int = 64
    epochs: int = 50
    samples_per_positive: int = 5
    batch_size: int = 1024
    dropout_rate: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    eval_every: int = 5
    checkpoint_every: int = 10
    random_seed: int = 42
```

**`EmbeddingDropout`:**
```python
class EmbeddingDropout:
    """Dropout for embeddings during training."""
    
    def __init__(self, dropout_rate: float = 0.1, random_seed: int = 42):
        self.dropout_rate = dropout_rate
        self.training = True
    
    def __call__(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply dropout, return (dropped_embeddings, mask)."""
        if not self.training or self.dropout_rate == 0:
            return embeddings, None
        
        keep_prob = 1.0 - self.dropout_rate
        mask = self.rng.random(embeddings.shape) < keep_prob
        return embeddings * mask / keep_prob, mask
```

**Usage:**
```python
from recsys.cf.model.bpr import (
    AdvancedBPRTrainer,
    OptimizerConfig,
    TrainingConfig,
    OptimizerType,
    SchedulerType
)

# Configure optimizer
opt_config = OptimizerConfig(
    optimizer_type=OptimizerType.ADAMW,
    learning_rate=0.01,
    weight_decay=0.01,
    scheduler_type=SchedulerType.WARMUP_COSINE,
    warmup_epochs=5
)

# Configure training
train_config = TrainingConfig(
    factors=64,
    epochs=50,
    batch_size=1024,
    dropout_rate=0.1,
    gradient_clip=1.0,
    early_stopping_patience=10
)

# Create advanced trainer
trainer = AdvancedBPRTrainer(
    U=U,
    V=V,
    optimizer_config=opt_config,
    training_config=train_config,
    checkpoint_dir=Path('checkpoints/bpr_advanced')
)

# Train
history = trainer.fit(
    positive_pairs=data.positive_pairs,
    user_pos_sets=data.user_pos_sets,
    hard_neg_sets=data.hard_neg_sets,
    num_items=data.num_items,
    user_pos_test=user_pos_test
)

# Get embeddings
U, V = trainer.get_embeddings()
```

---

### Step 7: Save Artifacts

#### Module: `recsys/cf/model/bpr/artifact_saver.py`

#### Key Components:

```python
@dataclass
class ScoreRange:
    """BPR score statistics for Task 08 normalization."""
    method: str
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p99: float

@dataclass
class BPRArtifacts:
    user_embeddings_path: Path
    item_embeddings_path: Path
    params_path: Path
    metrics_path: Path
    metadata_path: Path
    score_range: Optional[ScoreRange]
```

**Main Function: `save_bpr_complete()`**

**Usage:**
```python
from recsys.cf.model.bpr import save_bpr_complete, compute_bpr_score_range

artifacts = save_bpr_complete(
    user_embeddings=U,
    item_embeddings=V,
    params={
        'factors': 64,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'epochs': 50,
        'samples_per_epoch': 5,
        'hard_ratio': 0.3
    },
    metrics={
        'recall@10': 0.245,
        'ndcg@10': 0.198,
        'best_epoch': 35
    },
    training_history=history,
    validation_user_indices=[10, 25, 42, ...],
    data_version_hash='abc123def456',
    output_dir='artifacts/cf/bpr'
)

print(artifacts.summary())
# BPR Artifacts:
#   User embeddings: artifacts/cf/bpr/bpr_U.npy
#   Item embeddings: artifacts/cf/bpr/bpr_V.npy
#   Parameters: artifacts/cf/bpr/bpr_params.json
#   Metrics: artifacts/cf/bpr/bpr_metrics.json
#   Metadata: artifacts/cf/bpr/bpr_metadata.json
#   Score range: [-0.35, 1.62]
```

---

## BERT-Enhanced Training

### Module: `recsys/cf/model/bert_enhanced_als.py`

### Class: `BERTEnhancedALS`

ALS with BERT-initialized item embeddings.

**Purpose:**
- Initialize item factors from PhoBERT embeddings (768-dim → 64-dim)
- Transfer semantic knowledge to CF model
- **Critical for ≥2 threshold**: Higher regularization (λ=0.1) anchors sparse users to BERT semantic space

**Usage:**
```python
from recsys.cf.model.bert_enhanced_als import BERTEnhancedALS

bert_als = BERTEnhancedALS(
    bert_embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt',
    factors=64,
    projection_method='svd',  # or 'pca'
    regularization=0.1,       # Higher for sparse data
    iterations=15,
    alpha=10,
    random_state=42
)

# Fit (automatically initializes item factors from BERT)
model = bert_als.fit(
    X_train=X_train_implicit,
    item_to_idx=mappings['item_to_idx']
)

# Access embeddings
U = model.user_factors
V = model.item_factors

# Get metadata with BERT init info
metadata = bert_als.get_training_metadata()
print(f"BERT init: {metadata['bert_initialization']['enabled']}")
print(f"Explained variance: {metadata['bert_initialization']['explained_variance']:.3f}")
```

---

## Hyperparameter Tuning

### ALS Grid (Updated for ≥2 Threshold)
```yaml
factors: [32, 64, 128]
regularization: [0.01, 0.05, 0.1]  # Higher for sparse data
iterations: [10, 15, 20]
alpha: [5, 10, 20]  # Lower due to confidence range 1-6
```
- **Total**: 3×3×3×3 = 81 configs
- **Recommended start**: factors=64, reg=0.1, alpha=10, iter=15

### BPR Grid
```yaml
factors: [32, 64, 128]
learning_rate: [0.01, 0.05, 0.1]
regularization: [0.00001, 0.0001, 0.001]
epochs: [30, 50]
hard_ratio: [0.2, 0.3, 0.4]
```
- **Total**: 3×3×3×2×3 = 162 configs
- **Recommended start**: factors=64, lr=0.05, reg=0.0001, hard_ratio=0.3

---

## Performance Benchmarks

### Expected Training Times (CPU)
- **ALS**: 15 iterations, 26K users, 2.2K items → ~1-2 minutes
- **BPR**: 50 epochs, ~1.8M samples/epoch → ~20-30 minutes
- **Advanced BPR** (AdamW + dropout): ~30-45 minutes

### Expected Metrics
| Model | Recall@10 | NDCG@10 | Notes |
|-------|-----------|---------|-------|
| Popularity Baseline | 0.12-0.15 | 0.08-0.10 | - |
| ALS (default) | >0.20 | >0.15 | +40% vs baseline |
| ALS (BERT-init) | >0.22 | >0.17 | Better cold-start |
| BPR (basic) | >0.22 | >0.16 | - |
| BPR (advanced) | >0.25 | >0.19 | With contextual negs |

---

## Error Handling

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Memory Error (ALS) | OOM during training | Reduce factors, use GPU |
| Memory Error (BPR) | OOM during sampling | Reduce samples_per_epoch |
| Loss Oscillates (BPR) | Training unstable | Lower learning rate, add decay |
| Low Recall | Metrics below baseline | Check test set is positive-only |
| NaN in Embeddings | Training diverged | Lower LR, increase reg |

### Logging
- **Location**: `logs/cf/als.log`, `logs/cf/bpr.log`
- **Format**: `{timestamp} {level} {message}`
- **Rotation**: Keep last 10 runs

---

## Success Criteria

- [x] ALS pipeline trains and saves artifacts
- [x] BPR pipeline trains and saves artifacts
- [ ] Both exceed popularity baseline by ≥20% Recall@10
- [x] Artifacts include score_range for Task 08
- [x] Training scripts documented and configurable
- [x] Error handling robust (memory, convergence)
- [x] Advanced sampling strategies implemented (contextual, sentiment)
- [x] Advanced training options (AdamW, dropout, LR scheduling)

---

## Dependencies

```python
# requirements_cf.txt
numpy>=1.23.0
scipy>=1.9.0
pandas>=1.5.0
implicit>=0.6.0      # For ALS
scikit-learn>=1.2.0  # For metrics
pyyaml>=6.0          # For config
tqdm>=4.64.0         # Progress bars

# BERT features
torch>=1.13.0
transformers>=4.25.0
```

# Model Registry Documentation

## Tổng Quan

Model Registry là hệ thống quản lý phiên bản models, theo dõi performance, và chọn "best model" cho production serving. Hỗ trợ rollback, A/B testing, và audit trail cho reproducibility.

### Tính Năng Chính

| Tính năng | Mô tả |
|-----------|-------|
| **Versioning** | Track models với timestamp, data hash, git commit |
| **Auto-selection** | Tự động chọn best model theo metric |
| **Hot-reload** | Load model mới không cần restart service |
| **Caching** | In-memory cache cho fast serving |
| **Audit Trail** | Log tất cả registry operations |
| **BERT Registry** | Quản lý riêng BERT embeddings |

---

## Quick Start

### 1. Đăng Ký Model Mới

```python
from recsys.cf.registry import ModelRegistry

registry = ModelRegistry()

model_id = registry.register_model(
    artifacts_path='artifacts/cf/als/v2_20250116',
    model_type='als',
    hyperparameters={'factors': 128, 'regularization': 0.01},
    metrics={'recall@10': 0.245, 'ndcg@10': 0.195},
    training_info={'training_time_seconds': 102.8, 'num_users': 26000}
)
```

### 2. Chọn Best Model

```python
best = registry.select_best_model(metric='ndcg@10')
print(f"Best model: {best['model_id']}, NDCG@10: {best['value']}")
```

### 3. Load Model cho Serving

```python
from recsys.cf.registry import get_loader

loader = get_loader()
U, V, metadata = loader.load_current_best()
```

---

## Cấu Trúc Thư Mục

```
artifacts/cf/
├── als/
│   ├── v1_20250115_103000/
│   │   ├── als_U.npy           # User embeddings
│   │   ├── als_V.npy           # Item embeddings
│   │   ├── als_params.json     # Hyperparameters
│   │   └── als_metadata.json   # Training metadata
│   └── v2_20250116_141500/
├── bpr/
│   └── v1_20250115_120000/
├── bert_als/
│   └── v1_20250120_100000/
├── registry.json               # Main CF models registry
└── bert_registry.json          # BERT embeddings registry
```

### Required Model Files

| Model Type | Required Files |
|------------|----------------|
| ALS | `als_U.npy`, `als_V.npy`, `als_params.json`, `als_metadata.json` |
| BPR | `bpr_U.npy`, `bpr_V.npy`, `bpr_params.json`, `bpr_metadata.json` |
| BERT-ALS | `bert_als_U.npy`, `bert_als_V.npy`, `bert_als_params.json`, `bert_als_metadata.json` |

---

## Registry Schema

### File: `registry.json`

```json
{
  "current_best": {
    "model_id": "als_v2_20250116_141500",
    "model_type": "als",
    "version": "v2_20250116_141500",
    "path": "artifacts/cf/als/v2_20250116_141500",
    "selection_metric": "ndcg@10",
    "selection_value": 0.195,
    "selected_at": "2025-01-16T14:30:00",
    "selected_by": "auto"
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
        "recall@20": 0.325,
        "ndcg@10": 0.195,
        "ndcg@20": 0.229,
        "coverage": 0.310
      },
      
      "baseline_comparison": {
        "baseline_type": "popularity",
        "improvement_recall@10": 0.690,
        "improvement_ndcg@10": 0.912
      },
      
      "training_info": {
        "training_time_seconds": 102.8,
        "num_users": 26000,
        "num_items": 2200,
        "num_train_interactions": 320000
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

### Model Status

| Status | Mô tả |
|--------|-------|
| `active` | Available for selection |
| `archived` | Không available, nhưng còn metadata |
| `failed` | Training/validation failed |

---

## Core Classes

### 1. ModelRegistry

Class chính để quản lý registry.

#### Initialization

```python
from recsys.cf.registry import ModelRegistry, get_registry

# Option 1: Direct instantiation
registry = ModelRegistry(
    registry_path='artifacts/cf/registry.json',
    auto_create=True  # Tạo mới nếu không tồn tại
)

# Option 2: Singleton pattern (recommended)
registry = get_registry()
```

#### Method: `register_model()`

Đăng ký model mới vào registry.

```python
model_id = registry.register_model(
    artifacts_path='artifacts/cf/als/v2_20250116',
    model_type='als',  # 'als', 'bpr', 'bert_als'
    
    hyperparameters={
        'factors': 128,
        'regularization': 0.01,
        'iterations': 20,
        'alpha': 60
    },
    
    metrics={
        'recall@10': 0.245,
        'recall@20': 0.325,
        'ndcg@10': 0.195,
        'ndcg@20': 0.229,
        'coverage': 0.310
    },
    
    training_info={
        'training_time_seconds': 102.8,
        'num_users': 26000,
        'num_items': 2200,
        'num_train_interactions': 320000
    },
    
    data_version='abc123...',  # Hash từ Task 01
    git_commit='ghi789...',    # Auto-detect nếu None
    
    baseline_comparison={
        'baseline_type': 'popularity',
        'improvement_ndcg@10': 0.912
    },
    
    bert_integration={  # Optional, cho BERT-enhanced models
        'embeddings_version': 'bert_v1_20250115',
        'projection_method': 'svd',
        'explained_variance': 0.873
    },
    
    version=None,     # Auto-generated: v_{YYYYMMDD}_{HHMMSS}
    overwrite=False   # Nếu True, ghi đè model cùng ID
)

# Returns: 'als_v2_20250116_141500'
```

**Workflow:**
1. Validate artifacts (check required files)
2. Generate version ID
3. Create model entry với metadata
4. Update registry metadata
5. Save to JSON
6. Write audit log

**Errors:**
- `ValueError`: Missing required files
- `ValueError`: Unknown model_type

#### Method: `select_best_model()`

Chọn best model theo metric.

```python
best = registry.select_best_model(
    metric='ndcg@10',           # Metric để compare
    min_improvement=0.1,        # Minimum 10% improvement over baseline
    model_type=None             # Filter by type (None = all)
)

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_info': {...},
#     'metric': 'ndcg@10',
#     'value': 0.195
# }
```

**Workflow:**
1. Filter candidates (status='active')
2. Extract metric values
3. Sort descending
4. Select top model
5. Update `current_best`
6. Save registry
7. Write audit log

#### Method: `list_models()`

Liệt kê tất cả models.

```python
import pandas as pd

# List all models
df = registry.list_models()

# Filter và sort
als_models = registry.list_models(
    model_type='als',
    status='active',
    sort_by='ndcg@10',
    ascending=False
)

# DataFrame columns:
# model_id, model_type, version, status, created_at,
# recall@10, ndcg@10, coverage, training_time
```

#### Method: `archive_model()`

Archive model (soft delete).

```python
success = registry.archive_model('als_v1_20250115_103000')
# Returns: True nếu thành công, False nếu là current_best
```

**Note:** Không thể archive `current_best` model.

#### Method: `delete_model()`

Xóa model khỏi registry.

```python
success = registry.delete_model(
    model_id='als_v1_20250115_103000',
    delete_files=False  # True để xóa cả artifact files
)
```

**Note:** Không thể xóa `current_best` model.

#### Method: `compare_models()`

So sánh nhiều models.

```python
df = registry.compare_models(
    model_ids=['als_v1_...', 'als_v2_...', 'bpr_v1_...'],
    metrics=['recall@10', 'ndcg@10', 'coverage']
)

# Output DataFrame:
# model_id | model_type | factors | recall@10 | ndcg@10 | coverage
```

#### Method: `get_registry_stats()`

Thống kê registry.

```python
stats = registry.get_registry_stats()

# Returns:
# {
#     'total_models': 3,
#     'active_models': 2,
#     'archived_models': 1,
#     'by_type': {'als': 2, 'bpr': 1},
#     'current_best': 'als_v2_20250116_141500',
#     'last_updated': '2025-01-16T14:30:00'
# }
```

---

### 2. ModelLoader

Load models từ registry cho serving.

#### Initialization

```python
from recsys.cf.registry import ModelLoader, get_loader

# Option 1: Direct instantiation
loader = ModelLoader(
    registry_path='artifacts/cf/registry.json',
    cache_enabled=True,   # Enable embedding cache
    auto_load=True        # Auto-load current best on init
)

# Option 2: Singleton pattern (recommended)
loader = get_loader()
```

#### Method: `load_current_best()`

Load current best model.

```python
U, V, metadata = loader.load_current_best()

# Returns:
# - U: np.ndarray (num_users, factors) - User embeddings
# - V: np.ndarray (num_items, factors) - Item embeddings
# - metadata: dict - Model metadata
```

#### Method: `load_model(model_id)`

Load specific model.

```python
U, V, metadata = loader.load_model('als_v2_20250116_141500')
```

#### Method: `reload_model()`

Hot-reload model từ registry.

```python
model_changed = loader.reload_model()
# Returns: True nếu model changed, False nếu same
```

**Use case:** Gọi khi registry update (e.g., new best model selected).

#### Method: `get_current_model()`

Get current model state.

```python
state = loader.get_current_model()

# ModelState dataclass:
# - model_id: str
# - model_type: str
# - version: str
# - U: np.ndarray
# - V: np.ndarray
# - params: Dict
# - metadata: Dict
# - loaded_at: str
# - path: str
```

#### Method: `get_embeddings()`

Quick access to embeddings.

```python
U, V = loader.get_embeddings()
```

#### Method: `get_stats()`

Loader statistics.

```python
stats = loader.get_stats()

# Returns:
# {
#     'total_loads': 10,
#     'cache_hits': 8,
#     'cache_misses': 2,
#     'cache_hit_rate': 0.8,
#     'reload_count': 3,
#     'last_load_time_ms': 45.2,
#     'last_reload_at': '2025-01-16T14:30:00',
#     'cached_models': ['als_v2_...']
# }
```

#### Method: `preload_models()`

Preload multiple models vào cache.

```python
count = loader.preload_models([
    'als_v1_20250115_103000',
    'als_v2_20250116_141500',
    'bpr_v1_20250115_120000'
])
# Returns: 3 (number loaded successfully)
```

#### Caching Behavior

```python
# First load: cache miss, loads from disk (~50ms)
U1, V1, _ = loader.load_model('als_v1_...')

# Second load: cache hit (~0.1ms)
U2, V2, _ = loader.load_model('als_v1_...')

# Check cache hit rate
print(f"Hit rate: {loader.get_stats()['cache_hit_rate']:.1%}")
```

#### Thread Safety

`ModelLoader` sử dụng `threading.RLock()` cho thread-safe access:

```python
import threading
from recsys.cf.registry import get_loader

def worker():
    loader = get_loader()  # Singleton
    U, V, _ = loader.load_current_best()
    # Safe in multiple threads

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
```

---

### 3. BERTEmbeddingsRegistry

Registry riêng cho BERT embeddings.

#### Initialization

```python
from recsys.cf.registry import BERTEmbeddingsRegistry, get_bert_registry

# Option 1: Direct
bert_registry = BERTEmbeddingsRegistry(
    registry_path='artifacts/cf/bert_registry.json',
    auto_create=True
)

# Option 2: Singleton
bert_registry = get_bert_registry()
```

#### Method: `register_embeddings()`

Đăng ký BERT embeddings mới.

```python
version = bert_registry.register_embeddings(
    embedding_path='data/processed/content_based_embeddings',
    model_name='vinai/phobert-base',
    num_items=2244,
    embedding_dim=768,
    generation_config={
        'batch_size': 32,
        'max_length': 256
    },
    text_fields_used=['product_name', 'description', 'ingredients'],
    data_version='abc123...',
    git_commit='def456...'
)
```

#### Method: `get_current_best()`

Get current best embeddings.

```python
best = bert_registry.get_current_best()

# Returns:
# {
#     'version': 'bert_20250115_103000',
#     'info': {
#         'version': 'bert_20250115_103000',
#         'path': 'data/processed/content_based_embeddings',
#         'model_name': 'vinai/phobert-base',
#         'num_items': 2244,
#         'embedding_dim': 768,
#         ...
#     }
# }
```

#### Method: `link_to_model()`

Link embeddings tới CF model.

```python
bert_registry.link_to_model(
    embedding_version='bert_20250115_103000',
    model_id='als_v2_20250116_141500'
)
```

#### Method: `list_embeddings()`

Liệt kê tất cả embeddings.

```python
embeddings_list = bert_registry.list_embeddings(include_linked_models=True)

# Returns list of dicts:
# [
#     {
#         'version': 'bert_20250115_103000',
#         'model_name': 'vinai/phobert-base',
#         'num_items': 2244,
#         'embedding_dim': 768,
#         'is_current': True,
#         'linked_models': ['als_v2_...']
#     }
# ]
```

#### Method: `delete_embeddings()`

Xóa embeddings.

```python
success = bert_registry.delete_embeddings(
    version='bert_20250115_103000',
    delete_files=False
)

# Safety checks:
# - Không thể xóa current best
# - Không thể xóa nếu linked to CF models
```

#### Convenience Function

```python
from recsys.cf.registry import load_bert_embeddings

# Load current best
embeddings, metadata = load_bert_embeddings()

# Load specific version
embeddings, metadata = load_bert_embeddings(version='bert_20250115_103000')
```

---

## Utility Functions

### Versioning

```python
from recsys.cf.registry.utils import (
    generate_version_id,
    parse_version_id,
    compare_versions
)

# Generate version
version = generate_version_id(prefix='v')
# → 'v_20250116_141500'

# Parse version
parsed = parse_version_id('v_20250116_141500')
# {
#     'raw': 'v_20250116_141500',
#     'prefix': 'v',
#     'date': '20250116',
#     'time': '141500',
#     'datetime': '2025-01-16T14:15:00'
# }

# Compare versions
cmp = compare_versions('v_20250115_103000', 'v_20250116_141500')
# → -1 (v1 < v2)
```

### Data Versioning

```python
from recsys.cf.registry.utils import (
    compute_data_version,
    compute_file_hash,
    compute_directory_hash
)

# Compute data version từ multiple files
data_version = compute_data_version([
    'data/processed/interactions.parquet',
    'data/processed/user_item_mappings.json'
])

# Compute single file hash
file_hash = compute_file_hash(
    'data/processed/interactions.parquet',
    algorithm='md5'
)

# Compute directory hash
dir_hash = compute_directory_hash(
    'data/processed',
    extensions=['.parquet', '.json']
)
```

### Git Integration

```python
from recsys.cf.registry.utils import (
    get_git_commit,
    get_git_commit_short,
    get_git_branch,
    is_git_clean
)

# Get full commit hash
commit = get_git_commit()
# → 'def4567890abcdef...'

# Get short hash (7 chars)
commit_short = get_git_commit_short()
# → 'def4567'

# Get branch name
branch = get_git_branch()
# → 'main'

# Check working tree clean
clean = is_git_clean()
# → True/False
```

### Backup & Restore

```python
from recsys.cf.registry.utils import (
    backup_registry,
    restore_registry,
    list_backups
)

# Create backup
backup_path = backup_registry('artifacts/cf/registry.json')
# → 'artifacts/cf/registry_backup_20250116_120000.json'

# List backups
backups = list_backups('artifacts/cf/registry.json')
# [{'path': ..., 'name': ..., 'size_bytes': ..., 'created_at': ...}]

# Restore from backup
restore_registry(
    backup_path='artifacts/cf/registry_backup_20250116_120000.json',
    registry_path='artifacts/cf/registry.json',
    create_current_backup=True  # Backup current before restoring
)
```

### Artifact Management

```python
from recsys.cf.registry.utils import (
    copy_model_artifacts,
    cleanup_old_artifacts
)

# Copy model artifacts
copy_model_artifacts(
    src_path='artifacts/cf/als/v1_20250115',
    dest_path='artifacts/cf/als/v1_backup',
    model_type='als'
)

# Cleanup old artifacts (keep last 5 per type)
deleted = cleanup_old_artifacts(
    artifacts_dir='artifacts/cf',
    keep_count=5,
    keep_current_best=True,
    dry_run=True  # Set False to actually delete
)
```

### Metadata Utilities

```python
from recsys.cf.registry.utils import (
    create_model_metadata,
    load_model_metadata,
    save_model_metadata
)

# Create metadata
metadata = create_model_metadata(
    num_users=26000,
    num_items=2200,
    factors=128,
    model_type='als',
    data_version='abc123...',
    git_commit='def456...',
    score_range=(0.0, 1.5),
    extra={'custom_field': 'value'}
)

# Save metadata
save_model_metadata('artifacts/cf/als/v1', 'als', metadata)

# Load metadata
metadata = load_model_metadata('artifacts/cf/als/v1', 'als')
```

---

## Automation Scripts

### Script: `scripts/update_registry.py`

Đăng ký model mới và optional auto-select.

```powershell
python scripts/update_registry.py `
  --model-path artifacts/cf/als/v2_20250116_141500 `
  --auto-select  # Tự động select nếu metrics improve
```

**Workflow:**
1. Read model artifacts (params, metrics, metadata)
2. Register model in registry.json
3. If `--auto-select`: Compare với current_best, select nếu improve
4. Log results

### Script: `scripts/cleanup_old_models.py`

Cleanup old models.

```powershell
python scripts/cleanup_old_models.py `
  --keep-last 5 `     # Keep 5 most recent per type
  --archive-old       # Archive instead of delete
```

**Workflow:**
1. List all models per type
2. Sort by created_at (descending)
3. Keep top N, archive/delete rest
4. Preserve current_best

---

## Integration with Training

### Training Script Example

```python
from recsys.cf.registry import ModelRegistry
from recsys.cf.registry.utils import compute_data_version, get_git_commit

def train_and_register():
    # ... training code ...
    
    # Save artifacts
    save_artifacts(U, V, params, metrics, metadata, output_path)
    
    # Compute data version
    data_version = compute_data_version([
        'data/processed/interactions.parquet',
        'data/processed/user_item_mappings.json'
    ])
    
    # Register model
    registry = ModelRegistry()
    model_id = registry.register_model(
        artifacts_path=output_path,
        model_type='als',
        hyperparameters=params,
        metrics=metrics,
        training_info={
            'training_time_seconds': elapsed_time,
            'num_users': num_users,
            'num_items': num_items
        },
        data_version=data_version,
        git_commit=get_git_commit()
    )
    
    # Auto-select if improvement
    best = registry.select_best_model(
        metric='ndcg@10',
        min_improvement=0.05
    )
    
    if best:
        print(f"Selected: {best['model_id']}")
```

### Serving Integration

```python
from recsys.cf.registry import get_loader

# In FastAPI lifespan
@asynccontextmanager
async def lifespan(app):
    global loader
    loader = get_loader()
    U, V, metadata = loader.load_current_best()
    logger.info(f"Loaded model: {metadata['model_id']}")
    yield

# Hot-reload endpoint
@app.post("/reload_model")
async def reload():
    changed = loader.reload_model()
    return {"reloaded": changed}
```

---

## A/B Testing Support

### Canary Deployment

```json
{
  "ab_test": {
    "test_id": "ab_als_v2_2025_01_16",
    "start_time": "2025-01-16T15:00:00",
    "control_model": "als_v1_20250115_103000",
    "treatment_model": "als_v2_20250116_141500",
    "traffic_split": 0.1,
    "status": "running"
  }
}
```

### Service Integration

```python
import hashlib

def get_model_for_user(user_id: int, ab_config: dict) -> str:
    """Route user to control or treatment model."""
    if ab_config['status'] != 'running':
        return ab_config['control_model']
    
    # Hash user_id for consistent assignment
    hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    if (hash_val % 100) < ab_config['traffic_split'] * 100:
        return ab_config['treatment_model']
    return ab_config['control_model']
```

### Rollback

1. Stop A/B test: Set `traffic_split = 0`
2. Update `current_best` → previous model
3. Investigate issues
4. Archive or fix treatment model

---

## Audit Trail

### Log File: `logs/registry_audit.log`

```
2025-01-15 10:30:00 | REGISTER | als_v1_20250115_103000 | ndcg@10=0.189
2025-01-16 14:15:00 | REGISTER | als_v2_20250116_141500 | ndcg@10=0.195
2025-01-16 14:30:00 | SELECT_BEST | als_v2_20250116_141500 | improvement=+3.2%
2025-01-17 09:00:00 | ARCHIVE | als_v1_20250115_103000 | reason=manual
2025-01-17 10:00:00 | DELETE | als_v1_20250115_103000 | delete_files=False
```

### Actions Logged

| Action | Description |
|--------|-------------|
| `REGISTER` | Model registration |
| `SELECT_BEST` | Best model selection |
| `ARCHIVE` | Model archived |
| `DELETE` | Model deleted |
| `UPDATE_STATUS` | Status updates |

---

## Data Classes

### ModelState

```python
@dataclass
class ModelState:
    model_id: str
    model_type: str
    version: str
    U: np.ndarray       # User embeddings (num_users, factors)
    V: np.ndarray       # Item embeddings (num_items, factors)
    params: Dict[str, Any]
    metadata: Dict[str, Any]
    loaded_at: str      # ISO timestamp
    path: str
```

### LoaderStats

```python
@dataclass
class LoaderStats:
    total_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    reload_count: int = 0
    last_load_time_ms: float = 0
    last_reload_at: Optional[str] = None
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found"
```python
# Check model exists
models = registry.list_models()
print(models['model_id'].tolist())

# Check file paths
import os
model_info = registry.models['als_v1_...']
print(os.path.exists(model_info['path']))
```

#### 2. "Cannot archive current_best"
```python
# Select different model first
registry.select_best_model(metric='ndcg@10')
# Then archive old model
registry.archive_model('old_model_id')
```

#### 3. "Missing required files"
```python
# Check required files exist
from recsys.cf.registry import REQUIRED_MODEL_FILES

for file in REQUIRED_MODEL_FILES['als']:
    path = f"{artifacts_path}/{file}"
    print(f"{file}: {'✓' if os.path.exists(path) else '✗'}")
```

#### 4. "Cache miss performance"
```python
# Preload models
loader.preload_models(['als_v1_...', 'als_v2_...'])

# Check cache stats
stats = loader.get_stats()
print(f"Hit rate: {stats['cache_hit_rate']:.1%}")
```

### Debug Commands

```powershell
# List all models
python -c "from recsys.cf.registry import get_registry; print(get_registry().list_models())"

# Check current best
python -c "from recsys.cf.registry import get_registry; print(get_registry().registry.get('current_best'))"

# Get registry stats
python -c "from recsys.cf.registry import get_registry; print(get_registry().get_registry_stats())"

# Verify artifacts
python -c "from recsys.cf.registry import get_loader; print(get_loader().get_current_model())"
```

---

## Best Practices

1. **Always use singleton pattern** cho production serving:
   ```python
   loader = get_loader()  # Not ModelLoader()
   ```

2. **Auto-select with min_improvement**:
   ```python
   registry.select_best_model(metric='ndcg@10', min_improvement=0.05)
   ```

3. **Backup before major changes**:
   ```python
   backup_registry('artifacts/cf/registry.json')
   ```

4. **Track data_version** để ensure reproducibility:
   ```python
   data_version = compute_data_version([...])
   ```

5. **Use archive instead of delete**:
   ```python
   registry.archive_model(model_id)  # Not delete_model
   ```

6. **Monitor cache hit rate**:
   ```python
   if loader.get_stats()['cache_hit_rate'] < 0.8:
       loader.preload_models([...])
   ```

7. **Hot-reload in production**:
   ```python
   # Periodic check (e.g., every 5 minutes)
   if loader.reload_model():
       logger.info("Model reloaded!")
   ```

8. **Link BERT embeddings to CF models**:
   ```python
   bert_registry.link_to_model(
       embedding_version='bert_v1_...',
       model_id='als_v2_...'
   )
   ```

---

## Performance

### Benchmark

| Operation | Time |
|-----------|------|
| Register model | ~10ms |
| Select best model | ~5ms |
| Load model (cold) | ~50ms |
| Load model (cached) | ~0.1ms |
| Hot-reload check | ~2ms |
| List models | ~3ms |

### Memory Usage

- Model embeddings: ~50MB per model (128 factors × 26K users × 4 bytes)
- Cache overhead: ~5% per cached model
- Registry JSON: <1MB

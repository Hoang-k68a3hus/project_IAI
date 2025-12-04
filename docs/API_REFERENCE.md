# VieComRec API Reference

REST API cho hệ thống gợi ý mỹ phẩm Việt Nam.

**Base URL**: `http://localhost:8000`  
**OpenAPI Docs**: `http://localhost:8000/docs`

---

## 📋 Mục Lục

1. [Health & Info](#health--info)
2. [Recommendation](#recommendation)
3. [Search](#search)
4. [Cache Management](#cache-management)
5. [Model Management](#model-management)
6. [Scheduler Management](#scheduler-management)
7. [Evaluation](#evaluation)
8. [Rate Limiting](#rate-limiting)
9. [Error Handling](#error-handling)

---

## Health & Info

### GET /health

Kiểm tra trạng thái service và model.

**Response**:
```json
{
  "status": "healthy",
  "model_id": "bert_als_20251125_061805",
  "model_type": "bert_als",
  "num_users": 294857,
  "num_items": 1423,
  "trainable_users": 25717,
  "timestamp": "2025-11-28T10:27:21.589930"
}
```

| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy" hoặc error message |
| model_id | string | ID của model đang được sử dụng |
| model_type | string | Loại model (bert_als, als, bpr) |
| num_users | int | Tổng số users trong hệ thống |
| num_items | int | Tổng số products |
| trainable_users | int | Số users có thể dùng CF model (≥2 interactions) |

---

### GET /model_info

Thông tin chi tiết về model.

**Response**:
```json
{
  "model_id": "bert_als_20251125_061805",
  "model_type": "bert_als",
  "num_users": 294857,
  "num_items": 1423,
  "factors": 64,
  "loaded_at": "2025-11-28T09:03:47.104907",
  "score_range": {
    "method": "sample_users",
    "min": -0.495,
    "max": 1.048,
    "mean": 0.002,
    "p01": -0.075,
    "p99": 0.097
  },
  "trainable_users": 25717,
  "reranking_enabled": true
}
```

---

### GET /stats

Thống kê service.

**Response**:
```json
{
  "model_id": "bert_als_20251125_061805",
  "total_users": 294857,
  "trainable_users": 25717,
  "cold_start_users": 269140,
  "trainable_percentage": 8.72,
  "num_items": 1423,
  "popular_items_cached": 50,
  "user_histories_cached": 25717,
  "cache": {
    "warmed_up": true,
    "caches": {
      "user_profile": {"size": 0, "max_size": 50000, "hit_rate": 0},
      "item_similarity": {"size": 50, "max_size": 5000},
      "fallback": {"size": 0, "max_size": 10000}
    },
    "precomputed": {
      "popular_items": 50,
      "popular_similarities": 50
    }
  }
}
```

---

## Recommendation

### POST /recommend

Gợi ý sản phẩm cho một user.

**Request Body**:
```json
{
  "user_id": 14,
  "topk": 10,
  "exclude_seen": true,
  "filter_params": {
    "brand": "CeraVe",
    "category": "dạng gel",
    "min_price": 100000,
    "max_price": 500000
  },
  "rerank": true,
  "rerank_weights": {
    "cf_score": 0.3,
    "content_score": 0.4,
    "popularity_score": 0.2,
    "quality_score": 0.1
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| user_id | int | ✅ | - | ID của user |
| topk | int | ❌ | 10 | Số recommendations trả về |
| exclude_seen | bool | ❌ | true | Loại bỏ sản phẩm đã mua |
| filter_params | object | ❌ | null | Filters (brand, category, price) |
| rerank | bool | ❌ | true | Áp dụng hybrid reranking |
| rerank_weights | object | ❌ | default | Custom reranking weights |

**Response**:
```json
{
  "user_id": 14,
  "recommendations": [
    {
      "rank": 1,
      "product_id": 125899,
      "score": 0.847,
      "product_name": "Sữa rửa mặt CeraVe",
      "brand": "CeraVe",
      "category": "dạng gel",
      "price": 225000,
      "avg_rating": 4.8,
      "num_sold": 8400
    }
  ],
  "count": 10,
  "is_fallback": false,
  "fallback_method": null,
  "latency_ms": 45.2,
  "model_id": "bert_als_20251125_061805"
}
```

**Routing Logic**:
- **Trainable users** (≥2 interactions, ~8.7%): Sử dụng CF model (ALS/BPR)
- **Cold-start users** (<2 interactions, ~91.3%): Sử dụng fallback (hybrid content + popularity)

---

### POST /batch_recommend

Gợi ý cho nhiều users cùng lúc.

**Request Body**:
```json
{
  "user_ids": [14, 29, 1, 56, 73],
  "topk": 10,
  "exclude_seen": true
}
```

**Response**:
```json
{
  "results": [
    {"user_id": 14, "recommendations": [...], "is_fallback": false},
    {"user_id": 29, "recommendations": [...], "is_fallback": false},
    {"user_id": 1, "recommendations": [...], "is_fallback": true}
  ],
  "num_users": 5,
  "total_latency_ms": 156.3,
  "cf_users": 4,
  "fallback_users": 1
}
```

**Limits**: Max 1000 users per batch.

---

### POST /similar_items

Tìm sản phẩm tương tự (CF-based).

**Request Body**:
```json
{
  "product_id": 125899,
  "topk": 10,
  "use_cf": true
}
```

**Response**:
```json
{
  "product_id": 125899,
  "similar_items": [
    {"product_id": 134988, "score": 0.92},
    {"product_id": 116961, "score": 0.87}
  ],
  "count": 10,
  "method": "cf_item_similarity"
}
```

---

## Search

### POST /search

Tìm kiếm sản phẩm theo ngữ nghĩa (semantic search).

**Request Body**:
```json
{
  "query": "kem dưỡng ẩm cho da khô",
  "topk": 10,
  "filters": {
    "brand": "CeraVe",
    "category": "dạng kem",
    "min_price": 100000,
    "max_price": 1000000
  },
  "rerank": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | ✅ | - | Query tiếng Việt |
| topk | int | ❌ | 10 | Số kết quả |
| filters | object | ❌ | null | Brand/category/price filters |
| rerank | bool | ❌ | true | Rerank theo popularity/quality |

**Response**:
```json
{
  "query": "kem dưỡng ẩm cho da khô",
  "results": [
    {
      "rank": 1,
      "product_id": 49,
      "product_name": "Sữa rửa mặt CERAVE",
      "brand": "CeraVe",
      "category": "no_type",
      "price": 225000,
      "avg_rating": 4.8,
      "num_sold": 8400,
      "semantic_score": 0.312,
      "final_score": 0.545
    }
  ],
  "count": 10,
  "method": "hybrid",
  "latency_ms": 143.5
}
```

**Search Methods**:
- `hybrid`: Semantic search + reranking
- `semantic_only`: Chỉ semantic search (PhoBERT)
- `fallback_popular`: Fallback to popular items

---

### POST /search/similar

Tìm sản phẩm tương tự theo nội dung (content-based).

**Request Body**:
```json
{
  "product_id": 125899,
  "topk": 10,
  "exclude_self": true
}
```

**Response**:
```json
{
  "query": "similar:125899",
  "results": [
    {"rank": 1, "product_id": 134988, "semantic_score": 0.89, ...}
  ],
  "count": 10,
  "method": "content_similarity"
}
```

---

### POST /search/profile

Tìm kiếm dựa trên lịch sử mua hàng.

**Request Body**:
```json
{
  "product_history": [125899, 134988, 116961],
  "topk": 10,
  "exclude_history": true,
  "filters": null
}
```

**Response**:
```json
{
  "query": "profile:3_products",
  "results": [...],
  "count": 10,
  "method": "profile_similarity"
}
```

---

### GET /search/filters

Lấy danh sách filters có sẵn.

**Response**:
```json
{
  "brands": ["cerave", "la roche-posay", "innisfree", ...],
  "categories": ["dạng gel", "dạng kem", "dạng sữa", ...],
  "price_range": [1000, 2950000]
}
```

| Field | Count |
|-------|-------|
| brands | 282 |
| categories | 26 |
| price_range | 1,000đ - 2,950,000đ |

---

### GET /search/stats

Thống kê search service.

**Response**:
```json
{
  "searches_performed": 5,
  "similar_searches": 2,
  "profile_searches": 0,
  "total_latency_ms": 4748.67,
  "avg_latency_ms": 678.38,
  "errors": 0,
  "index": {
    "num_products": 2244,
    "num_brands": 282,
    "num_categories": 26
  },
  "encoder": {
    "queries_encoded": 5,
    "cache_hit_rate": 0.17,
    "model_loaded": true
  }
}
```

---

## Cache Management

### GET /cache_stats

Thống kê cache chi tiết.

**Response**:
```json
{
  "warmed_up": true,
  "caches": {
    "user_profile": {"size": 0, "max_size": 50000, "hits": 0, "misses": 0},
    "item_similarity": {"size": 50, "max_size": 5000},
    "fallback": {"size": 0, "max_size": 10000}
  },
  "precomputed": {
    "popular_items": 50,
    "popular_items_enriched": 50,
    "popular_similarities": 50
  }
}
```

---

### POST /cache_warmup

Warmup cache cho một nhóm users.

**Request Body** (optional):
```json
{
  "user_ids": [14, 29, 44, 56],
  "topk": 10
}
```

**Response**:
```json
{
  "status": "warmed",
  "users_warmed": 4,
  "duration_ms": 250.5
}
```

---

### POST /cache_clear

Xóa tất cả caches.

**Response**:
```json
{
  "status": "cleared"
}
```

---

## Model Management

### POST /reload_model

Hot-reload model mới nhất từ registry.

**Response**:
```json
{
  "status": "reloaded",
  "previous_model_id": "als_20251120_v1",
  "new_model_id": "bert_als_20251125_061805",
  "reloaded": true
}
```

Hoặc nếu không có model mới:
```json
{
  "status": "no_update",
  "previous_model_id": "bert_als_20251125_061805",
  "new_model_id": "bert_als_20251125_061805",
  "reloaded": false
}
```

---

## Scheduler Management

API quản lý lịch chạy tự động các tác vụ ML pipeline (data refresh, training, deployment, v.v.).

### GET /scheduler/status

Kiểm tra trạng thái tổng quan của scheduler.

**Response**:
```json
{
  "running": true,
  "uptime": null,
  "total_jobs": 6,
  "enabled_jobs": 5,
  "disabled_jobs": 1,
  "last_health_check": "2025-11-29T09:00:00.123456",
  "next_scheduled_task": {
    "job_id": "data_refresh",
    "description": "Daily data refresh from raw CSV files",
    "schedule": "hour=2, minute=0"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| running | bool | Scheduler đang chạy hay không |
| total_jobs | int | Tổng số jobs đã cấu hình |
| enabled_jobs | int | Số jobs đang bật |
| disabled_jobs | int | Số jobs đã tắt |
| last_health_check | string | Thời gian health check gần nhất |
| next_scheduled_task | object | Job tiếp theo sẽ chạy |

---

### GET /scheduler/jobs

Liệt kê tất cả scheduled jobs.

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "data_refresh",
      "description": "Daily data refresh from raw CSV files",
      "enabled": true,
      "schedule": {"hour": 2, "minute": 0},
      "module": "automation.data_refresh",
      "args": [],
      "last_run": "2025-11-29T02:00:00.123456",
      "last_status": "success"
    },
    {
      "job_id": "model_training",
      "description": "Weekly model training (ALS + BPR)",
      "enabled": true,
      "schedule": {"day_of_week": "sun", "hour": 3, "minute": 0},
      "module": "automation.model_training",
      "args": ["--auto-select"],
      "last_run": "2025-11-24T03:00:00.123456",
      "last_status": "success"
    }
  ],
  "total": 6,
  "scheduler_running": true
}
```

**Available Jobs**:

| Job ID | Schedule | Description |
|--------|----------|-------------|
| data_refresh | Daily 2:00 AM | Làm mới dữ liệu từ CSV |
| bert_embeddings | Tuesday 3:00 AM | Cập nhật PhoBERT embeddings |
| drift_detection | Monday 9:00 AM | Phát hiện data drift |
| model_training | Sunday 3:00 AM | Huấn luyện model ALS/BPR |
| model_deployment | Daily 5:00 AM | Kiểm tra và deploy model mới |
| health_check | Hourly | Health check hệ thống |

---

### GET /scheduler/jobs/{job_id}

Lấy thông tin chi tiết của một job.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | ID của job (vd: `data_refresh`) |

**Response**:
```json
{
  "job_id": "data_refresh",
  "status": "enabled",
  "enabled": true,
  "last_run": "2025-11-29T02:00:00.123456",
  "last_status": "success",
  "last_log_file": "logs/scheduler/data_refresh_20251129_020000.log"
}
```

---

### POST /scheduler/jobs/{job_id}/run

Chạy job thủ công ngay lập tức.

Job sẽ chạy trong background và trả về ngay. Kiểm tra logs để xem kết quả.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | ID của job |

**Response**:
```json
{
  "job_id": "data_refresh",
  "status": "started",
  "message": "Job 'data_refresh' has been triggered. Check logs for progress.",
  "log_file": "logs/scheduler/data_refresh_20251129_103045.log"
}
```

---

### POST /scheduler/jobs/{job_id}/enable

Bật một job đã tắt.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | ID của job |

**Response**:
```json
{
  "job_id": "data_refresh",
  "status": "enabled",
  "enabled": true,
  "last_run": "2025-11-29T02:00:00.123456",
  "last_status": "success"
}
```

---

### POST /scheduler/jobs/{job_id}/disable

Tắt một job (không chạy theo lịch).

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | ID của job |

**Response**:
```json
{
  "job_id": "model_training",
  "status": "disabled",
  "enabled": false,
  "last_run": "2025-11-24T03:00:00.123456",
  "last_status": "success"
}
```

---

### PUT /scheduler/jobs/{job_id}/schedule

Cập nhật lịch chạy của job.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | ID của job |

**Request Body**:
```json
{
  "schedule": {
    "hour": 3,
    "minute": 30
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| schedule.minute | int | ❌ | Phút (0-59) |
| schedule.hour | int | ❌ | Giờ (0-23) |
| schedule.day_of_week | string | ❌ | Ngày trong tuần: mon, tue, wed, thu, fri, sat, sun |
| schedule.day | int | ❌ | Ngày trong tháng (1-31) |

**Response**:
```json
{
  "job_id": "data_refresh",
  "status": "updated",
  "old_schedule": {"hour": 2, "minute": 0},
  "new_schedule": {"hour": 3, "minute": 30},
  "message": "Schedule updated. Restart scheduler for changes to take effect. New schedule: hour=3, minute=30"
}
```

**Note**: Sau khi cập nhật, cần restart scheduler để áp dụng thay đổi.

---

### GET /scheduler/logs

Lấy logs của scheduler chính.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| lines | int | 100 | Số dòng log trả về (1-1000) |

**Response**:
```json
{
  "task_name": "scheduler",
  "logs": [
    "2025-11-29 09:00:00 - AutomationScheduler - INFO - Health check starting...",
    "2025-11-29 09:00:01 - AutomationScheduler - INFO - ✓ Task completed: health_check"
  ],
  "log_file": "logs/scheduler/scheduler.log",
  "total_lines": 1523
}
```

---

### GET /scheduler/logs/{task_name}

Lấy logs của một task cụ thể.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| task_name | string | Tên task (vd: `data_refresh`, `model_training`) |

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| lines | int | 100 | Số dòng log trả về (1-1000) |

**Response**:
```json
{
  "task_name": "model_training",
  "logs": [
    "=== Manual run triggered at 2025-11-29T10:30:45.123456 ===",
    "Command: python -m automation.model_training --auto-select",
    "============================================================",
    "",
    "Starting model training...",
    "Training ALS model with 26K users...",
    "ALS training complete. Recall@10=0.245",
    "Model registered: bert_als_20251129_103045"
  ],
  "log_file": "logs/scheduler/model_training_20251129_103045.log",
  "total_lines": 156
}
```

---

### GET /scheduler/history

Lấy lịch sử thực thi các task.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| task_name | string | null | Lọc theo tên task |
| page | int | 1 | Trang (1-indexed) |
| page_size | int | 20 | Số items mỗi trang (1-100) |

**Response**:
```json
{
  "history": [
    {
      "task_name": "health_check",
      "status": "success",
      "timestamp": "2025-11-29T09:00:00.123456",
      "exit_code": 0,
      "log_file": "logs/scheduler/health_check_20251129_090000.log",
      "duration_seconds": 1.5,
      "error": null
    },
    {
      "task_name": "data_refresh",
      "status": "success",
      "timestamp": "2025-11-29T02:00:00.456789",
      "exit_code": 0,
      "log_file": "logs/scheduler/data_refresh_20251129_020000.log",
      "duration_seconds": 45.3,
      "error": null
    }
  ],
  "total": 156,
  "page": 1,
  "page_size": 20
}
```

| Status | Description |
|--------|-------------|
| success | Task hoàn thành thành công |
| failed | Task thất bại (exit code ≠ 0) |
| timeout | Task bị timeout (>1 giờ) |
| error | Lỗi khi thực thi |
| running | Đang chạy |

---

### POST /scheduler/reload

Lưu và reload cấu hình scheduler.

**Response**:
```json
{
  "status": "config_saved",
  "message": "Configuration saved. Please restart the scheduler for changes to take effect.",
  "total_jobs": 6,
  "enabled_jobs": 5
}
```

**Note**: Cần restart scheduler process để áp dụng thay đổi.

---

## Evaluation

### POST /evaluate/metrics

Tính toán metrics cho predictions.

**Request Body**:
```json
{
  "predictions": [[1, 2, 3], [4, 5, 6]],
  "ground_truth": [[2, 7], [5, 8]],
  "metric": "recall",
  "k": 3
}
```

| metric | Description |
|--------|-------------|
| recall | Recall@K |
| ndcg | NDCG@K |
| precision | Precision@K |
| mrr | Mean Reciprocal Rank |
| map | Mean Average Precision@K |

**Response**:
```json
{
  "metric": "recall",
  "k": 3,
  "value": 0.25,
  "per_user": [0.5, 0.0],
  "mean": 0.25,
  "std": 0.35,
  "min": 0.0,
  "max": 0.5
}
```

---

### POST /evaluate/statistical_test

So sánh thống kê giữa 2 models.

**Request Body**:
```json
{
  "model1_scores": [0.8, 0.75, 0.82, 0.79],
  "model2_scores": [0.7, 0.68, 0.72, 0.71],
  "test_type": "paired_ttest",
  "significance_level": 0.05
}
```

| test_type | Description |
|-----------|-------------|
| paired_ttest | Paired t-test |
| wilcoxon | Wilcoxon signed-rank test |

**Response**:
```json
{
  "test_type": "paired_ttest",
  "p_value": 0.012,
  "significant": true,
  "effect_size": 1.45,
  "confidence_interval": [0.05, 0.15]
}
```

---

### POST /evaluate/model

Đánh giá model trên test data.

**Request Body**:
```json
{
  "k_values": [10, 20],
  "metrics": ["recall", "ndcg"],
  "test_data": null,
  "user_pos_train": null
}
```

**Note**: Nếu không truyền test_data, sẽ load từ `data/processed/`.

---

### POST /evaluate/compare

So sánh model với baselines.

**Request Body**:
```json
{
  "baseline_names": ["popularity", "random"],
  "k_values": [10, 20]
}
```

---

### POST /evaluate/hybrid

Tính toán hybrid metrics (diversity, novelty, coverage).

**Request Body**:
```json
{
  "recommendations": {"user1": [1,2,3], "user2": [4,5,6]},
  "metrics": ["diversity", "novelty", "coverage"],
  "k_values": [10]
}
```

---

### POST /evaluate/report

Tạo báo cáo đánh giá.

**Request Body**:
```json
{
  "model_results": {...},
  "baseline_results": {...},
  "format": "markdown",
  "include_statistics": true
}
```

---

## Rate Limiting

API áp dụng rate limiting để bảo vệ service:

| Endpoint | Limit |
|----------|-------|
| /recommend | 60 requests/minute |
| /batch_recommend | 30 requests/minute |
| /search | 60 requests/minute |
| Khác | 120 requests/minute |

**Response khi vượt limit** (HTTP 429):
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Validation Error (422)

```json
{
  "detail": [
    {
      "loc": ["body", "user_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 14, "topk": 5}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "kem dưỡng ẩm", "topk": 5}'

# Search với filter
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sua rua mat", "topk": 5, "filters": {"brand": "cerave"}}'

# Scheduler - Xem danh sách jobs
curl http://localhost:8000/scheduler/jobs

# Scheduler - Chạy job thủ công
curl -X POST http://localhost:8000/scheduler/jobs/data_refresh/run

# Scheduler - Tắt một job
curl -X POST http://localhost:8000/scheduler/jobs/model_training/disable

# Scheduler - Cập nhật lịch chạy
curl -X PUT http://localhost:8000/scheduler/jobs/data_refresh/schedule \
  -H "Content-Type: application/json" \
  -d '{"schedule": {"hour": 3, "minute": 30}}'

# Scheduler - Xem logs
curl "http://localhost:8000/scheduler/logs/model_training?lines=50"
```

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Recommendation
response = requests.post(
    f"{BASE_URL}/recommend",
    json={"user_id": 14, "topk": 5}
)
recs = response.json()
print(f"Is fallback: {recs['is_fallback']}")
for item in recs['recommendations']:
    print(f"  {item['rank']}. {item['product_name']} (score: {item['score']:.3f})")

# Search
response = requests.post(
    f"{BASE_URL}/search",
    json={"query": "kem dưỡng ẩm cho da dầu", "topk": 5}
)
results = response.json()
for item in results['results']:
    print(f"  {item['rank']}. {item['product_name']} ({item['brand']})")

# ==================
# Scheduler Management
# ==================

# Xem trạng thái scheduler
response = requests.get(f"{BASE_URL}/scheduler/status")
status = response.json()
print(f"Scheduler running: {status['running']}")
print(f"Enabled jobs: {status['enabled_jobs']}/{status['total_jobs']}")

# Liệt kê tất cả jobs
response = requests.get(f"{BASE_URL}/scheduler/jobs")
jobs = response.json()
for job in jobs['jobs']:
    status_icon = "✅" if job['enabled'] else "❌"
    print(f"{status_icon} {job['job_id']}: {job['description']}")
    print(f"   Last run: {job['last_run']} - Status: {job['last_status']}")

# Chạy job thủ công
response = requests.post(f"{BASE_URL}/scheduler/jobs/data_refresh/run")
result = response.json()
print(f"Job triggered: {result['status']}")
print(f"Log file: {result['log_file']}")

# Bật/tắt job
requests.post(f"{BASE_URL}/scheduler/jobs/model_training/disable")
print("Model training job disabled")

requests.post(f"{BASE_URL}/scheduler/jobs/model_training/enable")
print("Model training job enabled")

# Cập nhật lịch chạy
response = requests.put(
    f"{BASE_URL}/scheduler/jobs/data_refresh/schedule",
    json={"schedule": {"hour": 3, "minute": 30}}
)
result = response.json()
print(f"Old schedule: {result['old_schedule']}")
print(f"New schedule: {result['new_schedule']}")

# Xem logs của task
response = requests.get(
    f"{BASE_URL}/scheduler/logs/model_training",
    params={"lines": 50}
)
logs = response.json()
print(f"Log file: {logs['log_file']}")
for line in logs['logs'][-10:]:
    print(line)

# Xem lịch sử thực thi
response = requests.get(
    f"{BASE_URL}/scheduler/history",
    params={"page": 1, "page_size": 10}
)
history = response.json()
for entry in history['history']:
    print(f"{entry['task_name']}: {entry['status']} at {entry['timestamp']}")
```

### PowerShell

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health | Format-List

# Recommendation
$body = @{ user_id = 14; topk = 5 } | ConvertTo-Json
$recs = Invoke-RestMethod http://localhost:8000/recommend -Method POST -Body $body -ContentType "application/json"
$recs.recommendations | Format-Table rank, product_name, score, brand

# Search
$body = @{ query = "sua rua mat"; topk = 5 } | ConvertTo-Json
$results = Invoke-RestMethod http://localhost:8000/search -Method POST -Body $body -ContentType "application/json"
$results.results | Format-Table rank, product_name, brand, final_score

# ==================
# Scheduler Management
# ==================

# Xem trạng thái scheduler
$status = Invoke-RestMethod http://localhost:8000/scheduler/status
Write-Host "Scheduler running: $($status.running)"
Write-Host "Jobs: $($status.enabled_jobs)/$($status.total_jobs) enabled"

# Liệt kê tất cả jobs
$jobs = Invoke-RestMethod http://localhost:8000/scheduler/jobs
$jobs.jobs | Format-Table job_id, enabled, last_status, @{N='Schedule';E={$_.schedule | ConvertTo-Json -Compress}}

# Chạy job thủ công
$result = Invoke-RestMethod http://localhost:8000/scheduler/jobs/data_refresh/run -Method POST
Write-Host "Job triggered: $($result.status)"
Write-Host "Log file: $($result.log_file)"

# Bật/tắt job
Invoke-RestMethod http://localhost:8000/scheduler/jobs/model_training/disable -Method POST
Write-Host "Model training job disabled"

Invoke-RestMethod http://localhost:8000/scheduler/jobs/model_training/enable -Method POST
Write-Host "Model training job enabled"

# Cập nhật lịch chạy
$scheduleBody = @{
    schedule = @{
        hour = 3
        minute = 30
    }
} | ConvertTo-Json
$result = Invoke-RestMethod http://localhost:8000/scheduler/jobs/data_refresh/schedule -Method PUT -Body $scheduleBody -ContentType "application/json"
Write-Host "Schedule updated from $($result.old_schedule | ConvertTo-Json -Compress) to $($result.new_schedule | ConvertTo-Json -Compress)"

# Xem logs của task
$logs = Invoke-RestMethod "http://localhost:8000/scheduler/logs/model_training?lines=50"
Write-Host "Log file: $($logs.log_file)"
$logs.logs | Select-Object -Last 10

# Xem lịch sử thực thi
$history = Invoke-RestMethod "http://localhost:8000/scheduler/history?page=1&page_size=10"
$history.history | Format-Table task_name, status, timestamp, exit_code
```

### JavaScript (Web Integration)

```javascript
const BASE_URL = 'http://localhost:8000';

// ==================
// Scheduler API Client
// ==================

class SchedulerClient {
    constructor(baseUrl = BASE_URL) {
        this.baseUrl = baseUrl;
    }

    async getStatus() {
        const response = await fetch(`${this.baseUrl}/scheduler/status`);
        return response.json();
    }

    async getJobs() {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs`);
        return response.json();
    }

    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs/${jobId}`);
        return response.json();
    }

    async runJob(jobId) {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs/${jobId}/run`, {
            method: 'POST'
        });
        return response.json();
    }

    async enableJob(jobId) {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs/${jobId}/enable`, {
            method: 'POST'
        });
        return response.json();
    }

    async disableJob(jobId) {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs/${jobId}/disable`, {
            method: 'POST'
        });
        return response.json();
    }

    async updateSchedule(jobId, schedule) {
        const response = await fetch(`${this.baseUrl}/scheduler/jobs/${jobId}/schedule`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ schedule })
        });
        return response.json();
    }

    async getLogs(taskName, lines = 100) {
        const response = await fetch(`${this.baseUrl}/scheduler/logs/${taskName}?lines=${lines}`);
        return response.json();
    }

    async getHistory(page = 1, pageSize = 20, taskName = null) {
        let url = `${this.baseUrl}/scheduler/history?page=${page}&page_size=${pageSize}`;
        if (taskName) url += `&task_name=${taskName}`;
        const response = await fetch(url);
        return response.json();
    }
}

// Usage example
const scheduler = new SchedulerClient();

// Xem trạng thái
scheduler.getStatus().then(status => {
    console.log(`Scheduler: ${status.running ? 'Running' : 'Stopped'}`);
    console.log(`Jobs: ${status.enabled_jobs}/${status.total_jobs} enabled`);
});

// Liệt kê jobs
scheduler.getJobs().then(data => {
    data.jobs.forEach(job => {
        const icon = job.enabled ? '✅' : '❌';
        console.log(`${icon} ${job.job_id}: ${job.description}`);
    });
});

// Chạy job thủ công
scheduler.runJob('data_refresh').then(result => {
    console.log(`Job triggered: ${result.status}`);
    console.log(`Check logs at: ${result.log_file}`);
});

// Cập nhật lịch
scheduler.updateSchedule('data_refresh', { hour: 3, minute: 30 }).then(result => {
    console.log(`Schedule updated: ${JSON.stringify(result.new_schedule)}`);
});
```

---

**Version**: 1.1.0  
**Last Updated**: November 2025  
**Changelog**:
- v1.1.0: Added Scheduler Management API for web integration
- v1.0.0: Initial release with recommendation, search, and evaluation endpoints

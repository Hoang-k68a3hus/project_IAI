# VieComRec Web API Reference

Tài liệu API cho tích hợp hệ thống recommendation vào web e-commerce mỹ phẩm.

**Base URL:** `http://localhost:8000` (Production: `https://api.viecomrec.example.com`)

## 📋 Mục Lục

1. [Recommendation APIs](#1-recommendation-apis)
2. [Search APIs](#2-search-apis)
3. [Data Ingestion APIs](#3-data-ingestion-apis)
4. [Scheduler/Admin APIs](#4-scheduleradmin-apis)
5. [Luồng Hoạt Động](#5-luồng-hoạt-động)

---

## 1. Recommendation APIs

### GET /health
Kiểm tra trạng thái hệ thống.

**Response:**
```json
{
  "status": "healthy",
  "model_id": "bert_als_20251125_061805",
  "model_type": "bert_als",
  "num_users": 294857,
  "num_items": 1423,
  "trainable_users": 25717,
  "timestamp": "2025-11-29T06:25:32.206104",
  "empty_mode": false
}
```

### POST /recommend
Lấy recommendations cho một user.

**Request:**
```json
{
  "user_id": 12345,
  "topk": 10,
  "exclude_seen": true,
  "filter_params": {
    "brand": "Innisfree",
    "category": "serum"
  },
  "rerank": true
}
```

**Response:**
```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "product_id": 28,
      "rank": 1,
      "score": 0.95,
      "product_name": "Sữa Rửa Mặt Cosrx Low pH",
      "brand": "COSRX",
      "price": 103000.0,
      "avg_star": 5.0,
      "num_sold_time": 26500,
      "content_score": 1.0,
      "popularity_score": 0.84,
      "cf_score": 0.95,
      "fallback": false
    }
    // ... more items
  ],
  "count": 10,
  "is_fallback": false,
  "fallback_method": null,
  "latency_ms": 1.86,
  "model_id": "bert_als_20251125_061805"
}
```

### POST /batch_recommend
Recommendations cho nhiều users cùng lúc (bulk).

**Request:**
```json
{
  "user_ids": [100, 200, 300],
  "topk": 5
}
```

### POST /similar_items
Tìm sản phẩm tương tự (content-based).

**Request:**
```json
{
  "product_id": 28,
  "topk": 5
}
```

---

## 2. Search APIs

### POST /search
Tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên (Vietnamese).

**Request:**
```json
{
  "query": "sữa rửa mặt cho da dầu mụn",
  "topk": 10,
  "filters": {
    "brand": "La Roche-Posay",
    "min_price": 100000,
    "max_price": 500000
  }
}
```

**Response:**
```json
{
  "query": "sữa rửa mặt cho da dầu mụn",
  "results": [
    {
      "product_id": 672,
      "product_name": "Sữa rửa mặt La Roche-Posay Effaclar",
      "brand": "La Roche-Posay",
      "price": 234000,
      "relevance_score": 0.92,
      "semantic_score": 0.88,
      "keyword_score": 0.95
    }
  ],
  "total_results": 25,
  "latency_ms": 45.2
}
```

### POST /search/similar
Tìm sản phẩm tương tự từ một product_id.

### POST /search/profile
Tìm sản phẩm phù hợp với profile người dùng.

### GET /search/filters
Lấy danh sách filters có sẵn (brands, categories, price ranges).

---

## 3. Data Ingestion APIs

### POST /ingest/review ⭐
Gửi đánh giá mới từ web (khi user review sản phẩm).

**Request:**
```json
{
  "user_id": 12345,
  "product_id": 28,
  "rating": 5.0,
  "comment": "Sản phẩm rất tốt, da mịn hơn sau 1 tuần!",
  "timestamp": "2025-11-29T10:30:00"  // Optional, auto-filled if missing
}
```

**Response:**
```json
{
  "status": "accepted",
  "interaction_id": "int_20251129_103000_123456",
  "message": "Review staged for processing. Will be included in next data refresh.",
  "timestamp": "2025-11-29T10:30:00.123456"
}
```

### POST /ingest/purchase ⭐
Gửi thông tin mua hàng (implicit positive feedback).

**Request:**
```json
{
  "user_id": 67890,
  "product_id": 419,
  "quantity": 2,
  "timestamp": "2025-11-29T11:00:00"  // Optional
}
```

**Response:**
```json
{
  "status": "accepted",
  "interaction_id": "int_20251129_110000_789012",
  "message": "Purchase staged for processing.",
  "timestamp": "2025-11-29T11:00:00.456789"
}
```

### POST /ingest/batch
Gửi batch nhiều interactions (cho sync hoặc import data).

**Request:**
```json
{
  "reviews": [
    {"user_id": 111, "product_id": 28, "rating": 4.5, "comment": "Khá tốt"},
    {"user_id": 222, "product_id": 672, "rating": 5.0, "comment": "Tuyệt vời"}
  ],
  "purchases": [
    {"user_id": 333, "product_id": 555},
    {"user_id": 444, "product_id": 28, "quantity": 3}
  ]
}
```

**Response:**
```json
{
  "status": "accepted",
  "total_received": 4,
  "reviews_count": 2,
  "purchases_count": 2,
  "message": "Batch of 4 interactions staged for processing.",
  "timestamp": "2025-11-29T12:00:00.123456"
}
```

### GET /ingest/stats
Xem thống kê ingestion.

**Response:**
```json
{
  "total_pending": 11,
  "reviews_pending": 5,
  "purchases_pending": 6,
  "last_ingestion": "2025-11-29T06:37:27",
  "last_processed": "2025-11-29T02:00:00",
  "today_count": 11,
  "staging_file_size_kb": 1.26
}
```

### GET /ingest/pending
Xem chi tiết data đang pending.

---

## 4. Scheduler/Admin APIs

### GET /scheduler/status
Trạng thái scheduler tự động.

### GET /scheduler/jobs
Danh sách tất cả jobs.

### POST /scheduler/jobs/{job_id}/run
Trigger chạy job thủ công.

### PUT /scheduler/jobs/{job_id}/schedule
Cập nhật schedule.

**Request:**
```json
{
  "schedule": {
    "hour": 3,
    "minute": 30
  }
}
```

---

## 5. Luồng Hoạt Động

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WEB APPLICATION                                                             │
│  ───────────────                                                             │
│  1. User xem sản phẩm → GET /health (check API)                             │
│  2. Hiển thị recommendations → POST /recommend                               │
│  3. User tìm kiếm → POST /search                                            │
│  4. User mua hàng → POST /ingest/purchase  ← ⭐ QUAN TRỌNG                  │
│  5. User viết review → POST /ingest/review ← ⭐ QUAN TRỌNG                  │
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  API SERVER (Docker: viecomrec-api:8000)                                     │
│  ───────────────────────────────────────                                     │
│  • Nhận requests từ web                                                      │
│  • Trả recommendations real-time                                             │
│  • Stage data mới vào data/staging/                                          │
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHEDULER (Tự động - Docker: viecomrec-scheduler)                           │
│  ─────────────────────────────────────────────────                           │
│  ┌────────────────────┬────────────────────┬────────────────────────────┐   │
│  │ Job                │ Schedule           │ Mô tả                      │   │
│  ├────────────────────┼────────────────────┼────────────────────────────┤   │
│  │ data_refresh       │ 2:00 AM daily      │ Load data mới từ staging   │   │
│  │ bert_embeddings    │ Tuesday 3:00 AM    │ Update BERT embeddings     │   │
│  │ drift_detection    │ Monday 8:30 AM     │ Detect data drift          │   │
│  │ model_training     │ Sunday 3:00 AM     │ Retrain ALS/BPR models     │   │
│  │ model_deployment   │ 5:00 AM daily      │ Deploy best model          │   │
│  │ health_check       │ Every hour :30     │ System health monitoring   │   │
│  └────────────────────┴────────────────────┴────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Web Integration Checklist

### Khi User Mua Hàng
```javascript
// Sau khi thanh toán thành công
async function onPurchaseComplete(userId, cart) {
  for (const item of cart) {
    await fetch('http://api.example.com/ingest/purchase', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        product_id: item.productId,
        quantity: item.quantity
      })
    });
  }
}
```

### Khi User Viết Review
```javascript
// Sau khi submit review form
async function onReviewSubmit(userId, productId, rating, comment) {
  await fetch('http://api.example.com/ingest/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      product_id: productId,
      rating: rating,  // 1.0 - 5.0
      comment: comment
    })
  });
}
```

### Hiển thị Recommendations
```javascript
// Trên trang chủ hoặc trang sản phẩm
async function getRecommendations(userId) {
  const response = await fetch('http://api.example.com/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      topk: 8,
      rerank: true
    })
  });
  return response.json();
}
```

---

## 📊 Data Flow Summary

| Bước | Thời điểm | Action | API |
|------|-----------|--------|-----|
| 1 | Real-time | User xem web | GET /health |
| 2 | Real-time | Hiển thị gợi ý | POST /recommend |
| 3 | Real-time | User search | POST /search |
| 4 | On purchase | Ghi nhận mua hàng | POST /ingest/purchase |
| 5 | On review | Ghi nhận đánh giá | POST /ingest/review |
| 6 | 2:00 AM | Process data mới | Scheduler: data_refresh |
| 7 | Sunday 3 AM | Retrain models | Scheduler: model_training |
| 8 | 5:00 AM | Deploy model mới | Scheduler: model_deployment |

---

## 🔐 Security Notes

- Rate limiting: 100 requests/minute per IP
- CORS: Chỉ cho phép origins được whitelist
- Production: Dùng HTTPS và API key authentication
- Input validation: Tất cả inputs được validate (rating 1-5, user_id >= 0)

---

## 📞 Support

- API Documentation (Swagger UI): http://localhost:8000/docs
- Health Dashboard: http://localhost:8501
- Logs: Docker logs `viecomrec-api`

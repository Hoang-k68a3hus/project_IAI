# VieComRec - Docker Full Guide 🐳

Hướng dẫn đầy đủ để chạy hệ thống gợi ý mỹ phẩm Việt Nam với Docker.

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Quick Start](#quick-start)
3. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
4. [Cấu Hình Chi Tiết](#cấu-hình-chi-tiết)
5. [API Endpoints](#api-endpoints)
6. [Test Hệ Thống](#test-hệ-thống)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

---

## 🖥️ Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu
- **RAM**: 8GB (khuyến nghị 16GB)
- **CPU**: 4 cores
- **Disk**: 10GB free space

### Phần Mềm
- Docker Desktop 4.x+ hoặc Docker Engine 20.x+
- Docker Compose v2+
- Git (để clone repo)

### Kiểm Tra Docker
```powershell
# Kiểm tra Docker version
docker --version
docker compose version

# Kiểm tra Docker đang chạy
docker info
```

---

## 🚀 Quick Start

### 1. Clone Repository
```powershell
git clone https://github.com/your-repo/viecomrec.git
cd viecomrec
```

### 2. Build và Start Services
```powershell
# Build image
docker-compose build

# Start tất cả services (API + Dashboard + Scheduler)
docker-compose up -d

# Xem logs để theo dõi startup
docker-compose logs -f
```

### 3. Đợi Khởi Động Hoàn Tất
API mất khoảng **2-3 phút** để khởi động do:
- Load CF model (~2 giây)
- Load PhoBERT embeddings (~350ms)
- Load PhoBERT model cho search (~2 phút trên CPU)

Kiểm tra status:
```powershell
# Xem container status
docker-compose ps

# Kiểm tra health
curl http://localhost:8000/health
# Hoặc PowerShell:
Invoke-RestMethod http://localhost:8000/health
```

### 4. Truy Cập Services

| Service | URL | Mô tả |
|---------|-----|-------|
| **API** | http://localhost:8000 | REST API chính |
| **API Docs** | http://localhost:8000/docs | Swagger UI documentation |
| **Dashboard** | http://localhost:8501 | Monitoring dashboard (Streamlit) |

---

## 🏗️ Kiến Trúc Hệ Thống

### Services

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Network                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│                 │                 │                             │
│   ┌─────────┐   │   ┌─────────┐   │   ┌───────────────────┐     │
│   │   API   │   │   │Dashboard│   │   │     Scheduler     │     │
│   │  :8000  │   │   │  :8501  │   │   │ (APScheduler)     │     │
│   └────┬────┘   │   └─────────┘   │   └─────────┬─────────┘     │
│        │        │                 │             │               │
│        └────────┴─────────────────┴─────────────┘               │
│                         │                                       │
├─────────────────────────┴───────────────────────────────────────┤
│                     Volume Mounts                               │
│  ./data (RO) │ ./artifacts (RO) │ ./logs (RW) │ ./config (RO)  │
└─────────────────────────────────────────────────────────────────┘
```

### Container Details

| Container | Image | Port | CPU | Memory |
|-----------|-------|------|-----|--------|
| viecomrec-api | viecomrec:latest | 8000 | 2 cores | 4GB |
| viecomrec-dashboard | viecomrec:latest | 8501 | 0.5 cores | 512MB |
| viecomrec-scheduler | viecomrec:latest | - | 0.5 cores | 512MB |

### Data Flow

```
User Request → API (FastAPI)
                 ├─→ CF Model (ALS/BPR) → Trainable Users (8.7%)
                 │     └─→ Hybrid Reranking
                 └─→ Fallback (PhoBERT) → Cold-Start Users (91.3%)
                       └─→ Content + Popularity
```

---

## ⚙️ Cấu Hình Chi Tiết

### docker-compose.yml Services

```yaml
services:
  api:
    # API service với 4 workers
    ports: ["8000:8000"]
    healthcheck: /health endpoint
    
  dashboard:
    # Streamlit monitoring dashboard
    ports: ["8501:8501"]
    
  scheduler:
    # APScheduler cho automation jobs
    # 6 cron jobs: health_check, data_refresh, bert_embeddings,
    #              drift_detection, model_training, model_deployment
```

### Environment Variables

| Variable | Default | Mô tả |
|----------|---------|-------|
| `ENV` | production | Environment mode |
| `LOG_LEVEL` | INFO | Logging level |
| `WORKERS` | 4 | Số Uvicorn workers |
| `SERVICE_URL` | http://localhost:8000 | API URL (cho scheduler) |

### Volume Mounts

| Host | Container | Mode | Mô tả |
|------|-----------|------|-------|
| `./data` | `/app/data` | RO | Raw data + processed data |
| `./artifacts` | `/app/artifacts` | RO | Model artifacts |
| `./logs` | `/app/logs` | RW | Application logs |
| `./config` | `/app/config` | RO | Configuration files |

---

## 📡 API Endpoints

### Health & Info

```powershell
# Health Check
Invoke-RestMethod http://localhost:8000/health

# Response:
# status          : healthy
# model_id        : bert_als_20251125_061805
# num_users       : 294857
# num_items       : 1423
# trainable_users : 25717
```

```powershell
# Model Info
Invoke-RestMethod http://localhost:8000/model_info

# Service Stats
Invoke-RestMethod http://localhost:8000/stats
```

### Recommendation

```powershell
# Single User Recommendation
$body = @{ user_id = 14; topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/recommend -Method POST -Body $body -ContentType "application/json"

# Response:
# user_id        : 14
# recommendations: [{rank, product_id, score, product_name, brand, ...}]
# is_fallback    : False  # True nếu cold-start user
# model_id       : bert_als_20251125_061805
```

```powershell
# Batch Recommendation (nhiều users)
$body = @{ user_ids = @(14, 29, 1); topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/batch_recommend -Method POST -Body $body -ContentType "application/json"

# Response:
# results      : [...]
# cf_users     : 2   # Users dùng CF model
# fallback_users: 1  # Users dùng fallback
```

```powershell
# Similar Items
$body = @{ product_id = 125899; topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/similar_items -Method POST -Body $body -ContentType "application/json"
```

### Search (Semantic)

```powershell
# Semantic Search
$body = @{ query = "sua rua mat cho da dau"; topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search -Method POST -Body $body -ContentType "application/json"

# Search với Filter
$body = @{ 
    query = "kem duong am"
    topk = 5
    filters = @{ brand = "cerave" }
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search -Method POST -Body $body -ContentType "application/json"
```

```powershell
# Similar Products (by product_id)
$body = @{ product_id = 125899; topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search/similar -Method POST -Body $body -ContentType "application/json"

# Search by User Profile
$body = @{ product_history = @(125899, 134988, 116961); topk = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search/profile -Method POST -Body $body -ContentType "application/json"
```

```powershell
# Get Available Filters
Invoke-RestMethod http://localhost:8000/search/filters

# Response:
# brands     : ["cerave", "la roche-posay", "innisfree", ...] (282 brands)
# categories : ["dạng gel", "dạng kem", ...] (26 categories)
# price_range: [1000, 2950000]
```

### Cache Management

```powershell
# Cache Stats
Invoke-RestMethod http://localhost:8000/cache_stats

# Clear Cache
Invoke-RestMethod http://localhost:8000/cache_clear -Method POST

# Warmup Cache
$body = @{ user_ids = @(14, 29, 44); topk = 10 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/cache_warmup -Method POST -Body $body -ContentType "application/json"
```

### Model Management

```powershell
# Hot-reload Model (check for new best model)
Invoke-RestMethod http://localhost:8000/reload_model -Method POST
```

### Evaluation (Advanced)

```powershell
# Compute Metrics
$body = @{
    predictions = @(@(1,2,3), @(4,5,6))
    ground_truth = @(@(2,7), @(5,8))
    metric = "recall"
    k = 3
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/evaluate/metrics -Method POST -Body $body -ContentType "application/json"

# Statistical Test
$body = @{
    model1_scores = @(0.8, 0.75, 0.82)
    model2_scores = @(0.7, 0.68, 0.72)
    test_type = "paired_ttest"
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/evaluate/statistical_test -Method POST -Body $body -ContentType "application/json"
```

---

## 🧪 Test Hệ Thống

### Quick Test

```powershell
# 1. Kiểm tra services đang chạy
docker-compose ps

# 2. Kiểm tra health
Invoke-RestMethod http://localhost:8000/health | Format-List

# 3. Test recommendation cho trainable user
$body = @{ user_id = 14; topk = 3 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/recommend -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 4

# 4. Test recommendation cho cold-start user (fallback)
$body = @{ user_id = 1; topk = 3 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/recommend -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 4

# 5. Test search
$body = @{ query = "sua rua mat"; topk = 3 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 4

# 6. Test dashboard
Start-Process http://localhost:8501
```

### Comprehensive Test

```powershell
# Chạy test script
python scripts/test_all_api.py --verbose

# Hoặc smoke test nhanh
python scripts/smoke_test.py
```

### Test Automation Modules (trong Docker)

```powershell
# Health Check
docker-compose exec api python -c "from automation.health_check import run_health_check; print(run_health_check())"

# Drift Detection
docker-compose exec api python -c "from automation.drift_detection import check_drift; print(check_drift())"

# Model Deployment (dry-run)
docker-compose exec api python -c "from automation.model_deployment import deploy_best_model; print(deploy_best_model(dry_run=True))"
```

### Kiểm Tra Scheduler

```powershell
# Xem scheduler logs
docker-compose logs scheduler --tail=50

# Các jobs đã đăng ký:
# - health_check: Mỗi giờ (minute=0)
# - data_refresh: 2:00 AM hàng ngày
# - bert_embeddings: Thứ 3, 3:00 AM
# - drift_detection: Thứ 2, 9:00 AM
# - model_training: Chủ nhật, 3:00 AM
# - model_deployment: 5:00 AM hàng ngày
```

---

## 🔧 Troubleshooting

### API Không Khởi Động

```powershell
# Xem logs chi tiết
docker-compose logs api --tail=100

# Kiểm tra file data có mount đúng không
docker-compose exec api ls -la /app/data/processed/

# Kiểm tra model artifacts
docker-compose exec api ls -la /app/artifacts/cf/
```

### Lỗi "Connection Refused"

```powershell
# API đang khởi động, đợi PhoBERT model load (~2 phút)
# Kiểm tra:
docker-compose logs api --tail=20

# Tìm dòng này là xong:
# "Application startup complete."
```

### Lỗi Memory

```powershell
# Kiểm tra memory usage
docker stats

# Tăng memory limit trong docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 6G
```

### Search Filters Trống

```powershell
# Đã fix trong version mới nhất
# Nếu vẫn trống, rebuild image:
docker-compose build api --no-cache
docker-compose up -d api
```

### Model Reload Không Hoạt Động

```powershell
# Kiểm tra registry
docker-compose exec api cat /app/artifacts/cf/registry.json

# Force reload
Invoke-RestMethod http://localhost:8000/reload_model -Method POST
```

---

## 🚢 Production Deployment

### 1. Build Production Image

```powershell
docker-compose build
# Hoặc với tag version:
docker build -t viecomrec:v1.0.0 .
```

### 2. Push to Registry

```powershell
docker tag viecomrec:latest your-registry.com/viecomrec:latest
docker push your-registry.com/viecomrec:latest
```

### 3. Deploy với Docker Compose

```powershell
# Production với restart policy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 4. Health Monitoring

```powershell
# Endpoint cho load balancer health check
curl http://localhost:8000/health

# Expected response (HTTP 200):
# {"status": "healthy", ...}
```

### 5. Logs và Metrics

```powershell
# Logs được ghi vào ./logs/
# - service/api.log
# - scheduler/scheduler.log
# - cf/als.log

# SQLite databases cho metrics:
# - logs/service_metrics.db
# - logs/training_metrics.db
# - logs/pipelines/pipeline_metrics.db
```

---

## 📊 System Stats

Sau khi khởi động thành công:

| Metric | Value |
|--------|-------|
| Total Users | 294,857 |
| Total Items | 1,423 |
| Trainable Users | 25,717 (8.7%) |
| Cold-Start Users | 269,140 (91.3%) |
| Brands | 282 |
| Categories | 26 |
| Price Range | 1,000đ - 2,950,000đ |
| Model | bert_als (BERT-initialized ALS) |

---

## 🆘 Support

### Logs Location
- API logs: `./logs/service/`
- Scheduler logs: `./logs/scheduler/`
- Training logs: `./logs/cf/`

### Databases
- Service metrics: `./logs/service_metrics.db`
- Training metrics: `./logs/training_metrics.db`

### Common Commands

```powershell
# Restart tất cả
docker-compose restart

# Restart chỉ API
docker-compose restart api

# Stop tất cả
docker-compose down

# Stop và xóa volumes
docker-compose down -v

# Xem resource usage
docker stats

# Shell vào container
docker-compose exec api bash
```

---

**Version**: 1.0.0  
**Last Updated**: November 2025

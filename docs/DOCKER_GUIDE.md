# Hướng Dẫn Chạy VieComRec với Docker

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cấu Trúc Project](#cấu-trúc-project)
4. [Hướng Dẫn Nhanh](#hướng-dẫn-nhanh)
5. [Các Dịch Vụ (Services)](#các-dịch-vụ-services)
6. [Hướng Dẫn Chi Tiết](#hướng-dẫn-chi-tiết)
7. [Quản Lý Dữ Liệu](#quản-lý-dữ-liệu)
8. [Cấu Hình Nâng Cao](#cấu-hình-nâng-cao)
9. [Xử Lý Sự Cố](#xử-lý-sự-cố)
10. [FAQ](#faq)

---

## Giới Thiệu

**VieComRec** (Vietnamese Cosmetics Recommender) là hệ thống gợi ý sản phẩm mỹ phẩm cho người dùng Việt Nam, sử dụng kết hợp Collaborative Filtering (ALS, BPR) và Content-Based (Vietnamese Embedding).

### Tại sao dùng Docker?

| Lợi ích | Mô tả |
|---------|-------|
| **Dễ triển khai** | Chạy được trên mọi máy có Docker |
| **Nhất quán** | Môi trường giống nhau mọi nơi |
| **Cách ly** | Không ảnh hưởng hệ thống gốc |
| **Tái tạo** | Dễ dàng reproduce kết quả |

### Kiến trúc Docker

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Network: viecomrec-net                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │   API Service   │   │    Dashboard    │   │    Scheduler    │   │
│  │   (FastAPI)     │   │   (Streamlit)   │   │  (APScheduler)  │   │
│  │   Port: 8000    │   │   Port: 8501    │   │   Background    │   │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘   │
│           │                     │                     │             │
│           └─────────────────────┴─────────────────────┘             │
│                                 │                                   │
│  ┌──────────────────────────────┴──────────────────────────────┐   │
│  │                    Shared Volumes                            │   │
│  │  ./data  │  ./artifacts  │  ./logs  │  ./config             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐                         │
│  │    Trainer      │   │  Data Pipeline  │   ← Chạy theo yêu cầu   │
│  │   (On-demand)   │   │   (On-demand)   │                         │
│  └─────────────────┘   └─────────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Yêu Cầu Hệ Thống

### Phần mềm bắt buộc

| Phần mềm | Phiên bản tối thiểu | Kiểm tra |
|----------|---------------------|----------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |

### Phần cứng khuyến nghị

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB | 20+ GB |

> ⚠️ **Lưu ý**: Image Docker có kích thước ~2.5GB do bao gồm PyTorch và Transformers.

### Cài đặt Docker

#### Windows

1. Tải [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Cài đặt và khởi động lại máy
3. Mở Docker Desktop và đợi Docker Engine khởi động

```powershell
# Kiểm tra cài đặt
docker --version
docker compose version
```

#### macOS

```bash
# Homebrew
brew install --cask docker

# Hoặc tải từ https://docs.docker.com/desktop/install/mac-install/
```

#### Linux (Ubuntu/Debian)

```bash
# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Thêm user vào group docker
sudo usermod -aG docker $USER
newgrp docker

# Cài đặt Docker Compose plugin
sudo apt-get install docker-compose-plugin
```

---

## Cấu Trúc Project

```
viecomrec/
├── Dockerfile                 # Multi-stage build (production + development)
├── docker-compose.yml         # Orchestration cho tất cả services
├── docker.ps1                 # Script quản lý Docker (PowerShell)
├── docker.sh                  # Script quản lý Docker (Bash)
├── requirements.docker.txt    # Dependencies với pinned versions
│
├── data/                      # 📦 Dữ liệu (cần có trước khi chạy)
│   ├── published_data/        #    Raw data (CSV files)
│   └── processed/             #    Processed data (sau khi chạy pipeline)
│
├── artifacts/                 # 🧠 Model artifacts
│   └── cf/                    #    ALS, BPR, BERT-ALS models
│
├── config/                    # ⚙️ Configuration files
│   ├── serving_config.yaml    #    API serving config
│   └── scheduler_config.json  #    Automation scheduler config
│
├── logs/                      # 📝 Application logs
│
├── service/                   # 🚀 Application code
│   ├── api.py                 #    FastAPI endpoints
│   ├── dashboard.py           #    Streamlit dashboard
│   └── search/                #    Smart search module
│
├── recsys/                    # 📊 Recommendation algorithms
│   └── cf/                    #    Collaborative filtering
│
├── automation/                # 🔄 Automation scripts
│   ├── scheduler.py           #    Job scheduler
│   └── model_training.py      #    Training automation
│
└── scripts/                   # 🛠️ Utility scripts
    └── run_task01_complete.py #    Data pipeline
```

---

## Hướng Dẫn Nhanh

### 🚀 Cách Nhanh Nhất: Sử dụng Docker Hub Image

Image đã được publish lên Docker Hub, bạn có thể pull trực tiếp:

```bash
# Pull image từ Docker Hub
docker pull maihoang07082005/viecomrec:latest

# Chạy API (cần mount data & artifacts từ local)
docker run -d -p 8000:8000 \
  -v ./data:/app/data \
  -v ./artifacts:/app/artifacts \
  -v ./logs:/app/logs \
  maihoang07082005/viecomrec:latest

# Truy cập API
# http://localhost:8000
# http://localhost:8000/docs (Swagger UI)
```

> ⚠️ **Lưu ý**: Bạn vẫn cần có thư mục `data/` và `artifacts/` với dữ liệu đã xử lý.

---

### Cách Đầy Đủ: Build từ Source

### Bước 1: Clone repository

```powershell
git clone https://github.com/Hoang-k68a3hus/project_IAI.git
cd viecomrec
```

### Bước 2: Chuẩn bị dữ liệu

Đảm bảo có các file dữ liệu trong thư mục `data/`:

```
data/
├── published_data/
│   ├── data_reviews_purchase.csv     # Reviews & interactions
│   ├── data_product.csv              # Product metadata
│   └── data_product_attribute.csv    # Product attributes
│
└── processed/                         # (Sẽ được tạo bởi pipeline)
    ├── interactions.parquet
    ├── X_train_confidence.npz
    ├── user_item_mappings.json
    └── content_based_embeddings/
        └── product_embeddings.pt
```

### Bước 3: Build Docker image

```powershell
# Windows (PowerShell)
.\docker.ps1 build

# Hoặc manual
docker build -t viecomrec:latest .
```

> ⏱️ Lần build đầu mất khoảng 5-10 phút do tải dependencies.

### Bước 4: Chạy Data Pipeline (nếu chưa có processed data)

```powershell
# Chạy pipeline để tạo processed data
.\docker.ps1 pipeline

# Hoặc manual
docker compose --profile pipeline up data-pipeline
```

### Bước 5: Khởi động services

```powershell
# Khởi động API + Dashboard
.\docker.ps1 start

# Hoặc manual
docker compose up -d api dashboard
```

### Bước 6: Kiểm tra

```powershell
# Kiểm tra status
.\docker.ps1 status

# Mở trình duyệt
# API:       http://localhost:8000
# Docs:      http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Bước 7: Thử nghiệm API

```powershell
# Health check
curl http://localhost:8000/health

# Lấy recommendations
curl -X POST http://localhost:8000/recommend `
  -H "Content-Type: application/json" `
  -d '{"user_id": 12345, "top_k": 10}'

# Tìm kiếm sản phẩm
curl -X POST http://localhost:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query": "kem dưỡng da cho da dầu", "topk": 10}'
```

---

## Các Dịch Vụ (Services)

### Tổng quan

| Service | Port | URL | Mô tả |
|---------|------|-----|-------|
| **api** | 8000 | http://localhost:8000 | FastAPI recommendation endpoints |
| **dashboard** | 8501 | http://localhost:8501 | Streamlit monitoring dashboard |
| **scheduler** | - | (background) | APScheduler automation jobs |
| **trainer** | - | (on-demand) | Model training pipeline |
| **data-pipeline** | - | (on-demand) | Data processing pipeline |

### 1. API Service

**Chức năng**: Cung cấp REST API cho recommendations và search.

**Endpoints chính**:

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/health` | GET | Health check |
| `/recommend` | POST | Lấy recommendations cho user |
| `/search` | POST | Smart search sản phẩm |
| `/search/similar` | POST | Tìm sản phẩm tương tự |
| `/docs` | GET | Swagger documentation |

**Khởi động**:
```powershell
docker compose up -d api
```

**Xem logs**:
```powershell
docker compose logs -f api
```

### 2. Dashboard Service

**Chức năng**: Monitoring và visualization.

**Features**:
- Model performance metrics
- Training history
- API statistics
- Data quality checks

**Khởi động**:
```powershell
docker compose up -d dashboard
```

### 3. Scheduler Service

**Chức năng**: Tự động hóa các jobs định kỳ.

**Jobs**:
- Health check (mỗi 5 phút)
- Data refresh (mỗi ngày)
- Model retraining (mỗi tuần)
- Drift detection (mỗi ngày)

**Khởi động** (chạy cùng api):
```powershell
docker compose up -d api dashboard scheduler
```

### 4. Trainer Service (On-demand)

**Chức năng**: Train các models (ALS, BPR, BERT-ALS).

**Chạy training**:
```powershell
# Sử dụng script
.\docker.ps1 train

# Hoặc manual
docker compose --profile training up trainer
```

### 5. Data Pipeline Service (On-demand)

**Chức năng**: Xử lý raw data → processed data.

**Chạy pipeline**:
```powershell
# Sử dụng script
.\docker.ps1 pipeline

# Hoặc manual
docker compose --profile pipeline up data-pipeline
```

---

## Hướng Dẫn Chi Tiết

### Sử dụng docker.ps1 (Windows)

Script `docker.ps1` cung cấp các lệnh tiện lợi:

```powershell
# Hiển thị help
.\docker.ps1 help

# Build image
.\docker.ps1 build          # Production image
.\docker.ps1 build-dev      # Development image (có pytest)

# Quản lý services
.\docker.ps1 start          # Khởi động API + Dashboard
.\docker.ps1 stop           # Dừng tất cả
.\docker.ps1 restart        # Khởi động lại
.\docker.ps1 status         # Xem trạng thái

# Xem logs
.\docker.ps1 logs           # Logs của API
.\docker.ps1 logs dashboard # Logs của Dashboard

# Chạy jobs
.\docker.ps1 train          # Training models
.\docker.ps1 pipeline       # Data processing

# Development
.\docker.ps1 shell          # Mở shell trong container
.\docker.ps1 test           # Chạy API tests

# Dọn dẹp
.\docker.ps1 clean          # Xóa containers và images
```

### Sử dụng docker.sh (Linux/macOS)

```bash
# Cấp quyền execute
chmod +x docker.sh

# Sử dụng tương tự docker.ps1
./docker.sh start
./docker.sh logs api
./docker.sh train
```

### Sử dụng Docker Compose trực tiếp

```powershell
# Khởi động services
docker compose up -d                    # Tất cả (trừ on-demand)
docker compose up -d api                # Chỉ API
docker compose up -d api dashboard      # API + Dashboard

# Xem logs
docker compose logs -f                  # Tất cả
docker compose logs -f api dashboard    # Cụ thể

# Dừng services
docker compose down                     # Dừng và xóa containers
docker compose stop                     # Chỉ dừng (giữ containers)

# Chạy on-demand jobs
docker compose --profile training up trainer
docker compose --profile pipeline up data-pipeline

# Scale services
docker compose up -d --scale api=3      # 3 API instances
```

### Chạy lệnh trong container

```powershell
# Mở shell
docker compose exec api /bin/bash

# Chạy script cụ thể
docker compose exec api python scripts/test_all_api.py

# Kiểm tra Python environment
docker compose exec api pip list

# Xem files
docker compose exec api ls -la /app/data/processed/
```

---

## Quản Lý Dữ Liệu

### Volume Mounts

| Host Path | Container Path | Mode | Mô tả |
|-----------|----------------|------|-------|
| `./data` | `/app/data` | rw | Raw & processed data |
| `./artifacts` | `/app/artifacts` | rw | Model files |
| `./logs` | `/app/logs` | rw | Application logs |
| `./config` | `/app/config` | ro | Configuration |

### Dữ liệu đầu vào cần có

```
data/published_data/
├── data_reviews_purchase.csv   # ~370K rows, UTF-8
│   Columns: user_id, product_id, rating, comment, cmt_date
│
├── data_product.csv            # ~2.2K rows
│   Columns: product_id, product_name, brand, category, price, num_sold_time
│
└── data_product_attribute.csv  # ~2.2K rows
    Columns: product_id, ingredient, skin_type, feature
```

### Tạo processed data

```powershell
# Chạy data pipeline
.\docker.ps1 pipeline

# Kiểm tra output
ls data/processed/
```

**Output files**:
```
data/processed/
├── interactions.parquet           # Cleaned interactions
├── X_train_confidence.npz         # Training matrix (ALS)
├── X_train_binary.npz             # Training matrix (BPR)
├── user_item_mappings.json        # ID mappings
├── user_pos_train.pkl             # Positive items per user
├── user_metadata.pkl              # User segment info
├── data_stats.json                # Normalization stats
└── content_based_embeddings/
    └── product_embeddings.pt      # Vietnamese Embedding vectors
```

### Tạo model artifacts

```powershell
# Chạy training
.\docker.ps1 train

# Kiểm tra output
ls artifacts/cf/
```

**Output files**:
```
artifacts/cf/
├── registry.json                  # Model registry
├── als/
│   └── 20251130_v1/
│       ├── U.npy                  # User embeddings
│       ├── V.npy                  # Item embeddings
│       ├── params.json            # Hyperparameters
│       └── metadata.json          # Training info
├── bpr/
│   └── ...
└── bert_als/
    └── ...
```

### Backup và Restore

```powershell
# Backup data
docker run --rm -v ${PWD}/data:/data -v ${PWD}/backup:/backup alpine `
    tar czf /backup/data-backup-$(Get-Date -Format "yyyyMMdd").tar.gz /data

# Backup artifacts
docker run --rm -v ${PWD}/artifacts:/artifacts -v ${PWD}/backup:/backup alpine `
    tar czf /backup/artifacts-backup-$(Get-Date -Format "yyyyMMdd").tar.gz /artifacts

# Restore
docker run --rm -v ${PWD}/backup:/backup -v ${PWD}/data:/data alpine `
    tar xzf /backup/data-backup-20251130.tar.gz -C /
```

---

## Cấu Hình Nâng Cao

### Environment Variables

| Variable | Default | Mô tả |
|----------|---------|-------|
| `ENV` | production | Môi trường (production/development) |
| `LOG_LEVEL` | INFO | Log level (DEBUG/INFO/WARNING/ERROR) |
| `WORKERS` | 1 | Số Uvicorn workers cho API |
| `SERVICE_URL` | http://localhost:8000 | Internal API URL |

### Thay đổi cấu hình

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - ENV=production
      - LOG_LEVEL=DEBUG      # Chi tiết hơn
      - WORKERS=4            # Tăng workers
```

### Custom docker-compose.override.yml

Tạo file `docker-compose.override.yml` để override cấu hình:

```yaml
# docker-compose.override.yml
services:
  api:
    ports:
      - "8080:8000"          # Đổi port
    environment:
      - LOG_LEVEL=DEBUG
      - WORKERS=2
    deploy:
      resources:
        limits:
          memory: 4G         # Giới hạn RAM
```

### Development Mode

```powershell
# Build development image
.\docker.ps1 build-dev

# Chạy với hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Tạo `docker-compose.dev.yml`:
```yaml
services:
  api:
    build:
      target: development
    volumes:
      - ./service:/app/service:ro   # Mount code cho hot reload
      - ./recsys:/app/recsys:ro
    command: uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload
```

### GPU Support (Optional)

Nếu có NVIDIA GPU:

```yaml
# docker-compose.gpu.yml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Xử Lý Sự Cố

### 1. Build thất bại với `implicit`

**Lỗi**:
```
error: command 'gcc' failed with exit status 1
```

**Nguyên nhân**: Thiếu build dependencies cho thư viện `implicit`.

**Giải pháp**: Đảm bảo Docker có đủ RAM (>4GB) và builder stage có đủ dependencies.

### 2. API khởi động chậm (>60s)

**Nguyên nhân**: Vietnamese Embedding model cần load lần đầu.

**Giải pháp**: 
- Đây là hành vi bình thường cho lần đầu
- Các request sau sẽ nhanh (~100-300ms)
- Tăng `start_period` trong healthcheck nếu cần

```yaml
healthcheck:
  start_period: 180s  # 3 phút
```

### 3. Out of Memory (OOM)

**Lỗi**:
```
Container killed: OOM
```

**Giải pháp**:

```powershell
# Tăng RAM cho Docker Desktop
# Settings → Resources → Memory: 8GB+

# Hoặc giảm workers
# docker-compose.yml:
#   environment:
#     - WORKERS=1
```

### 4. Port đã được sử dụng

**Lỗi**:
```
Error: bind: address already in use
```

**Giải pháp**:

```powershell
# Tìm process đang dùng port
netstat -ano | findstr :8000

# Hoặc đổi port trong docker-compose.yml
ports:
  - "8080:8000"  # Host 8080 → Container 8000
```

### 5. Không tìm thấy data files

**Lỗi**:
```
FileNotFoundError: data/processed/interactions.parquet
```

**Giải pháp**:

```powershell
# Chạy data pipeline trước
.\docker.ps1 pipeline

# Kiểm tra volume mount
docker compose exec api ls -la /app/data/processed/
```

### 6. Model không load được

**Lỗi**:
```
RuntimeError: Model registry not found
```

**Giải pháp**:

```powershell
# Chạy training
.\docker.ps1 train

# Kiểm tra artifacts
docker compose exec api ls -la /app/artifacts/cf/
```

### 7. Scheduler không chạy

**Kiểm tra**:

```powershell
# Xem logs scheduler
docker compose logs scheduler

# Kiểm tra config
docker compose exec scheduler cat /app/config/scheduler_config.json
```

### 8. Dashboard không kết nối được API

**Kiểm tra**:

```powershell
# API đang chạy?
docker compose ps api

# Network OK?
docker compose exec dashboard curl http://api:8000/health
```

### Debug Mode

```powershell
# Bật debug logging
docker compose exec api bash -c "LOG_LEVEL=DEBUG python -c 'import logging; logging.basicConfig(level=logging.DEBUG)'"

# Xem chi tiết logs
docker compose logs -f --tail=100 api

# Inspect container
docker inspect viecomrec-api
```

---

## FAQ

### Q: Lần đầu chạy cần những gì?

**A**: 
1. Docker + Docker Compose
2. Raw data trong `data/published_data/`
3. Chạy `.\docker.ps1 build`
4. Chạy `.\docker.ps1 pipeline` (nếu chưa có processed data)
5. Chạy `.\docker.ps1 train` (nếu chưa có models)
6. Chạy `.\docker.ps1 start`

### Q: Image Docker bao nhiêu GB?

**A**: ~2GB (production), ~2.3GB (development). Lớn do bao gồm PyTorch + Transformers.

### Q: Có image sẵn trên Docker Hub không?

**A**: Có! Pull trực tiếp:
```bash
docker pull maihoang07082005/viecomrec:latest
```
Link: https://hub.docker.com/r/maihoang07082005/viecomrec

### Q: Có thể chạy trên Windows không?

**A**: Có, sử dụng Docker Desktop for Windows và script `docker.ps1`.

### Q: Làm sao để update code?

**A**: 
```powershell
git pull
.\docker.ps1 build
.\docker.ps1 restart
```

### Q: Làm sao để chạy tests?

**A**:
```powershell
.\docker.ps1 test
# Hoặc
docker compose exec api python -m pytest tests/
```

### Q: Làm sao để xem API documentation?

**A**: Mở http://localhost:8000/docs (Swagger UI)

### Q: Có support GPU không?

**A**: Có, nhưng cần:
- NVIDIA Docker runtime
- File `docker-compose.gpu.yml` 
- PyTorch GPU version (thay đổi trong requirements.docker.txt)

### Q: Làm sao để scale API?

**A**:
```powershell
docker compose up -d --scale api=3
```
Cần thêm load balancer (nginx/traefik) phía trước.

### Q: Data có được persist không?

**A**: Có, qua volume mounts. Data nằm ở host machine (`./data`, `./artifacts`, `./logs`).

### Q: Làm sao để reset hoàn toàn?

**A**:
```powershell
.\docker.ps1 clean
# Xóa data nếu cần
rm -r data/processed/*
rm -r artifacts/*
rm -r logs/*
```

---

## Tài Liệu Liên Quan

- [DOCKER.md](../DOCKER.md) - Quick reference
- [README.md](../README.md) - Project overview
- [API Documentation](http://localhost:8000/docs) - Swagger UI
- [Smart Search Guide](./SMART_SEARCH_GUIDE.md) - Search module
- [Hybrid Reranking Guide](./HYBRID_RERANKING_GUIDE.md) - Reranking module

## Docker Hub

📦 **Image**: `maihoang07082005/viecomrec`

| Tag | Mô tả |
|-----|-------|
| `latest` | Phiên bản mới nhất |
| `1.0.2` | Phiên bản ổn định hiện tại |

```bash
# Pull image
docker pull maihoang07082005/viecomrec:latest

# Xem tags có sẵn
# https://hub.docker.com/r/maihoang07082005/viecomrec/tags
```

---

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra mục [Xử Lý Sự Cố](#xử-lý-sự-cố)
2. Xem logs: `.\docker.ps1 logs api`
3. Tạo issue trên GitHub với:
   - Mô tả lỗi
   - Output của `docker compose ps`
   - Output của `docker compose logs`
   - Thông tin hệ thống (OS, Docker version)

---

*Cập nhật: 30/11/2025*

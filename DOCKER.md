# VieComRec Docker Setup

## Quick Start

```powershell
# Build image
.\docker.ps1 build

# Start API + Dashboard
.\docker.ps1 start

# View logs
.\docker.ps1 logs api
```

## Services

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| Dashboard | 8501 | http://localhost:8501 |
| Docs | 8000 | http://localhost:8000/docs |

## Commands

```powershell
.\docker.ps1 build       # Build production image
.\docker.ps1 build-dev   # Build dev image (with pytest, etc.)
.\docker.ps1 start       # Start API + Dashboard
.\docker.ps1 stop        # Stop all services
.\docker.ps1 restart     # Restart services
.\docker.ps1 logs [svc]  # View logs (api/dashboard)
.\docker.ps1 train       # Run training pipeline
.\docker.ps1 pipeline    # Run data processing pipeline
.\docker.ps1 test        # Run API tests
.\docker.ps1 shell       # Shell into container
.\docker.ps1 status      # Show service status
.\docker.ps1 clean       # Remove containers/images
```

## Manual Docker Commands

```powershell
# Build
docker build -t viecomrec:latest .

# Run API only
docker run --rm -p 8000:8000 `
  -v ${PWD}/data:/app/data:ro `
  -v ${PWD}/artifacts:/app/artifacts:ro `
  -v ${PWD}/logs:/app/logs `
  viecomrec:latest

# Run Dashboard only
docker run --rm -p 8501:8501 `
  -v ${PWD}/logs:/app/logs:ro `
  viecomrec:latest `
  streamlit run service/dashboard.py --server.port 8501 --server.address 0.0.0.0

# Run training
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/artifacts:/app/artifacts `
  -v ${PWD}/logs:/app/logs `
  viecomrec:latest `
  python -m automation.model_training --auto-select

# Run data pipeline
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/artifacts:/app/artifacts `
  viecomrec:latest `
  python scripts/run_task01_complete.py
```

## Docker Compose

```powershell
# Start all services
docker compose up -d

# Start specific service
docker compose up -d api

# Run training job
docker compose --profile training up trainer

# Run data pipeline job
docker compose --profile pipeline up data-pipeline

# View logs
docker compose logs -f api

# Stop all
docker compose down
```

## Volume Mounts

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./data` | `/app/data` | ro/rw | Raw & processed data |
| `./artifacts` | `/app/artifacts` | ro/rw | Model artifacts |
| `./logs` | `/app/logs` | rw | Application logs |
| `./config` | `/app/config` | ro | Configuration files |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | production | Environment (production/development/training) |
| `LOG_LEVEL` | INFO | Logging level |
| `WORKERS` | 4 | Uvicorn workers for API |

## Image Sizes (Approximate)

- **Production**: ~2.5GB (includes PyTorch CPU, transformers)
- **Development**: ~2.8GB (adds pytest, dev tools)

## Troubleshooting

### Build fails with implicit
```powershell
# Ensure Docker has enough memory (4GB+)
# The implicit library compiles C++ code during pip install
```

### API takes long to start
```powershell
# First request loads PhoBERT model (~20-30s)
# Subsequent requests are fast (~300ms)
# Check warmup logs: docker compose logs api
```

### Out of memory
```powershell
# Reduce workers in docker-compose.yml
# Or increase Docker memory limit
```

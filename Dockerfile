# ============================================================================
# VieComRec - Vietnamese Cosmetics Recommender System
# Multi-stage Dockerfile for API, Training, and Dashboard
# ============================================================================

# Build stage for compiling implicit library
FROM python:3.11-slim AS builder

# Install build dependencies for implicit (ALS/BPR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# ============================================================================
# Production image
# ============================================================================
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Application config
    LOG_DIR=/app/logs \
    DATA_DIR=/app/data \
    ARTIFACTS_DIR=/app/artifacts \
    CONFIG_DIR=/app/config \
    # Default to production mode
    ENV=production

# Create directories
RUN mkdir -p /app/logs /app/data /app/artifacts /app/config

# Copy application code
COPY recsys/ /app/recsys/
COPY service/ /app/service/
COPY scripts/ /app/scripts/
COPY automation/ /app/automation/
COPY config/ /app/config/
COPY alerting.py /app/

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API server)
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose ports
EXPOSE 8000 8501


# ============================================================================
# Development image with additional tools
# ============================================================================
FROM production AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    black \
    isort \
    mypy

# Enable hot reload
ENV RELOAD=true

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#!/bin/bash
# ============================================================================
# VieComRec - Docker Build & Run Script
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Commands
# ============================================================================

build() {
    print_header "Building Docker Image"
    docker build -t viecomrec:latest .
    print_success "Image built successfully"
}

build_dev() {
    print_header "Building Development Image"
    docker build -t viecomrec:dev --target development .
    print_success "Development image built successfully"
}

start() {
    print_header "Starting Services (API + Dashboard)"
    docker compose up -d api dashboard
    print_success "Services started"
    echo ""
    echo "API:       http://localhost:8000"
    echo "Dashboard: http://localhost:8501"
    echo "Health:    http://localhost:8000/health"
}

stop() {
    print_header "Stopping Services"
    docker compose down
    print_success "Services stopped"
}

restart() {
    stop
    start
}

logs() {
    docker compose logs -f "${1:-api}"
}

train() {
    print_header "Running Training Pipeline"
    docker compose --profile training up trainer
    print_success "Training complete"
}

pipeline() {
    print_header "Running Data Pipeline"
    docker compose --profile pipeline up data-pipeline
    print_success "Data pipeline complete"
}

test_api() {
    print_header "Testing API"
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is ready"
            break
        fi
        sleep 2
    done
    
    # Run tests
    docker compose exec api python scripts/test_all_api.py
}

shell() {
    print_header "Opening Shell in API Container"
    docker compose exec api /bin/bash
}

clean() {
    print_header "Cleaning Docker Resources"
    docker compose down -v --rmi local
    docker image prune -f
    print_success "Cleaned up"
}

status() {
    print_header "Service Status"
    docker compose ps
}

help() {
    echo "VieComRec Docker Management Script"
    echo ""
    echo "Usage: ./docker.sh <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build production Docker image"
    echo "  build-dev   Build development Docker image"
    echo "  start       Start API and Dashboard services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs [svc]  View logs (default: api)"
    echo "  train       Run training pipeline"
    echo "  pipeline    Run data processing pipeline"
    echo "  test        Run API tests"
    echo "  shell       Open shell in API container"
    echo "  status      Show service status"
    echo "  clean       Remove containers and images"
    echo "  help        Show this help"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-help}" in
    build)      build ;;
    build-dev)  build_dev ;;
    start)      start ;;
    stop)       stop ;;
    restart)    restart ;;
    logs)       logs "$2" ;;
    train)      train ;;
    pipeline)   pipeline ;;
    test)       test_api ;;
    shell)      shell ;;
    status)     status ;;
    clean)      clean ;;
    help|*)     help ;;
esac

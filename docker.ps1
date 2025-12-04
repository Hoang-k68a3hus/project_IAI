# ============================================================================
# VieComRec Docker Management Script (PowerShell)
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "build-dev", "start", "stop", "restart", "logs", "train", "pipeline", "test", "shell", "status", "clean", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Service = "api"
)

function Write-Header {
    param([string]$Message)
    Write-Host "============================================================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "============================================================================" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# ============================================================================
# Commands (renamed to avoid conflict with Invoke-Build module)
# ============================================================================

function Start-DockerBuild {
    Write-Header "Building Docker Image"
    docker build -t viecomrec:latest .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Image built successfully"
    }
}

function Start-DockerBuildDev {
    Write-Header "Building Development Image"
    docker build -t viecomrec:dev --target development .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Development image built successfully"
    }
}

function Start-DockerServices {
    Write-Header "Starting Services (API + Dashboard)"
    docker compose up -d api dashboard
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started"
        Write-Host ""
        Write-Host "API:       http://localhost:8000" -ForegroundColor Cyan
        Write-Host "Dashboard: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "Health:    http://localhost:8000/health" -ForegroundColor Cyan
    }
}

function Stop-DockerServices {
    Write-Header "Stopping Services"
    docker compose down
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services stopped"
    }
}

function Restart-DockerServices {
    Stop-DockerServices
    Start-DockerServices
}

function Show-DockerLogs {
    param([string]$Svc = "api")
    docker compose logs -f $Svc
}

function Start-DockerTrain {
    Write-Header "Running Training Pipeline"
    docker compose --profile training up trainer
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Training complete"
    }
}

function Start-DockerPipeline {
    Write-Header "Running Data Pipeline"
    docker compose --profile pipeline up data-pipeline
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Data pipeline complete"
    }
}

function Start-DockerTest {
    Write-Header "Testing API"
    
    Write-Host "Waiting for API to be ready..."
    $ready = $false
    for ($i = 1; $i -le 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Success "API is ready"
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    
    if ($ready) {
        docker compose exec api python scripts/test_all_api.py
    } else {
        Write-Warn "API not ready after 60 seconds"
    }
}

function Start-DockerShell {
    Write-Header "Opening Shell in API Container"
    docker compose exec api /bin/bash
}

function Show-DockerStatus {
    Write-Header "Service Status"
    docker compose ps
}

function Start-DockerClean {
    Write-Header "Cleaning Docker Resources"
    docker compose down -v --rmi local
    docker image prune -f
    Write-Success "Cleaned up"
}

function Show-Help {
    Write-Host "VieComRec Docker Management Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\docker.ps1 command [service]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build       Build production Docker image"
    Write-Host "  build-dev   Build development Docker image"
    Write-Host "  start       Start API and Dashboard services"
    Write-Host "  stop        Stop all services"
    Write-Host "  restart     Restart all services"
    Write-Host "  logs [svc]  View logs (default: api)"
    Write-Host "  train       Run training pipeline"
    Write-Host "  pipeline    Run data processing pipeline"
    Write-Host "  test        Run API tests"
    Write-Host "  shell       Open shell in API container"
    Write-Host "  status      Show service status"
    Write-Host "  clean       Remove containers and images"
    Write-Host "  help        Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker.ps1 build          # Build image"
    Write-Host "  .\docker.ps1 start          # Start services"
    Write-Host "  .\docker.ps1 logs dashboard # View dashboard logs"
    Write-Host "  .\docker.ps1 train          # Run training"
}

# ============================================================================
# Main
# ============================================================================

switch ($Command) {
    "build"     { Start-DockerBuild }
    "build-dev" { Start-DockerBuildDev }
    "start"     { Start-DockerServices }
    "stop"      { Stop-DockerServices }
    "restart"   { Restart-DockerServices }
    "logs"      { Show-DockerLogs -Svc $Service }
    "train"     { Start-DockerTrain }
    "pipeline"  { Start-DockerPipeline }
    "test"      { Start-DockerTest }
    "shell"     { Start-DockerShell }
    "status"    { Show-DockerStatus }
    "clean"     { Start-DockerClean }
    default     { Show-Help }
}

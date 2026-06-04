# ============================================================================
# VieComRec Docker Management Script (PowerShell)
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "build-dev", "start", "stop", "restart", "logs", "train", "train-als", "train-bpr", "pipeline", "test", "mlops-test", "shell", "status", "clean", "help")]
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
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
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
    docker compose --profile training run --rm trainer python -m automation.model_training --auto-select
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Training complete"
    }
}

function Start-DockerTrainModel {
    param([string]$Model)
    Write-Header "Running $Model Training Pipeline"
    docker compose --profile training run --rm trainer python -m automation.model_training --model $Model --auto-select
    if ($LASTEXITCODE -eq 0) {
        Write-Success "$Model training complete"
    }
}

function Start-DockerPipeline {
    Write-Header "Running Data Pipeline"
    docker compose --profile pipeline run --rm data-pipeline python scripts/run_task01_complete.py
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

function Invoke-MlopsStep {
    param(
        [string]$Name,
        [string[]]$DockerArgs,
        [switch]$AllowWarning
    )

    Write-Host ""
    Write-Host "[$Name]" -ForegroundColor Cyan
    & docker @DockerArgs
    $code = $LASTEXITCODE

    if ($code -eq 0 -or ($AllowWarning.IsPresent -and $code -eq 1)) {
        Write-Success "$Name passed"
        return
    }

    throw "$Name failed with exit code $code"
}

function Start-DockerMlopsTest {
    Write-Header "Testing MLOps Pipelines"

    Invoke-MlopsStep -Name "automation modules" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "-e", "SERVICE_URL=http://api:8000",
        "trainer", "python", "automation/test_modules.py"
    )

    Invoke-MlopsStep -Name "trainer imports" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "trainer", "python", "-c",
        "import implicit, automation.model_training as m; print('trainer ok', implicit.__version__, m.TRAINING_CONFIG['als']['factors'])"
    )

    Invoke-MlopsStep -Name "deployment dry-run" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "-e", "SERVICE_URL=http://api:8000",
        "trainer", "python", "-m", "automation.model_deployment", "--dry-run"
    )

    Invoke-MlopsStep -Name "data refresh dry-run" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "trainer", "python", "-m", "automation.data_refresh", "--dry-run", "--skip-merge"
    )

    Invoke-MlopsStep -Name "bert embeddings check" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "trainer", "python", "-m", "automation.bert_embeddings", "--check-only"
    )

    Invoke-MlopsStep -Name "cleanup dry-run" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "trainer", "python", "-m", "automation.cleanup", "--dry-run"
    )

    Invoke-MlopsStep -Name "drift detection" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "trainer", "python", "-m", "automation.drift_detection"
    )

    Invoke-MlopsStep -Name "health check" -DockerArgs @(
        "compose", "--profile", "training", "run", "--rm",
        "-e", "SERVICE_URL=http://api:8000",
        "trainer", "python", "-m", "automation.health_check", "--json"
    ) -AllowWarning
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
    Write-Host "  train       Run training pipeline in Linux Docker"
    Write-Host "  train-als   Run ALS training in Linux Docker"
    Write-Host "  train-bpr   Run BPR training in Linux Docker"
    Write-Host "  pipeline    Run data processing pipeline"
    Write-Host "  test        Run API tests"
    Write-Host "  mlops-test  Run MLOps smoke suite in Linux Docker"
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
    Write-Host "  .\docker.ps1 train-als      # Run ALS training only"
    Write-Host "  .\docker.ps1 mlops-test     # Run MLOps smoke suite"
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
    "train-als" { Start-DockerTrainModel -Model "als" }
    "train-bpr" { Start-DockerTrainModel -Model "bpr" }
    "pipeline"  { Start-DockerPipeline }
    "test"      { Start-DockerTest }
    "mlops-test" { Start-DockerMlopsTest }
    "shell"     { Start-DockerShell }
    "status"    { Show-DockerStatus }
    "clean"     { Start-DockerClean }
    default     { Show-Help }
}

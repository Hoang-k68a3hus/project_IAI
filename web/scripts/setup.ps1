# Quick setup script for Windows
# Run from any directory: powershell -ExecutionPolicy Bypass -File .\web\scripts\setup.ps1

$ErrorActionPreference = "Stop"

$WebRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ServerDir = Join-Path $WebRoot "server"
$ClientDir = Join-Path $WebRoot "client"

Write-Host "VieComRec web setup" -ForegroundColor Cyan

Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow
$nodeVersion = node --version
$npmVersion = npm --version
Write-Host "Node.js $nodeVersion"
Write-Host "npm $npmVersion"

Write-Host "`nPreparing env files..." -ForegroundColor Yellow
$serverEnv = Join-Path $ServerDir ".env"
$serverEnvExample = Join-Path $ServerDir ".env.example"
$clientEnv = Join-Path $ClientDir ".env"
$clientEnvExample = Join-Path $ClientDir ".env.example"

if (-not (Test-Path $serverEnv) -and (Test-Path $serverEnvExample)) {
    Copy-Item -LiteralPath $serverEnvExample -Destination $serverEnv
    Write-Host "Created server/.env from example"
}

if (-not (Test-Path $clientEnv) -and (Test-Path $clientEnvExample)) {
    Copy-Item -LiteralPath $clientEnvExample -Destination $clientEnv
    Write-Host "Created client/.env from example"
}

Write-Host "`nStarting MongoDB with Docker..." -ForegroundColor Yellow
docker compose -f (Join-Path $WebRoot "docker-compose.yml") up -d mongodb

Write-Host "`nInstalling server dependencies..." -ForegroundColor Yellow
Push-Location $ServerDir
npm install
Pop-Location

Write-Host "`nInstalling client dependencies..." -ForegroundColor Yellow
Push-Location $ClientDir
npm install
Pop-Location

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Server: cd web/server; npm start"
Write-Host "Client: cd web/client; npm start"
Write-Host "VieComRec API: http://localhost:8000"
Write-Host "Web API: http://localhost:5000/api/health"
Write-Host "Web app: http://localhost:3000"

# Build script with BuildKit enabled
# Usage: .\build.ps1

Write-Host "Enabling BuildKit..." -ForegroundColor Green
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

Write-Host "Building Docker images..." -ForegroundColor Green
docker-compose build

Write-Host "`nBuild complete!" -ForegroundColor Green
Write-Host "To start services: docker-compose up -d" -ForegroundColor Cyan


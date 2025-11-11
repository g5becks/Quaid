#!/usr/bin/env pwsh

Write-Host "🚀 Setting up Quaid development environment..." -ForegroundColor Green

# Check if mise is installed
if (!(Get-Command mise -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing mise..." -ForegroundColor Yellow

    # Try winget first
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Using winget to install mise..." -ForegroundColor Cyan
        winget install jdx.mise
    }
    # Try scoop second
    elseif (Get-Command scoop -ErrorAction SilentlyContinue) {
        Write-Host "Using scoop to install mise..." -ForegroundColor Cyan
        scoop install mise
    }
    # Try chocolatey third
    elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Using chocolatey to install mise..." -ForegroundColor Cyan
        choco install mise
    }
    # No package manager found
    else {
        Write-Host "❌ Error: No supported package manager found!" -ForegroundColor Red
        Write-Host "Please install one of the following:" -ForegroundColor Yellow
        Write-Host "  - winget (comes with Windows 10+)" -ForegroundColor Cyan
        Write-Host "  - scoop: https://scoop.sh" -ForegroundColor Cyan
        Write-Host "  - chocolatey: https://chocolatey.org" -ForegroundColor Cyan
        exit 1
    }

    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
} else {
    Write-Host "✅ mise is already installed" -ForegroundColor Green
}

# Trust the project configuration
Write-Host "🔐 Trusting project configuration..." -ForegroundColor Yellow
mise trust

# Install development tools
Write-Host "🔧 Installing development tools..." -ForegroundColor Yellow
mise install

# Install workspace dependencies
Write-Host "📚 Installing workspace dependencies with uv..." -ForegroundColor Yellow
uv sync

# Install git hooks
Write-Host "🪝 Installing git hooks..." -ForegroundColor Yellow
mise exec lefthook -- lefthook install

Write-Host "🎉 Setup complete! You can now run:" -ForegroundColor Green
Write-Host "  uv run quaid --help    # Show CLI help" -ForegroundColor Cyan
Write-Host "  uv run pytest         # Run tests" -ForegroundColor Cyan
Write-Host "  uv run ruff check     # Run linter" -ForegroundColor Cyan
Write-Host "  uv run ruff format    # Run formatter" -ForegroundColor Cyan
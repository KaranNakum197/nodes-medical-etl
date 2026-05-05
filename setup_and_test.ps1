#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Medical ETL Pipeline - Quick Setup & Test Script with uv + venv
    
.DESCRIPTION
    Automated setup script using uv (fast pip alternative) and Python venv.
    Configures backend, agents, and runs integration tests.
    
.PARAMETER SkipTests
    Skip running integration tests after setup
    
.PARAMETER VerboseLogging
    Enable verbose logging during setup and tests
    
.EXAMPLE
    .\setup_and_test.ps1
    .\setup_and_test.ps1 -SkipTests
    .\setup_and_test.ps1 -VerboseLogging
#>

param(
    [switch]$SkipTests,
    [switch]$VerboseLogging
)

$ErrorActionPreference = "Stop"
$InformationPreference = "Continue"

# Colors for output
$colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error   = "Red"
    Info    = "Cyan"
}

function Write-Step {
    param([string]$Step, [string]$Color = "Info")
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host $Step -ForegroundColor $colors[$Color]
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
}

function Test-CommandExists {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
Write-Step "STEP 1: Checking Prerequisites" "Info"

# Check Python
Write-Host "✓ Checking Python installation..." -ForegroundColor Cyan
if (-not (Test-CommandExists "python")) {
    Write-Host "✗ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}
$pythonVersion = & python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green

# Check uv
Write-Host "✓ Checking uv installation..." -ForegroundColor Cyan
if (-not (Test-CommandExists "uv")) {
    Write-Host "⚠ uv not found. Installing uv..." -ForegroundColor Yellow
    pip install uv
    if (-not (Test-CommandExists "uv")) {
        Write-Host "✗ Failed to install uv" -ForegroundColor Red
        exit 1
    }
}
$uvVersion = & uv --version
Write-Host "  $uvVersion" -ForegroundColor Green

# Check current directory
$projectRoot = Get-Location
if (-not (Test-Path "backend/main.py" -PathType Leaf)) {
    Write-Host "✗ Not in project root directory. backend/main.py not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  Project root: $projectRoot" -ForegroundColor Green

Write-Host "✓ All prerequisites met" -ForegroundColor Green

# ============================================================================
# STEP 2: Create Virtual Environments
# ============================================================================
Write-Step "STEP 2: Creating Python Virtual Environments" "Info"

# Backend venv
Write-Host "Creating backend venv..." -ForegroundColor Cyan
$backendVenv = "backend\.venv"
if (Test-Path $backendVenv) {
    Write-Host "  Virtual environment already exists: $backendVenv" -ForegroundColor Yellow
} else {
    & python -m venv $backendVenv
    Write-Host "  ✓ Backend venv created" -ForegroundColor Green
}

# Agents venv
Write-Host "Creating agents venv..." -ForegroundColor Cyan
$agentsVenv = "agents\.venv"
if (Test-Path $agentsVenv) {
    Write-Host "  Virtual environment already exists: $agentsVenv" -ForegroundColor Yellow
} else {
    & python -m venv $agentsVenv
    Write-Host "  ✓ Agents venv created" -ForegroundColor Green
}

Write-Host "✓ Virtual environments ready" -ForegroundColor Green

# ============================================================================
# STEP 3: Backend Setup (ROCm PyTorch)
# ============================================================================
Write-Step "STEP 3: Installing Backend Dependencies (with ROCm PyTorch)" "Info"

Write-Host "Activating backend virtual environment..." -ForegroundColor Cyan
$backendActivate = if ($IsWindows) {
    "$backendVenv\Scripts\Activate.ps1"
} else {
    "$backendVenv/bin/Activate.ps1"
}

& $backendActivate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& python -m pip install --upgrade pip wheel setuptools

Write-Host "Installing backend dependencies with uv..." -ForegroundColor Cyan
Write-Host "  (This includes PyTorch ROCm ~2GB download)" -ForegroundColor Yellow

# Install with uv - it's much faster than pip
& uv pip install -r backend/requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/rocm5.7

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: Some packages may have failed to install" -ForegroundColor Yellow
    Write-Host "  Retrying with standard pip..." -ForegroundColor Yellow
    & python -m pip install -r backend/requirements.txt `
        --extra-index-url https://download.pytorch.org/whl/rocm5.7
}

Write-Host "✓ Backend dependencies installed" -ForegroundColor Green

# Verify key packages
Write-Host "Verifying key packages..." -ForegroundColor Cyan
$backendPackages = @("torch", "transformers", "fastapi", "pydantic")
foreach ($pkg in $backendPackages) {
    $result = & python -c "import $pkg; print($pkg)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $pkg" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg (missing)" -ForegroundColor Red
    }
}

# ============================================================================
# STEP 4: Agents Setup
# ============================================================================
Write-Step "STEP 4: Installing Agents Dependencies" "Info"

# Deactivate backend venv first
if (Test-CommandExists "deactivate") {
    & deactivate
}

# Activate agents venv
Write-Host "Activating agents virtual environment..." -ForegroundColor Cyan
$agentsActivate = if ($IsWindows) {
    "$agentsVenv\Scripts\Activate.ps1"
} else {
    "$agentsVenv/bin/Activate.ps1"
}

& $agentsActivate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& python -m pip install --upgrade pip wheel setuptools

Write-Host "Installing agents dependencies with uv..." -ForegroundColor Cyan
& uv pip install -r agents/requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: Some packages may have failed to install" -ForegroundColor Yellow
    Write-Host "  Retrying with standard pip..." -ForegroundColor Yellow
    & python -m pip install -r agents/requirements.txt
}

Write-Host "✓ Agents dependencies installed" -ForegroundColor Green

# Verify key packages
Write-Host "Verifying key packages..." -ForegroundColor Cyan
$agentPackages = @("crewai", "pydantic", "pdf2image", "Pillow")
foreach ($pkg in $agentPackages) {
    $importName = $pkg -replace "-", "_"
    $result = & python -c "import $importName; print(`'$pkg`')"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $pkg" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg (missing)" -ForegroundColor Red
    }
}

# ============================================================================
# STEP 5: Environment Configuration
# ============================================================================
Write-Step "STEP 5: Setting up Environment Configuration" "Info"

$envFile = ".env"
if (Test-Path $envFile) {
    Write-Host ".env file already exists" -ForegroundColor Yellow
} else {
    Write-Host "Creating .env configuration file..." -ForegroundColor Cyan
    $envContent = @"
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database (for future use)
DATABASE_URL=postgresql://user:password@localhost/medical_etl

# Orchestration LLM (optional)
OPENAI_API_KEY=
ORCHESTRATOR_MODEL=gpt-4o-mini

# Model Cache
HF_HOME=./model_cache
"@
    
    Set-Content -Path $envFile -Value $envContent
    Write-Host "  ✓ .env created with default values" -ForegroundColor Green
    Write-Host "  Edit .env to configure API credentials" -ForegroundColor Yellow
}

# ============================================================================
# STEP 6: Integration Tests
# ============================================================================
if (-not $SkipTests) {
    Write-Step "STEP 6: Running Integration Tests" "Info"
    
    # Must run tests in agents environment
    Write-Host "Running integration test suite..." -ForegroundColor Cyan
    
    $testFlags = if ($VerboseLogging) { "--verbose" } else { "" }
    & python integration_test.py $testFlags
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ All tests passed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Some tests failed - check output above" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping integration tests (use -SkipTests=`$false to run)" -ForegroundColor Yellow
}

# ============================================================================
# STEP 7: Summary & Next Steps
# ============================================================================
Write-Step "SETUP COMPLETE" "Success"

Write-Host ""
Write-Host "✓ Backend virtual environment: $backendVenv" -ForegroundColor Green
Write-Host "✓ Agents virtual environment:  $agentsVenv" -ForegroundColor Green
Write-Host "✓ Configuration file:          $envFile" -ForegroundColor Green
Write-Host "✓ Project root:                $projectRoot" -ForegroundColor Green
Write-Host ""

Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Start the FastAPI server (in new terminal):" -ForegroundColor Cyan
Write-Host "   $backendActivate" -ForegroundColor White
Write-Host "   python backend/main.py" -ForegroundColor White
Write-Host ""
Write-Host "2. Run the agent pipeline (in another terminal):" -ForegroundColor Cyan
Write-Host "   $agentsActivate" -ForegroundColor White
Write-Host "   python agents/crew.py medical_report.jpg" -ForegroundColor White
Write-Host ""
Write-Host "3. Test the API (in another terminal):" -ForegroundColor Cyan
Write-Host "   curl -X POST http://localhost:8000/extract -F 'file=@medical_report.jpg'" -ForegroundColor White
Write-Host ""
Write-Host "4. View API documentation:" -ForegroundColor Cyan
Write-Host "   http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

# Deactivate venv
& deactivate

Write-Host "Setup script completed!" -ForegroundColor Green

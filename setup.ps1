#Requires -Version 5.1
<#
.SYNOPSIS
    Phase 1 one-shot setup for Stream-to-Shorts MAS.

.DESCRIPTION
    1. Creates .venv with the system Python (3.10+)
    2. Installs requirements.txt (CPU-only torch via PyTorch wheel index)
    3. Checks for FFmpeg binary on PATH
    4. Pulls required Ollama models (qwen2.5:7b-instruct-q4_k_m, llama3.2:3b)
    5. Registers the custom Ollama model 'shorts-llm' with num_ctx 8192
    6. Runs verify_setup.py to confirm everything works
#>

$PROJECT_ROOT = $PSScriptRoot
$VENV_DIR     = Join-Path $PROJECT_ROOT ".venv"
$VENV_PYTHON  = Join-Path $VENV_DIR "Scripts\python.exe"
$VENV_PIP     = Join-Path $VENV_DIR "Scripts\pip.exe"

function Write-Step { param([string]$msg) Write-Host "" ; Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$msg) Write-Host "    [OK]   $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$msg) Write-Host "    [FAIL] $msg" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
Write-Step "Checking Python version"
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Python not found on PATH. Install Python 3.10+ and re-run."
    exit 1
}
$parts = $pyVersion.ToString().Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Write-Fail "Python $pyVersion detected - 3.10+ required."
    exit 1
}
Write-Ok "Python $pyVersion"

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
Write-Step "Creating virtual environment at .venv"
if (Test-Path $VENV_PYTHON) {
    Write-Warn ".venv already exists and has python.exe - skipping creation."
} else {
    python -m venv "$VENV_DIR"
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "python -m venv failed (exit $LASTEXITCODE)."
        exit 1
    }
    if (-not (Test-Path $VENV_PYTHON)) {
        Write-Fail ".venv was not created - check that Python is not restricted."
        exit 1
    }
    Write-Ok ".venv created at $VENV_DIR"
}

# ---------------------------------------------------------------------------
# 3. Install dependencies (CPU-only torch)
# ---------------------------------------------------------------------------
Write-Step "Upgrading pip inside .venv"
& "$VENV_PIP" install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Warn "pip upgrade returned $LASTEXITCODE (non-fatal)." }

Write-Step "Installing requirements.txt (CPU-only torch wheel index)"
& "$VENV_PIP" install `
    -r (Join-Path $PROJECT_ROOT "requirements.txt") `
    --extra-index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed (exit $LASTEXITCODE). Check output above."
    exit 1
}
Write-Ok "All packages installed"

# ---------------------------------------------------------------------------
# 4. FFmpeg binary check
# ---------------------------------------------------------------------------
Write-Step "Checking for FFmpeg binary on PATH"
$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpegCmd) {
    Write-Warn "FFmpeg not found on PATH."
    Write-Warn "Install with:  winget install Gyan.FFmpeg"
    Write-Warn "(Required before Phase 2 - non-fatal for Phase 1)"
} else {
    $ffmpegVer = (ffmpeg -version 2>&1 | Select-Object -First 1)
    Write-Ok "FFmpeg: $ffmpegVer"
}

# ---------------------------------------------------------------------------
# 5. Pull Ollama models
# ---------------------------------------------------------------------------
Write-Step "Checking installed Ollama models"
$modelList = (ollama list 2>&1) | Out-String

Write-Step "Pulling qwen2.5:7b-instruct-q4_k_m (~4.7 GB)"
if ($modelList -match "qwen2\.5:7b-instruct-q4_k_m") {
    Write-Warn "qwen2.5:7b-instruct-q4_k_m already present - skipping."
} else {
    ollama pull qwen2.5:7b-instruct-q4_k_m
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "ollama pull qwen2.5:7b-instruct-q4_k_m failed (exit $LASTEXITCODE)."
        exit 1
    }
    Write-Ok "qwen2.5:7b-instruct-q4_k_m pulled"
}

Write-Step "Pulling llama3.2:3b (~2 GB, fallback model)"
if ($modelList -match "llama3\.2:3b") {
    Write-Warn "llama3.2:3b already present - skipping."
} else {
    ollama pull llama3.2:3b
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "ollama pull llama3.2:3b failed (exit $LASTEXITCODE) - non-fatal, retry later."
    } else {
        Write-Ok "llama3.2:3b pulled"
    }
}

# ---------------------------------------------------------------------------
# 6. Register custom Ollama model 'shorts-llm' (num_ctx 8192)
# ---------------------------------------------------------------------------
Write-Step "Registering Ollama model 'shorts-llm' (num_ctx 8192)"
if ($modelList -match "shorts-llm") {
    Write-Warn "shorts-llm already registered - skipping."
} else {
    $modelfilePath = Join-Path $PROJECT_ROOT "config\Modelfile.qwen25"
    ollama create shorts-llm -f "$modelfilePath"
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "ollama create shorts-llm failed (exit $LASTEXITCODE)."
        exit 1
    }
    Write-Ok "shorts-llm registered"
}

# ---------------------------------------------------------------------------
# 7. Run verification script
# ---------------------------------------------------------------------------
Write-Step "Running verify_setup.py"
& "$VENV_PYTHON" (Join-Path $PROJECT_ROOT "verify_setup.py")
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Verification reported issues - see output above."
}

Write-Host ""
Write-Host "[Phase 1 complete] Environment is ready." -ForegroundColor Green

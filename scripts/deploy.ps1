<#
.SYNOPSIS
    Automated Production Deployment & Health Verification Script for AI Audio Forensics.
.DESCRIPTION
    Validates Python 3.14 environment, sets up directory structures, initializes SQLite
    database with WAL mode, verifies model artifact hashes, and bootstraps initial admin user.
#>

[CmdletBinding()]
param(
    [string]$PythonExe = ".venv314\Scripts\python.exe",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🛡️ AI AUDIO FORENSICS — PRODUCTION DEPLOYMENT & VERIFICATION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Python Virtual Environment
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment Python executable not found at $PythonExe. Please initialize .venv314 first."
}
$pyVer = & $PythonExe -c "import sys; print(sys.version.split()[0])"
Write-Host "[+] Verified Python runtime: $pyVer ($PythonExe)" -ForegroundColor Green

# 2. Verify and Create Standard Directories
$dirs = @(
    "storage\original_evidence",
    "storage\processed_audio",
    "storage\visualizations",
    "storage\reports",
    "storage\temp",
    "database\backups",
    "logs",
    "models",
    "config"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Host "[+] Created directory: $d" -ForegroundColor Yellow
    }
}
Write-Host "[+] Directory hierarchy verified." -ForegroundColor Green

# 3. Initialize SQLite Database & WAL Mode
Write-Host "[+] Initializing SQLite database with WAL concurrency..." -ForegroundColor Yellow
& $PythonExe -c "from database.database import init_db; init_db(); print('Database initialized successfully with WAL mode.')"

# 4. Verify Model Artifact Hashes
Write-Host "[+] Verifying machine learning model seals..." -ForegroundColor Yellow
& $PythonExe -c "
from src.security import verify_model_artifact
from pathlib import Path

models_dir = Path('models')
for m in models_dir.glob('*.joblib'):
    verify_model_artifact(m)
print('All model artifacts verified intact against cryptographic manifest.')
"

# 5. Bootstrap Initial Admin User if Empty
Write-Host "[+] Checking user provisioning status..." -ForegroundColor Yellow
& $PythonExe -c "
from database.database import ForensicDatabase
from src.auth import bootstrap_admin_user

db = ForensicDatabase()
users = db.list_users()
if not users:
    u, pwd = bootstrap_admin_user(db, username='admin')
    print(f'BOOTSTRAP NOTICE: Initial Admin User created (Username: admin, Password: {pwd})')
else:
    print(f'User directory intact ({len(users)} registered user(s)).')
"

# 6. Run Test Suite Validation (Optional)
if (-not $SkipTests) {
    Write-Host "[+] Running automated test suite regression check..." -ForegroundColor Yellow
    & $PythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Deployment verification failed: One or more automated tests failed."
    }
    Write-Host "[+] Automated regression suite passed cleanly." -ForegroundColor Green
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "✅ PRODUCTION DEPLOYMENT COMPLETE & VERIFIED" -ForegroundColor Green
Write-Host "To start the dashboard, execute: .\scripts\start.ps1" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

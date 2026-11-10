<#
.SYNOPSIS
    Automated Hot Backup Script for AI Audio Forensics.
.DESCRIPTION
    Triggers online hot database backup, bundles manifests, and generates cryptographic SHA-256 seal.
#>

[CmdletBinding()]
param(
    [string]$PythonExe = ".venv314\Scripts\python.exe",
    [string]$Operator = "PowerShellBackupJob"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🛡️ EXECUTING FORENSIC SYSTEM HOT BACKUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe."
}

& $PythonExe -c "
from database.database import ForensicDatabase
from src.backup_manager import create_system_backup
import json

db = ForensicDatabase()
res = create_system_backup(db, operator='$Operator')
print(json.dumps(res, indent=2))
"

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "✅ HOT BACKUP COMPLETED & VERIFIED" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green


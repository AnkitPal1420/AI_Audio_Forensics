<#
.SYNOPSIS
    Secure Startup Script for AI Audio Forensics Dashboard.
.DESCRIPTION
    Launches the Streamlit forensic laboratory dashboard strictly bound to local loopback (127.0.0.1).
    Prevents unauthorized wildcard LAN exposure (SEC-04).
#>

[CmdletBinding()]
param(
    [string]$PythonExe = ".venv314\Scripts\python.exe",
    [int]$Port = 8501,
    [string]$Address = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🛡️ STARTING AI AUDIO FORENSICS LABORATORY DASHBOARD" -ForegroundColor Cyan
Write-Host "Bound Address: http://${Address}:${Port} (Local Loopback Enforcement)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe. Run .\scripts\deploy.ps1 first."
}

# Launch Streamlit with enforced loopback binding
& $PythonExe -m streamlit run app.py `
    --server.address $Address `
    --server.port $Port `
    --server.headless true `
    --browser.gatherUsageStats false

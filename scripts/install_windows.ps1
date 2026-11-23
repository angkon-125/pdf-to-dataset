# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -m venv .venv
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m venv .venv
} else {
    throw 'Python 3.10+ was not found. Install Python, then re-run this script.'
}
$python = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host ''
Write-Host 'To start the Angkon UI:'
Write-Host '  .\.venv\Scripts\python.exe -m streamlit run app.py'
Write-Host 'Optional OCR requires a separate Tesseract installation.'

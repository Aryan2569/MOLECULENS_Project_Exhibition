# MOLECULENS - one-shot Windows setup
# Run from inside the moleculens folder:
#     .\setup_windows.ps1
#
# If PowerShell blocks it, run this once first:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"

Write-Host "`n=== [1/5] Checking Python ===" -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found on PATH. Install from python.org and tick 'Add python.exe to PATH'." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== [2/5] Checking GPU (optional) ===" -ForegroundColor Cyan
try {
    nvidia-smi | Select-Object -First 12
} catch {
    Write-Host "nvidia-smi not found - that's OK, everything runs on CPU too (just slower)." -ForegroundColor Yellow
}

Write-Host "`n=== [3/5] Creating virtual environment ===" -ForegroundColor Cyan
if (Test-Path ".\venv") {
    Write-Host "venv already exists, reusing it."
} else {
    python -m venv venv
}
.\venv\Scripts\Activate.ps1
Write-Host "venv activated."

Write-Host "`n=== [4/5] Installing dependencies (this takes a few minutes) ===" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "`n=== [5/5] Running smoke test ===" -ForegroundColor Cyan
python smoketest.py

Write-Host "`nSetup complete. Next:" -ForegroundColor Green
Write-Host "  python data\download.py"
Write-Host "  cd src"
Write-Host "  python train.py --dataset ESOL --model gcn --epochs 50"

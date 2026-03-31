# dev.ps1 - start backend API + frontend Vite in separate windows (PS 5.1 compatible)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\dev.ps1

param(
    [int]$ApiPort = 8080,
    [int]$FePort = 5173,
    [switch]$NoBrowser,
    [string]$CondaEnv = "mdm",
    [switch]$AllowCpuFallback
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step([string]$msg) {
    Write-Host ("[dev] " + $msg) -ForegroundColor Cyan
}

function Resolve-Python([string]$PreferredCondaEnv) {
    # 1) Explicit override
    if ($env:HILDANEXT_PYTHON -and (Test-Path $env:HILDANEXT_PYTHON)) {
        return $env:HILDANEXT_PYTHON
    }

    # 2) Preferred env candidates (works even without `conda activate`)
    $envCandidates = @(
        "$env:USERPROFILE\.conda\envs\$PreferredCondaEnv\python.exe",
        "C:\ProgramData\miniconda3\envs\$PreferredCondaEnv\python.exe",
        "C:\ProgramData\anaconda3\envs\$PreferredCondaEnv\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$PreferredCondaEnv\python.exe",
        "$env:USERPROFILE\anaconda3\envs\$PreferredCondaEnv\python.exe"
    )
    foreach ($c in $envCandidates) {
        if (Test-Path $c) {
            return $c
        }
    }

    # 3) Active conda env python (if it is not base)
    if ($env:CONDA_PREFIX) {
        $activeCondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        $activeEnvName = Split-Path -Leaf $env:CONDA_PREFIX
        if ((Test-Path $activeCondaPython) -and ($activeEnvName -ne "base")) {
            return $activeCondaPython
        }
    }

    # 4) Python on PATH
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return $pyCmd.Source
    }

    # 5) Last-resort global installs
    $candidates = @(
        "C:\ProgramData\miniconda3\python.exe",
        "C:\ProgramData\anaconda3\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return $c
        }
    }
    return $null
}

# 0) Resolve python path
$pythonExe = Resolve-Python -PreferredCondaEnv $CondaEnv

if (-not $pythonExe) {
    throw "python.exe not found. Activate conda env or set HILDANEXT_PYTHON."
}

Write-Step "Python = $pythonExe"
Write-Step "Preferred conda env = $CondaEnv"
if ($env:CONDA_PREFIX) {
    Write-Step "CONDA_PREFIX = $env:CONDA_PREFIX"
}

# 0.1) Quick CUDA diagnostic from selected python
try {
    $cudaDiag = & $pythonExe -c "import json; import torch; print(json.dumps({'python': __import__('sys').executable, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available()), 'gpu_count': int(torch.cuda.device_count())}))"
    Write-Step "Torch diag = $cudaDiag"
    if ($cudaDiag -and ($cudaDiag -like '*\"cuda_available\": false*')) {
        if ($AllowCpuFallback) {
            Write-Host "[dev] WARNING: selected python has CUDA unavailable. Backend will run in CPU fallback." -ForegroundColor Yellow
        } else {
            throw "CUDA unavailable in selected python. Stop to avoid CPU fallback. Use -AllowCpuFallback to bypass."
        }
    }
} catch {
    if ($_.Exception -and $_.Exception.Message -like "CUDA unavailable in selected python*") {
        throw
    }
    Write-Host "[dev] WARNING: unable to run torch CUDA diagnostic with selected python." -ForegroundColor Yellow
}

# 1) Kill old API listeners on the same port
$oldPids = netstat -ano |
    Select-String ":$ApiPort " |
    ForEach-Object { ($_ -split "\s+")[-1] } |
    Where-Object { $_ -match "^\d+$" } |
    Select-Object -Unique

foreach ($procId in $oldPids) {
    Write-Step "Kill old API process PID $procId on :$ApiPort"
    Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
}

# 2) Start backend API
$apiScript = Join-Path $root "start_server.py"
if (-not (Test-Path $apiScript)) {
    throw "start_server.py not found at $apiScript"
}

Write-Step "Starting API backend at http://127.0.0.1:$ApiPort"
Start-Process "powershell.exe" `
    -ArgumentList @("-NoExit", "-Command", "& '$pythonExe' '$apiScript'") `
    -WindowStyle Normal

# 3) Wait for API listener (max 60s)
Write-Step "Waiting for API on :$ApiPort ..."
$deadline = (Get-Date).AddSeconds(60)
$apiReady = $false

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1500
    $listening = netstat -ano | Select-String "127.0.0.1:$ApiPort.*LISTENING"
    if ($listening) {
        $apiReady = $true
        break
    }
}

if ($apiReady) {
    Write-Step "API ready on :$ApiPort"
} else {
    Write-Host "[dev] WARNING: API did not start within 60s - frontend will still launch." -ForegroundColor Yellow
}

# 4) Start frontend Vite
$feDir = Join-Path $root "frontend"
if (-not (Test-Path $feDir)) {
    throw "frontend folder not found at $feDir"
}

Write-Step "Starting Vite frontend at http://localhost:$FePort"
Start-Process "powershell.exe" `
    -ArgumentList @("-NoExit", "-Command", "Set-Location '$feDir'; npm run dev") `
    -WindowStyle Normal

# 5) Open browser
if (-not $NoBrowser) {
    Start-Sleep -Seconds 3
    $url = "http://localhost:$FePort/chat"
    Write-Step "Opening $url"
    Start-Process $url
}

Write-Host ""
Write-Host "HildaNext dev environment started" -ForegroundColor Green
Write-Host "Frontend   http://localhost:$FePort/chat" -ForegroundColor Green
Write-Host "API        http://127.0.0.1:$ApiPort" -ForegroundColor Green
Write-Host "Legacy WSD http://localhost:$FePort/legacy/wsd" -ForegroundColor Green

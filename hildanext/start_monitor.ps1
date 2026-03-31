# HildaNext WSD Monitor launcher
# Starts the FastAPI backend + Vite frontend dev server so you can monitor a live WSD run.
# Usage (from hildanext/ directory):
#   .\start_monitor.ps1
#   .\start_monitor.ps1 -ApiPort 8081 -FrontendPort 5174
#
# The WSD TRAINING run must be started separately (e.g. in another terminal):
#   .\start_wsd_full_logs.ps1 -AllowExternalDolmaPath
#
# Once both are running, open: http://localhost:5173/?scenario=live_wsd_run
# The page polls /api/frontend/wsd every 5 seconds for fresh metrics.

param(
    [string]$CondaEnv     = "mdm",
    [string]$Config       = "runs/configs/llada21_dolma_wsd_only.json",
    [string]$ApiHost      = "127.0.0.1",
    [int]$ApiPort         = 8080,
    [int]$FrontendPort    = 5173
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$frontendDir = Join-Path $root "frontend"
$envLocalFile = Join-Path $frontendDir ".env.local"

function Write-Line([string]$msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host ("[monitor] [$ts] $msg")
}

Write-Line "=== HildaNext WSD Monitor ==="
Write-Line "Config:   $Config"
Write-Line "API:      http://${ApiHost}:${ApiPort}"
Write-Line "Frontend: http://localhost:${FrontendPort}/?scenario=live_wsd_run"

# ------------------------------------------------------------------
# Step 1: Write .env.local so Vite picks up VITE_USE_BACKEND=true
# ------------------------------------------------------------------
Write-Line "Writing frontend/.env.local (VITE_USE_BACKEND=true)..."
"VITE_USE_BACKEND=true" | Set-Content -Path $envLocalFile -Encoding UTF8

# ------------------------------------------------------------------
# Step 2: Start FastAPI server (background, conda env mdm)
# ------------------------------------------------------------------
Write-Line "Starting FastAPI API server..."
$condaBase = conda info --base 2>$null
$condaExe  = if ($condaBase) { Join-Path $condaBase "Scripts\conda.exe" } else { "conda" }
$apiProcess = Start-Process `
    -FilePath $condaExe `
    -ArgumentList "run", "-n", $CondaEnv, "--no-capture-output",
                  "python", "-m", "hildanext.cli", "serve",
                  "--config", $Config, "--host", $ApiHost, "--port", $ApiPort `
    -PassThru `
    -NoNewWindow

Write-Line "API server PID: $($apiProcess.Id)"

# ------------------------------------------------------------------
# Step 3: Give the API a couple of seconds to bind
# ------------------------------------------------------------------
Start-Sleep -Seconds 3

# ------------------------------------------------------------------
# Step 4: Start Vite dev server (background)
# ------------------------------------------------------------------
Write-Line "Starting Vite dev server (port $FrontendPort)..."
$npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)?.Source ?? "npm.cmd"
$frontendProcess = Start-Process `
    -FilePath $npm `
    -ArgumentList "run", "dev", "--", "--port", $FrontendPort `
    -WorkingDirectory $frontendDir `
    -PassThru `
    -NoNewWindow

Write-Line "Frontend PID: $($frontendProcess.Id)"
Start-Sleep -Seconds 4

# ------------------------------------------------------------------
# Step 5: Open browser
# ------------------------------------------------------------------
$url = "http://localhost:${FrontendPort}/?scenario=live_wsd_run"
Write-Line "Opening $url"
try { Start-Process $url } catch { Write-Line "Could not open browser automatically." }

Write-Line "Both servers running. Press ENTER to stop."
Write-Line "TIP: The WSD training should be started separately in another terminal."

try {
    $null = Read-Host
} finally {
    Write-Line "Stopping servers..."
    Stop-Process -Id $apiProcess.Id     -ErrorAction SilentlyContinue
    Stop-Process -Id $frontendProcess.Id -ErrorAction SilentlyContinue

    Write-Line "Removing frontend/.env.local..."
    Remove-Item $envLocalFile -ErrorAction SilentlyContinue

    Write-Line "Done."
}

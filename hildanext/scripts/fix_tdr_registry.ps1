<# 
  TDR Registry Fix for WSD Training — GTX 1080 (Pascal)
  
  Sets TdrDelay and TdrDdiDelay to 60 seconds to prevent GPU timeout
  during long backward passes (composite_llada20 BDLM phases).
  
  REQUIRES: Run as Administrator
  REQUIRES: Reboot after applying
  
  Caveat: Microsoft documents these keys as for driver development/testing.
  This is a pragmatic workaround for training on consumer GPU, not a general fix.
  
  See: STAGE0_MASTER_PLAN.md Fix 0.3
#>

$ErrorActionPreference = "Stop"
$key = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as Administrator, then re-run this script."
    exit 1
}

# Show current values
Write-Host "`n=== Current TDR Settings ===" -ForegroundColor Cyan
try {
    $current = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
    $delay = $current.TdrDelay
    $ddiDelay = $current.TdrDdiDelay
    Write-Host "  TdrDelay:    $(if ($delay) { $delay } else { '(default: 2)' })"
    Write-Host "  TdrDdiDelay: $(if ($ddiDelay) { $ddiDelay } else { '(default: 5)' })"
} catch {
    Write-Host "  Could not read current values (registry key may not exist yet)"
}

# Apply new values
Write-Host "`n=== Applying TDR Fix ===" -ForegroundColor Yellow
Set-ItemProperty -Path $key -Name TdrDelay    -Value 60 -Type DWord
Set-ItemProperty -Path $key -Name TdrDdiDelay -Value 60 -Type DWord

# Verify
Write-Host "`n=== Verify ===" -ForegroundColor Green
$updated = Get-ItemProperty -Path $key
Write-Host "  TdrDelay:    $($updated.TdrDelay)"
Write-Host "  TdrDdiDelay: $($updated.TdrDdiDelay)"

if ($updated.TdrDelay -eq 60 -and $updated.TdrDdiDelay -eq 60) {
    Write-Host "`nSUCCESS: TDR timeout set to 60 seconds." -ForegroundColor Green
    Write-Host "IMPORTANT: You must REBOOT for changes to take effect." -ForegroundColor Yellow
    Write-Host "`nAfter reboot, run the benchmark suite:" -ForegroundColor Cyan
    Write-Host "  conda activate mdm"
    Write-Host "  cd e:\DIFFUSION\HildaNext\hildanext"
    Write-Host "  python test\test_wsd_benchmark_suite.py"
} else {
    Write-Host "`nERROR: Values not set correctly!" -ForegroundColor Red
    exit 1
}

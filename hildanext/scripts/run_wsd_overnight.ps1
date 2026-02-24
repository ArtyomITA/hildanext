# Resume-safe WSD runner loop for overnight training.
# Re-runs run-wsd on failure and resumes from latest checkpoint.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\run_wsd_overnight.ps1
param(
 [string]$CondaEnv="mdm",
 [string]$ConfigPath="runs/configs/llada21_dolma_wsd_only.json",
 [int]$RetryDelaySec=90,
 [int]$MaxRestarts=0
)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
function Log([string]$m){ $ts=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); Write-Host "[$ts] $m" }
if(Get-Command conda -ErrorAction SilentlyContinue){
 $hook=conda shell.powershell hook | Out-String
 Invoke-Expression $hook
 conda activate $CondaEnv
}else{
 throw "conda_not_found_in_path"
}
if($env:CONDA_DEFAULT_ENV -ne $CondaEnv){ throw "conda_activate_failed expected=$CondaEnv got=$($env:CONDA_DEFAULT_ENV)" }
$pyExe=Join-Path $env:CONDA_PREFIX "python.exe"
if(-not (Test-Path $pyExe)){ throw "conda_python_not_found path=$pyExe" }
$restart=0
while($true){
 $restart++
 Log ("START run-wsd attempt="+$restart)
 & $pyExe -u -m hildanext.cli run-wsd --config $ConfigPath --skip-preflight --no-archive
 $code=$LASTEXITCODE
 if($code -eq 0){
  Log "DONE run-wsd exit=0"
  break
 }
 Log ("FAIL run-wsd exit="+$code)
 if($MaxRestarts -gt 0 -and $restart -ge $MaxRestarts){
  throw "max_restarts_reached attempts=$restart"
 }
 Log ("SLEEP retry_delay_sec="+$RetryDelaySec)
 Start-Sleep -Seconds $RetryDelaySec
}

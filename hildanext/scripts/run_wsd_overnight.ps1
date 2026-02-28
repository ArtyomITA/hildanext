# Resume-safe WSD runner loop for overnight training.
# Re-runs run-wsd on failure and resumes from latest checkpoint.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\run_wsd_overnight.ps1
param(
 [string]$CondaEnv="mdm",
 [string]$ConfigPath="runs/configs/llada21_dolma_wsd_only.json",
 [int]$RetryDelaySec=90,
 [int]$MaxRestarts=0,
 [switch]$EnableNonBlockingFallback,
 [string]$FallbackConfigPath="runs/configs/llada21_dolma_wsd_only.fallback.json"
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
$cfgPathResolved=$ConfigPath
if(-not [System.IO.Path]::IsPathRooted($cfgPathResolved)){
 $cfgPathResolved=Join-Path $root $cfgPathResolved
}
$cfgPathResolved=[string](Resolve-Path -Path $cfgPathResolved -ErrorAction SilentlyContinue)
if([string]::IsNullOrWhiteSpace($cfgPathResolved)){
 throw "config_not_found path=$ConfigPath"
}
Log ("CONFIG="+$cfgPathResolved)
$activeConfigPath=$cfgPathResolved
$fallbackApplied=$false
$restart=0
while($true){
 $restart++
 Log ("START run-wsd attempt="+$restart+" config="+$activeConfigPath)
 & $pyExe -u -m hildanext.cli run-wsd --config $activeConfigPath --skip-preflight --no-archive --skip-dolma-prep
 $code=$LASTEXITCODE
 if($code -eq 0){
  Log "DONE run-wsd exit=0"
  break
 }
 Log ("FAIL run-wsd exit="+$code)
 if($EnableNonBlockingFallback -and (-not $fallbackApplied)){
  try{
   $fallbackTarget=$FallbackConfigPath
   if(-not [System.IO.Path]::IsPathRooted($fallbackTarget)){
    $fallbackTarget=Join-Path $root $fallbackTarget
   }
   $cfgObj=Get-Content -Path $cfgPathResolved -Raw -Encoding UTF8 | ConvertFrom-Json
   if($null -eq $cfgObj.runtime){
    $cfgObj | Add-Member -MemberType NoteProperty -Name runtime -Value ([pscustomobject]@{})
   }
   $cfgObj.runtime.strict_fallbacks=$false
   $wl=@()
   if($null -ne $cfgObj.runtime.fallback_whitelist){
    $wl=@($cfgObj.runtime.fallback_whitelist)
   }
   $wl=@($wl+@("flash_attention_unavailable","numpy_dll_unavailable","dinfer_missing") | Select-Object -Unique)
   $cfgObj.runtime.fallback_whitelist=$wl
   $cfgObj | ConvertTo-Json -Depth 100 | Set-Content -Path $fallbackTarget -Encoding UTF8
   $fallbackResolved=[string](Resolve-Path -Path $fallbackTarget -ErrorAction Stop)
   $activeConfigPath=$fallbackResolved
   $fallbackApplied=$true
   Log ("FALLBACK_APPLIED strict_fallbacks=false config="+$activeConfigPath)
   continue
  }catch{
   Log ("FALLBACK_APPLY_FAILED reason="+$_.Exception.Message)
  }
 }
 if($MaxRestarts -gt 0 -and $restart -ge $MaxRestarts){
  throw "max_restarts_reached attempts=$restart"
 }
 Log ("SLEEP retry_delay_sec="+$RetryDelaySec)
 Start-Sleep -Seconds $RetryDelaySec
}

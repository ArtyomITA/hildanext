# Standalone Dolma dataset preparation for Stage 0 WSD.
# Runs dolma-prep in three checkpointed steps; each step is skipped if already done.
# Run this BEFORE run_wsd_overnight.ps1.
#
# Steps:
#   1. dolma-prep   → data/tokenized/train.jsonl + eval.jsonl + .npy shards
#   2. dolma-verify → validates artifacts and doc-boundary signal
#   3. DONE stamp   → writes runs/cache/prep_done.stamp so overnight can trust it
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_dolma_prep.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_dolma_prep.ps1 -Force
param(
 [string]$CondaEnv="mdm",
 [string]$ConfigPath="runs/configs/llada21_dolma_wsd_only.json",
 [switch]$Force,           # re-run all steps even if stamp exists
 [int]$Workers=4           # tokenizer parallel workers (CPU only; no GPU needed)
)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$stamp=Get-Date -Format "yyyyMMdd_HHmmss"
$consoleDir=Join-Path $root "runs/console"
New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
$transcript=Join-Path $consoleDir ("dolma_prep_"+$stamp+".log")
Start-Transcript -Path $transcript -Force | Out-Null

function Log([string]$m){ $ts=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); Write-Host "[$ts] $m" }

try{
 # ── Conda activation ────────────────────────────────────────────────────────
 if(Get-Command conda -ErrorAction SilentlyContinue){
  $hook=conda shell.powershell hook | Out-String
  Invoke-Expression $hook
  conda activate $CondaEnv
 }else{ throw "conda_not_found_in_path" }
 if($env:CONDA_DEFAULT_ENV -ne $CondaEnv){ throw "conda_activate_failed expected=$CondaEnv got=$($env:CONDA_DEFAULT_ENV)" }
 $pyExe=Join-Path $env:CONDA_PREFIX "python.exe"
 if(-not (Test-Path $pyExe)){ throw "conda_python_not_found path=$pyExe" }
 $env:PYTHONUNBUFFERED="1"

 # ── Resolve config ──────────────────────────────────────────────────────────
 $cfgResolved=$ConfigPath
 if(-not [System.IO.Path]::IsPathRooted($cfgResolved)){ $cfgResolved=Join-Path $root $cfgResolved }
 $cfgResolved=[string](Resolve-Path -Path $cfgResolved -ErrorAction SilentlyContinue)
 if([string]::IsNullOrWhiteSpace($cfgResolved)){ throw "config_not_found path=$ConfigPath" }
 Log ("CONFIG="+$cfgResolved)

 # ── Checkpoint paths (derived from config) ──────────────────────────────────
 $cfgObj=Get-Content -Path $cfgResolved -Raw -Encoding UTF8 | ConvertFrom-Json
 $tokenizedDir=[string]$cfgObj.paths.tokenized_dir
 $trainTok=Join-Path $tokenizedDir "train.jsonl"
 $evalTok=Join-Path $tokenizedDir "eval.jsonl"
 $dolmaRaw=[string]$cfgObj.data.dolma_path          # e.g. .../raw
 $dolmaBase=Split-Path -Parent $dolmaRaw             # e.g. dolma_v1_6_sample_.../
 if((Split-Path -Leaf $dolmaRaw).ToLower() -ne "raw"){ $dolmaBase=$dolmaRaw }
 $numpyTokensDir=Join-Path $dolmaBase "tokens"
 $numpyDocDir=Join-Path $dolmaBase "doc_index"
 $metaFile=Join-Path $dolmaBase "meta.json"
 $doneStamp=Join-Path $root "runs/cache/prep_done.stamp"

 Log ("TOKENIZED_DIR="+$tokenizedDir)
 Log ("DOLMA_BASE="+$dolmaBase)

 # ── Check if already fully done ─────────────────────────────────────────────
 $tokenizedOk=(Test-Path $trainTok) -and (Test-Path $evalTok)
 $shardsOk=(Test-Path $metaFile) -and `
           ((Get-ChildItem -Path $numpyTokensDir -Filter "tokens_*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) -and `
           ((Get-ChildItem -Path $numpyDocDir   -Filter "doc_index_*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
 $stampOk=Test-Path $doneStamp

 if($stampOk -and $tokenizedOk -and $shardsOk -and (-not $Force)){
  Log "PREP_ALREADY_DONE stamp exists and all artifacts present. Use -Force to re-run."
  Log ("STAMP="+$doneStamp)
  exit 0
 }

 # ── STEP 1: dolma-prep (tokenize + build .npy shards) ───────────────────────
 $needPrep=(-not $tokenizedOk) -or (-not $shardsOk) -or $Force
 if($needPrep){
  Log "STEP=1 action=dolma-prep tokenized_ok=$tokenizedOk shards_ok=$shardsOk"
  $t0=[System.Diagnostics.Stopwatch]::StartNew()
  & $pyExe -u -m hildanext.cli dolma-prep --config $cfgResolved
  if($LASTEXITCODE -ne 0){ throw "dolma_prep_failed exit=$LASTEXITCODE" }
  $t0.Stop()
  Log ("STEP=1 done elapsed_sec="+[int]$t0.Elapsed.TotalSeconds)
 }else{
  Log "STEP=1 SKIP tokenized and shards already exist"
 }

 # ── STEP 2: dolma-verify ────────────────────────────────────────────────────
 Log "STEP=2 action=dolma-verify"
 $t1=[System.Diagnostics.Stopwatch]::StartNew()
 & $pyExe -u -m hildanext.cli dolma-verify --config $cfgResolved
 if($LASTEXITCODE -ne 0){ throw "dolma_verify_failed exit=$LASTEXITCODE" }
 $t1.Stop()
 Log ("STEP=2 done elapsed_sec="+[int]$t1.Elapsed.TotalSeconds)

 # ── STEP 3: write done stamp ─────────────────────────────────────────────────
 New-Item -ItemType Directory -Force -Path (Split-Path -Parent $doneStamp) | Out-Null
 $stampContent="prep_done=$(Get-Date -Format s) config=$cfgResolved"
 Set-Content -Path $doneStamp -Value $stampContent -Encoding UTF8
 Log ("STEP=3 STAMP_WRITTEN path="+$doneStamp)

 # ── Summary ─────────────────────────────────────────────────────────────────
 $trainSize=[math]::Round((Get-Item $trainTok).Length/1MB,1)
 $evalSize=[math]::Round((Get-Item $evalTok).Length/1MB,1)
 $shardCount=(Get-ChildItem -Path $numpyTokensDir -Filter "tokens_*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count
 Log ("SUMMARY train_jsonl_mb="+$trainSize+" eval_jsonl_mb="+$evalSize+" npy_shards="+$shardCount)
 Log "DONE - you can now launch run_wsd_overnight.ps1"
 Log ("TRANSCRIPT="+$transcript)

}finally{
 Stop-Transcript | Out-Null
}

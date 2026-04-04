# Stage0 WSD launcher -- corrected schedule W=1000, S=3000, D=1000, total=5000.
# Implements LLaDA 2.0 WSD recipe with 3-phase LR (warmup-constant-cosine).
#
# SCHEDULE (block_size / attention):
#   Warmup  steps 0-999:    block 1->1024 causal  composite_llada20  LR 0->5e-5
#   Stable  steps 1000-3999: block 1024   bidir   simple_blockdiag   LR 5e-5 constant
#   Decay   steps 4000-4999: block 1024->32 causal composite_llada20 LR 5e-5->5e-6 cosine
#
# DEFAULT: from-scratch (clean + regenerate everything).
# RESUME:  -NoFromScratch -SkipDataPrep -SkipPreflight
#
# Usage:
#   # Fresh run (default):
#   powershell -ExecutionPolicy Bypass -File .\start_wsd_full_logs.ps1
#
#   # Resume without reprocessing:
#   powershell -ExecutionPolicy Bypass -File .\start_wsd_full_logs.ps1 -NoFromScratch -SkipDataPrep -SkipPreflight
#
#   # Prep only (no GPU):
#   powershell -ExecutionPolicy Bypass -File .\start_wsd_full_logs.ps1 -NoRun
#
#   # Force AdamW instead of PagedAdamW8bit:
#   powershell -ExecutionPolicy Bypass -File .\start_wsd_full_logs.ps1 -Optimizer adamw
param(
 [string]$CondaEnv="mdm",
 [string]$DolmaPath="",
 [string]$DocIndexPath="E:/DIFFUSION/HildaNext/dolma_v1_6_sample_1767050862/doc_index",
 [string]$BaseConfig="runs/configs/default.json",
 [string]$OutConfig="runs/configs/llada21_dolma_wsd_only.json",
 [string]$Optimizer="",
 [switch]$NoFromScratch,
 [switch]$SkipInstall,
 [switch]$SkipDataPrep,
 [switch]$SkipVerify,
 [switch]$SkipPreflight,
 [switch]$NoRun,
 [switch]$NoArchive,
 [switch]$CleanTokenized,
 [switch]$AllowExternalDolmaPath
)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$stamp=Get-Date -Format "yyyyMMdd_HHmmss"
$consoleDir=Join-Path $root "runs/console"
New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
$transcript=Join-Path $consoleDir ("wsd_inline_"+$stamp+".log")
Start-Transcript -Path $transcript -Force | Out-Null
function Write-Line([string]$msg){
 $ts=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
 Write-Host ("["+$ts+"] "+$msg)
}
function Write-Recipe(){
 Write-Line "============ EFFECTIVE RECIPE ============"
 Write-Line "Schedule:  W=1000  S=3000  D=1000  Total=5000"
 Write-Line "Warmup:    steps 0-999    block 1->1024  causal   composite_llada20   LR ramp 0->5e-5"
 Write-Line "Stable:    steps 1000-3999 block 1024     bidir    simple_blockdiag    LR 5e-5 constant"
 Write-Line "Decay:     steps 4000-4999 block 1024->32  causal  composite_llada20   LR cosine 5e-5->5e-6"
 Write-Line "Ladder:    [1, 4, 32, 64, 128, 256, 512, 1024]"
 Write-Line "Decay:     [1024, 512, 256, 128, 64, 32]"
 Write-Line "seq_len:   1024"
 Write-Line "Optimizer: $(if($Optimizer){"$Optimizer (forced)"}else{"auto (PagedAdamW8bit or AdamW)"})"
 Write-Line "Loss:      M2T(1.0) + T2T(1.0), ELBO 1/t, continuous-time"
 Write-Line "MTF:       2 turns"
 Write-Line "Embed noise: 0.1 -> 0 over warmup (grad stability)"
 Write-Line "LR:        3-phase (warmup linear, stable constant, decay cosine)"
 Write-Line "FromScratch: $(if($NoFromScratch){'NO (resume)'}else{'YES (clean start)'})"
 Write-Line "============================================="
}
try{
 if(Get-Command conda -ErrorAction SilentlyContinue){
  $hook=conda shell.powershell hook | Out-String
  Invoke-Expression $hook
  conda activate $CondaEnv
 }else{
  throw "conda_not_found_in_path"
 }
 if($env:CONDA_DEFAULT_ENV -ne $CondaEnv){
  throw "conda_activate_failed expected=$CondaEnv got=$($env:CONDA_DEFAULT_ENV)"
 }
 $pyExe=Join-Path $env:CONDA_PREFIX "python.exe"
 if(-not (Test-Path $pyExe)){
  throw "conda_python_not_found path=$pyExe"
 }
 $env:HILDANEXT_DOLMA_DOC_INDEX_PATH=$DocIndexPath
 $env:PYTHONUNBUFFERED="1"
 # --- VRAM Lab optimal settings for GTX 1080 (8GB) ---
 $env:CUDA_MODULE_LOADING="LAZY"
 $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
 # Optimizer override via env var (picked up by _select_optimizer_name)
 if($Optimizer){
  $env:HILDANEXT_FORCE_OPTIMIZER=$Optimizer
  Write-Line "OPTIMIZER_OVERRIDE=$Optimizer"
 }
 $docIdxObj=Resolve-Path -Path $DocIndexPath -ErrorAction SilentlyContinue
 if(-not $docIdxObj){ throw "doc_index_path_missing path=$DocIndexPath" }
 $docIdxResolved=[string]$docIdxObj
 $dolmaEffective=$DolmaPath
 if([string]::IsNullOrWhiteSpace($dolmaEffective)){
  if((Split-Path -Leaf $docIdxResolved).ToLower() -eq "doc_index"){
   $dolmaEffective=Split-Path -Parent $docIdxResolved
  }else{
   $dolmaEffective=$docIdxResolved
  }
 }
 $dolmaObj=Resolve-Path -Path $dolmaEffective -ErrorAction SilentlyContinue
 if(-not $dolmaObj){ throw "dolma_path_missing path=$dolmaEffective" }
 $dolmaResolved=[string]$dolmaObj
 if((Test-Path $dolmaResolved) -and (Test-Path (Join-Path $dolmaResolved "raw"))){
  $rawPath=Join-Path $dolmaResolved "raw"
  if((Get-ChildItem -Path $rawPath -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0){
   $dolmaResolved=[string](Resolve-Path -Path $rawPath)
  }
 }
 $repoParent=(Split-Path -Parent $root)
 if((-not $AllowExternalDolmaPath) -and ($dolmaResolved.ToLower().StartsWith($repoParent.ToLower()) -eq $false)){
  throw "dolma_path_outside_repo_parent path=$dolmaResolved expected_prefix=$repoParent"
 }
 Write-Line ("ROOT="+$root)
 Write-Line ("ENV="+$env:CONDA_DEFAULT_ENV)
 Write-Line ("PYTHON="+$pyExe)
 Write-Line ("DOC_INDEX="+$env:HILDANEXT_DOLMA_DOC_INDEX_PATH)
 Write-Line ("DOLMA_PATH="+$dolmaResolved)
 Write-Recipe
 # --- Step 1: Install ---
 if(-not $SkipInstall){
  Write-Line "STEP 1/7: pip install -e backend"
  & $pyExe -m pip install -e backend
  if($LASTEXITCODE -ne 0){ throw "pip_install_failed exit=$LASTEXITCODE" }
 }else{
  Write-Line "STEP 1/7: SKIP (install)"
 }
 # --- Step 2: Generate config ---
 Write-Line "STEP 2/7: make-stage0-config"
 Write-Line ("  base="+$BaseConfig+" out="+$OutConfig+" dolma="+$dolmaResolved)
 & $pyExe -u -m hildanext.cli make-stage0-config --config $BaseConfig --out-config $OutConfig --dolma-path $dolmaResolved
 if($LASTEXITCODE -ne 0){ throw "make_stage0_config_failed exit=$LASTEXITCODE" }
 # --- Step 3: Clean (from scratch) ---
 if(-not $NoFromScratch){
  Write-Line "STEP 3/7: CLEAN from scratch"
  Remove-Item "runs/checkpoints/cpt" -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item "runs/logs/cpt*" -Force -ErrorAction SilentlyContinue
  Remove-Item "runs/reports/run-*_wsd_recipe.json" -Force -ErrorAction SilentlyContinue
  if($CleanTokenized){
   Write-Line "  CLEAN tokenized artifacts (re-prep forced)"
   Remove-Item "data/tokenized" -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item "data/processed" -Recurse -Force -ErrorAction SilentlyContinue
  }
 }else{
  Write-Line "STEP 3/7: SKIP (no cleanup, resume mode)"
 }
 # --- Step 4: Data prep + verify ---
 if(-not $SkipDataPrep){
  Write-Line "STEP 4/7: dolma-prep"
  & $pyExe -u -m hildanext.cli dolma-prep --config $OutConfig
  if($LASTEXITCODE -ne 0){ throw "dolma_prep_failed exit=$LASTEXITCODE" }
  if(-not $SkipVerify){
   Write-Line "  dolma-verify"
   & $pyExe -u -m hildanext.cli dolma-verify --config $OutConfig
   if($LASTEXITCODE -ne 0){ throw "dolma_verify_failed exit=$LASTEXITCODE" }
  }
 }else{
  Write-Line "STEP 4/7: SKIP (data prep)"
 }
 # --- Step 5: Preflight ---
 if($NoRun){
  Write-Line "STEP 5/7: NO_RUN -- stop after CPU preprocessing"
  Write-Line ("TRANSCRIPT="+$transcript)
  exit 0
 }
 if(-not $SkipPreflight){
  Write-Line "STEP 5/7: preflight-wsd"
  & $pyExe -u -m hildanext.cli preflight-wsd --config $OutConfig
  if($LASTEXITCODE -ne 0){ throw "preflight_wsd_failed exit=$LASTEXITCODE" }
 }else{
  Write-Line "STEP 5/7: SKIP (preflight)"
 }
 # --- Step 6: Training ---
 Write-Line "STEP 6/7: run-wsd (5000 steps)"
 $runArgs=@("-u","-m","hildanext.cli","run-wsd","--config",$OutConfig,"--skip-preflight")
 if($NoArchive -or $NoFromScratch){ $runArgs+="--no-archive" }
 Write-Line ("RUN python "+($runArgs -join " "))
 & $pyExe @runArgs
 if($LASTEXITCODE -ne 0){ throw "run_wsd_failed exit=$LASTEXITCODE" }
 # --- Step 7: Optional top-k merge ---
 Write-Line "STEP 7/7: merge-topk (averaging last stable checkpoints)"
 & $pyExe -u -m hildanext.cli merge-topk --config $OutConfig --top-k 3
 if($LASTEXITCODE -ne 0){
  Write-Line "WARN: merge-topk failed (non-fatal) exit=$LASTEXITCODE"
 }else{
  Write-Line "merge-topk complete"
 }
 Write-Line "DONE -- training complete"
 Write-Recipe
 Write-Line ("TRANSCRIPT="+$transcript)
}finally{
 if($Optimizer){ Remove-Item Env:HILDANEXT_FORCE_OPTIMIZER -ErrorAction SilentlyContinue }
 Stop-Transcript | Out-Null
}

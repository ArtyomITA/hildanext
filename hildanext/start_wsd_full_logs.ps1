# Stage0 inline launcher with mdm activation and live console logs.
# Single Python entrypoint: hildanext.cli run-stage0-inline.
# Usage: powershell -ExecutionPolicy Bypass -File .\start_wsd_full_logs.ps1
param(
 [string]$CondaEnv="mdm",
 [string]$DolmaPath="",
 [string]$DocIndexPath="E:/DIFFUSION/HildaNext/dolma_v1_6_sample_1767050862/doc_index",
 [string]$BaseConfig="runs/configs/default.json",
 [string]$OutConfig="runs/configs/llada21_dolma_wsd_only.json",
 [switch]$NoFromScratch,
 [switch]$SkipInstall,
 [switch]$SkipDataPrep,
 [switch]$SkipPreflight,
 [switch]$NoRun,
 [switch]$NoArchive,
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
 if(-not $SkipInstall){
  Write-Line "RUN python -m pip install -e backend"
  & $pyExe -m pip install -e backend
  if($LASTEXITCODE -ne 0){ throw "pip_install_failed exit=$LASTEXITCODE" }
 }
 Write-Line ("RUN python -u -m hildanext.cli make-stage0-config --config "+$BaseConfig+" --out-config "+$OutConfig+" --dolma-path "+$dolmaResolved)
 & $pyExe -u -m hildanext.cli make-stage0-config --config $BaseConfig --out-config $OutConfig --dolma-path $dolmaResolved
 if($LASTEXITCODE -ne 0){ throw "make_stage0_config_failed exit=$LASTEXITCODE" }
 if(-not $NoFromScratch){
  Write-Line "CLEAN from scratch"
  Remove-Item "runs/checkpoints/cpt" -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item "runs/logs/cpt*" -Force -ErrorAction SilentlyContinue
  Remove-Item "runs/reports/run-*_wsd_recipe.json" -Force -ErrorAction SilentlyContinue
 }
 if(-not $SkipDataPrep){
  Write-Line ("RUN python -u -m hildanext.cli dolma-prep --config "+$OutConfig)
  & $pyExe -u -m hildanext.cli dolma-prep --config $OutConfig
  if($LASTEXITCODE -ne 0){ throw "dolma_prep_failed exit=$LASTEXITCODE" }
  Write-Line ("RUN python -u -m hildanext.cli dolma-verify --config "+$OutConfig)
  & $pyExe -u -m hildanext.cli dolma-verify --config $OutConfig
  if($LASTEXITCODE -ne 0){ throw "dolma_verify_failed exit=$LASTEXITCODE" }
 }
 if($NoRun){
  Write-Line "NO_RUN set. Stop after CPU preprocessing."
  Write-Line ("TRANSCRIPT="+$transcript)
  exit 0
 }
 if(-not $SkipPreflight){
  Write-Line ("RUN python -u -m hildanext.cli preflight-wsd --config "+$OutConfig)
  & $pyExe -u -m hildanext.cli preflight-wsd --config $OutConfig
  if($LASTEXITCODE -ne 0){ throw "preflight_wsd_failed exit=$LASTEXITCODE" }
 }
 $runArgs=@("-u","-m","hildanext.cli","run-wsd","--config",$OutConfig,"--skip-preflight")
 if($NoArchive){ $runArgs+="--no-archive" }
 Write-Line ("RUN python "+($runArgs -join " "))
 & $pyExe @runArgs
 if($LASTEXITCODE -ne 0){ throw "run_wsd_failed exit=$LASTEXITCODE" }
 Write-Line "DONE"
 Write-Line ("TRANSCRIPT="+$transcript)
}finally{
 Stop-Transcript | Out-Null
}

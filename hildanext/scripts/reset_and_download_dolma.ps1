# Reset local datasets and download Dolma URLs in parallel.
# Uses huggingface/allenai/dolma urls list.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\reset_and_download_dolma.ps1 -Version v1_6-sample
param(
 [string]$DataRoot="E:\DIFFUSION\HildaNext\dolma_v1_6_sample_1767050862",
 [string]$Version="v1_6-sample",
 [int]$Parallel=16,
 [string]$RepoDir="",
 [switch]$NoReset,
 [switch]$UseAria2Only,
 [int]$MaxUrls=0
)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if([string]::IsNullOrWhiteSpace($RepoDir)){ $RepoDir=Join-Path $root "vendor/dolma_urls" }
$rawDir=Join-Path $DataRoot "raw"
$docIndexDir=Join-Path $DataRoot "doc_index"
function Log([string]$m){ $ts=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); Write-Host "[$ts] $m" }
if(-not $NoReset){
 $targets=@(
  (Join-Path $root "data/raw"),
  (Join-Path $root "data/processed"),
  (Join-Path $root "data/tokenized"),
  (Join-Path $root "data/dolma"),
  (Join-Path $root "runs/cache/dolma_fingerprint.json"),
  $DataRoot
 )
 foreach($t in $targets){
  if(Test-Path $t){
   Log "REMOVE $t"
   Remove-Item -Recurse -Force $t
  }
 }
}
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
New-Item -ItemType Directory -Force -Path $docIndexDir | Out-Null
if(Test-Path (Join-Path $RepoDir ".git")){
 Log "UPDATE URL REPO $RepoDir"
 git -C $RepoDir pull --ff-only
}else{
 if(Test-Path $RepoDir){ Remove-Item -Recurse -Force $RepoDir }
 New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RepoDir) | Out-Null
 Log "CLONE URL REPO $RepoDir"
 git clone https://huggingface.co/datasets/allenai/dolma $RepoDir
}
$urlFile=Join-Path $RepoDir ("urls/"+$Version+".txt")
if(-not (Test-Path $urlFile)){ throw "missing_url_list $urlFile" }
$urls=Get-Content $urlFile | Where-Object { $_ -and $_.Trim() -and (-not $_.Trim().StartsWith("#")) }
if($urls.Count -eq 0){ throw "empty_url_list $urlFile" }
if($MaxUrls -gt 0 -and $urls.Count -gt $MaxUrls){ $urls=$urls[0..($MaxUrls-1)] }
Log "VERSION=$Version URLS=$($urls.Count) PARALLEL=$Parallel"
if(Get-Command aria2c -ErrorAction SilentlyContinue){
 Log "DOWNLOAD with aria2c to $rawDir"
 aria2c --input-file="$urlFile" --dir="$rawDir" --continue=true --auto-file-renaming=false --max-concurrent-downloads=$Parallel --split=8 --min-split-size=16M --summary-interval=30
 if($LASTEXITCODE -ne 0){ throw "aria2c_failed exit=$LASTEXITCODE" }
}elseif($UseAria2Only){
 throw "aria2c_not_found"
}else{
 if(-not (Get-Command curl.exe -ErrorAction SilentlyContinue)){ throw "curl_not_found" }
 Log "DOWNLOAD with curl.exe thread-jobs to $rawDir"
 $jobs=@()
 $started=0
 $completed=0
 $fail=0
 foreach($u in $urls){
  $uri=[Uri]$u
  $name=[IO.Path]::GetFileName($uri.AbsolutePath)
  if([string]::IsNullOrWhiteSpace($name)){ $name=("dolma_"+[Guid]::NewGuid().ToString("N")+".bin") }
  $out=Join-Path $rawDir $name
  if(Test-Path $out){
   $completed++
   continue
  }
  while(($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $Parallel){
   $done=Wait-Job -Job $jobs -Any -Timeout 5
   if($done){
    try{
     $msg=Receive-Job -Job $done -ErrorAction Stop
     if($msg){ Write-Host $msg }
    }catch{
     $fail++
     Write-Host ("JOB_FAIL "+$done.Name+" "+$_.Exception.Message)
    }
    Remove-Job -Job $done -Force
    $jobs=$jobs | Where-Object { $_.Id -ne $done.Id }
    $completed++
    if(($completed % 25) -eq 0){ Log ("PROGRESS completed="+$completed+" started="+$started+" fail="+$fail) }
   }
  }
  $started++
  $jobs+=Start-ThreadJob -Name ("dl_"+$started) -ScriptBlock {
   param($url,$dst)
   & curl.exe -L --retry 8 --retry-delay 2 --fail --output $dst $url
   if($LASTEXITCODE -ne 0){ throw "curl_failed url=$url code=$LASTEXITCODE" }
   return ("DOWNLOADED "+$dst)
  } -ArgumentList $u,$out
 }
 while($jobs.Count -gt 0){
  $done=Wait-Job -Job $jobs -Any -Timeout 10
  if(-not $done){ continue }
  try{
   $msg=Receive-Job -Job $done -ErrorAction Stop
   if($msg){ Write-Host $msg }
  }catch{
   $fail++
   Write-Host ("JOB_FAIL "+$done.Name+" "+$_.Exception.Message)
  }
  Remove-Job -Job $done -Force
  $jobs=$jobs | Where-Object { $_.Id -ne $done.Id }
  $completed++
  if(($completed % 25) -eq 0){ Log ("PROGRESS completed="+$completed+" started="+$started+" fail="+$fail) }
 }
 if($fail -gt 0){ throw "curl_thread_jobs_failed count=$fail" }
}
$fileCount=(Get-ChildItem -Path $rawDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
Log "DONE RAW_FILES=$fileCount RAW_DIR=$rawDir DOC_INDEX_DIR=$docIndexDir"
Log "NEXT: powershell -ExecutionPolicy Bypass -File .\\start_wsd_full_logs.ps1 -DocIndexPath '$docIndexDir'"

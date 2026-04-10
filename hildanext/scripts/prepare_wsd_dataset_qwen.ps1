<#
.SYNOPSIS
A dedicated, minimal, robust pipeline for creating Qwen3-0.6B WSD-ready datasets.

.DESCRIPTION
This script prepares a dataset specifically tailored for Qwen3-0.6B continuous 
pre-training (WSD stages). It avoids legacy blind chunking of structured data,
properly uses the chat template (`<|im_start|>`, `<|im_end|>`), and selectively
maintains Qwen reasoning markers (`<think>`, `</think>`, `/no_think`).

It supports 4 modes:
1) raw_only (Dolma-only)
2) raw_no_think (Dolma + curated QA with empty thinking blocks)
3) raw_think (Dolma + curated reasoning data)
4) raw_both (Dolma + both curated flavors)

Outputs are packed seq_len=1024 arrays with `doc_ids` directly consumable by 
the main WSD trainer.

.PARAMETER Mode
Which dataset mix to target: raw_only|raw_no_think|raw_think|raw_both

.PARAMETER RawWeight
Percentage or weight of raw data (default: 0.90)

.PARAMETER NoThinkWeight
Percentage of no_think data (default: 0.08)

.PARAMETER ThinkWeight
Percentage of think data (default: 0.02)

.PARAMETER MaxRawDocs
Max number of raw documents to read (default: 100000)

.PARAMETER MaxNoThinkExamples
Cap on no_think structured examples (default: 4000)

.PARAMETER MaxThinkExamples
Cap on think structured examples (default: 1000)

.PARAMETER NoThinkStrategy
How to format negative reasoning: empty_think | slash_no_think | both (default: both)

.PARAMETER DryRun
If passed, only parses, renders samples, and produces an execution report. No big artifacts written.

.PARAMETER DownloadTinyOverlays
If passed, fetches a small subset of OpenThoughts trace (max 2000 examples).

.PARAMETER UseLocalCuratedOnly
If passed, skips extracting from the massive Dolma raw set and only uses local drop-in JSONL.
#>

[CmdletBinding()]
param (
    [ValidateSet('raw_only','raw_no_think','raw_think','raw_both')]
    [string]$Mode = 'raw_both',

    [double]$RawWeight = 0.90,
    [double]$NoThinkWeight = 0.08,
    [double]$ThinkWeight = 0.02,

    [int]$MaxRawDocs = 100000,
    [int]$MaxNoThinkExamples = 4000,
    [int]$MaxThinkExamples = 1000,

    [ValidateSet('empty_think','slash_no_think','both')]
    [string]$NoThinkStrategy = 'both',

    [double]$EvalRatio = 0.01,
    [int]$SeqLen = 1024,

    [switch]$DryRun,
    [switch]$ForceRebuild,
    [switch]$DownloadTinyOverlays,
    [switch]$UseLocalCuratedOnly,

    [string]$OutName = 'qwen_wsd_run',
    [string]$WorkspaceRoot = "E:\DIFFUSION\HildaNext",
    [string]$ModelDir = "E:\DIFFUSION\HildaNext\hildanext\models\qwen3-0.6b",
    [string]$DolmaDir = "E:\DIFFUSION\HildaNext\dolma_v1_6_sample_1767050862\raw"
)

$ErrorActionPreference = 'Stop'

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " QWEN WSD DATASET PREPARATION PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Mode              : $Mode"
Write-Host "OutName           : $OutName"
if ($DryRun) {
    Write-Host "Mode              : DRY RUN (NO ARTIFACTS WRITTEN)" -ForegroundColor Yellow
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
$DatasetPrepScript = Join-Path $BaseDir "backend\src\hildanext\dataset_prep_qwen.py"

if (-Not (Test-Path $DatasetPrepScript)) {
    Write-Error "Cannot find Python script: $DatasetPrepScript"
}

# Construct the Python arguments
$PyArgs = @(
    $DatasetPrepScript,
    "--mode", $Mode,
    "--raw_weight", $RawWeight.ToString([cultureinfo]::InvariantCulture),
    "--no_think_weight", $NoThinkWeight.ToString([cultureinfo]::InvariantCulture),
    "--think_weight", $ThinkWeight.ToString([cultureinfo]::InvariantCulture),
    "--max_raw_docs", $MaxRawDocs,
    "--max_nothink_examples", $MaxNoThinkExamples,
    "--max_think_examples", $MaxThinkExamples,
    "--nothink_strategy", $NoThinkStrategy,
    "--eval_ratio", $EvalRatio.ToString([cultureinfo]::InvariantCulture),
    "--seq_len", $SeqLen,
    "--out_name", $OutName,
    "--workspace_root", $WorkspaceRoot,
    "--model_dir", $ModelDir,
    "--dolma_dir", $DolmaDir
)

if ($DryRun) { $PyArgs += "--dry_run" }
if ($ForceRebuild) { $PyArgs += "--force_rebuild" }
if ($DownloadTinyOverlays) { $PyArgs += "--download_tiny_overlays" }
if ($UseLocalCuratedOnly) { $PyArgs += "--use_local_curated_only" }

# Execute
Write-Host "Invoking python pipeline..." -ForegroundColor DarkGray
python @PyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Dataset preparation failed with exit code $LASTEXITCODE."
} else {
    Write-Host "Pipeline completed successfully." -ForegroundColor Green
}

<#
.SYNOPSIS
Audits the prepared WSD dataset (either Legacy or the new Qwen-specific prep).

.DESCRIPTION
This script analyzes tokenization, packing, segmentation boundaries, and Qwen chat 
template rendering for WSD datasets. Evaluates whether the dataset introduces 
cross-doc leakage, broken `<think>` blocks, or missing ChatML `<|im_end|>` tags.

.PARAMETER DatasetRoot
Path to the root of the dataset to audit.

.PARAMETER ProcessedPath
Specific path to processed train.jsonl.

.PARAMETER TokenizedPath
Specific path to tokenized train.jsonl.

.PARAMETER ModelDir
Path to the tokenizer model directory.

.PARAMETER SeqLen
Packed sequence length.

.PARAMETER MaxSamples
Number of samples to visualize and render.

.PARAMETER Strict
Enforces a NO_GO verdict if any warnings are detected.

.PARAMETER FailOnWarning
Stops the pipeline if any warning is triggered.

.PARAMETER DryRun
Just lists files without full parsing.

.PARAMETER AuditLegacy
Flags if auditing a legacy dataset (generic string serialization).

.PARAMETER AuditQwenPrep
Flags if auditing the new Qwen-aware prep.

.PARAMETER OutName
Sub-name for report generation.
#>

[CmdletBinding()]
param (
    [string]$DatasetRoot = "",
    [string]$ProcessedPath = "",
    [string]$TokenizedPath = "",
    [string]$ModelDir = "E:\DIFFUSION\HildaNext\hildanext\models\qwen3-0.6b",
    [int]$SeqLen = 1024,
    [int]$MaxSamples = 8,
    [string]$OutName = "audit_report",
    [switch]$Strict,
    [switch]$FailOnWarning,
    [switch]$DryRun,
    [switch]$AuditLegacy,
    [switch]$AuditQwenPrep
)

$ErrorActionPreference = 'Stop'

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " QWEN WSD DATASET AUDIT PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
$PyScript = Join-Path $BaseDir "backend\src\hildanext\audit_wsd_dataset_qwen.py"

$Args = @(
    $PyScript,
    "--model_dir", $ModelDir,
    "--seq_len", $SeqLen,
    "--max_samples", $MaxSamples,
    "--out_name", $OutName
)

if ($DatasetRoot) { $Args += "--dataset_root"; $Args += $DatasetRoot }
if ($ProcessedPath) { $Args += "--processed_path"; $Args += $ProcessedPath }
if ($TokenizedPath) { $Args += "--tokenized_path"; $Args += $TokenizedPath }
if ($Strict) { $Args += "--strict" }
if ($FailOnWarning) { $Args += "--fail_on_warning" }
if ($AuditLegacy) { $Args += "--audit_legacy" }
if ($AuditQwenPrep) { $Args += "--audit_qwen_prep" }

Write-Host "Invoking python audit..." -ForegroundColor DarkGray
python @Args

if ($LASTEXITCODE -ne 0) {
    Write-Error "Audit failed with exit code $LASTEXITCODE."
} else {
    Write-Host "Audit completed successfully." -ForegroundColor Green
}

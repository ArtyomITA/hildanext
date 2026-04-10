# Qwen3-0.6B WSD Dataset Preparation

This specialized pipeline builds continuous pre-training (WSD) datasets targeting the Qwen architecture behaviors natively. Instead of flattening instruction-following or reasoning data into plain generic blocks, it strictly adheres to Qwen's `apply_chat_template` formatting while mixing raw Dolma pre-training data.

## Features

- **No Generic Serialization**: Drops old `"User: ... Assistant: ..."` text blobs in favor of true `<|im_start|>` / `<|im_end|>` chat template structures.
- **Mix Options**: Supports fine mixing ratios between plain vanilla raw data, direct chat data (`no_think`), and reasoning traces (`think`).
- **Doc-boundary Preserving**: Preserves `doc_ids` directly out of the box using robust continuous streaming, ready for the main WSD launcher block diagonal attention mask.
- **Qwen Thinking**: Can download a subset of `OpenThoughts-114k` natively to inject true `<think>` blocks, and conditionally synthesizes `empty_think` (`<think>\n\n</think>`) to teach the model to skip thinking when unprompted.

## Usage Guide (PowerShell)

You can trigger the dataset prep stage independently using `scripts\prepare_wsd_dataset_qwen.ps1`.

### Example 1: Dry run to preview outputs
Validates input parsing and dumps a dry run report with formatting traces, preventing large chunking artifacts.
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_wsd_dataset_qwen.ps1 `
  -Mode raw_no_think `
  -UseLocalCuratedOnly `
  -NoThinkStrategy both `
  -OutName qwen_wsd_raw_nothink `
  -DryRun
```

### Example 2: The full WSD short-phase Mix
Downloads up to 2000 reasoning overlays natively from HuggingFace to compose the dataset.
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_wsd_dataset_qwen.ps1 `
  -Mode raw_both `
  -DownloadTinyOverlays `
  -OutName qwen_wsd_raw_both
```

## Supported Local Curated Schemas

Drop structural files directly in `data\curated_sources\chat`, `data\curated_sources\no_think`, or `data\curated_sources\think`.

**Schema A (Standard)**
```json
{"messages":[{"role":"user","content":"How are you?"},{"role":"assistant","content":"I am a model."}]}
```

**Schema B (Prompt/Response)**
```json
{"prompt":"What is 2+2?", "response":"4"}
```

**Schema C (Raw)**
```json
{"text":"Once upon a time in a plain text dataset..."}
```

## Artifacts Produced
- `data\processed_qwen_wsd\<OutName>\*.jsonl` (Plain JSON representation for inspection)
- `data\tokenized_qwen_wsd\<OutName>\*.jsonl` (Pre-tokenized arrays, seq_len 1024, doc_ids)
- `runs\reports\<timestamp>_<OutName>_prep_report.json` (Diagnostic manifest summary)

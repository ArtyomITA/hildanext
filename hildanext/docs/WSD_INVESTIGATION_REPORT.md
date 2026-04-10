# WSD Pipeline Investigation Report
## HildaNext – Stage0 WSD Training Block Analysis
**Date**: 2026-03-02  
**Environment**: Windows 10, GTX 1080 8GB (Pascal sm_61), conda env "mdm", Python 3.11+

---

## 1. Executive Summary

The WSD (Warmup-Stable-Decay) training pipeline has been consistently **hanging during the `preflight_wsd` step** across 5+ consecutive runs (March 1-2, 2026). The hang occurs specifically inside the 1-step training probe that validates model training capability before launching the full WSD run.

**No training step has ever completed.** Zero checkpoints exist. The pipeline halts indefinitely after printing `RUN_START` inside the `_run()` function of `training.py`.

---

## 2. Pipeline Architecture

### 2.1 Execution Flow
```
start_wsd_full_logs.ps1
  ├── make-stage0-config     → Creates llada21_dolma_wsd_only.json
  ├── dolma-prep             → Tokenizes Dolma corpus (✅ PASSES)
  ├── dolma-verify           → Validates tokenized data (✅ PASSES)
  ├── preflight-wsd          → ❌ HANGS HERE
  │   ├── loads model for AR inference test (✅ Works)
  │   ├── del bundle + torch.cuda.empty_cache()
  │   └── run_wsd_conversion(probe_cfg, steps=1)  → ❌ HANGS
  │       └── _run() → DataLoader creation → DEADLOCKS
  └── run-wsd                → Never reached
```

### 2.2 Key Source Files
| File | Lines | Purpose |
|------|-------|---------|
| `backend/src/hildanext/wsd_stage0.py` | 855 | Stage0 pipeline orchestrator |
| `backend/src/hildanext/training.py` | ~675 | `_run()`, `run_wsd_conversion()`, `run_sft_training()` |
| `backend/src/hildanext/tokenization.py` | ~428 | Tokenizer loading, streaming tokenization, checkpoint/resume |
| `backend/src/hildanext/config.py` | 264 | `AppConfig` dataclass hierarchy |
| `backend/src/hildanext/diffusion.py` | 209 | M2T/T2T loss, WSD block computation |
| `backend/src/hildanext/inference.py` | 476 | `load_model_bundle()`, model loading |
| `backend/src/hildanext/formulas.py` | 120 | LLaDA2.1 set ops, WSD block scheduling |
| `backend/src/hildanext/utils.py` | 150 | `force_math_sdpa()`, `TinyCausalLM` |
| `backend/src/hildanext/trace.py` | 203 | Runtime trace for fallbacks/metrics |
| `backend/src/hildanext/masks.py` | 100 | Document attention masks |
| `backend/src/hildanext/io_utils.py` | 60 | JSON/JSONL I/O |
| `backend/src/hildanext/cli.py` | 407 | CLI dispatcher |
| `backend/src/hildanext/datasets.py` | 302 | Data ingestion (Dolma/TinyStories) |
| `backend/src/hildanext/recipe.py` | 284 | Preflight and overnight recipe runner |
| `start_wsd_full_logs.ps1` | 150 | Main launcher script |
| `run_wsd_overnight.ps1` | 120 | Resume-safe retry loop |

---

## 3. Configuration

### 3.1 Active WSD Config (`llada21_dolma_wsd_only.json`)
```json
{
  "model": {
    "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
    "mask_token": "[MASK]",
    "trust_remote_code": true,
    "torch_dtype": "float16"
  },
  "train": {
    "lr": 3e-5,
    "batch_size": 1,
    "accum_steps": 16,
    "max_steps": 8000,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "data_num_workers": 4,
    "data_prefetch_factor": 2,
    "data_persistent_workers": true,
    "data_pin_memory": true,
    "ckpt_every": 500,
    "eval_every": 250
  },
  "data": {
    "seq_len": 1024,
    "doc_mask_mode": "composite_llada20"
  },
  "wsd": {
    "warmup_steps": 200,
    "stable_steps": 6000,
    "decay_steps": 1800,
    "ladder_block_sizes": [1, 4, 32, 64, 128, 256, 512, 1024]
  },
  "stage0": {
    "steps_total_stage0": 8000,
    "lr_stage0": 3e-5,
    "seq_len": 1024,
    "grad_accum_steps": 16
  }
}
```

### 3.2 Probe Config (created by `preflight_wsd`)
The probe overrides:
- `train.max_steps = 1`
- `data.seq_len = 256`
- `stage0.seq_len = 256`
- `train.ckpt_every = 1`
- `train.eval_every = 2`
- Unique logs/checkpoints subdirectory

### 3.3 Model
- **Qwen3-0.6B-Base** (151,669 vocab, Apache-2.0)
- fp16 precision
- Math SDPA forced (no Flash Attention on Pascal/sm_61)
- Local weights at `Qwen3-0.6B/` with `model.safetensors`

---

## 4. Data Status

### 4.1 Raw Data
- **Dolma v1.6 sample**: 103 `.json.gz` files, ~16 GB raw
- Location: `dolma_v1_6_sample_1767050862/raw/`

### 4.2 Tokenized Data
| Split | Size | Rows | Seq Length |
|-------|------|------|------------|
| Train | 5.3 GB (`train.jsonl`) | 258,560 | 1024 |
| Eval | 48 MB (`eval.jsonl`) | ~5,000 | 1024 |

### 4.3 Processed Data
| Split | Size |
|-------|------|
| Train | 5.2 GB |
| Eval | 52 MB |

### 4.4 Checkpoints
**Zero checkpoints exist.** No training step has ever completed.

---

## 5. Root Cause Analysis

### 5.1 PRIMARY: RAM Exhaustion from Full Dataset Load
**Severity: CRITICAL**

`_run()` in `training.py` calls:
```python
ds = TokenizedDataset(tok_path)
```
which internally calls `read_jsonl(path)` with **NO `max_rows` limit**. This loads the entire 5.3 GB `train.jsonl` into a Python list:
- 258,560 rows
- Each row: `{"input_ids": [1024 ints], "attention_mask": [1024 ints], "doc_ids": [1024 ints], "response_mask": [1024 ints]}`
- Python list overhead: ~8 bytes per int × 4 fields × 1024 × 258,560 ≈ **8.5 GB RAM**
- Plus JSON parsing overhead, string temporaries → likely **10-15+ GB total RAM**

This happens even for the 1-step probe, wasting massive RAM for a single batch.

### 5.2 PRIMARY: Windows DataLoader Worker Deadlock
**Severity: CRITICAL**

Config sets `data_num_workers=4` with `persistent_workers=True`. On Windows:
- Python multiprocessing uses `spawn` (not `fork`)
- Each worker process must re-import all modules and re-initialize
- The broken NumPy DLL (see 5.3) causes worker initialization to fail silently
- `persistent_workers=True` means workers stay alive → if they deadlock during init, the main thread blocks forever on `for batch in loader`
- The symptom: `RUN_START` prints, then nothing → classic DataLoader worker deadlock

### 5.3 CONTRIBUTING: NumPy DLL Load Failure
**Severity: HIGH**

Every run logs (via `fallbacks.jsonl`):
```
Failed to initialize NumPy: DLL load failed while importing _multiarray_umath
```
This appears as a `numpy_dll_unavailable` fallback. The NumPy installation in the `mdm` conda environment has a broken DLL dependency, possibly due to:
- Mismatched NumPy version vs. compiled extensions
- Missing Visual C++ runtime
- Conda environment corruption

Impact: NumPy is used transitively by PyTorch DataLoader workers. Failed NumPy init in worker subprocesses can cause silent crashes or hangs.

### 5.4 CONTRIBUTING: Double Model Load in Preflight
**Severity: MEDIUM**

`preflight_wsd()` loads the model **twice**:
1. First for AR inference test via `load_model_bundle(for_training=False)`
2. Then deletes it + `torch.cuda.empty_cache()`
3. Then loads again for training probe via `run_wsd_conversion()` → `_run()` → `load_model_bundle(for_training=True)`

On GTX 1080 (8 GB VRAM):
- Qwen3-0.6B in fp16 ≈ 1.2 GB VRAM
- With optimizer states (AdamW): +2.4 GB
- Gradients: +1.2 GB
- Activations at seq_len=256: +0.5 GB
- Total: ~5.3 GB minimum for training

While the `del bundle + empty_cache()` between loads should free memory, any fragmentation or Python reference leak could keep the first model's VRAM partially allocated.

### 5.5 CONTRIBUTING: No Step-Level Timeout/Watchdog
**Severity: MEDIUM**

The training loop has **no timeout mechanism**:
```python
while opt_steps < max_steps:
    for step, batch in enumerate(loader, ...):  # ← blocks forever if workers deadlock
```

If the DataLoader workers fail to produce a batch, the main thread blocks indefinitely on `for batch in loader`. There is no heartbeat, no watchdog, no timeout.

---

## 6. Log Evidence

### 6.1 Console Logs (5 runs, all identical hang pattern)
```
wsd_inline_20260301_033848.log → CLEAN → dolma-prep → PIPELINE ARRESTED
wsd_inline_20260301_035134.log → Same, killed early
wsd_inline_20260301_035251.log → CLEAN → dolma-prep → dolma-verify → preflight-wsd → HUNG
wsd_inline_20260301_043053.log → Same, preflight-wsd → HUNG
wsd_inline_20260302_012544.log → Latest, same pattern
```

### 6.2 Training Probe Log (cpt_run.log)
```
[2026-03-01 02:04:29] RUN_START kind=cpt max_steps=1 grad_acc=16 seq_len=256 batch_size=1
```
**No further output.** No step 1/1 log, no loss, no RUN_END.

### 6.3 Fallbacks (fallbacks.jsonl)
37+ entries, recurring patterns:
- `flash_attention_unavailable` (every run, expected on Pascal)
- `numpy_dll_unavailable` (every run, problematic)
- No `oom` or `cuda_error` entries → the hang is silent

---

## 7. Formulas & Training Theory

### 7.1 WSD Schedule
- **Warmup** (steps 0–200): Linear LR ramp from 0 → `lr_base` (3e-5)
- **Stable** (steps 200–6200): Constant `lr_base`, bidirectional attention
- **Decay** (steps 6200–8000): Cosine decay to `lr_base × lr_min_ratio`
- **Ladder block sizes**: [1, 4, 32, 64, 128, 256, 512, 1024]

### 7.2 Loss Function: Continuous-Time ELBO
From `diffusion.py` `compute_m2t_t2t_losses()`:
```
t ~ Uniform(t_min, t_max)    # t_min=0.001, t_max=1.0
mask_ratio = t
L_m2t = (1/t) × CE(model(x_masked), x_original)  # inv_t weighting
```

### 7.3 LR Schedule (`_compute_lr`)
```
if step < warmup: lr = lr_base × (step / warmup)
else: progress = (step - warmup) / (max_steps - warmup)
      lr = lr_min + 0.5 × (lr_base - lr_min) × (1 + cos(π × progress))
```

---

## 8. What the Coding Agent Needs to Know

### 8.1 File Inventory for WSD
```
hildanext/
├── backend/src/hildanext/
│   ├── training.py        ← Main training loops (_run, run_wsd_conversion)
│   ├── wsd_stage0.py      ← Stage0 pipeline (preflight_wsd, run_wsd)
│   ├── config.py          ← All config dataclasses
│   ├── diffusion.py       ← M2T/T2T losses, WSD block computation
│   ├── inference.py       ← Model loading (load_model_bundle)
│   ├── tokenization.py    ← Tokenization pipeline
│   ├── formulas.py        ← LLaDA2.1 formulas, WSD scheduling
│   ├── utils.py           ← force_math_sdpa, TinyCausalLM
│   ├── trace.py           ← Runtime trace system
│   ├── masks.py           ← Document attention masks
│   ├── io_utils.py        ← JSON/JSONL I/O
│   ├── cli.py             ← CLI entry points
│   ├── datasets.py        ← Dolma/TinyStories data ingestion
│   └── recipe.py          ← Preflight/overnight recipe runner
├── data/
│   ├── configs/llada21_dolma_wsd_only.json  ← Active WSD config
│   ├── tokenized/         ← 5.3 GB train.jsonl + 48 MB eval.jsonl
│   └── processed/         ← 5.2 GB train + 52 MB eval
├── scripts/
│   ├── start_wsd_full_logs.ps1  ← Main launcher
│   └── run_wsd_overnight.ps1    ← Resume-safe retry
├── runs/
│   ├── checkpoints/       ← EMPTY (no steps completed)
│   ├── logs/              ← Console + run logs
│   └── reports/           ← Preflight JSON reports
└── docs/
    └── DESIGN.md          ← Architecture documentation
```

### 8.2 Critical Parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen3-0.6B-Base | config.json |
| Vocab size | 151,669 | model config |
| Precision | fp16 | train config |
| Attention | Math SDPA (forced) | utils.py |
| Seq len (full) | 1024 | data config |
| Seq len (probe) | 256 | preflight override |
| Micro batch | 1 | train config |
| Grad accumulation | 16 | train config |
| Total steps | 8,000 | stage0 config |
| LR | 3e-5 | train config |
| Workers | 4 | train config |
| VRAM | 8 GB (GTX 1080) | hardware |

### 8.3 Known Constraints
1. **No Flash Attention** — Pascal GPU (sm_61) doesn't support it
2. **NumPy DLL broken** — every run logs DLL load failure
3. **8 GB VRAM limit** — must fit model + optimizer + gradients + activations
4. **Windows multiprocessing** — uses `spawn`, not `fork`
5. **5.3 GB tokenized data** — loaded entirely into RAM

---

## 9. Logging Already Added

### 9.1 `training.py` `_run()` (previously added this session)
- Dataset load timing, file size, row count, RAM estimate
- DataLoader config (workers, prefetch, persistent, pin_memory)
- Model load VRAM before/after measurements
- Optimizer creation timing and VRAM
- Total init time
- Epoch start + first batch timing with `FIRST_BATCH_RECEIVED` marker
- Training loop enter marker `TRAINING_LOOP_ENTER`

### 9.2 `tokenization.py` `tokenize_split()` (previously added this session)
- Tokenizer load timing and vocab size stats

### 9.3 Still Needed
- Forward pass timing (before/after `compute_m2t_t2t_losses`)
- Backward pass timing (around `loss.backward()`)
- Per-batch VRAM snapshots
- `preflight_wsd()` step-by-step timing in `wsd_stage0.py`

---

## 10. Recommended Actions (for coding agent)

### DO NOT FIX (per user instruction — investigate only)
These are root causes identified but NOT to be fixed in this session:

1. **DataLoader `num_workers=4` on Windows with broken NumPy** → likely cause of deadlock
2. **Full dataset load for 1-step probe** → 5.3 GB loaded for one batch
3. **No timeout/watchdog** → infinite hang on deadlocked loader

### NEXT STEPS
1. Complete forward/backward pass logging in `training.py`
2. Add step-by-step timing to `preflight_wsd()` in `wsd_stage0.py`
3. Build VRAM stability benchmark harness (synthetic workload, real model)
4. Test configuration matrix (precision × optimizer × seq_len × grad_acc)
5. Identify safe operating envelope for GTX 1080 8 GB
6. Only then attempt to unblock the WSD pipeline

---

## 11. Resolution (April 2026)

### 11.1 Revised Root Cause

The initial hypotheses in sections 5.1–5.3 (RAM exhaustion, DataLoader deadlock, NumPy DLL) were **contributing factors but not the primary cause** of the system instability.

The **actual root cause** was identified through empirical VRAM probing:

> The `composite_llada20` mask mode doubles the effective sequence length from S=1024 to 2S=2048 by concatenating `[x_t | x_0]`. The `lm_head` linear layer (`[vocab=151936, hidden=1024]`) then computes logits on **all 2048 positions**, producing a tensor of shape `[1, 2048, 151936]` in fp16 = **594 MB**. At that point, ~5.4 GB is already allocated for backbone + gradients + activations. Attempting to allocate 594 MB more pushes past the 8 GB limit, but **does not produce a clean OOM**. Instead, the GPU enters **memory thrashing**: WDDM cannot schedule the Windows Desktop Window Manager (DWM), leading to progressive system degradation that requires a full reboot.

### 11.2 Key Discovery: SDPA Backend

Previous assumption: `force_math_sdpa()` selects the MATH backend (fp32 intermediates = ~268 MB/layer). Actual behavior: `torch._fused_sdp_choice` probe confirmed **EFFICIENT_ATTENTION** is selected for all mask shapes on Pascal (sm_61). Attention is NOT the VRAM bottleneck — the `lm_head` is.

### 11.3 Fix Applied: Option 3 + Option 2

Two complementary changes were implemented:

**Option 3 — Slim lm_head** (`diffusion.py::_forward()` L271–293):
- For the composite path, call `model.model(input_ids=ids2)` (backbone only) on all 2S tokens
- Then apply `model.lm_head()` only on the first S positions (the `x_t` tokens that need denoising)
- Saves ~296 MB (half the lm_head output)

**Option 2 — Halved seq_len for composite phases** (`training.py` L822–840):
- When `mask_mode == "composite_llada20"` (warmup + decay phases), truncate each batch from S=1024 to S≈512
- Truncation is **doc-boundary-aware**: snaps to nearest document boundary in range `[3/4 * half, half]`
- During the stable phase (bidirectional attention, `simple_blockdiag`), full S=1024 is used

### 11.4 Test Results (at `memory_fraction=0.85`)

| Phase | Mask Mode | Effective S | Peak VRAM | Headroom |
|-------|-----------|-------------|-----------|----------|
| Warmup/Decay | composite_llada20 | 512 → composite 1024 | 4972 MB | 1287 MB |
| Stable | simple_blockdiag | 1024 (no doubling) | 5676 MB | 583 MB |

Both phases pass with margin. Multi-turn-t2t (2 turns) also passes.

### 11.5 Updated Config

The WSD schedule was revised to W=1000 / S=3000 / D=1000 (5000 total steps), with:
- `batch_size=1`, `accum_steps=8`
- `seq_len=1024`
- `doc_attention_mask_mode="composite_llada20"` 
- `set_per_process_memory_fraction(0.85)` in `training.py`
- `multi_turn_t2t=2`
- Gradient checkpointing enabled

---

*Report generated for coding agent handoff. Resolution appended April 2026.*

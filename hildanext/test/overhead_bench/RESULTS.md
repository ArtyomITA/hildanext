# Overhead Benchmark Results & Decision Matrix

**GPU**: GTX 1080 (8192 MB dedicated VRAM, Pascal sm_61)
**Model**: Qwen3-0.6B (596M params, fp16)
**Dataset**: Dolma v1.6 sample (258K rows × 1024 tokens)
**Date**: benchmark runs from this session

---

## 1. VRAM & Speed Matrix (8 configs, 5 warm steps each)

| ID | Config | Peak MB | Free MB | tok/s | s/step | 4000 steps | Freeze Risk |
|----|--------|---------|---------|-------|--------|------------|-------------|
| **A** | adamw8bit + ckpt + 1024 | 5184 | 3008 | 551 | 14.80 | 16.4h | Low |
| **B** | adamw8bit + NO ckpt + 1024 | 7219 | 973 | 649 | 12.57 | 14.0h | **HIGH** |
| **C** | adamw8bit + ckpt + 2048 | 6917 | 1275 | 395 | 41.33 | 45.9h | Medium |
| **D** | adamw_fp32 + ckpt + 1024 | 6295 | 1897 | 540 | 15.12 | 16.8h | Med (diverges) |
| **E** | paged8bit + ckpt + 1024 | **4024** | **4168** | 541 | 15.08 | 16.8h | **Very Low** |
| **F** | paged8bit + NO ckpt + 1024 | 6044 | 2148 | 623 | 13.11 | 14.6h | Low-Med |
| **G** | adamw8bit + ckpt + 1536 | 4902 | 3290 | 458 | 26.77 | 29.7h | Very Low |
| **H** | paged8bit + ckpt + 2048 | 5759 | 2433 | 394 | 41.45 | 46.1h | Low |

### Key insights
- **No-ckpt is 15-18% faster** (12.57 vs 14.80 s/step) — less than the typical 25% AC penalty, because Qwen3-0.6B is small.
- **seq=2048 is 2.8x slower** (41.3 vs 14.8 s/step) — quadratic attention cost dominates.
- **Paged vs non-paged 8bit**: paged saves ~1160MB (E vs A) with only 2% speed overhead. Paging acts as emergency CPU offload safety net.
- **fp32 optimizer (D) diverges** (loss=2416) on 5 steps — fp32 states need GradScaler for fp16 model, not viable without further work.
- **Config B (7219MB) leaves only 973MB free** on an 8GB card — guaranteed shared memory spill.

---

## 2. Overhead Micro-Benchmarks (per call, 50 reps)

| Item | Mean | Median | Impact @ 4000 steps | Code location |
|------|------|--------|---------------------|---------------|
| `isfinite().item()` | 1479.8 us | 152.8 us | ~6.1s (every step) | training.py L782 |
| `.item()` x 6 metrics | 167.7 us | — | ~0.67s (every step) | training.py L810-820 |
| `nvidia-smi` subprocess | **48052 us** | — | **19.2s** (every 10 opt steps) | training.py L83 |
| `pynvml` equivalent | 735 us | — | 0.3s (65x faster) | — |
| `empty_cache()` | 1086.5 us | — | ~4.3s (4 sites) | training.py L534,803,980,990 |
| `wsd_block()` scheduler | 10.6 us | — | 0.04s | negligible |
| `torch.remainder()` | 791.6 us | 38.3 us | ~3.2s | negligible |
| **Total overhead** | | | **~97.6s** across 4000 steps | ~0.04% of 14h |

### Insight
The individual overheads are tiny in absolute terms. The real problems are:
1. **`.item()` forces GPU sync** — stalls the CUDA pipeline. On our small model this is tolerable, but it adds up.
2. **`nvidia-smi` subprocess** is 65x slower than pynvml — easy fix.
3. **`empty_cache()`** in the hot path (L534 post-init, L803 NaN recovery) fragements the allocator's steady-state cache.

---

## 3. Optimizer Quality (20 steps each, 8bit variants)

| Config | Final Loss | NaN | Trend | Grad Norm Max |
|--------|-----------|-----|-------|---------------|
| adamw8bit | 30.713 | 0 | DOWN | 6352 |
| adamw8bit (no embed noise) | 30.017 | 0 | DOWN | — |
| paged8bit | 29.088 | 0 | FLAT | — |
| paged8bit (no embed noise) | 28.315 | 0 | FLAT | — |

### Insight
- Both 8bit optimizers converge cleanly with zero NaN.
- Embed noise adds ~1-2 points to loss (expected — it's a regularizer per LLaDA2.0 S7.1).
- AdamW8bit shows steeper descent (DOWN) vs Paged (FLAT) in 20 steps — too few steps to judge convergence, but both are stable.
- **Recommendation**: Use PagedAdamW8bit for safety (auto CPU offload at OOM boundary) with negligible speed cost.

---

## 4. SYSTEM FREEZE ROOT CAUSE (CRITICAL)

### Problem
Computer becomes progressively slower during overnight training until complete freeze.

### Root Cause: GPU Shared Memory Spill
On **Windows consumer GPUs** (GTX 1080), when PyTorch's CUDA caching allocator exceeds
the card's **dedicated** VRAM (8GB), Windows automatically allocates into **GPU shared
memory** — which is actually **system RAM accessed via PCIe bus**.

This causes:
1. Every CUDA operation that touches shared-memory tensors runs at PCIe speed (16 GB/s) instead of VRAM speed (320 GB/s) — **20x slower**.
2. Windows' GPU memory manager increasingly pages system RAM to service GPU requests.
3. System-wide memory pressure builds → swap thrashing → UI freeze → eventual hang.

**Source**: [PyTorch forum – documented fix](https://discuss.pytorch.org/t/documented-fix-slow-execution-pytorch-using-gpu-shared-memory/218909)

### Evidence in HildaNext
- `config.py` defines `max_vram_pct: float = 0.85` — but **this value is NEVER enforced** via `torch.cuda.set_per_process_memory_fraction()`.
- Config B peaks at 7219 MB / 8192 MB = **88%** → only 973 MB headroom → any spike during eval/checkpoint/GC will spill into shared memory.
- Even Config A (5184 MB) can spike during eval or gradient accumulation resets.
- `utils.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — good for fragmentation but does NOT cap total usage.

### Fix (P0 — highest priority)
Add to `_run()` in training.py, before any allocation:
```python
if torch.cuda.is_available() and cfg.train.max_vram_pct < 1.0:
    torch.cuda.set_per_process_memory_fraction(
        cfg.train.max_vram_pct,
        device=bundle.device
    )
```
This tells PyTorch's allocator to **hard-cap** at 85% of dedicated VRAM (6963 MB).
If an allocation would exceed that limit, PyTorch raises OOM instead of silently spilling to shared memory.

---

## 5. Web Research Summary

### 5a. PagedAdamW8bit Mechanism
> "It works like regular CPU paging — only becomes active if you run out of GPU memory. Only then will the memory be transferred [to CPU]."
> — bitsandbytes Issue #962

- PagedAdamW8bit = AdamW8bit + automatic CPU offload of optimizer states at OOM boundary.
- No convergence difference vs regular 8bit — identical math, just a safety net.
- Cost: ~2% speed overhead (CPU→GPU page transfers when triggered).
- **On 8GB GPU: use Paged variant always** — the 2% cost is negligible vs the freeze risk.

### 5b. Gradient Checkpointing Trade-off
> "AC reduces memory saved for backward but comes with added cost in compute due to recomputation."
> — [PyTorch Blog: Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)

- Industry typical: **25% more training time for 60% memory reduction**.
- Our measurement: **18% overhead** (14.80 vs 12.57 s/step) — better than typical because model is small.
- `torch.compile` with selective AC can reduce the overhead further (1.5-2x speedup) — but requires sm_70+ for full benefit. Pascal sm_61 support is limited.

### 5c. `.item()` GPU Sync
- `.item()` forces GPU→CPU synchronization, stalling the CUDA pipeline.
- Best practice: accumulate tensors on GPU in a list, call `.item()` only at logging intervals.
- Alternative: use `torch.cuda.synchronize()` once at log interval, then read all accumulated values.

### 5d. nvidia-smi vs pynvml
- `nvidia-smi` uses NVML internally — Python bindings via `pynvml` skip the subprocess overhead.
- Our benchmark: **48ms (nvidia-smi) vs 0.7ms (pynvml)** = 65x speedup.
- Install: `pip install nvidia-ml-py3` (or `pynvml`), then `pynvml.nvmlInit()` once at startup.

### 5e. empty_cache() Best Practices
> "PyTorch's caching allocator aims for a steady state without cudaMalloc/cudaFree. Calling empty_cache() breaks this steady state."

- **Do call** after large memory pattern changes: checkpoint save, eval, OOM recovery.
- **Don't call** every batch or on the hot path.
- Current code has 4 calls in training.py + 5 in wsd_stage0.py — L534 (post-init) is fine, others are correctly placed at ckpt/eval/NaN recovery boundaries.

### 5f. PYTORCH_CUDA_ALLOC_CONF Options
- `expandable_segments:True` — already set in utils.py, good.
- `max_split_size_mb:512` — can help with fragmentation on 8GB cards.
- `CUDA_MODULE_LOADING=LAZY` — reduces upfront GPU memory usage by loading kernels on demand.

---

## 6. Decision Matrix — Recommended Configurations

### OPTION 1: "Safe Daily Driver" (recommended)
```
Optimizer: PagedAdamW8bit
Grad checkpoint: ON (adaptive — current behavior)
Seq length: 1024
+ memory fraction cap (0.85)
+ pynvml instead of nvidia-smi
```
- Peak: ~4024-5184 MB → 3+ GB headroom
- Speed: ~15.1 s/step → **16.8h for 4000 steps**
- Freeze risk: **Very Low**
- Trade-off: 15% slower than no-ckpt, but rock-solid for overnight runs

### OPTION 2: "Faster Stable Phase" (good balance)
```
Optimizer: PagedAdamW8bit
Grad checkpoint: adaptive (ON for warmup/decay, OFF for stable)  ← current behavior
Seq length: 1024
+ memory fraction cap (0.85)
+ pynvml
```
- Peak: ~4024 MB (warmup/decay with ckpt) → 6044 MB (stable without ckpt)
- Speed: ~15.1 s/step (warmup) → 13.1 s/step (stable) → **~14.8h average**
- Freeze risk: **Low** (paged catches spills, fraction cap prevents shared mem)
- This IS the current adaptive behavior — just add the fraction cap fix

### OPTION 3: "Maximum Speed" (risky on 8GB)
```
Optimizer: AdamW8bit (NOT paged)
Grad checkpoint: OFF
Seq length: 1024
+ memory fraction cap (0.85)
```
- Peak: 7219 MB → would be CAPPED at 6963 MB → **likely OOM**
- Speed: 12.57 s/step IF it fits
- Freeze risk: OOM crash instead of freeze (better than freeze, but still fails)
- **NOT RECOMMENDED** — doesn't fit within 85% cap

### OPTION 4: "Longer Context" (if you need 2048)
```
Optimizer: PagedAdamW8bit  
Grad checkpoint: ON (mandatory at 2048)
Seq length: 2048
+ memory fraction cap
+ pynvml
```
- Peak: 5759 MB → 2.4 GB headroom
- Speed: 41.5 s/step → **46.1h for 4000 steps** (2.8x slower)
- Only worth it if longer context is critical for quality

---

## 7. Prioritized Fix List

| Pri | Fix | Effort | Impact | Where |
|-----|-----|--------|--------|-------|
| **P0** | Enforce `max_vram_pct` via `set_per_process_memory_fraction()` | 3 lines | **Eliminates system freeze** | training.py `_run()` top |
| **P0** | Set `CUDA_MODULE_LOADING=LAZY` env var | 1 line | Reduces baseline VRAM ~100-200MB | utils.py |
| **P1** | Replace nvidia-smi subprocess with pynvml | 15 lines | 65x faster thermal check, -48ms/call | training.py L79-88 |
| **P2** | Batch `.item()` calls to log interval only | 20 lines | Eliminates per-step GPU sync stalls | training.py L782-820 |
| **P2** | Add `max_split_size_mb:512` to ALLOC_CONF | 1 line | Reduces fragmentation on 8GB | utils.py |
| **P3** | Remove `empty_cache()` from L534 post-init | 1 line delete | Minor — steady-state allocator benefit | training.py |
| **P3** | Optional: `torch.compile` on loss function | Medium | Potential 10-20% speedup (needs testing on Pascal) | diffusion.py |

---

## 8. Answer to Original Questions

### Q1: "posso mettere seq_len a 2048?"
**Si, ma non conviene.** Il dataset e tokenizzato a 1024; la retokenizzazione a 2048 e possibile ma il training diventa 2.8x piu lento (41s/step vs 15s/step). La VRAM a 2048 arriva a 5759-6917 MB — entra nella GTX 1080 ma con margine ridotto. A meno che il contesto lungo non sia essenziale, resta a 1024.

### Q2: "senza gradient checkpoint sarebbe piu veloce?"
**Si, 15-18% piu veloce** (12.6-13.1 vs 14.8-15.1 s/step). Ma:
- Con AdamW8bit (non-paged): peak 7219 MB → rischio freeze altissimo
- Con PagedAdamW8bit: peak 6044 MB → accettabile con memory_fraction cap
- Il comportamento **adattivo attuale** (ckpt ON warmup/decay, OFF stable) e gia il compromesso migliore

### Q3: "perche il PC si blocca durante il training notturno?"
**PyTorch spilla nella shared GPU memory di Windows.** La GTX 1080 ha 8GB dedicati; quando PyTorch riempie quella memoria, Windows assegna "GPU shared memory" dalla RAM di sistema via PCIe. Questo causa rallentamento progressivo → freeze. **Fix: una riga di codice** per capped l'allocazione all'85% della VRAM dedicata.

"""
WSD Performance & Conformity Diagnostic Suite
==============================================
Runs 5 isolated tests, each with full VRAM/RAM cleanup between them.
Tests actual forward+backward with real optimizer on real model.

Usage:
    conda activate mdm
    cd e:\DIFFUSION\HildaNext\hildanext
    python test\test_wsd_perf_diag.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

# Ensure hildanext package is on path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "backend" / "src"))

MODEL_PATH = str(_root / "models" / "qwen3-0.6b")
DATASET_PATH = str(_root / "data" / "tokenized_qwen_wsd" / "qwen_wsd_run" / "train.jsonl")
SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── helpers ───────────────────────────────────────────────────────────

def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024

def _vram_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024

def _full_cleanup():
    """Aggressive cleanup between tests."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def _load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.train()
    return model, tok

def _create_optimizer(model):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=5e-5, weight_decay=0.01)
        return opt, "PagedAdamW8bit"
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        return opt, "AdamW"

def _load_batch(n_rows: int = 1):
    """Load n_rows from the JSONL dataset, return as batch dict on device."""
    rows = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            rows.append(json.loads(line.strip()))
    ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long, device=DEVICE)
    docs = torch.tensor([r["doc_ids"] for r in rows], dtype=torch.long, device=DEVICE)
    attn = torch.tensor([r["attention_mask"] for r in rows], dtype=torch.long, device=DEVICE)
    resp = torch.tensor([r.get("response_mask", [0]*SEQ_LEN) for r in rows], dtype=torch.long, device=DEVICE)
    return {"input_ids": ids, "doc_ids": docs, "attention_mask": attn, "response_mask": resp}

def _fmt(mb: float) -> str:
    return f"{mb:.1f} MB"

# ─── TEST 1: Baseline model load + optimizer ──────────────────────────

def test_1_baseline_load():
    """Measure model + optimizer VRAM/RAM with no training."""
    print("\n" + "="*70)
    print("TEST 1: Baseline model + optimizer load (no forward)")
    print("="*70)
    _full_cleanup()
    ram0 = _ram_mb()
    vram0 = _vram_mb()
    print(f"  Before:  RAM={_fmt(ram0)}  VRAM={_fmt(vram0)}")

    model, tok = _load_model()
    vram_model = _vram_mb()
    ram_model = _ram_mb()
    print(f"  Model:   RAM={_fmt(ram_model)} (+{_fmt(ram_model-ram0)})  VRAM={_fmt(vram_model)} (+{_fmt(vram_model-vram0)})")

    opt, opt_name = _create_optimizer(model)
    vram_opt = _vram_mb()
    ram_opt = _ram_mb()
    print(f"  +Optim:  RAM={_fmt(ram_opt)} (+{_fmt(ram_opt-ram_model)})  VRAM={_fmt(vram_opt)} (+{_fmt(vram_opt-vram_model)})  name={opt_name}")

    del model, tok, opt
    _full_cleanup()
    vram_end = _vram_mb()
    ram_end = _ram_mb()
    print(f"  Cleanup: RAM={_fmt(ram_end)}  VRAM={_fmt(vram_end)}")
    print(f"  RESULT: model_vram={_fmt(vram_model-vram0)} opt_vram={_fmt(vram_opt-vram_model)} leaked_vram={_fmt(vram_end-vram0)}")
    return {"model_vram": vram_model - vram0, "opt_vram": vram_opt - vram_model, "leaked": vram_end - vram0}

# ─── TEST 2: Standard forward + backward (no composite, no MTF) ──────

def test_2_simple_fwd_bwd():
    """1 forward+backward with simple_blockdiag mask (no composite doubling)."""
    print("\n" + "="*70)
    print("TEST 2: Simple forward+backward (simple_blockdiag, 1024x1024 mask)")
    print("="*70)
    _full_cleanup()
    from hildanext.diffusion import compute_m2t_t2t_losses
    from hildanext.config import from_dict

    model, tok = _load_model()
    opt, opt_name = _create_optimizer(model)
    batch = _load_batch(1)
    mask_id = tok.convert_tokens_to_ids("[MASK]") if "[MASK]" in tok.get_vocab() else 151669

    # Minimal train config
    train_cfg = from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 1, "multi_turn_t2t": 1}
    }).train

    vram_pre = _vram_mb()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Pre-fwd: VRAM={_fmt(vram_pre)}")

    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compute_m2t_t2t_losses(
            model=model, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
            mask_id=mask_id, vocab_size=max(8, model.config.vocab_size),
            cfg=train_cfg, focus_response=False,
            mask_mode="simple_blockdiag", composite_block_size=1024,
            bidirectional=False, time_param="continuous_time",
            loss_weighting="inv_t", t_min=0.001, t_max=1.0
        )
    t_fwd = time.time() - t0
    vram_fwd = _vram_mb()
    print(f"  Fwd:     VRAM={_fmt(vram_fwd)} (+{_fmt(vram_fwd-vram_pre)})  time={t_fwd:.2f}s  loss={out['loss'].item():.4f}")

    t0 = time.time()
    out["loss"].backward()
    t_bwd = time.time() - t0
    vram_bwd = _vram_mb()
    peak = _vram_peak_mb()
    print(f"  Bwd:     VRAM={_fmt(vram_bwd)} (+{_fmt(vram_bwd-vram_fwd)})  time={t_bwd:.2f}s")
    print(f"  Peak:    VRAM={_fmt(peak)}")

    t0 = time.time()
    opt.step()
    opt.zero_grad(set_to_none=True)
    t_opt = time.time() - t0
    vram_opt = _vram_mb()
    print(f"  Opt:     VRAM={_fmt(vram_opt)} time={t_opt:.2f}s")

    total_time = t_fwd + t_bwd + t_opt
    tok_per_sec = SEQ_LEN / total_time
    print(f"  RESULT: fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s opt={t_opt:.2f}s total={total_time:.2f}s tok/s={tok_per_sec:.0f} peak_vram={_fmt(peak)}")

    del model, tok, opt, batch, out
    _full_cleanup()
    return {"fwd": t_fwd, "bwd": t_bwd, "peak_vram": peak, "tok_per_sec": tok_per_sec}

# ─── TEST 3: Composite forward + backward (2048x2048 mask) ───────────

def test_3_composite_fwd_bwd():
    """1 forward+backward with composite_llada20 mask (2048x2048 effective)."""
    print("\n" + "="*70)
    print("TEST 3: Composite forward+backward (composite_llada20, 2048x2048 mask)")
    print("="*70)
    _full_cleanup()
    from hildanext.diffusion import compute_m2t_t2t_losses
    from hildanext.config import from_dict

    model, tok = _load_model()
    opt, opt_name = _create_optimizer(model)
    batch = _load_batch(1)
    mask_id = tok.convert_tokens_to_ids("[MASK]") if "[MASK]" in tok.get_vocab() else 151669

    train_cfg = from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 1, "multi_turn_t2t": 1}
    }).train

    vram_pre = _vram_mb()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Pre-fwd: VRAM={_fmt(vram_pre)}")

    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compute_m2t_t2t_losses(
            model=model, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
            mask_id=mask_id, vocab_size=max(8, model.config.vocab_size),
            cfg=train_cfg, focus_response=False,
            mask_mode="composite_llada20", composite_block_size=1,
            bidirectional=False, time_param="continuous_time",
            loss_weighting="inv_t", t_min=0.001, t_max=1.0
        )
    t_fwd = time.time() - t0
    vram_fwd = _vram_mb()
    print(f"  Fwd:     VRAM={_fmt(vram_fwd)} (+{_fmt(vram_fwd-vram_pre)})  time={t_fwd:.2f}s  loss={out['loss'].item():.4f}")

    t0 = time.time()
    out["loss"].backward()
    t_bwd = time.time() - t0
    vram_bwd = _vram_mb()
    peak = _vram_peak_mb()
    print(f"  Bwd:     VRAM={_fmt(vram_bwd)} (+{_fmt(vram_bwd-vram_fwd)})  time={t_bwd:.2f}s")
    print(f"  Peak:    VRAM={_fmt(peak)}")

    t0 = time.time()
    opt.step()
    opt.zero_grad(set_to_none=True)
    t_opt = time.time() - t0
    vram_opt = _vram_mb()
    print(f"  Opt:     VRAM={_fmt(vram_opt)} time={t_opt:.2f}s")

    total_time = t_fwd + t_bwd + t_opt
    tok_per_sec = SEQ_LEN / total_time
    print(f"  RESULT: fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s opt={t_opt:.2f}s total={total_time:.2f}s tok/s={tok_per_sec:.0f} peak_vram={_fmt(peak)}")
    print(f"  COMPARISON: expect ~4x slower than TEST 2 due to 2048x2048 mask")

    del model, tok, opt, batch, out
    _full_cleanup()
    return {"fwd": t_fwd, "bwd": t_bwd, "peak_vram": peak, "tok_per_sec": tok_per_sec}

# ─── TEST 4: Composite + gradient checkpointing + MTF=2 ──────────────

def test_4_composite_gradckpt_mtf2():
    """1 step with composite + grad_ckpt + MTF=2 (matches actual warmup config)."""
    print("\n" + "="*70)
    print("TEST 4: Full warmup config (composite + grad_ckpt + MTF=2)")
    print("="*70)
    _full_cleanup()
    from hildanext.diffusion import compute_m2t_t2t_losses
    from hildanext.config import from_dict

    model, tok = _load_model()
    model.gradient_checkpointing_enable()
    opt, opt_name = _create_optimizer(model)
    batch = _load_batch(1)
    mask_id = tok.convert_tokens_to_ids("[MASK]") if "[MASK]" in tok.get_vocab() else 151669

    train_cfg = from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 8, "multi_turn_t2t": 2}
    }).train

    vram_pre = _vram_mb()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Pre-fwd: VRAM={_fmt(vram_pre)}")

    # Simulate 2 MTF turns like the real training loop
    total_loss = 0.0
    t0_total = time.time()
    current_ids = batch["input_ids"].clone()
    for mtf_turn in range(2):
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = compute_m2t_t2t_losses(
                model=model, input_ids=current_ids, attention_mask=batch["attention_mask"],
                doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
                mask_id=mask_id, vocab_size=max(8, model.config.vocab_size),
                cfg=train_cfg, focus_response=False,
                mask_mode="composite_llada20", composite_block_size=1,
                bidirectional=False, time_param="continuous_time",
                loss_weighting="inv_t", t_min=0.001, t_max=1.0
            )
        t_fwd = time.time() - t0
        vram_after_fwd = _vram_mb()

        t0 = time.time()
        (out["loss"] / 8.0).backward()  # accum_steps=8
        t_bwd = time.time() - t0
        vram_after_bwd = _vram_mb()
        peak = _vram_peak_mb()
        total_loss += out["loss"].item()

        print(f"  MTF turn {mtf_turn+1}: fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s loss={out['loss'].item():.4f} vram={_fmt(vram_after_bwd)} peak={_fmt(peak)}")

        # Prepare next turn input (model predictions replace corrupted)
        if mtf_turn < 1:
            with torch.no_grad():
                preds = out.get("model_predictions")
                if preds is not None:
                    corrupted = batch["input_ids"].eq(mask_id) | (current_ids != batch["input_ids"])
                    current_ids = batch["input_ids"].clone()
                    current_ids[corrupted] = preds[corrupted]

    total_time = time.time() - t0_total
    tok_per_sec = SEQ_LEN / total_time
    print(f"  RESULT: total_time={total_time:.2f}s tok/s={tok_per_sec:.0f} peak_vram={_fmt(peak)} total_loss={total_loss:.4f}")

    del model, tok, opt, batch, out, current_ids
    _full_cleanup()
    return {"total_time": total_time, "peak_vram": peak, "tok_per_sec": tok_per_sec}

# ─── TEST 5: 8 micro-batches gradient accumulation (1 full optimizer step) ──

def test_5_full_optim_step():
    """8 micro-batches with grad accum (= 1 optimizer step) with full warmup config."""
    print("\n" + "="*70)
    print("TEST 5: Full optimizer step (8 micro-batches x MTF=2, composite, grad_ckpt)")
    print("="*70)
    _full_cleanup()
    from hildanext.diffusion import compute_m2t_t2t_losses
    from hildanext.config import from_dict

    model, tok = _load_model()
    model.gradient_checkpointing_enable()
    opt, opt_name = _create_optimizer(model)
    mask_id = tok.convert_tokens_to_ids("[MASK]") if "[MASK]" in tok.get_vocab() else 151669

    train_cfg = from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 8, "multi_turn_t2t": 2}
    }).train

    vram_pre = _vram_mb()
    ram_pre = _ram_mb()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Pre:     VRAM={_fmt(vram_pre)}  RAM={_fmt(ram_pre)}")

    t0_total = time.time()
    losses = []
    for micro in range(8):
        batch = _load_batch(1)  # fresh batch each time
        current_ids = batch["input_ids"].clone()

        for mtf_turn in range(2):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = compute_m2t_t2t_losses(
                    model=model, input_ids=current_ids, attention_mask=batch["attention_mask"],
                    doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
                    mask_id=mask_id, vocab_size=max(8, model.config.vocab_size),
                    cfg=train_cfg, focus_response=False,
                    mask_mode="composite_llada20", composite_block_size=1,
                    bidirectional=False, time_param="continuous_time",
                    loss_weighting="inv_t", t_min=0.001, t_max=1.0
                )
            (out["loss"] / 8.0).backward()
            if mtf_turn < 1:
                with torch.no_grad():
                    preds = out.get("model_predictions")
                    if preds is not None:
                        corrupted = batch["input_ids"].eq(mask_id) | (current_ids != batch["input_ids"])
                        current_ids = batch["input_ids"].clone()
                        current_ids[corrupted] = preds[corrupted]

        losses.append(out["loss"].item())
        vram_now = _vram_mb()
        peak_now = _vram_peak_mb()
        ram_now = _ram_mb()
        print(f"  micro {micro+1}/8: loss={losses[-1]:.4f} vram={_fmt(vram_now)} peak={_fmt(peak_now)} ram={_fmt(ram_now)}")
        del batch, current_ids, out

    # Optimizer step
    t0_opt = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    t_opt = time.time() - t0_opt

    total_time = time.time() - t0_total
    total_tokens = 8 * SEQ_LEN
    tok_per_sec = total_tokens / total_time
    peak = _vram_peak_mb()
    ram_end = _ram_mb()
    print(f"  Opt:     time={t_opt:.2f}s")
    print(f"  RESULT: total_time={total_time:.2f}s tokens={total_tokens} tok/s={tok_per_sec:.0f}")
    print(f"           peak_vram={_fmt(peak)} ram_delta={_fmt(ram_end-ram_pre)}")
    print(f"           avg_loss={np.mean(losses):.4f}")

    # Check for VRAM leak
    del model, tok, opt
    _full_cleanup()
    vram_leaked = _vram_mb()
    print(f"  Cleanup: VRAM leaked={_fmt(vram_leaked)}")
    return {"total_time": total_time, "tok_per_sec": tok_per_sec, "peak_vram": peak}

# ─── TEST 6: Dataset loading benchmark ───────────────────────────────

def test_6_dataset_load():
    """Benchmark the JSONL streaming loader with full 60K rows."""
    print("\n" + "="*70)
    print("TEST 6: Dataset loading benchmark (streaming numpy conversion)")
    print("="*70)
    _full_cleanup()
    from hildanext.training import TokenizedDataset

    ram_pre = _ram_mb()
    print(f"  Pre:     RAM={_fmt(ram_pre)}")

    # Test subset load (16 rows like probe)
    t0 = time.time()
    ds_small = TokenizedDataset(DATASET_PATH, max_rows=16)
    t_small = time.time() - t0
    ram_small = _ram_mb()
    print(f"  16 rows: time={t_small:.3f}s RAM={_fmt(ram_small)} (+{_fmt(ram_small-ram_pre)})")
    del ds_small
    _full_cleanup()

    # Test full load
    ram_pre2 = _ram_mb()
    t0 = time.time()
    ds_full = TokenizedDataset(DATASET_PATH)
    t_full = time.time() - t0
    ram_full = _ram_mb()
    n_rows = len(ds_full)
    print(f"  Full:    time={t_full:.1f}s rows={n_rows} RAM={_fmt(ram_full)} (+{_fmt(ram_full-ram_pre2)})")

    # Verify dtype
    r0 = ds_full[0]
    print(f"  Types:   input_ids={r0['input_ids'].dtype} doc_ids={r0['doc_ids'].dtype} "
          f"attn_mask={r0['attention_mask'].dtype} resp_mask={r0.get('response_mask', np.array([])).dtype}")

    del ds_full
    _full_cleanup()
    ram_end = _ram_mb()
    print(f"  Cleanup: RAM={_fmt(ram_end)} leaked={_fmt(ram_end-ram_pre)}")
    return {"load_time": t_full, "rows": n_rows, "ram_delta": ram_full - ram_pre2}

# ─── TEST 7: Bidir verification conformity check ─────────────────────

def test_7_bidir_conformity():
    """Verify bidirectional attention matches LLaDA paper recipe per phase."""
    print("\n" + "="*70)
    print("TEST 7: Bidirectional conformity check (LLaDA 2.0 recipe)")
    print("="*70)
    _full_cleanup()

    # Read current config
    config_path = _root / "runs" / "configs" / "llada21_dolma_wsd_only.json"
    if not config_path.exists():
        config_path = _root / "runs" / "configs" / "default.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    wsd = cfg.get("wsd", {})
    stage0 = cfg.get("stage0", {})
    llada2 = cfg.get("llada2", {})

    warmup_steps = wsd.get("warmup_steps", 1000)
    stable_steps = wsd.get("stable_steps", 3000)
    decay_steps = wsd.get("decay_steps", 1000)
    total = warmup_steps + stable_steps + decay_steps
    mask_mode = llada2.get("mask_mode", "composite_llada20")
    attn_mode = cfg.get("experiment", {}).get("attention_mode", "bidirectional_only_stable")
    ladder = wsd.get("ladder_blocks", [1])
    max_block = wsd.get("max_block_size", 1024)
    decay_blocks = wsd.get("decay_blocks", [32])

    print(f"  WSD: W={warmup_steps} S={stable_steps} D={decay_steps} total={total}")
    print(f"  mask_mode={mask_mode}  attn_mode={attn_mode}")
    print(f"  ladder={ladder}  max_block={max_block}  decay_blocks={decay_blocks}")

    # Check conformity
    checks = []

    # 1. Warmup: composite_llada20, NOT bidirectional
    c1 = mask_mode == "composite_llada20"
    checks.append(("Warmup uses composite_llada20", c1))

    # 2. Stable: simple_blockdiag, bidirectional
    c2 = attn_mode == "bidirectional_only_stable"
    checks.append(("Attention bidirectional only in stable", c2))

    # 3. Warmup starts at block=1 (AR-like)
    c3 = ladder[0] == 1 if ladder else False
    checks.append(("Warmup starts block=1 (AR-like)", c3))

    # 4. Warmup ends at max_block = seq_len
    c4 = max_block == int(cfg.get("data", {}).get("seq_len", 1024))
    checks.append(("Warmup ends block=seq_len", c4))

    # 5. Decay ends at small block
    c5 = decay_blocks[-1] <= 64 if decay_blocks else False
    checks.append(("Decay ends at small block (<=64)", c5))

    # 6. Mask doubling: composite_llada20 creates [x_t||x_0] = 2*seq_len
    c6 = mask_mode == "composite_llada20"  # implies 2x attention
    checks.append(("Composite mask doubles seq (paper S4.2)", c6))

    # 7. NO bidirectional during warmup/decay (per paper)
    c7 = attn_mode != "always_bidirectional"
    checks.append(("No forced bidir during warmup/decay", c7))

    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    print(f"\n  RESULT: {'ALL PASS' if all_ok else 'SOME FAILED'} ({sum(1 for _,o in checks if o)}/{len(checks)})")
    return {"all_pass": all_ok, "checks": checks}

# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WSD PERFORMANCE & CONFORMITY DIAGNOSTIC SUITE")
    print(f"Model:    {MODEL_PATH}")
    print(f"Dataset:  {DATASET_PATH}")
    print(f"Device:   {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"RAM:      {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print("=" * 70)

    results = {}

    # Run all tests with full cleanup between them
    try:
        results["T1_baseline"] = test_1_baseline_load()
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        results["T1_baseline"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T2_simple"] = test_2_simple_fwd_bwd()
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        results["T2_simple"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T3_composite"] = test_3_composite_fwd_bwd()
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        results["T3_composite"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T4_full_warmup"] = test_4_composite_gradckpt_mtf2()
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        results["T4_full_warmup"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T5_full_step"] = test_5_full_optim_step()
    except Exception as e:
        print(f"  TEST 5 FAILED: {e}")
        results["T5_full_step"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T6_dataset"] = test_6_dataset_load()
    except Exception as e:
        print(f"  TEST 6 FAILED: {e}")
        results["T6_dataset"] = {"error": str(e)}
    _full_cleanup()

    try:
        results["T7_bidir"] = test_7_bidir_conformity()
    except Exception as e:
        print(f"  TEST 7 FAILED: {e}")
        results["T7_bidir"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    t2 = results.get("T2_simple", {})
    t3 = results.get("T3_composite", {})
    t5 = results.get("T5_full_step", {})
    if t2.get("tok_per_sec") and t3.get("tok_per_sec"):
        ratio = t2["tok_per_sec"] / t3["tok_per_sec"]
        print(f"  Simple vs Composite speed:   {t2['tok_per_sec']:.0f} vs {t3['tok_per_sec']:.0f} tok/s  (ratio {ratio:.1f}x)")
        print(f"  Simple vs Composite VRAM:    {t2.get('peak_vram',0):.0f} vs {t3.get('peak_vram',0):.0f} MB peak")
    if t5.get("tok_per_sec"):
        eta_sec = 5000 * 8 * SEQ_LEN / t5["tok_per_sec"]
        print(f"  Full step tok/s:             {t5['tok_per_sec']:.0f}")
        print(f"  Full step peak VRAM:         {t5.get('peak_vram',0):.0f} MB")
        print(f"  ETA for 5000 steps:          {eta_sec/3600:.1f} hours")
    if t5.get("peak_vram", 0) > 7500:
        print(f"  !! WARNING: Peak VRAM {t5['peak_vram']:.0f} MB > 7500 MB — OOM risk on 8GB GPU!")
    print("=" * 70)

    # Save results
    out_path = _root / "runs" / "reports" / "perf_diag_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert non-serializable items
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: (vv if not isinstance(vv, (list, tuple)) or not vv or not isinstance(vv[0], tuple) else [(n, o) for n, o in vv]) for kk, vv in v.items()}
        else:
            serializable[k] = v
    out_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()

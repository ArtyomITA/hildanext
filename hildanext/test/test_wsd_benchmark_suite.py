"""
Stage 0 Benchmark Suite — LLaDA 2.1-First WSD on GTX 1080
==========================================================
7 tests, each fully isolated with VRAM/RAM cleanup.
All tests use the same data and fixed seed for reproducibility.

Pre-requisites:
    - Fix 0.3 (TDR registry TdrDelay=60) applied and rebooted BEFORE running Test 3/4
    - conda activate mdm
    - cd e:\\DIFFUSION\\HildaNext\\hildanext
    - python test\\test_wsd_benchmark_suite.py [--test N] [--output FILE]

Reference: STAGE0_MASTER_PLAN.md Blocco 2
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "backend" / "src"))

MODEL_PATH = str(_root / "models" / "qwen3-0.6b")
DATASET_PATH = str(_root / "data" / "tokenized_qwen_wsd" / "qwen_wsd_run" / "train.jsonl")
SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ─── Helpers ───────────────────────────────────────────────────────────

def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def _vram_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2

def _full_cleanup():
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def _load_model(grad_ckpt: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.train()
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    return model, tok

def _load_batch(n_rows: int = 1):
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

def _get_mask_id(tok) -> int:
    v = tok.get_vocab()
    if "[MASK]" in v:
        return v["[MASK]"]
    return 151669

def _get_train_cfg(mtf: int = 1):
    from hildanext.config import from_dict
    return from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 1, "multi_turn_t2t": mtf}
    }).train

def _make_optimizer(model, name: str, lr: float = 5e-5, wd: float = 0.1,
                    betas: Tuple[float,float] = (0.9, 0.95)):
    """Create optimizer by name. Returns (optimizer, actual_name)."""
    if name == "adamw":
        fused = DEVICE == "cuda"
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                 betas=betas, fused=fused), "AdamW(fused)" if fused else "AdamW"
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=wd,
                                   betas=betas), "AdamW8bit"
    if name == "paged_adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, weight_decay=wd,
                                        betas=betas), "PagedAdamW8bit"
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(model.parameters(), lr=lr, relative_step=False,
                         scale_parameter=False, warmup_init=False), "Adafactor"
    raise ValueError(f"Unknown optimizer: {name}")

def _run_n_steps(model, opt, mask_id: int, vocab_size: int, n_steps: int,
                 mask_mode: str, block_size: int, bidirectional: bool,
                 mtf: int = 1, seq_len: int = SEQ_LEN,
                 use_amp: bool = True) -> Dict[str, Any]:
    """Run n optimizer steps and collect metrics."""
    from hildanext.diffusion import compute_m2t_t2t_losses
    train_cfg = _get_train_cfg(mtf=mtf)

    losses = []
    grad_norms = []
    bwd_times = []
    nan_count = 0

    for step in range(n_steps):
        batch = _load_batch(1)
        # Truncate if needed for reduced seq_len
        if seq_len < SEQ_LEN:
            for k in batch:
                batch[k] = batch[k][:, :seq_len]

        _set_seed(SEED + step)
        current_ids = batch["input_ids"].clone()
        step_loss = 0.0

        for mtf_turn in range(mtf):
            mtf_target = batch["input_ids"] if mtf_turn > 0 else None
            ctx = torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp)
            with ctx:
                out = compute_m2t_t2t_losses(
                    model=model, input_ids=current_ids,
                    attention_mask=batch["attention_mask"],
                    doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
                    mask_id=mask_id, vocab_size=vocab_size,
                    cfg=train_cfg, focus_response=False,
                    mask_mode=mask_mode, composite_block_size=block_size,
                    bidirectional=bidirectional, time_param="continuous_time",
                    loss_weighting="inv_t", t_min=0.001, t_max=1.0,
                    target_ids=mtf_target
                )
            turn_loss = out["loss"] / float(mtf)

            t_bwd_start = time.time()
            turn_loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # accurate backward timing
            bwd_times.append(time.time() - t_bwd_start)

            step_loss += out["loss"].detach().item()

            if mtf_turn < mtf - 1:
                with torch.no_grad():
                    preds = out.get("model_predictions")
                    if preds is not None:
                        corrupted = out.get("corrupted_positions",
                                            batch["input_ids"].eq(mask_id))
                        current_ids = batch["input_ids"].clone()
                        current_ids[corrupted] = preds[corrupted]

        step_loss /= float(mtf)
        if not math.isfinite(step_loss):
            nan_count += 1
        losses.append(step_loss)

        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        grad_norms.append(gn)
        opt.step()
        opt.zero_grad(set_to_none=True)

        del batch, current_ids, out

    return {
        "losses": losses,
        "loss_final": losses[-1] if losses else float("nan"),
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "bwd_time_mean": float(np.mean(bwd_times)) if bwd_times else 0.0,
        "bwd_time_max": float(np.max(bwd_times)) if bwd_times else 0.0,
        "nan_count": nan_count,
        "vram_peak_mb": _vram_peak_mb(),
    }


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Optimizer Comparison
# ═══════════════════════════════════════════════════════════════════════

def test_1_optimizer_comparison() -> Dict[str, Any]:
    """10 steps per optimizer, simple_blockdiag stable phase, seq_len=1024, grad_ckpt=True."""
    print("\n" + "=" * 70)
    print("TEST 1: Optimizer Comparison (10 steps each)")
    print("=" * 70)

    optimizers = ["adamw", "adamw8bit", "paged_adamw8bit", "adafactor"]
    results = {}

    for opt_name in optimizers:
        _full_cleanup()
        _set_seed()
        print(f"\n  --- {opt_name} ---")

        try:
            ram0 = _ram_mb()
            model, tok = _load_model(grad_ckpt=True)
            mask_id = _get_mask_id(tok)
            vocab_size = model.config.vocab_size

            opt, actual_name = _make_optimizer(model, opt_name)
            vram_after_init = _vram_mb()
            print(f"  Init VRAM: {vram_after_init:.1f} MB  optimizer={actual_name}")

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            metrics = _run_n_steps(model, opt, mask_id, vocab_size, n_steps=10,
                                   mask_mode="simple_blockdiag", block_size=1024,
                                   bidirectional=True, mtf=1)
            elapsed = time.time() - t0

            tok_per_sec = (10 * SEQ_LEN) / elapsed
            ram_delta = _ram_mb() - ram0

            results[opt_name] = {
                "optimizer": actual_name,
                "vram_peak_mb": metrics["vram_peak_mb"],
                "ram_delta_mb": ram_delta,
                "tok_per_sec": tok_per_sec,
                "loss_final": metrics["loss_final"],
                "loss_mean": metrics["loss_mean"],
                "grad_norm_mean": metrics["grad_norm_mean"],
                "nan_count": metrics["nan_count"],
                "elapsed_sec": elapsed,
                "oom": False,
            }
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB")
            print(f"  tok/s: {tok_per_sec:.0f}  loss@10: {metrics['loss_final']:.4f}")
            print(f"  grad_norm_mean: {metrics['grad_norm_mean']:.4f}  NaN: {metrics['nan_count']}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            results[opt_name] = {"optimizer": opt_name, "oom": True, "error": str(e)}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[opt_name] = {"optimizer": opt_name, "error": str(e)}
        finally:
            for v in list(locals().values()):
                if isinstance(v, torch.nn.Module):
                    del v
            _full_cleanup()

    print(f"\n  === SUMMARY ===")
    for name, r in results.items():
        if r.get("oom"):
            print(f"  {name}: OOM")
        elif r.get("error"):
            print(f"  {name}: ERROR — {r['error'][:80]}")
        else:
            print(f"  {name}: peak={r['vram_peak_mb']:.0f}MB tok/s={r['tok_per_sec']:.0f} "
                  f"loss={r['loss_final']:.4f} NaN={r['nan_count']}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Gradient Checkpointing
# ═══════════════════════════════════════════════════════════════════════

def test_2_gradient_checkpointing(best_optimizer: str = "paged_adamw8bit") -> Dict[str, Any]:
    """grad_ckpt ON vs OFF with the best optimizer from Test 1."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Gradient Checkpointing (optimizer={best_optimizer})")
    print("=" * 70)

    results = {}
    for grad_ckpt in [True, False]:
        label = "grad_ckpt_ON" if grad_ckpt else "grad_ckpt_OFF"
        _full_cleanup()
        _set_seed()
        print(f"\n  --- {label} ---")

        try:
            model, tok = _load_model(grad_ckpt=grad_ckpt)
            mask_id = _get_mask_id(tok)
            opt, actual_name = _make_optimizer(model, best_optimizer)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            metrics = _run_n_steps(model, opt, mask_id, model.config.vocab_size, n_steps=5,
                                   mask_mode="simple_blockdiag", block_size=1024,
                                   bidirectional=True, mtf=1)
            elapsed = time.time() - t0

            tok_per_sec = (5 * SEQ_LEN) / elapsed
            results[label] = {
                "grad_ckpt": grad_ckpt,
                "vram_peak_mb": metrics["vram_peak_mb"],
                "tok_per_sec": tok_per_sec,
                "loss_final": metrics["loss_final"],
                "oom": False,
            }
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB  tok/s: {tok_per_sec:.0f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM! grad_ckpt={grad_ckpt} cannot fit in 8GB")
            results[label] = {"grad_ckpt": grad_ckpt, "oom": True}
        finally:
            _full_cleanup()

    strictly_required = results.get("grad_ckpt_OFF", {}).get("oom", False)
    print(f"\n  Grad ckpt strictly required for 8GB? {'YES' if strictly_required else 'NO'}")
    results["strictly_required"] = strictly_required
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Mask Mode & Backward Time (CRITICAL for TDR)
# ═══════════════════════════════════════════════════════════════════════

def test_3_mask_mode_backward_time() -> Dict[str, Any]:
    """Measure backward_time for each mask mode + block_size combination."""
    print("\n" + "=" * 70)
    print("TEST 3: Mask Mode & Backward Time (TDR threshold check)")
    print("=" * 70)

    configs = [
        ("composite_bs1",   "composite_llada20", 1,    False),  # warmup start
        ("composite_bs32",  "composite_llada20", 32,   False),  # decay end
        ("composite_bs1024","composite_llada20", 1024, False),  # warmup end
        ("simple_bidir",    "simple_blockdiag",  1024, True),   # stable phase
    ]

    results = {}
    TDR_THRESHOLD = 60.0  # seconds, with TdrDelay=60

    for label, mask_mode, block_size, bidirectional in configs:
        _full_cleanup()
        _set_seed()
        effective_seq = SEQ_LEN * 2 if mask_mode == "composite_llada20" else SEQ_LEN
        print(f"\n  --- {label} (mask={mask_mode}, bs={block_size}, bidir={bidirectional}, eff_seq={effective_seq}) ---")

        try:
            model, tok = _load_model(grad_ckpt=True)
            mask_id = _get_mask_id(tok)
            opt, _ = _make_optimizer(model, "paged_adamw8bit")

            torch.cuda.reset_peak_memory_stats()
            metrics = _run_n_steps(model, opt, mask_id, model.config.vocab_size, n_steps=5,
                                   mask_mode=mask_mode, block_size=block_size,
                                   bidirectional=bidirectional, mtf=1)

            tok_per_sec = SEQ_LEN / (metrics["bwd_time_mean"] + 0.001)
            tdr_safe = metrics["bwd_time_max"] < TDR_THRESHOLD

            results[label] = {
                "mask_mode": mask_mode,
                "block_size": block_size,
                "bidirectional": bidirectional,
                "effective_seq": effective_seq,
                "bwd_time_mean_sec": metrics["bwd_time_mean"],
                "bwd_time_max_sec": metrics["bwd_time_max"],
                "vram_peak_mb": metrics["vram_peak_mb"],
                "tok_per_sec": tok_per_sec,
                "loss_mean": metrics["loss_mean"],
                "tdr_safe": tdr_safe,
                "oom": False,
            }
            status = "SAFE" if tdr_safe else "EXCEEDS TDR"
            print(f"  bwd_time: mean={metrics['bwd_time_mean']:.2f}s max={metrics['bwd_time_max']:.2f}s [{status}]")
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB  tok/s: {tok_per_sec:.0f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM!")
            results[label] = {"mask_mode": mask_mode, "block_size": block_size, "oom": True}
        finally:
            _full_cleanup()

    # Decision output
    composite_1024_ok = results.get("composite_bs1024", {}).get("tdr_safe", False)
    print(f"\n  === DECISION ===")
    if composite_1024_ok:
        print("  composite_llada20 at seq_len=1024 is TDR-safe → Option A (Full WSD) is feasible")
    else:
        print("  composite_llada20 at seq_len=1024 EXCEEDS TDR → need Test 4 (reduced seq_len)")
    results["composite_1024_tdr_safe"] = composite_1024_ok
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: BDLM at Reduced seq_len (only if Test 3 shows TDR risk)
# ═══════════════════════════════════════════════════════════════════════

def test_4_reduced_seqlen() -> Dict[str, Any]:
    """Composite at reduced seq_len to find feasible BDLM configuration."""
    print("\n" + "=" * 70)
    print("TEST 4: BDLM at Reduced seq_len")
    print("=" * 70)

    seq_lens = [512, 256]
    results = {}
    TDR_THRESHOLD = 60.0

    for sl in seq_lens:
        _full_cleanup()
        _set_seed()
        effective = sl * 2
        print(f"\n  --- seq_len={sl} (doubling → {effective}) ---")

        try:
            model, tok = _load_model(grad_ckpt=True)
            mask_id = _get_mask_id(tok)
            opt, _ = _make_optimizer(model, "paged_adamw8bit")

            torch.cuda.reset_peak_memory_stats()
            metrics = _run_n_steps(model, opt, mask_id, model.config.vocab_size, n_steps=5,
                                   mask_mode="composite_llada20", block_size=sl,
                                   bidirectional=False, mtf=1, seq_len=sl)

            tdr_safe = metrics["bwd_time_max"] < TDR_THRESHOLD
            results[f"seq_{sl}"] = {
                "seq_len": sl,
                "effective_seq": effective,
                "bwd_time_mean_sec": metrics["bwd_time_mean"],
                "bwd_time_max_sec": metrics["bwd_time_max"],
                "vram_peak_mb": metrics["vram_peak_mb"],
                "loss_mean": metrics["loss_mean"],
                "tdr_safe": tdr_safe,
                "oom": False,
            }
            status = "SAFE" if tdr_safe else "EXCEEDS TDR"
            print(f"  bwd_time: mean={metrics['bwd_time_mean']:.2f}s max={metrics['bwd_time_max']:.2f}s [{status}]")
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM!")
            results[f"seq_{sl}"] = {"seq_len": sl, "oom": True}
        finally:
            _full_cleanup()

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: MTF Impact
# ═══════════════════════════════════════════════════════════════════════

def test_5_mtf_impact() -> Dict[str, Any]:
    """MTF=1 vs MTF=2 in stable and warmup."""
    print("\n" + "=" * 70)
    print("TEST 5: MTF Impact (Multi-Turn Forward)")
    print("=" * 70)

    configs = [
        ("stable_mtf1",  "simple_blockdiag", 1024, True,  1),
        ("stable_mtf2",  "simple_blockdiag", 1024, True,  2),
        ("warmup_mtf2",  "composite_llada20", 1,   False, 2),  # warmup start
    ]

    results = {}
    for label, mask_mode, block_size, bidirectional, mtf in configs:
        _full_cleanup()
        _set_seed()
        print(f"\n  --- {label} (mtf={mtf}) ---")

        try:
            model, tok = _load_model(grad_ckpt=True)
            mask_id = _get_mask_id(tok)
            opt, _ = _make_optimizer(model, "paged_adamw8bit")

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            metrics = _run_n_steps(model, opt, mask_id, model.config.vocab_size, n_steps=5,
                                   mask_mode=mask_mode, block_size=block_size,
                                   bidirectional=bidirectional, mtf=mtf)
            elapsed = time.time() - t0
            tok_per_sec = (5 * SEQ_LEN) / elapsed

            results[label] = {
                "mtf": mtf,
                "mask_mode": mask_mode,
                "vram_peak_mb": metrics["vram_peak_mb"],
                "tok_per_sec": tok_per_sec,
                "loss_final": metrics["loss_final"],
                "loss_mean": metrics["loss_mean"],
                "oom": False,
            }
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB  tok/s: {tok_per_sec:.0f}")
            print(f"  loss@5: {metrics['loss_final']:.4f}  mean: {metrics['loss_mean']:.4f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM!")
            results[label] = {"mtf": mtf, "oom": True}
        finally:
            _full_cleanup()

    # Compare MTF=1 vs MTF=2 in stable
    s1 = results.get("stable_mtf1", {})
    s2 = results.get("stable_mtf2", {})
    if not s1.get("oom") and not s2.get("oom") and s1.get("tok_per_sec") and s2.get("tok_per_sec"):
        speedup = s1["tok_per_sec"] / max(1, s2["tok_per_sec"])
        print(f"\n  MTF=1 vs MTF=2 stable: {speedup:.2f}x speed ratio")
        results["mtf1_vs_mtf2_speed_ratio"] = speedup

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: ETA Per-Phase
# ═══════════════════════════════════════════════════════════════════════

def test_6_eta_per_phase(test3_results: Optional[Dict] = None) -> Dict[str, Any]:
    """Calculate realistic ETA using measured tok/s per phase."""
    print("\n" + "=" * 70)
    print("TEST 6: ETA Per-Phase Calculation")
    print("=" * 70)

    # WSD config: W=1000, S=3000, D=1000
    warmup_steps = 1000
    stable_steps = 3000
    decay_steps = 1000
    accum = 8  # gradient accumulation

    # Get tok/s from Test 3 results or re-measure
    if test3_results is None:
        print("  No Test 3 results provided — using estimates")
        # Conservative estimates from previous diagnostics
        warmup_tok_s = 74.0
        stable_tok_s = 275.0
        decay_tok_s = 120.0
    else:
        # Warmup: average of composite_bs1 and composite_bs1024
        c1 = test3_results.get("composite_bs1", {}).get("tok_per_sec", 74)
        c1024 = test3_results.get("composite_bs1024", {}).get("tok_per_sec", 120)
        warmup_tok_s = (c1 + c1024) / 2  # weighted average over ladder

        stable_tok_s = test3_results.get("simple_bidir", {}).get("tok_per_sec", 275)

        c32 = test3_results.get("composite_bs32", {}).get("tok_per_sec", 120)
        decay_tok_s = (c1024 + c32) / 2

    # Each optimizer step = accum micro-batches × SEQ_LEN tokens
    tokens_per_step = accum * SEQ_LEN

    warmup_sec = warmup_steps * tokens_per_step / max(1, warmup_tok_s)
    stable_sec = stable_steps * tokens_per_step / max(1, stable_tok_s)
    decay_sec = decay_steps * tokens_per_step / max(1, decay_tok_s)
    total_sec = warmup_sec + stable_sec + decay_sec

    results = {
        "warmup": {"steps": warmup_steps, "tok_per_sec": warmup_tok_s, "eta_hours": warmup_sec / 3600},
        "stable": {"steps": stable_steps, "tok_per_sec": stable_tok_s, "eta_hours": stable_sec / 3600},
        "decay":  {"steps": decay_steps,  "tok_per_sec": decay_tok_s,  "eta_hours": decay_sec / 3600},
        "total_eta_hours": total_sec / 3600,
        "total_tokens": (warmup_steps + stable_steps + decay_steps) * tokens_per_step,
    }

    print(f"  Warmup  ({warmup_steps} steps): {warmup_tok_s:.0f} tok/s → {warmup_sec/3600:.1f} hours")
    print(f"  Stable  ({stable_steps} steps): {stable_tok_s:.0f} tok/s → {stable_sec/3600:.1f} hours")
    print(f"  Decay   ({decay_steps} steps):  {decay_tok_s:.0f} tok/s → {decay_sec/3600:.1f} hours")
    print(f"  TOTAL:  {total_sec/3600:.1f} hours ({total_sec/3600/24:.1f} days)")
    print(f"  Total tokens: {results['total_tokens']:,}")

    feasible = total_sec / 3600 < 200  # Gate 4: < 200 hours
    results["feasible"] = feasible
    print(f"  Feasibility Gate (<200h): {'PASS' if feasible else 'FAIL'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: fp16 vs fp32 (Baseline Quality)
# ═══════════════════════════════════════════════════════════════════════

def test_7_fp16_vs_fp32() -> Dict[str, Any]:
    """10 steps with fp16 vs fp32 to verify no NaN/instability on Pascal."""
    print("\n" + "=" * 70)
    print("TEST 7: fp16 vs fp32 Quality Baseline")
    print("=" * 70)

    results = {}
    for dtype_label, use_amp, model_dtype in [
        ("fp16", True,  torch.float16),
        ("fp32", False, torch.float32),
    ]:
        _full_cleanup()
        _set_seed()
        print(f"\n  --- {dtype_label} ---")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=model_dtype, trust_remote_code=True
            ).to(DEVICE)
            model.train()
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
            except TypeError:
                model.gradient_checkpointing_enable()

            mask_id = _get_mask_id(tok)
            opt, _ = _make_optimizer(model, "paged_adamw8bit" if dtype_label == "fp16" else "adamw")

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            metrics = _run_n_steps(model, opt, mask_id, model.config.vocab_size, n_steps=10,
                                   mask_mode="simple_blockdiag", block_size=1024,
                                   bidirectional=True, mtf=1, use_amp=use_amp)
            elapsed = time.time() - t0
            tok_per_sec = (10 * SEQ_LEN) / elapsed

            results[dtype_label] = {
                "dtype": dtype_label,
                "vram_peak_mb": metrics["vram_peak_mb"],
                "tok_per_sec": tok_per_sec,
                "loss_final": metrics["loss_final"],
                "loss_mean": metrics["loss_mean"],
                "grad_norm_mean": metrics["grad_norm_mean"],
                "nan_count": metrics["nan_count"],
                "oom": False,
            }
            print(f"  VRAM peak: {metrics['vram_peak_mb']:.1f} MB  tok/s: {tok_per_sec:.0f}")
            print(f"  loss@10: {metrics['loss_final']:.4f}  NaN: {metrics['nan_count']}")
            print(f"  grad_norm_mean: {metrics['grad_norm_mean']:.4f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM!")
            results[dtype_label] = {"dtype": dtype_label, "oom": True}
        finally:
            _full_cleanup()

    # Stability check
    fp16_nan = results.get("fp16", {}).get("nan_count", 0)
    fp32_nan = results.get("fp32", {}).get("nan_count", 0)
    if fp16_nan > 0 and fp32_nan == 0:
        print(f"\n  WARNING: fp16 produced {fp16_nan} NaN but fp32 did not → check GradScaler")
    elif fp16_nan == 0:
        print(f"\n  fp16 stable: no NaN detected")
    results["fp16_stable"] = fp16_nan == 0

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 0 Benchmark Suite")
    parser.add_argument("--test", type=int, default=0, help="Run specific test (1-7), 0=all")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--best-optimizer", type=str, default="paged_adamw8bit",
                        help="Optimizer for Test 2 (default: paged_adamw8bit)")
    args = parser.parse_args()

    output_path = args.output or str(_root / "reports" / "benchmark_suite_results.json")

    print("=" * 70)
    print("STAGE 0 BENCHMARK SUITE — LLaDA 2.1-First WSD")
    print(f"Plan ref: STAGE0_MASTER_PLAN.md Blocco 2")
    print(f"Model:    {MODEL_PATH}")
    print(f"Dataset:  {DATASET_PATH}")
    print(f"Device:   {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        print(f"bf16:     {torch.cuda.is_bf16_supported()}")
    print(f"RAM:      {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Seed:     {SEED}")
    print("=" * 70)

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "vram_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0,
            "ram_gb": psutil.virtual_memory().total / 1024**3,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        },
        "config": {
            "model": MODEL_PATH,
            "seq_len": SEQ_LEN,
            "seed": SEED,
        },
    }

    run_all = args.test == 0

    # Test 1
    if run_all or args.test == 1:
        try:
            all_results["test_1_optimizer"] = test_1_optimizer_comparison()
        except Exception as e:
            print(f"\n  TEST 1 FAILED: {e}")
            all_results["test_1_optimizer"] = {"error": str(e)}
        _full_cleanup()

    # Test 2
    if run_all or args.test == 2:
        try:
            all_results["test_2_grad_ckpt"] = test_2_gradient_checkpointing(args.best_optimizer)
        except Exception as e:
            print(f"\n  TEST 2 FAILED: {e}")
            all_results["test_2_grad_ckpt"] = {"error": str(e)}
        _full_cleanup()

    # Test 3
    if run_all or args.test == 3:
        try:
            all_results["test_3_mask_mode"] = test_3_mask_mode_backward_time()
        except Exception as e:
            print(f"\n  TEST 3 FAILED: {e}")
            all_results["test_3_mask_mode"] = {"error": str(e)}
        _full_cleanup()

    # Test 4
    if run_all or args.test == 4:
        # Only run if Test 3 showed TDR risk, or explicitly requested
        test3 = all_results.get("test_3_mask_mode", {})
        if args.test == 4 or not test3.get("composite_1024_tdr_safe", True):
            try:
                all_results["test_4_reduced_seqlen"] = test_4_reduced_seqlen()
            except Exception as e:
                print(f"\n  TEST 4 FAILED: {e}")
                all_results["test_4_reduced_seqlen"] = {"error": str(e)}
            _full_cleanup()
        else:
            print("\n  TEST 4 SKIPPED: composite_llada20@1024 is TDR-safe")
            all_results["test_4_reduced_seqlen"] = {"skipped": True, "reason": "composite_1024_tdr_safe"}

    # Test 5
    if run_all or args.test == 5:
        try:
            all_results["test_5_mtf"] = test_5_mtf_impact()
        except Exception as e:
            print(f"\n  TEST 5 FAILED: {e}")
            all_results["test_5_mtf"] = {"error": str(e)}
        _full_cleanup()

    # Test 6
    if run_all or args.test == 6:
        try:
            test3 = all_results.get("test_3_mask_mode")
            all_results["test_6_eta"] = test_6_eta_per_phase(test3)
        except Exception as e:
            print(f"\n  TEST 6 FAILED: {e}")
            all_results["test_6_eta"] = {"error": str(e)}
        _full_cleanup()

    # Test 7
    if run_all or args.test == 7:
        try:
            all_results["test_7_fp16_fp32"] = test_7_fp16_vs_fp32()
        except Exception as e:
            print(f"\n  TEST 7 FAILED: {e}")
            all_results["test_7_fp16_fp32"] = {"error": str(e)}
        _full_cleanup()

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    main()

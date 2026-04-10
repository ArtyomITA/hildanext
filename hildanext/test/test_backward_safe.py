"""
Safe Backward-Time Benchmark — single-process with per-step safety.
====================================================================
Runs composite tests at reduced eff_seq (<=1024) to avoid TDR freezes.
Each config runs in the same process with explicit CUDA sync and
VRAM monitoring. Model loaded once, reused across configs.

Usage:
    conda activate mdm
    cd e:\\DIFFUSION\\HildaNext\\hildanext
    python -u test/test_backward_safe.py

Previous results (preserved from earlier runs):
    Test 1: PagedAdamW8bit wins  -- 5220 MB, 511 tok/s, 0 NaN
    Test 2: grad_ckpt ON         -- 2914 MB, 451 tok/s (recommended)
    Test 3 old: composite_bs1 at eff_seq=2048 -> FROZE PC
"""
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "backend" / "src"))

MODEL_PATH = str(_root / "models" / "qwen3-0.6b")
DATASET_PATH = str(_root / "data" / "tokenized_qwen_wsd" / "qwen_wsd_run" / "train.jsonl")
DEVICE = "cuda"
SEED = 42


def _cleanup():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def _vram_mb():
    return torch.cuda.memory_allocated() / 1024**2


# (label, mask_mode, seq_len, block_size, bidirectional, n_steps)
CONFIGS = [
    # Baseline: simple_blockdiag at 1024 (proven safe, eff_seq=1024)
    ("simple_bidir_1024",    "simple_blockdiag",  1024, 1024, True,  2),

    # Composite at seq_len=256 -> eff_seq=512 (very safe)
    ("composite_256_bs1",    "composite_llada20",  256,   1, False, 2),
    ("composite_256_bs32",   "composite_llada20",  256,  32, False, 2),
    ("composite_256_bs256",  "composite_llada20",  256, 256, False, 2),

    # Composite at seq_len=512 -> eff_seq=1024 (PRIMARY TARGET)
    ("composite_512_bs1",    "composite_llada20",  512,   1, False, 2),
    ("composite_512_bs32",   "composite_llada20",  512,  32, False, 2),
    ("composite_512_bs512",  "composite_llada20",  512, 512, False, 2),
]


def run_all():
    print("=" * 70, flush=True)
    print("SAFE BACKWARD BENCHMARK -- single-process, per-step safety", flush=True)
    print(f"Configs: {len(CONFIGS)}", flush=True)
    print("=" * 70, flush=True)

    print("\n--- Previous Results (Test 1 & 2) ---", flush=True)
    print("  Test 1 Winner: PagedAdamW8bit -- 5220 MB, 511 tok/s, 0 NaN", flush=True)
    print("  Test 2: grad_ckpt ON -- 2914 MB, 451 tok/s (recommended)", flush=True)
    print("  Test 3 old: composite_bs1 at eff_seq=2048 -> FROZE PC", flush=True)
    print("-" * 40, flush=True)

    # ---- Load model ONCE ----
    print("\n[LOAD] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import bitsandbytes as bnb
    from hildanext.diffusion import compute_m2t_t2t_losses
    from hildanext.config import from_dict

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.train()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    mask_id = tok.get_vocab().get("[MASK]", 151669)
    vocab_size = model.config.vocab_size
    print(f"[LOAD] Model loaded. VRAM: {_vram_mb():.0f} MB", flush=True)

    train_cfg = from_dict({
        "train": {"mask_ratio": 0.15, "t2t_noise_ratio": 0.1, "m2t_weight": 1.0,
                  "t2t_weight": 1.0, "accum_steps": 1, "multi_turn_t2t": 1}
    }).train

    # ---- Preload dataset rows ----
    rows = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1:
                break
            rows.append(json.loads(line.strip()))
    full_ids = rows[0]["input_ids"]
    full_docs = rows[0]["doc_ids"]
    full_attn = rows[0]["attention_mask"]
    full_resp = rows[0].get("response_mask", [0] * 1024)

    # ---- Run each config ----
    all_results = {}

    for label, mask_mode, seq_len, block_size, bidirectional, n_steps in CONFIGS:
        _cleanup()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        eff_seq = seq_len * 2 if mask_mode == "composite_llada20" else seq_len
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}: mask={mask_mode} seq={seq_len} bs={block_size} "
              f"eff_seq={eff_seq} bidir={bidirectional}", flush=True)
        print(f"{'='*60}", flush=True)

        # Build batch at this seq_len
        ids = torch.tensor([full_ids[:seq_len]], dtype=torch.long, device=DEVICE)
        docs = torch.tensor([full_docs[:seq_len]], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([full_attn[:seq_len]], dtype=torch.long, device=DEVICE)
        resp = torch.tensor([full_resp[:seq_len]], dtype=torch.long, device=DEVICE)

        # Fresh optimizer per config
        opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=5e-5, weight_decay=0.1,
                                        betas=(0.9, 0.95))
        torch.cuda.reset_peak_memory_stats()

        bwd_times = []
        fwd_times = []
        losses = []
        nan_count = 0
        oom = False

        try:
            for step in range(n_steps):
                torch.manual_seed(SEED + step)
                opt.zero_grad(set_to_none=True)

                # Forward
                t_fwd = time.time()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = compute_m2t_t2t_losses(
                        model=model, input_ids=ids,
                        attention_mask=attn,
                        doc_ids=docs, response_mask=resp,
                        mask_id=mask_id, vocab_size=vocab_size,
                        cfg=train_cfg, focus_response=False,
                        mask_mode=mask_mode, composite_block_size=block_size,
                        bidirectional=bidirectional, time_param="continuous_time",
                        loss_weighting="inv_t", t_min=0.001, t_max=1.0,
                        target_ids=None
                    )
                torch.cuda.synchronize()
                fwd_time = time.time() - t_fwd
                fwd_times.append(fwd_time)

                loss_val = out["loss"].detach().item()
                losses.append(loss_val)
                if not math.isfinite(loss_val):
                    nan_count += 1

                # Backward
                t_bwd = time.time()
                out["loss"].backward()
                torch.cuda.synchronize()
                bwd_time = time.time() - t_bwd
                bwd_times.append(bwd_time)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                vram_now = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  step {step}: fwd={fwd_time:.2f}s bwd={bwd_time:.2f}s "
                      f"loss={loss_val:.4f} vram_peak={vram_now:.0f}MB", flush=True)

                del out
                # Safety: if bwd_time > 30s, skip remaining steps
                if bwd_time > 30.0:
                    print(f"  WARNING: bwd_time={bwd_time:.1f}s > 30s, skipping remaining steps",
                          flush=True)
                    break

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}", flush=True)
            oom = True
            _cleanup()
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            oom = True
            _cleanup()

        # Collect results
        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        result = {
            "mask_mode": mask_mode,
            "seq_len": seq_len,
            "block_size": block_size,
            "bidirectional": bidirectional,
            "eff_seq": eff_seq,
            "fwd_time_mean": float(np.mean(fwd_times)) if fwd_times else 0,
            "bwd_time_mean": float(np.mean(bwd_times)) if bwd_times else 0,
            "bwd_time_max": float(np.max(bwd_times)) if bwd_times else 0,
            "bwd_times": [round(t, 3) for t in bwd_times],
            "vram_peak_mb": round(vram_peak, 1),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_final": losses[-1] if losses else float("nan"),
            "nan_count": nan_count,
            "oom": oom,
            "tdr_safe": (max(bwd_times) < 60.0) if bwd_times else False,
        }
        all_results[label] = result

        # Clean optimizer state between configs
        del opt
        _cleanup()

    # ---- Summary ----
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY -- Safe Backward Benchmark", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Config':<25} {'eff_seq':>7} {'bwd_mean':>9} {'bwd_max':>8} "
          f"{'VRAM':>7} {'loss':>8} {'status':>10}", flush=True)
    print("-" * 80, flush=True)

    for lbl, r in all_results.items():
        if r.get("oom"):
            print(f"{lbl:<25} {r['eff_seq']:>7} {'--':>9} {'--':>8} "
                  f"{'--':>7} {'--':>8} {'OOM':>10}", flush=True)
        else:
            safe = "SAFE" if r["tdr_safe"] else "TDR!"
            print(f"{lbl:<25} {r['eff_seq']:>7} {r['bwd_time_mean']:>8.2f}s "
                  f"{r['bwd_time_max']:>7.2f}s {r['vram_peak_mb']:>6.0f}MB "
                  f"{r['loss_mean']:>8.4f} {safe:>10}", flush=True)

    # ---- Decision ----
    print("\n--- DECISION ---", flush=True)
    c512 = {k: v for k, v in all_results.items()
            if "512" in k and not v.get("oom", True)}
    if c512:
        worst_bwd = max(v["bwd_time_max"] for v in c512.values())
        worst_vram = max(v["vram_peak_mb"] for v in c512.values())
        all_safe = all(v.get("tdr_safe", False) for v in c512.values())
        print(f"  Composite at seq_len=512 (eff_seq=1024): "
              f"worst_bwd={worst_bwd:.2f}s worst_vram={worst_vram:.0f}MB "
              f"{'ALL SAFE' if all_safe else 'TDR RISK'}", flush=True)
        if all_safe and worst_vram < 7500:
            print("  -> Option B FEASIBLE: Warmup/Decay at seq_len=512, "
                  "Stable at seq_len=1024", flush=True)
        else:
            print("  -> Need further reduction or TDR fix + reboot", flush=True)
    else:
        print("  No composite_512 results -> run test first", flush=True)

    # ---- Save ----
    out_path = _root / "reports" / "backward_safe_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    run_all()

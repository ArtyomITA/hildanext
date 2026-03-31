#!/usr/bin/env python
"""vram_bench.py — CLI micro-test benchmark for GTX 1080 VRAM and stability.

Runs a short synthetic training loop on the real Qwen3-0.6B backbone,
logging per-step JSONL metrics: VRAM, loss, grad_norm, tok/s, NaN/Inf.

NOT a real training run — uses random input_ids, no real data pipeline.
Max 300 steps per run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG = logging.getLogger("vram_bench")

def _ts() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _make_run_id(args: argparse.Namespace) -> str:
    parts = [
        args.precision,
        args.optimizer,
        f"sl{args.seq_len}",
        f"ga{args.grad_acc}",
    ]
    if args.checkpoint != "none":
        parts.append(f"ckpt_{args.checkpoint}")
    if args.sdpa_backend != "MATH":
        parts.append(f"sdpa_{args.sdpa_backend}")
    if args.variable_seq:
        parts.append("varseq")
    return "__".join(parts)


def _set_seed(seed: int = 123) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading (real Qwen3-0.6B backbone)
# ---------------------------------------------------------------------------

def _load_model(model_dir: str, device: torch.device) -> Any:
    """Load Qwen3-0.6B with float32 weights — precision mode applied later."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # will cast later if needed
    )
    model = model.to(device)
    return model


def _get_vocab_size(model: Any) -> int:
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return int(model.config.vocab_size)
    if hasattr(model, "lm_head"):
        return int(model.lm_head.out_features)
    return 32000


# ---------------------------------------------------------------------------
# Precision / AMP helpers
# ---------------------------------------------------------------------------

def _apply_precision(model: Any, precision: str, device: torch.device):
    """Cast model weights; return (model, use_amp, use_scaler).

    On GTX 1080 (8 GB) a 0.6B model in fp32 (~2.4 GB) plus AdamW fp32
    momentum+variance (~4.8 GB) = ~7.2 GB BEFORE activations → OOM.
    Therefore all practical modes load the model in fp16.

    amp_fp16: model fp16, autocast enabled (protects loss/CE from overflow),
             GradScaler NOT used (fp16 params can't be unscaled).
             This is the working "mixed precision" on 8 GB.

    fp16_no_scaler: model fp16, no autocast, no scaler.
                    Raw fp16 — demonstrates instability.

    fp32: model fp32, no autocast — only for tiny seq_len smoke tests,
          will OOM with normal seq_len.
    """
    if precision == "fp32":
        return model, False, False
    if precision == "fp16_no_scaler":
        model = model.half()
        return model, False, False
    if precision == "amp_fp16":
        model = model.half()
        return model, True, False  # autocast yes, scaler no (fp16 params)
    raise ValueError(f"Unknown precision: {precision}")


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _make_optimizer(model: Any, name: str, lr: float = 3e-5):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if name == "bnb_adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    if name == "bnb_paged_adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(model.parameters(), lr=lr, weight_decay=0.01,
                         relative_step=False, scale_parameter=False)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# SDPA backend control
# ---------------------------------------------------------------------------

def _force_sdpa_backend(backend: str) -> None:
    """Force SDPA backend. On Pascal only MATH is safe."""
    if backend == "MATH":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "EFFICIENT":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "ALL":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        raise ValueError(f"Unknown SDPA backend: {backend}")


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def _apply_checkpointing(model: Any, mode: str) -> None:
    if mode == "none":
        return
    if mode == "full":
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            LOG.info("Gradient checkpointing enabled via HF API")
        else:
            LOG.warning("Model does not support gradient_checkpointing_enable()")
    # Disable KV-cache (incompatible with grad checkpointing)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


# ---------------------------------------------------------------------------
# Shifted cross-entropy (preserve left-shift like training.py)
# ---------------------------------------------------------------------------

def _shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits[:,:-1] predicts labels[:,1:] — position 0 never predicted."""
    if logits.shape[1] < 2:
        return torch.tensor(0.0, device=logits.device)
    l = logits[:, :-1, :].contiguous()
    y = labels[:, 1:].contiguous()
    return F.cross_entropy(l.view(-1, l.shape[-1]), y.view(-1))


# ---------------------------------------------------------------------------
# VRAM metrics
# ---------------------------------------------------------------------------

def _vram_metrics(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    idx = device.index or 0
    return {
        "cuda_mem_alloc_mb": round(torch.cuda.memory_allocated(idx) / 1024 / 1024, 2),
        "cuda_mem_res_mb": round(torch.cuda.memory_reserved(idx) / 1024 / 1024, 2),
        "cuda_peak_alloc_mb": round(torch.cuda.max_memory_allocated(idx) / 1024 / 1024, 2),
        "cuda_peak_res_mb": round(torch.cuda.max_memory_reserved(idx) / 1024 / 1024, 2),
    }


# ---------------------------------------------------------------------------
# Core training loop (synthetic)
# ---------------------------------------------------------------------------

def run_bench(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a micro-benchmark and return summary dict."""
    _set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("No CUDA device — aborting.")

    # Resolve model dir
    model_dir = args.model_dir
    if not model_dir:
        # Default: relative to this script's location
        model_dir = str(Path(__file__).resolve().parents[2] / "models" / "qwen3-0.6b")
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    run_id = _make_run_id(args)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else Path(__file__).resolve().parents[2] / "logs" / f"run_{run_id}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[vram_bench] run_id={run_id}")
    print(f"[vram_bench] model_dir={model_dir}")
    print(f"[vram_bench] precision={args.precision} optimizer={args.optimizer}")
    print(f"[vram_bench] seq_len={args.seq_len} micro_bs={args.micro_bs} grad_acc={args.grad_acc} steps={args.steps}")
    print(f"[vram_bench] checkpoint={args.checkpoint} sdpa_backend={args.sdpa_backend}")
    print(f"[vram_bench] variable_seq={args.variable_seq}")
    print(f"[vram_bench] out_jsonl={out_jsonl}")

    # --- Force SDPA ---
    _force_sdpa_backend(args.sdpa_backend)

    # --- Load model ---
    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    model = _load_model(model_dir, device)
    vocab_size = _get_vocab_size(model)
    print(f"[vram_bench] model loaded in {time.time() - t_load:.1f}s  vocab_size={vocab_size}")

    # --- Precision ---
    model, use_amp, use_scaler = _apply_precision(model, args.precision, device)
    print(f"[vram_bench] precision applied: use_amp={use_amp} use_scaler={use_scaler}")

    # --- Checkpointing ---
    _apply_checkpointing(model, args.checkpoint)

    # --- Optimizer ---
    optimizer = _make_optimizer(model, args.optimizer, lr=3e-5)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    print(f"[vram_bench] optimizer={args.optimizer} scaler={'yes' if scaler else 'no'}")

    model.train()

    # --- Variable seq_len support ---
    seq_lens = [args.seq_len]
    if args.variable_seq:
        lo = max(64, args.seq_len - 128)
        hi = args.seq_len + 128
        seq_lens = [lo, args.seq_len, hi]

    # --- Config snapshot for JSONL ---
    config_snap = {
        "precision": args.precision,
        "optimizer": args.optimizer,
        "seq_len": args.seq_len,
        "micro_bs": args.micro_bs,
        "grad_acc": args.grad_acc,
        "checkpoint": args.checkpoint,
        "sdpa_backend": args.sdpa_backend,
        "variable_seq": args.variable_seq,
    }

    fh = open(out_jsonl, "w", encoding="utf-8")
    total_nan_inf = 0
    oom = False
    steps_done = 0
    t_run_start = time.time()
    last_loss = float("nan")
    last_grad_norm = float("nan")

    try:
        for step in range(1, args.steps + 1):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            # Pick seq_len for this step
            cur_sl = seq_lens[step % len(seq_lens)] if args.variable_seq else args.seq_len

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_nan_inf = 0

            for acc_i in range(args.grad_acc):
                input_ids = torch.randint(0, vocab_size, (args.micro_bs, cur_sl), device=device)
                labels = input_ids.clone()

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(input_ids=input_ids).logits
                        loss = _shifted_cross_entropy(logits, labels) / args.grad_acc
                else:
                    logits = model(input_ids=input_ids).logits
                    loss = _shifted_cross_entropy(logits, labels) / args.grad_acc

                if not torch.isfinite(loss):
                    step_nan_inf += 1

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += float(loss.detach().item()) if torch.isfinite(loss) else 0.0

            # Grad norm & clip
            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))

            if not math.isfinite(grad_norm):
                step_nan_inf += 1

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            torch.cuda.synchronize()
            t1 = time.time()
            step_time = t1 - t0
            tok_per_s = (cur_sl * args.micro_bs * args.grad_acc) / max(1e-6, step_time)

            total_nan_inf += step_nan_inf
            last_loss = accum_loss * args.grad_acc  # un-normalized
            last_grad_norm = grad_norm

            scaler_scale = float(scaler.get_scale()) if scaler is not None else None

            vm = _vram_metrics(device)

            row = {
                "ts": _ts(),
                "run_id": run_id,
                "step": step,
                "phase": "train",
                "config": config_snap,
                "metrics": {
                    "loss": round(last_loss, 6),
                    "grad_norm": round(grad_norm, 4),
                    "scaler_scale": scaler_scale,
                    "time_step_s": round(step_time, 4),
                    "tok_per_s": round(tok_per_s, 1),
                    "seq_len_actual": cur_sl,
                    **vm,
                    "oom": False,
                    "nan_inf": step_nan_inf,
                },
            }
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            fh.flush()

            if step == 1 or step % 5 == 0 or step == args.steps:
                print(
                    f"  step={step}/{args.steps}  loss={last_loss:.4f}  gn={grad_norm:.3f}"
                    f"  scaler={scaler_scale}  tok/s={tok_per_s:.0f}"
                    f"  alloc={vm.get('cuda_mem_alloc_mb',0):.0f}MB"
                    f"  peak={vm.get('cuda_peak_alloc_mb',0):.0f}MB"
                    f"  nan_inf={step_nan_inf}",
                    flush=True,
                )

            steps_done = step

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
            print(f"[vram_bench] OOM at step {steps_done + 1}: {e}", flush=True)
            # Log OOM row
            vm = _vram_metrics(device)
            row = {
                "ts": _ts(),
                "run_id": run_id,
                "step": steps_done + 1,
                "phase": "oom",
                "config": config_snap,
                "metrics": {**vm, "oom": True, "nan_inf": 0},
            }
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        else:
            raise
    finally:
        fh.close()
        # Explicit cleanup for sweep_seq reuse
        del optimizer
        if scaler is not None:
            del scaler
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - t_run_start
    summary = {
        "run_id": run_id,
        "steps_done": steps_done,
        "steps_requested": args.steps,
        "oom": oom,
        "total_nan_inf": total_nan_inf,
        "last_loss": round(last_loss, 6) if math.isfinite(last_loss) else None,
        "last_grad_norm": round(last_grad_norm, 4) if math.isfinite(last_grad_norm) else None,
        "total_time_s": round(total_time, 2),
        "out_jsonl": str(out_jsonl),
        "config": config_snap,
    }

    # Write summary JSON alongside the JSONL
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[vram_bench] DONE  steps={steps_done}  oom={oom}  nan_inf={total_nan_inf}  time={total_time:.1f}s")
    print(f"[vram_bench] summary -> {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VRAM micro-benchmark for GTX 1080")
    parser.add_argument("--precision", choices=["fp32", "fp16_no_scaler", "amp_fp16"], default="amp_fp16")
    parser.add_argument("--optimizer", choices=["adamw", "bnb_adamw8bit", "bnb_paged_adamw8bit", "adafactor", "sgd"], default="adamw")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--micro_bs", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", choices=["none", "full"], default="none")
    parser.add_argument("--sdpa_backend", choices=["MATH", "EFFICIENT", "ALL"], default="MATH")
    parser.add_argument("--variable_seq", action="store_true", default=False)
    parser.add_argument("--leak_check", action="store_true", default=False)
    parser.add_argument("--out_jsonl", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--mode", choices=["train"], default="train")
    args = parser.parse_args()

    if args.steps > 300:
        print("[vram_bench] WARNING: capping steps to 300 (micro-test only)")
        args.steps = 300

    run_bench(args)


if __name__ == "__main__":
    main()

"""VRAM Matrix Benchmark.

Tests every combination of (optimizer, grad_ckpt, seq_len) and reports:
  - Peak VRAM (alloc + reserved)
  - Throughput (tok/s)
  - Step time
  - OOM yes/no
  - NaN count

Usage:
    python -m test.overhead_bench.bench_vram_matrix [--opt-steps N] [--config PATH]
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import torch

from test.overhead_bench._common import (
    RESULTS_DIR, load_cfg, load_bundle, make_loader, make_optimizer,
    reset_vram, vram_stats, append_jsonl, write_json, gpu_info,
    forward_backward, fix_stdout_encoding,
)


# ── Test matrix ──────────────────────────────────────────────────────
MATRIX = [
    # label, optimizer, grad_ckpt, seq_len
    ("A_adamw8bit_ckpt_1024",     "AdamW8bit",       True,  1024),
    ("B_adamw8bit_nockpt_1024",   "AdamW8bit",       False, 1024),
    ("C_adamw8bit_ckpt_2048",     "AdamW8bit",       True,  2048),
    ("D_adamwfp32_ckpt_1024",     "AdamW_fused",     True,  1024),
    ("E_paged8bit_ckpt_1024",     "PagedAdamW8bit",  True,  1024),
    ("F_paged8bit_nockpt_1024",   "PagedAdamW8bit",  False, 1024),
    ("G_adamw8bit_ckpt_1536",     "AdamW8bit",       True,  1536),
    ("H_paged8bit_ckpt_2048",     "PagedAdamW8bit",  True,  2048),
]


def _toggle_grad_ckpt(model, enable: bool):
    """Enable or disable gradient checkpointing on the model."""
    if enable:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False, "preserve_rng_state": False
                    }
                )
            except TypeError:
                model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


def _run_single_config(
    label: str,
    opt_name: str,
    grad_ckpt: bool,
    seq_len: int,
    model,
    bundle,
    cfg,
    loader_iter,
    n_opt_steps: int,
    grad_acc: int,
) -> dict:
    """Run one config for n_opt_steps optimizer steps. Returns result dict."""
    device = bundle.device
    use_amp = device.type == "cuda"
    lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)
    ct_t_min = float(cfg.experiment.t_min)
    ct_t_max = float(cfg.experiment.t_max)

    # Toggle grad checkpointing
    _toggle_grad_ckpt(model, grad_ckpt)
    model.train()

    # Create optimizer
    try:
        opt, opt_label = make_optimizer(opt_name, model, lr, wd)
    except Exception as e:
        return {"label": label, "status": "opt_unavailable", "error": str(e)[:200]}

    # Reset VRAM stats after optimizer creation
    reset_vram()
    vram_before = vram_stats()

    # Install embed noise for warmup stability
    from hildanext.diffusion import (
        _install_embed_noise_hook, _remove_embed_noise_hook,
        set_embed_noise_std,
    )
    _install_embed_noise_hook(model, bundle.mask_id, noise_std=0.1)

    # Warmup: 2 micro-batches (not timed)
    for _ in range(2):
        try:
            batch = next(loader_iter)
        except StopIteration:
            return {"label": label, "status": "data_exhausted"}
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        try:
            forward_backward(model, batch, bundle, cfg, use_amp,
                             ct_t_min, ct_t_max, grad_acc=grad_acc)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                opt.zero_grad(set_to_none=True)
                del opt
                _remove_embed_noise_hook()
                reset_vram()
                return {"label": label, "status": "oom_warmup",
                        "error": str(e)[:200]}
            raise
        opt.zero_grad(set_to_none=True)

    # Reset peak after warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    fwd_times, bwd_times, opt_times = [], [], []
    losses, grad_norms = [], []
    nan_count = 0
    total_tokens = 0
    oom = False

    t_start = time.perf_counter()
    micro_step = 0

    for opt_step_i in range(n_opt_steps):
        for acc_i in range(grad_acc):
            micro_step += 1
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            total_tokens += int(batch["attention_mask"].sum().item())

            try:
                loss_val, out, fwd_s, bwd_s = forward_backward(
                    model, batch, bundle, cfg, use_amp,
                    ct_t_min, ct_t_max, grad_acc=grad_acc,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom = True
                    break
                raise

            fwd_times.append(fwd_s)
            bwd_times.append(bwd_s)
            losses.append(loss_val)
            if not math.isfinite(loss_val):
                nan_count += 1
            del out

        if oom:
            break

        # Optimizer step
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        grad_norms.append(gn)

        if math.isfinite(gn) and gn < 10000:
            t_opt0 = time.perf_counter()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_opt1 = time.perf_counter()
            opt_times.append(t_opt1 - t_opt0)
        else:
            opt_times.append(0.0)

        opt.zero_grad(set_to_none=True)

        # Decay embed noise
        frac = max(0.0, 1.0 - float(opt_step_i + 1) / float(n_opt_steps))
        set_embed_noise_std(0.1 * frac)

    t_end = time.perf_counter()
    wall = t_end - t_start

    peak = vram_stats()
    _remove_embed_noise_hook()

    # Cleanup optimizer
    opt.zero_grad(set_to_none=True)
    del opt
    reset_vram()

    return {
        "label": label,
        "status": "oom" if oom else "ok",
        "optimizer": opt_name,
        "grad_ckpt": grad_ckpt,
        "seq_len": seq_len,
        "opt_steps_done": len(grad_norms),
        "micro_batches": micro_step,
        "wall_s": round(wall, 2),
        "total_tokens": total_tokens,
        "tok_per_s": round(total_tokens / max(0.001, wall), 1),
        "avg_fwd_s": round(sum(fwd_times) / max(1, len(fwd_times)), 4),
        "avg_bwd_s": round(sum(bwd_times) / max(1, len(bwd_times)), 4),
        "avg_opt_step_s": round(sum(opt_times) / max(1, len(opt_times)), 4),
        "peak_alloc_mb": peak.get("peak_alloc_mb", 0),
        "peak_reserved_mb": peak.get("reserved_mb", 0),
        "vram_current_mb": peak.get("alloc_mb", 0),
        "nan_count": nan_count,
        "loss_mean": round(sum(l for l in losses if math.isfinite(l)) /
                          max(1, sum(1 for l in losses if math.isfinite(l))), 4),
        "loss_last": round(losses[-1], 4) if losses else 0,
        "grad_norm_mean": round(sum(g for g in grad_norms if math.isfinite(g)) /
                               max(1, sum(1 for g in grad_norms if math.isfinite(g))), 2),
        "grad_norm_max": round(max((g for g in grad_norms if math.isfinite(g)),
                                   default=0), 2),
    }


def _est_4000_steps(result: dict) -> str:
    """Estimate time for 4000 optimizer steps from measured step time."""
    if result["status"] != "ok" or result["opt_steps_done"] == 0:
        return "N/A"
    step_s = result["wall_s"] / result["opt_steps_done"]
    total_s = step_s * 4000
    h = total_s / 3600
    return f"{h:.1f}h ({step_s:.2f}s/step)"


def main():
    fix_stdout_encoding()
    ap = argparse.ArgumentParser(description="VRAM Matrix Benchmark")
    ap.add_argument("--opt-steps", type=int, default=5,
                    help="Optimizer steps per config (default 5)")
    ap.add_argument("--grad-acc", type=int, default=8,
                    help="Gradient accumulation steps (default 8)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated labels to run (empty=all)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — aborting", flush=True)
        return

    info = gpu_info()
    print(f"{'=' * 72}", flush=True)
    print(f"VRAM MATRIX BENCHMARK", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"GPU: {info['gpu']}  VRAM: {info.get('vram_gb', '?')} GB  "
          f"PyTorch: {info.get('torch', '?')}  Temp: {info.get('temp_c', '?')}°C",
          flush=True)
    print(f"opt_steps={args.opt_steps}  grad_acc={args.grad_acc}", flush=True)

    only = set(x.strip() for x in args.only.split(",") if x.strip()) if args.only else set()

    # Load base config (seq_len=1024)
    cfg = load_cfg(args.config)

    # Load model WITH grad_ckpt (we toggle per-test)
    print(f"\n--- Loading model ---", flush=True)
    bundle = load_bundle(cfg, for_training=True)
    model = bundle.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  dtype={bundle.actual_dtype}", flush=True)
    print(f"VRAM model-only: {vram_stats().get('alloc_mb', 0):.0f} MB", flush=True)

    # Create loader (uses current cfg.data.seq_len=1024; for 2048 tests we
    # handle it below by creating a temporary config)
    print(f"\n--- Loading dataset ---", flush=True)
    loader = make_loader(cfg)
    print(f"Dataset: {len(loader.dataset)} rows  seq_len={cfg.data.seq_len}", flush=True)

    # For seq_len != 1024, we need different data. We'll handle that inside the loop
    # by patching the batch to the right length (truncate or pad).
    # This is simpler than re-tokenizing for a quick benchmark.

    results = []
    jsonl_path = RESULTS_DIR / "vram_matrix.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    print(f"\n{'=' * 72}", flush=True)
    print(f"RUNNING {len(MATRIX)} CONFIGURATIONS", flush=True)
    print(f"{'=' * 72}\n", flush=True)

    for label, opt_name, grad_ckpt, seq_len in MATRIX:
        if only and label not in only:
            continue

        print(f"--- [{label}] optimizer={opt_name} grad_ckpt={grad_ckpt} "
              f"seq_len={seq_len} ---", flush=True)

        # If seq_len differs from shard, use a wrapper that adjusts batch size
        test_cfg = cfg
        if seq_len != cfg.data.seq_len:
            from hildanext.config import clone_with_updates
            test_cfg = clone_with_updates(cfg, {
                "data": {"seq_len": seq_len},
                "wsd": {"max_block_size": seq_len,
                        "ladder_blocks": [b for b in cfg.wsd.ladder_blocks if b <= seq_len] + [seq_len]},
            })

        # Create a fresh loader iterator (enough for warmup + n_opt_steps * grad_acc)
        total_needed = 2 + args.opt_steps * args.grad_acc + 5
        loader_iter = iter(make_loader(cfg))  # always use base seq_len from shards

        # Wrap iterator to handle seq_len mismatch by padding/truncating
        def _adapt_iter(it, target_seq):
            """Yield batches adjusted to target_seq from shards of different width."""
            shard_seq = cfg.data.seq_len
            for batch in it:
                if target_seq == shard_seq:
                    yield batch
                elif target_seq < shard_seq:
                    yield {k: v[:, :target_seq] if v.dim() > 1 else v
                           for k, v in batch.items()}
                else:
                    # Pad by repeating (simple, just for VRAM measurement)
                    reps = (target_seq + shard_seq - 1) // shard_seq
                    padded = {}
                    for k, v in batch.items():
                        if v.dim() > 1:
                            tiled = v.repeat(1, reps)[:, :target_seq]
                            padded[k] = tiled
                        else:
                            padded[k] = v
                    yield padded

        adapted = _adapt_iter(loader_iter, seq_len)

        from hildanext.utils import seed_everything
        seed_everything(42)

        try:
            r = _run_single_config(
                label=label,
                opt_name=opt_name,
                grad_ckpt=grad_ckpt,
                seq_len=seq_len,
                model=model,
                bundle=bundle,
                cfg=test_cfg,
                loader_iter=adapted,
                n_opt_steps=args.opt_steps,
                grad_acc=args.grad_acc,
            )
        except Exception as e:
            r = {"label": label, "status": "error", "error": str(e)[:300]}
            traceback.print_exc()

        results.append(r)
        append_jsonl(jsonl_path, [r])

        # Restore grad_ckpt to default after each test
        _toggle_grad_ckpt(model, True)
        reset_vram()

        # Print inline result
        status = r.get("status", "?")
        if status == "ok":
            print(f"  [OK] peak={r['peak_alloc_mb']:.0f}MB  "
                  f"tok/s={r['tok_per_s']}  "
                  f"NaN={r['nan_count']}  "
                  f"loss={r['loss_last']:.4f}  "
                  f"4000step~{_est_4000_steps(r)}",
                  flush=True)
        else:
            print(f"  [FAIL] status={status}  {r.get('error', '')[:100]}", flush=True)
        print(flush=True)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 100}", flush=True)
    print(f"{'VRAM MATRIX — SUMMARY':^100}", flush=True)
    print(f"{'=' * 100}", flush=True)
    hdr = f"{'Label':<32} {'Status':<10} {'PeakMB':>8} {'tok/s':>8} {'NaN':>5} {'LossLast':>10} {'4000 steps':>16}"
    print(hdr, flush=True)
    print("-" * 100, flush=True)
    for r in results:
        s = r.get("status", "?")
        if s == "ok":
            line = (f"{r['label']:<32} {'OK':<10} "
                    f"{r['peak_alloc_mb']:>8.0f} "
                    f"{r['tok_per_s']:>8.1f} "
                    f"{r['nan_count']:>5} "
                    f"{r['loss_last']:>10.4f} "
                    f"{_est_4000_steps(r):>16}")
        else:
            line = f"{r.get('label', '?'):<32} {s.upper():<10} {'—':>8} {'—':>8} {'—':>5} {'—':>10} {'—':>16}"
        print(line, flush=True)
    print("-" * 100, flush=True)

    # Write summary
    summary = {"gpu": info, "results": results}
    write_json(RESULTS_DIR / "vram_matrix.summary.json", summary)
    print(f"\nResults saved to {RESULTS_DIR / 'vram_matrix.summary.json'}", flush=True)


# Bring into scope for non-module usage
from test.overhead_bench._common import vram_stats, reset_vram  # noqa: re-export


if __name__ == "__main__":
    main()

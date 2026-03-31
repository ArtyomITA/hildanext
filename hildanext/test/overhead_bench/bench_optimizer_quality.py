"""Optimizer Quality Benchmark.

Runs N optimizer steps with each viable optimizer and measures:
  - Loss convergence (mean, trend)
  - NaN/Inf count and rate
  - Gradient norm statistics
  - Masked token accuracy trajectory
  - Effect of embed noise hook
  - AdamW fp32 NaN investigation (with/without GradScaler)

Usage:
    python -m test.overhead_bench.bench_optimizer_quality [--opt-steps N] [--config PATH]
"""
from __future__ import annotations
import argparse, gc, math, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import torch

from test.overhead_bench._common import (
    RESULTS_DIR, load_cfg, load_bundle, make_loader, make_optimizer,
    reset_vram, vram_stats, append_jsonl, write_json, gpu_info,
    forward_backward, fix_stdout_encoding,
)


# ── Configs to test ──────────────────────────────────────────────────
CONFIGS = [
    # label, optimizer, embed_noise, use_grad_scaler
    ("adamw8bit",               "AdamW8bit",       True,  False),
    ("adamw8bit_no_noise",      "AdamW8bit",       False, False),
    ("paged8bit",               "PagedAdamW8bit",  True,  False),
    ("paged8bit_no_noise",      "PagedAdamW8bit",  False, False),
    ("adamw_fp32",              "AdamW_fused",     True,  False),
    ("adamw_fp32_gradscaler",   "AdamW_fused",     True,  True),
    ("adafactor",               "Adafactor",       True,  False),
]


def _run_quality_test(
    label: str,
    opt_name: str,
    embed_noise: bool,
    use_grad_scaler: bool,
    model,
    bundle,
    cfg,
    loader_iter,
    n_opt_steps: int,
    grad_acc: int,
) -> dict:
    device = bundle.device
    use_amp = device.type == "cuda"
    lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)
    ct_t_min = float(cfg.experiment.t_min)
    ct_t_max = float(cfg.experiment.t_max)

    # Ensure grad_ckpt is ON (needed for fp32 to fit)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False, "preserve_rng_state": False
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model.train()

    try:
        opt, _ = make_optimizer(opt_name, model, lr, wd)
    except Exception as e:
        return {"label": label, "status": "opt_unavailable", "error": str(e)[:200]}

    scaler = None
    if use_grad_scaler and torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda")

    if embed_noise:
        from hildanext.diffusion import (
            _install_embed_noise_hook, _remove_embed_noise_hook,
            set_embed_noise_std,
        )
        _install_embed_noise_hook(model, bundle.mask_id, noise_std=0.1)

    reset_vram()

    # Warmup 2 micro-batches
    for _ in range(2):
        try:
            batch = next(loader_iter)
        except StopIteration:
            return {"label": label, "status": "data_exhausted"}
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        vocab_cap = max(8, bundle.vocab_size)
        batch["input_ids"] = torch.remainder(batch["input_ids"], vocab_cap)

        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                from hildanext.diffusion import compute_m2t_t2t_losses
                out = compute_m2t_t2t_losses(
                    model=model, input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    doc_ids=batch["doc_ids"], response_mask=batch["response_mask"],
                    mask_id=bundle.mask_id, vocab_size=vocab_cap,
                    cfg=cfg.train, focus_response=False,
                    mask_mode="simple_blockdiag",
                    composite_block_size=cfg.llada2.composite_block_size,
                    trace=None, cfg_obj=cfg, bidirectional=False,
                    time_param="continuous_time", loss_weighting="inv_t",
                    t_min=ct_t_min, t_max=ct_t_max,
                )
            scaler.scale(out["loss"]).backward()
            scaler.step(opt)
            scaler.update()
        else:
            try:
                forward_backward(model, batch, bundle, cfg, use_amp,
                                 ct_t_min, ct_t_max, grad_acc=1)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if embed_noise:
                        _remove_embed_noise_hook()
                    del opt
                    reset_vram()
                    return {"label": label, "status": "oom", "error": str(e)[:200]}
                raise
        opt.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed quality run
    step_records = []
    t_start = time.perf_counter()

    for opt_step_i in range(n_opt_steps):
        step_losses = []
        step_nan = 0
        step_accs = []

        for acc_i in range(grad_acc):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            try:
                if scaler:
                    vocab_cap = max(8, bundle.vocab_size)
                    batch["input_ids"] = torch.remainder(batch["input_ids"], vocab_cap)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        from hildanext.diffusion import compute_m2t_t2t_losses
                        out = compute_m2t_t2t_losses(
                            model=model, input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            doc_ids=batch["doc_ids"],
                            response_mask=batch["response_mask"],
                            mask_id=bundle.mask_id, vocab_size=vocab_cap,
                            cfg=cfg.train, focus_response=False,
                            mask_mode="simple_blockdiag",
                            composite_block_size=cfg.llada2.composite_block_size,
                            trace=None, cfg_obj=cfg, bidirectional=False,
                            time_param="continuous_time", loss_weighting="inv_t",
                            t_min=ct_t_min, t_max=ct_t_max,
                        )
                    loss_val = float(out["loss"].detach().item())
                    scaler.scale(out["loss"] / float(grad_acc)).backward()
                else:
                    loss_val, out, _, _ = forward_backward(
                        model, batch, bundle, cfg, use_amp,
                        ct_t_min, ct_t_max, grad_acc=grad_acc,
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    step_records.append({
                        "step": opt_step_i + 1, "status": "oom"
                    })
                    if embed_noise:
                        _remove_embed_noise_hook()
                    del opt
                    if scaler:
                        del scaler
                    reset_vram()
                    return {
                        "label": label, "status": "oom",
                        "steps_done": opt_step_i,
                        "step_records": step_records,
                    }
                raise

            step_losses.append(loss_val)
            if not math.isfinite(loss_val):
                step_nan += 1

            acc_v = out.get("masked_token_acc")
            if acc_v is not None:
                step_accs.append(float(acc_v))
            del out

        # Optimizer step
        if scaler:
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if math.isfinite(gn) and gn < 10000:
                scaler.step(opt)
            scaler.update()
        else:
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if math.isfinite(gn) and gn < 10000:
                opt.step()

        opt.zero_grad(set_to_none=True)

        # Decay embed noise
        if embed_noise:
            frac = max(0.0, 1.0 - float(opt_step_i + 1) / float(n_opt_steps))
            set_embed_noise_std(0.1 * frac)

        finite_losses = [l for l in step_losses if math.isfinite(l)]
        step_records.append({
            "step": opt_step_i + 1,
            "loss_mean": round(sum(finite_losses) / max(1, len(finite_losses)), 4)
                         if finite_losses else None,
            "loss_min": round(min(finite_losses), 4) if finite_losses else None,
            "nan_count": step_nan,
            "grad_norm": round(gn, 2) if math.isfinite(gn) else None,
            "mta_mean": round(sum(step_accs) / max(1, len(step_accs)), 4)
                        if step_accs else None,
        })

    t_end = time.perf_counter()
    wall = t_end - t_start

    if embed_noise:
        _remove_embed_noise_hook()

    peak = vram_stats()
    opt.zero_grad(set_to_none=True)
    del opt
    if scaler:
        del scaler
    reset_vram()

    # Aggregate
    all_losses = [r["loss_mean"] for r in step_records
                  if r.get("loss_mean") is not None]
    all_gn = [r["grad_norm"] for r in step_records
              if r.get("grad_norm") is not None]
    all_nan = sum(r.get("nan_count", 0) for r in step_records)
    all_mta = [r["mta_mean"] for r in step_records
               if r.get("mta_mean") is not None]

    # Loss trend (first half vs second half)
    half = max(1, len(all_losses) // 2)
    loss_first = sum(all_losses[:half]) / max(1, half) if all_losses else 0
    loss_second = sum(all_losses[half:]) / max(1, len(all_losses) - half) if all_losses else 0

    return {
        "label": label,
        "status": "ok",
        "optimizer": opt_name,
        "embed_noise": embed_noise,
        "grad_scaler": use_grad_scaler,
        "opt_steps_done": len(step_records),
        "wall_s": round(wall, 2),
        "peak_alloc_mb": peak.get("peak_alloc_mb", 0),
        "total_nan": all_nan,
        "nan_rate": round(all_nan / max(1, len(step_records) * grad_acc), 4),
        "loss_first_half": round(loss_first, 4),
        "loss_second_half": round(loss_second, 4),
        "loss_improving": loss_second < loss_first if all_losses else False,
        "loss_last": round(all_losses[-1], 4) if all_losses else None,
        "grad_norm_mean": round(sum(all_gn) / max(1, len(all_gn)), 2) if all_gn else None,
        "grad_norm_max": round(max(all_gn), 2) if all_gn else None,
        "mta_last": round(all_mta[-1], 4) if all_mta else None,
        "step_records": step_records,
    }


def main():
    fix_stdout_encoding()
    ap = argparse.ArgumentParser(description="Optimizer Quality Benchmark")
    ap.add_argument("--opt-steps", type=int, default=30,
                    help="Optimizer steps per config (default 30)")
    ap.add_argument("--grad-acc", type=int, default=8,
                    help="Gradient accumulation (default 8)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated labels to test")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — aborting", flush=True)
        return

    info = gpu_info()
    print(f"{'=' * 72}", flush=True)
    print(f"OPTIMIZER QUALITY BENCHMARK", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"GPU: {info['gpu']}  opt_steps={args.opt_steps}  grad_acc={args.grad_acc}",
          flush=True)

    only = set(x.strip() for x in args.only.split(",") if x.strip()) if args.only else set()

    cfg = load_cfg(args.config)

    print(f"\n--- Loading model ---", flush=True)
    bundle = load_bundle(cfg, for_training=True)
    model = bundle.model
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    results = []
    jsonl_path = RESULTS_DIR / "optimizer_quality.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    for label, opt_name, embed_noise, use_scaler in CONFIGS:
        if only and label not in only:
            continue

        print(f"\n--- [{label}] opt={opt_name} noise={embed_noise} "
              f"scaler={use_scaler} ---", flush=True)

        from hildanext.utils import seed_everything
        seed_everything(42)
        loader = make_loader(cfg)
        loader_iter = iter(loader)

        try:
            r = _run_quality_test(
                label=label, opt_name=opt_name,
                embed_noise=embed_noise, use_grad_scaler=use_scaler,
                model=model, bundle=bundle, cfg=cfg,
                loader_iter=loader_iter,
                n_opt_steps=args.opt_steps, grad_acc=args.grad_acc,
            )
        except Exception as e:
            r = {"label": label, "status": "error", "error": str(e)[:300]}
            traceback.print_exc()

        results.append(r)
        # Write step records separately
        steps_only = r.pop("step_records", [])
        append_jsonl(jsonl_path, [r])
        if steps_only:
            steps_path = RESULTS_DIR / f"quality_{label}_steps.jsonl"
            if steps_path.exists():
                steps_path.unlink()
            append_jsonl(steps_path, steps_only)

        s = r.get("status", "?")
        if s == "ok":
            improving = "DOWN" if r.get("loss_improving") else "FLAT"
            print(f"  [OK] loss={r['loss_last']}  NaN={r['total_nan']}  "
                  f"trend={improving}  gn_max={r['grad_norm_max']}  "
                  f"mta={r['mta_last']}  peak={r['peak_alloc_mb']:.0f}MB",
                  flush=True)
        else:
            print(f"  [FAIL] status={s}  {r.get('error', '')[:100]}", flush=True)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 110}", flush=True)
    print(f"{'OPTIMIZER QUALITY — SUMMARY':^110}", flush=True)
    print(f"{'=' * 110}", flush=True)
    hdr = (f"{'Label':<28} {'Status':<8} {'NaN':>5} {'NaN%':>6} "
           f"{'Loss1H':>8} {'Loss2H':>8} {'Trend':>6} {'GNmax':>8} "
           f"{'MTA':>6} {'PeakMB':>8}")
    print(hdr, flush=True)
    print("-" * 110, flush=True)
    for r in results:
        s = r.get("status", "?")
        if s == "ok":
            trend = "DOWN" if r.get("loss_improving") else "FLAT"
            line = (f"{r['label']:<28} {'OK':<8} "
                    f"{r['total_nan']:>5} "
                    f"{r['nan_rate']:>6.2%} "
                    f"{r['loss_first_half']:>8.2f} "
                    f"{r['loss_second_half']:>8.2f} "
                    f"{trend:>6} "
                    f"{r['grad_norm_max'] or 0:>8.1f} "
                    f"{r['mta_last'] or 0:>6.4f} "
                    f"{r['peak_alloc_mb']:>8.0f}")
        else:
            line = (f"{r.get('label', '?'):<28} {s.upper():<8} "
                    f"{'—':>5} {'—':>6} {'—':>8} {'—':>8} {'—':>6} "
                    f"{'—':>8} {'—':>6} {'—':>8}")
        print(line, flush=True)
    print("-" * 110, flush=True)

    summary = {"gpu": info, "results": results}
    write_json(RESULTS_DIR / "optimizer_quality.summary.json", summary)
    print(f"\nResults: {RESULTS_DIR / 'optimizer_quality.summary.json'}", flush=True)


if __name__ == "__main__":
    main()

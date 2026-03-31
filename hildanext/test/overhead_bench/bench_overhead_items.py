"""Overhead Micro-Benchmarks.

Measures the cost of individual overhead items in the hot path:
  1. torch.isfinite(loss).item()  — GPU sync cost
  2. .item() on 6 metric tensors  — GPU sync cost (batched)
  3. nvidia-smi subprocess vs pynvml  — temperature query
  4. torch.cuda.empty_cache()  — allocator flush
  5. wsd_block() call  — pure Python cost
  6. torch.remainder(input_ids, vocab_cap)  — GPU kernel

Usage:
    python -m test.overhead_bench.bench_overhead_items [--iters N] [--config PATH]
"""
from __future__ import annotations
import argparse, os, sys, statistics, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import torch

from test.overhead_bench._common import (
    RESULTS_DIR, load_cfg, load_bundle, make_loader, write_json, gpu_info,
    _gpu_temp, _gpu_temp_pynvml, fix_stdout_encoding,
)


def _bench_isfinite_item(loss_tensor: torch.Tensor, n: int) -> list:
    """Benchmark torch.isfinite(t).item() which forces GPU sync."""
    times = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = bool(torch.isfinite(loss_tensor).item())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return times


def _bench_item_calls(tensors: list, n: int) -> list:
    """Benchmark calling .item() on multiple GPU tensors."""
    times = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in tensors:
            _ = float(t.detach().item())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_deferred_item(tensors: list, n: int) -> list:
    """Benchmark .item() on CPU copies (deferred: move to CPU first, then read)."""
    times = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_vals = [t.detach().cpu() for t in tensors]
        vals = [float(c) for c in cpu_vals]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_nvidia_smi(n: int) -> list:
    """Benchmark nvidia-smi subprocess call."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = _gpu_temp()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_pynvml(n: int) -> list:
    """Benchmark pynvml temperature query."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = _gpu_temp_pynvml()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_empty_cache(n: int) -> list:
    """Benchmark torch.cuda.empty_cache()."""
    if not torch.cuda.is_available():
        return []
    times = []
    for _ in range(n):
        # Allocate + free some memory to make empty_cache do work
        tmp = torch.randn(1024, 1024, device="cuda")
        del tmp
        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_wsd_block(n: int, cfg) -> list:
    """Benchmark wsd_block() call (pure Python)."""
    from hildanext.formulas import llada2_wsd_block
    times = []
    for i in range(n):
        step = i % 500
        t0 = time.perf_counter()
        _ = llada2_wsd_block(
            step,
            warmup_steps=cfg.wsd.warmup_steps,
            stable_steps=cfg.wsd.stable_steps,
            decay_steps=cfg.wsd.decay_steps,
            start_block=cfg.wsd.start_block_size,
            max_block=cfg.wsd.max_block_size,
            end_block=cfg.wsd.end_block_size,
            ladder_blocks=cfg.wsd.ladder_blocks,
            decay_blocks=cfg.wsd.decay_blocks,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _bench_remainder(n: int, seq_len: int = 1024) -> list:
    """Benchmark torch.remainder on GPU tensor."""
    if not torch.cuda.is_available():
        return []
    ids = torch.randint(0, 160000, (1, seq_len), device="cuda", dtype=torch.long)
    vocab_cap = 151936
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.remainder(ids, vocab_cap)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def _stats(times: list) -> dict:
    if not times:
        return {"n": 0, "mean_us": 0, "median_us": 0, "p95_us": 0, "min_us": 0, "max_us": 0}
    times_sorted = sorted(times)
    p95_idx = min(len(times_sorted) - 1, int(len(times_sorted) * 0.95))
    return {
        "n": len(times),
        "mean_us": round(statistics.mean(times), 1),
        "median_us": round(statistics.median(times), 1),
        "p95_us": round(times_sorted[p95_idx], 1),
        "min_us": round(min(times), 1),
        "max_us": round(max(times), 1),
    }


def main():
    fix_stdout_encoding()
    ap = argparse.ArgumentParser(description="Overhead Micro-Benchmarks")
    ap.add_argument("--iters", type=int, default=100,
                    help="Iterations per benchmark (default 100)")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    N = args.iters

    info = gpu_info()
    print(f"{'=' * 72}", flush=True)
    print(f"OVERHEAD MICRO-BENCHMARKS", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"GPU: {info.get('gpu', 'N/A')}  iters={N}", flush=True)

    cfg = load_cfg(args.config)

    results = {}

    # ── 1. isfinite().item() ─────────────────────────────────────────
    print(f"\n1. torch.isfinite(loss).item() — GPU sync cost", flush=True)
    if torch.cuda.is_available():
        loss_t = torch.tensor(42.0, device="cuda", requires_grad=False)
        times = _bench_isfinite_item(loss_t, N)
        s = _stats(times)
        results["isfinite_item"] = s
        print(f"   mean={s['mean_us']:.1f}μs  median={s['median_us']:.1f}μs  "
              f"p95={s['p95_us']:.1f}μs", flush=True)
    else:
        results["isfinite_item"] = {"status": "no_cuda"}
        print(f"   SKIPPED (no CUDA)", flush=True)

    # ── 2. .item() on 6 metric tensors ──────────────────────────────
    print(f"\n2. .item() × 6 tensors — per-micro-batch metric extraction", flush=True)
    if torch.cuda.is_available():
        metric_ts = [torch.tensor(float(i), device="cuda") for i in range(6)]
        times_direct = _bench_item_calls(metric_ts, N)
        times_deferred = _bench_deferred_item(metric_ts, N)
        s_direct = _stats(times_direct)
        s_deferred = _stats(times_deferred)
        results["item_6_direct"] = s_direct
        results["item_6_deferred_cpu"] = s_deferred
        print(f"   direct:   mean={s_direct['mean_us']:.1f}μs  "
              f"median={s_direct['median_us']:.1f}μs", flush=True)
        print(f"   deferred: mean={s_deferred['mean_us']:.1f}μs  "
              f"median={s_deferred['median_us']:.1f}μs", flush=True)
        savings = s_direct['mean_us'] - s_deferred['mean_us']
        print(f"   -> savings per micro-batch if deferred: {savings:.1f}us", flush=True)
    else:
        results["item_6_direct"] = {"status": "no_cuda"}
        print(f"   SKIPPED (no CUDA)", flush=True)

    # ── 3. nvidia-smi vs pynvml ─────────────────────────────────────
    n_temp = min(N, 20)  # nvidia-smi is slow, don't do 100
    print(f"\n3. GPU temperature: nvidia-smi subprocess vs pynvml", flush=True)
    times_smi = _bench_nvidia_smi(n_temp)
    s_smi = _stats(times_smi)
    results["nvidia_smi"] = s_smi
    print(f"   nvidia-smi: mean={s_smi['mean_us']:.0f}μs "
          f"({s_smi['mean_us']/1000:.1f}ms)  n={n_temp}", flush=True)

    times_nvml = _bench_pynvml(n_temp)
    if times_nvml:
        s_nvml = _stats(times_nvml)
        results["pynvml"] = s_nvml
        print(f"   pynvml:    mean={s_nvml['mean_us']:.0f}μs "
              f"({s_nvml['mean_us']/1000:.1f}ms)  n={n_temp}", flush=True)
        speedup = s_smi['mean_us'] / max(1, s_nvml['mean_us'])
        print(f"   -> pynvml is {speedup:.0f}x faster", flush=True)
    else:
        results["pynvml"] = {"status": "unavailable"}
        print(f"   pynvml: UNAVAILABLE (pip install pynvml)", flush=True)

    # ── 4. empty_cache() ────────────────────────────────────────────
    print(f"\n4. torch.cuda.empty_cache() — allocator flush", flush=True)
    times_ec = _bench_empty_cache(N)
    if times_ec:
        s_ec = _stats(times_ec)
        results["empty_cache"] = s_ec
        print(f"   mean={s_ec['mean_us']:.1f}μs ({s_ec['mean_us']/1000:.2f}ms)  "
              f"p95={s_ec['p95_us']:.1f}μs", flush=True)
    else:
        results["empty_cache"] = {"status": "no_cuda"}

    # ── 5. wsd_block() ──────────────────────────────────────────────
    print(f"\n5. wsd_block() — pure Python phase computation", flush=True)
    times_wsd = _bench_wsd_block(N, cfg)
    s_wsd = _stats(times_wsd)
    results["wsd_block"] = s_wsd
    print(f"   mean={s_wsd['mean_us']:.1f}μs  "
          f"median={s_wsd['median_us']:.1f}μs", flush=True)
    # Cost over 4000 steps × 8 micro-batches
    total_wsd_us = s_wsd['mean_us'] * 4000 * 8
    print(f"   -> total over 4000x8=32000 calls: {total_wsd_us/1e6:.2f}s", flush=True)

    # ── 6. torch.remainder ──────────────────────────────────────────
    print(f"\n6. torch.remainder(input_ids, vocab_cap) — GPU kernel", flush=True)
    times_rem = _bench_remainder(N)
    if times_rem:
        s_rem = _stats(times_rem)
        results["remainder"] = s_rem
        print(f"   mean={s_rem['mean_us']:.1f}μs  "
              f"median={s_rem['median_us']:.1f}μs", flush=True)
    else:
        results["remainder"] = {"status": "no_cuda"}

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}", flush=True)
    print(f"{'OVERHEAD SUMMARY':^72}", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"{'Item':<40} {'Per call':>12} {'Per 4000 steps':>16}", flush=True)
    print(f"{'-' * 72}", flush=True)

    # Per micro-batch items (×32000 = 4000 steps × 8 accum)
    per_mb = 32000
    for key, label, freq in [
        ("isfinite_item", "isfinite().item()", per_mb),
        ("item_6_direct", ".item() × 6 metrics", per_mb),
        ("remainder", "torch.remainder()", per_mb),
        ("wsd_block", "wsd_block()", per_mb),
    ]:
        s = results.get(key, {})
        if isinstance(s, dict) and "mean_us" in s:
            total_s = s["mean_us"] * freq / 1e6
            print(f"  {label:<38} {s['mean_us']:>10.1f}μs {total_s:>14.1f}s", flush=True)

    # Per optimizer step items (×4000)
    per_os = 4000
    for key, label, freq in [
        ("empty_cache", "empty_cache() post-ckpt", 20),  # ~20 times in 4000 steps
    ]:
        s = results.get(key, {})
        if isinstance(s, dict) and "mean_us" in s:
            total_s = s["mean_us"] * freq / 1e6
            print(f"  {label:<38} {s['mean_us']:>10.1f}μs {total_s:>14.3f}s", flush=True)

    # Periodic items
    for key, label, freq in [
        ("nvidia_smi", "nvidia-smi (every 10 steps)", 400),
        ("pynvml", "pynvml (every 10 steps)", 400),
    ]:
        s = results.get(key, {})
        if isinstance(s, dict) and "mean_us" in s:
            total_s = s["mean_us"] * freq / 1e6
            print(f"  {label:<38} {s['mean_us']:>10.0f}μs {total_s:>14.1f}s", flush=True)

    print(f"{'-' * 72}", flush=True)

    # Total overhead estimate
    total_overhead = 0
    for key, freq in [
        ("isfinite_item", per_mb), ("item_6_direct", per_mb),
        ("remainder", per_mb), ("wsd_block", per_mb),
        ("empty_cache", 20), ("nvidia_smi", 400),
    ]:
        s = results.get(key, {})
        if isinstance(s, dict) and "mean_us" in s:
            total_overhead += s["mean_us"] * freq / 1e6
    print(f"  {'TOTAL OVERHEAD (est. 4000 steps)':<38} {'':>12} {total_overhead:>14.1f}s",
          flush=True)

    write_json(RESULTS_DIR / "overhead_items.summary.json", results)
    print(f"\nResults: {RESULTS_DIR / 'overhead_items.summary.json'}", flush=True)


if __name__ == "__main__":
    main()

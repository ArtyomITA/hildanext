#!/usr/bin/env python
"""sweep_seq.py — Sweep seq_len values to find the OOM threshold.

For each seq_len in --seq_lens, runs a short micro-bench (N steps)
and records peak VRAM + whether OOM occurred.
Outputs a summary table and JSONL logs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# We invoke vram_bench.run_bench programmatically
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vram_bench import run_bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Sweep seq_len for OOM threshold")
    parser.add_argument("--precision", default="amp_fp16")
    parser.add_argument("--optimizer", default="bnb_paged_adamw8bit")
    parser.add_argument("--checkpoint", default="full")
    parser.add_argument("--sdpa_backend", default="MATH")
    parser.add_argument("--micro_bs", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq_lens", type=str, default="1024,1536,2048,2560")
    parser.add_argument("--model_dir", type=str, default="")
    args = parser.parse_args()

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    results: List[Dict] = []

    print(f"[sweep_seq] Sweeping seq_lens={seq_lens}")
    print(f"[sweep_seq] precision={args.precision}  optimizer={args.optimizer}  checkpoint={args.checkpoint}")
    print()

    oom_threshold = None

    for sl in seq_lens:
        print(f"{'='*60}")
        print(f"[sweep_seq] Testing seq_len={sl}")
        print(f"{'='*60}")

        # Build a namespace matching vram_bench CLI args
        bench_args = argparse.Namespace(
            precision=args.precision,
            optimizer=args.optimizer,
            seq_len=sl,
            micro_bs=args.micro_bs,
            grad_acc=args.grad_acc,
            steps=args.steps,
            checkpoint=args.checkpoint,
            sdpa_backend=args.sdpa_backend,
            variable_seq=False,
            leak_check=False,
            out_jsonl="",
            model_dir=args.model_dir,
            mode="train",
        )

        t0 = time.time()
        try:
            summary = run_bench(bench_args)
            elapsed = time.time() - t0

            # Read peak VRAM from JSONL
            jsonl_path = Path(summary["out_jsonl"])
            peak_alloc = 0.0
            peak_res = 0.0
            if jsonl_path.exists():
                for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    m = row.get("metrics", {})
                    peak_alloc = max(peak_alloc, m.get("cuda_peak_alloc_mb", 0))
                    peak_res = max(peak_res, m.get("cuda_peak_res_mb", 0))

            entry = {
                "seq_len": sl,
                "oom": summary.get("oom", False),
                "steps_done": summary.get("steps_done", 0),
                "total_nan_inf": summary.get("total_nan_inf", 0),
                "peak_alloc_mb": round(peak_alloc, 1),
                "peak_res_mb": round(peak_res, 1),
                "last_loss": summary.get("last_loss"),
                "time_s": round(elapsed, 1),
            }
            results.append(entry)

            if summary.get("oom") and oom_threshold is None:
                oom_threshold = sl

            print(f"[sweep_seq] seq_len={sl}  oom={entry['oom']}  peak_alloc={peak_alloc:.0f}MB  peak_res={peak_res:.0f}MB")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"[sweep_seq] seq_len={sl}  FAILED: {e}")
            entry = {
                "seq_len": sl,
                "oom": "out of memory" in str(e).lower(),
                "steps_done": 0,
                "total_nan_inf": 0,
                "peak_alloc_mb": 0,
                "peak_res_mb": 0,
                "last_loss": None,
                "time_s": round(elapsed, 1),
                "error": str(e),
            }
            results.append(entry)
            if entry["oom"] and oom_threshold is None:
                oom_threshold = sl

        # Free memory between runs
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("[sweep_seq] SUMMARY")
    print(f"{'='*60}")
    header = f"{'seq_len':>8} {'OOM':>5} {'steps':>6} {'nan_inf':>8} {'peak_alloc_MB':>14} {'peak_res_MB':>12} {'loss':>10} {'time_s':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        loss_s = f"{r['last_loss']:.4f}" if r.get("last_loss") is not None else "N/A"
        print(
            f"{r['seq_len']:>8} {str(r['oom']):>5} {r['steps_done']:>6} "
            f"{r['total_nan_inf']:>8} {r['peak_alloc_mb']:>14.1f} {r['peak_res_mb']:>12.1f} "
            f"{loss_s:>10} {r['time_s']:>8.1f}"
        )
    if oom_threshold is not None:
        print(f"\n[sweep_seq] OOM threshold at seq_len={oom_threshold}")
    else:
        print(f"\n[sweep_seq] No OOM encountered across tested seq_lens.")

    # Write summary JSON
    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary = {
        "results": results,
        "oom_threshold": oom_threshold,
        "config": {
            "precision": args.precision,
            "optimizer": args.optimizer,
            "checkpoint": args.checkpoint,
            "sdpa_backend": args.sdpa_backend,
            "micro_bs": args.micro_bs,
            "grad_acc": args.grad_acc,
            "steps_per_sl": args.steps,
        },
    }
    out_path = out_dir / "sweep_seq_summary.json"
    out_path.write_text(json.dumps(sweep_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[sweep_seq] Summary saved to {out_path}")


if __name__ == "__main__":
    main()

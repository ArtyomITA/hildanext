"""Sequence Length Sweep Benchmark.

Tests seq_len=1024, 1536, 2048 with re-tokenization on-the-fly.
For seq_len > shard_seq_len, tiles the shard data (sufficient for VRAM/speed measurement).
Also optionally re-builds proper 2048-wide Dolma shards via _build_tokenized_artifacts.

Usage:
    python -m test.overhead_bench.bench_seq_len [--seq-lens 1024,1536,2048] [--opt-steps N]
    python -m test.overhead_bench.bench_seq_len --retokenize 2048   # full re-tokenize
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import numpy as np
import torch

from test.overhead_bench._common import (
    RESULTS_DIR, load_cfg, load_bundle, make_loader, make_optimizer,
    reset_vram, vram_stats, append_jsonl, write_json, gpu_info,
    forward_backward, fix_stdout_encoding,
)


def _retokenize_dolma(cfg, target_seq_len: int, out_dir: Path) -> dict:
    """Re-pack existing Dolma shards to new seq_len.

    Reads all tokens from existing shards, concatenates, and re-packs at target_seq_len.
    Doc boundaries are re-computed as contiguous blocks.
    """
    from hildanext.training import MmapShardedDataset

    dolma_root = str(Path(cfg.data.dolma_path).parent)
    ds = MmapShardedDataset(dolma_root)
    src_seq = ds.seq_len

    print(f"  Re-tokenizing: {ds.total_rows} rows x {src_seq} -> target {target_seq_len}",
          flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = out_dir / "tokens"
    doc_dir = out_dir / "doc_index"
    tok_dir.mkdir(exist_ok=True)
    doc_dir.mkdir(exist_ok=True)

    # Stream all token IDs and doc IDs
    shard_rows = 1000
    cur_tok, cur_doc = [], []
    shard_i, total_rows = 0, 0
    buf_tokens, buf_docs = [], []

    for idx in range(ds.total_rows):
        item = ds[idx]
        buf_tokens.extend(item["input_ids"].tolist())
        buf_docs.extend(item["doc_ids"].tolist())

        # Pack into target_seq_len rows
        while len(buf_tokens) >= target_seq_len:
            row_tok = buf_tokens[:target_seq_len]
            row_doc = buf_docs[:target_seq_len]
            buf_tokens = buf_tokens[target_seq_len:]
            buf_docs = buf_docs[target_seq_len:]

            cur_tok.append(row_tok)
            cur_doc.append(row_doc)
            total_rows += 1

            if len(cur_tok) >= shard_rows:
                np.save(tok_dir / f"tokens_{shard_i:05d}.npy",
                        np.asarray(cur_tok, dtype=np.uint32))
                np.save(doc_dir / f"doc_index_{shard_i:05d}.npy",
                        np.asarray(cur_doc, dtype=np.int32))
                shard_i += 1
                cur_tok, cur_doc = [], []
                if shard_i % 20 == 0:
                    print(f"    shard={shard_i} rows={total_rows:,}", flush=True)

    # Flush remainder
    if cur_tok:
        np.save(tok_dir / f"tokens_{shard_i:05d}.npy",
                np.asarray(cur_tok, dtype=np.uint32))
        np.save(doc_dir / f"doc_index_{shard_i:05d}.npy",
                np.asarray(cur_doc, dtype=np.int32))
        shard_i += 1

    meta = {
        "seq_len": target_seq_len,
        "rows_total": total_rows,
        "shards": shard_i,
        "tokens_dir": str(tok_dir),
        "doc_index_dir": str(doc_dir),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  Done: {shard_i} shards, {total_rows:,} rows at seq_len={target_seq_len}",
          flush=True)
    return meta


def _run_seq_test(
    seq_len: int,
    model, bundle, cfg,
    loader_iter,
    n_opt_steps: int,
    grad_acc: int,
    opt_name: str = "AdamW8bit",
) -> dict:
    device = bundle.device
    use_amp = device.type == "cuda"
    lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)
    ct_t_min = float(cfg.experiment.t_min)
    ct_t_max = float(cfg.experiment.t_max)

    # grad_ckpt ON for safety
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
        return {"seq_len": seq_len, "status": "opt_unavailable", "error": str(e)[:200]}

    from hildanext.diffusion import (
        _install_embed_noise_hook, _remove_embed_noise_hook,
        set_embed_noise_std,
    )
    _install_embed_noise_hook(model, bundle.mask_id, noise_std=0.1)
    reset_vram()

    # Warmup
    for _ in range(2):
        try:
            batch = next(loader_iter)
        except StopIteration:
            _remove_embed_noise_hook()
            del opt
            return {"seq_len": seq_len, "status": "data_exhausted"}
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        try:
            forward_backward(model, batch, bundle, cfg, use_amp,
                             ct_t_min, ct_t_max, grad_acc=1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _remove_embed_noise_hook()
                opt.zero_grad(set_to_none=True)
                del opt
                reset_vram()
                return {"seq_len": seq_len, "status": "oom_warmup",
                        "error": str(e)[:200]}
            raise
        opt.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    fwd_times, bwd_times, opt_times = [], [], []
    losses, grad_norms = [], []
    nan_count, total_tokens = 0, 0
    oom = False

    t_start = time.perf_counter()

    for step_i in range(n_opt_steps):
        for _ in range(grad_acc):
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

        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        grad_norms.append(gn)
        if math.isfinite(gn) and gn < 10000:
            t_opt0 = time.perf_counter()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            opt_times.append(time.perf_counter() - t_opt0)
        opt.zero_grad(set_to_none=True)

        frac = max(0.0, 1.0 - float(step_i + 1) / float(n_opt_steps))
        set_embed_noise_std(0.1 * frac)

    t_end = time.perf_counter()
    wall = t_end - t_start

    peak = vram_stats()
    _remove_embed_noise_hook()
    opt.zero_grad(set_to_none=True)
    del opt
    reset_vram()

    step_time = wall / max(1, len(grad_norms))
    est_4000_h = step_time * 4000 / 3600

    return {
        "seq_len": seq_len,
        "optimizer": opt_name,
        "status": "oom" if oom else "ok",
        "opt_steps_done": len(grad_norms),
        "wall_s": round(wall, 2),
        "total_tokens": total_tokens,
        "tok_per_s": round(total_tokens / max(0.001, wall), 1),
        "avg_fwd_s": round(sum(fwd_times) / max(1, len(fwd_times)), 4),
        "avg_bwd_s": round(sum(bwd_times) / max(1, len(bwd_times)), 4),
        "avg_opt_step_s": round(sum(opt_times) / max(1, len(opt_times)), 4),
        "step_time_s": round(step_time, 3),
        "est_4000_steps_h": round(est_4000_h, 1),
        "eff_tokens_per_step": seq_len * grad_acc,
        "peak_alloc_mb": peak.get("peak_alloc_mb", 0),
        "peak_reserved_mb": peak.get("reserved_mb", 0),
        "nan_count": nan_count,
        "loss_last": round(losses[-1], 4) if losses else 0,
    }


def main():
    fix_stdout_encoding()
    ap = argparse.ArgumentParser(description="Seq Len Sweep Benchmark")
    ap.add_argument("--seq-lens", type=str, default="1024,1536,2048",
                    help="Comma-separated seq lengths to test")
    ap.add_argument("--opt-steps", type=int, default=5)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--optimizer", type=str, default="AdamW8bit")
    ap.add_argument("--retokenize", type=int, default=0,
                    help="If >0, re-tokenize Dolma at this seq_len and exit")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # ── Re-tokenize mode ─────────────────────────────────────────────
    if args.retokenize > 0:
        target = args.retokenize
        out_dir = RESULTS_DIR / f"dolma_retok_{target}"
        print(f"Re-tokenizing Dolma shards to seq_len={target}", flush=True)
        meta = _retokenize_dolma(cfg, target, out_dir)
        print(f"\nMetadata: {json.dumps(meta, indent=2)}", flush=True)
        print(f"Shards written to: {out_dir}", flush=True)
        return

    # ── Sweep mode ───────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("CUDA not available — aborting", flush=True)
        return

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
    info = gpu_info()
    shard_seq = cfg.data.seq_len

    print(f"{'=' * 72}", flush=True)
    print(f"SEQ LEN SWEEP BENCHMARK", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"GPU: {info['gpu']}  shard_seq={shard_seq}  test={seq_lens}  "
          f"optimizer={args.optimizer}", flush=True)

    bundle = load_bundle(cfg, for_training=True)
    model = bundle.model
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    results = []

    for seq_len in seq_lens:
        print(f"\n--- seq_len={seq_len} ---", flush=True)

        # Check for re-tokenized shards
        retok_dir = RESULTS_DIR / f"dolma_retok_{seq_len}"
        if seq_len != shard_seq and (retok_dir / "meta.json").exists():
            print(f"  Using re-tokenized shards from {retok_dir}", flush=True)
            from hildanext.training import MmapShardedDataset
            from hildanext.config import clone_with_updates
            test_cfg = clone_with_updates(cfg, {
                "data": {"seq_len": seq_len,
                         "dolma_path": str(retok_dir / "raw")},
                "wsd": {"max_block_size": seq_len},
            })
            ds = MmapShardedDataset(str(retok_dir))
            from torch.utils.data import DataLoader
            from hildanext.training import _collate
            loader = DataLoader(
                ds, batch_size=1, shuffle=True, num_workers=0,
                collate_fn=_collate, pin_memory=True,
            )
        elif seq_len != shard_seq:
            print(f"  No retokenized shards for {seq_len}. Using tile/truncate from "
                  f"shard_seq={shard_seq}", flush=True)
            from hildanext.config import clone_with_updates
            test_cfg = clone_with_updates(cfg, {
                "data": {"seq_len": seq_len},
                "wsd": {"max_block_size": seq_len},
            })
            base_loader = make_loader(cfg)

            def _adapt(it, target):
                for batch in it:
                    if target <= shard_seq:
                        yield {k: v[:, :target] if v.dim() > 1 else v
                               for k, v in batch.items()}
                    else:
                        reps = (target + shard_seq - 1) // shard_seq
                        yield {k: v.repeat(1, reps)[:, :target] if v.dim() > 1 else v
                               for k, v in batch.items()}

            loader = _adapt(iter(base_loader), seq_len)
            loader = list(loader)[:2 + args.opt_steps * args.grad_acc + 5]
            loader = iter(loader)
        else:
            test_cfg = cfg
            loader = make_loader(cfg)

        from hildanext.utils import seed_everything
        seed_everything(42)

        try:
            r = _run_seq_test(
                seq_len=seq_len,
                model=model, bundle=bundle, cfg=test_cfg,
                loader_iter=iter(loader) if not hasattr(loader, '__next__') else loader,
                n_opt_steps=args.opt_steps,
                grad_acc=args.grad_acc,
                opt_name=args.optimizer,
            )
        except Exception as e:
            r = {"seq_len": seq_len, "status": "error", "error": str(e)[:300]}
            traceback.print_exc()

        results.append(r)

        s = r.get("status", "?")
        if s == "ok":
            print(f"  [OK] peak={r['peak_alloc_mb']:.0f}MB  tok/s={r['tok_per_s']}  "
                  f"step={r['step_time_s']:.2f}s  4000->{r['est_4000_steps_h']}h  "
                  f"NaN={r['nan_count']}", flush=True)
        else:
            print(f"  [FAIL] status={s}  {r.get('error', '')[:100]}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 90}", flush=True)
    print(f"{'SEQ LEN SWEEP — SUMMARY':^90}", flush=True)
    print(f"{'=' * 90}", flush=True)
    hdr = (f"{'SeqLen':>8} {'Status':<8} {'PeakMB':>8} {'tok/s':>8} "
           f"{'step_s':>8} {'4000h':>8} {'eff_tok':>8} {'NaN':>5}")
    print(hdr, flush=True)
    print("-" * 90, flush=True)
    for r in results:
        s = r.get("status", "?")
        if s == "ok":
            line = (f"{r['seq_len']:>8} {'OK':<8} "
                    f"{r['peak_alloc_mb']:>8.0f} "
                    f"{r['tok_per_s']:>8.1f} "
                    f"{r['step_time_s']:>8.3f} "
                    f"{r['est_4000_steps_h']:>8.1f} "
                    f"{r['eff_tokens_per_step']:>8} "
                    f"{r['nan_count']:>5}")
        else:
            line = f"{r.get('seq_len', '?'):>8} {s.upper():<8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>5}"
        print(line, flush=True)
    print("-" * 90, flush=True)

    # Highlight: tokens per wall-clock hour
    print(f"\nEffective training throughput (tokens/hour):", flush=True)
    for r in results:
        if r.get("status") == "ok":
            tph = r["tok_per_s"] * 3600
            print(f"  seq={r['seq_len']:>5}: {tph:,.0f} tok/h  "
                  f"({tph/1e6:.1f}M tok/h)", flush=True)

    write_json(RESULTS_DIR / "seq_len_sweep.summary.json",
               {"gpu": info, "results": results})
    print(f"\nResults: {RESULTS_DIR / 'seq_len_sweep.summary.json'}", flush=True)


if __name__ == "__main__":
    main()

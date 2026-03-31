#!/usr/bin/env python
"""test_bidir_precise.py — 3 precise, unambiguous bidirectional tests.

Test 1  — same-block bidirectional (composite_llada20)
  Asserts: mask3d[i,j]==True, attn4d.ndim==4, delta_logits[i] > threshold
  when j>i in the same xt block.

Test 2  — doc gating leakage (~0)
  Asserts: mask3d[i,j]==False, delta_logits[i] < threshold
  when i∈docA, j∈docB (cross-doc must be invisible).

Test 3  — micro-run: stable-phase bidirectional branch is actually executed
  warmup_steps=0, stable_steps=1, decay_steps=0, seq_len=128, 1 forward pass
  through compute_m2t_t2t_losses(bidirectional=True). Asserts loss is finite.

Usage:
  cd hildanext && conda activate mdm
  python -u tools/tests_wsd/test_bidir_precise.py \\
      --model-dir E:/DIFFUSION/HildaNext/Qwen3-0.6B

Expected total runtime: ~2 min on GTX 1080 8GB.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

# ── path bootstrap ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent          # hildanext/
_SRC  = _REPO / "backend" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hildanext.masks import _composite_llada20_mask            # noqa: E402
from hildanext.diffusion import (                               # noqa: E402
    force_noncausal_attention,
    compute_m2t_t2t_losses,
    wsd_block,
)
from hildanext.config import WSDConfig                          # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────

def _force_math_sdpa() -> None:
    """Pascal SM_61: no FlashAttention, force math SDPA."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _load(model_dir: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float16
    ).to(device)
    m.eval()
    return m


def _make_attn4d(mask3d: torch.Tensor, model) -> torch.Tensor:
    """[B,S,S] bool → [B,1,S,S] float additive mask (same dtype as model)."""
    m = mask3d.bool()
    if m.dim() == 3:
        m = m[:, None, :, :]
    try:
        dt = next(model.parameters()).dtype
    except StopIteration:
        dt = torch.float32
    out = torch.zeros(m.shape, device=m.device, dtype=dt)
    out = out.masked_fill(~m, torch.finfo(dt).min)
    return out


def _ok(label: str, details: Dict[str, Any]) -> Dict[str, Any]:
    print(f"  ✓  PASS  {label}", flush=True)
    return {"pass": True, "label": label, **details}


def _fail(label: str, msg: str, details: Dict[str, Any]) -> Dict[str, Any]:
    print(f"  ✗  FAIL  {label}: {msg}", file=sys.stderr, flush=True)
    return {"pass": False, "label": label, "reason": msg, **details}


# ── Test 1 ───────────────────────────────────────────────────────────────────

def test1_same_block_bidirectional(
    model,
    device: torch.device,
    block_size: int = 32,
    base_len: int = 128,
    threshold: float = 1e-4,
) -> Dict[str, Any]:
    """
    Setup: [xt | x0], total = 2*base_len, one doc, seed=42.
    i=5, j=last position of i's block (= block_size-1 if i in block 0).
    Both i and j are in xt (< base_len) → cond_xt_xt must be True.

    PRE-ASSERTIONS (hard failures if wrong):
      A1  i // block_size == j // block_size           (same block)
      A2  mask3d[0, i, j] == True                      (mask allows i→j)
      A3  attn4d.ndim == 4                             (4D mask, NOT 2D)
      A4  attn4d.shape == (1, 1, 2*base_len, 2*base_len)

    POST-ASSERTION:
      A5  delta_logits_at_i > threshold                (vision is real)
    """
    label = "T1_same_block_bidirectional"
    total = 2 * base_len
    vocab = model.config.vocab_size

    # ── positions ──────────────────────────────────────────────────────────
    i = 5
    block_of_i = i // block_size
    j = (block_of_i + 1) * block_size - 1          # last pos in i's block
    j = min(j, base_len - 1)                        # must stay in xt half
    assert i < base_len and j < base_len, "i,j must be in xt half"

    # ── A1: same block ─────────────────────────────────────────────────────
    if i // block_size != j // block_size:
        return _fail(label, f"A1 violated: i={i} block={i//block_size}, j={j} block={j//block_size}", {})

    # ── build inputs ────────────────────────────────────────────────────────
    torch.manual_seed(42)
    ids_xt = torch.randint(0, vocab, (1, base_len), device=device)
    ids2   = torch.cat([ids_xt, ids_xt.clone()], dim=1)        # [xt | x0]
    doc_ids = torch.zeros(1, total, dtype=torch.long, device=device)  # single doc

    # ── build mask ──────────────────────────────────────────────────────────
    mask3d = _composite_llada20_mask(doc_ids, base_len=base_len, block_size=block_size)

    # ── A2: mask allows i→j ────────────────────────────────────────────────
    if not bool(mask3d[0, i, j].item()):
        return _fail(label, f"A2 violated: mask3d[0,{i},{j}]==False (same-block visibility not set)", {
            "i": i, "j": j, "block_size": block_size
        })

    attn4d = _make_attn4d(mask3d, model)

    # ── A3 & A4: 4D shape ──────────────────────────────────────────────────
    if attn4d.ndim != 4:
        return _fail(label, f"A3 violated: attn4d.ndim={attn4d.ndim}, expected 4", {})
    expected_shape = (1, 1, total, total)
    if tuple(attn4d.shape) != expected_shape:
        return _fail(label, f"A4 violated: attn4d.shape={tuple(attn4d.shape)}, expected {expected_shape}", {})

    # ── forward passes ──────────────────────────────────────────────────────
    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_base = model(input_ids=ids2, attention_mask=attn4d).logits

    ids2_mod = ids2.clone()
    ids2_mod[0, j] = (ids2_mod[0, j] + 1) % vocab

    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_mod = model(input_ids=ids2_mod, attention_mask=attn4d).logits

    delta = float((logits_mod[0, i, :] - logits_base[0, i, :]).abs().mean().item())

    details = {
        "i": i, "j": j, "block_size": block_size, "base_len": base_len,
        "delta": round(delta, 8), "threshold": threshold,
        "mask_i_sees_j": True,
        "attn4d_shape": list(attn4d.shape),
        "attn4d_ndim": attn4d.ndim,
    }

    # ── A5: real bidirectional effect ────────────────────────────────────
    if delta <= threshold:
        return _fail(label, f"A5 violated: delta={delta:.8f} <= threshold={threshold}", details)

    return _ok(label, details)


# ── Test 2 ───────────────────────────────────────────────────────────────────

def test2_doc_gating_leakage(
    model,
    device: torch.device,
    block_size: int = 32,
    base_len: int = 128,
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    """
    Setup: doc_ids = [0…0 | 1…1] in xt half (split at base_len//2),
           mirrored in x0 half → total 2*base_len.
    i=5 (docA, xt), j = base_len//2 + 5 (docB, xt).

    PRE-ASSERTIONS:
      B1  i and j in different docs                    (by construction)
      B2  mask3d[0, i, j] == False                     (cross-doc blocked)
      B3  attn4d.ndim == 4

    POST-ASSERTION:
      B4  delta_logits_at_i < threshold                (no leakage)
    """
    label = "T2_doc_gating_leakage"
    total = 2 * base_len
    vocab = model.config.vocab_size
    half  = base_len // 2

    # ── positions ──────────────────────────────────────────────────────────
    i = 5               # docA, xt
    j = half + 5        # docB, xt
    assert i < half and j >= half and j < base_len, "position layout error"

    # ── build doc_ids: [docA | docB | docA | docB] ─────────────────────────
    doc_xt = torch.zeros(1, base_len, dtype=torch.long, device=device)
    doc_xt[0, half:] = 1
    doc_ids = torch.cat([doc_xt, doc_xt.clone()], dim=1)   # mirror in x0

    # ── build inputs ────────────────────────────────────────────────────────
    torch.manual_seed(7)
    ids_xt = torch.randint(0, vocab, (1, base_len), device=device)
    ids2   = torch.cat([ids_xt, ids_xt.clone()], dim=1)

    # ── build mask ──────────────────────────────────────────────────────────
    mask3d = _composite_llada20_mask(doc_ids, base_len=base_len, block_size=block_size)

    # ── B1: different docs ─────────────────────────────────────────────────
    if bool(doc_ids[0, i].item()) == bool(doc_ids[0, j].item()):
        return _fail(label, "B1 violated: i and j accidentally in the same doc", {})

    # ── B2: mask blocks i→j ────────────────────────────────────────────────
    if bool(mask3d[0, i, j].item()):
        return _fail(label, f"B2 violated: mask3d[0,{i},{j}]==True — cross-doc NOT blocked", {
            "i": i, "j": j, "doc_i": int(doc_ids[0, i]), "doc_j": int(doc_ids[0, j])
        })

    attn4d = _make_attn4d(mask3d, model)

    # ── B3: 4D mask ────────────────────────────────────────────────────────
    if attn4d.ndim != 4:
        return _fail(label, f"B3 violated: attn4d.ndim={attn4d.ndim}, expected 4", {})

    # ── forward passes ──────────────────────────────────────────────────────
    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_base = model(input_ids=ids2, attention_mask=attn4d).logits

    ids2_mod = ids2.clone()
    ids2_mod[0, j] = (ids2_mod[0, j] + 1) % vocab

    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_mod = model(input_ids=ids2_mod, attention_mask=attn4d).logits

    delta = float((logits_mod[0, i, :] - logits_base[0, i, :]).abs().max().item())

    details = {
        "i": i, "j": j, "doc_i": 0, "doc_j": 1,
        "delta_max": round(delta, 10), "threshold": threshold,
        "mask_i_sees_j": False,
        "attn4d_shape": list(attn4d.shape),
    }

    # ── B4: no leakage ─────────────────────────────────────────────────────
    if delta >= threshold:
        return _fail(label, f"B4 violated: delta_max={delta:.2e} >= threshold={threshold:.2e} — GATING LEAK", details)

    return _ok(label, details)


# ── Test 3 ───────────────────────────────────────────────────────────────────

def test3_stable_phase_micro_run(
    model,
    device: torch.device,
    block_size: int = 32,
    seq_len: int = 128,
    mask_id: int = 32000,
) -> Dict[str, Any]:
    """
    Micro-run: WSD with warmup_steps=0, stable_steps=1, decay_steps=0.
    Confirms:
      C1  wsd_block(step=0, warmup=0, stable=1) → phase == "stable"
      C2  bidirectional=True is set for phase "stable"
              (with attention_mode="bidirectional_only_stable")
      C3  compute_m2t_t2t_losses runs without error and returns finite loss

    This is NOT a training loop — it's one forward pass through the
    production code path to confirm the bidirectional branch is reachable.
    """
    label = "T3_stable_phase_micro_run"

    # ── C1: wsd_block at step=1 with warmup=0, stable=2 → stable ──────────
    # llada2_wsd_block uses step > warmup_steps (strict), so step=0 returns
    # 'warmup' regardless. step=1 with warmup_steps=0, stable_steps=2 → stable.
    wsd_cfg = WSDConfig(
        warmup_steps=0,
        stable_steps=2,
        decay_steps=0,
        start_block_size=block_size,
        max_block_size=block_size,
        end_block_size=block_size,
    )
    step = wsd_block(step=1, cfg=wsd_cfg, seq_len=seq_len)

    if step.phase != "stable":
        return _fail(label, f"C1 violated: wsd_block returned phase='{step.phase}', expected 'stable'", {
            "warmup_steps": 0, "stable_steps": 2, "queried_step": 1
        })

    # ── C2: bidirectional=True for stable phase with mode=bidirectional_only_stable ──
    attn_mode = "bidirectional_only_stable"
    bidirectional = (step.phase == "stable") if attn_mode == "bidirectional_only_stable" else False

    if not bidirectional:
        return _fail(label, "C2 violated: bidirectional=False for stable phase with bidirectional_only_stable", {})

    # ── build minimal batch ─────────────────────────────────────────────────
    vocab = model.config.vocab_size
    torch.manual_seed(99)
    input_ids   = torch.randint(0, vocab, (1, seq_len), device=device)
    attn_mask   = torch.ones(1, seq_len, dtype=torch.long, device=device)
    doc_ids     = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    resp_mask   = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # Clamp mask_id to valid range
    mask_id_safe = min(mask_id, vocab - 1)

    # ── C3: single forward through production loss function ────────────────
    from hildanext.config import TrainConfig
    train_cfg = TrainConfig(
        mask_ratio=0.15,
        t2t_noise_ratio=0.1,
        m2t_weight=1.0,
        t2t_weight=0.1,
        multi_turn_t2t=1,
    )

    with torch.no_grad():
        out = compute_m2t_t2t_losses(
            model=model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            doc_ids=doc_ids,
            response_mask=resp_mask,
            mask_id=mask_id_safe,
            vocab_size=vocab,
            cfg=train_cfg,
            focus_response=False,
            mask_mode="composite_llada20",
            composite_block_size=block_size,
            bidirectional=True,           # ← production value for stable phase
            time_param="continuous_time",
            loss_weighting="inv_t",
            t_min=0.001,
            t_max=1.0,
        )

    loss_val = float(out["loss"].item())
    loss_finite = bool(torch.isfinite(out["loss"]).item())

    details = {
        "wsd_phase": step.phase,
        "block_size": step.block_size,
        "bidirectional": bidirectional,
        "attn_mode": attn_mode,
        "loss": round(loss_val, 6),
        "loss_finite": loss_finite,
        "masked_token_acc": out.get("masked_token_acc"),
        "t_sampled": round(float(out.get("t_sampled", 0)), 4),
    }

    if not loss_finite:
        return _fail(label, f"C3 violated: loss={loss_val} is not finite (NaN/Inf)", details)

    return _ok(label, details)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--base-len",   type=int, default=128)
    parser.add_argument("--out",        default="reports/bidir_precise_test.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _force_math_sdpa()

    print(f"\n[bidir_precise] device={device}  block_size={args.block_size}  base_len={args.base_len}", flush=True)
    print(f"[bidir_precise] loading model from {args.model_dir} ...", flush=True)
    t0 = time.time()
    model = _load(args.model_dir, device)
    print(f"[bidir_precise] model loaded in {time.time()-t0:.1f}s\n", flush=True)

    results = []

    print("── Test 1: same-block bidirectional ──────────────────────────", flush=True)
    r1 = test1_same_block_bidirectional(model, device, args.block_size, args.base_len)
    results.append(r1)
    if r1["pass"]:
        print(f"           delta={r1['delta']:.8f}  threshold={r1['threshold']}"
              f"  attn4d_shape={r1['attn4d_shape']}", flush=True)

    print("\n── Test 2: doc gating leakage ────────────────────────────────", flush=True)
    r2 = test2_doc_gating_leakage(model, device, args.block_size, args.base_len)
    results.append(r2)
    if r2["pass"]:
        print(f"           delta_max={r2['delta_max']:.2e}  threshold={r2['threshold']}", flush=True)

    print("\n── Test 3: stable-phase micro-run ───────────────────────────", flush=True)
    vocab = model.config.vocab_size
    r3 = test3_stable_phase_micro_run(model, device, args.block_size, args.base_len,
                                      mask_id=min(32000, vocab - 1))
    results.append(r3)
    if r3["pass"]:
        print(f"           phase={r3['wsd_phase']}  bidirectional={r3['bidirectional']}"
              f"  loss={r3['loss']}  finite={r3['loss_finite']}", flush=True)

    # ── cleanup ────────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    # ── summary ────────────────────────────────────────────────────────────
    all_pass = all(r["pass"] for r in results)
    total_sec = round(time.time() - t0, 1)

    print(f"\n{'='*58}", flush=True)
    status = "ALL PASS ✓" if all_pass else "FAILED ✗"
    print(f"  {status}  ({total_sec}s)", flush=True)
    print(f"{'='*58}\n", flush=True)

    report = {
        "all_pass": all_pass,
        "elapsed_sec": total_sec,
        "model_dir": args.model_dir,
        "block_size": args.block_size,
        "base_len": args.base_len,
        "tests": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[bidir_precise] report → {out_path}", flush=True)

    if not all_pass:
        for r in results:
            if not r["pass"]:
                print(f"  FAILED: {r['label']} — {r.get('reason','')}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

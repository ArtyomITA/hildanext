#!/usr/bin/env python
"""test_bidirectional_composite_runtime.py — Block-aware bidirectional test
for the composite_llada20 mask path.

Tests:
  1. SAME-BLOCK bidirectional: perturb a future token j within the same
     composite block as probe token i (both in xt half).  Under true
     bidirectional attention, logits[i] MUST change.
  2. DOC-GATING leakage: perturb a token in doc B and verify logits of
     a token in doc A do NOT change (gating must hold).

Usage:
  python -m tools.tests_wsd.test_bidirectional_composite_runtime \
      --model-dir models/qwen3-0.6b [--out reports/bidir_test.json]

Can also be imported and called from preflight:
  from tools.tests_wsd.test_bidirectional_composite_runtime import run_all
  result = run_all(model_dir="models/qwen3-0.6b")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_model(model_dir: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model


def _make_4d_mask(mask3d: torch.Tensor, model) -> torch.Tensor:
    """Convert [B,S,S] bool mask → [B,1,S,S] float additive mask."""
    m = mask3d.bool()
    if m.dim() == 3:
        m = m[:, None, :, :]
    dt = torch.float32
    try:
        dt = next(model.parameters()).dtype
    except Exception:
        dt = torch.float32
    out = torch.zeros(m.shape, device=m.device, dtype=dt)
    out = out.masked_fill(~m, torch.finfo(dt).min)
    return out


# ---------------------------------------------------------------------------
# Test 1: same-block bidirectional (composite_llada20)
# ---------------------------------------------------------------------------

def test_same_block_bidirectional(
    model,
    device: torch.device,
    block_size: int = 32,
    seq_len: int = 128,
    threshold: float = 1e-5,
) -> Dict[str, Any]:
    """Perturb token j > i in the SAME xt block → logits[i] must change.

    Uses ``force_noncausal_attention`` to disable the internal causal mask,
    then passes the composite_llada20 4D mask which allows same-block
    bidirectional visibility.
    """
    # Lazy import — allows running without installing hildanext as package
    try:
        from hildanext.diffusion import force_noncausal_attention
        from hildanext.masks import _composite_llada20_mask
    except ImportError:
        # Fallback: add backend/src to path
        _root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(_root / "backend" / "src"))
        from hildanext.diffusion import force_noncausal_attention
        from hildanext.masks import _composite_llada20_mask

    vocab_size = model.config.vocab_size
    base_len = seq_len  # xt has base_len tokens, x0 also base_len → total 2*base_len
    total_len = 2 * base_len

    # Place i and j in same block of xt (first half)
    i = 5  # probe position in xt
    block_end = ((i // block_size) + 1) * block_size - 1  # last pos in i's block
    j = min(block_end, base_len - 1)  # stay in same block AND within xt
    if j <= i:
        j = i + 1  # minimal
    assert j < base_len, f"j={j} must be < base_len={base_len}"
    assert i // block_size == j // block_size, f"i,j not in same block"

    # Build input
    torch.manual_seed(42)
    input_ids_base = torch.randint(0, vocab_size, (1, base_len), device=device)
    x0 = input_ids_base.clone()  # clean copy
    # Composite: ids2 = [xt, x0]
    ids2_base = torch.cat([input_ids_base, x0], dim=1)
    # All same doc
    doc_ids = torch.zeros(1, total_len, dtype=torch.long, device=device)

    # Build composite mask
    mask3d = _composite_llada20_mask(doc_ids, base_len=base_len, block_size=block_size)
    attn4d = _make_4d_mask(mask3d, model)

    # Verify mask allows i→j in xt-xt region
    mask_i_sees_j = bool(mask3d[0, i, j].item())

    # Run 1: original
    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_base = model(input_ids=ids2_base, attention_mask=attn4d).logits

    # Run 2: perturb position j in xt
    ids2_mod = ids2_base.clone()
    ids2_mod[0, j] = (ids2_mod[0, j] + 1) % vocab_size

    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_mod = model(input_ids=ids2_mod, attention_mask=attn4d).logits

    # Measure delta at probe position i (in xt half)
    delta = float((logits_mod[0, i, :] - logits_base[0, i, :]).abs().mean().item())

    # Also measure WITHOUT force_noncausal_attention (to show the difference)
    with torch.no_grad():
        logits_causal_base = model(input_ids=ids2_base, attention_mask=attn4d).logits
        logits_causal_mod = model(input_ids=ids2_mod, attention_mask=attn4d).logits
    delta_without_override = float(
        (logits_causal_mod[0, i, :] - logits_causal_base[0, i, :]).abs().mean().item()
    )

    passed = delta > threshold
    backend = "math_sdpa"
    try:
        if torch.backends.cuda.flash_sdp_enabled():
            backend = "flash_sdpa"
        elif torch.backends.cuda.mem_efficient_sdp_enabled():
            backend = "mem_efficient_sdpa"
    except Exception:
        pass

    result = {
        "test": "same_block_bidirectional_composite",
        "pass": passed,
        "delta_with_override": round(delta, 8),
        "delta_without_override": round(delta_without_override, 8),
        "threshold": threshold,
        "mask_i_sees_j": mask_i_sees_j,
        "i": i,
        "j": j,
        "block_size": block_size,
        "seq_len": seq_len,
        "total_len": total_len,
        "backend": backend,
        "override_active": True,
    }
    return result


# ---------------------------------------------------------------------------
# Test 2: doc-gating leakage
# ---------------------------------------------------------------------------

def test_doc_gating_leakage(
    model,
    device: torch.device,
    block_size: int = 32,
    seq_len: int = 128,
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    """Perturb a token in doc B → logits of doc A token must NOT change.

    If delta > threshold → gating is leaking (FAIL).
    """
    try:
        from hildanext.diffusion import force_noncausal_attention
        from hildanext.masks import _composite_llada20_mask
    except ImportError:
        _root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(_root / "backend" / "src"))
        from hildanext.diffusion import force_noncausal_attention
        from hildanext.masks import _composite_llada20_mask

    vocab_size = model.config.vocab_size
    base_len = seq_len
    total_len = 2 * base_len

    # Doc A occupies positions 0..half-1, doc B occupies half..base_len-1
    half = base_len // 2
    doc_ids_xt = torch.zeros(1, base_len, dtype=torch.long, device=device)
    doc_ids_xt[0, half:] = 1  # doc B
    doc_ids = torch.cat([doc_ids_xt, doc_ids_xt], dim=1)

    torch.manual_seed(42)
    input_ids_base = torch.randint(0, vocab_size, (1, base_len), device=device)
    x0 = input_ids_base.clone()
    ids2_base = torch.cat([input_ids_base, x0], dim=1)

    mask3d = _composite_llada20_mask(doc_ids, base_len=base_len, block_size=block_size)
    attn4d = _make_4d_mask(mask3d, model)

    # Probe: position in doc A (e.g., i=5)
    # Perturb: position in doc B (e.g., j=half+5)
    i = min(5, half - 1)
    j = half + 5
    assert j < base_len, f"j={j} >= base_len={base_len}"

    # Verify mask blocks i→j (cross-doc)
    mask_i_sees_j = bool(mask3d[0, i, j].item())

    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_base = model(input_ids=ids2_base, attention_mask=attn4d).logits

    ids2_mod = ids2_base.clone()
    ids2_mod[0, j] = (ids2_mod[0, j] + 1) % vocab_size

    with torch.no_grad():
        with force_noncausal_attention(model):
            logits_mod = model(input_ids=ids2_mod, attention_mask=attn4d).logits

    delta = float((logits_mod[0, i, :] - logits_base[0, i, :]).abs().mean().item())

    # For doc gating, delta should be ~0 (no leakage)
    passed = delta < threshold

    result = {
        "test": "doc_gating_leakage_composite",
        "pass": passed,
        "delta": round(delta, 10),
        "threshold": threshold,
        "mask_i_sees_j": mask_i_sees_j,
        "i": i,
        "j": j,
        "doc_i": 0,
        "doc_j": 1,
        "block_size": block_size,
        "seq_len": seq_len,
    }
    return result


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_all(
    model_dir: str,
    device: Optional[torch.device] = None,
    block_size: int = 32,
    seq_len: int = 128,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all bidirectional tests. Returns combined report dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force MATH SDPA (Pascal / no FlashAttn)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    t0 = time.time()
    print(f"[bidir_test] loading model from {model_dir}", flush=True)
    model = _load_model(model_dir, device)
    t_load = time.time() - t0
    print(f"[bidir_test] model loaded in {t_load:.1f}s", flush=True)

    print("[bidir_test] running same_block_bidirectional ...", flush=True)
    r1 = test_same_block_bidirectional(model, device, block_size=block_size, seq_len=seq_len)
    print(f"[bidir_test]   PASS={r1['pass']} delta_with={r1['delta_with_override']:.8f} delta_without={r1['delta_without_override']:.8f}", flush=True)

    print("[bidir_test] running doc_gating_leakage ...", flush=True)
    r2 = test_doc_gating_leakage(model, device, block_size=block_size, seq_len=seq_len)
    print(f"[bidir_test]   PASS={r2['pass']} delta={r2['delta']:.10f}", flush=True)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    report = {
        "bidirectional_test": r1,
        "doc_gating_test": r2,
        "all_pass": r1["pass"] and r2["pass"],
        "bidirectional_pass": r1["pass"],
        "doc_gating_pass": r2["pass"],
        "model_dir": model_dir,
        "block_size": block_size,
        "seq_len": seq_len,
        "elapsed_sec": round(elapsed, 1),
    }

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[bidir_test] report saved to {out_path}", flush=True)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bidirectional composite attention runtime test")
    parser.add_argument("--model-dir", type=str, default="models/qwen3-0.6b")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--out", type=str, default="reports/bidir_composite_test.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = run_all(
        model_dir=args.model_dir,
        device=device,
        block_size=args.block_size,
        seq_len=args.seq_len,
        out_path=args.out,
    )
    status = "PASS" if report["all_pass"] else "FAIL"
    print(f"\n[bidir_test] === {status} ===", flush=True)
    if not report["all_pass"]:
        if not report["bidirectional_pass"]:
            print("[bidir_test] BIDIRECTIONAL_STABLE FAILED: force_noncausal_attention did not make attention bidirectional", file=sys.stderr, flush=True)
        if not report["doc_gating_pass"]:
            print("[bidir_test] DOC_GATING LEAKED: cross-document attention detected", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

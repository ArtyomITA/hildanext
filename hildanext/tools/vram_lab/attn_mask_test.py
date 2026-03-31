#!/usr/bin/env python
"""attn_mask_test.py — Unit test for hidden causal masks in SDPA and model forward.

Tests whether attention is truly bidirectional by perturbing future tokens
and checking if earlier positions' output changes.
  delta ~ 0  =>  causal (future tokens invisible)
  delta > 0  =>  bidirectional (future tokens influence past positions)

Also tests SDPA directly with is_causal=True/False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


def causal_leak_test(attn_fn, x: torch.Tensor) -> float:
    """Test if attn_fn is causal by perturbing the last token.

    Returns delta: ~0 means causal; >0 means bidirectional.
    """
    with torch.no_grad():
        y1 = attn_fn(x)
        x2 = x.clone()
        x2[:, -1, :] += 1.0
        y2 = attn_fn(x2)
        delta = (y2[:, 0, :] - y1[:, 0, :]).abs().mean().item()
    return delta


def sdpa_attn(x: torch.Tensor, is_causal: bool) -> torch.Tensor:
    """Wrapper for SDPA with q=k=v=x."""
    q = k = v = x.unsqueeze(1)  # [B, H=1, T, D]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    return out.squeeze(1)


def test_sdpa_causal_leak(device: torch.device) -> Dict[str, Any]:
    """Test SDPA with is_causal=True and is_causal=False."""
    B, T, D = 1, 32, 64
    x = torch.randn(B, T, D, device=device)

    delta_false = causal_leak_test(lambda t: sdpa_attn(t, False), x)
    delta_true = causal_leak_test(lambda t: sdpa_attn(t, True), x)

    # Threshold 1e-6: fp16 reduces numerical delta for simple q=k=v test
    eps = 1e-6
    result = {
        "test": "sdpa_causal_leak",
        "delta_is_causal_false": round(delta_false, 8),
        "delta_is_causal_true": round(delta_true, 8),
        "is_causal_false_bidirectional": delta_false > eps,
        "is_causal_true_causal": delta_true < eps,
        "pass": delta_false > eps and delta_true < eps,
    }
    return result


def test_model_forward_causal_leak(
    model_dir: str, device: torch.device
) -> Dict[str, Any]:
    """Load real Qwen3-0.6B and test if forward is causal or bidirectional.

    We feed random input_ids, get logits, perturb last token's embedding,
    and check if position-0 logits change. Since we can't easily hook
    into the internal attention, we test at the input_ids level:
    change input_ids[-1] and see if logits[0] changes.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    seq_len = 32

    # --- Test 1: Default forward (likely causal for Qwen) ---
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        logits_base = model(input_ids=input_ids).logits

    input_ids_mod = input_ids.clone()
    input_ids_mod[0, -1] = (input_ids[0, -1] + 1) % vocab_size

    with torch.no_grad():
        logits_mod = model(input_ids=input_ids_mod).logits

    delta_default = (logits_mod[0, 0, :] - logits_base[0, 0, :]).abs().mean().item()

    # --- Test 2: With full bidirectional attention_mask (all 1s, 2D => no causal mask) ---
    # Qwen3 uses SDPA under the hood; its behavior may depend on
    # input shapes and HF internals.
    attn_mask_full = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_full_base = model(input_ids=input_ids, attention_mask=attn_mask_full).logits
        logits_full_mod = model(input_ids=input_ids_mod, attention_mask=attn_mask_full).logits

    delta_full_mask = (logits_full_mod[0, 0, :] - logits_full_base[0, 0, :]).abs().mean().item()

    # --- Test 3: force_math_sdpa and re-test ---
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    with torch.no_grad():
        logits_math_base = model(input_ids=input_ids).logits
        logits_math_mod = model(input_ids=input_ids_mod).logits

    delta_math_sdpa = (logits_math_mod[0, 0, :] - logits_math_base[0, 0, :]).abs().mean().item()

    del model
    torch.cuda.empty_cache()

    result = {
        "test": "model_forward_causal_leak",
        "model_dir": model_dir,
        "delta_default_forward": round(delta_default, 6),
        "delta_full_attention_mask": round(delta_full_mask, 6),
        "delta_math_sdpa": round(delta_math_sdpa, 6),
        "default_is_causal": delta_default < 1e-4,
        "full_mask_is_causal": delta_full_mask < 1e-4,
        "math_sdpa_is_causal": delta_math_sdpa < 1e-4,
        "notes": (
            "delta~0 => causal (future tokens invisible to position 0). "
            "delta>0 => bidirectional. "
            "For diffusion (WSD stable phase) we WANT bidirectional."
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Test for hidden causal masks")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model_dir
    if not model_dir:
        model_dir = str(Path(__file__).resolve().parents[2] / "models" / "qwen3-0.6b")

    print(f"[attn_mask_test] device={device}")
    print(f"[attn_mask_test] model_dir={model_dir}")
    print()

    results = {}

    # Test 1: SDPA direct
    print("[attn_mask_test] Running SDPA causal leak test...")
    r1 = test_sdpa_causal_leak(device)
    results["sdpa"] = r1
    print(f"  is_causal=False => delta={r1['delta_is_causal_false']:.6f} (bidirectional={r1['is_causal_false_bidirectional']})")
    print(f"  is_causal=True  => delta={r1['delta_is_causal_true']:.6f}  (causal={r1['is_causal_true_causal']})")
    print(f"  PASS: {r1['pass']}")
    print()

    # Test 2: Model forward
    print("[attn_mask_test] Running Qwen3-0.6B forward causal leak test...")
    r2 = test_model_forward_causal_leak(model_dir, device)
    results["model_forward"] = r2
    print(f"  Default forward => delta={r2['delta_default_forward']:.6f} (causal={r2['default_is_causal']})")
    print(f"  Full attn mask  => delta={r2['delta_full_attention_mask']:.6f} (causal={r2['full_mask_is_causal']})")
    print(f"  MATH SDPA       => delta={r2['delta_math_sdpa']:.6f} (causal={r2['math_sdpa_is_causal']})")
    print()

    # Save results
    out_path = Path(args.out_json) if args.out_json else Path(__file__).resolve().parents[2] / "reports" / "attn_mask_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[attn_mask_test] Results saved to {out_path}")


if __name__ == "__main__":
    main()

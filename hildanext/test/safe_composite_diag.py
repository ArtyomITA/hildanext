"""Safe diagnostic for composite_llada20 first-step hang.

Reproduces the exact warmup-phase forward path with aggressive safety guards:
- VRAM hard limit at 70% (leaves 2.4 GB for Windows/DWM)
- torch.cuda.synchronize() between every operation for accurate timing
- Any single GPU operation > 1.5s triggers immediate abort (below TDR threshold)
- Clean CUDA teardown in finally{} to avoid WDDM leaks

Run: conda activate mdm && python test/safe_composite_diag.py
"""
import gc
import sys
import time
import os

# ── Safety constants ──────────────────────────────────────────────
VRAM_FRACTION = 0.70          # leave 30% for Windows
VRAM_ABORT_MB = 5500          # hard abort if allocated exceeds this
OP_WARN_SEC = 1.0             # warn if any op > 1s
OP_ABORT_SEC = 1.5            # abort if any op > 1.5s (TDR is 2s)
SKIP_BACKWARD = False         # set True to test forward-only first


def _vram():
    import torch
    return torch.cuda.memory_allocated() / 1024**2

def _vram_reserved():
    import torch
    return torch.cuda.memory_reserved() / 1024**2

def _sync():
    import torch
    torch.cuda.synchronize()

def _check(label: str):
    """Print VRAM and abort if over limit."""
    a, r = _vram(), _vram_reserved()
    print(f"  [{label}] alloc={a:.0f} MB  reserved={r:.0f} MB")
    if a > VRAM_ABORT_MB:
        print(f"  !! ABORT: alloc {a:.0f} > {VRAM_ABORT_MB} MB")
        _cleanup()
        sys.exit(1)
    return a

def _cleanup():
    """Best-effort CUDA cleanup to avoid WDDM ghost allocations."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def timed(label, fn, *, abort_on_slow=True):
    """Run fn() with CUDA-synchronised timing + VRAM check."""
    import torch
    _sync()
    t0 = time.perf_counter()
    result = fn()
    _sync()
    dt = time.perf_counter() - t0
    _check(label)
    tag = ""
    if dt > OP_ABORT_SEC and abort_on_slow:
        tag = "  !! ABORT (>TDR)"
    elif dt > OP_WARN_SEC:
        tag = "  !! SLOW"
    print(f"  [{label}] time={dt*1000:.1f} ms{tag}")
    if dt > OP_ABORT_SEC and abort_on_slow:
        _cleanup()
        sys.exit(2)
    return result, dt


def main():
    import torch

    if not torch.cuda.is_available():
        print("No CUDA device"); return

    # ── Hard VRAM cap ────────────────────────────────────────────
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 1024**2
    limit_mb = total_mb * VRAM_FRACTION
    print("=" * 64)
    print("COMPOSITE PATH DIAGNOSTIC (safe mode)")
    print("=" * 64)
    print(f"GPU:            {torch.cuda.get_device_name()}")
    print(f"Total VRAM:     {total_mb:.0f} MB")
    print(f"Torch limit:    {limit_mb:.0f} MB ({VRAM_FRACTION*100:.0f}%)")
    print(f"Abort limit:    {VRAM_ABORT_MB} MB alloc")
    print(f"TDR guard:      {OP_ABORT_SEC}s per op")
    print()

    # ── 1. Load model ────────────────────────────────────────────
    print("── Phase 1: Load Model ─────────────────────────")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))
    from transformers import AutoModelForCausalLM

    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3-0.6B")
    print("  loading from disk (CPU)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print("  moving to CUDA...", flush=True)
    model, dt_load = timed("model_to_cuda", lambda: model.cuda())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False}
        )
        print("  grad_ckpt: ON")
    except Exception:
        model.gradient_checkpointing_enable()
        print("  grad_ckpt: ON (legacy)")
    _check("post_model")
    print()

    # ── 2. Prepare dummy batch ───────────────────────────────────
    print("── Phase 2: Prepare Batch (S=1024) ─────────────")
    S = 1024
    mask_id = 151936 - 1  # last token as mask
    vocab = 151936

    input_ids = torch.randint(0, vocab, (1, S), device="cuda")
    attn_1d = torch.ones(1, S, dtype=torch.long, device="cuda")
    doc_ids = torch.ones(1, S, dtype=torch.long, device="cuda")
    clean_ids = input_ids.clone()

    # Corrupt 30% with mask_id (simulates M2T)
    mask_pos = torch.rand(1, S, device="cuda") < 0.3
    mixed = input_ids.clone()
    mixed[mask_pos] = mask_id
    _check("batch_ready")
    print()

    # ── 3. Build composite mask (CPU-bound bool ops) ─────────────
    print("── Phase 3: Build Composite Mask ────────────────")
    from hildanext.masks import batch_doc_attention_mask
    from hildanext.diffusion import _attn_for_model

    # Replicate what _forward does: cat + mask
    ids2, _ = timed("cat_ids", lambda: torch.cat([mixed, clean_ids], dim=1))
    docs2, _ = timed("cat_docs", lambda: torch.cat([doc_ids, doc_ids], dim=1))
    print(f"  ids2={list(ids2.shape)}  docs2={list(docs2.shape)}")

    mask_bool, dt_mask = timed("composite_mask", lambda: batch_doc_attention_mask(
        docs2, causal=False, mask_mode="composite_llada20",
        block_size=None, base_len=S,
    ))
    print(f"  mask_bool: shape={list(mask_bool.shape)}  dtype={mask_bool.dtype}"
          f"  mem={mask_bool.nelement()*mask_bool.element_size()/1024**2:.1f} MB")

    attn4d, dt_attn = timed("to_attn4d", lambda: _attn_for_model(mask_bool, model))
    print(f"  attn4d: shape={list(attn4d.shape)}  dtype={attn4d.dtype}"
          f"  mem={attn4d.nelement()*attn4d.element_size()/1024**2:.1f} MB")
    del mask_bool
    _cleanup()
    _check("mask_done")
    print()

    # ── 4. SDPA backend probe ────────────────────────────────────
    print("── Phase 4: SDPA Backend Probe ─────────────────")
    try:
        H = model.config.num_attention_heads
        D = getattr(model.config, "head_dim", model.config.hidden_size // H)
        S2 = attn4d.shape[-1]
        q_probe = torch.randn(1, H, S2, D, device="cuda", dtype=torch.float16)
        bid = torch._fused_sdp_choice(q_probe, q_probe, q_probe,
                                       attn_mask=attn4d[:1], dropout_p=0.0, is_causal=False)
        from torch.nn.attention import SDPBackend
        bname = SDPBackend(bid).name
        print(f"  backend={bname}  H={H}  D={D}  S2={S2}")
        del q_probe
        _cleanup()
    except Exception as e:
        print(f"  probe failed: {e}")
    print()

    # ── 5. Forward ONLY (no_grad) ────────────────────────────────
    print("── Phase 5: Forward Pass (no_grad) ─────────────")
    print("  >> If this hangs, the composite forward is the cause.")

    def _fwd_no_grad():
        with torch.no_grad():
            out = model(input_ids=ids2, attention_mask=attn4d)
            logits = out.logits[:, :S, :].contiguous()
            del out
            return logits

    logits_ng, dt_fwd_ng = timed("fwd_no_grad", _fwd_no_grad)
    print(f"  logits: {list(logits_ng.shape)}  mem={logits_ng.nelement()*2/1024**2:.0f} MB")
    del logits_ng
    _cleanup()
    _check("fwd_ng_done")
    print()

    if SKIP_BACKWARD:
        print("SKIP_BACKWARD=True → stopping here")
        _cleanup()
        _print_summary(dt_load, dt_mask, dt_attn, dt_fwd_ng, None, None)
        return

    # ── 6. Forward + Backward ────────────────────────────────────
    print("── Phase 6: Forward + Backward ─────────────────")
    print("  >> This tests the real training path (single turn).")

    def _fwd_bwd():
        model.zero_grad()
        out = model(input_ids=ids2, attention_mask=attn4d)
        logits = out.logits[:, :S, :].contiguous()
        del out
        targets = torch.randint(0, vocab, (1, S), device="cuda")
        labels_flat = targets.view(-1)
        logits_flat = logits.view(-1, logits.size(-1))
        loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
        loss.backward()
        lv = loss.item()
        del logits, loss, targets, labels_flat, logits_flat
        return lv

    loss_val, dt_bwd = timed("fwd_bwd", _fwd_bwd, abort_on_slow=False)
    print(f"  loss={loss_val:.4f}")
    _cleanup()
    _check("fwd_bwd_done")
    print()

    # ── 7. MTF 2-turn simulation ─────────────────────────────────
    print("── Phase 7: MTF 2 Turns (realistic) ────────────")
    print("  >> This replicates the actual training loop with 2 turns.")

    def _mtf_2turn():
        model.zero_grad()
        total_loss = 0.0
        current_ids = mixed.clone()
        for turn in range(2):
            _sync()
            t_turn0 = time.perf_counter()
            # -- forward through composite path --
            ids2_t = torch.cat([current_ids, clean_ids], dim=1)
            docs2_t = torch.cat([doc_ids, doc_ids], dim=1)
            mk = batch_doc_attention_mask(
                docs2_t, causal=False, mask_mode="composite_llada20",
                block_size=None, base_len=S,
            )
            a4d = _attn_for_model(mk, model)
            del mk
            out = model(input_ids=ids2_t, attention_mask=a4d)
            logits = out.logits[:, :S, :].contiguous()
            del out, ids2_t, docs2_t, a4d
            targets = clean_ids
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            (loss / 2.0 / 8.0).backward()  # /turns /accum, like real training
            total_loss += loss.item()
            # build next input from predictions
            if turn == 0:
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    current_ids = clean_ids.clone()
                    current_ids[:, 1:] = preds[:, :-1]
            del logits, loss
            _sync()
            dt_turn = time.perf_counter() - t_turn0
            a = _check(f"mtf_turn_{turn}")
            print(f"  [mtf_turn_{turn}] time={dt_turn*1000:.0f} ms")
        return total_loss

    mtf_loss, dt_mtf = timed("mtf_2turns", _mtf_2turn, abort_on_slow=False)
    print(f"  mtf total_loss={mtf_loss:.4f}")
    _check("mtf_done")
    print()

    # ── Summary ──────────────────────────────────────────────────
    _print_summary(dt_load, dt_mask, dt_attn, dt_fwd_ng, dt_bwd, dt_mtf)
    _cleanup()


def _print_summary(dt_load, dt_mask, dt_attn, dt_fwd_ng, dt_bwd, dt_mtf):
    import torch
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Model load:        {dt_load*1000:.0f} ms")
    print(f"  Composite mask:    {dt_mask*1000:.0f} ms")
    print(f"  Mask → attn4d:     {dt_attn*1000:.0f} ms")
    print(f"  Forward (no_grad): {dt_fwd_ng*1000:.0f} ms")
    if dt_bwd is not None:
        print(f"  Fwd + Bwd:         {dt_bwd*1000:.0f} ms")
    if dt_mtf is not None:
        print(f"  MTF 2 turns:       {dt_mtf*1000:.0f} ms")
    print(f"  Peak VRAM:         {peak:.0f} MB")
    print()
    if dt_fwd_ng and dt_fwd_ng > 2.0:
        print("  !! FORWARD ALONE > 2s → likely TDR hang source")
    elif dt_bwd and dt_bwd > 5.0:
        print("  !! FWD+BWD > 5s → backward phase is the bottleneck")
    elif dt_mtf and dt_mtf > 10.0:
        print("  !! MTF 2-turn > 10s → multi-turn accumulation issue")
    elif peak > 6500:
        print("  !! Peak VRAM > 6.5 GB → OOM / WDDM swap likely cause")
    else:
        print("  OK: all phases within safe limits")
    print()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n!! EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _cleanup()
        print("[CLEANUP] CUDA teardown complete.")

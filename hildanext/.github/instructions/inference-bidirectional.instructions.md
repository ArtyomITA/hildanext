---
description: "Use when editing inference decode loop, bidirectional attention, diffusion forward, attention masks, or LLaDA formulas. Covers force_noncausal_attention, diagnostic_dllm_decode, threshold-edit policies, composite masks, and critical off-by-one pitfalls."
applyTo: "backend/src/hildanext/{inference,diffusion,masks,formulas,config}.py"
---

# Inference & Bidirectional Attention — Deep Reference

> Canonical source: code in `backend/src/hildanext/`. For design rationale see [BIDIRECTIONAL_ATTENTION_HANDOFF.md](../../docs/BIDIRECTIONAL_ATTENTION_HANDOFF.md).

---

## 1. Bidirectional Attention — Two Pathways, Both Neutralized

Qwen3 has **two** causal pathways on transformers 4.57.3:

| # | Pathway | Where | Neutralized by |
|---|---------|-------|----------------|
| 1 | `Qwen3Attention.is_causal = True` | Every `self_attn` layer | `force_noncausal_attention` sets `is_causal=False` |
| 2 | `create_causal_mask()` builds lower-triangular | `Qwen3Model.forward()` | **4D** attention mask triggers early-exit, bypassing causal generation |

**On transformers 4.57**: when a 4D mask is provided, `_preprocess_mask_arguments` returns it as-is (early-exit) → pathway 2 is already neutralized. Pathway 1 also: SDPA checks `attention_mask is not None` → `is_causal=False`.

**Therefore**: the 4D composite mask alone makes the model bidirectional. `force_noncausal_attention` is a **safety net** for paths without 4D mask (inference via `_predict_bidirectional`) or future transformers versions.

### `force_noncausal_attention(model)` — Exact Mechanism

```python
# 1. Flip is_causal=False on all attention sub-modules
for name, mod in model.named_modules():
    if hasattr(mod, "is_causal"):
        mod.is_causal = False

# 2. Monkey-patch create_causal_mask in 3 modules:
#    - transformers.masking_utils
#    - transformers.models.qwen3.modeling_qwen3
#    - transformers.models.qwen2.modeling_qwen2
def _noncausal_mask(*args, **kwargs):
    result = _orig(*args, **kwargs)
    if result is None: return None
    # Identity check: if transformers returned user's 4D mask unchanged → preserve it
    if user_mask is not None and result is user_mask:
        return result       # composite block-diagonal mask passes through
    # Otherwise: fresh causal mask → zero it out (fully permissive)
    return torch.zeros_like(result) if isinstance(result, torch.Tensor) else result
```

### Critical Pitfalls

- **Identity check (`result is user_mask`)**: relies on Python object identity. If a transformers update copies the tensor before returning, composite masks get zeroed → cross-document attention leak.
- **Returning `None` instead of `zeros_like`**: causes `causal_mask_mapping["full_attention"] = None` → `attention_mask=None` at every layer → doc-gating leak. This was a real bug found and fixed.
- **Not thread-safe**: patches global module attributes. Safe only because `serialize_inference=True` enforces a single concurrent forward.

---

## 2. Decode Loop — `diagnostic_dllm_decode()` Algorithm

```
1. encode(prompt) → input_ids [1, P]
2. seq = [input_ids | mask_id × max_new]   shape [1, P + max_new]
3. num_blocks = ⌈max_new / block_size⌉
   steps_per_block = max(1, max_steps // num_blocks)
4. for block_idx in 0..num_blocks-1:
     for bstep in 0..steps_per_block-1:
       A. pred, conf = _predict_bidirectional(model, seq, P, max_new, ...)
            → full-sequence forward, reads gen logits at logits[:, P-1 : P-1+max_new]
       B. Extract block slice of pred/conf
       C. Apply decode policy → updated tokens
       D. Write back to seq
       E. Check stop guards: EOS → cycle → plateau
       F. Optional tau degeneration fallback
       G. if remain_masks == 0: converged, break
5. Decode, truncate at EOS, return text + stats
```

### Left-Shift Alignment (CRITICAL)

```python
gen_logits = logits[:, prompt_len - 1 : prompt_len - 1 + max_new, :]
```

The `-1` is essential: logit at position `j` predicts token at `j+1`. Forgetting it silently reads wrong logits with no error.

### `_predict_bidirectional()` — Confidence vs Selection

- **Clean confidence**: `softmax(logits)` → probability of selected token
- **Noisy selection**: `argmax(logits + Gumbel(temperature))` → token ID
- At `temperature=0.0` (default): Gumbel noise is zero → deterministic argmax
- Returns: `pred [1, max_new]` (long), `conf [1, max_new]` (float32)

---

## 3. Decode Policies

Controlled by `cfg.runtime.dllm_base_policy` (default `"shadow_llada21_cap"`).

| Policy | τ_mask | τ_edit | Δ edits | Remask | Budget top-up |
|--------|--------|--------|---------|--------|---------------|
| `current_base` | ✗ | ✗ | ✗ | ✗ | top-k of masked by confidence |
| `threshold_cap` | ✓ | ✗ | ✗ | ✗ | top-k fallback if γ < budget |
| `shadow_llada21` | ✓ | ✓ | ✓ | optional | ✗ |
| `shadow_llada21_cap` | ✓ | ✓ | ✓ | optional | ✓ top-up masked to fill budget |
| `shadow_llada21_delayed_edit` | ✓ | ✓ | ✓ (step≥2) | optional | ✓ |

**Γ set** = masked positions with `confidence ≥ tau_mask` (unmask)
**Δ set** = non-masked positions where `pred ≠ current AND confidence ≥ tau_edit` (re-edit)

Computed by `llada21_sets()` in formulas.py.

---

## 4. Effort Knob

```python
_EFFORT_PARAMS = {
    "instant":  {"max_steps": 1,   "tau_scale": 2.0},
    "low":      {"max_steps": 16,  "tau_scale": 1.5},
    "medium":   {"max_steps": 64,  "tau_scale": 1.0},
    "high":     {"max_steps": 128, "tau_scale": 0.7},
    "adaptive": {"max_steps": 256, "tau_scale": 1.0},
}
```

`_resolve_effort(effort, cfg_steps, tau_mask, tau_edit)` → `(steps, tau_mask, tau_edit)`.
τ values are scaled by `tau_scale`, clamped to `[0.0, 1.0]`.

> **Note**: DESIGN.md has stale effort-knob values. `inference.py` is the canonical source.

---

## 5. Stop Guards (3 independent)

1. **EOS** (`eos_guard_enabled`): eos_token_id in gen region → truncate at first EOS.
2. **Cycle** (`cycle_guard_enabled`): `hash(tuple(tokens))` repeated with `changed==0` → stop. Cost scales with `max_new`.
3. **Plateau**: `changed==0` for `plateau_patience` consecutive steps with `remain > 0` → stop.

**Tau degeneration fallback** (off by default): if no-update streak ≥ `degenerate_patience`, scale taus by 0.85, floor at 0.05.

---

## 6. Composite Mask — `_composite_llada20_mask`

Sequence structure: `[xt₁…xt_L | x0₁…x0_L]` (noisy | clean), `total = 2L`.

| Query in | Key in | Condition |
|----------|--------|-----------|
| xt | xt | same block — bidirectional within block |
| xt | x0 | `block(xt) > block(x0)` — noisy sees earlier clean |
| x0 | x0 | `block(x0) ≥ block(x0')` — clean sees same/earlier clean |
| x0 | xt | **never** |

All rules AND with `same_doc(i,j)` + `doc_id ≥ 0` (valid position). Output: 4D additive float mask (`-inf` blocked, `0` visible).

---

## 7. Config Fields That Matter for Inference

### RuntimeConfig (decode control)

| Field | Default | Note |
|-------|---------|------|
| `dllm_base_policy` | `shadow_llada21_cap` | Active policy |
| `dllm_base_use_bidirectional` | **False** | **This is the actual switch** for bidirectional inference |
| `dllm_base_use_remask` | False | |
| `dllm_shadow_enabled` | **False** | Shadow decode (doubles cost) |
| `serialize_inference` | True | Required for thread-safety of monkey-patch |

### ExperimentConfig

| Field | Default | Note |
|-------|---------|------|
| `attention_mode` | `bidirectional_only_stable` | **Training only** — decode loop ignores this |
| `effort` | `medium` | Maps to steps + tau via `_EFFORT_PARAMS` |

**Pitfall**: `ExperimentConfig.attention_mode` controls training phase selection in `training.py`, NOT the inference path. For inference bidirectionality, **only** `RuntimeConfig.dllm_base_use_bidirectional` matters.

### Dead Fields (defined but unused)

- `RemaskConfig.percentile_safety`, `block_size`, `block_stride` — `apply_remask()` reads only `target_ratio` and `min_ratio`
- `InferenceConfig.strict_decode_invariants` — no assertions check it
- `_predict_autoregressive_candidates_*()` — two AR helpers exist but are NOT called from the main decode path

---

## 8. Remask — `apply_remask()`

```python
cand = (tokens != mask_id)  # decoded positions
k = max(min_ratio * total, target_ratio * total)
# re-MASK the k positions with LOWEST confidence among decoded
```

Used in `shadow_llada21*` policies when remask is enabled. Applied between intermediate steps, never on the last step.

---

## 9. M2T/T2T Loss (Training Forward)

- **M2T**: positions replaced with `mask_id` at ratio `t`, CE with `1/t` weighting
- **T2T**: positions replaced with random different tokens at `t2t_noise_ratio`, plain CE
- **Disjoint sets**: M2T and T2T positions never overlap
- **`_causal_loss()` optimization**: when <50% positions valid, gathers only valid logits → ~6× memory saving at 15% mask ratio for V=151936
- **Left-shift**: `logits[:, :-1]` predicts `labels[:, 1:]` (`preserve_left_shift`)

---

## 10. Rules When Editing These Files

1. **Never break the engine fallback chain** in `build_engine()` — every fallback must be logged via `trace`.
2. **Never re-enable `use_cache=True`** — incompatible with full-sequence dLLM forward.
3. **Never return `None` from the `create_causal_mask` patch** — causes doc-gating leak.
4. **Never change the left-shift alignment** (`prompt_len - 1`) without understanding the off-by-one.
5. **Keep `force_noncausal_attention` idempotent** — it saves/restores state in `finally`.
6. **Test bidirectional changes** with the runtime test: `tools/tests_wsd/test_bidirectional_composite_runtime.py`.
7. **If upgrading transformers**: verify that the 4D-mask early-exit in `_preprocess_mask_arguments` still works. Check that `result is user_mask` identity holds.
8. **All new config parameters** go in dataclasses in `config.py` — never hard-code values.

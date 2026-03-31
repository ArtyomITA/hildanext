# RCD (Residual Context Diffusion) inference engine.
# Paper: "Residual Context Diffusion Language Models" (Hu et al., 2026)
# arXiv:2601.22954  —  Algorithm 2, Eq.(1-3), Section 3.1/3.3.
#
# DEVIATIONS FROM PAPER (inference-only, no RCD-trained weights):
#   1. No two-stage RCD training was performed on the model checkpoint.
#      The model is a standard dLLM; quality gains will be smaller than
#      paper Tables 1-2 which use RCD-trained weights.
#   2. Warm-start: paper uses a dedicated smaller reference model (Mref).
#      Default here falls back to the same target model (approximate).
#   3. External inputs_embeds injection: paper's custom models have
#      RCD-native forward(). We use HF inputs_embeds interface.
#
# RCD residual source = probability distribution projected through the
# model's own input embedding codebook.  NOT raw hidden states.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math as _math
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic request schema
# ---------------------------------------------------------------------------
class InferenceRCDMRequest(BaseModel):
    prompt: str
    mode: str = "S_MODE"
    tau_mask: Optional[float] = None
    tau_edit: Optional[float] = None
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    seed: Optional[int] = None
    effort: str = Field(default="medium")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=2000)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    enable_thinking: Optional[bool] = None
    # ---- RCD-specific knobs (all optional, sane defaults) ----
    rcd_alpha_mode: str = Field(default="normalized_entropy", description="normalized_entropy (Eq.3)")
    rcd_temperature_residual: float = Field(default=1.0, ge=0.01, le=10.0, description="T_res for entropy alignment (Sec 3.3)")
    rcd_store_step_diagnostics: bool = Field(default=True)
    rcd_force_mask_only_injection: bool = Field(default=True, description="inject residual only on [MASK] positions (Eq.2)")
    rcd_warm_start: bool = Field(default=True, description="warm-start first step (Sec 3.3)")
    rcd_reference_model: Optional[str] = Field(default=None, description="path to separate Mref checkpoint")
    rcd_same_model_warm_start_fallback: bool = Field(default=True, description="use target model if no Mref")


# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------
@dataclass
class RCDStepDiagnostics:
    step: int
    committed_count: int
    tokens_per_step: float
    avg_alpha: float
    avg_residual_norm: float
    masked_remaining: int
    elapsed_ms: float


@dataclass
class RCDInferenceDiagnostics:
    steps: List[RCDStepDiagnostics] = field(default_factory=list)
    warm_start_used: bool = False
    reference_model_used: bool = False
    t_res: float = 1.0
    total_denoising_steps: int = 0
    total_elapsed_ms: float = 0.0
    finish_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "committed_count": s.committed_count,
                    "tokens_per_step": s.tokens_per_step,
                    "avg_alpha": s.avg_alpha,
                    "avg_residual_norm": s.avg_residual_norm,
                    "masked_remaining": s.masked_remaining,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
            "warm_start_used": self.warm_start_used,
            "reference_model_used": self.reference_model_used,
            "t_res": self.t_res,
            "total_denoising_steps": self.total_denoising_steps,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
        }


# ---------------------------------------------------------------------------
# Pure helper functions — paper-faithful math
# ---------------------------------------------------------------------------

def compute_normalized_entropy(probs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Eq. (3): α_i = −Σ_j p_{i,j} log p_{i,j} / log(V).
    Input: probs shape (..., V), already a valid probability distribution.
    Output: α shape (...) in [0, 1].
    """
    assert probs.shape[-1] == vocab_size, f"probs last dim {probs.shape[-1]} != vocab_size {vocab_size}"
    log_v = _math.log(vocab_size)
    # clamp to avoid log(0)
    p = probs.clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=-1)  # H(x)
    alpha = entropy / log_v
    return alpha.clamp(0.0, 1.0)


def compute_rcd_residuals_from_probs(probs: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Eq. (1): Δ_i = Σ_j p_{i,j} · E_{j,:}.
    probs: (B, S, V)  —  probability distributions per position.
    embedding_weight: (V, D)  —  model input embedding codebook.
    Returns: (B, S, D) residual vectors.
    NOTE: This uses probability-weighted embedding sums, NOT raw hidden states.
    """
    # probs @ E  ->  (B, S, V) @ (V, D)  ->  (B, S, D)
    return torch.matmul(probs, embedding_weight)


def build_rcd_inputs_embeds(
    input_ids: torch.Tensor,
    mask_id: int,
    embedding_layer: torch.nn.Module,
    alpha_prev: torch.Tensor,
    delta_prev: torch.Tensor,
    force_mask_only: bool = True,
) -> torch.Tensor:
    """Eq. (2): Build ẽ_i for the next denoising step.
    For masked positions:
        ẽ_i = (1 - α_i^{prev}) · E(x_i) + α_i^{prev} · Δ_i^{prev}
    For committed (unmasked) positions:
        ẽ_i = E(x_i)
    input_ids: (B, S)
    alpha_prev: (B, S) in [0,1]
    delta_prev: (B, S, D) residual embeddings from previous step
    Returns: (B, S, D) input embeddings for model forward
    """
    base_embeds = embedding_layer(input_ids)  # (B, S, D)
    if not force_mask_only:
        # Apply to all positions (non-standard)
        alpha_exp = alpha_prev.unsqueeze(-1)  # (B, S, 1)
        return (1.0 - alpha_exp) * base_embeds + alpha_exp * delta_prev

    # Paper-faithful: inject only on [MASK] positions
    is_mask = (input_ids == mask_id).unsqueeze(-1)  # (B, S, 1)
    alpha_exp = alpha_prev.unsqueeze(-1)  # (B, S, 1)
    interpolated = (1.0 - alpha_exp) * base_embeds + alpha_exp * delta_prev
    # Where mask: use interpolated; where committed: use base_embeds
    return torch.where(is_mask, interpolated, base_embeds)


def initialize_rcd_warm_start(
    model: Any,
    input_ids: torch.Tensor,
    mask_id: int,
    embedding_weight: torch.Tensor,
    vocab_size: int,
    t_res: float = 1.0,
    force_noncausal_ctx: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warm-start: run one forward pass to get initial (α₀, Δ₀).
    Paper Section 3.3: "At the very first denoising step, no previous
    residual exists. To jump-start ... invoke Mref once."
    Here we use the provided model (may be target model as fallback).
    Returns: (alpha_0, delta_0) with shapes (B, S) and (B, S, D).
    """
    model.eval()
    with torch.inference_mode():
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(input_ids=input_ids)
        else:
            out = model(input_ids=input_ids)
    logits = out.logits  # (B, S, V)
    # Temperature-scaled softmax for residual probs (Sec 3.3)
    probs = F.softmax(logits.float() / max(t_res, 1e-6), dim=-1)
    alpha_0 = compute_normalized_entropy(probs, vocab_size)
    delta_0 = compute_rcd_residuals_from_probs(probs, embedding_weight)
    return alpha_0, delta_0


# ---------------------------------------------------------------------------
# RCD decode loop
# ---------------------------------------------------------------------------

def rcd_decode(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    mask_id: int,
    vocab_size: int,
    prompt: str,
    max_new_tokens: int = 256,
    tau_mask: float = 0.3,
    tau_edit: float = 0.5,
    max_steps: int = 10,
    t_res: float = 1.0,
    alpha_mode: str = "normalized_entropy",
    force_mask_only: bool = True,
    warm_start: bool = True,
    warm_start_model: Any = None,
    store_diagnostics: bool = True,
    seed: int = 42,
    is_dummy: bool = False,
    force_noncausal_ctx: Any = None,
) -> Tuple[str, Dict[str, Any], RCDInferenceDiagnostics]:
    """Full RCD inference loop (Algorithm 2).
    Returns: (text, stats_dict, diagnostics).
    """
    from .formulas import llada21_apply
    from .diffusion import apply_remask
    from .config import RemaskConfig
    from .utils import seed_everything, tokens_per_second

    seed_everything(seed)
    diag = RCDInferenceDiagnostics(t_res=t_res)

    # Encode prompt
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])

    # Build sequence: [prompt tokens] [MASK * max_new_tokens]
    seq = torch.full((1, prompt_len + max_new_tokens), mask_id, dtype=torch.long, device=device)
    seq[:, :prompt_len] = input_ids

    # Get embedding layer reference
    embed_layer = _get_embedding_layer(model)
    embed_weight = embed_layer.weight.detach()  # (V, D) — the codebook E

    # Clamp vocab_size to embedding table size
    actual_v = embed_weight.shape[0]
    effective_v = min(vocab_size, actual_v)

    # Initialize residual state buffers
    S = seq.shape[1]
    D = embed_weight.shape[1]
    alpha_prev = torch.zeros(1, S, device=device, dtype=torch.float32)
    delta_prev = embed_layer(seq).detach()  # initialize to base embeddings

    # ---- Warm-start (Algorithm 2, lines 2-6) ----
    ws_model = warm_start_model if warm_start_model is not None else model
    reference_model_used = warm_start_model is not None
    if warm_start:
        alpha_prev, delta_prev = initialize_rcd_warm_start(
            ws_model, seq, mask_id, embed_weight, effective_v,
            t_res=t_res, force_noncausal_ctx=force_noncausal_ctx,
        )
        diag.warm_start_used = True
        diag.reference_model_used = reference_model_used

    remask_cfg = RemaskConfig()
    model.eval()
    t0 = time.time()
    converged = False
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            eos_token_id = int(eos_token_id)
        except Exception:
            eos_token_id = None

    finish_reason = "length"
    eos_cut_idx: Optional[int] = None

    with torch.inference_mode():
        for step in range(max_steps):
            step_t0 = time.time()

            # ---- Build input embeddings with RCD residuals (Eq.2) ----
            inputs_embeds = build_rcd_inputs_embeds(
                seq, mask_id, embed_layer, alpha_prev, delta_prev,
                force_mask_only=force_mask_only,
            )

            # ---- Forward pass with inputs_embeds ----
            if force_noncausal_ctx is not None:
                with force_noncausal_ctx(model):
                    out = model(inputs_embeds=inputs_embeds)
            else:
                out = model(inputs_embeds=inputs_embeds)

            logits = out.logits  # (B, S, V)

            # ---- Prediction: confidence + top1 from generation region ----
            gen_logits = logits[:, prompt_len:, :]
            gen_probs = F.softmax(gen_logits.float(), dim=-1)
            confidence, pred_ids = gen_probs.max(dim=-1)  # (B, S_gen)

            gen_before = seq[:, prompt_len:]
            masked_before = gen_before.eq(mask_id)

            # ---- Selection + Update via existing llada21_apply ----
            updated, sets = llada21_apply(
                gen_before, pred_ids, confidence,
                mask_id, tau_mask, tau_edit,
            )

            # Remask on non-final steps
            if step + 1 < max_steps:
                updated = apply_remask(updated, confidence, mask_id, remask_cfg)

            seq[:, prompt_len:] = updated

            # ---- Compute new residual state for next step (Algorithm 2, lines 22-26) ----
            # Use temperature-scaled probs for entropy alignment (Sec 3.3)
            full_logits = logits  # (B, S, V)
            res_probs = F.softmax(full_logits.float() / max(t_res, 1e-6), dim=-1)
            # α from temperature-scaled distribution (Eq.3 + Sec 3.3)
            alpha_prev = compute_normalized_entropy(res_probs, effective_v)
            # Δ from original probs projected through codebook (Eq.1)
            # Paper uses p_{i,j}^{t_k} (not temperature-scaled) for the residual vector
            orig_probs = F.softmax(full_logits.float(), dim=-1)
            delta_prev = compute_rcd_residuals_from_probs(orig_probs, embed_weight)

            # ---- Diagnostics ----
            remain = int(updated.eq(mask_id).sum().item())
            committed = int(sets.gamma_count) + int(sets.delta_count)
            step_elapsed = (time.time() - step_t0) * 1000.0

            if store_diagnostics:
                gen_alpha = alpha_prev[:, prompt_len:]
                gen_delta = delta_prev[:, prompt_len:, :]
                diag.steps.append(RCDStepDiagnostics(
                    step=step + 1,
                    committed_count=committed,
                    tokens_per_step=float(committed) if committed > 0 else 0.0,
                    avg_alpha=float(gen_alpha.mean().item()),
                    avg_residual_norm=float(gen_delta.norm(dim=-1).mean().item()),
                    masked_remaining=remain,
                    elapsed_ms=step_elapsed,
                ))

            # ---- Stop conditions ----
            if eos_token_id is not None:
                eos_pos = (updated[0] == eos_token_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    eos_cut_idx = int(eos_pos[0][0].item())
                    finish_reason = "eos"
                    converged = True
                    break

            if remain == 0:
                if sets.delta_count == 0:
                    finish_reason = "converged"
                    converged = True
                    break
                finish_reason = "converged"

    elapsed = max(1e-6, time.time() - t0)
    out_ids = seq[0, prompt_len:]
    if eos_cut_idx is not None:
        out_ids = out_ids[:max(0, eos_cut_idx + 1)]

    tokens_generated = int((out_ids != mask_id).sum().item())
    text = _decode_text_rcd(tokenizer, out_ids, mask_id, is_dummy)

    if not converged:
        finish_reason = "length"

    diag.total_denoising_steps = len(diag.steps)
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason

    stats = {
        "engine": "rcd",
        "mode": "RCD",
        "tau_mask": tau_mask,
        "tau_edit": tau_edit,
        "steps": len(diag.steps),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_second(tokens_generated, elapsed),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "dummy_model": is_dummy,
        "rcd_diagnostics": diag.to_dict(),
    }

    return text.strip(), stats, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embedding_layer(model: Any) -> torch.nn.Module:
    """Extract the input embedding layer from a HuggingFace model or TinyCausalLM."""
    # TinyCausalLM uses .embed
    if hasattr(model, "embed") and isinstance(model.embed, torch.nn.Embedding):
        return model.embed
    # HuggingFace models: model.model.embed_tokens or model.transformer.wte
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    # Fallback: get_input_embeddings()
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    raise RuntimeError("Cannot find input embedding layer on model")


def _decode_text_rcd(tok: Any, out_ids: torch.Tensor, mask_id: int, is_dummy: bool) -> str:
    text = ""
    if hasattr(tok, "decode"):
        try:
            text = tok.decode(out_ids, skip_special_tokens=True)
        except Exception:
            text = ""
    if text and text.strip():
        t = text.strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    raw = [int(x) for x in out_ids.detach().cpu().tolist() if int(x) != int(mask_id)]
    if raw:
        t = " ".join(f"tok{x}" for x in raw[:64]).strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    return "[DUMMY] dummy-output" if is_dummy else ""

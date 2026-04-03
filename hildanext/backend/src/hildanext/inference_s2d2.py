# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation.
# Paper: "S2D2" (Han, Wang, Gao, Xu, Srivastava, 2026)
# arXiv:2603.25702  — Algorithm 3, Algorithm 4, Sections 4.1–4.3.
#
# KEY IDEA: Reuse the SAME block-diffusion model in two roles:
#   - Standard block-diffusion decoding acts as the DRAFTER
#   - The same model with block-size-1 (AR mode) acts as the VERIFIER
# Verification is applied ONLY to the first contiguous masked span.
# Accepted tokens are committed through speculative rejection sampling.
# If verification is skipped or ends early via routing, decoding falls
# back to ordinary confidence-threshold diffusion decoding.
#
# NOT equivalent to global AR decoding (paper Section 4.4):
#   - Verification is local, not global
#   - Routing can skip verification
#   - Rejection stops the speculative segment
#   - Drafting & cache updates remain under block-diffusion attention
#
# POSITION-ALIGNED IMPLEMENTATION (LLaDA/Qwen-dLLM family):
#   Uses the "2L trick" from Section 4.2 / Eq.(3) for verifier mask.
#
# *da non cancellare* — FAST-dLLM v2 TODO:
#   Quando integreremo Fast-dLLM v2 (right-shifted model):
#   1. Sub-block decoding (SB) va aggiunto: B fisso + SB variabile (Tab.2)
#   2. build_block_draft_attention_mask va esteso con Eq.(4) paper:
#      split committed prefix (causal) vs masked suffix (full) DENTRO il blocco
#   3. Draft KV cache cross-block (Alg.1 lines 5-7) per velocità reale
#   4. Ratio tempering γ in acceptance (Appendix A.8): (q_i/p_i)^γ
#   5. UCB contextual bandit routing (Appendix A.6)
#   6. Il verifier per right-shifted NON usa 2L trick: la causal mask
#      standard basta già (Fig.1(c) paper)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math as _math
import time

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class InferenceS2D2Request(BaseModel):
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
    # ---- S2D2-specific knobs (all follow paper Sec. 4.1–4.3) ----
    s2d2_block_size: int = Field(default=32, ge=1, le=512, description="Block size B for drafting")
    s2d2_denoising_steps: int = Field(default=0, ge=0, le=256, description="T denoising steps per block; 0=auto from effort")
    s2d2_routing_policy: str = Field(
        default="min_span",
        description="Routing policy: min_span | score_threshold | hysteresis | always | never",
    )
    s2d2_min_verify_span: int = Field(default=2, ge=1, le=512, description="tau_span for min-span policy (Alg.4)")
    s2d2_score_threshold: float = Field(default=0.0, description="tau_score for score-threshold policy")
    s2d2_score_cost: float = Field(default=1.0, ge=0.0, le=10.0, description="c — cost hyperparameter in Eq.(6)")
    s2d2_score_mode: str = Field(default="static", description="static | dynamic for Eq.(6)")
    s2d2_hysteresis_on: float = Field(default=1.0, description="tau_on for hysteresis policy")
    s2d2_hysteresis_off: float = Field(default=-5.0, description="tau_off for hysteresis policy")
    s2d2_acceptance_estimator: str = Field(
        default="entropy",
        description="Token acceptance estimator: entropy | margin (Sec 4.3, Eq.5)",
    )
    s2d2_entropy_beta: float = Field(default=1.0, ge=0.01, le=10.0, description="beta for entropy estimator alpha_i = exp(-beta H_tilde_i)")
    s2d2_margin_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="tau_margin for margin estimator")
    s2d2_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="tau — confidence threshold for diffusion fallback (Alg.2/3)")
    s2d2_store_diagnostics: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class S2D2StepDiag:
    step: int
    block_index: int
    block_start: int
    block_end: int
    masked_before: int
    contiguous_span_len: int
    span_start: int
    span_end: int
    route_score: float
    verify_invoked: bool
    verifier_mode: str
    verifier_input_tokens: int
    verifier_cache_used: bool
    accepted_count: int
    rejection_position: int
    committed_count: int
    masked_after: int
    elapsed_ms: float


@dataclass
class S2D2VerifierInputs:
    input_ids: torch.Tensor
    query_start: int
    query_end: int
    attention_mask_bool: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    verifier_mode: str = "position_aligned_2l"


@dataclass
class S2D2Diagnostics:
    steps: List[S2D2StepDiag] = field(default_factory=list)
    total_denoising_steps: int = 0
    total_elapsed_ms: float = 0.0
    finish_reason: str = ""
    verifier_invocations: int = 0
    verifier_skips: int = 0
    accepted_prefix_lengths: List[int] = field(default_factory=list)
    avg_accepted_prefix_length: float = 0.0
    routing_policy_used: str = "min_span"
    acceptance_estimator_used: str = "entropy"
    fallback_to_diffusion_count: int = 0
    verifier_mask_path: str = "position_aligned_2l"
    verifier_mode_used: str = "position_aligned_2l"
    block_size: int = 32
    kv_cache_mode: str = "disabled"
    verifier_cache_prefills: int = 0
    verifier_cache_hits: int = 0
    last_route_score: float = 0.0
    last_block_index: int = -1
    last_span_start: int = -1
    last_span_end: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "block_index": s.block_index,
                    "block_start": s.block_start,
                    "block_end": s.block_end,
                    "masked_before": s.masked_before,
                    "contiguous_span_len": s.contiguous_span_len,
                    "span_start": s.span_start,
                    "span_end": s.span_end,
                    "route_score": s.route_score,
                    "verify_invoked": s.verify_invoked,
                    "verifier_mode": s.verifier_mode,
                    "verifier_input_tokens": s.verifier_input_tokens,
                    "verifier_cache_used": s.verifier_cache_used,
                    "accepted_count": s.accepted_count,
                    "rejection_position": s.rejection_position,
                    "committed_count": s.committed_count,
                    "masked_after": s.masked_after,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
            "total_denoising_steps": self.total_denoising_steps,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
            "verifier_invocations": self.verifier_invocations,
            "verifier_skips": self.verifier_skips,
            "accepted_prefix_lengths": self.accepted_prefix_lengths,
            "avg_accepted_prefix_length": self.avg_accepted_prefix_length,
            "routing_policy_used": self.routing_policy_used,
            "acceptance_estimator_used": self.acceptance_estimator_used,
            "fallback_to_diffusion_count": self.fallback_to_diffusion_count,
            "verifier_mask_path": self.verifier_mask_path,
            "verifier_mode_used": self.verifier_mode_used,
            "block_size": self.block_size,
            "kv_cache_mode": self.kv_cache_mode,
            "verifier_cache_prefills": self.verifier_cache_prefills,
            "verifier_cache_hits": self.verifier_cache_hits,
            "last_route_score": self.last_route_score,
            "last_block_index": self.last_block_index,
            "last_span_start": self.last_span_start,
            "last_span_end": self.last_span_end,
        }


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------
def find_first_contiguous_mask_span(
    seq: torch.Tensor,
    mask_id: int,
    prompt_len: int,
    region_end: Optional[int] = None,
) -> Tuple[int, int]:
    """Find the first contiguous [MASK] span in seq[prompt_len:region_end]."""
    end = int(region_end) if region_end is not None else int(seq.shape[1])
    if end <= prompt_len:
        return -1, -1
    region = seq[0, prompt_len:end]
    is_mask = region.eq(mask_id)
    if not is_mask.any():
        return -1, -1
    nonzero = is_mask.nonzero(as_tuple=False).view(-1)
    if nonzero.numel() == 0:
        return -1, -1
    start_local = int(nonzero[0].item())
    end_local = start_local + 1
    while end_local < region.shape[0] and bool(is_mask[end_local]):
        end_local += 1
    return prompt_len + start_local, prompt_len + end_local


def estimate_expected_accept_prefix(
    draft_logits: torch.Tensor,
    span_start: int,
    span_end: int,
    estimator: str = "entropy",
    entropy_beta: float = 1.0,
    margin_threshold: float = 0.1,
    vocab_size: int = 1,
) -> float:
    """Estimate expected accepted prefix length K_hat via Eq.(5)."""
    span_logits = draft_logits[0, span_start:span_end, :]
    L = span_logits.shape[0]
    if L == 0:
        return 0.0
    probs = F.softmax(span_logits.float(), dim=-1)

    if estimator == "margin":
        top2_vals, _ = probs.topk(min(2, probs.shape[-1]), dim=-1)
        if top2_vals.shape[-1] >= 2:
            margins = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            margins = torch.ones(L, device=probs.device)
        alphas = (margins >= margin_threshold).float()
    else:
        log_v = _math.log(max(vocab_size, 2))
        p_clamped = probs.clamp(min=1e-12)
        H = -(p_clamped * p_clamped.log()).sum(dim=-1)
        H_tilde = H / log_v
        alphas = torch.exp(-entropy_beta * H_tilde)

    return float(torch.cumprod(alphas, dim=0).sum().item())


def compute_verification_score(
    k_hat: float,
    cost: float,
    n_hi: int = 0,
    mode: str = "static",
) -> float:
    """Eq.(6): s = K_hat - c  (static)  or  s = K_hat - c*N_hi  (dynamic)."""
    if mode == "dynamic":
        return k_hat - cost * n_hi
    return k_hat - cost


def route_s2d2_verification(
    *,
    span_len: int,
    draft_logits: torch.Tensor,
    span_start: int,
    span_end: int,
    confidence: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_id: int,
    tau: float,
    policy: str,
    min_verify_span: int,
    score_threshold: float,
    score_cost: float,
    score_mode: str,
    hysteresis_state: bool,
    hysteresis_on: float,
    hysteresis_off: float,
    estimator: str,
    entropy_beta: float,
    margin_threshold: float,
    vocab_size: int,
) -> Tuple[bool, bool, float]:
    """Algorithm 4: DOVERIFY routing decision."""
    del mask_id
    if policy == "always":
        return True, hysteresis_state, 0.0
    if policy == "never":
        return False, hysteresis_state, 0.0
    if policy == "min_span":
        return span_len >= min_verify_span, hysteresis_state, float(span_len)

    k_hat = estimate_expected_accept_prefix(
        draft_logits,
        span_start,
        span_end,
        estimator=estimator,
        entropy_beta=entropy_beta,
        margin_threshold=margin_threshold,
        vocab_size=vocab_size,
    )
    n_hi = 0
    if score_mode == "dynamic" and confidence is not None and mask_positions is not None:
        n_hi = int((confidence[mask_positions] > tau).sum().item()) if mask_positions.numel() > 0 else 0
    score = compute_verification_score(k_hat, score_cost, n_hi, score_mode)

    if policy == "score_threshold":
        return score >= score_threshold, hysteresis_state, score
    if policy == "hysteresis":
        new_state = hysteresis_state
        if hysteresis_state and score < hysteresis_off:
            new_state = False
        elif (not hysteresis_state) and score >= hysteresis_on:
            new_state = True
        return new_state, new_state, score
    return True, hysteresis_state, score


def _attn_for_model(mask: torch.Tensor, model: Any) -> torch.Tensor:
    m = mask.bool()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m[:, None, :, :]
    dtype = torch.float32
    try:
        dtype = next(model.parameters()).dtype
    except Exception:
        dtype = torch.float32
    out = torch.zeros(m.shape, device=m.device, dtype=dtype)
    return out.masked_fill(~m, torch.finfo(dtype).min)


def _model_name_candidates(model: Any, tokenizer: Any = None) -> List[str]:
    out: List[str] = []
    for obj in (model, getattr(model, "config", None), tokenizer):
        if obj is None:
            continue
        for attr in ("name_or_path", "_name_or_path", "model_type"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                out.append(value.lower())
        cls_name = obj.__class__.__name__
        if isinstance(cls_name, str):
            out.append(cls_name.lower())
    return out


def detect_s2d2_verifier_mode(model: Any, tokenizer: Any = None) -> str:
    """Detect verifier mode from model name.

    *da non cancellare* — Fast-dLLM v2 reminder:
    Quando aggiungiamo Fast-dLLM v2 il path "right_shifted_ar" si attiverà.
    Serve verificare che il sequential AR verifier con KV cache funzioni
    correttamente con l'architettura right-shifted (la causal mask standard
    del modello fornisce già la vista verifier, Fig.1(c) paper).
    Testare anche sub-block caching che per Fast-dLLM "is not lossless" (§5.1).
    """
    names = _model_name_candidates(model, tokenizer)
    if any(("dream" in x) or ("fast-dllm" in x) or ("fast_dllm" in x) for x in names):
        return "right_shifted_ar"
    if any(("llada" in x) or ("qwen" in x) or ("sdar" in x) for x in names):
        return "position_aligned_2l"
    return "position_aligned_2l"


def build_block_draft_attention_mask(
    seq_len: int,
    block_start: int,
    block_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Block-wise draft mask: causal prefix, fully-visible current block.

    *da non cancellare* — Eq.(4) paper reminder:
    Questa è una versione semplificata: tutto il blocco è fully-visible.
    L'Eq.(4) del paper prevede di splittare il blocco in committed prefix
    x^b_{<j} (causal A_j) e masked suffix (fully-visible 1_{B-j,B-j}).
    Quando implementeremo Fast-dLLM v2 / partially-causal drafting,
    aggiungere parametro `first_masked_in_block: int` e costruire la mask
    M_draft^(j) come da paper Section 4.2.
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if block_start > 0:
        prefix_tri = torch.tril(torch.ones((block_start, block_start), dtype=torch.bool, device=device))
        mask[:block_start, :block_start] = prefix_tri
    if block_end > block_start:
        mask[block_start:block_end, :block_start] = True
        mask[block_start:block_end, block_start:block_end] = True
    return mask.unsqueeze(0)


def build_verifier_inputs_position_aligned(
    drafted_tokens: torch.Tensor,
    mask_id: int,
    span_start: int,
    span_end: int,
    full_seq: torch.Tensor,
) -> S2D2VerifierInputs:
    """Paper-faithful 2L verifier input for position-aligned dLLMs."""
    L = int(span_end - span_start)
    prefix = full_seq[:, :span_start]
    drafted = drafted_tokens.view(1, L)
    query = torch.full((1, L), mask_id, dtype=full_seq.dtype, device=full_seq.device)
    input_ids = torch.cat([prefix, drafted, query], dim=1)

    prefix_len = int(prefix.shape[1])
    total_len = int(prefix_len + 2 * L)
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=full_seq.device)

    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=full_seq.device)
        )

    drafted_start = prefix_len
    query_start = prefix_len + L
    for i in range(L):
        row_d = drafted_start + i
        row_q = query_start + i
        if prefix_len > 0:
            mask[row_d, :prefix_len] = True
            mask[row_q, :prefix_len] = True
        mask[row_d, drafted_start : drafted_start + i + 1] = True
        if i > 0:
            mask[row_q, drafted_start : drafted_start + i] = True
        mask[row_q, row_q] = True

    pos_prefix = torch.arange(prefix_len, device=full_seq.device, dtype=torch.long)
    pos_local = torch.arange(span_start, span_end, device=full_seq.device, dtype=torch.long)
    position_ids = torch.cat([pos_prefix, pos_local, pos_local], dim=0).unsqueeze(0)

    return S2D2VerifierInputs(
        input_ids=input_ids,
        query_start=query_start,
        query_end=query_start + L,
        attention_mask_bool=mask.unsqueeze(0),
        position_ids=position_ids,
        verifier_mode="position_aligned_2l",
    )


def _prefill_causal_cache(model: Any, prefix_ids: torch.Tensor) -> Tuple[Any, Optional[torch.Tensor]]:
    if prefix_ids.shape[1] == 0:
        return None, None
    out = model(input_ids=prefix_ids, use_cache=True)
    return getattr(out, "past_key_values", None), out.logits[:, -1, :]


def _extend_causal_cache(
    model: Any,
    past_kv: Any,
    next_logits: Optional[torch.Tensor],
    tokens: torch.Tensor,
) -> Tuple[Any, Optional[torch.Tensor]]:
    logits = next_logits
    cache = past_kv
    for i in range(tokens.shape[1]):
        tok = tokens[:, i : i + 1]
        if cache is None:
            out = model(input_ids=tok, use_cache=True)
        else:
            out = model(input_ids=tok, past_key_values=cache, use_cache=True)
        cache = getattr(out, "past_key_values", None)
        logits = out.logits[:, -1, :]
    return cache, logits


def _clone_past_key_values(past_kv: Any) -> Any:
    if past_kv is None:
        return None
    cloned = []
    for layer in past_kv:
        if isinstance(layer, (tuple, list)):
            cloned.append(tuple(x.clone() if torch.is_tensor(x) else x for x in layer))
        else:
            cloned.append(layer)
    return tuple(cloned)


def _past_seq_len(past_kv: Any) -> int:
    if past_kv is None or len(past_kv) == 0:
        return 0
    layer0 = past_kv[0]
    if isinstance(layer0, (tuple, list)):
        for item in layer0:
            if torch.is_tensor(item) and item.ndim >= 3:
                return int(item.shape[-2])
    return 0


def _draft_forward_block(
    *,
    model: Any,
    seq: torch.Tensor,
    block_start: int,
    block_end: int,
    force_noncausal_ctx: Any = None,
) -> Any:
    draft_mask = build_block_draft_attention_mask(seq.shape[1], block_start, block_end, seq.device)
    attn4d = _attn_for_model(draft_mask, model)
    if force_noncausal_ctx is not None:
        with force_noncausal_ctx(model):
            return model(input_ids=seq, attention_mask=attn4d)
    return model(input_ids=seq, attention_mask=attn4d)


def run_s2d2_verification(
    *,
    model: Any,
    verifier_inputs: S2D2VerifierInputs,
    draft_probs: torch.Tensor,
    drafted_ids: torch.Tensor,
    span_start: int,
    span_end: int,
    force_noncausal_ctx: Any = None,
    block_prefix_cache: Any = None,
    block_prefix_next_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, bool]:
    """Algorithm 3 lines 10–25: speculative verification with rejection sampling."""
    L = int(span_end - span_start)
    if L == 0:
        return drafted_ids, torch.zeros(0), 0, -1, False

    model.eval()
    verifier_cache_used = False

    if verifier_inputs.verifier_mode == "right_shifted_ar":
        draft_span_probs = draft_probs[0, span_start:span_end, :]
        accepted_ids = drafted_ids.clone()
        accepted_count = 0
        rejection_pos = -1
        prefix_ids = verifier_inputs.input_ids

        if block_prefix_cache is not None and block_prefix_next_logits is not None:
            cache = _clone_past_key_values(block_prefix_cache)
            logits = block_prefix_next_logits.clone()
            verifier_cache_used = True
            extra_prefix = prefix_ids[:, _past_seq_len(block_prefix_cache) :]
            if extra_prefix.shape[1] > 0:
                cache, logits = _extend_causal_cache(model, cache, logits, extra_prefix)
        else:
            cache, logits = _prefill_causal_cache(model, prefix_ids)

        ver_token_logits: List[torch.Tensor] = []
        for i in range(L):
            if logits is None:
                if prefix_ids.shape[1] == 0:
                    seed_tok = drafted_ids[i : i + 1].view(1, 1)
                    out = model(input_ids=seed_tok, use_cache=True)
                    cache = getattr(out, "past_key_values", None)
                    logits = out.logits[:, -1, :]
                else:
                    cache, logits = _prefill_causal_cache(model, prefix_ids)
            ver_token_logits.append(logits.squeeze(0))
            token_id = int(drafted_ids[i].item())
            p_i = max(float(draft_span_probs[i, token_id].item()), 1e-12)
            q_probs = F.softmax(logits.float(), dim=-1)
            q_i = float(q_probs[0, token_id].item())
            ratio = min(1.0, q_i / p_i)
            r = float(torch.rand(1, device=logits.device).item())
            if r < ratio:
                accepted_count += 1
                out = model(input_ids=drafted_ids[i : i + 1].view(1, 1), past_key_values=cache, use_cache=True)
                cache = getattr(out, "past_key_values", None)
                logits = out.logits[:, -1, :]
            else:
                rejection_pos = i
                residual = (q_probs[0] - draft_span_probs[i]).clamp(min=0.0)
                residual_sum = residual.sum()
                if residual_sum > 1e-12:
                    residual = residual / residual_sum
                    resampled_id = int(torch.multinomial(residual, 1).item())
                else:
                    resampled_id = int(torch.multinomial(q_probs[0], 1).item())
                accepted_ids[i] = resampled_id
                break
        ver_logits = torch.stack(ver_token_logits, dim=0) if ver_token_logits else torch.zeros((0, draft_probs.shape[-1]))
        return accepted_ids, ver_logits, accepted_count, rejection_pos, verifier_cache_used

    attn4d = None
    if verifier_inputs.attention_mask_bool is not None:
        attn4d = _attn_for_model(verifier_inputs.attention_mask_bool, model)

    with torch.inference_mode():
        kwargs: Dict[str, Any] = {"input_ids": verifier_inputs.input_ids}
        if attn4d is not None:
            kwargs["attention_mask"] = attn4d
        if verifier_inputs.position_ids is not None:
            kwargs["position_ids"] = verifier_inputs.position_ids
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(**kwargs)
        else:
            out = model(**kwargs)

    ver_logits = out.logits
    ver_span_logits = ver_logits[0, verifier_inputs.query_start : verifier_inputs.query_end, :]
    ver_probs = F.softmax(ver_span_logits.float(), dim=-1)
    draft_span_probs = draft_probs[0, span_start:span_end, :]
    accepted_ids = drafted_ids.clone()
    accepted_count = 0
    rejection_pos = -1

    for i in range(L):
        token_id = int(drafted_ids[i].item())
        p_i = max(float(draft_span_probs[i, token_id].item()), 1e-12)
        q_i = float(ver_probs[i, token_id].item())
        ratio = min(1.0, q_i / p_i)
        r = float(torch.rand(1, device=ver_probs.device).item())
        if r < ratio:
            accepted_count += 1
        else:
            rejection_pos = i
            residual = (ver_probs[i] - draft_span_probs[i]).clamp(min=0.0)
            residual_sum = residual.sum()
            if residual_sum > 1e-12:
                residual = residual / residual_sum
                resampled_id = int(torch.multinomial(residual, 1).item())
            else:
                resampled_id = int(torch.multinomial(ver_probs[i], 1).item())
            accepted_ids[i] = resampled_id
            break
    return accepted_ids, ver_span_logits, accepted_count, rejection_pos, verifier_cache_used


# ---------------------------------------------------------------------------
# S2D2 decode loop — Algorithm 3
# ---------------------------------------------------------------------------
def s2d2_decode(
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
    block_size: int = 32,
    denoising_steps_per_block: int = 0,
    confidence_threshold: float = 0.3,
    routing_policy: str = "min_span",
    min_verify_span: int = 2,
    score_threshold: float = 0.0,
    score_cost: float = 1.0,
    score_mode: str = "static",
    hysteresis_on: float = 1.0,
    hysteresis_off: float = -5.0,
    acceptance_estimator: str = "entropy",
    entropy_beta: float = 1.0,
    margin_threshold: float = 0.1,
    store_diagnostics: bool = True,
    seed: int = 42,
    is_dummy: bool = False,
    force_noncausal_ctx: Any = None,
) -> Tuple[str, Dict[str, Any], S2D2Diagnostics]:
    """Full S2D2 inference loop implementing Algorithm 3 block-wise."""
    from .utils import seed_everything, tokens_per_second

    del tau_mask, tau_edit

    seed_everything(seed)
    verifier_mode = detect_s2d2_verifier_mode(model, tokenizer)
    diag = S2D2Diagnostics(
        routing_policy_used=routing_policy,
        acceptance_estimator_used=acceptance_estimator,
        block_size=block_size,
        verifier_mode_used=verifier_mode,
        verifier_mask_path="position_aligned_2l" if verifier_mode == "position_aligned_2l" else "right_shifted_ar",
        kv_cache_mode="block_prefix_verifier" if verifier_mode == "right_shifted_ar" else "disabled",
    )

    enc = tokenizer([prompt], return_tensors="pt")
    seq = enc["input_ids"].to(device)
    prompt_len = int(seq.shape[1])

    T = denoising_steps_per_block if denoising_steps_per_block > 0 else max(1, max_steps)
    total_blocks = max(1, _math.ceil(max_new_tokens / block_size))

    model.eval()
    t0 = time.time()
    hysteresis_state = False
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            eos_token_id = int(eos_token_id)
        except Exception:
            eos_token_id = None
    finish_reason = "length"
    converged = False
    eos_cut_idx: Optional[int] = None
    total_step_counter = 0

    with torch.inference_mode():
        for block_idx in range(total_blocks):
            remaining = int(max_new_tokens - max(0, seq.shape[1] - prompt_len))
            if remaining <= 0:
                break
            current_block_size = min(int(block_size), remaining)
            new_block = torch.full((1, current_block_size), mask_id, dtype=torch.long, device=device)
            blk_start = int(seq.shape[1])
            seq = torch.cat([seq, new_block], dim=1)
            blk_end = int(seq.shape[1])

            block_prefix_cache = None
            block_prefix_next_logits = None
            if verifier_mode == "right_shifted_ar" and blk_start > 0 and hasattr(model, "forward"):
                try:
                    block_prefix_cache, block_prefix_next_logits = _prefill_causal_cache(model, seq[:, :blk_start])
                    if block_prefix_cache is not None:
                        diag.verifier_cache_prefills += 1
                except Exception:
                    block_prefix_cache, block_prefix_next_logits = None, None

            block_converged = False
            for _ in range(T):
                step_t0 = time.time()
                total_step_counter += 1

                block_seq = seq[0, blk_start:blk_end]
                is_mask_block = block_seq.eq(mask_id)
                masked_before = int(is_mask_block.sum().item())
                if masked_before == 0:
                    block_converged = True
                    break

                draft_out = _draft_forward_block(
                    model=model,
                    seq=seq,
                    block_start=blk_start,
                    block_end=blk_end,
                    force_noncausal_ctx=force_noncausal_ctx,
                )
                logits = draft_out.logits
                probs = F.softmax(logits.float(), dim=-1)
                confidence, pred_ids = probs.max(dim=-1)

                span_start, span_end = find_first_contiguous_mask_span(seq, mask_id, blk_start, blk_end)
                contiguous_span_len = max(0, span_end - span_start) if span_start >= 0 else 0
                block_confidence = confidence[0, blk_start:blk_end]
                mask_positions_block = is_mask_block.nonzero(as_tuple=False).view(-1)

                do_verify, hysteresis_state, route_score = route_s2d2_verification(
                    span_len=contiguous_span_len,
                    draft_logits=logits,
                    span_start=span_start if span_start >= 0 else blk_start,
                    span_end=span_end if span_end >= 0 else blk_start,
                    confidence=block_confidence,
                    mask_positions=mask_positions_block,
                    mask_id=mask_id,
                    tau=confidence_threshold,
                    policy=routing_policy,
                    min_verify_span=min_verify_span,
                    score_threshold=score_threshold,
                    score_cost=score_cost,
                    score_mode=score_mode,
                    hysteresis_state=hysteresis_state,
                    hysteresis_on=hysteresis_on,
                    hysteresis_off=hysteresis_off,
                    estimator=acceptance_estimator,
                    entropy_beta=entropy_beta,
                    margin_threshold=margin_threshold,
                    vocab_size=vocab_size,
                )

                accepted_count = 0
                rejection_pos = -1
                committed_this_step = 0
                verifier_input_tokens = 0
                verifier_cache_used = False

                if do_verify and span_start >= 0 and contiguous_span_len > 0:
                    diag.verifier_invocations += 1
                    drafted_ids = pred_ids[0, span_start:span_end]
                    if verifier_mode == "position_aligned_2l":
                        verifier_inputs = build_verifier_inputs_position_aligned(
                            drafted_ids,
                            mask_id,
                            span_start,
                            span_end,
                            seq,
                        )
                    else:
                        verifier_inputs = S2D2VerifierInputs(
                            input_ids=seq[:, :span_start].clone(),
                            query_start=0,
                            query_end=contiguous_span_len,
                            verifier_mode="right_shifted_ar",
                        )
                    verifier_input_tokens = int(verifier_inputs.input_ids.shape[1])

                    accepted_ids, _ver_logits, accepted_count, rejection_pos, verifier_cache_used = run_s2d2_verification(
                        model=model,
                        verifier_inputs=verifier_inputs,
                        draft_probs=probs,
                        drafted_ids=drafted_ids,
                        span_start=span_start,
                        span_end=span_end,
                        force_noncausal_ctx=force_noncausal_ctx,
                        block_prefix_cache=block_prefix_cache,
                        block_prefix_next_logits=block_prefix_next_logits,
                    )
                    if verifier_cache_used:
                        diag.verifier_cache_hits += 1

                    if rejection_pos < 0:
                        seq[0, span_start:span_end] = accepted_ids
                        committed_this_step = contiguous_span_len
                    else:
                        commit_end = span_start + rejection_pos + 1
                        seq[0, span_start:commit_end] = accepted_ids[: rejection_pos + 1]
                        committed_this_step = rejection_pos + 1
                    diag.accepted_prefix_lengths.append(accepted_count)
                else:
                    diag.verifier_skips += 1
                    diag.fallback_to_diffusion_count += 1
                    block_pred_ids = pred_ids[0, blk_start:blk_end]
                    selected = is_mask_block & block_confidence.ge(confidence_threshold)
                    if is_mask_block.any() and not selected.any():
                        best_idx = block_confidence.clone()
                        best_idx[~is_mask_block] = -float("inf")
                        best_pos = int(best_idx.argmax().item())
                        selected[best_pos] = True
                    if selected.any():
                        updated_block = seq[0, blk_start:blk_end].clone()
                        updated_block[selected] = block_pred_ids[selected]
                        seq[0, blk_start:blk_end] = updated_block
                        committed_this_step = int(selected.sum().item())

                masked_after = int(seq[0, blk_start:blk_end].eq(mask_id).sum().item())
                step_elapsed = (time.time() - step_t0) * 1000.0

                diag.last_route_score = float(route_score)
                diag.last_block_index = int(block_idx)
                diag.last_span_start = int(span_start)
                diag.last_span_end = int(span_end)

                if store_diagnostics:
                    diag.steps.append(
                        S2D2StepDiag(
                            step=total_step_counter,
                            block_index=block_idx,
                            block_start=blk_start,
                            block_end=blk_end,
                            masked_before=masked_before,
                            contiguous_span_len=contiguous_span_len,
                            span_start=span_start,
                            span_end=span_end,
                            route_score=float(route_score),
                            verify_invoked=do_verify and contiguous_span_len > 0,
                            verifier_mode=verifier_mode,
                            verifier_input_tokens=verifier_input_tokens,
                            verifier_cache_used=verifier_cache_used,
                            accepted_count=accepted_count,
                            rejection_position=rejection_pos,
                            committed_count=committed_this_step,
                            masked_after=masked_after,
                            elapsed_ms=step_elapsed,
                        )
                    )

                gen_out = seq[0, prompt_len:]
                if eos_token_id is not None:
                    eos_pos = (gen_out == eos_token_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        eos_cut_idx = int(eos_pos[0][0].item())
                        finish_reason = "eos"
                        converged = True
                        block_converged = True
                        break
                if masked_after == 0:
                    finish_reason = "converged"
                    block_converged = True
                    break

            if converged:
                break
            if not block_converged:
                finish_reason = "length"
                break

    elapsed = max(1e-6, time.time() - t0)
    out_ids = seq[0, prompt_len:]
    if eos_cut_idx is not None:
        out_ids = out_ids[: max(0, eos_cut_idx + 1)]

    tokens_generated = int((out_ids != mask_id).sum().item())
    text = _decode_text(tokenizer, out_ids, mask_id, is_dummy)

    if not converged and finish_reason != "eos":
        finish_reason = "length"

    diag.total_denoising_steps = total_step_counter
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason
    if diag.accepted_prefix_lengths:
        diag.avg_accepted_prefix_length = sum(diag.accepted_prefix_lengths) / len(diag.accepted_prefix_lengths)

    stats = {
        "engine": "s2d2",
        "mode": "S2D2",
        "block_size": block_size,
        "denoising_steps_per_block": T,
        "confidence_threshold": confidence_threshold,
        "steps": total_step_counter,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_second(tokens_generated, elapsed),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "dummy_model": is_dummy,
        "routing_policy": routing_policy,
        "acceptance_estimator": acceptance_estimator,
        "verifier_invocations": diag.verifier_invocations,
        "verifier_skips": diag.verifier_skips,
        "avg_accepted_prefix_length": diag.avg_accepted_prefix_length,
        "fallback_to_diffusion_count": diag.fallback_to_diffusion_count,
        "verifier_mode": verifier_mode,
        "kv_cache_mode": diag.kv_cache_mode,
        "s2d2_diagnostics": diag.to_dict(),
    }

    return text.strip(), stats, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_text(tok: Any, out_ids: torch.Tensor, mask_id: int, is_dummy: bool) -> str:
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

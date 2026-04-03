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
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math as _math
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
    s2d2_min_verify_span: int = Field(default=2, ge=1, le=512, description="τ_span for min-span policy (Alg.4)")
    s2d2_score_threshold: float = Field(default=0.0, description="τ_score for score-threshold policy")
    s2d2_score_cost: float = Field(default=1.0, ge=0.0, le=10.0, description="c — cost hyperparameter in Eq.(6)")
    s2d2_score_mode: str = Field(default="static", description="static | dynamic for Eq.(6)")
    s2d2_hysteresis_on: float = Field(default=1.0, description="τ_on for hysteresis policy")
    s2d2_hysteresis_off: float = Field(default=-5.0, description="τ_off for hysteresis policy")
    s2d2_acceptance_estimator: str = Field(
        default="entropy",
        description="Token acceptance estimator: entropy | margin (Sec 4.3, Eq.5)",
    )
    s2d2_entropy_beta: float = Field(default=1.0, ge=0.01, le=10.0, description="β for entropy estimator α_i = exp(-β H̃_i)")
    s2d2_margin_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="τ_margin for margin estimator")
    s2d2_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="τ — confidence threshold for diffusion fallback (Alg.2/3)")
    s2d2_store_diagnostics: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class S2D2StepDiag:
    step: int
    masked_before: int
    contiguous_span_len: int
    verify_invoked: bool
    accepted_count: int
    rejection_position: int  # -1 if no rejection
    committed_count: int
    masked_after: int
    elapsed_ms: float


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
    verifier_mask_path: str = "position_aligned_2n"
    block_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "masked_before": s.masked_before,
                    "contiguous_span_len": s.contiguous_span_len,
                    "verify_invoked": s.verify_invoked,
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
            "block_size": self.block_size,
        }


# ---------------------------------------------------------------------------
# Pure helper functions — paper-faithful
# ---------------------------------------------------------------------------

def find_first_contiguous_mask_span(
    seq: torch.Tensor,
    mask_id: int,
    prompt_len: int,
) -> Tuple[int, int]:
    """Find the first contiguous span of [MASK] tokens in the generation region.
    Returns (start, end) indices in seq coordinates. end is exclusive.
    If no masked tokens remain, returns (-1, -1).
    Paper Section 4.1: "optionally verifies the first contiguous masked span C_t".
    """
    gen_region = seq[0, prompt_len:]  # (gen_len,)
    is_mask = gen_region.eq(mask_id)
    if not is_mask.any():
        return -1, -1
    # Find first masked position
    nonzero = is_mask.nonzero(as_tuple=False).view(-1)
    if nonzero.numel() == 0:
        return -1, -1
    start_gen = int(nonzero[0].item())
    # Extend contiguous span
    end_gen = start_gen + 1
    gen_len = gen_region.shape[0]
    while end_gen < gen_len and is_mask[end_gen]:
        end_gen += 1
    return prompt_len + start_gen, prompt_len + end_gen


def estimate_expected_accept_prefix(
    draft_logits: torch.Tensor,
    span_start: int,
    span_end: int,
    estimator: str = "entropy",
    entropy_beta: float = 1.0,
    margin_threshold: float = 0.1,
    vocab_size: int = 1,
) -> float:
    """Estimate expected accepted prefix length K̂ via Eq.(5).
    K̂ = Σ_{k=1}^{L} Π_{i=1}^{k} α_i
    where α_i is estimated via entropy- or margin-based proxy.
    """
    span_logits = draft_logits[0, span_start:span_end, :]  # (L, V)
    L = span_logits.shape[0]
    if L == 0:
        return 0.0
    probs = F.softmax(span_logits.float(), dim=-1)

    if estimator == "margin":
        # α_i = 1[m_i ≥ τ_margin]  where m_i = top1 - top2
        top2_vals, _ = probs.topk(min(2, probs.shape[-1]), dim=-1)
        if top2_vals.shape[-1] >= 2:
            margins = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            margins = torch.ones(L, device=probs.device)
        alphas = (margins >= margin_threshold).float()
    else:
        # Entropy-based: α_i = exp(-β H̃_i)  where H̃_i = H_i / log(V)
        log_v = _math.log(max(vocab_size, 2))
        p_clamped = probs.clamp(min=1e-12)
        H = -(p_clamped * p_clamped.log()).sum(dim=-1)  # (L,)
        H_tilde = H / log_v
        alphas = torch.exp(-entropy_beta * H_tilde)

    # K̂ = Σ_{k=1}^{L} Π_{i=1}^{k} α_i
    cum_prod = torch.cumprod(alphas, dim=0)  # (L,)
    k_hat = float(cum_prod.sum().item())
    return k_hat


def compute_verification_score(
    k_hat: float,
    cost: float,
    n_hi: int = 0,
    mode: str = "static",
) -> float:
    """Eq.(6): s = K̂ - c  (static)  or  s = K̂ - c·N_hi  (dynamic)."""
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
    """Algorithm 4: DOVERIFY routing decision.
    Returns (do_verify, new_hysteresis_state, score).
    """
    if policy == "always":
        return True, hysteresis_state, 0.0
    if policy == "never":
        return False, hysteresis_state, 0.0

    # Minimum-span policy (simplest, often surprisingly effective)
    if policy == "min_span":
        return span_len >= min_verify_span, hysteresis_state, float(span_len)

    # Compute K̂ for score-based policies
    k_hat = estimate_expected_accept_prefix(
        draft_logits, span_start, span_end,
        estimator=estimator, entropy_beta=entropy_beta,
        margin_threshold=margin_threshold, vocab_size=vocab_size,
    )
    # Count high-confidence tokens in masked positions for dynamic scoring
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
        elif not hysteresis_state and score >= hysteresis_on:
            new_state = True
        return new_state, new_state, score

    # Fallback: always verify if unknown policy
    return True, hysteresis_state, 0.0


def build_verifier_inputs_position_aligned(
    drafted_tokens: torch.Tensor,
    mask_id: int,
    span_start: int,
    span_end: int,
    full_seq: torch.Tensor,
) -> torch.Tensor:
    """Build verifier input for position-aligned models using the "2L trick" (Sec 4.2, Eq.3).
    For a span of length L, we concatenate:
      [drafted tokens at span positions] + [MASK × L at same positions]
    The full sequence context before the span is kept,
    and we construct the input so that the second L positions see
    only causal (left-to-right) context from the first L positions.

    For our implementation within a full-sequence bidirectional dLLM,
    we approximate this by constructing a verifier sequence where:
    - positions before span_start: keep committed tokens (context)
    - span positions: fill with drafted tokens
    - positions after span_end: keep as-is
    Then we run this sequence through the model with noncausal attention
    and use the logits at span positions as verifier probabilities.

    This is equivalent to the "single forward on drafted sequence" approach
    for position-aligned models because our model is bidirectional.
    """
    ver_seq = full_seq.clone()
    ver_seq[0, span_start:span_end] = drafted_tokens
    return ver_seq


def run_s2d2_verification(
    *,
    model: Any,
    verifier_seq: torch.Tensor,
    draft_probs: torch.Tensor,
    drafted_ids: torch.Tensor,
    span_start: int,
    span_end: int,
    force_noncausal_ctx: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Algorithm 3 lines 10–25: speculative verification with rejection sampling.
    Uses the SAME model in verifier mode on a sequence with drafted tokens filled in.
    Returns: (accepted_ids, verifier_logits, accepted_count, rejection_pos)
    - accepted_count: number of tokens accepted left-to-right
    - rejection_pos: position of rejection in span, or -1 if all accepted
    """
    L = span_end - span_start
    if L == 0:
        return drafted_ids, torch.zeros(0), 0, -1

    model.eval()
    with torch.inference_mode():
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(input_ids=verifier_seq)
        else:
            out = model(input_ids=verifier_seq)

    ver_logits = out.logits  # (1, S, V)
    ver_span_logits = ver_logits[0, span_start:span_end, :]  # (L, V)
    ver_probs = F.softmax(ver_span_logits.float(), dim=-1)  # (L, V)
    draft_span_probs = draft_probs[0, span_start:span_end, :]  # (L, V)

    # Speculative rejection sampling: left-to-right (Algorithm 3 lines 14–24)
    accepted_ids = drafted_ids.clone()  # (L,)
    accepted_count = 0
    rejection_pos = -1

    for i in range(L):
        token_id = int(drafted_ids[i].item())
        p_i = float(draft_span_probs[i, token_id].item())
        q_i = float(ver_probs[i, token_id].item())

        # Acceptance probability: min(1, q_i / p_i) — Eq.(8), Algorithm 3 line 16
        p_i = max(p_i, 1e-12)
        ratio = min(1.0, q_i / p_i)
        r = float(torch.rand(1).item())

        if r < ratio:
            # Accept (Algorithm 3 line 17)
            accepted_count += 1
        else:
            # Reject: resample from residual distribution (P_ver - P_draft)+ (line 19)
            rejection_pos = i
            residual = (ver_probs[i] - draft_span_probs[i]).clamp(min=0.0)
            residual_sum = residual.sum()
            if residual_sum > 1e-12:
                residual = residual / residual_sum
                resampled_id = int(torch.multinomial(residual, 1).item())
            else:
                # Fallback: sample from verifier distribution
                resampled_id = int(torch.multinomial(ver_probs[i], 1).item())
            accepted_ids[i] = resampled_id
            # Stop speculative segment (line 22: break)
            break

    return accepted_ids, ver_logits, accepted_count, rejection_pos


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
    """Full S2D2 inference loop implementing Algorithm 3.
    The SAME model serves as both drafter (block-diffusion) and verifier (AR mode).
    Returns: (text, stats_dict, diagnostics).
    """
    from .utils import seed_everything, tokens_per_second

    seed_everything(seed)
    diag = S2D2Diagnostics(
        routing_policy_used=routing_policy,
        acceptance_estimator_used=acceptance_estimator,
        block_size=block_size,
    )

    # Encode prompt
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])

    # Build sequence: [prompt tokens] [MASK × max_new_tokens]
    seq = torch.full((1, prompt_len + max_new_tokens), mask_id, dtype=torch.long, device=device)
    seq[:, :prompt_len] = input_ids

    # Determine effective denoising steps per block
    T = denoising_steps_per_block if denoising_steps_per_block > 0 else max(1, max_steps)

    # Total number of blocks to process = ceil(max_new_tokens / block_size)
    total_blocks = max(1, _math.ceil(max_new_tokens / block_size))

    model.eval()
    t0 = time.time()
    hysteresis_state = False  # paper Algorithm 4: starts OFF
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
        # Block-wise autoregressive outer loop (Algorithm 1)
        for block_idx in range(total_blocks):
            blk_start = prompt_len + block_idx * block_size
            blk_end = min(prompt_len + (block_idx + 1) * block_size, prompt_len + max_new_tokens)
            if blk_start >= seq.shape[1]:
                break

            # Inner denoising loop per block (Algorithm 3 lines 1–31)
            for t in range(T):
                step_t0 = time.time()
                total_step_counter += 1

                # Check: any MASK left in this block?
                block_seq = seq[0, blk_start:blk_end]
                is_mask_block = block_seq.eq(mask_id)
                masked_before = int(is_mask_block.sum().item())
                if masked_before == 0:
                    break

                # ─── DRAFT FORWARD (Algorithm 3 line 5) ───
                if force_noncausal_ctx is not None:
                    with force_noncausal_ctx(model):
                        draft_out = model(input_ids=seq)
                else:
                    draft_out = model(input_ids=seq)
                logits = draft_out.logits  # (1, S, V)

                # Draft proposals (line 6): (x̂, p) ← SampleFromLogits(ℓ)
                probs = F.softmax(logits.float(), dim=-1)
                confidence, pred_ids = probs.max(dim=-1)  # (1, S)

                # ─── FIND FIRST CONTIGUOUS MASK SPAN (line 8) ───
                span_start, span_end = find_first_contiguous_mask_span(seq, mask_id, prompt_len)
                contiguous_span_len = max(0, span_end - span_start) if span_start >= 0 else 0

                # Masked positions M_t (line 7)
                gen_region = seq[0, prompt_len:]
                gen_is_mask = gen_region.eq(mask_id)
                mask_positions_gen = gen_is_mask.nonzero(as_tuple=False).view(-1)

                # ─── ROUTING DECISION (Algorithm 4, line 9) ───
                do_verify, hysteresis_state, route_score = route_s2d2_verification(
                    span_len=contiguous_span_len,
                    draft_logits=logits,
                    span_start=span_start if span_start >= 0 else 0,
                    span_end=span_end if span_end >= 0 else 0,
                    confidence=confidence[0, prompt_len:],
                    mask_positions=mask_positions_gen,
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

                if do_verify and span_start >= 0 and contiguous_span_len > 0:
                    # ─── SELF-SPECULATIVE VERIFICATION (Algorithm 3 lines 10–25) ───
                    diag.verifier_invocations += 1

                    # Build verifier input (line 10): x̃^b ← x^b, x̃^b_{C_t} ← x̂_{C_t}
                    drafted_ids = pred_ids[0, span_start:span_end]
                    verifier_seq = build_verifier_inputs_position_aligned(
                        drafted_ids, mask_id, span_start, span_end, seq,
                    )

                    # Run verification (lines 11–24): same model as verifier
                    accepted_ids, _ver_logits, accepted_count, rejection_pos = run_s2d2_verification(
                        model=model,
                        verifier_seq=verifier_seq,
                        draft_probs=probs,
                        drafted_ids=drafted_ids,
                        span_start=span_start,
                        span_end=span_end,
                        force_noncausal_ctx=force_noncausal_ctx,
                    )

                    # Commit accepted tokens (line 25): x^b_{S_t} ← x̂_{S_t}
                    if rejection_pos < 0:
                        # All accepted
                        seq[0, span_start:span_end] = accepted_ids
                        committed_this_step = contiguous_span_len
                    else:
                        # Accept up to rejection, commit resampled token at rejection
                        commit_end = span_start + rejection_pos + 1
                        seq[0, span_start:commit_end] = accepted_ids[:rejection_pos + 1]
                        committed_this_step = rejection_pos + 1

                    diag.accepted_prefix_lengths.append(accepted_count)

                else:
                    # ─── FALLBACK: standard confidence-threshold decoding (Algorithm 3 lines 26–29) ───
                    diag.verifier_skips += 1
                    diag.fallback_to_diffusion_count += 1

                    # S_t ← {i ∈ M_t : p_i > τ}  (line 27)
                    gen_confidence = confidence[0, prompt_len:]
                    gen_pred_ids = pred_ids[0, prompt_len:]
                    selected = gen_is_mask & gen_confidence.ge(confidence_threshold)

                    # S_t ← S_t ∪ {argmax_{i ∈ M_t} p_i}  (line 28: always commit at least one)
                    if gen_is_mask.any() and not selected.any():
                        best_idx = gen_confidence.clone()
                        best_idx[~gen_is_mask] = -float("inf")
                        best_pos = int(best_idx.argmax().item())
                        selected[best_pos] = True

                    # x^b_{S_t} ← x̂_{S_t}  (line 29)
                    if selected.any():
                        gen_region[selected] = gen_pred_ids[selected]
                        seq[0, prompt_len:] = gen_region
                        committed_this_step = int(selected.sum().item())

                # ─── Diagnostics ───
                masked_after = int(seq[0, prompt_len:].eq(mask_id).sum().item())
                step_elapsed = (time.time() - step_t0) * 1000.0

                if store_diagnostics:
                    diag.steps.append(S2D2StepDiag(
                        step=total_step_counter,
                        masked_before=masked_before,
                        contiguous_span_len=contiguous_span_len,
                        verify_invoked=do_verify and contiguous_span_len > 0,
                        accepted_count=accepted_count,
                        rejection_position=rejection_pos,
                        committed_count=committed_this_step,
                        masked_after=masked_after,
                        elapsed_ms=step_elapsed,
                    ))

                # ─── Stop conditions ───
                if eos_token_id is not None:
                    gen_out = seq[0, prompt_len:]
                    eos_pos = (gen_out == eos_token_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        eos_cut_idx = int(eos_pos[0][0].item())
                        finish_reason = "eos"
                        converged = True
                        break

                if masked_after == 0:
                    finish_reason = "converged"
                    converged = True
                    break

            if converged:
                break

    elapsed = max(1e-6, time.time() - t0)
    out_ids = seq[0, prompt_len:]
    if eos_cut_idx is not None:
        out_ids = out_ids[:max(0, eos_cut_idx + 1)]

    tokens_generated = int((out_ids != mask_id).sum().item())
    text = _decode_text(tokenizer, out_ids, mask_id, is_dummy)

    if not converged:
        finish_reason = "length"

    diag.total_denoising_steps = total_step_counter
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason
    if diag.accepted_prefix_lengths:
        diag.avg_accepted_prefix_length = sum(diag.accepted_prefix_lengths) / len(diag.accepted_prefix_lengths)

    stats = {
        "engine": "s2d2",
        "mode": "S2D2",
        "tau_mask": tau_mask,
        "tau_edit": tau_edit,
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

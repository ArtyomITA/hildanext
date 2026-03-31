# Inferenza2 — Hybrid OTS-over-RCD inference engine for dLLMs.
#
# Combines:
#   OTS (Order-Token Search) — outer search/branching/pruning controller
#     Paper: "Improving Diffusion LM Decoding through Joint Search
#             in Generation Order and Token Space" (Shen et al., 2026)
#     arXiv:2601.20339
#
#   RCD (Residual Context Diffusion) — inner denoising-state enhancement
#     Paper: "Residual Context Diffusion Language Models" (Hu et al., 2026)
#     arXiv:2601.22954
#
# HYBRID DESIGN:
#   OTS is the OUTER controller: maintains beams, expands candidates at
#   search checkpoints, prunes via diffusion-native scoring.
#   RCD is the INNER mechanism: each beam carries its own residual state
#   (alpha, delta).  Every forward pass uses RCD-style inputs_embeds
#   construction (Eq.2 from RCD paper).  Candidate expansion uses
#   logits from the RCD-enhanced forward.
#
# KEY RULE: after beams diverge, each beam evolves its own independent
# RCD residual state.  No global shared state.
#
# APPROXIMATIONS (documented):
#   1. No RCD-trained weights — model is a standard dLLM.
#   2. Warm-start uses target model as Mref fallback.
#   3. HF inputs_embeds interface (not RCD-native forward).
#   4. Scoring input: OTS diffusion-native scorer uses plain input_ids
#      for the scoring forward (not RCD-augmented), since the paper's
#      scoring function re-masks positions in x0 which already contains
#      the model's full prediction.  This is a principled choice: the
#      scorer evaluates whether the MODEL supports the denoising action,
#      not whether the RCD-augmented state supports it.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import copy
import time
import math as _math

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Imports from existing engines (reuse, don't reimplement)
# ---------------------------------------------------------------------------
# RCD math helpers
from .inference_rcd import (
    compute_normalized_entropy,
    compute_rcd_residuals_from_probs,
    build_rcd_inputs_embeds,
    initialize_rcd_warm_start,
    _get_embedding_layer,
)

# OTS search helpers
from .inference_ots import (
    _add_gumbel_noise,
    _transfer_tokens,
    prune_ots_beams as _prune_ots_beams,
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class Inference2Request(BaseModel):
    """Unified request schema for the inferenza2 hybrid engine."""
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

    # ---- Hybrid-level knobs ----
    hybrid_enable_rcd: bool = Field(default=True, description="Enable RCD residual enhancement")
    hybrid_enable_ots: bool = Field(default=True, description="Enable OTS search/branching")
    hybrid_mode: str = Field(default="rcd_plus_ots", description="rcd_plus_ots | rcd_only | ots_only")
    hybrid_store_diagnostics: bool = Field(default=True)
    hybrid_store_search_trace: bool = Field(default=True)
    hybrid_debug_return_intermediate: bool = Field(default=False)
    hybrid_ablation_mode: Optional[str] = Field(
        default=None,
        description="full_hybrid | rcd_only | ots_only — overrides hybrid_mode if set",
    )

    # ---- RCD-specific knobs ----
    rcd_alpha_mode: str = Field(default="normalized_entropy", description="normalized_entropy (Eq.3)")
    rcd_temperature_residual: float = Field(default=1.0, ge=0.01, le=10.0, description="T_res (Sec 3.3)")
    rcd_force_mask_only_injection: bool = Field(default=True, description="Inject residual only on [MASK]")
    rcd_warm_start: bool = Field(default=True, description="Warm-start with reference/target model")
    rcd_reference_model: Optional[str] = Field(default=None, description="Path to Mref checkpoint")
    rcd_same_model_warm_start_fallback: bool = Field(default=True, description="Use target model if no Mref")

    # ---- OTS-specific knobs ----
    ots_beam_size: int = Field(default=3, ge=1, le=32, description="K beams")
    ots_gumbel_temperature: float = Field(default=0.6, ge=0.0, le=5.0, description="τ Gumbel noise")
    ots_search_interval: int = Field(default=0, ge=0, le=256, description="Steps between checkpoints; 0=auto")
    ots_pruning_mode: str = Field(default="diffusion_likelihood", description="diffusion_likelihood | fallback_confidence")
    ots_allow_fallback_simple_score: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class HybridCheckpointDiag:
    """Diagnostics for one search checkpoint."""
    checkpoint_idx: int
    step_range: Tuple[int, int]
    beams_before_expand: int
    candidates_after_expand: int
    beams_after_prune: int
    scores: List[float]
    masked_remaining: List[int]
    avg_alpha_per_beam: List[float]
    avg_residual_norm_per_beam: List[float]
    elapsed_ms: float


@dataclass
class HybridStepDiag:
    """Diagnostics for a regular (non-checkpoint) denoising step."""
    step: int
    avg_alpha: float
    avg_residual_norm: float
    masked_remaining: int
    elapsed_ms: float


@dataclass
class Inference2Diagnostics:
    """Rich diagnostics for the inferenza2 hybrid engine."""
    # OTS-side
    checkpoints: List[HybridCheckpointDiag] = field(default_factory=list)
    total_beams_explored: int = 0
    chosen_beam_index: int = 0
    chosen_beam_score: float = 0.0
    total_search_checkpoints: int = 0
    beam_size: int = 1
    pruning_mode_used: str = "diffusion_likelihood"

    # RCD-side
    warm_start_used: bool = False
    reference_model_used: bool = False
    t_res: float = 1.0
    step_diagnostics: List[HybridStepDiag] = field(default_factory=list)

    # General
    total_denoising_steps: int = 0
    total_elapsed_ms: float = 0.0
    finish_reason: str = ""

    # Hybrid-specific
    hybrid_mode_active: str = "rcd_plus_ots"
    rcd_enabled: bool = True
    ots_enabled: bool = True
    approximations_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoints": [
                {
                    "checkpoint_idx": c.checkpoint_idx,
                    "step_range": list(c.step_range),
                    "beams_before_expand": c.beams_before_expand,
                    "candidates_after_expand": c.candidates_after_expand,
                    "beams_after_prune": c.beams_after_prune,
                    "scores": c.scores,
                    "masked_remaining": c.masked_remaining,
                    "avg_alpha_per_beam": c.avg_alpha_per_beam,
                    "avg_residual_norm_per_beam": c.avg_residual_norm_per_beam,
                    "elapsed_ms": c.elapsed_ms,
                }
                for c in self.checkpoints
            ],
            "step_diagnostics": [
                {
                    "step": s.step,
                    "avg_alpha": s.avg_alpha,
                    "avg_residual_norm": s.avg_residual_norm,
                    "masked_remaining": s.masked_remaining,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.step_diagnostics
            ],
            "total_beams_explored": self.total_beams_explored,
            "chosen_beam_index": self.chosen_beam_index,
            "chosen_beam_score": self.chosen_beam_score,
            "total_search_checkpoints": self.total_search_checkpoints,
            "beam_size": self.beam_size,
            "pruning_mode_used": self.pruning_mode_used,
            "warm_start_used": self.warm_start_used,
            "reference_model_used": self.reference_model_used,
            "t_res": self.t_res,
            "total_denoising_steps": self.total_denoising_steps,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
            "hybrid_mode_active": self.hybrid_mode_active,
            "rcd_enabled": self.rcd_enabled,
            "ots_enabled": self.ots_enabled,
            "approximations_used": self.approximations_used,
        }


# ---------------------------------------------------------------------------
# Hybrid beam state — carries BOTH OTS trajectory + RCD residual state
# ---------------------------------------------------------------------------
@dataclass
class HybridBeamState:
    """One partial denoising trajectory with beam-local RCD residual state.

    OTS fields:
        seq — current token ids (1, total_len)
        prompt_len — number of prompt tokens
        cumulative_score — sum of block scores from OTS pruning
        block_scores — per-checkpoint action-level scores
        step — last denoising step index
        reveal_trace — (step, n_positions_revealed) per checkpoint

    RCD fields (beam-local, NOT shared across beams after divergence):
        alpha_prev — (1, S) normalized entropy weights from previous step
        delta_prev — (1, S, D) residual vectors from previous step
    """
    seq: torch.Tensor               # (1, total_len)
    prompt_len: int
    cumulative_score: float = 0.0
    block_scores: List[float] = field(default_factory=list)
    step: int = 0
    reveal_trace: List[Tuple[int, int]] = field(default_factory=list)
    # RCD state — per beam, per step
    alpha_prev: Optional[torch.Tensor] = None   # (1, S)
    delta_prev: Optional[torch.Tensor] = None   # (1, S, D)


def clone_hybrid_beam(state: HybridBeamState) -> HybridBeamState:
    """Deep-clone a hybrid beam, including independent RCD state."""
    return HybridBeamState(
        seq=state.seq.clone(),
        prompt_len=state.prompt_len,
        cumulative_score=state.cumulative_score,
        block_scores=list(state.block_scores),
        step=state.step,
        reveal_trace=list(state.reveal_trace),
        alpha_prev=state.alpha_prev.clone() if state.alpha_prev is not None else None,
        delta_prev=state.delta_prev.clone() if state.delta_prev is not None else None,
    )


# ---------------------------------------------------------------------------
# Hybrid forward — RCD-enhanced model forward for a single beam
# ---------------------------------------------------------------------------
def _hybrid_forward(
    model: Any,
    beam: HybridBeamState,
    mask_id: int,
    embed_layer: torch.nn.Module,
    use_rcd: bool = True,
    force_noncausal_ctx: Any = None,
) -> torch.Tensor:
    """Run model forward with RCD-enhanced inputs_embeds if enabled.

    If use_rcd is False or beam has no RCD state, falls back to plain
    input_ids forward (equivalent to standard OTS).

    Returns: logits (1, S, V).
    """
    if use_rcd and beam.alpha_prev is not None and beam.delta_prev is not None:
        # Build RCD-augmented input embeddings (Eq.2 from RCD paper)
        inputs_embeds = build_rcd_inputs_embeds(
            beam.seq, mask_id, embed_layer,
            beam.alpha_prev, beam.delta_prev,
            force_mask_only=True,
        )
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(inputs_embeds=inputs_embeds)
        else:
            out = model(inputs_embeds=inputs_embeds)
    else:
        # Plain forward (no RCD)
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(input_ids=beam.seq)
        else:
            out = model(input_ids=beam.seq)
    return out.logits


# ---------------------------------------------------------------------------
# Hybrid RCD state update
# ---------------------------------------------------------------------------
def update_hybrid_rcd_state(
    beam: HybridBeamState,
    logits: torch.Tensor,
    embed_weight: torch.Tensor,
    effective_v: int,
    t_res: float,
) -> None:
    """Update beam-local RCD residual state after a forward pass.

    Algorithm 2, lines 22-26 (RCD paper):
    - alpha from temperature-scaled entropy (Eq.3 + Sec 3.3)
    - delta from probability-weighted embedding projection (Eq.1)
    """
    # Temperature-scaled probs for entropy weight
    res_probs = F.softmax(logits.float() / max(t_res, 1e-6), dim=-1)
    beam.alpha_prev = compute_normalized_entropy(res_probs, effective_v)
    # Delta from original (non-temperature-scaled) probs through codebook
    orig_probs = F.softmax(logits.float(), dim=-1)
    beam.delta_prev = compute_rcd_residuals_from_probs(orig_probs, embed_weight)


# ---------------------------------------------------------------------------
# Hybrid candidate expansion
# ---------------------------------------------------------------------------
def expand_hybrid_candidates(
    model: Any,
    beam: HybridBeamState,
    beam_size: int,
    mask_id: int,
    tokens_per_step: int,
    gumbel_temp: float,
    embed_layer: torch.nn.Module,
    embed_weight: torch.Tensor,
    effective_v: int,
    t_res: float,
    use_rcd: bool = True,
    force_noncausal_ctx: Any = None,
) -> List[Tuple[HybridBeamState, torch.Tensor, torch.Tensor]]:
    """Expand one hybrid beam into K children.

    Uses RCD-enhanced forward to get logits, then applies Gumbel noise
    for diverse exploration (OTS Alg.1 lines 9-12).

    Each child inherits a CLONE of the parent's RCD state, then after
    expansion each child's RCD state is updated independently based on
    the expansion forward's logits.

    Returns: list of (child_state, newly_revealed_mask, x0_full)
    """
    children: List[Tuple[HybridBeamState, torch.Tensor, torch.Tensor]] = []
    prompt_len = beam.prompt_len

    with torch.inference_mode():
        logits = _hybrid_forward(
            model, beam, mask_id, embed_layer,
            use_rcd=use_rcd, force_noncausal_ctx=force_noncausal_ctx,
        )

    gen_logits = logits[:, prompt_len:, :]

    for _ in range(beam_size):
        # Gumbel-perturbed logits → diverse x0 candidates
        noisy = _add_gumbel_noise(gen_logits, gumbel_temp)
        x0_gen = noisy.argmax(dim=-1)  # (1, gen_len)
        conf = F.softmax(noisy.float(), dim=-1).max(dim=-1).values  # (1, gen_len)

        child = clone_hybrid_beam(beam)
        gen_before = child.seq[:, prompt_len:]

        # Transfer tokens: joint order-token exploration
        updated, revealed_gen = _transfer_tokens(
            gen_before, x0_gen, conf, mask_id, tokens_per_step,
        )
        child.seq[:, prompt_len:] = updated

        # Full revealed mask (including prompt dim for scoring)
        full_revealed = torch.zeros_like(child.seq, dtype=torch.bool)
        full_revealed[:, prompt_len:] = revealed_gen

        # x0_full for scoring: prompt tokens + model's full prediction
        x0_full = child.seq.clone()
        x0_full[:, prompt_len:] = x0_gen

        # Update child's RCD state independently
        # Use the logits from the expansion forward (shared across children
        # from same parent, but each child's alpha/delta will diverge on
        # subsequent steps since their sequences now differ)
        if use_rcd:
            update_hybrid_rcd_state(child, logits, embed_weight, effective_v, t_res)

        children.append((child, full_revealed, x0_full))

    return children


# ---------------------------------------------------------------------------
# Hybrid scoring — diffusion-native + optional RCD context
# ---------------------------------------------------------------------------
def score_hybrid_candidate(
    model: Any,
    candidate: HybridBeamState,
    newly_revealed: torch.Tensor,
    x0_full: torch.Tensor,
    mask_id: int,
    force_noncausal_ctx: Any = None,
) -> float:
    """Diffusion-native action-level likelihood scorer for hybrid engine.

    Follows OTS Eq.2 / Fig.3:
    1. Take model's full-sequence prediction x0
    2. In x0, mask ONLY newly revealed positions
    3. Forward the DLM on this scoring input
    4. Sum log-probs of actual tokens at revealed positions

    DESIGN DECISION: The scoring forward uses plain input_ids
    (not RCD-augmented inputs_embeds).  Rationale:
    - The scoring function evaluates the MODEL's support for the
      denoising action, independent of RCD enhancement.
    - Using RCD state in scoring would conflate two signals.
    - The OTS paper's scoring assumes standard model forward.
    - This is the most principled modular approximation.

    If this approximation proves limiting, a future version could
    score with RCD-augmented forward; the interface is isolated
    for easy upgrade.
    """
    prompt_len = candidate.prompt_len
    gen_revealed = newly_revealed[:, prompt_len:]
    n_revealed = int(gen_revealed.sum().item())
    if n_revealed == 0:
        return 0.0

    # Build scoring input: x0_full with newly revealed positions re-masked
    score_input = x0_full.clone()
    score_input[newly_revealed] = mask_id

    # Plain forward (not RCD-augmented — see docstring)
    with torch.inference_mode():
        if force_noncausal_ctx is not None:
            with force_noncausal_ctx(model):
                out = model(input_ids=score_input)
        else:
            out = model(input_ids=score_input)
        score_logits = out.logits

    # Log-prob of actual tokens at newly revealed positions
    log_probs = F.log_softmax(score_logits.float(), dim=-1)
    target_ids = candidate.seq[newly_revealed]
    revealed_idx = newly_revealed.nonzero(as_tuple=False)
    score_sum = 0.0
    for i in range(n_revealed):
        b, s = int(revealed_idx[i, 0]), int(revealed_idx[i, 1])
        tok = int(target_ids[i])
        score_sum += float(log_probs[b, s, tok].item())
    return score_sum


def score_hybrid_candidate_fallback(
    candidate: HybridBeamState,
    confidence: torch.Tensor,
    newly_revealed: torch.Tensor,
) -> float:
    """Fallback scoring: sum of confidence at newly revealed positions.
    APPROXIMATION: NOT paper-faithful.  Reported in diagnostics."""
    n = int(newly_revealed.sum().item())
    if n == 0:
        return 0.0
    return float(confidence[newly_revealed].sum().item())


def prune_hybrid_beams(
    beams: List[HybridBeamState],
    beam_size: int,
) -> List[HybridBeamState]:
    """Keep top-K beams by cumulative score."""
    if len(beams) <= beam_size:
        return beams
    beams.sort(key=lambda b: b.cumulative_score, reverse=True)
    return beams[:beam_size]


# ---------------------------------------------------------------------------
# Warm-start initialization for hybrid beams
# ---------------------------------------------------------------------------
def initialize_hybrid_warm_start(
    model: Any,
    init_seq: torch.Tensor,
    mask_id: int,
    embed_weight: torch.Tensor,
    effective_v: int,
    t_res: float = 1.0,
    force_noncausal_ctx: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warm-start: run one forward to get initial RCD state (α₀, Δ₀).
    Delegates to the proven RCD helper."""
    return initialize_rcd_warm_start(
        model, init_seq, mask_id, embed_weight, effective_v,
        t_res=t_res, force_noncausal_ctx=force_noncausal_ctx,
    )


# ---------------------------------------------------------------------------
# Main hybrid decode loop
# ---------------------------------------------------------------------------
def inferenza2_decode(
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
    seed: int = 42,
    is_dummy: bool = False,
    force_noncausal_ctx: Any = None,
    # RCD knobs
    rcd_enabled: bool = True,
    t_res: float = 1.0,
    force_mask_only: bool = True,
    warm_start: bool = True,
    warm_start_model: Any = None,
    # OTS knobs
    ots_enabled: bool = True,
    beam_size: int = 3,
    gumbel_temperature: float = 0.6,
    search_interval: int = 0,
    pruning_mode: str = "diffusion_likelihood",
    allow_fallback_score: bool = False,
    # Diagnostics
    store_diagnostics: bool = True,
    store_trace: bool = True,
) -> Tuple[str, Dict[str, Any], Inference2Diagnostics]:
    """Inferenza2: OTS-over-RCD hybrid decode loop.

    Outer loop: OTS search with beams, expansion, pruning.
    Inner mechanism: each beam uses RCD-enhanced denoising.

    Returns: (text, stats_dict, diagnostics).
    """
    from .formulas import llada21_apply
    from .diffusion import apply_remask
    from .config import RemaskConfig
    from .utils import seed_everything, tokens_per_second

    seed_everything(seed)

    # Resolve effective mode
    effective_beam_size = beam_size if ots_enabled else 1
    use_rcd = rcd_enabled
    use_ots_search = ots_enabled and effective_beam_size > 1

    # Determine hybrid mode label
    if use_rcd and use_ots_search:
        hybrid_mode = "rcd_plus_ots"
    elif use_rcd:
        hybrid_mode = "rcd_only"
    elif use_ots_search:
        hybrid_mode = "ots_only"
    else:
        hybrid_mode = "baseline"

    diag = Inference2Diagnostics(
        beam_size=effective_beam_size,
        pruning_mode_used=pruning_mode,
        t_res=t_res,
        hybrid_mode_active=hybrid_mode,
        rcd_enabled=use_rcd,
        ots_enabled=use_ots_search,
    )

    # Track approximations
    if use_rcd:
        diag.approximations_used.append("rcd_no_trained_weights")
        diag.approximations_used.append("rcd_hf_inputs_embeds_interface")
    if use_ots_search:
        diag.approximations_used.append("ots_scoring_uses_plain_forward_not_rcd_augmented")

    # Validate pruning mode
    if pruning_mode == "diffusion_likelihood":
        use_diffusion_score = True
    elif pruning_mode == "fallback_confidence":
        use_diffusion_score = False
        diag.approximations_used.append("fallback_confidence_scoring")
    else:
        if not allow_fallback_score:
            raise ValueError(
                f"Unknown pruning_mode '{pruning_mode}' and "
                "ots_allow_fallback_simple_score is disabled"
            )
        use_diffusion_score = False
        diag.pruning_mode_used = "fallback_confidence"
        diag.approximations_used.append("fallback_confidence_scoring")

    # ---- Encode prompt ----
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    gen_len = max_new_tokens

    # Build initial sequence: [prompt] [MASK^L]
    init_seq = torch.full(
        (1, prompt_len + gen_len), mask_id, dtype=torch.long, device=device,
    )
    init_seq[:, :prompt_len] = input_ids

    # ---- RCD setup: embedding layer + codebook ----
    embed_layer: Optional[torch.nn.Module] = None
    embed_weight: Optional[torch.Tensor] = None
    effective_v = vocab_size
    if use_rcd:
        embed_layer = _get_embedding_layer(model)
        embed_weight = embed_layer.weight.detach()
        actual_v = embed_weight.shape[0]
        effective_v = min(vocab_size, actual_v)

    # ---- Search interval ----
    if search_interval <= 0:
        n_blocks = max(2, gen_len // 32)
        search_interval = max(1, max_steps // n_blocks)
    tokens_per_step_base = max(1, gen_len // max(1, max_steps))

    # ---- Initialize RCD warm-start state ----
    init_alpha: Optional[torch.Tensor] = None
    init_delta: Optional[torch.Tensor] = None
    if use_rcd and warm_start:
        ws_model = warm_start_model if warm_start_model is not None else model
        init_alpha, init_delta = initialize_hybrid_warm_start(
            ws_model, init_seq, mask_id, embed_weight, effective_v,
            t_res=t_res, force_noncausal_ctx=force_noncausal_ctx,
        )
        diag.warm_start_used = True
        diag.reference_model_used = warm_start_model is not None
        if warm_start_model is None:
            diag.approximations_used.append("rcd_warm_start_same_model_fallback")

    # ---- Initialize K identical beams (OTS Alg.1 line 2) ----
    beams: List[HybridBeamState] = []
    for _ in range(effective_beam_size):
        b = HybridBeamState(
            seq=init_seq.clone(),
            prompt_len=prompt_len,
        )
        # Each beam gets its own clone of the warm-start RCD state
        if use_rcd:
            if init_alpha is not None and init_delta is not None:
                b.alpha_prev = init_alpha.clone()
                b.delta_prev = init_delta.clone()
            else:
                # No warm start: zero alpha, base embeddings as delta
                S = init_seq.shape[1]
                b.alpha_prev = torch.zeros(1, S, device=device, dtype=torch.float32)
                b.delta_prev = embed_layer(init_seq).detach()
        beams.append(b)

    remask_cfg = RemaskConfig()
    model.eval()
    t0 = time.time()
    checkpoint_idx = 0
    total_expansions = 0

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            eos_token_id = int(eos_token_id)
        except Exception:
            eos_token_id = None

    finish_reason = "length"

    # =======================================================================
    # MAIN DECODE LOOP — OTS outer controller, RCD inner mechanism
    # =======================================================================
    with torch.inference_mode():
        for step in range(max_steps):
            is_search_checkpoint = (
                use_ots_search
                and step > 0
                and step % search_interval == 0
            )

            if is_search_checkpoint:
                # ============================================================
                # SEARCH CHECKPOINT: expand + score + prune
                # ============================================================
                ckpt_t0 = time.time()
                all_candidates: List[HybridBeamState] = []
                n_before = len(beams)

                for beam in beams:
                    children = expand_hybrid_candidates(
                        model, beam, effective_beam_size, mask_id,
                        tokens_per_step_base, gumbel_temperature,
                        embed_layer, embed_weight, effective_v, t_res,
                        use_rcd=use_rcd,
                        force_noncausal_ctx=force_noncausal_ctx,
                    )
                    for child, revealed, x0_full in children:
                        # Score the child
                        if use_diffusion_score:
                            block_score = score_hybrid_candidate(
                                model, child, revealed, x0_full,
                                mask_id, force_noncausal_ctx,
                            )
                        else:
                            # Fallback confidence scoring
                            gen_logits = _hybrid_forward(
                                model, beam, mask_id, embed_layer,
                                use_rcd=use_rcd,
                                force_noncausal_ctx=force_noncausal_ctx,
                            )[:, prompt_len:, :]
                            conf = F.softmax(gen_logits.float(), dim=-1).max(dim=-1).values
                            full_conf = torch.zeros(1, child.seq.shape[1], device=device)
                            full_conf[:, prompt_len:] = conf
                            block_score = score_hybrid_candidate_fallback(
                                child, full_conf, revealed,
                            )

                        child.block_scores.append(block_score)
                        child.cumulative_score = sum(child.block_scores)
                        child.step = step

                        n_revealed = int(revealed.sum().item())
                        child.reveal_trace.append((step, n_revealed))
                        all_candidates.append(child)

                total_expansions += len(all_candidates)

                # Prune to top-K
                beams = prune_hybrid_beams(all_candidates, effective_beam_size)

                ckpt_elapsed = (time.time() - ckpt_t0) * 1000.0
                if store_trace:
                    diag.checkpoints.append(HybridCheckpointDiag(
                        checkpoint_idx=checkpoint_idx,
                        step_range=(max(0, step - search_interval), step),
                        beams_before_expand=n_before,
                        candidates_after_expand=len(all_candidates),
                        beams_after_prune=len(beams),
                        scores=[b.cumulative_score for b in beams],
                        masked_remaining=[
                            int(b.seq[:, prompt_len:].eq(mask_id).sum().item())
                            for b in beams
                        ],
                        avg_alpha_per_beam=[
                            float(b.alpha_prev[:, prompt_len:].mean().item())
                            if b.alpha_prev is not None else 0.0
                            for b in beams
                        ],
                        avg_residual_norm_per_beam=[
                            float(b.delta_prev[:, prompt_len:, :].norm(dim=-1).mean().item())
                            if b.delta_prev is not None else 0.0
                            for b in beams
                        ],
                        elapsed_ms=ckpt_elapsed,
                    ))
                checkpoint_idx += 1

            else:
                # ============================================================
                # REGULAR DENOISING STEP — RCD-enhanced forward for each beam
                # ============================================================
                step_t0 = time.time()
                new_beams: List[HybridBeamState] = []

                for beam in beams:
                    # Step a,b: RCD-enhanced forward
                    logits = _hybrid_forward(
                        model, beam, mask_id, embed_layer,
                        use_rcd=use_rcd,
                        force_noncausal_ctx=force_noncausal_ctx,
                    )

                    # Step c: Obtain predictions + confidence
                    gen_logits = logits[:, prompt_len:, :]
                    noisy = _add_gumbel_noise(gen_logits, gumbel_temperature)
                    x0_gen = noisy.argmax(dim=-1)
                    conf = F.softmax(gen_logits.float(), dim=-1).max(dim=-1).values

                    gen_before = beam.seq[:, prompt_len:]

                    # Selection + Update via proven llada21_apply
                    updated, sets = llada21_apply(
                        gen_before, x0_gen, conf,
                        mask_id, tau_mask, tau_edit,
                    )

                    # Remask on non-final steps
                    if step + 1 < max_steps:
                        updated = apply_remask(updated, conf, mask_id, remask_cfg)

                    beam.seq[:, prompt_len:] = updated
                    beam.step = step

                    # Step d: Update beam-local RCD state
                    if use_rcd:
                        update_hybrid_rcd_state(
                            beam, logits, embed_weight, effective_v, t_res,
                        )

                    new_beams.append(beam)

                beams = new_beams

                # Step diagnostics
                step_elapsed = (time.time() - step_t0) * 1000.0
                if store_diagnostics:
                    # Average across all beams
                    avg_alpha = 0.0
                    avg_rnorm = 0.0
                    avg_remain = 0
                    for b in beams:
                        avg_remain += int(b.seq[:, prompt_len:].eq(mask_id).sum().item())
                        if b.alpha_prev is not None:
                            avg_alpha += float(b.alpha_prev[:, prompt_len:].mean().item())
                        if b.delta_prev is not None:
                            avg_rnorm += float(b.delta_prev[:, prompt_len:, :].norm(dim=-1).mean().item())
                    n_beams = max(1, len(beams))
                    diag.step_diagnostics.append(HybridStepDiag(
                        step=step,
                        avg_alpha=avg_alpha / n_beams,
                        avg_residual_norm=avg_rnorm / n_beams,
                        masked_remaining=avg_remain // n_beams,
                        elapsed_ms=step_elapsed,
                    ))

            # ---- Check early stop ----
            any_alive = False
            for beam in beams:
                gen = beam.seq[:, prompt_len:]
                remain = int(gen.eq(mask_id).sum().item())
                if remain > 0:
                    any_alive = True
                if eos_token_id is not None:
                    eos_pos = (gen[0] == eos_token_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        finish_reason = "eos"

            if not any_alive:
                finish_reason = "converged"
                break

    # =======================================================================
    # Select best beam (OTS Alg.1 line 30)
    # =======================================================================
    best_idx = 0
    best_score = beams[0].cumulative_score if beams else 0.0
    for i, b in enumerate(beams):
        if b.cumulative_score > best_score:
            best_score = b.cumulative_score
            best_idx = i

    best = beams[best_idx]
    elapsed = max(1e-6, time.time() - t0)

    # Decode text
    out_ids = best.seq[0, prompt_len:]
    if eos_token_id is not None:
        eos_pos = (out_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            out_ids = out_ids[:max(0, int(eos_pos[0][0].item()) + 1)]

    tokens_generated = int((out_ids != mask_id).sum().item())
    text = _decode_text_hybrid(tokenizer, out_ids, mask_id, is_dummy)

    diag.total_beams_explored = total_expansions + effective_beam_size
    diag.chosen_beam_index = best_idx
    diag.chosen_beam_score = best_score
    diag.total_denoising_steps = max_steps
    diag.total_search_checkpoints = checkpoint_idx
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason

    stats: Dict[str, Any] = {
        "engine": "inferenza2",
        "mode": "INFERENZA2",
        "hybrid_mode": hybrid_mode,
        "rcd_enabled": use_rcd,
        "ots_enabled": use_ots_search,
        "tau_mask": tau_mask,
        "tau_edit": tau_edit,
        "beam_size": effective_beam_size,
        "gumbel_temperature": gumbel_temperature,
        "search_interval": search_interval,
        "steps": max_steps,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_second(tokens_generated, elapsed),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "dummy_model": is_dummy,
        "pruning_mode": diag.pruning_mode_used,
        "total_search_checkpoints": checkpoint_idx,
        "chosen_beam_index": best_idx,
        "chosen_beam_score": best_score,
        "t_res": t_res,
        "warm_start_used": diag.warm_start_used,
        "approximations_used": diag.approximations_used,
        "inferenza2_diagnostics": diag.to_dict(),
    }

    return text.strip(), stats, diag


# ---------------------------------------------------------------------------
# Text decode helper
# ---------------------------------------------------------------------------
def _decode_text_hybrid(tok: Any, out_ids: torch.Tensor, mask_id: int, is_dummy: bool) -> str:
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

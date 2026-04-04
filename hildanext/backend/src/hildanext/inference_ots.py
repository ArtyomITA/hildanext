# Order-Token Search (OTS) inference engine for dLLM.
# Paper: "Improving Diffusion LM Decoding through Joint Search
#         in Generation Order and Token Space" (Shen et al., 2026)
# arXiv:2601.20339 — Algorithm 1, Eq.(2), Sec. 4.1/4.2.
#
# KEY IDEA: OTS searches in the JOINT space of generation order + token
# values.  It is NOT plain beam search.  Standard AR beam search only
# varies tokens for a fixed left-to-right order.  OTS varies BOTH which
# positions are revealed and which tokens are written.
#
# PAPER-FAITHFUL PARTS:
#   - Multi-beam denoising with expansion at block boundaries
#   - Gumbel-noise diverse sampling (Alg.1 line 10)
#   - Diffusion-native block scoring: re-mask newly revealed block in
#     the model's own full prediction x0, then measure log p of those
#     positions (Eq.2 / Fig.3)
#   - Cumulative trajectory score = sum of block scores (Sec 4.2)
#   - Top-K pruning after each search checkpoint
#
# APPROXIMATIONS (documented in code):
#   - The codebase is full-sequence bidirectional (not semi-AR block
#     diffusion).  We simulate "blocks" by grouping denoising steps
#     into checkpoint intervals of `search_interval` steps.
#   - Within each simulated block we denoise only the positions selected
#     at the block boundary, then score/prune at the end of the interval.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class InferenceOTSRequest(BaseModel):
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
    # ---- OTS-specific knobs ----
    ots_beam_size: int = Field(default=3, ge=1, le=32, description="K — number of beams")
    ots_gumbel_temperature: float = Field(default=0.6, ge=0.0, le=5.0, description="τ for Gumbel noise (Alg.1)")
    ots_search_interval: int = Field(default=0, ge=0, le=256, description="N steps between search checkpoints; 0=auto")
    ots_block_size: int = Field(default=32, ge=1, le=512, description="Contiguous block size (paper default 32, Alg.1 block_idx)")
    ots_pruning_mode: str = Field(default="diffusion_likelihood", description="diffusion_likelihood | fallback_confidence")
    ots_store_search_trace: bool = Field(default=True)
    ots_return_diagnostics: bool = Field(default=True)
    ots_allow_fallback_simple_score: bool = Field(default=False)
    ots_seed: Optional[int] = None
    ots_deterministic: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class OTSCheckpointDiag:
    checkpoint_idx: int
    step_range: Tuple[int, int]
    beams_before_expand: int
    candidates_after_expand: int
    beams_after_prune: int
    scores: List[float]
    masked_remaining: List[int]
    elapsed_ms: float


@dataclass
class OTSSearchDiagnostics:
    checkpoints: List[OTSCheckpointDiag] = field(default_factory=list)
    total_beams_explored: int = 0
    chosen_beam_index: int = 0
    chosen_beam_score: float = 0.0
    total_denoising_steps: int = 0
    total_search_checkpoints: int = 0
    pruning_mode_used: str = "diffusion_likelihood"
    total_elapsed_ms: float = 0.0
    finish_reason: str = ""
    beam_size: int = 1

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
                    "elapsed_ms": c.elapsed_ms,
                }
                for c in self.checkpoints
            ],
            "total_beams_explored": self.total_beams_explored,
            "chosen_beam_index": self.chosen_beam_index,
            "chosen_beam_score": self.chosen_beam_score,
            "total_denoising_steps": self.total_denoising_steps,
            "total_search_checkpoints": self.total_search_checkpoints,
            "pruning_mode_used": self.pruning_mode_used,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
            "beam_size": self.beam_size,
        }


# ---------------------------------------------------------------------------
# Search state per beam
# ---------------------------------------------------------------------------
@dataclass
class OTSBeamState:
    """Tracks one partial denoising trajectory."""
    seq: torch.Tensor            # (1, total_len) current token ids
    prompt_len: int
    cumulative_score: float = 0.0
    block_scores: List[float] = field(default_factory=list)
    step: int = 0
    # Lightweight trace: list of (step, positions_revealed) per checkpoint
    reveal_trace: List[Tuple[int, int]] = field(default_factory=list)
    # Positions that have already been scored and should stay fixed.
    committed_mask: Optional[torch.Tensor] = None
    # Positions currently being denoised inside the active block.
    current_block_mask: Optional[torch.Tensor] = None
    # Model's full prediction from x_t (parent, pre-expansion) with Gumbel
    # noise, stored for Eq.(2) scoring.  Avoids re-running the model on x_s.
    x0_for_scoring: Optional[torch.Tensor] = None
    # Index of the contiguous block currently being processed (Alg.1 line 8).
    # Advances by 1 after each scored checkpoint.
    block_idx: int = 0


def clone_search_state(state: OTSBeamState) -> OTSBeamState:
    return OTSBeamState(
        seq=state.seq.clone(),
        prompt_len=state.prompt_len,
        cumulative_score=state.cumulative_score,
        block_scores=list(state.block_scores),
        step=state.step,
        reveal_trace=list(state.reveal_trace),
        committed_mask=(
            state.committed_mask.clone() if state.committed_mask is not None else None
        ),
        current_block_mask=(
            state.current_block_mask.clone() if state.current_block_mask is not None else None
        ),
        x0_for_scoring=(
            state.x0_for_scoring.clone() if state.x0_for_scoring is not None else None
        ),
        block_idx=state.block_idx,
    )


def _zero_mask_like(seq: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(seq, dtype=torch.bool)


def _ensure_beam_masks(state: OTSBeamState) -> None:
    if state.committed_mask is None:
        state.committed_mask = _zero_mask_like(state.seq)
        state.committed_mask[:, :state.prompt_len] = True
    if state.current_block_mask is None:
        state.current_block_mask = _zero_mask_like(state.seq)


# ---------------------------------------------------------------------------
# Core helpers (paper-faithful where possible)
# ---------------------------------------------------------------------------

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Alg.1 line 10: l̃ = add_gumbel_noise(l, τ).
    Gumbel-max trick: argmax(logits + Gumbel(0,1)*τ) ≈ sample from
    softmax(logits/τ), producing diverse candidates from same logits."""
    if temperature <= 0:
        return logits
    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10)))
    return logits + gumbel * temperature


def _transfer_tokens(
    current: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    mask_id: int,
    tokens_to_reveal: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transfer top-`tokens_to_reveal` confident predictions from x0 into
    the current sequence, only at currently masked positions.
    Returns: (updated_seq, newly_revealed_mask).
    Jointly varies ORDER (which positions get revealed based on confidence
    ranking) and TOKENS (x0 was sampled with Gumbel noise → different
    token values). This is the joint order-token exploration."""
    out = current.clone()
    is_mask = (current == mask_id)
    if not is_mask.any() or tokens_to_reveal <= 0:
        return out, torch.zeros_like(current, dtype=torch.bool)
    # Confidence only at masked positions; committed positions get -inf
    score = confidence.clone()
    score[~is_mask] = -float("inf")
    flat = score.view(-1)
    k = min(tokens_to_reveal, int(is_mask.sum().item()))
    if k <= 0:
        return out, torch.zeros_like(current, dtype=torch.bool)
    _, top_idx = torch.topk(flat, k=k, largest=True)
    out.view(-1)[top_idx] = x0.view(-1)[top_idx]
    revealed = torch.zeros_like(current, dtype=torch.bool)
    revealed.view(-1)[top_idx] = True
    return out, revealed


def _forward_model(
    model: Any,
    seq: torch.Tensor,
    force_noncausal_ctx: Any = None,
) -> torch.Tensor:
    """Run model forward, return logits (B, S, V)."""
    model.eval()
    if force_noncausal_ctx is not None:
        with force_noncausal_ctx(model):
            out = model(input_ids=seq)
    else:
        out = model(input_ids=seq)
    return out.logits


def _denoise_active_block_step(
    current: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int,
    tau_mask: float,
    tau_edit: float,
) -> torch.Tensor:
    """Denoise only positions that belong to the current active block."""
    out = current.clone()
    active = active_mask.bool()
    if not active.any():
        return out

    reveal = active & current.eq(mask_id) & confidence.ge(float(tau_mask))
    if reveal.any():
        out[reveal] = x0[reveal]

    edit = active & current.ne(mask_id) & x0.ne(current) & confidence.ge(float(tau_edit))
    if edit.any():
        out[edit] = x0[edit]

    return out


def _apply_block_remask(
    tokens: torch.Tensor,
    confidence: torch.Tensor,
    active_mask: torch.Tensor,
    mask_id: int,
    target_ratio: float,
    min_ratio: float,
) -> torch.Tensor:
    """Low-confidence remasking restricted to the active block."""
    out = tokens.clone()
    candidate = active_mask & out.ne(mask_id)
    total = int(candidate.sum().item())
    if total <= 0:
        return out

    k = max(int(total * float(min_ratio)), int(total * float(target_ratio)))
    if k <= 0:
        return out

    score = confidence.clone()
    score[~candidate] = 2.0
    flat = score.view(-1)
    k = min(k, int(candidate.sum().item()))
    if k <= 0:
        return out

    _, idx = torch.topk(flat, k=k, largest=False)
    out.view(-1)[idx] = mask_id
    return out


def _build_x0_full(
    model: Any,
    beam: OTSBeamState,
    force_noncausal_ctx: Any = None,
) -> torch.Tensor:
    """Model full prediction x0 for the current partial sequence."""
    logits = _forward_model(model, beam.seq, force_noncausal_ctx)
    x0_full = beam.seq.clone()
    # Left-shift alignment: logits[j] predicts token[j+1]
    x0_full[:, beam.prompt_len:] = logits[:, beam.prompt_len-1:-1, :].argmax(dim=-1)
    return x0_full


def score_ots_candidate(
    model: Any,
    candidate_state: OTSBeamState,
    newly_revealed: torch.Tensor,
    x0_full: torch.Tensor,
    mask_id: int,
    force_noncausal_ctx: Any = None,
) -> float:
    """Diffusion-native likelihood estimator (Eq.2, Fig.3).

    Paper procedure:
    1. Take the model's full-sequence prediction x0 (all tokens predicted
       including unrevealed ones).
    2. In x0, mask ONLY the newly revealed positions (the "current block").
    3. Run the DLM forward on this partially-masked x0.
    4. Measure log-probability of the actual token at each newly-revealed
       position.  Sum = block score.

    This scores the specific denoising ACTION that produced the candidate,
    not a monolithic full-sequence likelihood.  This is the key insight
    from Section 4.2: incremental action scoring is more stable than
    scoring entire partially-denoised states.

    NOT AR-style scoring: we condition on the model's own full prediction
    (including predicted future content), which gives context for scoring
    the block.  AR-style would only condition on revealed-so-far.
    """
    prompt_len = candidate_state.prompt_len
    gen_revealed = newly_revealed[:, prompt_len:]
    n_revealed = int(gen_revealed.sum().item())
    if n_revealed == 0:
        return 0.0

    # Build scoring input: x0_full with newly revealed positions re-masked
    score_input = x0_full.clone()
    score_input[newly_revealed] = mask_id

    # Forward pass on this scoring input
    with torch.inference_mode():
        score_logits = _forward_model(model, score_input, force_noncausal_ctx)

    # Log-prob of actual tokens at newly revealed positions.
    # USE x0_full as targets (paper Eq.(2): log p_θ(x0 | b(xs,xt,x0))).
    # x0_full is the Gumbel-sampled full prediction from x_t; it has
    # definite (non-MASK) tokens at every position including those not
    # yet committed in x_s.  Using candidate_state.seq would give MASK
    # as the target for uncommitted block positions, distorting scores.
    log_probs = F.log_softmax(score_logits.float(), dim=-1)  # (B, S, V)
    target_ids = x0_full[newly_revealed]  # (n_revealed,)  ← x0, not x_s
    # Gather log probs
    revealed_idx = newly_revealed.nonzero(as_tuple=False)  # (n_revealed, 2)
    score_sum = 0.0
    for i in range(n_revealed):
        b, s = int(revealed_idx[i, 0]), int(revealed_idx[i, 1])
        tok = int(target_ids[i])
        # Left-shift alignment: logits[j] predicts token[j+1], so read [j-1]
        score_sum += float(log_probs[b, max(0, s - 1), tok].item())
    return score_sum


def score_ots_candidate_fallback(
    candidate_state: OTSBeamState,
    confidence: torch.Tensor,
    newly_revealed: torch.Tensor,
) -> float:
    """Fallback scoring: summed confidence of newly revealed tokens.
    APPROXIMATION: this is NOT paper-faithful.  It's a simple proxy
    when the dedicated diffusion-native score is unavailable/disabled."""
    n = int(newly_revealed.sum().item())
    if n == 0:
        return 0.0
    return float(confidence[newly_revealed].sum().item())


def expand_ots_candidates(
    model: Any,
    beam: OTSBeamState,
    beam_size: int,
    mask_id: int,
    block_size: int,
    gumbel_temp: float,
    *,
    tokens_per_step: int = 0,
    force_noncausal_ctx: Any = None,
) -> List[Tuple[OTSBeamState, torch.Tensor, torch.Tensor]]:
    """Expand one beam into `beam_size` child candidates.
    Each child is produced by sampling x0 with independent Gumbel noise
    (Alg.1 lines 9-12).  Token filling is restricted to the CURRENT
    CONTIGUOUS BLOCK (beam.block_idx * block_size .. +block_size), making
    the generation order structured and semi-AR compatible (Alg.1 line 8).
    Scoring uses ALL positions in that block against x_full_seq (Alg.1
    lines 13-15, Eq.(2)).

    Returns: list of (child_state, block_mask_full, x0_full)
    """
    children = []
    seq = beam.seq
    prompt_len = beam.prompt_len

    with torch.inference_mode():
        logits = _forward_model(model, seq, force_noncausal_ctx)

    # Left-shift alignment: logits[j] predicts token[j+1]
    gen_logits = logits[:, prompt_len-1:-1, :]

    for _ in range(beam_size):
        # Gumbel-perturbed logits → diverse x0 candidates (Alg.1 line 10-11)
        noisy = _add_gumbel_noise(gen_logits, gumbel_temp)
        x0_gen = noisy.argmax(dim=-1)  # (1, gen_len)
        conf = F.softmax(noisy.float(), dim=-1).max(dim=-1).values  # (1, gen_len)

        child = clone_search_state(beam)
        gen_len_local = seq.shape[1] - prompt_len

        # Current contiguous block boundaries in gen space (Alg.1 line 8).
        blk_s = min(beam.block_idx * block_size, gen_len_local)
        blk_e = min(blk_s + block_size, gen_len_local)

        # x_candidate (Alg.1 line 12): fill only the current block's positions.
        # This enforces semi-AR generation order — earlier blocks are context,
        # later blocks are untouched until their turn.
        gen_before = child.seq[:, prompt_len:]
        # Reveal budget per step: paper Alg.1 line 12 "Only apply L/S
        # predicted tokens".  tokens_per_step=0 falls back to block_size
        # for backward compatibility.
        reveal_budget = tokens_per_step if tokens_per_step > 0 else block_size
        if blk_s < blk_e:
            gen_blk = gen_before[:, blk_s:blk_e]
            updated_blk, _ = _transfer_tokens(
                gen_blk,
                x0_gen[:, blk_s:blk_e],
                conf[:, blk_s:blk_e],
                mask_id,
                reveal_budget,
            )
            updated = gen_before.clone()
            updated[:, blk_s:blk_e] = updated_blk
        else:
            updated = gen_before.clone()
        child.seq[:, prompt_len:] = updated

        # current_block_mask = the FULL contiguous block slice (Alg.1 line 14:
        # x_masked ← mask_tokens(x_full_seq, block_idx)), not sparse positions.
        # This is the structured change of point 5.
        block_mask_full = torch.zeros_like(child.seq, dtype=torch.bool)
        if blk_s < blk_e:
            block_mask_full[:, prompt_len + blk_s: prompt_len + blk_e] = True
        child.current_block_mask = block_mask_full

        # x0_full = x_full_seq (Alg.1 line 13): start from x_t, fill ALL
        # remaining masked gen positions with Gumbel-sampled x0_gen.
        # Conditioning on x_t (not x_s) keeps scoring faithful to Eq.(2).
        x0_full = beam.seq.clone()                          # x_t
        gen_is_mask_xt = beam.seq[:, prompt_len:].eq(mask_id)
        x0_gen_fill = x0_full[:, prompt_len:].clone()
        x0_gen_fill[gen_is_mask_xt] = x0_gen[gen_is_mask_xt]
        x0_full[:, prompt_len:] = x0_gen_fill
        child.x0_for_scoring = x0_full  # Alg.1 line 13; used in score_block

        children.append((child, block_mask_full, x0_full))

    return children


def prune_ots_beams(
    beams: List[OTSBeamState],
    beam_size: int,
) -> List[OTSBeamState]:
    """Keep top-K beams by cumulative score (Alg.1 line 21)."""
    if len(beams) <= beam_size:
        return beams
    beams.sort(key=lambda b: b.cumulative_score, reverse=True)
    return beams[:beam_size]


# ---------------------------------------------------------------------------
# Main OTS decode loop
# ---------------------------------------------------------------------------

def ots_decode(
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
    beam_size: int = 3,
    block_size: int = 32,
    gumbel_temperature: float = 0.6,
    search_interval: int = 0,
    pruning_mode: str = "diffusion_likelihood",
    allow_fallback_score: bool = False,
    store_trace: bool = True,
    seed: int = 42,
    is_dummy: bool = False,
    force_noncausal_ctx: Any = None,
) -> Tuple[str, Dict[str, Any], OTSSearchDiagnostics]:
    """Full Order-Token Search decode loop.

    Approximate OTS structure:
    1. Start from a single masked root trajectory.
    2. At each block boundary, expand every live beam via Gumbel-noise
       sampling to choose the next block positions.
    3. Denoise only that active block for `search_interval` micro-steps.
    4. Score the finished block with the diffusion-native scorer (Eq.2),
       add it to the cumulative trajectory score, then prune to top-K.
    5. Return the best surviving beam.
    """
    from .utils import seed_everything, tokens_per_second

    seed_everything(seed)
    diag = OTSSearchDiagnostics(beam_size=beam_size, pruning_mode_used=pruning_mode)

    # Validate pruning mode
    if pruning_mode == "diffusion_likelihood":
        use_diffusion_score = True
    elif pruning_mode == "fallback_confidence":
        use_diffusion_score = False
    else:
        if not allow_fallback_score:
            raise ValueError(
                f"Unknown pruning_mode '{pruning_mode}' and "
                "ots_allow_fallback_simple_score is disabled"
            )
        use_diffusion_score = False
        diag.pruning_mode_used = "fallback_confidence"

    # Encode prompt
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    gen_len = max_new_tokens

    # Build initial sequence
    init_seq = torch.full(
        (1, prompt_len + gen_len), mask_id, dtype=torch.long, device=device,
    )
    init_seq[:, :prompt_len] = input_ids

    # Search interval: paper uses block_size=32 as default.
    # We use steps-per-block equivalent since codebase is full-sequence.
    #
    # Paper §A.5.1: S = L/2, block_size = 32 (benchmark defaults).
    # For interactive use the effort system controls max_steps.  Ensure
    # at minimum enough steps to process every block at least once
    # (Alg.1 needs ≥B search steps where B = gen_len / block_size).
    n_blocks = max(1, gen_len // max(1, block_size))
    if max_steps < n_blocks:
        max_steps = n_blocks

    if search_interval <= 0:
        # Auto: N = S/B steps between search checkpoints (§A.5.1).
        search_interval = max(1, max_steps // n_blocks)

    # Paper Alg.1 line 12: reveal L/S tokens per step.
    tokens_per_step = max(1, gen_len // max(1, max_steps))

    # Keep a single root beam, then branch at each block boundary.
    root = OTSBeamState(
        seq=init_seq.clone(),
        prompt_len=prompt_len,
    )
    _ensure_beam_masks(root)
    beams: List[OTSBeamState] = [root]

    model.eval()
    t0 = time.time()
    checkpoint_idx = 0
    total_expansions = 0
    steps_run = 0

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            eos_token_id = int(eos_token_id)
        except Exception:
            eos_token_id = None

    finish_reason = "length"

    with torch.inference_mode():
        for block_start in range(0, max_steps, search_interval):
            block_end = min(max_steps, block_start + search_interval)
            ckpt_t0 = time.time()
            n_before = len(beams)
            all_candidates: List[OTSBeamState] = []
            child_count = max(1, beam_size)

            # ---- SEARCH (Alg.1 lines 9-12): expand each beam → K children ----
            for beam in beams:
                children = expand_ots_candidates(
                    model, beam, child_count, mask_id,
                    block_size, gumbel_temperature,
                    tokens_per_step=tokens_per_step,
                    force_noncausal_ctx=force_noncausal_ctx,
                )
                for child, _revealed, _x0_full in children:
                    _ensure_beam_masks(child)
                    all_candidates.append(child)

            total_expansions += len(all_candidates)
            if not all_candidates:
                break

            # ---- SCORE immediately + PRUNE (Alg.1 lines 14-21) ----
            # Paper scores RIGHT AFTER expansion, not after micro-steps.
            # Each candidate's x0_for_scoring was set by expand_ots_candidates
            # from the parent state x_t (Eq.2 conditioning).
            for beam in all_candidates:
                _ensure_beam_masks(beam)
                score_mask = (
                    beam.current_block_mask.clone()
                    if beam.current_block_mask is not None
                    else _zero_mask_like(beam.seq)
                )
                score_mask[:, :prompt_len] = False

                if use_diffusion_score:
                    x0_full = (
                        beam.x0_for_scoring
                        if beam.x0_for_scoring is not None
                        else _build_x0_full(model, beam, force_noncausal_ctx)
                    )
                    block_score = score_ots_candidate(
                        model,
                        beam,
                        score_mask,
                        x0_full,
                        mask_id,
                        force_noncausal_ctx,
                    )
                else:
                    logits = _forward_model(model, beam.seq, force_noncausal_ctx)
                    conf = F.softmax(logits[:, prompt_len:, :].float(), dim=-1).max(dim=-1).values
                    full_conf = torch.zeros(1, beam.seq.shape[1], device=device)
                    full_conf[:, prompt_len:] = conf
                    block_score = score_ots_candidate_fallback(
                        beam,
                        full_conf,
                        score_mask,
                    )

                beam.block_scores.append(block_score)
                beam.cumulative_score += block_score
                beam.reveal_trace.append((block_start, int(score_mask.sum().item())))

            beams = prune_ots_beams(all_candidates, beam_size)

            # ---- NON-SEARCH STEPS (Alg.1 lines 23-26) ----
            # After pruning we denoise only the surviving K beams (not K²).
            # Paper Alg.1 else-branch: Gumbel noise → transfer_tokens.
            # NO remasking — Algorithm 1 has none.
            for step in range(block_start + 1, block_end):
                for beam in beams:
                    gen_before = beam.seq[:, prompt_len:]
                    gen_len_local = gen_before.shape[1]
                    blk_s = min(beam.block_idx * block_size, gen_len_local)
                    blk_e = min(blk_s + block_size, gen_len_local)
                    if blk_s < blk_e and gen_before[:, blk_s:blk_e].eq(mask_id).any():
                        logits = _forward_model(model, beam.seq, force_noncausal_ctx)
                        gen_logits = logits[:, prompt_len:, :]
                        noisy = _add_gumbel_noise(gen_logits, gumbel_temperature)
                        x0_gen = noisy.argmax(dim=-1)
                        conf = F.softmax(noisy.float(), dim=-1).max(dim=-1).values
                        gen_blk = gen_before[:, blk_s:blk_e]
                        updated_blk, _ = _transfer_tokens(
                            gen_blk, x0_gen[:, blk_s:blk_e],
                            conf[:, blk_s:blk_e], mask_id, tokens_per_step,
                        )
                        updated = gen_before.clone()
                        updated[:, blk_s:blk_e] = updated_blk
                        beam.seq[:, prompt_len:] = updated
                    beam.step = step
                steps_run = max(steps_run, step + 1)

            steps_run = max(steps_run, block_start + 1)

            # ---- Commit scored block, advance block_idx ----
            for beam in beams:
                if beam.committed_mask is not None and beam.current_block_mask is not None:
                    beam.committed_mask |= beam.current_block_mask
                beam.current_block_mask = _zero_mask_like(beam.seq)
                beam.block_idx += 1

            ckpt_elapsed = (time.time() - ckpt_t0) * 1000.0
            if store_trace:
                diag.checkpoints.append(OTSCheckpointDiag(
                    checkpoint_idx=checkpoint_idx,
                    step_range=(block_start, block_end),
                    beams_before_expand=n_before,
                    candidates_after_expand=len(all_candidates),
                    beams_after_prune=len(beams),
                    scores=[b.cumulative_score for b in beams],
                    masked_remaining=[
                        int(b.seq[:, prompt_len:].eq(mask_id).sum().item())
                        for b in beams
                    ],
                    elapsed_ms=ckpt_elapsed,
                ))
            checkpoint_idx += 1

            # Check early stop after each scored block.
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

    # ---- Select best beam (Alg.1 line 30) ----
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
    text = _decode_text_ots(tokenizer, out_ids, mask_id, is_dummy)

    diag.total_beams_explored = total_expansions + 1
    diag.chosen_beam_index = best_idx
    diag.chosen_beam_score = best_score
    diag.total_denoising_steps = steps_run
    diag.total_search_checkpoints = checkpoint_idx
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason

    stats = {
        "engine": "ots",
        "mode": "OTS",
        "tau_mask": tau_mask,
        "tau_edit": tau_edit,
        "beam_size": beam_size,
        "gumbel_temperature": gumbel_temperature,
        "search_interval": search_interval,
        "steps": max_steps,
        "block_size": block_size,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_second(tokens_generated, elapsed),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "dummy_model": is_dummy,
        "pruning_mode": diag.pruning_mode_used,
        "total_search_checkpoints": checkpoint_idx,
        "chosen_beam_index": best_idx,
        "chosen_beam_score": best_score,
        "ots_diagnostics": diag.to_dict(),
    }

    return text.strip(), stats, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_text_ots(tok: Any, out_ids: torch.Tensor, mask_id: int, is_dummy: bool) -> str:
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

# EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models.
# Paper: "EntRGi" (Tejaswi, Rout, Caramanis, Shakkottai, Sanghavi, 2026)
# arXiv:2602.05000 — Algorithm 1, Section 3.1, Section 3.2.
#
# KEY IDEA: Modify dLLM logits at masked positions using reward gradients
# from a frozen reward model. The reward model input at masked positions
# is an entropy-aware interpolation between:
#   - soft token embedding ē = Σ_j q_j E^R_j  (differentiable)
#   - sampled hard token embedding ẽ = E^R[x]  (via STE / stop-gradient)
#
# The interpolation weight w = H(q) / log(K) balances:
#   - Low entropy (confident) → trust soft embedding (continuous relaxation)
#   - High entropy (uncertain) → trust hard embedding (STE, reward-model reliable)
#
# Both the dLLM and the reward model stay FROZEN. This is test-time steering.
#
# REWARD MODEL: Skywork-Reward-V2-Qwen3-0.6B (same Qwen3 tokenizer family)
# The reward model is loaded separately and cached. If unavailable, the
# engine falls back to standard confidence-based denoising.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import math as _math
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level reward model cache
# ---------------------------------------------------------------------------
_REWARD_CACHE: Dict[str, Any] = {}


def load_reward_model(
    model_name: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any]:
    """Load and cache a frozen reward model + tokenizer.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    cache_key = f"{model_name}@{device}"
    if cache_key in _REWARD_CACHE:
        return _REWARD_CACHE[cache_key]

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _log.info("Loading reward model '%s' on %s ...", model_name, device)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        mdl = mdl.to(device)
        mdl.eval()
        for p in mdl.parameters():
            p.requires_grad_(False)
        _log.info("Reward model loaded: %s (%d params)", model_name, sum(p.numel() for p in mdl.parameters()))
        _REWARD_CACHE[cache_key] = (mdl, tok)
        return mdl, tok
    except Exception as exc:
        _log.warning("Could not load reward model '%s': %s", model_name, exc)
        _REWARD_CACHE[cache_key] = (None, None)
        return None, None


def _get_reward_embed_weight(reward_model: Any) -> torch.Tensor:
    """Extract the input embedding weight matrix from a reward model."""
    if hasattr(reward_model, "get_input_embeddings"):
        emb = reward_model.get_input_embeddings()
        if emb is not None:
            return emb.weight.detach()
    if hasattr(reward_model, "model"):
        inner = reward_model.model
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens.weight.detach()
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner.model.embed_tokens.weight.detach()
    raise RuntimeError("Cannot extract embedding weight from reward model")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class InferenceEntRGiRequest(BaseModel):
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
    # ---- EntRGi-specific knobs (Algorithm 1 hyper-parameters) ----
    entrgi_reward_model: str = Field(
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        description="HF model ID or local path for the reward model",
    )
    entrgi_guidance_scale: float = Field(
        default=0.5, ge=0.0, le=10.0,
        description="η — gradient step size (paper default 0.5)",
    )
    entrgi_guidance_steps: int = Field(
        default=3, ge=1, le=10,
        description="M — reward gradient steps per denoising step (paper default 3)",
    )
    entrgi_temperature: float = Field(
        default=0.7, ge=0.01, le=5.0,
        description="τ — softmax temperature for guidance logits (paper default 0.7)",
    )
    entrgi_confidence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Confidence threshold for unmasking (standard dLLM τ_mask)",
    )
    entrgi_disable_guidance: bool = Field(
        default=False,
        description="If True, skip reward guidance (ablation / fallback mode)",
    )
    entrgi_store_diagnostics: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class EntRGiStepDiag:
    step: int
    masked_before: int
    guidance_applied: bool
    guidance_steps_run: int
    avg_entropy: float
    avg_entropy_weight: float
    reward_before: float
    reward_after: float
    committed_count: int
    masked_after: int
    elapsed_ms: float


@dataclass
class EntRGiDiagnostics:
    steps: List[EntRGiStepDiag] = field(default_factory=list)
    total_denoising_steps: int = 0
    total_elapsed_ms: float = 0.0
    finish_reason: str = ""
    reward_model_name: str = ""
    reward_model_loaded: bool = False
    guidance_scale: float = 0.5
    guidance_steps: int = 3
    avg_masked_entropy: float = 0.0
    avg_entropy_weight: float = 0.0
    number_of_guidance_calls: int = 0
    number_of_guided_positions: int = 0
    fallback_to_standard_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step, "masked_before": s.masked_before,
                    "guidance_applied": s.guidance_applied,
                    "guidance_steps_run": s.guidance_steps_run,
                    "avg_entropy": s.avg_entropy, "avg_entropy_weight": s.avg_entropy_weight,
                    "reward_before": s.reward_before, "reward_after": s.reward_after,
                    "committed_count": s.committed_count, "masked_after": s.masked_after,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
            "total_denoising_steps": self.total_denoising_steps,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
            "reward_model_name": self.reward_model_name,
            "reward_model_loaded": self.reward_model_loaded,
            "guidance_scale": self.guidance_scale,
            "guidance_steps": self.guidance_steps,
            "avg_masked_entropy": self.avg_masked_entropy,
            "avg_entropy_weight": self.avg_entropy_weight,
            "number_of_guidance_calls": self.number_of_guidance_calls,
            "number_of_guided_positions": self.number_of_guided_positions,
            "fallback_to_standard_count": self.fallback_to_standard_count,
        }


# ---------------------------------------------------------------------------
# Core EntRGi helpers — paper-faithful math
# ---------------------------------------------------------------------------

def compute_entropy_weights(q: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Algorithm 1 line 9: w^l = Entropy(q^l) / log K.
    q: (N, V) valid probability distribution.
    Returns: (N,) weights in [0, 1].
    """
    log_q = q.clamp(min=1e-12).log()
    entropy = -(q * log_q).sum(dim=-1)  # (N,)
    w = (entropy / _math.log(max(vocab_size, 2))).clamp(0.0, 1.0)
    return w


def apply_entrgi_guidance(
    *,
    logits: torch.Tensor,
    seq: torch.Tensor,
    mask_id: int,
    prompt_len: int,
    reward_model: Any,
    reward_embed_weight: torch.Tensor,
    guidance_scale: float,
    guidance_steps: int,
    temperature: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply M steps of EntRGi gradient guidance to logits at masked positions.

    Implements Algorithm 1, lines 5-12.
    Both the dLLM and the reward model are frozen.
    Gradients flow: R(ê) → ê → ē (soft embed) → q (softmax) → ψ (logits).

    Returns: (guided_logits, info_dict)
    """
    S = seq.shape[1]
    V_reward, D = reward_embed_weight.shape
    V_logits = logits.shape[-1]
    V_common = min(V_logits, V_reward)

    # Find masked positions
    gen_is_mask = seq[0, prompt_len:].eq(mask_id)
    masked_gen_idx = gen_is_mask.nonzero(as_tuple=False).view(-1)
    num_masked = masked_gen_idx.shape[0]

    info: Dict[str, float] = {
        "avg_entropy": 0.0, "avg_w": 0.0,
        "reward_before": 0.0, "reward_after": 0.0,
        "num_guided": float(num_masked),
    }

    if num_masked == 0:
        return logits, info

    masked_seq_idx = masked_gen_idx + prompt_len  # in full-seq coords

    # Extract logits at masked positions (only over shared vocabulary)
    psi = logits[0, masked_seq_idx, :V_common].detach().clone().requires_grad_(True)

    # Prepare base embeddings for non-masked positions (detached)
    seq_for_embed = seq[0].clone()
    seq_for_embed[seq_for_embed >= V_reward] = 0  # clip out-of-vocab (mask token)
    base_embeds = F.embedding(seq_for_embed, reward_embed_weight).detach()  # (S, D)

    # Attention mask: all ones (all positions attend)
    attn_mask = torch.ones(1, S, dtype=torch.long, device=device)

    reward_before = 0.0
    reward_after = 0.0
    all_entropies: List[float] = []
    all_weights: List[float] = []

    for j in range(guidance_steps):
        if psi.grad is not None:
            psi.grad.zero_()

        # Alg 1, line 6: q = softmax(ψ/τ)
        q = F.softmax(psi / temperature, dim=-1)  # (num_masked, V_common)

        # Alg 1, line 8: soft embedding ē = Σ q_i E^R_i  (differentiable)
        e_bar = torch.matmul(q, reward_embed_weight[:V_common].to(q.dtype))  # (num_masked, D)

        # Alg 1, line 7: sample hard token, get embedding ẽ
        with torch.no_grad():
            sampled_ids = torch.multinomial(q.detach(), 1).squeeze(-1)
            e_tilde = reward_embed_weight[sampled_ids].to(q.dtype)  # (num_masked, D)

        # Alg 1, line 9: entropy weight w = H(q) / log K
        with torch.no_grad():
            w = compute_entropy_weights(q.detach(), V_common)  # (num_masked,)
            all_entropies.append(float(-(q.detach().clamp(1e-12) * q.detach().clamp(1e-12).log()).sum(-1).mean().item()))
            all_weights.append(float(w.mean().item()))

        # Alg 1, line 10: ê = ē + sg(w(ẽ - ē))
        # stop-gradient on the shift so gradients flow only through ē
        with torch.no_grad():
            shift = w.unsqueeze(-1) * (e_tilde - e_bar.detach())
        mixed = e_bar + shift.detach()  # (num_masked, D) — grad through e_bar

        # Build full reward-model input (differentiable at masked positions)
        # Use matmul with one-hot scatter for differentiable placement
        if num_masked < S:
            idx_mat = torch.zeros(num_masked, S, device=device, dtype=mixed.dtype)
            idx_mat[torch.arange(num_masked, device=device), masked_seq_idx] = 1.0
            scattered = idx_mat.T @ mixed  # (S, D) — differentiable

            mask_indicator = torch.zeros(S, 1, device=device, dtype=mixed.dtype)
            mask_indicator[masked_seq_idx] = 1.0

            full_embeds = ((1.0 - mask_indicator) * base_embeds.to(mixed.dtype)
                           + mask_indicator * scattered).unsqueeze(0)
        else:
            # All positions masked (rare edge case)
            full_embeds = mixed.unsqueeze(0)

        # Forward through frozen reward model
        try:
            r_out = reward_model(inputs_embeds=full_embeds, attention_mask=attn_mask)
            reward = r_out.logits.squeeze()
        except Exception:
            # Reward model failed — skip this guidance step
            break

        if j == 0:
            reward_before = float(reward.item())

        # Alg 1, line 11: ψ ← ψ + η ∇ψ R(ê)
        reward.backward()

        if psi.grad is not None:
            with torch.no_grad():
                psi.data.add_(guidance_scale * psi.grad)

        reward_after = float(reward.item())

    # Aggregate diagnostics
    info["avg_entropy"] = sum(all_entropies) / max(len(all_entropies), 1)
    info["avg_w"] = sum(all_weights) / max(len(all_weights), 1)
    info["reward_before"] = reward_before
    info["reward_after"] = reward_after

    # Write guided logits back
    result = logits.clone()
    with torch.no_grad():
        result[0, masked_seq_idx, :V_common] = psi.detach()
    return result, info


# ---------------------------------------------------------------------------
# Main EntRGi decode loop
# ---------------------------------------------------------------------------

def entrgi_decode(
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
    reward_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    guidance_scale: float = 0.5,
    guidance_steps: int = 3,
    entrgi_temperature: float = 0.7,
    confidence_threshold: float = 0.3,
    disable_guidance: bool = False,
    store_diagnostics: bool = True,
    seed: int = 42,
    is_dummy: bool = False,
    force_noncausal_ctx: Any = None,
) -> Tuple[str, Dict[str, Any], EntRGiDiagnostics]:
    """Full EntRGi inference loop (Algorithm 1).
    The dLLM and reward model both remain frozen.
    Returns: (text, stats_dict, diagnostics).
    """
    from .utils import seed_everything, tokens_per_second

    seed_everything(seed)
    diag = EntRGiDiagnostics(
        reward_model_name=reward_model_name,
        guidance_scale=guidance_scale,
        guidance_steps=guidance_steps,
    )

    # Load reward model (cached, lazy)
    reward_model = None
    reward_embed_weight = None
    if not is_dummy and not disable_guidance:
        rm, _rm_tok = load_reward_model(reward_model_name, device)
        if rm is not None:
            reward_model = rm
            try:
                reward_embed_weight = _get_reward_embed_weight(rm)
                diag.reward_model_loaded = True
            except Exception as exc:
                _log.warning("Could not extract reward embeddings: %s", exc)
                reward_model = None

    # Encode prompt
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])

    # Build sequence: [prompt] [MASK × max_new_tokens]
    seq = torch.full((1, prompt_len + max_new_tokens), mask_id, dtype=torch.long, device=device)
    seq[:, :prompt_len] = input_ids

    model.eval()
    t0 = time.time()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            eos_token_id = int(eos_token_id)
        except Exception:
            eos_token_id = None
    finish_reason = "length"
    converged = False
    eos_cut_idx: Optional[int] = None

    all_entropies: List[float] = []
    all_weights: List[float] = []

    for step in range(max_steps):
        step_t0 = time.time()

        gen_region = seq[0, prompt_len:]
        gen_is_mask = gen_region.eq(mask_id)
        masked_before = int(gen_is_mask.sum().item())
        if masked_before == 0:
            finish_reason = "converged"
            converged = True
            break

        # ── STEP 1: dLLM forward (frozen, no grad) ──
        with torch.no_grad():
            if force_noncausal_ctx is not None:
                with force_noncausal_ctx(model):
                    draft_out = model(input_ids=seq)
            else:
                draft_out = model(input_ids=seq)
        logits = draft_out.logits.detach().clone()  # (1, S, V)

        # ── STEP 2: EntRGi reward guidance (Algorithm 1 lines 5-12) ──
        guidance_applied = False
        step_info: Dict[str, float] = {
            "avg_entropy": 0.0, "avg_w": 0.0,
            "reward_before": 0.0, "reward_after": 0.0,
        }

        if reward_model is not None and masked_before > 0:
            try:
                logits, step_info = apply_entrgi_guidance(
                    logits=logits,
                    seq=seq,
                    mask_id=mask_id,
                    prompt_len=prompt_len,
                    reward_model=reward_model,
                    reward_embed_weight=reward_embed_weight,
                    guidance_scale=guidance_scale,
                    guidance_steps=guidance_steps,
                    temperature=entrgi_temperature,
                    device=device,
                )
                guidance_applied = True
                diag.number_of_guidance_calls += 1
                diag.number_of_guided_positions += masked_before
                all_entropies.append(step_info["avg_entropy"])
                all_weights.append(step_info["avg_w"])
            except Exception as exc:
                _log.warning("EntRGi guidance failed at step %d: %s", step + 1, exc)
                diag.fallback_to_standard_count += 1
        else:
            diag.fallback_to_standard_count += 1

        # ── STEP 3: Standard unmasking with (guided) logits (Alg 1 lines 13-15) ──
        with torch.no_grad():
            gen_logits = logits[:, prompt_len:, :]
            gen_probs = F.softmax(gen_logits.float(), dim=-1)
            confidence, pred_ids = gen_probs.max(dim=-1)  # (1, gen_len)

            # Select positions to unmask: masked & confident enough
            selected = gen_is_mask & confidence[0].ge(confidence_threshold)
            # Always unmask at least one position per step
            if gen_is_mask.any() and not selected.any():
                best = confidence[0].clone()
                best[~gen_is_mask] = -float("inf")
                best_pos = int(best.argmax().item())
                selected[best_pos] = True

            # Commit tokens
            if selected.any():
                gen_region[selected] = pred_ids[0][selected]
                seq[0, prompt_len:] = gen_region

        masked_after = int(seq[0, prompt_len:].eq(mask_id).sum().item())
        committed = masked_before - masked_after
        step_elapsed = (time.time() - step_t0) * 1000.0

        if store_diagnostics:
            diag.steps.append(EntRGiStepDiag(
                step=step + 1,
                masked_before=masked_before,
                guidance_applied=guidance_applied,
                guidance_steps_run=guidance_steps if guidance_applied else 0,
                avg_entropy=step_info.get("avg_entropy", 0.0),
                avg_entropy_weight=step_info.get("avg_w", 0.0),
                reward_before=step_info.get("reward_before", 0.0),
                reward_after=step_info.get("reward_after", 0.0),
                committed_count=committed,
                masked_after=masked_after,
                elapsed_ms=step_elapsed,
            ))

        # Stop conditions
        if eos_token_id is not None:
            eos_pos = (seq[0, prompt_len:] == eos_token_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                eos_cut_idx = int(eos_pos[0][0].item())
                finish_reason = "eos"
                converged = True
                break

        if masked_after == 0:
            finish_reason = "converged"
            converged = True
            break

    elapsed = max(1e-6, time.time() - t0)
    out_ids = seq[0, prompt_len:]
    if eos_cut_idx is not None:
        out_ids = out_ids[:max(0, eos_cut_idx + 1)]

    tokens_generated = int((out_ids != mask_id).sum().item())
    text = _decode_text(tokenizer, out_ids, mask_id, is_dummy)

    if not converged:
        finish_reason = "length"

    diag.total_denoising_steps = step + 1 if max_steps > 0 else 0
    diag.total_elapsed_ms = elapsed * 1000.0
    diag.finish_reason = finish_reason
    if all_entropies:
        diag.avg_masked_entropy = sum(all_entropies) / len(all_entropies)
    if all_weights:
        diag.avg_entropy_weight = sum(all_weights) / len(all_weights)

    stats = {
        "engine": "entrgi",
        "mode": "EntRGi",
        "tau_mask": tau_mask,
        "tau_edit": tau_edit,
        "steps": diag.total_denoising_steps,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_per_second(tokens_generated, elapsed),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "dummy_model": is_dummy,
        "guidance_scale": guidance_scale,
        "guidance_steps": guidance_steps,
        "reward_model_name": reward_model_name,
        "reward_model_loaded": diag.reward_model_loaded,
        "number_of_guidance_calls": diag.number_of_guidance_calls,
        "avg_masked_entropy": diag.avg_masked_entropy,
        "avg_entropy_weight": diag.avg_entropy_weight,
        "fallback_to_standard_count": diag.fallback_to_standard_count,
        "entrgi_diagnostics": diag.to_dict(),
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

# EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models.
# Paper: "EntRGi" (Tejaswi, Rout, Caramanis, Shakkottai, Sanghavi, 2026)
# arXiv:2602.05000 — Algorithm 1, Section 3.1, Section 3.2.
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math as _math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

_REWARD_CACHE: Dict[str, Any] = {}
_REWARD_ALIGNMENT_CACHE: Dict[str, Tuple[Optional[torch.Tensor], str]] = {}


def load_reward_model(
    model_name: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any]:
    """Load and cache a frozen reward model + tokenizer."""
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
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
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
    entrgi_reward_model: str = Field(
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        description="HF model ID or local path for the reward model",
    )
    entrgi_guidance_scale: float = Field(default=0.5, ge=0.0, le=10.0, description="eta")
    entrgi_guidance_steps: int = Field(default=3, ge=1, le=10, description="M")
    entrgi_temperature: float = Field(default=0.7, ge=0.01, le=5.0, description="tau")
    entrgi_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Deprecated compatibility knob; paper-faithful decoding uses lowest-entropy selection U(q)",
    )
    entrgi_disable_guidance: bool = Field(default=False)
    entrgi_store_diagnostics: bool = Field(default=True)


@dataclass
class EntRGiStepDiag:
    step: int
    masked_before: int
    guidance_applied: bool
    guidance_steps_run: int
    selection_budget: int
    selected_count: int
    avg_entropy: float
    avg_entropy_weight: float
    avg_selected_entropy: float
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
    reward_tokenizer_aligned: bool = False
    tokenizer_alignment_mode: str = "unverified"
    guidance_scale: float = 0.5
    guidance_steps: int = 3
    selection_policy_used: str = "lowest_entropy_budget"
    avg_masked_entropy: float = 0.0
    avg_entropy_weight: float = 0.0
    avg_selected_entropy: float = 0.0
    number_of_guidance_calls: int = 0
    number_of_guided_positions: int = 0
    fallback_to_standard_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "masked_before": s.masked_before,
                    "guidance_applied": s.guidance_applied,
                    "guidance_steps_run": s.guidance_steps_run,
                    "selection_budget": s.selection_budget,
                    "selected_count": s.selected_count,
                    "avg_entropy": s.avg_entropy,
                    "avg_entropy_weight": s.avg_entropy_weight,
                    "avg_selected_entropy": s.avg_selected_entropy,
                    "reward_before": s.reward_before,
                    "reward_after": s.reward_after,
                    "committed_count": s.committed_count,
                    "masked_after": s.masked_after,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
            "total_denoising_steps": self.total_denoising_steps,
            "total_elapsed_ms": self.total_elapsed_ms,
            "finish_reason": self.finish_reason,
            "reward_model_name": self.reward_model_name,
            "reward_model_loaded": self.reward_model_loaded,
            "reward_tokenizer_aligned": self.reward_tokenizer_aligned,
            "tokenizer_alignment_mode": self.tokenizer_alignment_mode,
            "guidance_scale": self.guidance_scale,
            "guidance_steps": self.guidance_steps,
            "selection_policy_used": self.selection_policy_used,
            "avg_masked_entropy": self.avg_masked_entropy,
            "avg_entropy_weight": self.avg_entropy_weight,
            "avg_selected_entropy": self.avg_selected_entropy,
            "number_of_guidance_calls": self.number_of_guidance_calls,
            "number_of_guided_positions": self.number_of_guided_positions,
            "fallback_to_standard_count": self.fallback_to_standard_count,
        }


def _safe_tokenizer_vocab_size(tokenizer: Any) -> int:
    vals = [
        int(getattr(tokenizer, "vocab_size", 0) or 0),
        int(len(tokenizer)) if hasattr(tokenizer, "__len__") else 0,
    ]
    return max(vals) if vals else 0


def _tokenizer_name(tokenizer: Any) -> str:
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            return value
    return tokenizer.__class__.__name__


def _id_to_token(tokenizer: Any, token_id: int) -> Optional[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            tok = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(tok, list):
                tok = tok[0] if tok else None
            if isinstance(tok, str):
                return tok
        except Exception:
            pass
    if hasattr(tokenizer, "decode"):
        try:
            tok = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            return tok if isinstance(tok, str) and tok != "" else None
        except Exception:
            pass
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
            if isinstance(vocab, dict):
                for tok, idx in vocab.items():
                    if int(idx) == int(token_id):
                        return str(tok)
        except Exception:
            pass
    return None


def _token_to_id(tokenizer: Any, token: str) -> Optional[int]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            idx = tokenizer.convert_tokens_to_ids(token)
            if isinstance(idx, int):
                return int(idx)
        except Exception:
            pass
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
            if isinstance(vocab, dict) and token in vocab:
                return int(vocab[token])
        except Exception:
            pass
    return None


def build_reward_id_map(
    tokenizer: Any,
    reward_tokenizer: Any,
    model_vocab_size: int,
    reward_vocab_size: int,
) -> Tuple[Optional[torch.Tensor], str]:
    """Build a base-token-id -> reward-token-id map over the model vocabulary."""
    if model_vocab_size <= 0 or reward_vocab_size <= 0:
        return None, "invalid_vocab_size"
    if tokenizer is reward_tokenizer and reward_vocab_size >= model_vocab_size:
        return torch.arange(model_vocab_size, dtype=torch.long), "identity"

    try:
        base_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
        reward_vocab = reward_tokenizer.get_vocab() if hasattr(reward_tokenizer, "get_vocab") else None
        if isinstance(base_vocab, dict) and isinstance(reward_vocab, dict) and base_vocab == reward_vocab and reward_vocab_size >= model_vocab_size:
            return torch.arange(model_vocab_size, dtype=torch.long), "shared_vocab_dict"
    except Exception:
        pass

    base_tok_vocab = _safe_tokenizer_vocab_size(tokenizer)
    reward_tok_vocab = _safe_tokenizer_vocab_size(reward_tokenizer)
    if base_tok_vocab < model_vocab_size:
        return None, "base_vocab_too_small"
    if reward_tok_vocab <= 0:
        return None, "reward_vocab_unavailable"

    cache_key = f"{id(tokenizer)}:{id(reward_tokenizer)}:{model_vocab_size}:{reward_vocab_size}"
    cached = _REWARD_ALIGNMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mapping = torch.full((model_vocab_size,), -1, dtype=torch.long)
    for token_id in range(model_vocab_size):
        tok = _id_to_token(tokenizer, token_id)
        if tok is None:
            result = (None, "base_token_decode_failed")
            _REWARD_ALIGNMENT_CACHE[cache_key] = result
            return result
        reward_id = _token_to_id(reward_tokenizer, tok)
        if reward_id is None or reward_id < 0 or reward_id >= reward_vocab_size:
            result = (None, "reward_token_missing")
            _REWARD_ALIGNMENT_CACHE[cache_key] = result
            return result
        reward_tok = _id_to_token(reward_tokenizer, reward_id)
        if reward_tok != tok:
            result = (None, "token_roundtrip_mismatch")
            _REWARD_ALIGNMENT_CACHE[cache_key] = result
            return result
        mapping[token_id] = int(reward_id)

    result = (mapping, "token_string_map")
    _REWARD_ALIGNMENT_CACHE[cache_key] = result
    return result


def _prepare_reward_guidance_view(
    tokenizer: Any,
    reward_tokenizer: Any,
    reward_embed_weight: torch.Tensor,
    model_vocab_size: int,
    mask_id: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    reward_vocab_size = int(reward_embed_weight.shape[0])
    reward_id_map, mode = build_reward_id_map(tokenizer, reward_tokenizer, model_vocab_size, reward_vocab_size)
    if reward_id_map is None:
        return None, mode

    actual_token_ids_cpu = torch.arange(model_vocab_size, dtype=torch.long)
    actual_token_ids_cpu = actual_token_ids_cpu[actual_token_ids_cpu.ne(int(mask_id))]
    if actual_token_ids_cpu.numel() == 0:
        return None, "no_actual_tokens"

    reward_actual_ids = reward_id_map.index_select(0, actual_token_ids_cpu)
    if bool((reward_actual_ids < 0).any()):
        return None, "actual_token_mapping_incomplete"

    aligned_reward_embeds = reward_embed_weight.index_select(0, reward_actual_ids.to(reward_embed_weight.device))
    return {
        "reward_id_map_cpu": reward_id_map,
        "actual_token_ids_cpu": actual_token_ids_cpu,
        "reward_actual_ids_cpu": reward_actual_ids,
        "aligned_reward_embeds": aligned_reward_embeds,
    }, mode


def compute_entropy_weights(q: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Algorithm 1 line 9: w = H(q) / log K."""
    log_q = q.clamp(min=1e-12).log()
    entropy = -(q * log_q).sum(dim=-1)
    return (entropy / _math.log(max(vocab_size, 2))).clamp(0.0, 1.0)


def select_lowest_entropy_mask_positions(
    masked_q: torch.Tensor,
    remaining_steps: int,
) -> Tuple[torch.Tensor, int, float, float]:
    """Dream-like U(q): choose a budgeted subset of currently masked positions with lowest entropy."""
    num_masked = int(masked_q.shape[0])
    if num_masked == 0:
        return torch.zeros((0,), dtype=torch.long, device=masked_q.device), 0, 0.0, 0.0

    probs = masked_q.clamp(min=1e-12)
    entropies = -(probs * probs.log()).sum(dim=-1)
    budget = min(num_masked, max(1, int(_math.ceil(num_masked / max(1, remaining_steps)))))
    chosen = torch.topk(entropies, k=budget, largest=False).indices
    avg_entropy = float(entropies.mean().item())
    avg_selected_entropy = float(entropies[chosen].mean().item()) if chosen.numel() > 0 else 0.0
    return chosen, budget, avg_entropy, avg_selected_entropy


def apply_entrgi_guidance(
    *,
    logits: torch.Tensor,
    seq: torch.Tensor,
    mask_id: int,
    prompt_len: int,
    reward_model: Any,
    reward_view: Dict[str, Any],
    guidance_scale: float,
    guidance_steps: int,
    temperature: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply M steps of EntRGi gradient guidance to masked-position logits."""
    S = int(seq.shape[1])
    V_logits = int(logits.shape[-1])
    actual_token_ids_cpu = reward_view["actual_token_ids_cpu"]
    reward_actual_ids_cpu = reward_view["reward_actual_ids_cpu"]
    valid_vocab = actual_token_ids_cpu.lt(V_logits)
    actual_token_ids = actual_token_ids_cpu[valid_vocab].to(device=device)
    reward_actual_ids = reward_actual_ids_cpu[valid_vocab].to(device=device)
    aligned_reward_embeds = _get_reward_embed_weight(reward_model).to(device=device).index_select(0, reward_actual_ids).to(dtype=logits.dtype)

    gen_is_mask = seq[0, prompt_len:].eq(mask_id)
    masked_gen_idx = gen_is_mask.nonzero(as_tuple=False).view(-1)
    num_masked = int(masked_gen_idx.shape[0])
    info: Dict[str, float] = {
        "avg_entropy": 0.0,
        "avg_w": 0.0,
        "reward_before": 0.0,
        "reward_after": 0.0,
        "num_guided": float(num_masked),
    }
    if num_masked == 0 or actual_token_ids.numel() == 0:
        return logits, info

    masked_seq_idx = masked_gen_idx + prompt_len
    psi = logits[0, masked_seq_idx, :].index_select(-1, actual_token_ids).detach().clone().requires_grad_(True)

    reward_id_map_cpu = reward_view["reward_id_map_cpu"]
    reward_seq_ids = reward_id_map_cpu.index_select(0, seq[0].detach().cpu().long()).to(device=device)
    if bool((reward_seq_ids < 0).any()):
        raise RuntimeError("Reward tokenizer alignment missing for committed sequence tokens")
    base_embeds = F.embedding(reward_seq_ids, _get_reward_embed_weight(reward_model).to(device=device)).detach()
    attn_mask = torch.ones(1, S, dtype=torch.long, device=device)

    reward_before = 0.0
    reward_after = 0.0
    all_entropies: List[float] = []
    all_weights: List[float] = []

    for j in range(guidance_steps):
        if psi.grad is not None:
            psi.grad.zero_()

        q = F.softmax(psi / temperature, dim=-1)
        e_bar = torch.matmul(q, aligned_reward_embeds.to(q.dtype))

        with torch.no_grad():
            sampled_local = torch.multinomial(q.detach(), 1).squeeze(-1)
            e_tilde = aligned_reward_embeds.index_select(0, sampled_local).to(q.dtype)
            w = compute_entropy_weights(q.detach(), int(actual_token_ids.numel()))
            entropy = -(q.detach().clamp(min=1e-12) * q.detach().clamp(min=1e-12).log()).sum(dim=-1)
            all_entropies.append(float(entropy.mean().item()))
            all_weights.append(float(w.mean().item()))
            shift = w.unsqueeze(-1) * (e_tilde - e_bar.detach())

        mixed = e_bar + shift.detach()
        idx_mat = torch.zeros(num_masked, S, device=device, dtype=mixed.dtype)
        idx_mat[torch.arange(num_masked, device=device), masked_seq_idx] = 1.0
        scattered = idx_mat.T @ mixed
        mask_indicator = torch.zeros(S, 1, device=device, dtype=mixed.dtype)
        mask_indicator[masked_seq_idx] = 1.0
        full_embeds = ((1.0 - mask_indicator) * base_embeds.to(mixed.dtype) + mask_indicator * scattered).unsqueeze(0)

        reward = reward_model(inputs_embeds=full_embeds, attention_mask=attn_mask).logits.squeeze()
        if j == 0:
            reward_before = float(reward.item())
        reward.backward()

        if psi.grad is not None:
            with torch.no_grad():
                psi.add_(guidance_scale * psi.grad)

    # Compute reward_after once, only after the final gradient step (avoids M extra forwards).
    if guidance_steps > 0:
        with torch.no_grad():
            final_q = F.softmax(psi / temperature, dim=-1)
            final_e_bar = torch.matmul(final_q, aligned_reward_embeds.to(final_q.dtype))
            final_sampled = torch.multinomial(final_q.detach(), 1).squeeze(-1)
            final_e_tilde = aligned_reward_embeds.index_select(0, final_sampled).to(final_q.dtype)
            final_w = compute_entropy_weights(final_q.detach(), int(actual_token_ids.numel()))
            final_shift = final_w.unsqueeze(-1) * (final_e_tilde - final_e_bar)
            final_mixed = final_e_bar + final_shift
            final_scattered = idx_mat.T @ final_mixed
            final_full = ((1.0 - mask_indicator) * base_embeds.to(final_mixed.dtype) + mask_indicator * final_scattered).unsqueeze(0)
            reward_after = float(
                reward_model(inputs_embeds=final_full, attention_mask=attn_mask).logits.squeeze().item()
            )

    info["avg_entropy"] = sum(all_entropies) / max(len(all_entropies), 1)
    info["avg_w"] = sum(all_weights) / max(len(all_weights), 1)
    info["reward_before"] = reward_before
    info["reward_after"] = reward_after

    result = logits.clone()
    with torch.no_grad():
        result[0, masked_seq_idx, :].index_copy_(1, actual_token_ids, psi.detach())
        if 0 <= int(mask_id) < V_logits:
            result[0, masked_seq_idx, int(mask_id)] = torch.finfo(result.dtype).min
    return result, info


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
    """Full EntRGi inference loop implementing Algorithm 1."""
    from .utils import seed_everything, tokens_per_second

    del tau_mask, tau_edit, confidence_threshold

    seed_everything(seed)
    diag = EntRGiDiagnostics(
        reward_model_name=reward_model_name,
        guidance_scale=guidance_scale,
        guidance_steps=guidance_steps,
    )

    reward_model = None
    reward_view = None
    if not is_dummy and not disable_guidance:
        rm, reward_tokenizer = load_reward_model(reward_model_name, device)
        if rm is not None and reward_tokenizer is not None:
            try:
                reward_embed_weight = _get_reward_embed_weight(rm)
                reward_view, alignment_mode = _prepare_reward_guidance_view(
                    tokenizer=tokenizer,
                    reward_tokenizer=reward_tokenizer,
                    reward_embed_weight=reward_embed_weight,
                    model_vocab_size=int(vocab_size),
                    mask_id=int(mask_id),
                )
                diag.reward_model_loaded = True
                diag.tokenizer_alignment_mode = alignment_mode
                diag.reward_tokenizer_aligned = reward_view is not None
                if reward_view is not None:
                    reward_model = rm
                else:
                    _log.warning(
                        "Disabling EntRGi guidance because tokenizer alignment failed: base=%s reward=%s mode=%s",
                        _tokenizer_name(tokenizer),
                        _tokenizer_name(reward_tokenizer),
                        alignment_mode,
                    )
            except Exception as exc:
                _log.warning("Could not prepare reward guidance view: %s", exc)
                diag.tokenizer_alignment_mode = "alignment_exception"
        else:
            diag.tokenizer_alignment_mode = "reward_model_unavailable"
    else:
        diag.tokenizer_alignment_mode = "guidance_disabled"

    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
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
    all_selected_entropies: List[float] = []

    for step in range(max_steps):
        step_t0 = time.time()

        gen_region = seq[0, prompt_len:]
        gen_is_mask = gen_region.eq(mask_id)
        masked_before = int(gen_is_mask.sum().item())
        if masked_before == 0:
            finish_reason = "converged"
            converged = True
            break

        with torch.no_grad():
            if force_noncausal_ctx is not None:
                with force_noncausal_ctx(model):
                    draft_out = model(input_ids=seq)
            else:
                draft_out = model(input_ids=seq)
        logits = draft_out.logits.detach().clone()

        guidance_applied = False
        step_info: Dict[str, float] = {
            "avg_entropy": 0.0,
            "avg_w": 0.0,
            "reward_before": 0.0,
            "reward_after": 0.0,
        }
        if reward_model is not None and reward_view is not None:
            try:
                logits, step_info = apply_entrgi_guidance(
                    logits=logits,
                    seq=seq,
                    mask_id=mask_id,
                    prompt_len=prompt_len,
                    reward_model=reward_model,
                    reward_view=reward_view,
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

        with torch.no_grad():
            masked_gen_idx = gen_is_mask.nonzero(as_tuple=False).view(-1)
            masked_seq_idx = masked_gen_idx + prompt_len
            actual_token_ids = torch.arange(logits.shape[-1], device=device, dtype=torch.long)
            actual_token_ids = actual_token_ids[actual_token_ids.ne(int(mask_id))]
            masked_logits = logits[0, masked_seq_idx, :].index_select(-1, actual_token_ids)
            masked_q = F.softmax(masked_logits.float() / entrgi_temperature, dim=-1)
            remaining_steps = max(1, max_steps - step)
            chosen_local, selection_budget, avg_masked_entropy, avg_selected_entropy = select_lowest_entropy_mask_positions(
                masked_q,
                remaining_steps,
            )
            all_selected_entropies.append(avg_selected_entropy)

            sampled_local = torch.multinomial(masked_q.index_select(0, chosen_local), 1).squeeze(-1)
            sampled_token_ids = actual_token_ids.index_select(0, sampled_local)
            selected_positions = masked_gen_idx.index_select(0, chosen_local)
            gen_region[selected_positions] = sampled_token_ids
            seq[0, prompt_len:] = gen_region

        masked_after = int(seq[0, prompt_len:].eq(mask_id).sum().item())
        committed = masked_before - masked_after
        step_elapsed = (time.time() - step_t0) * 1000.0

        if store_diagnostics:
            diag.steps.append(
                EntRGiStepDiag(
                    step=step + 1,
                    masked_before=masked_before,
                    guidance_applied=guidance_applied,
                    guidance_steps_run=guidance_steps if guidance_applied else 0,
                    selection_budget=selection_budget,
                    selected_count=int(chosen_local.numel()),
                    avg_entropy=step_info.get("avg_entropy", avg_masked_entropy),
                    avg_entropy_weight=step_info.get("avg_w", 0.0),
                    avg_selected_entropy=avg_selected_entropy,
                    reward_before=step_info.get("reward_before", 0.0),
                    reward_after=step_info.get("reward_after", 0.0),
                    committed_count=committed,
                    masked_after=masked_after,
                    elapsed_ms=step_elapsed,
                )
            )

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
        out_ids = out_ids[: max(0, eos_cut_idx + 1)]

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
    if all_selected_entropies:
        diag.avg_selected_entropy = sum(all_selected_entropies) / len(all_selected_entropies)

    stats = {
        "engine": "entrgi",
        "mode": "EntRGi",
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
        "reward_tokenizer_aligned": diag.reward_tokenizer_aligned,
        "tokenizer_alignment_mode": diag.tokenizer_alignment_mode,
        "selection_policy_used": diag.selection_policy_used,
        "number_of_guidance_calls": diag.number_of_guidance_calls,
        "avg_masked_entropy": diag.avg_masked_entropy,
        "avg_entropy_weight": diag.avg_entropy_weight,
        "avg_selected_entropy": diag.avg_selected_entropy,
        "fallback_to_standard_count": diag.fallback_to_standard_count,
        "entrgi_diagnostics": diag.to_dict(),
    }

    return text.strip(), stats, diag


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

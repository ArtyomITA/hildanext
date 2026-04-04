# Inference engines: dInfer adapter + Transformers fallback.
# Main entrypoints: build_engine,load_model_bundle.
# Implements threshold-edit decode with degeneration guards and telemetry.
from __future__ import annotations
from dataclasses import dataclass,field
from collections import Counter
from pathlib import Path
from typing import Any,Dict,Optional,Tuple,List
import os
import sys
import time
import gc
import math as _math
import torch
from .config import AppConfig
from .tokenization import load_tokenizer,ensure_mask_token
from .diffusion import apply_remask,force_noncausal_attention
from .formulas import llada21_sets
from .toon import dumps_toon
from .utils import TinyCausalLM,choose_device,dtype_from_name,force_math_sdpa,seed_everything,tokens_per_second,env_issues
from .trace import use_trace,trace_from_cfg,exception_with_stack

@dataclass
class ModelBundle:
    model:Any
    tokenizer:Any
    device:torch.device
    mask_id:int
    vocab_size:int
    is_dummy:bool
    load_reason:str=""
    env_issues:Dict[str,str]=field(default_factory=dict)
    model_name_or_path:str=""
    requested_dtype:str=""
    actual_dtype:str=""
    fallbacks:List[Dict[str,Any]]=field(default_factory=list)

def _model_dir_ready(model_dir:str)->bool:
    p=Path(model_dir)
    if not p.exists():
        return False
    if not (p/"config.json").exists():
        return False
    return any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

def _infer_param_dtype(model:Any)->str:
    try:
        return str(next(model.parameters()).dtype)
    except Exception:
        return "unknown"


def _weight_shard_count(model_dir:str)->int:
    p=Path(model_dir)
    if not p.exists():
        return 0
    seen:set[str]=set()
    for pat in [
        "model-*.safetensors",
        "pytorch_model-*.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ]:
        for fp in p.glob(pat):
            if fp.is_file():
                seen.add(str(fp.resolve()))
    return len(seen)


def _configure_hf_parallel_loading(model_dir:str)->None:
    shard_count=_weight_shard_count(model_dir)
    if shard_count<2:
        return
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING","true")
    workers=max(1,min(8,shard_count))
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS",str(workers))


def _from_pretrained_best_effort(AutoModelForCausalLM:Any,model_dir:str,dtype:Any,trust_remote_code:bool)->Any:
    # Try safer/faster combinations first, then fall back for older stacks.
    base={"trust_remote_code":trust_remote_code}
    attempts=[
        {**base,"local_files_only":True,"low_cpu_mem_usage":True,"use_safetensors":True},
        {**base,"local_files_only":True,"low_cpu_mem_usage":True},
        {**base,"local_files_only":True},
        dict(base),
    ]
    last_err:Exception|None=None
    for kwargs in attempts:
        try:
            try:
                return AutoModelForCausalLM.from_pretrained(model_dir,dtype=dtype,**kwargs)
            except TypeError:
                return AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=dtype,**kwargs)
        except Exception as e:
            last_err=e
    if last_err is not None:
        raise last_err
    raise RuntimeError("from_pretrained_failed")

def load_model_bundle(cfg:AppConfig,for_training:bool=False,trace=None)->ModelBundle:
    tr=use_trace(cfg,trace)
    force_math_sdpa()
    if tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="force_math_sdpa",
            reason="flash_attention_unavailable",
            extra_dict={"device_hint":cfg.runtime.device}
        )
    issues=env_issues()
    if tr is not None:
        for n,d in issues.items():
            tr.record_env_issue(name=f"{n}_unavailable",detail=str(d),module="inference",func="load_model_bundle")
    device=choose_device(cfg.runtime.device)
    if cfg.runtime.device=="cuda" and device.type!="cuda" and tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="cpu_fallback",
            reason="cuda_unavailable",
            extra_dict={"device_hint":cfg.runtime.device}
        )
    tok=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code,trace=tr,cfg=cfg)
    model=None
    is_dummy=False
    reason=""
    requested=str(cfg.train.dtype)
    if not cfg.runtime.force_dummy_model:
        if _model_dir_ready(cfg.paths.model_dir):
            try:
                from transformers import AutoModelForCausalLM
                td=dtype_from_name(cfg.train.dtype,device)
                _configure_hf_parallel_loading(cfg.paths.model_dir)
                model=_from_pretrained_best_effort(
                    AutoModelForCausalLM,
                    cfg.paths.model_dir,
                    td,
                    cfg.model.trust_remote_code,
                )
                model=model.to(device)
            except Exception as e:
                reason=f"load_failed:{e}"
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="inference",
                        func="load_model_bundle",
                        action="dummy_model_fallback",
                        reason="model_load_failed",
                        exception_str=exception_with_stack(e),
                        extra_dict={"model_dir":cfg.paths.model_dir}
                    )
                model=None
        else:
            reason="model_dir_invalid_or_missing"
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="inference",
                    func="load_model_bundle",
                    action="dummy_model_fallback",
                    reason="model_dir_invalid_or_missing",
                    extra_dict={"model_dir":cfg.paths.model_dir}
                )
    else:
        reason="force_dummy_model=true"
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="inference",
                func="load_model_bundle",
                action="dummy_model_fallback",
                reason="force_dummy_model",
                extra_dict={"model_dir":cfg.paths.model_dir}
            )
    if model is None:
        is_dummy=True
        vocab=max(int(getattr(tok,"vocab_size",0) or 0),len(tok) if hasattr(tok,"__len__") else 0,32768)
        model=TinyCausalLM(vocab_size=vocab,hidden_size=256).to(device)
    if for_training and cfg.train.grad_ckpt and hasattr(model,"gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False,"preserve_rng_state":False})
        except TypeError:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        except Exception:
            pass
    if for_training and hasattr(model,"config") and hasattr(model.config,"use_cache"):
        try:
            model.config.use_cache=False
        except Exception:
            pass
    if (not for_training) and hasattr(model,"config") and hasattr(model.config,"use_cache"):
        try:
            # dLLM bidirectional forward does not use KV-cache
            model.config.use_cache=False
        except Exception:
            pass
    mask_id=ensure_mask_token(tok,cfg.model.mask_token,model=model)
    model_vocab=int(getattr(getattr(model,"lm_head",None),"out_features",0) or 0)
    tok_vocab=int(len(tok) if hasattr(tok,"__len__") else 0)
    if is_dummy and mask_id>=model_vocab:
        new_vocab=max(mask_id+1,tok_vocab,model_vocab)
        model=TinyCausalLM(vocab_size=new_vocab,hidden_size=256).to(device)
        model_vocab=int(getattr(model.lm_head,"out_features",new_vocab))
    vocab_size=max(tok_vocab,model_vocab)
    model.train(mode=for_training)
    actual_dtype=_infer_param_dtype(model)
    if requested.lower() in {"bfloat16","bf16"} and "float16" in actual_dtype and tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="dtype_fallback",
            reason="bf16_to_fp16",
            extra_dict={"requested_dtype":requested,"actual_dtype":actual_dtype}
        )
    return ModelBundle(
        model=model,
        tokenizer=tok,
        device=device,
        mask_id=mask_id,
        vocab_size=vocab_size,
        is_dummy=is_dummy,
        load_reason=reason,
        env_issues=issues,
        model_name_or_path=cfg.paths.model_dir,
        requested_dtype=requested,
        actual_dtype=actual_dtype,
        fallbacks=tr.snapshot_fallbacks(limit=64) if tr is not None else []
    )

# --- P0.2 effort knob ---
_EFFORT_PARAMS:Dict[str,Dict[str,Any]]={
    "instant": {"max_steps":1,   "tau_scale":2.0},
    "low":     {"max_steps":16,  "tau_scale":1.5},
    "medium":  {"max_steps":64,  "tau_scale":1.0},
    "high":    {"max_steps":128, "tau_scale":0.7},
    "adaptive":{"max_steps":256, "tau_scale":1.0},
}

def _resolve_effort(effort:str,cfg_steps:int,tau_mask:float,tau_edit:float)->Tuple[int,float,float]:
    """Map effort level → (steps, tau_mask, tau_edit).
    'adaptive' = run until mask_ratio==0, up to 128 steps.
    tau is clamped to [0, 1].
    """
    p=_EFFORT_PARAMS.get(str(effort).lower(),_EFFORT_PARAMS["medium"])
    steps=p["max_steps"] if p["max_steps"] is not None else cfg_steps
    scale=float(p["tau_scale"])
    return (
        max(1,int(steps)),
        min(1.0,max(0.0,tau_mask*scale)),
        min(1.0,max(0.0,tau_edit*scale)),
    )


def mode_thresholds(cfg:AppConfig,mode:str,tau_mask:Optional[float],tau_edit:Optional[float])->Tuple[float,float]:
    m=(mode or "S_MODE").upper()
    if tau_mask is not None and tau_edit is not None:
        return float(tau_mask),float(tau_edit)
    if m=="Q_MODE":
        return float(cfg.inference.q_mode_tau_mask),float(cfg.inference.q_mode_tau_edit)
    return float(cfg.inference.s_mode_tau_mask),float(cfg.inference.s_mode_tau_edit)

def _top1_with_confidence(logits:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
    """Return (top1_prob, top1_id) without materializing full softmax."""
    logits_fp32=logits.float()
    topv,p=torch.max(logits_fp32,dim=-1)
    lse=torch.logsumexp(logits_fp32,dim=-1)
    c=torch.exp(topv-lse).to(torch.float32)
    return c,p.to(torch.long)


def _add_gumbel_noise(logits:torch.Tensor,temperature:float)->torch.Tensor:
    """Gumbel-max sampling noise (LLaDA style). temperature=0 → deterministic."""
    if temperature==0:
        return logits
    logits=logits.to(torch.float64)
    noise=torch.rand_like(logits,dtype=torch.float64)
    noise=noise.clamp_(min=1e-20)
    gumbel_noise=(-torch.log(noise))**temperature
    return logits.exp()/gumbel_noise


def _forward_full_sequence(model:Any,seq:torch.Tensor,use_bidirectional:bool=False)->Any:
    if use_bidirectional:
        with force_noncausal_attention(model):
            try:
                return model(input_ids=seq,use_cache=False)
            except TypeError:
                return model.forward(input_ids=seq,use_cache=False)
    try:
        return model(input_ids=seq,use_cache=False)
    except TypeError:
        return model.forward(input_ids=seq,use_cache=False)


def _predict_bidirectional(model:Any,seq:torch.Tensor,prompt_len:int,max_new:int,
                           mask_id:int,temperature:float=0.0,use_bidirectional:bool=False)->Tuple[torch.Tensor,torch.Tensor]:
    """Full-sequence forward pass for dLLM inference.

    When use_bidirectional=True uses force_noncausal_attention to make
    Qwen3ForCausalLM behave as a bidirectional encoder.
    When use_bidirectional=False (default) uses standard causal attention,
    matching WSD decay-phase training semantics.
    Applies left-shift: logits[j] predicts token[j+1], so we read
    logits[prompt_len-1 : prompt_len-1+max_new] for gen positions.
    Confidence is computed from ORIGINAL logits (pre-noise) following
    LLaDA generate.py: Gumbel noise is only used for diverse token
    selection via argmax, not for confidence estimation.
    """
    out=_forward_full_sequence(model,seq,use_bidirectional=use_bidirectional)
    # Left-shift alignment: logits[j] → token[j+1]
    # For gen position i (seq index prompt_len+i), read logits[prompt_len-1+i]
    gen_logits=out.logits[:,prompt_len-1:prompt_len-1+max_new,:]
    # Confidence from ORIGINAL logits (LLaDA style: p = softmax(logits))
    probs=torch.nn.functional.softmax(gen_logits.float(),dim=-1)
    # Gumbel noise only for token selection (argmax), not confidence
    noisy=_add_gumbel_noise(gen_logits,temperature)
    pred=torch.argmax(noisy.float(),dim=-1)
    # Confidence = probability of the selected token under the clean distribution
    conf=torch.gather(probs,dim=-1,index=pred.unsqueeze(-1).long()).squeeze(-1).to(torch.float32)
    return pred.to(torch.long),conf


def _predict_autoregressive_candidates_full(model:Any,seq:torch.Tensor,prompt_len:int,max_new:int,mask_id:int)->Tuple[torch.Tensor,torch.Tensor]:
    work=seq.clone()
    pred=torch.zeros((1,max_new),dtype=torch.long,device=seq.device)
    conf=torch.zeros((1,max_new),dtype=torch.float32,device=seq.device)
    for i in range(max_new):
        g=prompt_len+i
        prefix=work[:,:g]
        if prefix.shape[1]==0:
            prefix=work[:,:1]
        out=model(input_ids=prefix)
        logits=out.logits[:,-1,:]
        c,p=_top1_with_confidence(logits)
        pred[:,i]=p
        conf[:,i]=c
        if int(work[0,g].item())==int(mask_id):
            work[:,g]=p
    return pred,conf


def _predict_autoregressive_candidates_cached(model:Any,seq:torch.Tensor,prompt_len:int,max_new:int,mask_id:int)->Tuple[torch.Tensor,torch.Tensor]:
    """Faster candidate scan using KV-cache incremental decode."""
    work=seq.clone()
    pred=torch.zeros((1,max_new),dtype=torch.long,device=seq.device)
    conf=torch.zeros((1,max_new),dtype=torch.float32,device=seq.device)
    prefix=work[:,:prompt_len]
    if prefix.shape[1]==0:
        prefix=work[:,:1]
    out=model(input_ids=prefix,use_cache=True)
    logits=out.logits[:,-1,:]
    past_kv=getattr(out,"past_key_values",None)
    for i in range(max_new):
        g=prompt_len+i
        c,p=_top1_with_confidence(logits)
        pred[:,i]=p
        conf[:,i]=c
        if int(work[0,g].item())==int(mask_id):
            next_tok=p.view(1,1)
            work[:,g]=p
        else:
            next_tok=work[:,g].view(1,1)
        if i+1>=max_new:
            break
        if past_kv is None:
            # Some models may ignore cache; fallback to growing-prefix step.
            out=model(input_ids=work[:,:g+1],use_cache=True)
        else:
            out=model(input_ids=next_tok,past_key_values=past_kv,use_cache=True)
        logits=out.logits[:,-1,:]
        past_kv=getattr(out,"past_key_values",None)
    return pred,conf


def _predict_autoregressive_candidates(model:Any,seq:torch.Tensor,prompt_len:int,max_new:int,mask_id:int)->Tuple[torch.Tensor,torch.Tensor]:
    try:
        return _predict_autoregressive_candidates_cached(model,seq,prompt_len,max_new,mask_id)
    except Exception:
        return _predict_autoregressive_candidates_full(model,seq,prompt_len,max_new,mask_id)


def _token_debug_text(tok:Any,token_id:int)->str:
    try:
        text=tok.decode([int(token_id)],skip_special_tokens=False)
    except Exception:
        try:
            text=tok.decode(int(token_id),skip_special_tokens=False)
        except Exception:
            text=""
    text=str(text or "").replace("\n","\\n").strip()
    return text if text else f"tok{int(token_id)}"


def _topk_token_debug(tok:Any,pred_ids:torch.Tensor,topn:int=5)->Tuple[Dict[str,Any]|None,float,List[Dict[str,Any]]]:
    flat=[int(x) for x in pred_ids.reshape(-1).detach().cpu().tolist()]
    if not flat:
        return None,0.0,[]
    counts=Counter(flat).most_common(max(1,int(topn)))
    total=max(1,len(flat))
    dom_id,dom_count=counts[0]
    dominant={"id":int(dom_id),"text":_token_debug_text(tok,dom_id)}
    topk=[
        {
            "id":int(tok_id),
            "text":_token_debug_text(tok,tok_id),
            "count":int(cnt),
            "share":float(cnt/total),
        }
        for tok_id,cnt in counts
    ]
    return dominant,float(dom_count/total),topk


def _select_mask_topk(
    masked:torch.Tensor,
    confidence:torch.Tensor,
    k:int,
    *,
    exclude:torch.Tensor|None=None,
)->torch.Tensor:
    selected=torch.zeros_like(masked,dtype=torch.bool)
    if int(k)<=0:
        return selected
    selectable=masked.clone()
    if exclude is not None:
        selectable=selectable & (~exclude.bool())
    selectable_count=int(selectable.sum().item())
    if selectable_count<=0:
        return selected
    take=min(int(k),selectable_count)
    masked_conf=torch.where(selectable,confidence,torch.full_like(confidence,-float("inf")))
    _,topk_idx=torch.topk(masked_conf.view(-1),k=take)
    selected.view(-1)[topk_idx]=True
    return selected


def _apply_decode_policy_step(
    *,
    tok:Any,
    tokens:torch.Tensor,
    pred_ids:torch.Tensor,
    confidence:torch.Tensor,
    mask_id:int,
    tau_mask:float,
    tau_edit:float,
    budget:int,
    policy:str,
    use_bidirectional_effective:bool,
    remask_cfg:Any|None=None,
    use_remask:bool=False,
    is_last_step:bool=False,
    step_in_block:int=1,
    block_mask_ratio:float=1.0,
)->Tuple[torch.Tensor,Dict[str,Any]]:
    if tokens.shape!=pred_ids.shape or tokens.shape!=confidence.shape:
        raise ValueError("decode step tensors must share the same shape")
    masked=tokens.eq(int(mask_id))
    eligible_gamma=masked & confidence.ge(float(tau_mask))
    eligible_delta=tokens.ne(int(mask_id)) & pred_ids.ne(tokens) & confidence.ge(float(tau_edit))
    updated=tokens.clone()
    applied_gamma=0
    applied_delta=0
    remask_applied=False
    remask_count=0
    noop_reason=""
    tau_mask_active=False
    tau_edit_active=False
    delta_edit_enabled=False
    remask_enabled=False
    low_conf_fallback_applied=False
    low_conf_fallback_count=0
    delta_stage_ready=True
    delta_gate_reason=""
    accept_strategy="none"
    policy_norm=str(policy or "current_base").strip().lower()
    if policy_norm=="current_base":
        n_masked=int(masked.sum().item())
        if n_masked<=0:
            noop_reason="no_masked_tokens"
        elif int(budget)<=0:
            noop_reason="budget_zero"
        else:
            k=min(max(0,int(budget)),n_masked)
            if k<=0:
                noop_reason="budget_zero"
            else:
                topk_mask=_select_mask_topk(masked,confidence,k)
                updated[topk_mask]=pred_ids[topk_mask]
                applied_gamma=int(k)
                accept_strategy="topk_budget"
    elif policy_norm=="threshold_cap":
        tau_mask_active=True
        n_masked=int(masked.sum().item())
        if n_masked<=0:
            noop_reason="no_masked_tokens"
        elif int(budget)<=0:
            noop_reason="budget_zero"
        elif int(eligible_gamma.sum().item())>=max(1,int(budget)):
            updated[eligible_gamma]=pred_ids[eligible_gamma]
            applied_gamma=int(eligible_gamma.sum().item())
            accept_strategy="threshold_accept_all"
        else:
            topk_mask=_select_mask_topk(masked,confidence,int(budget))
            updated[topk_mask]=pred_ids[topk_mask]
            applied_gamma=int(topk_mask.sum().item())
            low_conf_fallback_applied=bool(applied_gamma>0)
            low_conf_fallback_count=int(applied_gamma)
            accept_strategy="threshold_fallback_topk"
            if applied_gamma==0:
                noop_reason="threshold_blocked_budget_zero"
    elif policy_norm in {"shadow_llada21","shadow_llada21_cap","shadow_llada21_delayed_edit"}:
        tau_mask_active=True
        tau_edit_active=True
        delta_edit_enabled=True
        sets=llada21_sets(tokens,pred_ids,confidence,mask_id,float(tau_mask),float(tau_edit))
        gamma_mask=sets.gamma.clone()
        delta_mask=sets.delta.clone()
        if policy_norm=="shadow_llada21_delayed_edit" and int(step_in_block)<2:
            delta_stage_ready=False
            delta_gate_reason="await_initial_draft"
            delta_mask=torch.zeros_like(delta_mask,dtype=torch.bool)
        if sets.gamma_count>0:
            updated[gamma_mask]=pred_ids[gamma_mask]
        if policy_norm in {"shadow_llada21_cap","shadow_llada21_delayed_edit"}:
            gamma_budget_gap=max(0,int(budget)-int(sets.gamma_count))
            if gamma_budget_gap>0 and int(masked.sum().item())>0:
                topup_mask=_select_mask_topk(masked,confidence,gamma_budget_gap,exclude=gamma_mask)
                if int(topup_mask.sum().item())>0:
                    updated[topup_mask]=pred_ids[topup_mask]
                    gamma_mask=gamma_mask | topup_mask
                    low_conf_fallback_applied=True
                    low_conf_fallback_count=int(topup_mask.sum().item())
                    accept_strategy="threshold_edit_fallback_topup"
            elif sets.gamma_count>0:
                accept_strategy="threshold_edit"
        else:
            if sets.gamma_count>0 or sets.delta_count>0:
                accept_strategy="threshold_edit"
        if int(delta_mask.sum().item())>0:
            updated[delta_mask]=pred_ids[delta_mask]
        applied_gamma=int(gamma_mask.sum().item())
        applied_delta=int(delta_mask.sum().item())
        remask_enabled=bool(use_remask and remask_cfg is not None and not is_last_step)
        if remask_enabled:
            remasked=apply_remask(updated,confidence,mask_id,remask_cfg)
            remask_count=int(((updated!=remasked) & remasked.eq(int(mask_id))).sum().item())
            if remask_count>0:
                updated=remasked
                remask_applied=True
        if applied_gamma==0 and applied_delta==0 and not remask_applied:
            if policy_norm=="shadow_llada21_delayed_edit" and not delta_stage_ready:
                noop_reason="delta_gated_until_draft"
            else:
                noop_reason="threshold_blocked_no_fallback" if not low_conf_fallback_applied else "threshold_blocked"
        elif remask_enabled and remask_count==0 and applied_gamma==0 and applied_delta==0:
            noop_reason="remask_noop"
    else:
        raise ValueError(f"unknown decode policy: {policy}")
    if not noop_reason and applied_gamma==0 and applied_delta==0 and not remask_applied:
        noop_reason="no_updates"
    dominant_token,dominant_share,topk_tokens=_topk_token_debug(tok,pred_ids)
    row={
        "decode_policy":policy_norm,
        "use_bidirectional_effective":bool(use_bidirectional_effective),
        "tau_mask_active":bool(tau_mask_active),
        "tau_edit_active":bool(tau_edit_active),
        "delta_edit_enabled":bool(delta_edit_enabled),
        "remask_enabled":bool(remask_enabled),
        "eligible_gamma_count":int(eligible_gamma.sum().item()),
        "applied_gamma_count":int(applied_gamma),
        "eligible_delta_count":int(eligible_delta.sum().item()),
        "applied_delta_count":int(applied_delta),
        "gamma_count":int(applied_gamma),
        "delta_count":int(applied_delta),
        "remask_applied":bool(remask_applied),
        "remask_count":int(remask_count),
        "low_conf_fallback_applied":bool(low_conf_fallback_applied),
        "low_conf_fallback_count":int(low_conf_fallback_count),
        "delta_stage_ready":bool(delta_stage_ready),
        "delta_gate_reason":str(delta_gate_reason),
        "accept_strategy":str(accept_strategy),
        "block_mask_ratio_before":float(block_mask_ratio),
        "dominant_token":dominant_token,
        "dominant_token_share":float(dominant_share),
        "topk_tokens":topk_tokens,
        "noop_reason":str(noop_reason or ""),
    }
    return updated,row


def diagnostic_dllm_decode(
    *,
    model:Any,
    tokenizer:Any,
    prompt:str,
    device:torch.device,
    mask_id:int,
    max_new_tokens:int,
    max_steps:int,
    block_size:int,
    tau_mask:float,
    tau_edit:float,
    temperature:float=0.0,
    decode_policy:str="current_base",
    use_bidirectional:bool=False,
    remask_cfg:Any|None=None,
    eos_guard_enabled:bool=True,
    plateau_patience:int=2,
    plateau_delta_max:int=1,
    cycle_guard_enabled:bool=True,
    allow_tau_fallback_on_degenerate:bool=False,
    degenerate_patience:int=2,
    degenerate_tau_scale:float=0.85,
    min_tau_mask:float=0.05,
    step_callback:Any|None=None,
    is_dummy:bool=False,
)->Dict[str,Any]:
    model.eval()
    enc=tokenizer([prompt],return_tensors="pt")
    input_ids=enc["input_ids"].to(device)
    prompt_len=int(input_ids.shape[1])
    max_new=max(1,int(max_new_tokens))
    seq=torch.full((1,prompt_len+max_new),int(mask_id),dtype=torch.long,device=device)
    seq[:,:prompt_len]=input_ids
    logs:List[Dict[str,Any]]=[]
    converged=False
    stop_guard_triggered=False
    stop_guard_reason=""
    eos_cut_idx:int|None=None
    eos_token_id_raw=getattr(tokenizer,"eos_token_id",None)
    eos_token_id:int|None=None
    if eos_token_id_raw is not None:
        try:
            eos_token_id=int(eos_token_id_raw)
        except Exception:
            eos_token_id=None
    current_tau_mask=float(tau_mask)
    current_tau_edit=float(tau_edit)
    block_len=max(1,int(block_size))
    num_blocks=max(1,(max_new+block_len-1)//block_len)
    steps_per_block=max(1,max(1,int(max_steps))//num_blocks)
    global_step=0
    plateau_streak=0
    no_update_streak=0
    tau_fallback_count=0
    seen_complete_hashes:set[int]=set()
    with torch.inference_mode():
        for block_idx in range(num_blocks):
            block_start=block_idx*block_len
            block_end=min((block_idx+1)*block_len,max_new)
            block_sz=block_end-block_start
            base_budget=block_sz//steps_per_block
            rem_budget=block_sz%steps_per_block
            block_schedule=[base_budget+(1 if i<rem_budget else 0) for i in range(steps_per_block)]
            for bstep in range(steps_per_block):
                gen_before=seq[:,prompt_len:]
                pred,conf=_predict_bidirectional(
                    model,
                    seq,
                    prompt_len,
                    max_new,
                    mask_id,
                    temperature=float(temperature),
                    use_bidirectional=bool(use_bidirectional),
                )
                masked_before=gen_before.eq(int(mask_id))
                avg_masked=float(conf[masked_before].mean().item()) if masked_before.any() else None
                avg_tokens=float(conf[~masked_before].mean().item()) if (~masked_before).any() else None
                block_gen=gen_before[:,block_start:block_end]
                block_pred=pred[:,block_start:block_end]
                block_conf=conf[:,block_start:block_end]
                block_mask_ratio_before=float(block_gen.eq(int(mask_id)).float().mean().item()) if int(block_gen.numel())>0 else 0.0
                is_last_step=bool((global_step+1)>=max(1,int(max_steps)))
                block_updated,step_row=_apply_decode_policy_step(
                    tok=tokenizer,
                    tokens=block_gen,
                    pred_ids=block_pred,
                    confidence=block_conf,
                    mask_id=mask_id,
                    tau_mask=float(current_tau_mask),
                    tau_edit=float(current_tau_edit),
                    budget=int(block_schedule[bstep]),
                    policy=str(decode_policy),
                    use_bidirectional_effective=bool(use_bidirectional),
                    remask_cfg=remask_cfg,
                    use_remask=bool(remask_cfg is not None),
                    is_last_step=is_last_step,
                    step_in_block=int(bstep+1),
                    block_mask_ratio=float(block_mask_ratio_before),
                )
                updated=gen_before.clone()
                updated[:,block_start:block_end]=block_updated
                changed_count=int(updated.ne(gen_before).sum().item())
                remain_count=int(updated.eq(int(mask_id)).sum().item())
                remain=float(remain_count/max(1,int(updated.numel())))
                progress_count=max(0,int(masked_before.sum().item())-remain_count)
                step_row.update(
                    {
                        "step":int(global_step+1),
                        "step_in_block":int(bstep+1),
                        "block":int(block_idx),
                        "block_start":int(block_start),
                        "block_end":int(block_end),
                        "budget":int(block_schedule[bstep]),
                        "mask_ratio":float(remain),
                        "progress_count":int(progress_count),
                        "changed_count":int(changed_count),
                        "avg_conf_masked":avg_masked,
                        "avg_conf_tokens":avg_tokens,
                        "tau_mask":float(current_tau_mask),
                        "tau_edit":float(current_tau_edit),
                    }
                )
                if remain_count>0 and changed_count==0:
                    plateau_streak+=1
                    no_update_streak+=1
                else:
                    plateau_streak=0
                    no_update_streak=0
                tau_fallback_applied=False
                if (
                    remain_count>0
                    and allow_tau_fallback_on_degenerate
                    and str(decode_policy).strip().lower()!="current_base"
                    and no_update_streak>=max(1,int(degenerate_patience))
                ):
                    next_tau_mask=max(float(min_tau_mask),float(current_tau_mask)*float(degenerate_tau_scale))
                    next_tau_edit=max(float(min_tau_mask),float(current_tau_edit)*float(degenerate_tau_scale))
                    if next_tau_mask<float(current_tau_mask) or next_tau_edit<float(current_tau_edit):
                        current_tau_mask=float(next_tau_mask)
                        current_tau_edit=float(next_tau_edit)
                        tau_fallback_applied=True
                        tau_fallback_count+=1
                        no_update_streak=0
                step_row["tau_fallback_applied"]=bool(tau_fallback_applied)
                step_row["tau_mask_next"]=float(current_tau_mask)
                step_row["tau_edit_next"]=float(current_tau_edit)
                seq[:,prompt_len:]=updated
                complete_tokens=updated[0]
                if eos_guard_enabled and eos_token_id is not None:
                    eos_positions=(complete_tokens==int(eos_token_id)).nonzero(as_tuple=False)
                    if int(eos_positions.numel())>0:
                        eos_cut_idx=int(eos_positions[0][0].item())
                        stop_guard_triggered=True
                        stop_guard_reason="eos"
                if (
                    not stop_guard_triggered
                    and cycle_guard_enabled
                    and remain_count>0
                    and changed_count==0
                ):
                    complete_hash=hash(tuple(int(x) for x in complete_tokens.detach().cpu().tolist()))
                    if complete_hash in seen_complete_hashes:
                        stop_guard_triggered=True
                        stop_guard_reason="cycle"
                    else:
                        seen_complete_hashes.add(complete_hash)
                if (
                    not stop_guard_triggered
                    and remain_count>0
                    and changed_count<=max(0,int(plateau_delta_max))
                    and changed_count==0
                    and plateau_streak>=max(1,int(plateau_patience))
                ):
                    stop_guard_triggered=True
                    stop_guard_reason="plateau"
                if stop_guard_triggered:
                    step_row["stop_guard_triggered"]=True
                    step_row["stop_guard_reason"]=str(stop_guard_reason)
                    logs.append(step_row)
                    if callable(step_callback):
                        try:
                            step_callback(dict(step_row))
                        except Exception:
                            pass
                    converged=True
                    break
                logs.append(step_row)
                if callable(step_callback):
                    try:
                        step_callback(dict(step_row))
                    except Exception:
                        pass
                global_step+=1
                if remain_count==0:
                    converged=True
                    break
                block_remain=int(updated[:,block_start:block_end].eq(int(mask_id)).sum().item())
                if block_remain==0:
                    break
            if converged:
                break
        final_remain=float(seq[:,prompt_len:].eq(int(mask_id)).float().mean().item())
        if final_remain==0.0 and not stop_guard_triggered:
            converged=True
    out_ids=seq[0,prompt_len:]
    if eos_cut_idx is not None:
        out_ids=out_ids[:max(0,int(eos_cut_idx)+1)]
    tokens_generated=int((out_ids!=int(mask_id)).sum().item()) if int(out_ids.numel())>0 else 0
    unique_tokens=int(len({int(x) for x in out_ids.detach().cpu().tolist() if int(x)!=int(mask_id)}))
    text=_decode_text(tokenizer,out_ids,mask_id,is_dummy)
    stc=next((int(l["step"]) for l in logs if float(l.get("mask_ratio",1.0))==0.0),len(logs))
    finish_reason=str(stop_guard_reason) if stop_guard_reason else ("converged" if converged else "length")
    return {
        "text":str(text).strip(),
        "logs":logs,
        "steps":int(len(logs)),
        "steps_to_converge":int(stc),
        "tokens_generated":int(tokens_generated),
        "unique_token_count":int(unique_tokens),
        "finish_reason":finish_reason,
        "truncated":True if finish_reason=="length" else False,
        "stop_guard_triggered":bool(stop_guard_triggered),
        "stop_guard_reason":str(stop_guard_reason),
        "decode_policy":str(decode_policy),
        "use_bidirectional_effective":bool(use_bidirectional),
        "tau_fallback_count":int(tau_fallback_count),
        "final_mask_ratio":float(final_remain),
    }

def _decode_text(tok:Any,out_ids:torch.Tensor,mask_id:int,is_dummy:bool)->str:
    text=""
    if hasattr(tok,"decode"):
        try:
            text=tok.decode(out_ids,skip_special_tokens=True)
        except Exception:
            text=""
    if text and text.strip():
        t=text.strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    raw=[int(x) for x in out_ids.detach().cpu().tolist() if int(x)!=int(mask_id)]
    if raw:
        t=" ".join(f"tok{x}" for x in raw[:64]).strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    return "[DUMMY] dummy-output" if is_dummy else ""

class BaseEngine:
    name="base"
    def __init__(self,cfg:AppConfig,trace=None):
        self.cfg=cfg
        self.trace=use_trace(cfg,trace)
        self.last_stats:Dict[str,Any]={}
    def generate(
        self,
        prompt:str,
        mode:str="S_MODE",
        tau_mask:float|None=None,
        tau_edit:float|None=None,
        max_new_tokens:int|None=None,
        seed:int|None=None,
        effort:str="medium",
        temperature:float|None=None,
        top_p:float|None=None,
        top_k:int|None=None,
        presence_penalty:float|None=None,
        repetition_penalty:float|None=None,
    )->str:
        raise NotImplementedError
    def close(self)->None:
        return

class TransformersEngine(BaseEngine):
    name="transformers"
    def __init__(self,cfg:AppConfig,fallback_reason:str="",trace=None):
        super().__init__(cfg,trace=trace)
        self.bundle=load_model_bundle(cfg,for_training=False,trace=self.trace)
        self.fallback_reason=fallback_reason or self.bundle.load_reason
    def _decode(
        self,
        prompt:str,
        mode:str,
        tau_mask:float|None,
        tau_edit:float|None,
        max_new_tokens:int|None,
        seed:int|None,
        effort:str="medium",
        temperature:float|None=None,
        top_p:float|None=None,
        top_k:int|None=None,
        presence_penalty:float|None=None,
        repetition_penalty:float|None=None,
    )->str:
        if seed is None:
            seed=self.cfg.runtime.seed
        seed_everything(seed)
        tau_m,tau_e=mode_thresholds(self.cfg,mode,tau_mask,tau_edit)
        max_new=max(1,int(max_new_tokens or self.cfg.inference.max_new_tokens))
        # apply effort knob (overrides steps and scales tau)
        eff_steps,tau_m,tau_e=_resolve_effort(effort,max(1,int(self.cfg.inference.max_steps)),tau_m,tau_e)
        tok=self.bundle.tokenizer
        model=self.bundle.model
        device=self.bundle.device
        mask_id=self.bundle.mask_id
        t0=time.time()
        model.eval()
        if device.type=="cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device.index or 0)
            except Exception:
                pass
        allow_fallback=bool(self.cfg.inference.allow_tau_fallback_on_degenerate)
        eos_guard_enabled=bool(getattr(self.cfg.runtime,"dllm_stop_eos_enabled",True))
        plateau_patience=max(1,int(getattr(self.cfg.runtime,"dllm_stop_plateau_patience",2)))
        plateau_delta_ratio=max(0.0,float(getattr(self.cfg.runtime,"dllm_stop_plateau_delta_ratio",0.01)))
        plateau_delta_max=max(1,int(_math.ceil(float(max_new)*plateau_delta_ratio)))
        cycle_guard_enabled=bool(getattr(self.cfg.runtime,"dllm_stop_cycle_enabled",True))
        step_callback=getattr(self.cfg.runtime,"inference_step_callback",None)
        if not callable(step_callback):
            step_callback=None
        _temperature=float(temperature if temperature is not None else 0.0)
        base_policy=str(getattr(self.cfg.runtime,"dllm_base_policy","current_base") or "current_base")
        base_use_bidirectional=bool(getattr(self.cfg.runtime,"dllm_base_use_bidirectional",False))
        base_use_remask=bool(getattr(self.cfg.runtime,"dllm_base_use_remask",False))
        shadow_enabled=bool(getattr(self.cfg.runtime,"dllm_shadow_enabled",True))
        shadow_policy=str(getattr(self.cfg.runtime,"dllm_shadow_policy","shadow_llada21") or "shadow_llada21")
        shadow_use_bidirectional=bool(getattr(self.cfg.runtime,"dllm_shadow_use_bidirectional",False))
        shadow_use_remask=bool(getattr(self.cfg.runtime,"dllm_shadow_use_remask",True))
        base_result=diagnostic_dllm_decode(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            device=device,
            mask_id=mask_id,
            max_new_tokens=max_new,
            max_steps=eff_steps,
            block_size=max(1,int(getattr(self.cfg.inference,"block_size",128))),
            tau_mask=float(tau_m),
            tau_edit=float(tau_e),
            temperature=float(_temperature),
            decode_policy=base_policy,
            use_bidirectional=bool(base_use_bidirectional),
            remask_cfg=self.cfg.inference.remask if base_use_remask else None,
            eos_guard_enabled=eos_guard_enabled,
            plateau_patience=plateau_patience,
            plateau_delta_max=plateau_delta_max,
            cycle_guard_enabled=cycle_guard_enabled,
            allow_tau_fallback_on_degenerate=allow_fallback if base_policy!="current_base" else False,
            degenerate_patience=max(1,int(self.cfg.inference.degenerate_patience)),
            degenerate_tau_scale=float(self.cfg.inference.degenerate_tau_scale),
            min_tau_mask=float(self.cfg.inference.min_tau_mask),
            step_callback=step_callback,
            is_dummy=self.bundle.is_dummy,
        )
        shadow_result=None
        if shadow_enabled:
            shadow_result=diagnostic_dllm_decode(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                device=device,
                mask_id=mask_id,
                max_new_tokens=max_new,
                max_steps=eff_steps,
                block_size=max(1,int(getattr(self.cfg.inference,"block_size",128))),
                tau_mask=float(tau_m),
                tau_edit=float(tau_e),
                temperature=float(_temperature),
                decode_policy=shadow_policy,
                use_bidirectional=bool(shadow_use_bidirectional),
                remask_cfg=self.cfg.inference.remask if shadow_use_remask else None,
                eos_guard_enabled=eos_guard_enabled,
                plateau_patience=plateau_patience,
                plateau_delta_max=plateau_delta_max,
                cycle_guard_enabled=cycle_guard_enabled,
                allow_tau_fallback_on_degenerate=allow_fallback,
                degenerate_patience=max(1,int(self.cfg.inference.degenerate_patience)),
                degenerate_tau_scale=float(self.cfg.inference.degenerate_tau_scale),
                min_tau_mask=float(self.cfg.inference.min_tau_mask),
                step_callback=None,
                is_dummy=self.bundle.is_dummy,
            )
        elapsed=max(1e-6,time.time()-t0)
        text=str(base_result.get("text","")).strip()
        logs=list(base_result.get("logs",[]))
        tokens_generated=int(base_result.get("tokens_generated",0) or 0)
        vram_peak=None
        if self.bundle.device.type=="cuda":
            try:
                vram_peak=float(torch.cuda.max_memory_allocated(self.bundle.device.index or 0))
            except Exception:
                pass
        self.last_stats={
            "engine":self.name,
            "mode":mode,
            "effort":effort,
            "tau_mask":tau_m,
            "tau_edit":tau_e,
            "steps":int(base_result.get("steps",len(logs))),
            "steps_to_converge":int(base_result.get("steps_to_converge",len(logs))),
            "logs":logs,
            "tokens_generated":tokens_generated,
            "tokens_per_sec":tokens_per_second(tokens_generated,elapsed),
            "vram_peak_bytes":vram_peak,
            "json_valid_rate":None,
            "fallback_reason":self.fallback_reason,
            "dummy_model":self.bundle.is_dummy,
            "load_reason":self.bundle.load_reason,
            "device":self.bundle.device.type,
            "model_name_or_path":self.bundle.model_name_or_path,
            "requested_dtype":self.bundle.requested_dtype,
            "actual_dtype":self.bundle.actual_dtype,
            "finish_reason":str(base_result.get("finish_reason","length")),
            "truncated":bool(base_result.get("truncated",False)),
            "stop_guard_triggered":bool(base_result.get("stop_guard_triggered",False)),
            "stop_guard_reason":str(base_result.get("stop_guard_reason","")),
            "decode_policy":str(base_result.get("decode_policy","current_base")),
            "use_bidirectional_effective":bool(base_result.get("use_bidirectional_effective",False)),
            "unique_token_count":int(base_result.get("unique_token_count",0) or 0),
            "final_mask_ratio":float(base_result.get("final_mask_ratio",1.0)),
            "shadow_enabled":bool(shadow_enabled),
            "base_policy_configured":base_policy,
            "base_use_bidirectional_configured":bool(base_use_bidirectional),
            "base_use_remask_configured":bool(base_use_remask),
            "shadow_policy_configured":shadow_policy,
            "shadow_use_bidirectional_configured":bool(shadow_use_bidirectional),
            "shadow_use_remask_configured":bool(shadow_use_remask),
            "shadow":shadow_result,
            "tokenizer_fix_mistral_regex":bool(getattr(tok,"fix_mistral_regex",False)),
            "env_issues":self.bundle.env_issues,
            "fallbacks":self.trace.snapshot_fallbacks(limit=64) if self.trace is not None else []
        }
        try:
            self.last_stats["summary_toon"]=dumps_toon(
                {
                    "mode":mode,
                    "effort":effort,
                    "tokenizer_fix_mistral_regex":bool(getattr(tok,"fix_mistral_regex",False)),
                    "base":{
                        "decode_policy":self.last_stats["decode_policy"],
                        "use_bidirectional_effective":self.last_stats["use_bidirectional_effective"],
                        "finish_reason":self.last_stats["finish_reason"],
                        "steps":self.last_stats["steps"],
                        "tokens_generated":self.last_stats["tokens_generated"],
                        "unique_token_count":self.last_stats["unique_token_count"],
                        "final_mask_ratio":self.last_stats["final_mask_ratio"],
                        "first_step":logs[0] if logs else {},
                        "last_step":logs[-1] if logs else {},
                    },
                    "shadow":{
                        "enabled":bool(shadow_enabled),
                        "decode_policy":str((shadow_result or {}).get("decode_policy","")),
                        "use_bidirectional_effective":bool((shadow_result or {}).get("use_bidirectional_effective",False)),
                        "finish_reason":str((shadow_result or {}).get("finish_reason","")),
                        "steps":int((shadow_result or {}).get("steps",0) or 0),
                        "tokens_generated":int((shadow_result or {}).get("tokens_generated",0) or 0),
                        "unique_token_count":int((shadow_result or {}).get("unique_token_count",0) or 0),
                        "final_mask_ratio":float((shadow_result or {}).get("final_mask_ratio",1.0)),
                    },
                },
                root_key="dllm",
            )
        except Exception:
            pass
        return text.strip()
    def generate(
        self,
        prompt:str,
        mode:str="S_MODE",
        tau_mask:float|None=None,
        tau_edit:float|None=None,
        max_new_tokens:int|None=None,
        seed:int|None=None,
        effort:str="medium",
        temperature:float|None=None,
        top_p:float|None=None,
        top_k:int|None=None,
        presence_penalty:float|None=None,
        repetition_penalty:float|None=None,
    )->str:
        return self._decode(
            prompt,
            mode,
            tau_mask,
            tau_edit,
            max_new_tokens,
            seed,
            effort=effort,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
    def close(self)->None:
        try:
            if hasattr(self,"bundle") and self.bundle is not None:
                if hasattr(self.bundle,"model"):
                    del self.bundle.model
                if hasattr(self.bundle,"tokenizer"):
                    del self.bundle.tokenizer
                self.bundle=None
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

class DInferEngine(BaseEngine):
    name="dinfer"
    def __init__(self,cfg:AppConfig,trace=None):
        super().__init__(cfg,trace=trace)
        vendor_py=Path(cfg.paths.vendor_dinfer)/"python"
        if not vendor_py.exists():
            raise RuntimeError("dInfer python package path missing")
        sys.path.insert(0,str(vendor_py))
        from dinfer import SamplingParams,DiffusionLLMServing
        self.SamplingParams=SamplingParams
        self.DiffusionLLMServing=DiffusionLLMServing
        self.tokenizer=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code,trace=self.trace,cfg=cfg)
        self.server=None
        self._init_server()
    def _init_server(self)->None:
        sample=self.SamplingParams(
            threshold=self.cfg.inference.q_mode_tau_mask,
            cache="prefix",
            temperature=0.0,
            early_stop=True,
            cont_weight=0.0,
            prefix_look=0,
            after_look=0,
            warmup_steps=0,
            enable_torch_compile=False,
            parallel_decoding="threshold",
            use_credit=False,
            use_bd=True,
            max_length=max(1024,self.cfg.data.seq_len),
            batch_size=1,
            mini_batch_size=1
        )
        self.server=self.DiffusionLLMServing(
            self.cfg.paths.model_dir,
            model_type=self.cfg.runtime.dinfer_model_type,
            sample_params=sample,
            backend=self.cfg.runtime.dinfer_backend,
            num_gpus=1,
            dp_size=1,
            tpep_size=1
        )
    def close(self)->None:
        try:
            if self.server is not None and hasattr(self.server,"stop_serving"):
                self.server.stop_serving()
        except Exception:
            pass
    def generate(
        self,
        prompt:str,
        mode:str="S_MODE",
        tau_mask:float|None=None,
        tau_edit:float|None=None,
        max_new_tokens:int|None=None,
        seed:int|None=None,
        effort:str="medium",
        temperature:float|None=None,
        top_p:float|None=None,
        top_k:int|None=None,
        presence_penalty:float|None=None,
        repetition_penalty:float|None=None,
    )->str:
        tau_m,_=mode_thresholds(self.cfg,mode,tau_mask,tau_edit)
        eff_steps,tau_m,_=_resolve_effort(effort,8,tau_m,tau_m)
        if hasattr(self.server,"sample_params"):
            try:
                self.server.sample_params.threshold=float(tau_m)
            except Exception:
                pass
        max_new=int(max_new_tokens or self.cfg.inference.max_new_tokens)
        x=self.tokenizer([prompt],return_tensors="pt")["input_ids"]
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t0=time.time()
        with torch.inference_mode():
            out=self.server.generate(x,gen_length=max_new,block_length=self.cfg.inference.block_size)
        elapsed=max(1e-6,time.time()-t0)
        prompt_len=int(x.shape[1])
        text=self.tokenizer.decode(out[0,prompt_len:],skip_special_tokens=True)
        if not text.strip():
            text="[DUMMY] dummy-output"
        self.last_stats={
            "engine":self.name,
            "mode":mode,
            "effort":effort,
            "tau_mask":tau_m,
            "tau_edit":tau_mask,
            "steps":-1,
            "steps_to_converge":None,
            "tokens_per_sec":tokens_per_second(int(max_new),elapsed),
            "vram_peak_bytes":None,
            "json_valid_rate":None,
            "device":"cuda" if torch.cuda.is_available() else "cpu",
            "actual_dtype":"unknown",
            "finish_reason":"length",
            "truncated":True,
            "fallbacks":self.trace.snapshot_fallbacks(limit=64) if self.trace is not None else [],
        }
        return text.strip()

def build_engine(cfg:AppConfig,trace=None)->BaseEngine:
    tr=use_trace(cfg,trace)
    err=""
    if cfg.runtime.use_dinfer:
        try:
            return DInferEngine(cfg,trace=tr)
        except Exception as e:
            err=str(e)
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="inference",
                    func="build_engine",
                    action="engine_fallback",
                    reason="dinfer_missing",
                    exception_str=exception_with_stack(e),
                    extra_dict={"engine_requested":"dinfer","backend":cfg.runtime.dinfer_backend}
                )
    return TransformersEngine(cfg,fallback_reason=err,trace=tr)

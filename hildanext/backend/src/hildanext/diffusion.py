# Diffusion schedule/objective utilities for SAFE training and decode.
# Main entrypoints: wsd_block,compute_m2t_t2t_losses,apply_remask.
# Implements WSD + M2T/T2T with doc-mask aware forward fallback.
from __future__ import annotations
import contextlib
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict,Tuple,Optional
import torch
import torch.nn.functional as F
from .config import WSDConfig,TrainConfig,RemaskConfig
from .masks import batch_doc_attention_mask
from .formulas import llada2_wsd_block
from .trace import use_trace,exception_with_stack

# ---------------------------------------------------------------------------
# force_noncausal_attention — context manager
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def force_noncausal_attention(model:torch.nn.Module):
    """Temporarily disable causal masking inside a HuggingFace transformer model.

    Two independent causal pathways exist in transformers >=4.57:
      1) ``module.is_causal = True`` on each attention layer → controls the
         ``is_causal`` flag passed to ``F.scaled_dot_product_attention``.
      2) ``create_causal_mask`` called in ``Qwen3Model.forward`` → builds a
         4D lower-triangular mask added on top of the user-supplied mask.

    This context manager neutralises **both**:
      - Sets ``is_causal = False`` on every attention sub-module.
      - Monkey-patches ``create_causal_mask`` in ``transformers.models.qwen3``
        (and ``masking_utils``) to return the user-provided mask unchanged
        (preserving padding / custom composite masks).

    The original state is always restored in ``finally``.
    """
    # --- 1) Flip is_causal on all attention sub-modules ---
    saved_causal:list[tuple]=[]
    for name,mod in model.named_modules():
        if hasattr(mod,"is_causal"):
            saved_causal.append((mod,mod.is_causal))
            mod.is_causal=False

    # --- 2) Monkey-patch create_causal_mask ---
    #   transformers 4.57 Qwen3 calls create_causal_mask in forward().
    #   The original function already early-exits when a 4D mask is provided
    #   (returns the user's 4D mask as-is via _preprocess_mask_arguments).
    #   When NO 4D mask is provided, it builds a causal lower-triangular mask.
    #   Our patch must: (a) pass through any user-provided 4D mask unchanged,
    #   (b) when no 4D mask is given, return a fully-permissive (non-causal) mask
    #   so that attention is bidirectional.
    _patched_modules:list[tuple]=[]
    try:
        import transformers.masking_utils as _mu
        _orig_mu=_mu.create_causal_mask
        def _noncausal_mask(*args,**kwargs):
            # Call original — it already early-exits for 4D masks, returning them as-is.
            # For 2D/None masks, it would build a causal mask. We call it, then
            # if the result is a causal mask, replace it with a non-causal one.
            result=_orig_mu(*args,**kwargs)
            if result is None:
                return None
            # If result is the user's 4D mask (early-exit path), return as-is.
            # The user's 4D mask already encodes composite block structure.
            # We detect the early-exit case: the result equals the attention_mask kwarg.
            user_mask=kwargs.get("attention_mask",None)
            if user_mask is None and len(args)>=3:
                user_mask=args[2]  # positional: config, input_embeds, attention_mask
            if user_mask is not None and result is user_mask:
                return result  # pass through the user's composite mask
            # Otherwise result is a freshly-built causal mask — make it non-causal
            # by filling with zeros (all positions visible).
            if isinstance(result,torch.Tensor):
                return torch.zeros_like(result)
            return result
        _mu.create_causal_mask=_noncausal_mask
        _patched_modules.append((_mu,"create_causal_mask",_orig_mu))
    except (ImportError,AttributeError):
        pass

    try:
        import transformers.models.qwen3.modeling_qwen3 as _qm
        if hasattr(_qm,"create_causal_mask"):
            _orig_qm=_qm.create_causal_mask
            _qm.create_causal_mask=_noncausal_mask  # type: ignore[has-type]
            _patched_modules.append((_qm,"create_causal_mask",_orig_qm))
    except (ImportError,AttributeError):
        pass

    # Also patch for Qwen2 (some transformers versions alias Qwen3→Qwen2)
    try:
        import transformers.models.qwen2.modeling_qwen2 as _q2
        if hasattr(_q2,"create_causal_mask"):
            _orig_q2=_q2.create_causal_mask
            _q2.create_causal_mask=_noncausal_mask  # type: ignore[has-type]
            _patched_modules.append((_q2,"create_causal_mask",_orig_q2))
    except (ImportError,AttributeError):
        pass

    try:
        yield model
    finally:
        # --- Restore is_causal ---
        for mod,orig_val in saved_causal:
            mod.is_causal=orig_val
        # --- Restore create_causal_mask ---
        for mod_obj,attr_name,orig_fn in _patched_modules:
            setattr(mod_obj,attr_name,orig_fn)

@dataclass
class WSDStep:
    phase:str
    block_size:int

def wsd_block(step:int,cfg:WSDConfig,seq_len:int|None=None)->WSDStep:
    phase,block=llada2_wsd_block(
        step=step,
        warmup_steps=cfg.warmup_steps,
        stable_steps=cfg.stable_steps,
        decay_steps=cfg.decay_steps,
        start_block=cfg.start_block_size,
        max_block=cfg.max_block_size,
        end_block=cfg.end_block_size,
        seq_len=seq_len,
        ladder_blocks=cfg.ladder_blocks,
        decay_blocks=cfg.decay_blocks,
        enforce_divisibility=cfg.enforce_divisibility
    )
    return WSDStep(phase=phase,block_size=block)

def _pick_positions(base_mask:torch.Tensor,p:float)->torch.Tensor:
    p=min(max(float(p),0.0),1.0)
    r=torch.rand_like(base_mask.float())
    m=(r<p) & base_mask.bool()
    if not torch.any(m):
        idx=torch.nonzero(base_mask,as_tuple=False)
        if idx.numel()>0:
            m[idx[0,0],idx[0,1]]=True
    return m

def _causal_loss(logits:torch.Tensor,labels:torch.Tensor,num_chunks:int=8)->torch.Tensor:
    """Cross-entropy with left-shift. When few positions are valid (masked diffusion),
    gathers only those positions' logits to avoid materializing a full (B,S,V) view.
    For V=151936 and ~15% mask ratio: 297MB → ~44MB logits in backward."""
    if logits.shape[1]<2:
        return torch.tensor(0.0,device=logits.device)
    y=labels[:,1:].contiguous()
    if not torch.any(y.ne(-100)):
        return logits.sum()*0.0
    S=logits.shape[1]-1
    V=logits.shape[-1]
    valid_mask=y.ne(-100)
    n_valid=int(valid_mask.sum().item())
    # Gather path: when <50% positions are valid, gather is much cheaper
    if n_valid>0 and n_valid<S*y.shape[0]*0.5:
        flat_logits=logits[:,:-1,:].reshape(-1,V)
        flat_labels=y.reshape(-1)
        flat_valid=valid_mask.reshape(-1)
        idx=flat_valid.nonzero(as_tuple=True)[0]
        gathered_logits=flat_logits[idx]
        gathered_labels=flat_labels[idx]
        return F.cross_entropy(gathered_logits,gathered_labels,reduction="mean")
    # Chunked fallback for dense labels
    if S*V<500_000 or num_chunks<=1:
        l=logits[:,:-1,:].contiguous()
        return F.cross_entropy(l.view(-1,V),y.view(-1),ignore_index=-100)
    chunk_size=max(1,(S+num_chunks-1)//num_chunks)
    total_loss=torch.tensor(0.0,device=logits.device,dtype=torch.float32)
    total_count=0
    for i in range(0,S,chunk_size):
        end=min(i+chunk_size,S)
        l_chunk=logits[:,i:end,:].contiguous()
        y_chunk=y[:,i:end].contiguous()
        valid=y_chunk.ne(-100).sum().item()
        if valid==0:
            continue
        ce=F.cross_entropy(l_chunk.view(-1,V),y_chunk.view(-1),ignore_index=-100,reduction="sum")
        total_loss=total_loss+ce
        total_count+=valid
    if total_count==0:
        return logits.sum()*0.0
    return total_loss/float(total_count)

def _attn_for_model(mask:torch.Tensor,model)->torch.Tensor:
    m=mask.bool()
    if m.dim()==3:
        m=m[:,None,:,:]
    dt=torch.float32
    try:
        dt=next(model.parameters()).dtype
    except Exception:
        dt=torch.float32
    out=torch.zeros(m.shape,device=m.device,dtype=dt)
    out=out.masked_fill(~m,torch.finfo(dt).min)
    return out

_sdpa_backend_logged=False
def reset_sdpa_probe():
    global _sdpa_backend_logged
    _sdpa_backend_logged=False
_embed_noise_hook=None
_embed_noise_mask_id=None
_embed_noise_std=0.0

def _install_embed_noise_hook(model,mask_id:int,noise_std:float=0.1):
    """LLaDA2.0 Sec 7.1: add Gaussian noise to embedding output for masked tokens.
    Prevents gradient explosion from near-zero mask embeddings during WSD warmup."""
    global _embed_noise_hook,_embed_noise_mask_id,_embed_noise_std
    _embed_noise_mask_id=mask_id
    _embed_noise_std=noise_std
    embed_layer=None
    for name,mod in model.named_modules():
        if name.endswith("embed_tokens") or (hasattr(mod,"weight") and isinstance(mod,torch.nn.Embedding) and mod.weight.shape[0]>1000):
            embed_layer=mod
            break
    if embed_layer is None:
        return
    def _hook(module,input,output):
        if _embed_noise_std<=0 or _embed_noise_mask_id is None:
            return output
        if len(input)>0 and isinstance(input[0],torch.Tensor):
            ids=input[0]
            mask_pos=(ids==_embed_noise_mask_id)
            if mask_pos.any():
                noise=torch.randn_like(output)*_embed_noise_std
                noise[~mask_pos]=0.0
                return output+noise
        return output
    _embed_noise_hook=embed_layer.register_forward_hook(_hook)
    print(f"[EMBED_NOISE] installed hook on embedding, mask_id={mask_id}, std={noise_std}",flush=True)

def _remove_embed_noise_hook():
    global _embed_noise_hook,_embed_noise_std
    if _embed_noise_hook is not None:
        _embed_noise_hook.remove()
        _embed_noise_hook=None
    _embed_noise_std=0.0

def set_embed_noise_std(std:float):
    global _embed_noise_std
    _embed_noise_std=std

def _forward(model,input_ids:torch.Tensor,attn_1d:torch.Tensor,doc_ids:torch.Tensor,mask_mode:str,clean_ids:Optional[torch.Tensor]=None,composite_block_size:int|None=None,trace=None,cfg=None,bidirectional:bool=False):
    global _sdpa_backend_logged
    tr=use_trace(cfg,trace)
    try:
        if (mask_mode or "").lower()=="composite_llada20":
            x0=clean_ids if clean_ids is not None else input_ids
            l=int(input_ids.shape[1])
            ids2=torch.cat([input_ids,x0],dim=1)
            docs2=torch.cat([doc_ids,doc_ids],dim=1)
            mask2=batch_doc_attention_mask(docs2,causal=False,mask_mode=mask_mode,block_size=composite_block_size,base_len=l)
            attn4d=_attn_for_model(mask2,model)
            if not _sdpa_backend_logged and torch.cuda.is_available():
                _sdpa_backend_logged=True
                try:
                    H=model.config.num_attention_heads
                    D=model.config.head_dim if hasattr(model.config,"head_dim") else model.config.hidden_size//H
                    S2=attn4d.shape[-1]
                    q_probe=torch.randn(1,H,S2,D,device=attn4d.device,dtype=attn4d.dtype)
                    bid=torch._fused_sdp_choice(q_probe,q_probe,q_probe,attn_mask=attn4d[:1],dropout_p=0.0,is_causal=False)
                    from torch.nn.attention import SDPBackend
                    bname=SDPBackend(bid).name
                    del q_probe
                    print(f"[SDPA_DIAG] backend={bname} mask_shape={list(attn4d.shape)} dtype={attn4d.dtype} H={H} D={D}",flush=True)
                except Exception as e:
                    print(f"[SDPA_DIAG] probe_failed: {e}",flush=True)
            # Option-3 optimisation: run backbone on full 2S tokens for
            # attention context, but apply lm_head only on the first S (x_t)
            # positions.  Saves ~594 MB VRAM (lm_head on S vs 2S with V=151936).
            _backbone=getattr(model,"model",None)
            _lm_head=getattr(model,"lm_head",None)
            _use_slim=(_backbone is not None and _lm_head is not None)
            if _use_slim:
                if bidirectional:
                    with force_noncausal_attention(model):
                        backbone_out=_backbone(input_ids=ids2,attention_mask=attn4d)
                else:
                    backbone_out=_backbone(input_ids=ids2,attention_mask=attn4d)
                hidden_xt=backbone_out[0][:,:l,:].contiguous()
                del backbone_out
                logits=_lm_head(hidden_xt)
                del hidden_xt
            else:
                # Fallback: full model call (non-standard architectures)
                if bidirectional:
                    with force_noncausal_attention(model):
                        out=model(input_ids=ids2,attention_mask=attn4d)
                else:
                    out=model(input_ids=ids2,attention_mask=attn4d)
                logits=out.logits[:,:l,:].contiguous()
                del out
            return SimpleNamespace(logits=logits)
        doc_mask=batch_doc_attention_mask(doc_ids,causal=not bidirectional,mask_mode=mask_mode)
        attn4d=_attn_for_model(doc_mask,model)
        if bidirectional:
            with force_noncausal_attention(model):
                return model(input_ids=input_ids,attention_mask=attn4d)
        return model(input_ids=input_ids,attention_mask=attn4d)
    except Exception as e:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="diffusion",
                func="_forward",
                action="attention_mask_fallback",
                reason="doc_attention_mask_failed",
                exception_str=exception_with_stack(e),
                extra_dict={"mask_mode":mask_mode}
            )
        return model(input_ids=input_ids,attention_mask=attn_1d)

def _make_m2t_batch(input_ids:torch.Tensor,attn_mask:torch.Tensor,response_mask:torch.Tensor|None,mask_id:int,ratio:float)->Tuple[torch.Tensor,torch.Tensor]:
    cand=attn_mask.bool()
    if response_mask is not None:
        cand=cand & response_mask.bool()
    pos=_pick_positions(cand,ratio)
    noisy=input_ids.clone()
    noisy[pos]=mask_id
    labels=torch.full_like(input_ids,-100)
    labels[pos]=input_ids[pos]
    return noisy,labels

def _continuous_time_m2t_batch(input_ids:torch.Tensor,attn_mask:torch.Tensor,response_mask:torch.Tensor|None,mask_id:int,t_min:float=0.001,t_max:float=1.0)->Tuple[torch.Tensor,torch.Tensor,float,float]:
    """Continuous-time M2T: sample t~U(t_min,t_max), mask each token i.i.d. with p=t.
    Returns (noisy_ids, labels, t_sampled, mask_ratio_actual).
    Ref: pplx-embed Sec 2.1 — absorbing-state corruption with continuous t."""
    t=float(torch.empty(1).uniform_(t_min,t_max).item())
    cand=attn_mask.bool()
    if response_mask is not None:
        cand=cand & response_mask.bool()
    rand=torch.rand_like(input_ids.float())
    mask=(rand<t) & cand
    if not mask.any():
        idx=torch.nonzero(cand,as_tuple=False)
        if idx.numel()>0:
            mask[idx[0,0],idx[0,1]]=True
    noisy=input_ids.clone()
    noisy[mask]=mask_id
    labels=torch.full_like(input_ids,-100)
    labels[mask]=input_ids[mask]
    n_cand=max(1,int(cand.sum().item()))
    mask_ratio=float(mask.sum().item())/n_cand
    return noisy,labels,t,mask_ratio

def t2t_corrupt_tokens(input_ids:torch.Tensor,attn_mask:torch.Tensor,response_mask:torch.Tensor|None,ratio:float,vocab_size:int)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    cand=attn_mask.bool()
    if response_mask is not None:
        cand=cand & response_mask.bool()
    pos=_pick_positions(cand,ratio)
    noisy=input_ids.clone()
    rnd=torch.randint(low=0,high=max(8,vocab_size),size=input_ids.shape,device=input_ids.device)
    same=pos & rnd.eq(input_ids)
    if same.any():
        rnd[same]=(rnd[same]+1)%max(8,vocab_size)
    noisy[pos]=rnd[pos]
    labels=torch.full_like(input_ids,-100)
    labels[pos]=input_ids[pos]
    return noisy,labels,pos

def _make_t2t_batch(input_ids:torch.Tensor,attn_mask:torch.Tensor,response_mask:torch.Tensor|None,ratio:float,vocab_size:int)->Tuple[torch.Tensor,torch.Tensor]:
    x,y,_=t2t_corrupt_tokens(input_ids,attn_mask,response_mask,ratio,vocab_size)
    return x,y

def compute_m2t_t2t_losses(model,input_ids:torch.Tensor,attention_mask:torch.Tensor,doc_ids:torch.Tensor,response_mask:torch.Tensor|None,mask_id:int,vocab_size:int,cfg:TrainConfig,focus_response:bool,mask_mode:str="simple_blockdiag",composite_block_size:int|None=None,trace=None,cfg_obj=None,bidirectional:bool=False,time_param:str="discrete",loss_weighting:str="none",t_min:float=0.001,t_max:float=1.0,target_ids:Optional[torch.Tensor]=None)->Dict[str,torch.Tensor]:
    """Merged M2T+T2T: single forward pass with mixed corruption (LLaDA 2.1 recipe).
    M2T positions: masked with [MASK], T2T positions: replaced with random tokens.
    Disjoint sets → single forward, separate loss streams.
    If target_ids is provided (MTF turn>1), input_ids is the base to corrupt
    (may contain model predictions), target_ids holds the ground-truth labels."""
    _target=target_ids if target_ids is not None else input_ids
    focus=response_mask if focus_response else None
    t_sampled=float(cfg.mask_ratio)
    mask_ratio_actual=0.0
    cand=attention_mask.bool()
    if focus is not None:
        cand=cand & focus.bool()
    if time_param=="continuous_time":
        t=float(torch.empty(1).uniform_(t_min,t_max).item())
        rand_m=torch.rand_like(input_ids.float())
        m2t_pos=(rand_m<t) & cand
        if not m2t_pos.any():
            idx=torch.nonzero(cand,as_tuple=False)
            if idx.numel()>0:
                m2t_pos[idx[0,0],idx[0,1]]=True
        t_sampled=t
        n_cand=max(1,int(cand.sum().item()))
        mask_ratio_actual=float(m2t_pos.sum().item())/n_cand
    else:
        rand_m=torch.rand_like(input_ids.float())
        m2t_pos=(rand_m<float(cfg.mask_ratio)) & cand
        if not m2t_pos.any():
            idx=torch.nonzero(cand,as_tuple=False)
            if idx.numel()>0:
                m2t_pos[idx[0,0],idx[0,1]]=True
        t_sampled=float(cfg.mask_ratio)
        n_cand=max(1,int(cand.sum().item()))
        mask_ratio_actual=float(m2t_pos.sum().item())/n_cand
    remaining=cand & ~m2t_pos
    rand_t=torch.rand_like(input_ids.float())
    t2t_pos=(rand_t<float(cfg.t2t_noise_ratio)) & remaining
    mixed=input_ids.clone()
    mixed[m2t_pos]=mask_id
    if t2t_pos.any():
        rnd=torch.randint(low=0,high=max(8,vocab_size),size=input_ids.shape,device=input_ids.device)
        same=t2t_pos & rnd.eq(input_ids)
        if same.any():
            rnd[same]=(rnd[same]+1)%max(8,vocab_size)
        mixed[t2t_pos]=rnd[t2t_pos]
    m2t_labels=torch.full_like(input_ids,-100)
    m2t_labels[m2t_pos]=_target[m2t_pos]
    t2t_labels=torch.full_like(input_ids,-100)
    t2t_labels[t2t_pos]=_target[t2t_pos]
    out=_forward(model,mixed,attention_mask,doc_ids,mask_mode=mask_mode,clean_ids=_target,composite_block_size=composite_block_size,trace=trace,cfg=cfg_obj,bidirectional=bidirectional)
    loss_m2t_raw=_causal_loss(out.logits,m2t_labels)
    if loss_weighting=="inv_t":
        t_clamp=max(float(t_sampled),0.001)
        loss_m2t=loss_m2t_raw/t_clamp
    else:
        loss_m2t=loss_m2t_raw
    loss_t2t=_causal_loss(out.logits,t2t_labels)
    loss=cfg.m2t_weight*loss_m2t+cfg.t2t_weight*loss_t2t
    with torch.no_grad():
        preds=out.logits.argmax(-1)
        shifted_preds=preds[:,:-1]
        shifted_labels=m2t_labels[:,1:]
        shifted_mask=shifted_labels.ne(-100)
        masked_token_acc=float((shifted_preds[shifted_mask]==shifted_labels[shifted_mask]).float().mean().item()) if shifted_mask.any() else None
        pred_positions_count=int(shifted_mask.sum().item())
        # MTF: build full-position predictions (left-shift: logit[i] predicts pos i+1)
        _mtf_preds=_target.clone()
        _mtf_preds[:,1:]=preds[:,:-1]
        _corrupted=m2t_pos|t2t_pos
    return {"loss":loss,"loss_m2t":loss_m2t_raw.detach(),"loss_m2t_scaled":loss_m2t.detach(),"loss_t2t":loss_t2t.detach(),"masked_token_acc":masked_token_acc,"t_sampled":t_sampled,"mask_ratio_actual":mask_ratio_actual,"pred_positions_count":pred_positions_count,"model_predictions":_mtf_preds,"corrupted_positions":_corrupted}

def apply_remask(tokens:torch.Tensor,confidence:torch.Tensor,mask_id:int,cfg:RemaskConfig)->torch.Tensor:
    x=tokens.clone()
    cand=(x!=mask_id)
    total=int(cand.sum().item())
    if total<=0:
        return x
    k=max(int(total*cfg.min_ratio),int(total*cfg.target_ratio))
    if k<=0:
        return x
    conf=confidence.clone()
    conf[~cand]=2.0
    flat=conf.view(-1)
    k=min(k,flat.numel())
    vals,idx=torch.topk(flat,k=k,largest=False)
    x.view(-1)[idx]=mask_id
    return x

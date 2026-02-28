# Diffusion schedule/objective utilities for SAFE training and decode.
# Main entrypoints: wsd_block,compute_m2t_t2t_losses,apply_remask.
# Implements WSD + M2T/T2T with doc-mask aware forward fallback.
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict,Tuple,Optional
import torch
import torch.nn.functional as F
from .config import WSDConfig,TrainConfig,RemaskConfig
from .masks import batch_doc_attention_mask
from .formulas import llada2_wsd_block
from .trace import use_trace,exception_with_stack

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

def _causal_loss(logits:torch.Tensor,labels:torch.Tensor)->torch.Tensor:
    if logits.shape[1]<2:
        return torch.tensor(0.0,device=logits.device)
    l=logits[:,:-1,:].contiguous()
    y=labels[:,1:].contiguous()
    if not torch.any(y.ne(-100)):
        return l.sum()*0.0
    return F.cross_entropy(l.view(-1,l.shape[-1]),y.view(-1),ignore_index=-100)

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

def _forward(model,input_ids:torch.Tensor,attn_1d:torch.Tensor,doc_ids:torch.Tensor,mask_mode:str,clean_ids:Optional[torch.Tensor]=None,composite_block_size:int|None=None,trace=None,cfg=None):
    tr=use_trace(cfg,trace)
    try:
        if (mask_mode or "").lower()=="composite_llada20":
            x0=clean_ids if clean_ids is not None else input_ids
            l=int(input_ids.shape[1])
            ids2=torch.cat([input_ids,x0],dim=1)
            docs2=torch.cat([doc_ids,doc_ids],dim=1)
            mask2=batch_doc_attention_mask(docs2,causal=False,mask_mode=mask_mode,block_size=composite_block_size,base_len=l)
            out=model(input_ids=ids2,attention_mask=_attn_for_model(mask2,model))
            logits=out.logits[:,:l,:]
            return SimpleNamespace(logits=logits)
        doc_mask=batch_doc_attention_mask(doc_ids,causal=True,mask_mode=mask_mode)
        return model(input_ids=input_ids,attention_mask=_attn_for_model(doc_mask,model))
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

def compute_m2t_t2t_losses(model,input_ids:torch.Tensor,attention_mask:torch.Tensor,doc_ids:torch.Tensor,response_mask:torch.Tensor|None,mask_id:int,vocab_size:int,cfg:TrainConfig,focus_response:bool,mask_mode:str="simple_blockdiag",composite_block_size:int|None=None,trace=None,cfg_obj=None)->Dict[str,torch.Tensor]:
    focus=response_mask if focus_response else None
    m2t_x,m2t_y=_make_m2t_batch(input_ids,attention_mask,focus,mask_id,cfg.mask_ratio)
    out_m2t=_forward(model,m2t_x,attention_mask,doc_ids,mask_mode=mask_mode,clean_ids=input_ids,composite_block_size=composite_block_size,trace=trace,cfg=cfg_obj)
    loss_m2t=_causal_loss(out_m2t.logits,m2t_y)
    t2t_losses=[]
    turns=max(1,int(cfg.multi_turn_t2t))
    for _ in range(turns):
        t2t_x,t2t_y=_make_t2t_batch(input_ids,attention_mask,focus,cfg.t2t_noise_ratio,vocab_size)
        out_t2t=_forward(model,t2t_x,attention_mask,doc_ids,mask_mode=mask_mode,clean_ids=input_ids,composite_block_size=composite_block_size,trace=trace,cfg=cfg_obj)
        t2t_losses.append(_causal_loss(out_t2t.logits,t2t_y))
    loss_t2t=torch.stack(t2t_losses).mean()
    loss=cfg.m2t_weight*loss_m2t+cfg.t2t_weight*loss_t2t
    with torch.no_grad():
        preds=out_m2t.logits.argmax(-1)
        mask_pos=m2t_y.ne(-100)
        masked_token_acc=float((preds[mask_pos]==m2t_y[mask_pos]).float().mean().item()) if mask_pos.any() else None
    return {"loss":loss,"loss_m2t":loss_m2t.detach(),"loss_t2t":loss_t2t.detach(),"masked_token_acc":masked_token_acc}

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

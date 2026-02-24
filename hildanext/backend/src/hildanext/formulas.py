# Formula checks for LLaDA, LLaDA2.0 and LLaDA2.1.
# Main entrypoints: llada_m2t_loss,llada2_wsd_block,llada21_sets.
# Utilities are deterministic and test-oriented.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,Tuple,List
import torch
import torch.nn.functional as F

@dataclass
class LLaDA21SetResult:
    gamma:torch.Tensor
    delta:torch.Tensor
    gamma_count:int
    delta_count:int

def _unique_sorted_pos(vals:List[int])->List[int]:
    out=sorted(set(max(1,int(x)) for x in vals))
    return out

def _align_divisor(block:int,seq_len:int)->int:
    b=max(1,int(block))
    s=max(1,int(seq_len))
    if s%b==0:
        return b
    for k in range(b,0,-1):
        if s%k==0:
            return k
    return 1

def _ladder_step(step:int,warmup_steps:int,ladder:List[int])->int:
    if not ladder:
        return 1
    if warmup_steps<=1:
        return int(ladder[-1])
    idx=min(len(ladder)-1,int(round((step/max(1,warmup_steps-1))*(len(ladder)-1))))
    return int(ladder[idx])

def llada_m2t_loss(logits:torch.Tensor,target_ids:torch.Tensor,masked_pos:torch.Tensor)->torch.Tensor:
    if logits.dim()!=3:
        raise ValueError("logits must be [B,S,V]")
    if target_ids.shape!=masked_pos.shape:
        raise ValueError("target_ids and masked_pos shape mismatch")
    labels=torch.full_like(target_ids,-100)
    labels[masked_pos.bool()]=target_ids[masked_pos.bool()]
    if logits.shape[1]<2:
        return logits.sum()*0.0
    l=logits[:,:-1,:].contiguous()
    y=labels[:,1:].contiguous()
    if not torch.any(y.ne(-100)):
        return l.sum()*0.0
    return F.cross_entropy(l.view(-1,l.shape[-1]),y.view(-1),ignore_index=-100)

def llada2_wsd_block(step:int,warmup_steps:int,stable_steps:int,decay_steps:int,start_block:int,max_block:int,end_block:int,seq_len:int|None=None,ladder_blocks:List[int]|None=None,decay_blocks:List[int]|None=None,enforce_divisibility:bool=False)->Tuple[str,int]:
    w=max(1,int(warmup_steps))
    s=max(0,int(stable_steps))
    d=max(1,int(decay_steps))
    ladder=_unique_sorted_pos(list(ladder_blocks or []))
    if ladder:
        if seq_len is not None and int(seq_len)>0 and (int(seq_len) not in ladder):
            ladder.append(int(seq_len))
            ladder=_unique_sorted_pos(ladder)
        stable_target=int(seq_len) if seq_len is not None and int(seq_len)>0 else int(ladder[-1])
    else:
        stable_target=max(1,int(max_block))
    if step<w:
        if ladder:
            b=_ladder_step(step,w,ladder)
        else:
            t=step/max(1,w-1)
            b=int(round(start_block+t*(max_block-start_block)))
        if enforce_divisibility and seq_len is not None and int(seq_len)>0:
            b=_align_divisor(b,int(seq_len))
        return "warmup",max(1,int(b))
    if step<w+s:
        b=max(1,int(stable_target))
        if enforce_divisibility and seq_len is not None and int(seq_len)>0:
            b=_align_divisor(b,int(seq_len))
        return "stable",b
    j=min(d-1,step-(w+s))
    dec=_unique_sorted_pos(list(decay_blocks or []))
    if dec:
        if int(stable_target) not in dec:
            dec.append(int(stable_target))
            dec=_unique_sorted_pos(dec)
        b=_ladder_step(j,d,sorted(dec,reverse=True))
    else:
        t=j/max(1,d-1)
        start_decay=max(1,int(stable_target))
        b=int(round(start_decay+t*(int(end_block)-start_decay)))
    if enforce_divisibility and seq_len is not None and int(seq_len)>0:
        b=_align_divisor(b,int(seq_len))
    return "decay",max(1,b)

def llada21_sets(tokens:torch.Tensor,pred_ids:torch.Tensor,confidence:torch.Tensor,mask_id:int,tau_mask:float,tau_edit:float)->LLaDA21SetResult:
    if tokens.shape!=pred_ids.shape or tokens.shape!=confidence.shape:
        raise ValueError("shape mismatch for llada21_sets")
    masked=tokens.eq(int(mask_id))
    gamma=masked & confidence.ge(float(tau_mask))
    delta=tokens.ne(int(mask_id)) & pred_ids.ne(tokens) & confidence.ge(float(tau_edit))
    return LLaDA21SetResult(gamma=gamma,delta=delta,gamma_count=int(gamma.sum().item()),delta_count=int(delta.sum().item()))

def llada21_apply(tokens:torch.Tensor,pred_ids:torch.Tensor,confidence:torch.Tensor,mask_id:int,tau_mask:float,tau_edit:float)->Tuple[torch.Tensor,LLaDA21SetResult]:
    out=tokens.clone()
    sets=llada21_sets(tokens,pred_ids,confidence,mask_id,tau_mask,tau_edit)
    if sets.gamma_count>0:
        out[sets.gamma]=pred_ids[sets.gamma]
    if sets.delta_count>0:
        out[sets.delta]=pred_ids[sets.delta]
    return out,sets

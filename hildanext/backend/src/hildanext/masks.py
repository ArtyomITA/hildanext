# Attention mask builders for packed multi-document sequences.
# Main entrypoints: doc_attention_mask,batch_doc_attention_mask.
# Masks prevent cross-document attention in packed training.
from __future__ import annotations
import torch

def doc_attention_mask(doc_ids:torch.Tensor,causal:bool=False)->torch.Tensor:
    if doc_ids.dim()!=1:
        raise ValueError("doc_ids must be 1D")
    valid=doc_ids.ge(0)
    same=doc_ids[:,None].eq(doc_ids[None,:])
    m=same & valid[:,None] & valid[None,:]
    if causal:
        tri=torch.tril(torch.ones_like(m,dtype=torch.bool))
        m=m & tri
    return m

def _composite_llada20_mask(doc_ids:torch.Tensor,base_len:int,block_size:int)->torch.Tensor:
    if doc_ids.dim()!=2:
        raise ValueError("doc_ids must be 2D")
    b,s=doc_ids.shape
    if s!=2*base_len:
        raise ValueError("composite_llada20 expects seq length 2*base_len")
    if block_size<=0:
        raise ValueError("block_size must be >0")
    idx=torch.arange(s,device=doc_ids.device).unsqueeze(0).expand(b,s)
    is_xt=idx<base_len
    is_x0=~is_xt
    local=idx.clone()
    local[is_x0]=local[is_x0]-base_len
    blk=torch.div(local,block_size,rounding_mode="floor")
    bi=blk[:,:,None]
    bj=blk[:,None,:]
    i_xt=is_xt[:,:,None]
    j_xt=is_xt[:,None,:]
    j_x0=is_x0[:,None,:]
    i_x0=is_x0[:,:,None]
    cond_xt_xt=i_xt & j_xt & bi.eq(bj)
    cond_xt_x0=i_xt & j_x0 & bi.gt(bj)
    cond_x0_x0=i_x0 & j_x0 & bi.ge(bj)
    base=cond_xt_xt | cond_xt_x0 | cond_x0_x0
    valid=doc_ids.ge(0)
    same=doc_ids[:,:,None].eq(doc_ids[:,None,:])
    gating=same & valid[:,:,None] & valid[:,None,:]
    return base & gating

def batch_doc_attention_mask(doc_ids:torch.Tensor,causal:bool=False,mask_mode:str="simple_blockdiag",block_size:int|None=None,base_len:int|None=None)->torch.Tensor:
    if doc_ids.dim()!=2:
        raise ValueError("doc_ids must be 2D")
    b,s=doc_ids.shape
    mode=(mask_mode or "simple_blockdiag").lower()
    if mode=="composite_llada20":
        l=base_len if base_len is not None else s//2
        bs=int(block_size if block_size is not None else max(1,l))
        return _composite_llada20_mask(doc_ids=doc_ids,base_len=int(l),block_size=bs)
    out=torch.zeros((b,s,s),dtype=torch.bool,device=doc_ids.device)
    if mode not in {"simple_blockdiag","composite_placeholder"}:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    for i in range(b):
        out[i]=doc_attention_mask(doc_ids[i],causal=causal)
    return out

def response_focus_mask(response_mask:torch.Tensor,base_mask:torch.Tensor)->torch.Tensor:
    if response_mask.shape!=base_mask.shape:
        raise ValueError("shape mismatch")
    return response_mask.bool() & base_mask.bool()

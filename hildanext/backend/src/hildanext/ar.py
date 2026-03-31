# AR baseline generation helper for side-by-side checks.
# Main entrypoints: generate_ar (loads model fresh), generate_ar_from_bundle (reuses pre-loaded bundle).
# Supports greedy or sampling decode with seed-stable behavior.
from __future__ import annotations
from typing import Any,Dict,Optional,Literal,Tuple
import time
import torch
from .config import AppConfig
from .inference import ModelBundle, load_model_bundle
from .utils import seed_everything,tokens_per_second
from .trace import use_trace

DecodeMode=Literal["greedy","sampling"]


def _apply_penalties(
    logits:torch.Tensor,
    generated:torch.Tensor,
    presence_penalty:float,
    repetition_penalty:float,
)->torch.Tensor:
    if generated.numel()==0:
        return logits
    out=logits.clone()
    unique_ids=torch.unique(generated.view(-1))
    if float(repetition_penalty)>1.0:
        vals=out[:,unique_ids]
        out[:,unique_ids]=torch.where(vals<0,vals*float(repetition_penalty),vals/float(repetition_penalty))
    if float(presence_penalty)!=0.0:
        out[:,unique_ids]=out[:,unique_ids]-float(presence_penalty)
    return out


def _apply_top_k_top_p(logits:torch.Tensor,top_k:int,top_p:float)->torch.Tensor:
    out=logits.clone()
    vocab=int(out.shape[-1])
    k=int(max(0,min(int(top_k),vocab)))
    if k>0 and k<vocab:
        topk_vals,_=torch.topk(out,k=k,dim=-1)
        kth=topk_vals[:,-1,None]
        out=torch.where(out<kth,torch.full_like(out,float("-inf")),out)
    p=float(max(0.0,min(1.0,top_p)))
    if 0.0<p<1.0:
        sorted_logits,sorted_idx=torch.sort(out,dim=-1,descending=True)
        sorted_probs=torch.softmax(sorted_logits,dim=-1)
        cdf=torch.cumsum(sorted_probs,dim=-1)
        remove=cdf>p
        remove[...,0]=False
        sorted_logits=torch.where(remove,torch.full_like(sorted_logits,float("-inf")),sorted_logits)
        restored=torch.full_like(out,float("-inf"))
        restored.scatter_(dim=-1,index=sorted_idx,src=sorted_logits)
        out=restored
    return out


def _sample_from_logits(
    logits:torch.Tensor,
    generated:torch.Tensor,
    decode_mode:DecodeMode,
    temperature:float,
    top_p:float,
    top_k:int,
    presence_penalty:float,
    repetition_penalty:float,
)->torch.Tensor:
    if decode_mode=="greedy":
        return torch.argmax(logits,dim=-1,keepdim=True)
    t=float(temperature)
    if t<=0.0:
        return torch.argmax(logits,dim=-1,keepdim=True)
    adj=_apply_penalties(
        logits=logits,
        generated=generated,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    adj=adj/float(t)
    adj=_apply_top_k_top_p(adj,top_k=top_k,top_p=top_p)
    probs=torch.softmax(adj,dim=-1)
    return torch.multinomial(probs,num_samples=1)


def _ar_decode_cached(
    model:Any,
    seq:torch.Tensor,
    prompt_len:int,
    max_new_tokens:int,
    decode_mode:DecodeMode,
    temperature:float,
    top_p:float,
    top_k:int,
    presence_penalty:float,
    repetition_penalty:float,
    eos_token_id:Optional[int],
)->Tuple[torch.Tensor,str]:
    prefix=seq[:,:prompt_len]
    if prefix.shape[1]==0:
        prefix=seq[:,:1]
    out=model(input_ids=prefix,use_cache=True)
    logits=out.logits[:,-1,:]
    past_kv=getattr(out,"past_key_values",None)
    generated:Optional[torch.Tensor]=None
    finish_reason="length"
    produced=0
    for i in range(int(max_new_tokens)):
        g=prompt_len+i
        gen_ids=generated if generated is not None else torch.empty((1,0),dtype=torch.long,device=seq.device)
        nxt=_sample_from_logits(
            logits=logits,
            generated=gen_ids,
            decode_mode=decode_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
        seq[:,g]=nxt.view(-1)
        generated=nxt if generated is None else torch.cat([generated,nxt],dim=1)
        produced=i+1
        tok_id=int(nxt.view(-1)[0].item())
        if eos_token_id is not None and tok_id==int(eos_token_id):
            finish_reason="eos"
            break
        if produced>=int(max_new_tokens):
            break
        if past_kv is None:
            out=model(input_ids=seq[:,:g+1],use_cache=True)
        else:
            out=model(input_ids=nxt,past_key_values=past_kv,use_cache=True)
        logits=out.logits[:,-1,:]
        past_kv=getattr(out,"past_key_values",None)
    return seq[:,prompt_len:prompt_len+produced],finish_reason


def _ar_decode_full_prefix(
    model:Any,
    seq:torch.Tensor,
    prompt_len:int,
    max_new_tokens:int,
    decode_mode:DecodeMode,
    temperature:float,
    top_p:float,
    top_k:int,
    presence_penalty:float,
    repetition_penalty:float,
    eos_token_id:Optional[int],
)->Tuple[torch.Tensor,str]:
    generated:Optional[torch.Tensor]=None
    finish_reason="length"
    produced=0
    for i in range(int(max_new_tokens)):
        g=prompt_len+i
        prefix=seq[:,:g]
        if prefix.shape[1]==0:
            prefix=seq[:,:1]
        logits=model(input_ids=prefix).logits[:,-1,:]
        gen_ids=generated if generated is not None else torch.empty((1,0),dtype=torch.long,device=seq.device)
        nxt=_sample_from_logits(
            logits=logits,
            generated=gen_ids,
            decode_mode=decode_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
        seq[:,g]=nxt.view(-1)
        generated=nxt if generated is None else torch.cat([generated,nxt],dim=1)
        produced=i+1
        tok_id=int(nxt.view(-1)[0].item())
        if eos_token_id is not None and tok_id==int(eos_token_id):
            finish_reason="eos"
            break
    return seq[:,prompt_len:prompt_len+produced],finish_reason


def _run_ar_decode(
    bundle:ModelBundle,
    prompt:str,
    max_new_tokens:int,
    seed:int,
    decode_mode:DecodeMode="greedy",
    temperature:float=0.7,
    top_p:float=0.9,
    top_k:int=20,
    presence_penalty:float=0.0,
    repetition_penalty:float=1.0,
)->Dict[str,Any]:
    model=bundle.model
    tok=bundle.tokenizer
    device=bundle.device
    seed_everything(seed)
    enc=tok([prompt],return_tensors="pt")
    input_ids=enc["input_ids"].to(device)
    prompt_len=int(input_ids.shape[1])
    seq=torch.full((1,prompt_len+int(max_new_tokens)),bundle.mask_id,dtype=torch.long,device=device)
    if prompt_len>0:
        seq[:,:prompt_len]=input_ids
    eos_token_id=getattr(tok,"eos_token_id",None)
    if eos_token_id is not None:
        try:
            eos_token_id=int(eos_token_id)
        except Exception:
            eos_token_id=None
    if device.type=="cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device.index or 0)
        except Exception:
            pass
    t0=time.time()
    model.eval()
    with torch.inference_mode():
        try:
            new_ids,finish_reason=_ar_decode_cached(
                model=model,
                seq=seq,
                prompt_len=prompt_len,
                max_new_tokens=int(max_new_tokens),
                decode_mode=decode_mode,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
            )
        except Exception:
            new_ids,finish_reason=_ar_decode_full_prefix(
                model=model,
                seq=seq,
                prompt_len=prompt_len,
                max_new_tokens=int(max_new_tokens),
                decode_mode=decode_mode,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
            )
    text=tok.decode(new_ids[0],skip_special_tokens=True) if hasattr(tok,"decode") else " ".join(str(int(x)) for x in new_ids[0].tolist())
    if not text.strip():
        raw=[int(x) for x in new_ids[0].detach().cpu().tolist() if int(x)!=bundle.mask_id]
        text=" ".join(f"tok{x}" for x in raw[:64]) if raw else ("dummy-ar-output" if bundle.is_dummy else "")
    if bundle.is_dummy and not text.startswith("[DUMMY] "):
        text=f"[DUMMY] {text}"
    elapsed=max(1e-6,time.time()-t0)
    vram_peak=None
    if device.type=="cuda":
        try:
            vram_peak=float(torch.cuda.max_memory_allocated(device.index or 0))
        except Exception:
            pass
    generated_n=int(new_ids.shape[1])
    truncated=finish_reason=="length" and generated_n>=int(max_new_tokens)
    return {
        "text":text.strip(),
        "engine":"ar-greedy" if decode_mode=="greedy" else "ar-sampling",
        "dummy_model":bundle.is_dummy,
        "load_reason":bundle.load_reason,
        "actual_dtype":bundle.actual_dtype,
        "device":bundle.device.type,
        "tokens_generated":generated_n,
        "tokens_per_sec":tokens_per_second(generated_n,elapsed),
        "vram_peak_bytes":vram_peak,
        "finish_reason":finish_reason,
        "truncated":bool(truncated),
    }


def generate_ar_from_bundle(
    bundle:ModelBundle,
    prompt:str,
    max_new_tokens:int=64,
    seed:int=42,
    decode_mode:DecodeMode="greedy",
    temperature:float=0.7,
    top_p:float=0.9,
    top_k:int=20,
    presence_penalty:float=0.0,
    repetition_penalty:float=1.0,
)->Dict[str,Any]:
    """AR generation reusing an already-loaded ModelBundle (fast path, no disk I/O)."""
    mode="sampling" if str(decode_mode).lower()=="sampling" else "greedy"
    return _run_ar_decode(
        bundle=bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        decode_mode=mode,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )


def generate_ar(
    cfg:AppConfig,
    prompt:str,
    max_new_tokens:int=64,
    seed:Optional[int]=None,
    trace=None,
    decode_mode:DecodeMode="greedy",
    temperature:float=0.7,
    top_p:float=0.9,
    top_k:int=20,
    presence_penalty:float=0.0,
    repetition_penalty:float=1.0,
)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    bundle=load_model_bundle(cfg,for_training=False,trace=tr)
    s=cfg.runtime.seed if seed is None else int(seed)
    result=generate_ar_from_bundle(
        bundle=bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=s,
        decode_mode=decode_mode,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    result["fallbacks"]=tr.snapshot_fallbacks(limit=32) if tr is not None else []
    # Explicitly free model to reclaim VRAM before caller loads another model.
    del bundle
    torch.cuda.empty_cache()
    return result

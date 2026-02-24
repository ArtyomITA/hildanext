# Quantization helpers and VRAM/timing benchmark for SAFE backend.
# Main entrypoints: load_quantized,run_quant_bench.
# Handles unavailable modes with non-crashing structured reasons.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Dict,List,Tuple
import json
import math
import time
import torch
import torch.nn.functional as F
from .config import AppConfig,clone_with_updates
from .inference import build_engine,mode_thresholds
from .tokenization import load_tokenizer,ensure_mask_token
from .utils import choose_device,seed_everything,tokens_per_second,mem_stats,TinyCausalLM
from .trace import use_trace,exception_with_stack

@dataclass
class QuantLoadResult:
    model:Any|None
    tokenizer:Any
    info:Dict[str,Any]

def _safe_vocab(tokenizer)->int:
    try:
        return int(len(tokenizer))
    except Exception:
        return int(getattr(tokenizer,"vocab_size",32768) or 32768)

def _can_use_bnb(device:torch.device)->Tuple[bool,str]:
    if device.type!="cuda":
        return False,"cuda required for bitsandbytes modes"
    try:
        import bitsandbytes as _  # noqa: F401
        return True,""
    except Exception as e:
        return False,f"bitsandbytes unavailable: {e}"

def _fallback_tiny(tokenizer,device:torch.device,reason:str,mode:str,trace=None)->QuantLoadResult:
    vocab=max(32768,_safe_vocab(tokenizer))
    model=TinyCausalLM(vocab_size=vocab,hidden_size=256).to(device)
    mask_id=ensure_mask_token(tokenizer,"<|mask|>",model=model)
    info={
        "ok":False,
        "available":False,
        "reason":reason,
        "mode":mode,
        "device":str(device),
        "dtype":"float32",
        "mask_id":mask_id,
        "vocab_size":max(vocab,_safe_vocab(tokenizer)),
        "fallback_dummy":True
    }
    if trace is not None:
        trace.record_fallback(
            event="fallback",
            module="quant",
            func="_fallback_tiny",
            action="tiny_model_fallback",
            reason=str(reason),
            extra_dict={"mode":mode,"device":str(device)}
        )
    return QuantLoadResult(model=model,tokenizer=tokenizer,info=info)

def load_quantized(model_dir:str,mode:str,device:str="auto",trust_remote_code:bool=True,trace=None,cfg=None)->Tuple[Any|None,Any,Dict[str,Any]]:
    tr=use_trace(cfg,trace)
    dev=choose_device(device)
    m=(mode or "fp32").lower()
    tok=load_tokenizer(model_dir,trust_remote_code=trust_remote_code,trace=tr,cfg=cfg)
    model=None
    info={"ok":False,"available":False,"reason":"","mode":m,"device":str(dev),"dtype":"float32","mask_id":-1,"vocab_size":_safe_vocab(tok),"fallback_dummy":False}
    try:
        from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    except Exception:
        AutoModelForCausalLM=None
        BitsAndBytesConfig=None
    if AutoModelForCausalLM is None:
        q=_fallback_tiny(tok,dev,"transformers_unavailable",m,trace=tr)
        return q.model,q.tokenizer,q.info
    try:
        kwargs={"trust_remote_code":trust_remote_code}
        if m in {"fp32","float32"}:
            kwargs["dtype"]=torch.float32
            info["dtype"]="float32"
        elif m in {"fp16","float16"}:
            if dev.type!="cuda":
                q=_fallback_tiny(tok,dev,"fp16_requested_without_cuda",m,trace=tr)
                return q.model,q.tokenizer,q.info
            kwargs["dtype"]=torch.float16
            info["dtype"]="float16"
        elif m in {"bf16","bfloat16"}:
            if dev.type!="cuda":
                kwargs["dtype"]=torch.float32
                info["dtype"]="float32"
                info["reason"]="bf16 emulated as fp32 on cpu"
            else:
                kwargs["dtype"]=torch.bfloat16
                info["dtype"]="bfloat16"
        elif m in {"int8","nf4"}:
            ok,why=_can_use_bnb(dev)
            if not ok:
                q=_fallback_tiny(tok,dev,why,m,trace=tr)
                return q.model,q.tokenizer,q.info
            if BitsAndBytesConfig is None:
                q=_fallback_tiny(tok,dev,"BitsAndBytesConfig_unavailable",m,trace=tr)
                return q.model,q.tokenizer,q.info
            if m=="int8":
                kwargs["quantization_config"]=BitsAndBytesConfig(load_in_8bit=True)
                info["dtype"]="int8"
            else:
                kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16 if dev.type=="cuda" else torch.float32)
                info["dtype"]="nf4"
        else:
            q=_fallback_tiny(tok,dev,f"unsupported_mode:{mode}",m,trace=tr)
            return q.model,q.tokenizer,q.info
        model=AutoModelForCausalLM.from_pretrained(model_dir,**kwargs)
        if m not in {"int8","nf4"}:
            model=model.to(dev)
        mask_id=ensure_mask_token(tok,"<|mask|>",model=model)
        vocab_size=max(_safe_vocab(tok),int(getattr(getattr(model,"lm_head",None),"out_features",0) or 0))
        info.update({"ok":True,"available":True,"reason":info["reason"],"mask_id":mask_id,"vocab_size":vocab_size})
        return model,tok,info
    except Exception as e:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="quant",
                func="load_quantized",
                action="load_failure",
                reason="quantized_load_failed",
                exception_str=exception_with_stack(e),
                extra_dict={"mode":m,"model_dir":model_dir}
            )
        q=_fallback_tiny(tok,dev,f"load_failure:{e}",m,trace=tr)
        return q.model,q.tokenizer,q.info

def _decode_tokens(tokenizer,ids:torch.Tensor,mask_id:int)->str:
    try:
        txt=tokenizer.decode(ids,skip_special_tokens=True)
    except Exception:
        raw=[int(x) for x in ids.detach().cpu().tolist() if int(x)!=mask_id]
        txt=" ".join(f"tok{x}" for x in raw[:32])
    return txt.strip() if txt and txt.strip() else "dummy-output"

def _run_ar_once(model,tokenizer,prompt:str,max_new_tokens:int,device:torch.device,seed:int)->Tuple[str,float]:
    seed_everything(seed)
    enc=tokenizer([prompt],return_tensors="pt")
    ids=enc["input_ids"].to(device)
    prompt_len=int(ids.shape[1])
    t0=time.time()
    model.eval()
    seq=ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits=model(input_ids=seq).logits[:,-1,:]
            nxt=torch.argmax(logits,dim=-1,keepdim=True)
            seq=torch.cat([seq,nxt],dim=1)
        new_ids=seq[0,prompt_len:]
    elapsed=max(1e-6,time.time()-t0)
    return _decode_tokens(tokenizer,new_ids,mask_id=-1),tokens_per_second(int(new_ids.numel()),elapsed)

def _run_dllm_once(model,tokenizer,prompt:str,max_new_tokens:int,device:torch.device,mask_id:int,tau_mask:float,tau_edit:float,steps:int,seed:int)->Tuple[str,float]:
    seed_everything(seed)
    enc=tokenizer([prompt],return_tensors="pt")
    input_ids=enc["input_ids"].to(device)
    prompt_len=int(input_ids.shape[1])
    seq=torch.full((1,prompt_len+max_new_tokens),int(mask_id),dtype=torch.long,device=device)
    seq[:,:prompt_len]=input_ids
    t0=time.time()
    model.eval()
    with torch.no_grad():
        for _ in range(max(1,steps)):
            out=model(input_ids=seq)
            logits=out.logits
            pos=torch.arange(prompt_len,prompt_len+max_new_tokens,device=device)
            src=torch.clamp(pos-1,min=0)
            pred_logits=logits[:,src,:]
            probs=torch.softmax(pred_logits,dim=-1)
            conf,pred=torch.max(probs,dim=-1)
            gen=seq[:,prompt_len:]
            gamma=gen.eq(mask_id) & conf.ge(float(tau_mask))
            if gamma.any():
                gen[gamma]=pred[gamma]
            delta=gen.ne(mask_id) & pred.ne(gen) & conf.ge(float(tau_edit))
            if delta.any():
                gen[delta]=pred[delta]
            seq[:,prompt_len:]=gen
            if not gen.eq(mask_id).any():
                break
    elapsed=max(1e-6,time.time()-t0)
    out_ids=seq[0,prompt_len:]
    return _decode_tokens(tokenizer,out_ids,mask_id),tokens_per_second(int(max_new_tokens),elapsed)

def _run_train_probe(model,tokenizer,prompt:str,device:torch.device,seq_len:int=64)->Dict[str,Any]:
    t0=time.time()
    row={"ok":False,"loss":None,"elapsed":None,"peak_mem":None,"reason":""}
    try:
        model.train()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device.index or 0)
        enc=tokenizer([prompt],return_tensors="pt")
        ids=enc["input_ids"].to(device)
        if ids.shape[1]<seq_len:
            pad=torch.full((1,seq_len-ids.shape[1]),int(getattr(tokenizer,"pad_token_id",0) or 0),dtype=torch.long,device=device)
            ids=torch.cat([ids,pad],dim=1)
        else:
            ids=ids[:,:seq_len]
        out=model(input_ids=ids)
        logits=out.logits
        if logits.shape[1]<2:
            raise RuntimeError("sequence too short for train probe")
        y=ids
        loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),y[:,1:].reshape(-1))
        loss.backward()
        row["ok"]=True
        row["loss"]=float(loss.detach().item())
    except Exception as e:
        row["reason"]=str(e)
    row["elapsed"]=float(max(1e-6,time.time()-t0))
    if device.type=="cuda":
        try:
            row["peak_mem"]=float(torch.cuda.max_memory_allocated(device.index or 0))
        except Exception:
            row["peak_mem"]=None
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    return row

def _finite_or_none(x:Any)->float|None:
    if x is None:
        return None
    try:
        y=float(x)
        if math.isfinite(y):
            return y
        return None
    except Exception:
        return None

def run_quant_bench(cfg:AppConfig,modes:List[str],prompt:str,max_new_tokens:int,engine_name:str="transformers",seed:int|None=None,out_json:str|Path|None=None,train_probe:bool=False,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    reports=[]
    req_engine=(engine_name or "transformers").lower()
    s=cfg.runtime.seed if seed is None else int(seed)
    for i,mode in enumerate(modes):
        dev=choose_device(cfg.runtime.device)
        if dev.type=="cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev.index or 0)
            except Exception:
                pass
        model,tokenizer,info=load_quantized(cfg.paths.model_dir,mode,device=cfg.runtime.device,trust_remote_code=cfg.model.trust_remote_code,trace=tr,cfg=cfg)
        row={
            "mode":mode,
            "ok":bool(info.get("ok",False)),
            "available":bool(info.get("available",False)),
            "reason":str(info.get("reason","")),
            "device":str(info.get("device",str(dev))),
            "dtype":str(info.get("dtype","")),
            "engine_requested":req_engine,
            "engine_used":"transformers-local",
            "elapsed":None,
            "tokens_per_sec":{"ar":None,"dllm":None},
            "peak_mem":None,
            "mem":mem_stats(dev),
            "train_probe":None
        }
        t0=time.time()
        if model is not None and bool(info.get("available",False)):
            try:
                mask_id=int(info.get("mask_id",-1))
                if mask_id<0:
                    mask_id=ensure_mask_token(tokenizer,cfg.model.mask_token,model=model)
                ar_text,ar_tps=_run_ar_once(model,tokenizer,prompt,max_new_tokens,dev,s+i)
                tau_m,tau_e=mode_thresholds(cfg,"Q_MODE",None,None)
                dllm_tps=None
                if req_engine=="dinfer":
                    try:
                        c2=clone_with_updates(cfg,{"runtime":{"use_dinfer":True}})
                        eng=build_engine(c2,trace=tr)
                        _=eng.generate(prompt=prompt,mode="Q_MODE",max_new_tokens=max_new_tokens,seed=s+i)
                        dllm_tps=float(getattr(eng,"last_stats",{}).get("tokens_per_sec",0.0))
                        row["engine_used"]=eng.name
                        eng.close()
                    except Exception as e:
                        row["engine_used"]="transformers-local-fallback"
                        row["reason"]=(row["reason"]+"; " if row["reason"] else "")+f"dinfer fallback: {e}"
                        if tr is not None:
                            tr.record_fallback(
                                event="fallback",
                                module="quant",
                                func="run_quant_bench",
                                action="engine_fallback",
                                reason="dinfer_missing",
                                exception_str=exception_with_stack(e),
                                extra_dict={"engine_requested":req_engine}
                            )
                        _,dllm_tps=_run_dllm_once(model,tokenizer,prompt,max_new_tokens,dev,mask_id,tau_m,tau_e,cfg.inference.max_steps,s+i)
                else:
                    _,dllm_tps=_run_dllm_once(model,tokenizer,prompt,max_new_tokens,dev,mask_id,tau_m,tau_e,cfg.inference.max_steps,s+i)
                row["ok"]=bool(info.get("ok",False))
                row["tokens_per_sec"]={"ar":_finite_or_none(ar_tps),"dllm":_finite_or_none(dllm_tps)}
                row["sample_ar_non_empty"]=bool(ar_text.strip())
                if train_probe:
                    row["train_probe"]=_run_train_probe(model,tokenizer,prompt,dev,seq_len=min(64,max_new_tokens+16))
            except Exception as e:
                row["ok"]=False
                row["reason"]=(row["reason"]+"; " if row["reason"] else "")+f"bench error: {e}"
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="quant",
                        func="run_quant_bench",
                        action="bench_error",
                        reason="quant_bench_error",
                        exception_str=exception_with_stack(e),
                        extra_dict={"mode":mode}
                    )
        elif model is not None and not bool(info.get("available",False)):
            row["engine_used"]="unavailable"
        row["elapsed"]=_finite_or_none(max(1e-6,time.time()-t0))
        if dev.type=="cuda":
            try:
                row["peak_mem"]=float(torch.cuda.max_memory_allocated(dev.index or 0))
                row["mem"]=mem_stats(dev)
            except Exception:
                row["peak_mem"]=None
        reports.append(row)
    payload={
        "model_dir":cfg.paths.model_dir,
        "prompt":prompt,
        "max_new_tokens":int(max_new_tokens),
        "engine":req_engine,
        "results":reports,
        "fallbacks":tr.snapshot_fallbacks(limit=128) if tr is not None else []
    }
    out=Path(out_json) if out_json else Path(cfg.paths.root)/"runs"/"reports"/"quant_vram.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=True,indent=2),encoding="utf-8")
    return payload

# AR baseline generation helper for side-by-side checks.
# Main entrypoints: generate_ar (loads model fresh), generate_ar_from_bundle (reuses pre-loaded bundle).
# Uses explicit greedy decoding to keep behavior deterministic.
from __future__ import annotations
from typing import Any,Dict,Optional
import time
import torch
from .config import AppConfig
from .inference import ModelBundle, load_model_bundle
from .utils import seed_everything,tokens_per_second
from .trace import use_trace

def _run_ar_greedy(bundle:ModelBundle,prompt:str,max_new_tokens:int,seed:int)->Dict[str,Any]:
    """Pure greedy AR decode using a pre-loaded ModelBundle. Does NOT free the bundle."""
    model=bundle.model
    tok=bundle.tokenizer
    device=bundle.device
    seed_everything(seed)
    enc=tok([prompt],return_tensors="pt")
    input_ids=enc["input_ids"].to(device)
    prompt_len=int(input_ids.shape[1])
    t0=time.time()
    model.eval()
    seq=input_ids
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            logits=model(input_ids=seq).logits[:,-1,:]
            nxt=torch.argmax(logits,dim=-1,keepdim=True)
            seq=torch.cat([seq,nxt],dim=1)
    new_ids=seq[0,prompt_len:]
    text=tok.decode(new_ids,skip_special_tokens=True) if hasattr(tok,"decode") else " ".join(str(int(x)) for x in new_ids.tolist())
    if not text.strip():
        raw=[int(x) for x in new_ids.detach().cpu().tolist() if int(x)!=bundle.mask_id]
        text=" ".join(f"tok{x}" for x in raw[:64]) if raw else ("dummy-ar-output" if bundle.is_dummy else "")
    if bundle.is_dummy and not text.startswith("[DUMMY] "):
        text=f"[DUMMY] {text}"
    elapsed=max(1e-6,time.time()-t0)
    return {
        "text":text.strip(),
        "engine":"ar-greedy",
        "dummy_model":bundle.is_dummy,
        "load_reason":bundle.load_reason,
        "actual_dtype":bundle.actual_dtype,
        "tokens_generated":int(new_ids.numel()),
        "tokens_per_sec":tokens_per_second(int(new_ids.numel()),elapsed)
    }

def generate_ar_from_bundle(bundle:ModelBundle,prompt:str,max_new_tokens:int=64,seed:int=42)->Dict[str,Any]:
    """AR generation reusing an already-loaded ModelBundle (fast path, no disk I/O)."""
    return _run_ar_greedy(bundle,prompt,max_new_tokens,seed)

def generate_ar(cfg:AppConfig,prompt:str,max_new_tokens:int=64,seed:Optional[int]=None,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    bundle=load_model_bundle(cfg,for_training=False,trace=tr)
    s=cfg.runtime.seed if seed is None else int(seed)
    result=_run_ar_greedy(bundle,prompt,max_new_tokens,s)
    result["fallbacks"]=tr.snapshot_fallbacks(limit=32) if tr is not None else []
    # Explicitly free model to reclaim VRAM before caller loads another model.
    del bundle
    torch.cuda.empty_cache()
    return result

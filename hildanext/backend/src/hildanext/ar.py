# AR baseline generation helper for side-by-side checks.
# Main entrypoint: generate_ar.
# Uses explicit greedy decoding to keep behavior deterministic.
from __future__ import annotations
from typing import Any,Dict,Optional
import time
import torch
from .config import AppConfig
from .inference import load_model_bundle
from .utils import seed_everything,tokens_per_second
from .trace import use_trace

def generate_ar(cfg:AppConfig,prompt:str,max_new_tokens:int=64,seed:Optional[int]=None,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    bundle=load_model_bundle(cfg,for_training=False,trace=tr)
    model=bundle.model
    tok=bundle.tokenizer
    device=bundle.device
    s=cfg.runtime.seed if seed is None else int(seed)
    seed_everything(s)
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
        "fallbacks":tr.snapshot_fallbacks(limit=32) if tr is not None else [],
        "tokens_generated":int(new_ids.numel()),
        "tokens_per_sec":tokens_per_second(int(new_ids.numel()),elapsed)
    }

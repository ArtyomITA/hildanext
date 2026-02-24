# Runtime utilities and minimal fallback model/tokenizer.
# Main entrypoints: seed_everything,choose_device,force_math_sdpa.
# TinyCausalLM and SimpleTokenizer keep smoke flow non-blocking.
from __future__ import annotations
from types import SimpleNamespace
from typing import Dict,List,Iterable,Tuple
import math
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def force_math_sdpa()->None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

def choose_device(device_hint:str="auto")->torch.device:
    if device_hint=="cpu":
        return torch.device("cpu")
    if device_hint=="cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dtype_from_name(name:str,device:torch.device)->torch.dtype:
    n=(name or "float32").lower()
    if n in {"bf16","bfloat16"}:
        if device.type!="cuda":
            return torch.float32
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if n in {"fp16","float16","half"}:
        return torch.float16 if device.type=="cuda" else torch.float32
    return torch.float32

def env_issues()->Dict[str,str]:
    out:Dict[str,str]={}
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _=torch.tensor([1.0]).cpu().numpy()
            for w in ws:
                msg=str(getattr(w,"message",""))
                if "Failed to initialize NumPy" in msg or "_multiarray_umath" in msg:
                    out["numpy_dll"]=msg
    except Exception as e:
        out["numpy_dll"]=str(e)
    return out

def mem_stats(device:torch.device)->Dict[str,float]:
    if device.type!="cuda":
        return {"alloc_gb":0.0,"reserved_gb":0.0}
    alloc=float(torch.cuda.memory_allocated(device.index or 0))/1024**3
    reserv=float(torch.cuda.memory_reserved(device.index or 0))/1024**3
    return {"alloc_gb":round(alloc,4),"reserved_gb":round(reserv,4)}

def tokens_per_second(token_count:int,elapsed:float)->float:
    if elapsed<=0:
        return 0.0
    return float(token_count)/float(elapsed)

def chunked(seq:List[int],size:int)->Iterable[List[int]]:
    for i in range(0,len(seq),size):
        yield seq[i:i+size]

class SimpleTokenizer:
    def __init__(self,vocab_size:int=32768):
        self.vocab_size=vocab_size
        self.pad_token="<pad>"
        self.eos_token="<eos>"
        self.unk_token="<unk>"
        self.mask_token="<mask>"
        self.pad_token_id=0
        self.eos_token_id=1
        self.unk_token_id=2
        self.mask_token_id=3
        self._special={self.pad_token:0,self.eos_token:1,self.unk_token:2,self.mask_token:3}
    def __len__(self)->int:
        return self.vocab_size
    def get_vocab(self)->Dict[str,int]:
        return dict(self._special)
    def convert_tokens_to_ids(self,token:str)->int:
        if token in self._special:
            return self._special[token]
        return 4+(abs(hash(token))%(self.vocab_size-4))
    def add_special_tokens(self,payload:Dict[str,List[str]|str])->int:
        added=0
        vals:List[str]=[]
        for v in payload.values():
            if isinstance(v,list):
                vals.extend(v)
            else:
                vals.append(v)
        for tok in vals:
            if tok not in self._special:
                self._special[tok]=len(self._special)
                added+=1
        self.vocab_size=max(self.vocab_size,len(self._special)+4)
        if self.mask_token in self._special:
            self.mask_token_id=self._special[self.mask_token]
        return added
    def encode(self,text:str,add_special_tokens:bool=False)->List[int]:
        toks=[self.convert_tokens_to_ids(t) for t in text.strip().split()]
        if add_special_tokens:
            toks=toks+[self.eos_token_id]
        return toks
    def __call__(self,texts:List[str]|str,return_tensors:str|None=None)->Dict[str,torch.Tensor|List[List[int]]]:
        if isinstance(texts,str):
            texts=[texts]
        ids=[self.encode(t,add_special_tokens=False) for t in texts]
        if return_tensors=="pt":
            max_len=max(len(x) for x in ids) if ids else 1
            arr=torch.full((len(ids),max_len),self.pad_token_id,dtype=torch.long)
            for i,row in enumerate(ids):
                arr[i,:len(row)]=torch.tensor(row,dtype=torch.long)
            return {"input_ids":arr}
        return {"input_ids":ids}
    def decode(self,ids:List[int]|torch.Tensor,skip_special_tokens:bool=True)->str:
        if isinstance(ids,torch.Tensor):
            ids=ids.detach().cpu().tolist()
        out=[]
        for i in ids:
            if skip_special_tokens and i in self._special.values():
                continue
            out.append(f"tok{i}")
        return " ".join(out).strip()

class TinyCausalLM(nn.Module):
    def __init__(self,vocab_size:int=32768,hidden_size:int=256):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,hidden_size)
        self.lm_head=nn.Linear(hidden_size,vocab_size,bias=False)
    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor|None=None):
        h=self.embed(input_ids)
        logits=self.lm_head(h)
        return SimpleNamespace(logits=logits)

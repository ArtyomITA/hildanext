# Data ingestion for Dolma sample and TinyStories.
# Main entrypoint: prepare_data.
# Supports local files, optional HF download, and synthetic fallback.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Tuple
import json
import random
import numpy as np
from .config import AppConfig
from .io_utils import ensure_dir,write_jsonl,write_json,now_iso
from .trace import use_trace

TEXT_KEYS=("text","content","body","document","story")

def _pick_text(obj:Dict[str,Any])->str:
    for k in TEXT_KEYS:
        v=obj.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def _records_from_jsonl(path:Path,source:str,max_samples:int)->List[Dict[str,Any]]:
    out:List[Dict[str,Any]]=[]
    with path.open("r",encoding="utf-8",errors="ignore") as f:
        for i,line in enumerate(f):
            if max_samples and len(out)>=max_samples:
                break
            if not line.strip():
                continue
            obj=json.loads(line)
            text=_pick_text(obj)
            if not text:
                continue
            out.append({"id":f"{source}-{i}","source":source,"text":text})
    return out

def _records_from_json(path:Path,source:str,max_samples:int)->List[Dict[str,Any]]:
    raw=json.loads(path.read_text(encoding="utf-8",errors="ignore"))
    out:List[Dict[str,Any]]=[]
    if isinstance(raw,dict):
        raw=[raw]
    if not isinstance(raw,list):
        return out
    for i,obj in enumerate(raw):
        if max_samples and len(out)>=max_samples:
            break
        if not isinstance(obj,dict):
            continue
        text=_pick_text(obj)
        if not text:
            continue
        out.append({"id":f"{source}-{i}","source":source,"text":text})
    return out

def _records_from_txt(path:Path,source:str,max_samples:int)->List[Dict[str,Any]]:
    out:List[Dict[str,Any]]=[]
    for i,line in enumerate(path.read_text(encoding="utf-8",errors="ignore").splitlines()):
        if max_samples and len(out)>=max_samples:
            break
        t=line.strip()
        if not t:
            continue
        out.append({"id":f"{source}-{i}","source":source,"text":t})
    return out

def _records_from_pretokenized_dir(path:Path,source:str,max_samples:int)->List[Dict[str,Any]]:
    out:List[Dict[str,Any]]=[]
    docs_files=sorted(path.glob("*_docs.json"))
    for docs_file in docs_files:
        if max_samples and len(out)>=max_samples:
            break
        npy_file=docs_file.with_name(docs_file.name.replace("_docs.json",".npy"))
        if not npy_file.exists():
            continue
        try:
            docs=json.loads(docs_file.read_text(encoding="utf-8"))
            arr=np.load(npy_file,mmap_mode="r")
        except Exception:
            continue
        pos=0
        for i,d in enumerate(docs):
            if max_samples and len(out)>=max_samples:
                break
            ln=int(d.get("len",0))
            if ln<=0 or pos+ln>arr.shape[0]:
                break
            toks=arr[pos:pos+ln].astype(np.int64).tolist()
            pos+=ln
            out.append({"id":d.get("doc_id",f"{docs_file.stem}-{i}"),"source":source,"token_ids":toks,"text":""})
    return out

def _records_from_packed_dir(path:Path,source:str,max_samples:int)->List[Dict[str,Any]]:
    out:List[Dict[str,Any]]=[]
    p=path/"packed" if (path/"packed").exists() else path
    meta={}
    mp=p/"meta.json"
    if mp.exists():
        try:
            meta=json.loads(mp.read_text(encoding="utf-8",errors="ignore"))
        except Exception:
            meta={}
    seq_len=int(meta.get("seq_len",1024) or 1024)
    for fp in sorted(p.glob("packed_*.npy")):
        if max_samples and len(out)>=max_samples:
            break
        try:
            arr=np.load(fp,mmap_mode="r")
        except Exception:
            continue
        if arr.ndim==2:
            rows=min(arr.shape[0],max_samples-len(out) if max_samples else arr.shape[0])
            for i in range(rows):
                toks=arr[i].astype(np.int64).tolist()
                out.append({"id":f"{fp.stem}-{i}","source":source,"token_ids":toks,"text":""})
            continue
        flat=arr.reshape(-1)
        total=min(flat.shape[0]//max(1,seq_len),max_samples-len(out) if max_samples else flat.shape[0]//max(1,seq_len))
        for i in range(total):
            s=i*seq_len
            e=s+seq_len
            toks=flat[s:e].astype(np.int64).tolist()
            out.append({"id":f"{fp.stem}-{i}","source":source,"token_ids":toks,"text":""})
    return out

def load_local_records(path_like:str,source:str,max_samples:int)->List[Dict[str,Any]]:
    if not path_like:
        return []
    p=Path(path_like)
    if not p.exists():
        return []
    if p.is_file():
        if p.suffix.lower()==".jsonl":
            return _records_from_jsonl(p,source,max_samples)
        if p.suffix.lower()==".json":
            return _records_from_json(p,source,max_samples)
        if p.suffix.lower() in {".txt",".md"}:
            return _records_from_txt(p,source,max_samples)
        return []
    tok_dir=p/"tokenized"
    if tok_dir.exists():
        recs=_records_from_pretokenized_dir(tok_dir,source,max_samples)
        if recs:
            return recs
    packed_recs=_records_from_packed_dir(p,source,max_samples)
    if packed_recs:
        return packed_recs
    files=[x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".jsonl",".json",".txt",".md"}]
    out:List[Dict[str,Any]]=[]
    for fp in files:
        if max_samples and len(out)>=max_samples:
            break
        rem=max_samples-len(out) if max_samples else 0
        if fp.suffix.lower()==".jsonl":
            out.extend(_records_from_jsonl(fp,source,rem))
        elif fp.suffix.lower()==".json":
            out.extend(_records_from_json(fp,source,rem))
        else:
            out.extend(_records_from_txt(fp,source,rem))
    return out[:max_samples] if max_samples else out

def _download_tinystories(max_samples:int)->List[Dict[str,Any]]:
    try:
        from datasets import load_dataset
    except Exception:
        return []
    try:
        ds=load_dataset("roneneldan/TinyStories",split=f"train[:{max_samples}]")
    except Exception:
        return []
    out=[]
    for i,row in enumerate(ds):
        text=str(row.get("text","")).strip()
        if text:
            out.append({"id":f"tinystories-{i}","source":"tinystories","text":text})
    return out

def _download_dolma(max_samples:int)->List[Dict[str,Any]]:
    try:
        from datasets import load_dataset
    except Exception:
        return []
    tries=["allenai/dolma","allenai/dolma-v1_6"]
    for name in tries:
        try:
            ds=load_dataset(name,split=f"train[:{max_samples}]")
            out=[]
            for i,row in enumerate(ds):
                txt=_pick_text(row if isinstance(row,dict) else {})
                if txt:
                    out.append({"id":f"dolma-{i}","source":"dolma","text":txt})
            if out:
                return out
        except Exception:
            continue
    return []

def _synthetic_records(prefix:str,n:int)->List[Dict[str,Any]]:
    out=[]
    for i in range(n):
        text=f"{prefix} sample {i} keeps stable SAFE training with masked denoising and token editing."
        out.append({"id":f"{prefix}-{i}","source":prefix,"text":text})
    return out

def _to_sft_pairs(records:List[Dict[str,Any]],max_samples:int)->List[Dict[str,Any]]:
    out=[]
    for i,r in enumerate(records):
        if max_samples and len(out)>=max_samples:
            break
        text=r.get("text","")
        words=text.split()
        if len(words)<8:
            continue
        pivot=max(4,min(len(words)-1,len(words)//3))
        prompt=" ".join(words[:pivot])
        response=" ".join(words[pivot:])
        out.append({"id":f"sft-{i}","source":"sft","prompt":prompt,"response":response})
    if not out:
        for i in range(max(8,max_samples or 8)):
            prompt=f"Write a tiny story about number {i}."
            response=f"This is story {i}. It has a beginning, a middle, and a clean ending."
            out.append({"id":f"sft-fallback-{i}","source":"sft","prompt":prompt,"response":response})
    return out[:max_samples] if max_samples else out

def _split(records:List[Dict[str,Any]],eval_ratio:float,seed:int)->Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    rng=random.Random(seed)
    idx=list(range(len(records)))
    rng.shuffle(idx)
    cut=max(1,int(len(records)*(1.0-eval_ratio))) if records else 0
    tr=[records[i] for i in idx[:cut]]
    ev=[records[i] for i in idx[cut:]]
    if not ev and tr:
        ev=tr[-max(1,min(8,len(tr)//10)):]
        tr=tr[:-len(ev)] if len(tr)>len(ev) else tr
    return tr,ev

def _build_humaneval_dummy(n:int=8)->List[Dict[str,Any]]:
    rows=[]
    for i in range(n):
        prompt=f"def solve_{i}(x:int)->int:\n    \"\"\"Return x plus {i}.\"\"\"\n"
        rows.append({"id":f"humaneval-{i}","prompt":prompt,"tests":[f"assert solve_{i}(2)=={i+2}"]})
    return rows

def prepare_data(cfg:AppConfig,download:bool=False,max_samples:int|None=None,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    max_s=max_samples or cfg.data.max_samples
    raw_dir=ensure_dir(cfg.paths.raw_dir)
    proc_dir=ensure_dir(cfg.paths.processed_dir)
    dolma=load_local_records(cfg.data.dolma_path,"dolma",max_s)
    tiny=load_local_records(cfg.data.tinystories_path,"tinystories",max_s)
    if download and not tiny:
        tiny=_download_tinystories(max_s)
    if download and not dolma:
        dolma=_download_dolma(max_s)
    if not dolma:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="datasets",
                func="prepare_data",
                action="synthetic_dolma",
                reason="dolma_unavailable",
                extra_dict={"dolma_path":cfg.data.dolma_path,"download":bool(download)}
            )
        dolma=_synthetic_records("dolma",max(32,min(256,max_s)))
    if not tiny:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="datasets",
                func="prepare_data",
                action="synthetic_tinystories",
                reason="tinystories_unavailable",
                extra_dict={"tinystories_path":cfg.data.tinystories_path,"download":bool(download)}
            )
        tiny=_synthetic_records("tinystories",max(32,min(256,max_s)))
    cpt_records=(dolma+tiny)[:max_s]
    sft_records=_to_sft_pairs(tiny,max_s)
    train,evals=_split(cpt_records,cfg.data.eval_ratio,cfg.runtime.seed)
    sft_train,sft_eval=_split(sft_records,cfg.data.eval_ratio,cfg.runtime.seed+1)
    paths={
        "train":str(proc_dir/"train.jsonl"),
        "eval":str(proc_dir/"eval.jsonl"),
        "sft_train":str(proc_dir/"sft_train.jsonl"),
        "sft_eval":str(proc_dir/"sft_eval.jsonl"),
        "humaneval_dummy":str(proc_dir/"humaneval_dummy.jsonl")
    }
    write_jsonl(paths["train"],train)
    write_jsonl(paths["eval"],evals)
    write_jsonl(paths["sft_train"],sft_train)
    write_jsonl(paths["sft_eval"],sft_eval)
    write_jsonl(paths["humaneval_dummy"],_build_humaneval_dummy())
    manifest={
        "created_at":now_iso(),
        "sources":{"dolma_path":cfg.data.dolma_path,"tinystories_path":cfg.data.tinystories_path},
        "counts":{"dolma":len(dolma),"tinystories":len(tiny),"train":len(train),"eval":len(evals),"sft_train":len(sft_train),"sft_eval":len(sft_eval)},
        "paths":paths,
        "fallbacks":tr.snapshot_fallbacks(limit=32) if tr is not None else []
    }
    write_json(raw_dir/"manifest.prepare_data.json",manifest)
    return manifest

# Stage-0 benchmark dataset utilities (HellaSwag).
# Entrypoint: load_hellaswag_items.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Tuple
import json
import random
import shutil
from urllib.request import Request,urlopen
from .config import AppConfig

HELLASWAG_VAL_URL="https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
HELLASWAG_TRAIN_URL="https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"
MMLU_PRO_DATASET_ID="TIGER-Lab/MMLU-Pro"
GSM8K_DATASET_ID="openai/gsm8k"


def _bench_root(cfg:AppConfig)->Path:
    return Path(cfg.paths.root)/"data"/"benchmarks"/"hellaswag"


def _hellaswag_path(cfg:AppConfig,split:str)->Path:
    s=str(split or "val").strip().lower()
    if s not in {"val","validation","train"}:
        s="val"
    name="hellaswag_train.jsonl" if s=="train" else "hellaswag_val.jsonl"
    return _bench_root(cfg)/name


def _download_to(url:str,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    req=Request(url,headers={"User-Agent":"HildaNext/1.0"})
    with urlopen(req,timeout=120) as resp,dst.open("wb") as out:
        shutil.copyfileobj(resp,out)


def ensure_hellaswag_split(cfg:AppConfig,split:str="val",force_download:bool=False)->Tuple[Path,str]:
    s=str(split or "val").strip().lower()
    if s not in {"val","validation","train"}:
        s="val"
    path=_hellaswag_path(cfg,s)
    src=HELLASWAG_TRAIN_URL if s=="train" else HELLASWAG_VAL_URL
    if force_download or (not path.exists()) or int(path.stat().st_size)<1024:
        _download_to(src,path)
        return path,"downloaded"
    return path,"cache"


def ensure_hellaswag_validation(cfg:AppConfig,force_download:bool=False)->Tuple[Path,str]:
    return ensure_hellaswag_split(cfg,split="val",force_download=force_download)


def _row_to_item(row:Dict[str,Any])->Dict[str,Any]|None:
    endings=[str(x).strip() for x in list(row.get("endings") or [])]
    if len(endings)<4:
        return None
    endings=endings[:4]
    stem=str(row.get("ctx") or "").strip()
    if not stem:
        stem=f"{str(row.get('ctx_a') or '').strip()} {str(row.get('ctx_b') or '').strip()}".strip()
    if not stem:
        return None
    try:
        label=int(str(row.get("label","0")).strip())
    except Exception:
        label=0
    label=max(0,min(3,label))
    source_id=str(row.get("source_id","unknown"))
    ind=str(row.get("ind","0"))
    return {
        "id":f"{source_id}:{ind}",
        "stem":stem,
        "endings":endings,
        "label":label,
    }


def load_hellaswag_items(
    cfg:AppConfig,
    limit:int=64,
    seed:int=42,
    force_download:bool=False,
    split:str="val",
)->Dict[str,Any]:
    path,source=ensure_hellaswag_split(cfg,split=split,force_download=force_download)
    rows:List[Dict[str,Any]]=[]
    with path.open("r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                row=json.loads(line)
            except Exception:
                continue
            if not isinstance(row,dict):
                continue
            item=_row_to_item(row)
            if item is not None:
                rows.append(item)
    if not rows:
        raise RuntimeError("hellaswag_dataset_empty_or_invalid")
    total=len(rows)
    k=max(1,min(int(limit),total))
    if k<total:
        rng=random.Random(int(seed))
        idxs=sorted(rng.sample(range(total),k))
        items=[rows[i] for i in idxs]
    else:
        items=rows
    return {
        "dataset_path":str(path),
        "source":source,
        "split":"train" if str(split or "val").strip().lower()=="train" else "val",
        "total_items":total,
        "items":items,
    }


def _sample_indices(total:int,limit:int,seed:int)->List[int]:
    if total<=0:
        return []
    k=max(1,min(int(limit),int(total)))
    if k>=total:
        return list(range(total))
    rng=random.Random(int(seed))
    return sorted(rng.sample(range(total),k))


def _load_hf_dataset(name:str,split:str,config_name:str|None=None):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "datasets_library_missing_or_broken: install with `pip install datasets`"
        ) from e
    if config_name:
        return load_dataset(name,config_name,split=split)
    return load_dataset(name,split=split)


def load_mmlu_pro_items(cfg:AppConfig,limit:int=150,seed:int=42,split:str="test")->Dict[str,Any]:
    _=cfg
    ds=_load_hf_dataset(MMLU_PRO_DATASET_ID,split=split)
    total=int(len(ds))
    idxs=_sample_indices(total=total,limit=limit,seed=seed)
    labels="ABCDEFGHIJ"
    items:List[Dict[str,Any]]=[]
    for idx in idxs:
        row=ds[int(idx)]
        if not isinstance(row,dict):
            continue
        question=str(row.get("question","")).strip()
        opts=[str(x).strip() for x in list(row.get("options") or []) if str(x).strip()]
        if not question or len(opts)<2:
            continue
        valid_labels=labels[:min(len(opts),10)]
        ans=str(row.get("answer","")).strip().upper()
        if ans not in valid_labels:
            try:
                ans_i=int(row.get("answer_index",-1))
            except Exception:
                ans_i=-1
            if 0<=ans_i<len(valid_labels):
                ans=valid_labels[ans_i]
        if ans not in valid_labels:
            continue
        qid=row.get("question_id",idx)
        items.append({
            "id":f"mmlu-pro:{qid}",
            "question":question,
            "options":opts[:10],
            "answer_label":ans,
            "category":str(row.get("category","")).strip(),
            "source":str(row.get("src","")).strip(),
        })
    if not items:
        raise RuntimeError("mmlu_pro_dataset_empty_or_invalid")
    return {
        "dataset_id":MMLU_PRO_DATASET_ID,
        "source":"huggingface",
        "split":split,
        "total_items":total,
        "items":items,
    }


def load_gsm8k_items(
    cfg:AppConfig,
    limit:int=150,
    seed:int=42,
    split:str="test",
    config_name:str="main",
)->Dict[str,Any]:
    _=cfg
    ds=_load_hf_dataset(GSM8K_DATASET_ID,config_name=config_name,split=split)
    total=int(len(ds))
    idxs=_sample_indices(total=total,limit=limit,seed=seed)
    items:List[Dict[str,Any]]=[]
    for idx in idxs:
        row=ds[int(idx)]
        if not isinstance(row,dict):
            continue
        q=str(row.get("question","")).strip()
        a=str(row.get("answer","")).strip()
        if not q or not a:
            continue
        items.append({
            "id":f"gsm8k:{idx}",
            "question":q,
            "answer_target":a,
        })
    if not items:
        raise RuntimeError("gsm8k_dataset_empty_or_invalid")
    return {
        "dataset_id":GSM8K_DATASET_ID,
        "config":config_name,
        "source":"huggingface",
        "split":split,
        "total_items":total,
        "items":items,
    }

# Stage0 WSD-only pipeline with strict real-data gating.
# Entrypoints: dolma_manifest,prepare_dolma_only,preflight_wsd,run_wsd.
# Provides preflight,archive,run report outputs for overnight runs.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,Iterator,List,Tuple
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
import numpy as np
import torch
from .config import AppConfig,clone_with_updates,save_config
from .io_utils import ensure_dir,write_json,read_json,read_jsonl
from .trace import use_trace,exception_with_stack
from .tokenization import tokenize_split
from .inference import load_model_bundle
from .ar import generate_ar
from .training import run_wsd_conversion
from .masks import doc_attention_mask

ACCEPTED_EXT=(".jsonl",".jsonl.gz",".json.gz",".zst",".parquet",".txt",".json")
DEFAULT_EXTERNAL_DOC_INDEX_DIR=Path("E:/DIFFUSION/HildaNext/dolma_v1_6_sample_1767050862/doc_index")

def _count_bytes(p:Path)->int:
    if p.is_file():
        return int(p.stat().st_size)
    total=0
    for f in p.rglob("*"):
        if f.is_file():
            total+=int(f.stat().st_size)
    return total

def _infer_ext(name:str)->str:
    low=name.lower()
    for e in ACCEPTED_EXT:
        if low.endswith(e):
            return e
    if low.endswith(".npy"):
        return ".npy"
    if low.endswith(".bin"):
        return ".bin"
    if "." in low:
        return "."+low.split(".")[-1]
    return ""

def _inspect_path(p:Path)->Dict[str,Any]:
    if not p.exists():
        return {"path":str(p),"file_count":0,"total_bytes":0,"extensions":{},"examples":[],"verdict":"REAL_MISSING","kind":"missing"}
    if p.is_file():
        ext=_infer_ext(p.name)
        ok=ext in ACCEPTED_EXT
        return {"path":str(p),"file_count":1,"total_bytes":int(p.stat().st_size),"extensions":{ext:1},"examples":[str(p)],"verdict":"REAL_OK" if ok else "REAL_UNSUPPORTED","kind":"file"}
    files=[]
    ext_count:Dict[str,int]={}
    total=0
    has_supported=False
    has_tokens=False
    has_doc_index=False
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        files.append(str(f))
        total+=int(f.stat().st_size)
        ext=_infer_ext(f.name)
        ext_count[ext]=ext_count.get(ext,0)+1
        if ext in ACCEPTED_EXT:
            has_supported=True
        if f.name.lower().startswith("tokens_") and ext in {".npy",".bin"}:
            has_tokens=True
        if f.name.lower().startswith("doc_index_") and ext==".npy":
            has_doc_index=True
    verdict="REAL_OK" if has_supported or (has_tokens and has_doc_index) else ("REAL_UNSUPPORTED" if has_doc_index else "REAL_MISSING")
    kind="pretokenized" if has_tokens and has_doc_index else "dir"
    return {"path":str(p),"file_count":len(files),"total_bytes":total,"extensions":ext_count,"examples":files[:5],"verdict":verdict,"kind":kind}

def _candidate_paths(cfg:AppConfig)->List[Path]:
    out:List[Path]=[]
    seen=set()
    def add(p:Path):
        k=str(p).lower().strip()
        if not k or k in seen:
            return
        seen.add(k)
        out.append(p)
    if cfg.data.dolma_path:
        add(Path(cfg.data.dolma_path))
    env=os.environ.get("HILDANEXT_DOLMA_PATH","").strip()
    if env:
        add(Path(env))
    roots=[Path("E:/DIFFUSION/HildaNext"),Path(cfg.paths.root),Path(cfg.paths.root).parent]
    for r in roots:
        if not r.exists():
            continue
        for d in sorted(r.glob("dolma_v1_6_sample_*")):
            add(d)
    return out

def dolma_manifest(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    scans=[_inspect_path(p) for p in _candidate_paths(cfg)]
    chosen=None
    for s in scans:
        if s.get("verdict")=="REAL_OK":
            chosen=s
            break
    if chosen is None:
        chosen=scans[0] if scans else {"path":"","file_count":0,"total_bytes":0,"extensions":{},"examples":[],"verdict":"REAL_MISSING","kind":"missing"}
    rep={
        "run_id":getattr(tr,"run_id",""),
        "root_path":chosen.get("path",""),
        "file_count":int(chosen.get("file_count",0)),
        "total_bytes":int(chosen.get("total_bytes",0)),
        "extensions":chosen.get("extensions",{}),
        "5_example_files":chosen.get("examples",[])[:5],
        "verdict":chosen.get("verdict","REAL_MISSING"),
        "kind":chosen.get("kind",""),
        "candidates":scans
    }
    out=Path(cfg.paths.root)/"runs"/"reports"/f"{getattr(tr,'run_id','run')}_dolma_manifest.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    if tr is not None:
        tr.record_notice(module="wsd_stage0",func="dolma_manifest",action="dolma_manifest",reason=rep["verdict"],extra_dict={"root_path":rep["root_path"],"file_count":rep["file_count"]})
    return rep

def _extract_text(obj:Dict[str,Any])->str:
    for k in ("text","content","body","document","story"):
        v=obj.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def _iter_json_lines(fh:Any,max_docs:int|None,seen:int)->Iterator[Tuple[str,str,int]]:
    n=seen
    for i,line in enumerate(fh):
        if max_docs is not None and n>=max_docs:
            break
        if not line or not str(line).strip():
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        text=_extract_text(obj if isinstance(obj,dict) else {})
        if not text:
            continue
        doc_id=str(obj.get("id",f"doc_{i}")) if isinstance(obj,dict) else f"doc_{i}"
        n+=1
        yield doc_id,text,n

def stream_docs(path:Path,max_docs:int|None=None)->Iterator[Tuple[str,str]]:
    n=0
    low=path.name.lower()
    if path.is_file() and low.endswith(".jsonl"):
        with path.open("r",encoding="utf-8",errors="ignore") as f:
            for doc_id,text,n in _iter_json_lines(f,max_docs,n):
                yield doc_id,text
        return
    if path.is_file() and (low.endswith(".jsonl.gz") or low.endswith(".json.gz")):
        with gzip.open(path,"rt",encoding="utf-8",errors="ignore") as f:
            for doc_id,text,n in _iter_json_lines(f,max_docs,n):
                yield doc_id,text
        return
    if path.is_file() and low.endswith(".zst"):
        import zstandard as zstd
        with path.open("rb") as fh:
            dctx=zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                txt=io.TextIOWrapper(reader,encoding="utf-8",errors="ignore")
                for doc_id,text,n in _iter_json_lines(txt,max_docs,n):
                    yield doc_id,text
        return
    if path.is_file() and low.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024):
            cols=batch.to_pydict()
            rows=batch.num_rows
            for i in range(rows):
                if max_docs is not None and n>=max_docs:
                    return
                obj={k:(v[i] if i<len(v) else None) for k,v in cols.items()}
                text=_extract_text(obj)
                if not text:
                    continue
                doc_id=str(obj.get("id",f"doc_{n}"))
                n+=1
                yield doc_id,text
        return
    if path.is_file() and low.endswith(".txt"):
        with path.open("r",encoding="utf-8",errors="ignore") as f:
            for i,line in enumerate(f):
                if max_docs is not None and n>=max_docs:
                    break
                text=str(line).strip()
                if not text:
                    continue
                n+=1
                yield f"doc_{i}",text
        return
    if path.is_dir():
        files=[f for f in sorted(path.rglob("*")) if f.is_file() and _infer_ext(f.name) in ACCEPTED_EXT]
        for fp in files:
            if max_docs is not None and n>=max_docs:
                break
            for doc_id,text in stream_docs(fp,None if max_docs is None else max_docs-n):
                if max_docs is not None and n>=max_docs:
                    break
                n+=1
                yield f"{fp.stem}:{doc_id}",text
        return
    raise RuntimeError(f"unsupported dolma path: {path}")

def _inspect_existing_doc_index(root:Path)->Dict[str,Any]:
    di=root if root.name.lower()=="doc_index" else (root/"doc_index")
    if not di.exists() or not di.is_dir():
        return {"exists":False,"compatible":False,"reason":"missing"}
    files=sorted(di.glob("doc_index_*.npy"))
    if not files:
        return {"exists":True,"compatible":False,"reason":"empty"}
    samples=[]
    ok=True
    for fp in files[:5]:
        try:
            arr=np.load(fp,mmap_mode="r")
            mn=int(arr.min()) if arr.size else 0
            mx=int(arr.max()) if arr.size else 0
            sh=[int(x) for x in arr.shape]
            dt=str(arr.dtype)
            if arr.ndim not in (1,2):
                ok=False
            samples.append({"file":str(fp),"shape":sh,"dtype":dt,"min":mn,"max":mx})
        except Exception as e:
            ok=False
            samples.append({"file":str(fp),"error":str(e)})
    tok_files=list((root/"tokens").glob("tokens_*.npy")) if (root/"tokens").exists() else []
    compat=ok and len(tok_files)>0
    if root.name.lower()=="doc_index":
        compat=ok and len(files)>0
    return {"exists":True,"compatible":compat,"reason":"ok" if compat else "no_tokens_or_shape_invalid","files_checked":len(samples),"samples":samples,"path":str(di)}

def _resolve_external_doc_index_dir()->Path:
    p=os.environ.get("HILDANEXT_DOLMA_DOC_INDEX_PATH","").strip()
    if p:
        q=Path(p)
        if q.name.lower()=="doc_index":
            return q
        return q/"doc_index"
    return DEFAULT_EXTERNAL_DOC_INDEX_DIR

def _iter_external_doc_rows(doc_dir:Path,target_seq_len:int,max_rows:int)->Iterator[List[int]]:
    yielded=0
    files=sorted(doc_dir.glob("doc_index_*.npy"))
    for fp in files:
        arr=np.load(fp,mmap_mode="r")
        if arr.ndim==1:
            arr=arr.reshape(1,-1)
        cols=int(arr.shape[1]) if arr.ndim==2 else 0
        if cols<=0:
            continue
        for r in range(int(arr.shape[0])):
            row=arr[r].astype(np.int32,copy=False)
            if cols==target_seq_len:
                yield [int(x) for x in row.tolist()]
                yielded+=1
            elif cols>target_seq_len and cols%target_seq_len==0:
                chunks=cols//target_seq_len
                for c in range(chunks):
                    s=c*target_seq_len
                    e=s+target_seq_len
                    yield [int(x) for x in row[s:e].tolist()]
                    yielded+=1
                    if max_rows>0 and yielded>=max_rows:
                        return
            elif cols>target_seq_len:
                yield [int(x) for x in row[:target_seq_len].tolist()]
                yielded+=1
            else:
                pad=[int(x) for x in row.tolist()]+([-1]*(target_seq_len-cols))
                yield pad
                yielded+=1
            if max_rows>0 and yielded>=max_rows:
                return

def _apply_external_doc_index(train_tok:Path,eval_tok:Path,doc_dir:Path,seq_len:int)->Dict[str,Any]:
    train_rows=read_jsonl(train_tok)
    eval_rows=read_jsonl(eval_tok)
    total=len(train_rows)+len(eval_rows)
    repl=list(_iter_external_doc_rows(doc_dir,target_seq_len=seq_len,max_rows=total))
    if len(repl)<total:
        raise RuntimeError("external_doc_index_insufficient_rows")
    i=0
    changed=0
    for rows,p in ((train_rows,train_tok),(eval_rows,eval_tok)):
        out=[]
        for row in rows:
            doc_ids=[int(x) for x in repl[i]]
            i+=1
            if len(doc_ids)!=seq_len:
                raise RuntimeError("external_doc_index_shape_mismatch")
            row["doc_ids"]=doc_ids
            attn=[int(x) for x in row.get("attention_mask",[])]
            if len(attn)==seq_len:
                for k in range(seq_len):
                    if attn[k]<=0:
                        doc_ids[k]=-1
            row["doc_ids"]=doc_ids
            out.append(row)
            changed+=1
        with p.open("w",encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row,ensure_ascii=True)+"\n")
    return {"path":str(doc_dir),"rows_total":total,"rows_replaced":changed}

def _segment_text(text:str,max_words:int=120)->List[str]:
    t=str(text or "").strip()
    if not t:
        return []
    parts=[x.strip() for x in t.replace("\r\n","\n").split("\n\n") if x.strip()]
    if not parts:
        parts=[t]
    out=[]
    for p in parts:
        words=p.split()
        if len(words)<=max_words:
            out.append(" ".join(words))
            continue
        for i in range(0,len(words),max_words):
            chunk=words[i:i+max_words]
            if chunk:
                out.append(" ".join(chunk))
    return [x for x in out if x.strip()]

def _split_processed(cfg:AppConfig,source_path:Path,max_docs:int,eval_pct:float,seed:int)->Dict[str,Any]:
    max_docs_eff=None if int(max_docs)<=0 else int(max_docs)
    proc=ensure_dir(cfg.paths.processed_dir)
    tr_path=proc/"train.jsonl"
    ev_path=proc/"eval.jsonl"
    n_tr=0
    n_ev=0
    tok_tr=0
    tok_ev=0
    with tr_path.open("w",encoding="utf-8") as trf,ev_path.open("w",encoding="utf-8") as evf:
        c=0
        for i,(doc_id,text) in enumerate(stream_docs(source_path,max_docs=max_docs_eff)):
            for j,seg in enumerate(_segment_text(text,max_words=120)):
                h=int(hashlib.sha1(f"{seed}:{doc_id}:{j}".encode("utf-8")).hexdigest()[:8],16)/0xFFFFFFFF
                row={"id":f"dolma-{i}-{j}","source":"dolma","text":seg}
                line=json.dumps(row,ensure_ascii=True)+"\n"
                tok_est=len(seg.split())
                if h<float(eval_pct):
                    evf.write(line)
                    n_ev+=1
                    tok_ev+=tok_est
                else:
                    trf.write(line)
                    n_tr+=1
                    tok_tr+=tok_est
                c+=1
                if max_docs_eff and c>=max_docs_eff:
                    break
            if max_docs_eff and c>=max_docs_eff:
                break
    return {"train_path":str(tr_path),"eval_path":str(ev_path),"num_docs_train":n_tr,"num_docs_eval":n_ev,"num_tokens_train_est":tok_tr,"num_tokens_eval_est":tok_ev}

def _build_tokenized_artifacts(cfg:AppConfig,train_tok_path:Path,eval_tok_path:Path,shard_rows:int=1000)->Dict[str,Any]:
    dp=Path(cfg.data.dolma_path)
    base=dp.parent if dp.name.lower()=="raw" else dp
    base=ensure_dir(base)
    tok_dir=ensure_dir(base/"tokens")
    doc_dir=ensure_dir(base/"doc_index")
    seq_len=int(cfg.data.seq_len)
    shard=0
    rows=0
    cur_tok=[]
    cur_doc=[]
    t0_art=time.time()
    total_input=sum(p.stat().st_size for p in [train_tok_path,eval_tok_path] if p.exists())
    print(f"[artifacts] START  building .npy shards from {total_input//1024//1024} MB tokenized data  shard_rows={shard_rows}",flush=True)
    for p in [train_tok_path,eval_tok_path]:
        with p.open("r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                row=json.loads(line)
                ids=[int(x) for x in row.get("input_ids",[])]
                docs=[int(x) for x in row.get("doc_ids",[])]
                if len(ids)!=seq_len or len(docs)!=seq_len:
                    continue
                cur_tok.append(ids)
                cur_doc.append(docs)
                rows+=1
                if len(cur_tok)>=shard_rows:
                    np.save(tok_dir/f"tokens_{shard:05d}.npy",np.asarray(cur_tok,dtype=np.uint32))
                    np.save(doc_dir/f"doc_index_{shard:05d}.npy",np.asarray(cur_doc,dtype=np.int32))
                    shard+=1
                    cur_tok=[]
                    cur_doc=[]
                    if shard%10==0:
                        print(f"[artifacts]   shard={shard}  rows={rows:,}  elapsed={int(time.time()-t0_art)}s",flush=True)
    if cur_tok:
        np.save(tok_dir/f"tokens_{shard:05d}.npy",np.asarray(cur_tok,dtype=np.uint32))
        np.save(doc_dir/f"doc_index_{shard:05d}.npy",np.asarray(cur_doc,dtype=np.int32))
        shard+=1
    meta={"seq_len":seq_len,"rows_total":rows,"shards":shard,"tokens_dir":str(tok_dir),"doc_index_dir":str(doc_dir)}
    write_json(base/"meta.json",meta)
    print(f"[artifacts] DONE   shards={shard}  rows_total={rows:,}  elapsed={int(time.time()-t0_art)}s",flush=True)
    return meta

def _dolma_fingerprint(cfg:AppConfig,manifest:Dict[str,Any])->Dict[str,Any]:
    core={"root_path":manifest.get("root_path",""),"file_count":manifest.get("file_count",0),"total_bytes":manifest.get("total_bytes",0),"examples":manifest.get("5_example_files",[])}
    raw=json.dumps(core,sort_keys=True,ensure_ascii=True)
    fp={"sha1":hashlib.sha1(raw.encode("utf-8")).hexdigest(),"manifest":core}
    out=Path(cfg.paths.root)/"runs"/"cache"/"dolma_fingerprint.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,fp)
    return fp

def _check_doc_boundary_signal(tokenized_train:Path)->Dict[str,Any]:
    rows=read_jsonl(tokenized_train,max_rows=64)
    changed=False
    for r in rows:
        docs=[int(x) for x in r.get("doc_ids",[]) if int(x)>=0]
        for i in range(max(0,len(docs)-1)):
            if docs[i]!=docs[i+1]:
                changed=True
                break
        if changed:
            break
    return {"rows_checked":len(rows),"doc_boundary_changed":changed}

def _check_no_leakage(tokenized_train:Path)->Dict[str,Any]:
    rows=read_jsonl(tokenized_train,max_rows=8)
    ok=True
    checks=0
    for r in rows:
        docs=torch.tensor([int(x) for x in r.get("doc_ids",[])],dtype=torch.long)
        m=doc_attention_mask(docs,causal=False)
        for i in range(docs.shape[0]):
            di=int(docs[i].item())
            if di<0:
                continue
            for j in range(docs.shape[0]):
                dj=int(docs[j].item())
                if dj<0:
                    continue
                checks+=1
                if di!=dj and bool(m[i,j].item()):
                    ok=False
    return {"rows_checked":len(rows),"pair_checks":checks,"no_cross_doc_leakage":ok}

def _verify_artifacts(cfg:AppConfig)->Dict[str,Any]:
    dp=Path(cfg.data.dolma_path)
    base=dp.parent if dp.name.lower()=="raw" else dp
    meta_path=base/"meta.json"
    tok_dir=base/"tokens"
    doc_dir=base/"doc_index"
    tok_files=sorted(tok_dir.glob("tokens_*.npy")) if tok_dir.exists() else []
    doc_files=sorted(doc_dir.glob("doc_index_*.npy")) if doc_dir.exists() else []
    meta=read_json(meta_path) if meta_path.exists() else {}
    ok=meta_path.exists() and len(tok_files)>0 and len(doc_files)>0 and len(tok_files)==len(doc_files)
    return {"ok":ok,"meta_path":str(meta_path),"tokens_shards":len(tok_files),"doc_index_shards":len(doc_files),"meta":meta}

def _ensure_llada21_objective(cfg:AppConfig)->None:
    if str(cfg.stage0.objective_mode)!="llada21_mixture":
        raise RuntimeError("llada21_mixture_missing")
    if not bool(cfg.stage0.t2t_enabled):
        raise RuntimeError("llada21_mixture_missing")

def _select_optimizer_name()->str:
    if not torch.cuda.is_available():
        return "adamw"
    try:
        import bitsandbytes  # noqa: F401
        return "adamw8bit"
    except Exception:
        return "adamw"

def _apply_stage0_to_cfg(cfg:AppConfig,run_id:str|None=None)->AppConfig:
    total=max(1,int(cfg.stage0.steps_total_stage0))
    warm=max(1,int(total*float(cfg.stage0.warmup_frac)))
    stable=max(1,int(total*float(cfg.stage0.stable_frac)))
    decay=max(1,total-warm-stable)
    return clone_with_updates(cfg,{
        "runtime":{"run_id":run_id or cfg.runtime.run_id,"use_dinfer":False,"force_dummy_model":False,"device":"cuda"},
        "data":{"seq_len":int(cfg.stage0.seq_len),"doc_mask_mode":str(cfg.stage0.doc_attention_mask_mode),"eval_ratio":float(cfg.data.eval_pct_stage0)},
        "llada2":{"mask_mode":str(cfg.stage0.doc_attention_mask_mode),"composite_block_size":max(1,min(int(cfg.stage0.seq_len),32))},
        "train":{"dtype":"fp16","batch_size":int(cfg.stage0.micro_batch_size),"accum_steps":int(cfg.stage0.grad_accum_steps),"lr":float(cfg.stage0.lr_stage0),"max_steps":total,"m2t_weight":float(cfg.stage0.m2t_weight),"t2t_weight":float(cfg.stage0.t2t_weight),"mask_ratio":float(cfg.stage0.mask_ratio_m2t),"t2t_noise_ratio":float(cfg.stage0.t2t_edit_ratio),"ckpt_every":int(cfg.stage0.save_every_steps),"eval_every":max(1,int(cfg.stage0.eval_every_steps) if int(cfg.stage0.eval_every_steps)>0 else total+1),"log_every_steps":int(cfg.stage0.log_every_steps),"keep_last_checkpoints":int(cfg.stage0.keep_last_checkpoints),"optimizer":_select_optimizer_name(),"grad_ckpt":True},
        "wsd":{"warmup_steps":warm,"stable_steps":stable,"decay_steps":decay,"start_block_size":1,"max_block_size":int(cfg.stage0.seq_len),"end_block_size":int(cfg.stage0.decay_blocks[-1] if cfg.stage0.decay_blocks else 32),"ladder_blocks":[int(x) for x in cfg.stage0.ladder_blocks],"decay_blocks":[int(x) for x in cfg.stage0.decay_blocks],"enforce_divisibility":True}
    })

def prepare_dolma_only(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    man=dolma_manifest(cfg,trace=tr)
    if man.get("verdict")!="REAL_OK":
        if tr is not None:
            tr.record_fallback(event="fallback",module="wsd_stage0",func="prepare_dolma_only",action="synthetic_dolma",reason="dolma_unavailable",extra_dict={"manifest":man})
        raise RuntimeError("dolma_unavailable")
    source=Path(man["root_path"])
    existing_doc_index=_inspect_existing_doc_index(source) if source.is_dir() else {"exists":False,"compatible":False,"reason":"source_not_dir"}
    ext_doc_dir=_resolve_external_doc_index_dir()
    ext_doc_info=_inspect_existing_doc_index(ext_doc_dir if ext_doc_dir.exists() else Path("__missing__"))
    # Checkpoint 1: skip _split_processed if output already exists and is non-empty (resume after crash)
    proc_dir=Path(cfg.paths.processed_dir)
    proc_train=proc_dir/"train.jsonl"
    proc_eval=proc_dir/"eval.jsonl"
    _MIN_PROC_BYTES=1024*1024  # at least 1 MB to be considered valid
    if proc_train.exists() and proc_eval.exists() and proc_train.stat().st_size>_MIN_PROC_BYTES and proc_eval.stat().st_size>_MIN_PROC_BYTES:
        # Count lines with progress (36 GB file can take a few minutes)
        def _count_lines_progress(p:Path,label:str)->int:
            n=0
            t0_c=time.time()
            last_c=t0_c
            sz=p.stat().st_size
            print(f"[prep] counting lines in {label} ({sz//1024//1024} MB) ...",flush=True)
            with p.open("r",encoding="utf-8",errors="ignore") as _f:
                for _ in _f:
                    n+=1
                    now=time.time()
                    if now-last_c>=15:
                        print(f"[prep]   {label}: {n:,} lines so far  ({int(now-t0_c)}s)",flush=True)
                        last_c=now
            print(f"[prep] {label}: {n:,} lines total  ({int(time.time()-t0_c)}s)",flush=True)
            return n
        n_tr=_count_lines_progress(proc_train,"train")
        n_ev=_count_lines_progress(proc_eval,"eval")
        split={"train_path":str(proc_train),"eval_path":str(proc_eval),"num_docs_train":n_tr,"num_docs_eval":n_ev,"num_tokens_train_est":0,"num_tokens_eval_est":0,"skipped":True,"reason":"processed_already_exists"}
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="prepare_dolma_only",action="split_processed_skip",reason="processed_already_exists",extra_dict={"train_rows":n_tr,"eval_rows":n_ev,"train_bytes":proc_train.stat().st_size})
    else:
        split=_split_processed(cfg,source_path=source,max_docs=int(cfg.data.max_samples),eval_pct=float(cfg.data.eval_pct_stage0),seed=99)
    if split["num_docs_train"]<=0:
        if tr is not None:
            tr.record_fallback(event="fallback",module="wsd_stage0",func="prepare_dolma_only",action="download_false_empty",reason="dataset_empty",extra_dict={"root_path":man["root_path"],"max_docs":cfg.data.max_samples})
        raise RuntimeError("dataset_empty")
    tok_dir=ensure_dir(cfg.paths.tokenized_dir)
    train_tok=Path(tok_dir)/"train.jsonl"
    eval_tok=Path(tok_dir)/"eval.jsonl"
    # Checkpoint 2: skip tokenize_split if tokenized output already exists and is non-empty
    if train_tok.exists() and eval_tok.exists() and train_tok.stat().st_size>_MIN_PROC_BYTES and eval_tok.stat().st_size>_MIN_PROC_BYTES:
        tok_train={"input":split["train_path"],"output":str(train_tok),"records_in":split["num_docs_train"],"records_out":-1,"seq_len":int(cfg.data.seq_len),"skipped":True,"reason":"tokenized_already_exists"}
        tok_eval={"input":split["eval_path"],"output":str(eval_tok),"records_in":split["num_docs_eval"],"records_out":-1,"seq_len":int(cfg.data.seq_len),"skipped":True,"reason":"tokenized_already_exists"}
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="prepare_dolma_only",action="tokenize_skip",reason="tokenized_already_exists",extra_dict={"train_bytes":train_tok.stat().st_size})
    else:
        tok_train=tokenize_split(cfg,split["train_path"],str(train_tok),max_records=None,trace=tr)
        tok_eval=tokenize_split(cfg,split["eval_path"],str(eval_tok),max_records=None,trace=tr)
    ext_apply={"path":str(ext_doc_dir),"rows_total":0,"rows_replaced":0,"mode":"internal_doc_ids"}
    if ext_doc_info.get("exists") and ext_doc_info.get("compatible"):
        ext_apply=_apply_external_doc_index(train_tok,eval_tok,doc_dir=ext_doc_dir,seq_len=int(cfg.data.seq_len))
        ext_apply["mode"]="external_doc_index"
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="prepare_dolma_only",action="external_doc_index_apply",reason="ok",extra_dict=ext_apply)
    else:
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="prepare_dolma_only",action="external_doc_index_skip",reason="incompatible_or_missing",extra_dict={"path":str(ext_doc_dir),"info":ext_doc_info})
    art=_build_tokenized_artifacts(cfg,train_tok_path=train_tok,eval_tok_path=eval_tok,shard_rows=1000)
    verify_art=_verify_artifacts(cfg)
    fp=_dolma_fingerprint(cfg,man)
    rep={"manifest":man,"existing_doc_index":existing_doc_index,"external_doc_index":ext_doc_info,"external_doc_index_apply":ext_apply,"split":split,"tokenize":{"train":tok_train,"eval":tok_eval},"artifacts":art,"verify_artifacts":verify_art,"fingerprint":fp}
    out=Path(cfg.paths.root)/"runs"/"reports"/f"{getattr(tr,'run_id','run')}_dolma_prep.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    return rep

def verify_dolma_only(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    man=dolma_manifest(cfg,trace=tr)
    art=_verify_artifacts(cfg)
    train_tok=Path(cfg.paths.tokenized_dir)/"train.jsonl"
    boundary=_check_doc_boundary_signal(train_tok) if train_tok.exists() else {"rows_checked":0,"doc_boundary_changed":False}
    leak=_check_no_leakage(train_tok) if train_tok.exists() else {"rows_checked":0,"pair_checks":0,"no_cross_doc_leakage":False}
    boundary_warn=not bool(boundary.get("doc_boundary_changed"))
    rep={"ok":bool(man.get("verdict")=="REAL_OK" and art.get("ok") and leak.get("no_cross_doc_leakage")),"manifest":man,"artifacts":art,"boundary":boundary,"leakage":leak,"warnings":{"doc_boundary_changed_missing":boundary_warn}}
    if boundary_warn and tr is not None:
        tr.record_notice(module="wsd_stage0",func="verify_dolma_only",action="doc_boundary_changed_missing",reason="warn",extra_dict={"rows_checked":boundary.get("rows_checked",0)})
    out=Path(cfg.paths.root)/"runs"/"reports"/f"{getattr(tr,'run_id','run')}_dolma_verify.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    return rep

def preflight_wsd(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    run_cfg=_apply_stage0_to_cfg(cfg,getattr(tr,"run_id",""))
    rep={"ok":False,"run_id":getattr(tr,"run_id",""),"error":"","env":{},"data":{},"model":{},"train_probe":{},"fallbacks":[]}
    numpy_ok=False
    try:
        env=os.environ.get("CONDA_DEFAULT_ENV","")
        exe=str(sys.executable).replace("/","\\").lower()
        is_mdm=env.lower()=="mdm" or "\\envs\\mdm\\" in exe
        rep["env"]["conda_env"]=env
        rep["env"]["is_mdm"]=is_mdm
        rep["env"]["python"]=sys.executable
        rep["env"]["cuda_available"]=torch.cuda.is_available()
        if not is_mdm:
            raise RuntimeError("env_not_mdm")
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        try:
            import numpy as _np
            _=_np.zeros(1)
            numpy_ok=True
        except Exception as e:
            numpy_ok=False
            if tr is not None:
                tr.record_env_issue(name="numpy_dll_unavailable",detail=str(e),module="wsd_stage0",func="preflight_wsd")
            raise RuntimeError("numpy_dll_unavailable")
        free=float(shutil.disk_usage(run_cfg.paths.root).free)/1024**3
        rep["disk_free_gb"]=free
        if free<float(run_cfg.runtime.min_disk_free_gb):
            raise RuntimeError("low_disk_space")
        _ensure_llada21_objective(run_cfg)
        verify_before=verify_dolma_only(run_cfg,trace=tr)
        if verify_before.get("ok"):
            prep={"skipped":True,"reason":"artifacts_ready"}
            verify=verify_before
            if tr is not None:
                tr.record_notice(module="wsd_stage0",func="preflight_wsd",action="prepare_dolma_skip",reason="artifacts_ready")
        else:
            prep=prepare_dolma_only(run_cfg,trace=tr)
            verify=verify_dolma_only(run_cfg,trace=tr)
        manifest_verdict=verify.get("manifest",{}).get("verdict","")
        split_info=prep.get("split",{})
        rep["data"]={"manifest_verdict":manifest_verdict,"split":split_info,"verify":verify,"prep":prep}
        if not verify.get("ok"):
            raise RuntimeError("dolma_verify_failed")
        bundle=load_model_bundle(run_cfg,for_training=False,trace=tr)
        rep["model"]={"dummy_model":bundle.is_dummy,"load_reason":bundle.load_reason,"dtype":bundle.actual_dtype,"device":str(bundle.device),"model_name_or_path":bundle.model_name_or_path}
        if bundle.is_dummy:
            if tr is not None:
                tr.record_fallback(event="fallback",module="wsd_stage0",func="preflight_wsd",action="dummy_model_fallback",reason="dummy_model_fallback",extra_dict={"load_reason":bundle.load_reason})
            raise RuntimeError("dummy_model_fallback")
        ar=generate_ar(run_cfg,prompt="Write one short safe line.",max_new_tokens=16,seed=run_cfg.runtime.seed,trace=tr)
        rep["model"]["sample_text_ar"]=ar.get("text","")
        if not str(ar.get("text","")).strip():
            raise RuntimeError("ar_empty_output")
        probe_cfg=clone_with_updates(run_cfg,{"paths":{"logs_dir":str(Path(run_cfg.paths.logs_dir)/f"{tr.run_id}_preflight_probe"),"checkpoints_dir":str(Path(run_cfg.paths.checkpoints_dir)/f"{tr.run_id}_preflight_probe")},"train":{"max_steps":1,"ckpt_every":1,"eval_every":2},"data":{"seq_len":256},"stage0":{"seq_len":256}})
        probe=run_wsd_conversion(probe_cfg,steps=1,trace=tr,resume=False,ckpt_every=1,eval_every=2)
        ckpt=Path(probe["checkpoints_dir"])/"step_00001"
        ckpt_ok=ckpt.exists()
        load_ok=False
        if ckpt_ok:
            try:
                from transformers import AutoModelForCausalLM
                _=AutoModelForCausalLM.from_pretrained(str(ckpt),trust_remote_code=run_cfg.model.trust_remote_code)
                load_ok=True
            except Exception as e:
                if tr is not None:
                    tr.record_fallback(event="fallback",module="wsd_stage0",func="preflight_wsd",action="checkpoint_reload_failed",reason="checkpoint_reload_failed",exception_str=exception_with_stack(e),extra_dict={"path":str(ckpt)})
                load_ok=False
        rep["train_probe"]={"summary":probe,"checkpoint_exists":ckpt_ok,"checkpoint_load_ok":load_ok}
        if not (ckpt_ok and load_ok):
            raise RuntimeError("checkpoint_probe_failed")
        events=tr.all_events() if tr is not None else []
        blocking=[e for e in events if tr.is_blocking(e,numpy_ok=numpy_ok)] if tr is not None else []
        rep["fallbacks"]=events
        rep["fallbacks_total_count"]=tr.count_fallbacks() if tr is not None else 0
        rep["fallbacks_count"]=len(blocking)
        rep["fallbacks_blocking_count"]=len(blocking)
        rep["ok"]=len(blocking)==0
        if not rep["ok"]:
            raise RuntimeError("strict_run_gate_failed")
        print("OK PREPARED")
    except Exception as e:
        rep["ok"]=False
        rep["error"]=str(e)
        rep["fallbacks"]=tr.all_events() if tr is not None else []
        rep["fallbacks_total_count"]=tr.count_fallbacks() if tr is not None else 0
        blocking=[x for x in rep["fallbacks"] if tr.is_blocking(x,numpy_ok=numpy_ok)] if tr is not None else []
        rep["fallbacks_count"]=len(blocking)
        rep["fallbacks_blocking_count"]=len(blocking)
        out=Path(run_cfg.paths.root)/"runs"/"reports"/f"{tr.run_id}_preflight.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        write_json(out,rep)
        raise
    out=Path(run_cfg.paths.root)/"runs"/"reports"/f"{tr.run_id}_preflight.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    return rep

def archive_runs(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    runs=Path(cfg.paths.root)/"runs"
    stamp=time.strftime("%Y%m%d_%H%M%S",time.gmtime())
    dst=runs/"_archive"/stamp
    dst.mkdir(parents=True,exist_ok=True)
    ops=[]
    for p in runs.glob("*"):
        if not p.is_dir():
            continue
        n=p.name.lower()
        if n=="_archive":
            continue
        if not (n.startswith("checkpoints") or n.startswith("logs") or n.startswith("reports")):
            continue
        target=dst/p.name
        i=1
        while target.exists():
            target=dst/f"{p.name}_{i}"
            i+=1
        b=_count_bytes(p)
        ok=True
        err=""
        try:
            shutil.move(str(p),str(target))
        except Exception as e:
            ok=False
            err=str(e)
        row={"src":str(p),"dst":str(target),"bytes":b,"ok":ok,"error":err}
        ops.append(row)
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="archive_runs",action="archive",reason="ok" if ok else "error",extra_dict=row)
    rep={"archive_dir":str(dst),"ops":ops}
    out=Path(cfg.paths.root)/"runs"/"reports"/f"{getattr(tr,'run_id','run')}_archive.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    return rep

def run_wsd(cfg:AppConfig,config_path:str,trace=None,skip_dolma_prep:bool=False)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    run_cfg=_apply_stage0_to_cfg(cfg,getattr(tr,"run_id",""))
    _ensure_llada21_objective(run_cfg)
    if skip_dolma_prep:
        # Prep must have been done separately; just verify tokenized .jsonl exist.
        train_tok=Path(run_cfg.paths.tokenized_dir)/"train.jsonl"
        eval_tok=Path(run_cfg.paths.tokenized_dir)/"eval.jsonl"
        if not train_tok.exists() or not eval_tok.exists():
            raise RuntimeError(f"skip_dolma_prep_set_but_tokenized_missing paths={train_tok},{eval_tok}")
        prep={"skipped":True,"reason":"skip_dolma_prep_flag"}
        verify={"ok":True,"skipped":True,"reason":"skip_dolma_prep_flag"}
        if tr is not None:
            tr.record_notice(module="wsd_stage0",func="run_wsd",action="prepare_dolma_skip",reason="skip_dolma_prep_flag",extra_dict={"train_tok":str(train_tok),"eval_tok":str(eval_tok)})
    else:
        verify_before=verify_dolma_only(run_cfg,trace=tr)
        if verify_before.get("ok"):
            prep={"skipped":True,"reason":"artifacts_ready"}
            verify=verify_before
            if tr is not None:
                tr.record_notice(module="wsd_stage0",func="run_wsd",action="prepare_dolma_skip",reason="artifacts_ready")
        else:
            prep=prepare_dolma_only(run_cfg,trace=tr)
            verify=verify_dolma_only(run_cfg,trace=tr)
        if not verify.get("ok"):
            raise RuntimeError("dolma_verify_failed")
    summary=run_wsd_conversion(run_cfg,steps=int(run_cfg.stage0.steps_total_stage0),trace=tr,resume=True,ckpt_every=int(run_cfg.stage0.save_every_steps),eval_every=max(1,int(run_cfg.stage0.eval_every_steps) if int(run_cfg.stage0.eval_every_steps)>0 else int(run_cfg.stage0.steps_total_stage0)+1))
    ck_root=Path(summary["checkpoints_dir"])
    ckpts=[str(x) for x in sorted(ck_root.glob("step_*"))]
    fp=read_json(Path(run_cfg.paths.root)/"runs"/"cache"/"dolma_fingerprint.json")
    events=tr.all_events() if tr is not None else []
    blocking=[e for e in events if tr.is_blocking(e,numpy_ok=True)] if tr is not None else []
    ok=bool(int(summary.get("steps",0))>=int(run_cfg.stage0.steps_total_stage0) and len(blocking)==0)
    rep={"ok":ok,"run_id":getattr(tr,"run_id",""),"config_path":config_path,"dolma_fingerprint":fp,"checkpoints_paths":ckpts,"fallbacks_count":len(blocking),"fallbacks_total_count":tr.count_fallbacks() if tr is not None else 0,"fallbacks_blocking_count":len(blocking),"fallbacks":events,"summary":summary,"prep":prep,"verify":verify}
    out=Path(run_cfg.paths.root)/"runs"/"reports"/f"{tr.run_id}_wsd_recipe.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rep)
    if not ok:
        raise RuntimeError("wsd_not_complete")
    return rep

def create_stage0_config(cfg:AppConfig,path:Path,dolma_path:str)->AppConfig:
    out_cfg=clone_with_updates(cfg,{
        "data":{"dolma_path":dolma_path,"tinystories_path":"","max_samples":0,"eval_pct_stage0":0.01,"eval_ratio":0.01},
        "runtime":{"use_dinfer":False,"strict_fallbacks":True,"device":"cuda","blocking_fallback_actions":["synthetic_dolma","dummy_model_fallback","download_false_empty","dataset_empty"],"blocking_fallback_reasons":["dolma_unavailable","dataset_empty"],"fallback_whitelist":["flash_attention_unavailable","numpy_dll_unavailable"]},
        "stage0":{"steps_total_stage0":10000,"lr_stage0":3e-5,"micro_batch_size":1,"grad_accum_steps":32,"seq_len":256,"log_every_steps":10,"eval_every_steps":0,"save_every_steps":200,"keep_last_checkpoints":3,"objective_mode":"llada21_mixture","t2t_enabled":True,"mask_ratio_m2t":0.15,"t2t_edit_ratio":0.10,"m2t_weight":1.0,"t2t_weight":1.0,"warmup_frac":0.2,"stable_frac":0.6,"decay_frac":0.2,"ladder_blocks":[1,4,32,64,256],"decay_blocks":[256,128,64,32],"doc_packing":True,"doc_attention_mask_mode":"composite_llada20"},
        "train":{"dtype":"fp16","batch_size":1,"accum_steps":32,"grad_ckpt":True,"optimizer":"adamw","ckpt_every":200,"eval_every":10001,"log_every_steps":10,"keep_last_checkpoints":3,"data_num_workers":4,"data_prefetch_factor":2,"data_persistent_workers":True,"data_pin_memory":True,"cooldown_every_steps":50,"cooldown_seconds":180},
        "wsd":{"ladder_blocks":[1,4,32,64,256],"decay_blocks":[256,128,64,32],"end_block_size":32,"enforce_divisibility":True},
        "inference":{"max_steps":8,"max_new_tokens":16}
    })
    path.parent.mkdir(parents=True,exist_ok=True)
    save_config(out_cfg,path)
    return out_cfg

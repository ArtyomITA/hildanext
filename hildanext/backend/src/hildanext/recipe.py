# Preflight and overnight recipe runner for LLaDA2.1 on Qwen3-0.6B.
# Main entrypoints: preflight,run_recipe_llada21,dinfer_smoke.
# Provides watchdog,resume and uniform report outputs.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Tuple
import os
import sys
import json
import shutil
import time
import math
import torch
from .config import AppConfig,clone_with_updates
from .io_utils import ensure_dir,write_json,read_jsonl,write_jsonl
from .datasets import prepare_data
from .tokenization import tokenize_all
from .training import run_wsd_conversion,run_sft_training
from .inference import build_engine,load_model_bundle
from .ar import generate_ar
from .trace import use_trace,exception_with_stack


def _disk_free_gb(path:str)->float:
    p=Path(path)
    if not p.exists():
        p=p.parent
    u=shutil.disk_usage(str(p))
    return float(u.free)/1024**3

def _safe_ratio(text:str)->float:
    if not text:
        return 0.0
    ok=0
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in ".,;:!?-'\"()[]{}":
            ok+=1
    return float(ok)/max(1.0,float(len(text)))

def _ensure_sft_from_tiny(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    proc=Path(cfg.paths.processed_dir)
    sft_train=proc/"sft_train.jsonl"
    sft_eval=proc/"sft_eval.jsonl"
    rows=read_jsonl(sft_train)
    if len(rows)>=int(cfg.recipe.min_sft_records):
        return {"wrapped":False,"count":len(rows)}
    train_rows=read_jsonl(proc/"train.jsonl")
    out=[]
    for i,r in enumerate(train_rows):
        txt=str(r.get("text","") or "").strip()
        if not txt:
            continue
        words=txt.split()
        if len(words)<8:
            continue
        topic=" ".join(words[:min(5,len(words))])
        prompt=f"Write a short story about {topic}."
        response=txt
        out.append({"id":f"wrap-{i}","source":"sft_wrap","prompt":prompt,"response":response})
    if not out:
        out=[{"id":f"wrap-{i}","source":"sft_wrap","prompt":f"Write a short story about {i}.","response":f"Story {i} begins, develops, and ends clearly."} for i in range(max(8,int(cfg.recipe.min_sft_records)))]
    n=min(len(out),max(int(cfg.recipe.min_sft_records),32))
    split=max(1,int(0.9*n))
    tr_rows=out[:split]
    ev_rows=out[split:n]
    write_jsonl(sft_train,tr_rows)
    write_jsonl(sft_eval,ev_rows)
    if tr is not None:
        tr.record_fallback(
            event="fallback",
            module="recipe",
            func="_ensure_sft_from_tiny",
            action="dataset_wrap",
            reason="sft_dataset_wrapped_from_tinystories",
            extra_dict={"count_before":len(rows),"count_after":len(tr_rows)}
        )
    return {"wrapped":True,"count":len(tr_rows)}

def _run_stage_with_watchdog(cfg:AppConfig,kind:str,steps:int,trace=None)->Tuple[Dict[str,Any],AppConfig]:
    tr=use_trace(cfg,trace)
    cur=cfg
    for attempt in range(1,4):
        try:
            if kind=="cpt":
                rep=run_wsd_conversion(cur,steps=steps,trace=tr,resume=True,ckpt_every=cur.recipe.ckpt_every,eval_every=cur.recipe.eval_every)
            else:
                rep=run_sft_training(cur,steps=steps,trace=tr,resume=True,ckpt_every=cur.recipe.ckpt_every,eval_every=cur.recipe.eval_every)
            rep["attempt"]=attempt
            rep["seq_len"]=cur.data.seq_len
            rep["accum_steps"]=cur.train.accum_steps
            return rep,cur
        except RuntimeError as e:
            msg=str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                new_seq=max(128,int(cur.data.seq_len)//2)
                new_acc=max(8,int(cur.train.accum_steps)//2)
                cur=clone_with_updates(cur,{"data":{"seq_len":new_seq},"train":{"accum_steps":new_acc}})
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="recipe",
                        func="_run_stage_with_watchdog",
                        action="oom_autodownscale",
                        reason="oom_autodownscale",
                        exception_str=exception_with_stack(e),
                        extra_dict={"kind":kind,"attempt":attempt,"seq_len":new_seq,"accum_steps":new_acc}
                    )
                continue
            if "loss_nan_inf" in msg or "nan" in msg or "inf" in msg:
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="recipe",
                        func="_run_stage_with_watchdog",
                        action="abort",
                        reason="loss_nan_inf",
                        exception_str=exception_with_stack(e),
                        extra_dict={"kind":kind,"attempt":attempt}
                    )
                raise
            raise
        except Exception as e:
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="recipe",
                    func="_run_stage_with_watchdog",
                    action="abort",
                    reason="stage_exception",
                    exception_str=exception_with_stack(e),
                    extra_dict={"kind":kind,"attempt":attempt}
                )
            raise
    raise RuntimeError(f"{kind} stage failed after retries")

def preflight(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    ensure_dir(cfg.paths.logs_dir)
    ensure_dir(Path(cfg.paths.root)/"runs"/"reports")
    env_name=os.environ.get("CONDA_DEFAULT_ENV","")
    is_mdm=env_name.lower()=="mdm" or "\\envs\\mdm\\" in str(sys.executable).replace("/","\\").lower()
    if not is_mdm and tr is not None:
        tr.record_fallback(event="fallback",module="recipe",func="preflight",action="env_check",reason="env_not_mdm",extra_dict={"env":env_name,"python":sys.executable})
    if not torch.cuda.is_available() and tr is not None:
        tr.record_fallback(event="fallback",module="recipe",func="preflight",action="cuda_check",reason="cuda_unavailable",extra_dict={"device_count":torch.cuda.device_count()})
    free_gb=_disk_free_gb(cfg.paths.root)
    if free_gb<float(cfg.runtime.min_disk_free_gb) and tr is not None:
        tr.record_fallback(event="fallback",module="recipe",func="preflight",action="disk_check",reason="low_disk_space",extra_dict={"free_gb":free_gb,"required_gb":cfg.runtime.min_disk_free_gb})
    test_cfg=clone_with_updates(cfg,{"runtime":{"use_dinfer":False,"force_dummy_model":False},"train":{"max_steps":1,"batch_size":1,"accum_steps":1},"data":{"max_samples":min(64,int(cfg.data.max_samples)),"seq_len":min(256,int(cfg.data.seq_len))}})
    bundle=load_model_bundle(test_cfg,for_training=False,trace=tr)
    prep=prepare_data(test_cfg,download=False,max_samples=test_cfg.data.max_samples,trace=tr)
    tok=tokenize_all(test_cfg,max_records=test_cfg.data.max_samples,trace=tr)
    cpt=run_wsd_conversion(test_cfg,steps=1,trace=tr,resume=False,ckpt_every=1,eval_every=1)
    sft=run_sft_training(test_cfg,steps=1,trace=tr,resume=False,ckpt_every=1,eval_every=1)
    ar=generate_ar(test_cfg,prompt="Write one short safe line.",max_new_tokens=16,seed=test_cfg.runtime.seed,trace=tr)
    engine=build_engine(test_cfg,trace=tr)
    dllm_text=engine.generate(prompt="Write one short safe line.",mode="S_MODE",max_new_tokens=16,seed=test_cfg.runtime.seed)
    dllm_stats=dict(engine.last_stats or {})
    engine.close()
    ar_ratio=_safe_ratio(str(ar.get("text","")))
    dllm_ratio=_safe_ratio(str(dllm_text))
    if not bundle.is_dummy and ar_ratio<float(cfg.runtime.gibberish_ratio_warn) and tr is not None:
        tr.record_notice(module="recipe",func="preflight",action="quality_warn",reason="text_gibberish_suspected",extra_dict={"path":"ar","ratio":ar_ratio})
    if not bundle.is_dummy and dllm_ratio<float(cfg.runtime.gibberish_ratio_warn) and tr is not None:
        tr.record_notice(module="recipe",func="preflight",action="quality_warn",reason="text_gibberish_suspected",extra_dict={"path":"dllm","ratio":dllm_ratio})
    peak_mem=None
    if torch.cuda.is_available():
        try:
            peak_mem=float(torch.cuda.max_memory_allocated(0))
        except Exception:
            peak_mem=None
    fallbacks_count=tr.count_fallbacks() if tr is not None else 0
    blocking_fallbacks_count=tr.count_blocking_fallbacks() if tr is not None else 0
    ok=bool(is_mdm and torch.cuda.is_available() and (not bundle.is_dummy) and cpt.get("steps",0)>=1 and sft.get("steps",0)>=1 and str(dllm_text).strip())
    if bool(cfg.runtime.strict_fallbacks) and blocking_fallbacks_count>0:
        ok=False
    report={
        "ok":ok,
        "run_id":tr.run_id if tr is not None else "",
        "env":{"is_mdm":is_mdm,"conda_env":env_name,"python":sys.executable,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count()},
        "env_issues":bundle.env_issues,
        "fallbacks_count":blocking_fallbacks_count,
        "fallbacks_total_count":fallbacks_count,
        "fallbacks_blocking_count":blocking_fallbacks_count,
        "fallbacks":tr.snapshot_fallbacks(limit=256) if tr is not None else [],
        "disk_free_gb":free_gb,
        "peak_mem":peak_mem,
        "dtype":bundle.actual_dtype,
        "engine_used":dllm_stats.get("engine",""),
        "sample_text_ar":ar.get("text",""),
        "sample_text_dllm":dllm_text,
        "sample_decode_stats":dllm_stats,
        "prepare_counts":prep.get("counts",{}),
        "tokenize":{k:v.get("records_out",0) for k,v in tok.items() if isinstance(v,dict)},
        "dry":{"cpt":cpt,"sft":sft},
        "quality":{"ar_ratio":ar_ratio,"dllm_ratio":dllm_ratio}
    }
    out=Path(cfg.paths.root)/"runs"/"reports"/"preflight.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,report)
    return report

def run_recipe_llada21(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    for p in [cfg.paths.raw_dir,cfg.paths.processed_dir,cfg.paths.tokenized_dir,cfg.paths.logs_dir,cfg.paths.checkpoints_dir,Path(cfg.paths.root)/"runs"/"reports"]:
        ensure_dir(p)
    base=clone_with_updates(cfg,{"runtime":{"force_dummy_model":False},"train":{"dtype":"fp16","batch_size":1,"accum_steps":max(16,int(cfg.train.accum_steps)) if torch.cuda.is_available() else int(cfg.train.accum_steps),"grad_ckpt":True},"llada2":{"mask_mode":"composite_llada20"}})
    prep=prepare_data(base,download=False,max_samples=base.data.max_samples,trace=tr)
    tok=tokenize_all(base,max_records=base.data.max_samples,trace=tr)
    sft_wrap={"wrapped":False,"count":0}
    if base.recipe.sft_wrap_from_tiny:
        sft_wrap=_ensure_sft_from_tiny(base,trace=tr)
    stage0_cfg=clone_with_updates(base,{"llada2":{"mask_mode":"composite_llada20"}})
    t0=time.time()
    stage0,used0=_run_stage_with_watchdog(stage0_cfg,"cpt",int(base.recipe.stage0_steps),trace=tr)
    stage1_cfg=clone_with_updates(used0,{"data":{"doc_mask_mode":"simple_blockdiag"}})
    stage1,used1=_run_stage_with_watchdog(stage1_cfg,"sft",int(base.recipe.stage1_steps),trace=tr)
    eng=build_engine(used1,trace=tr)
    samples=[]
    for i,p in enumerate((used1.recipe.eval_prompts or ["Write one sentence about rain.","Q: 5+7? A:","Complete safely: The quick brown fox"])[:3]):
        txt=eng.generate(prompt=p,mode="S_MODE",max_new_tokens=24,seed=used1.runtime.seed+i)
        samples.append({"prompt":p,"text":txt,"stats":dict(eng.last_stats or {})})
    eng.close()
    report={
        "ok":True,
        "run_id":tr.run_id if tr is not None else "",
        "duration_sec":float(max(1e-6,time.time()-t0)),
        "prepare":prep,
        "tokenize":tok,
        "sft_wrap":sft_wrap,
        "stage0":stage0,
        "stage1":stage1,
        "samples":samples,
        "fallbacks_count":tr.count_fallbacks() if tr is not None else 0,
        "fallbacks_blocking_count":tr.count_blocking_fallbacks() if tr is not None else 0,
        "fallbacks":tr.snapshot_fallbacks(limit=512) if tr is not None else []
    }
    out=Path(cfg.paths.root)/"runs"/"reports"/"recipe_llada21.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,report)
    return report

def dinfer_smoke(cfg:AppConfig,trace=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    root=Path(cfg.paths.root)
    vendor=root/"vendor"/"dinfer"/"python"
    if not vendor.exists():
        if tr is not None:
            tr.record_fallback(event="fallback",module="recipe",func="dinfer_smoke",action="skip",reason="dinfer_missing",extra_dict={"engine_requested":"dinfer","path":str(vendor)})
        rep={"ok":False,"skipped":True,"reason":"dinfer_missing","fallbacks":tr.snapshot_fallbacks(limit=64) if tr is not None else []}
        write_json(root/"runs"/"reports"/"dinfer_smoke.json",rep)
        return rep
    sys.path.insert(0,str(vendor))
    try:
        from dinfer.decoding.parallel_strategy import get_transfer_index_threshold
    except Exception as e:
        if tr is not None:
            tr.record_fallback(event="fallback",module="recipe",func="dinfer_smoke",action="skip",reason="dinfer_missing",exception_str=exception_with_stack(e),extra_dict={"engine_requested":"dinfer"})
        rep={"ok":False,"skipped":True,"reason":"dinfer_import_failed","fallbacks":tr.snapshot_fallbacks(limit=64) if tr is not None else []}
        write_json(root/"runs"/"reports"/"dinfer_smoke.json",rep)
        return rep
    seed_every=11
    torch.manual_seed(seed_every)
    b,l,v=1,12,64
    mask_id=63
    x=torch.randint(0,v-1,(b,l),dtype=torch.long)
    mask_index=torch.zeros((b,l),dtype=torch.bool)
    mask_index[:,3:6]=True
    mask_index[:,8:11]=True
    x[mask_index]=mask_id
    logits=torch.randn(b,l,v,dtype=torch.float32)
    from .formulas import llada21_apply
    x0,transfer=get_transfer_index_threshold(logits=logits,temperature=0.0,remasking='low_confidence',mask_index=mask_index,x=x.clone(),num_transfer_tokens=None,mask_id=mask_id,threshold=0.0)
    p=torch.softmax(logits.to(torch.float32),dim=-1)
    pred=torch.argmax(logits,dim=-1)
    conf=torch.gather(p,dim=-1,index=pred.unsqueeze(-1)).squeeze(-1)
    local_updated,sets=llada21_apply(tokens=x,pred_ids=pred,confidence=conf,mask_id=mask_id,tau_mask=0.0,tau_edit=2.0)
    dinfer_updated=torch.where(transfer,x0,x)
    ok=bool(torch.equal(local_updated,dinfer_updated) and int(sets.gamma_count)==int(transfer.sum().item()))
    rep={"ok":ok,"skipped":False,"local_gamma":int(sets.gamma_count),"dinfer_transfer":int(transfer.sum().item()),"fallbacks":tr.snapshot_fallbacks(limit=64) if tr is not None else []}
    write_json(root/"runs"/"reports"/"dinfer_smoke.json",rep)
    return rep

# FastAPI server for health, generation, and job endpoints.
# Main entrypoints: create_app,run_server.
# Engine is shared and supports dInfer fallback logic.
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any,Dict,Optional,List,Callable,Tuple,Literal,Iterator
import json as _json
import os
import re as _re
import math as _math
import queue as _queue
from decimal import Decimal as _Decimal, InvalidOperation as _InvalidOperation
import subprocess
import sys
import threading
import time
import uuid
import gc
from collections import deque
from datetime import datetime as _datetime
import torch
from fastapi import FastAPI,HTTPException,Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from .config import AppConfig,load_config
from .inference import build_engine
from .stage0_benchmarks import load_hellaswag_items,load_mmlu_pro_items,load_gsm8k_items
from .trace import trace_from_cfg,set_active_trace,reset_active_trace,use_trace,exception_with_stack

class _LazyEngine:
    """Wraps build_engine() with lazy loading.

    The real engine (and its model weights) are loaded only on the first call
    to .generate() or .bundle.  WSD, run-control, and health endpoints never
    trigger a model load.
    """
    def __init__(self,cfg:AppConfig,trace)->None:
        self._cfg=cfg
        self._trace=trace
        self._engine=None
        self._lock=threading.Lock()
        self._loaded=False

    def _ensure(self):
        """Load the real engine if not yet loaded (thread-safe)."""
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self._engine=build_engine(self._cfg,trace=self._trace)
                self._loaded=True

    # ---- proxied properties / methods --------------------------------
    @property
    def name(self)->str:
        return self._engine.name if self._loaded else "lazy(not loaded)"

    @property
    def bundle(self):
        self._ensure()
        return getattr(self._engine,"bundle",None)

    @property
    def last_stats(self)->dict:
        return getattr(self._engine,"last_stats",{}) if self._loaded else {}

    def generate(self,**kwargs)->str:
        self._ensure()
        return self._engine.generate(**kwargs)

    def close(self)->None:
        if self._loaded and self._engine is not None:
            self._engine.close()
        self._engine=None
        self._loaded=False
        try:
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

class GenerateRequest(BaseModel):
    prompt:str
    mode:str="S_MODE"
    tau_mask:Optional[float]=None
    tau_edit:Optional[float]=None
    max_new_tokens:int=Field(default=256,ge=1,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="medium",description="instant|low|medium|high|adaptive — controls decode steps and tau scaling")
    temperature:Optional[float]=Field(default=None,ge=0.0,le=2.0)
    top_p:Optional[float]=Field(default=None,ge=0.0,le=1.0)
    top_k:Optional[int]=Field(default=None,ge=0,le=2000)
    presence_penalty:Optional[float]=Field(default=None,ge=0.0,le=5.0)
    repetition_penalty:Optional[float]=Field(default=None,ge=0.0,le=5.0)
    system_prompt:Optional[str]=None
    messages:Optional[List[Dict[str,str]]]=None
    enable_thinking:Optional[bool]=None

class ArGenerateRequest(BaseModel):
    prompt:str
    max_new_tokens:int=Field(default=256,ge=1,le=512)
    seed:Optional[int]=None
    decode_mode:Literal["greedy","sampling"]="greedy"
    temperature:float=Field(default=0.7,ge=0.0,le=2.0)
    top_p:float=Field(default=0.9,ge=0.0,le=1.0)
    top_k:int=Field(default=20,ge=0,le=2000)
    presence_penalty:float=Field(default=0.0,ge=0.0,le=5.0)
    repetition_penalty:float=Field(default=1.0,ge=0.0,le=5.0)
    system_prompt:Optional[str]=None
    messages:Optional[List[Dict[str,str]]]=None
    enable_thinking:Optional[bool]=None


class LoadWeightsRequest(BaseModel):
    scope:str=Field(default="BOTH",description="AR | DLLM | BOTH")
    prompt:str=Field(default="__hildanext_load_weights__")
    mode:str=Field(default="S_MODE")
    max_new_tokens:int=Field(default=1,ge=1,le=8)
    seed:Optional[int]=None
    effort:str=Field(default="instant",description="instant|low|medium|high|adaptive")


class HellaSwagItemRequest(BaseModel):
    stem:str
    endings:List[str]=Field(min_length=4,max_length=4)
    label_target:Optional[int]=Field(default=None,ge=0,le=3)
    scope:Literal["AR","DLLM","BOTH","RCD","OTS"]="DLLM"
    context_window:Optional[int]=Field(default=None,ge=256,le=8192)
    decode_strategy:Literal["greedy","sampling"]="greedy"
    temperature:Optional[float]=Field(default=None,ge=0.0,le=2.0)
    top_p:Optional[float]=Field(default=None,ge=0.0,le=1.0)
    n_shots:int=Field(default=0,ge=0,le=5)
    mode:str=Field(default="Q_MODE")
    max_new_tokens:int=Field(default=256,ge=1,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="low")
    detailed_log_token:Optional[str]=Field(default=None,max_length=64)

class MmluProItemRequest(BaseModel):
    question:str
    options:List[str]=Field(min_length=2,max_length=10)
    answer_label:str=Field(min_length=1,max_length=1)
    scope:Literal["AR","DLLM","BOTH","RCD","OTS"]="DLLM"
    context_window:Optional[int]=Field(default=None,ge=256,le=8192)
    decode_strategy:Literal["greedy","sampling"]="greedy"
    temperature:Optional[float]=Field(default=None,ge=0.0,le=2.0)
    top_p:Optional[float]=Field(default=None,ge=0.0,le=1.0)
    n_shots:int=Field(default=0,ge=0,le=5)
    force_cot:bool=True
    mode:str=Field(default="S_MODE")
    max_new_tokens:int=Field(default=1024,ge=16,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="medium")
    detailed_log_token:Optional[str]=Field(default=None,max_length=64)


class Gsm8kItemRequest(BaseModel):
    question:str
    answer_target:str
    scope:Literal["AR","DLLM","BOTH","RCD","OTS"]="DLLM"
    context_window:Optional[int]=Field(default=None,ge=256,le=8192)
    decode_strategy:Literal["greedy","sampling"]="greedy"
    temperature:Optional[float]=Field(default=None,ge=0.0,le=2.0)
    top_p:Optional[float]=Field(default=None,ge=0.0,le=1.0)
    n_shots:int=Field(default=0,ge=0,le=8)
    target_format:Literal["hash","boxed"]="hash"
    mode:str=Field(default="S_MODE")
    max_new_tokens:int=Field(default=1024,ge=16,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="medium")
    detailed_log_token:Optional[str]=Field(default=None,max_length=64)


class Stage0StabilityRequest(BaseModel):
    prompt:str=Field(default="The capital of France is")
    scope:Literal["AR","DLLM","BOTH","RCD","OTS"]="DLLM"
    context_window:Optional[int]=Field(default=None,ge=256,le=8192)
    decode_strategy:Literal["greedy","sampling"]="greedy"
    temperature:Optional[float]=Field(default=None,ge=0.0,le=2.0)
    top_p:Optional[float]=Field(default=None,ge=0.0,le=1.0)
    total_steps:int=Field(default=50,ge=10,le=100)
    mask_schedule:Literal["linear","cosine"]="cosine"
    mode:str=Field(default="S_MODE")
    max_new_tokens:int=Field(default=1024,ge=8,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="medium")
    detailed_log_token:Optional[str]=Field(default=None,max_length=64)


class Stage0DetailedLogStartRequest(BaseModel):
    benchmark:Literal["hellaswag","mmlu-pro","gsm8k","stability"]
    scope:Literal["AR","DLLM","BOTH","RCD","OTS"]="DLLM"
    context_window:Optional[int]=Field(default=None,ge=256,le=8192)
    decode_strategy:Literal["greedy","sampling"]="greedy"
    effort:str=Field(default="medium")
    max_new_tokens:int=Field(default=1024,ge=1,le=4096)
    run_label:Optional[str]=Field(default=None,max_length=80)


class Stage0DetailedLogFinishRequest(BaseModel):
    token:str=Field(min_length=8,max_length=64)
    status:Literal["completed","stopped","error"]="completed"
    summary:Optional[Dict[str,Any]]=None

class GenerateResponse(BaseModel):
    text:str
    stats:Dict[str,Any]
    engine:str

class JobResponse(BaseModel):
    job_id:str
    status:str
    result:Optional[GenerateResponse]=None
    error:Optional[str]=None

# ---- /frontend/wsd helpers ----------------------------------------

def _read_jsonl_tail(path:Path,n:int=2000)->List[dict]:
    """Read last *n* JSONL lines, skip malformed rows. Returns [] if file missing."""
    if not path.exists():
        return []
    rows:List[dict]=[]
    try:
        with path.open("r",encoding="utf-8",errors="replace") as f:
            lines=f.readlines()
        for line in lines[-n:]:
            line=line.strip()
            if not line:
                continue
            try:
                rows.append(_json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return rows

def _map_metric_row(r:dict)->dict:
    """cpt.jsonl row → frontend WsdMetricRow (camelCase)."""
    return {
        "kind":r.get("kind","cpt"),
        "step":int(r.get("step",0)),
        "phase":r.get("phase","warmup"),
        "blockSize":int(r.get("block_size",1)),
        "loss":float(r.get("loss",0)),
        "lossM2T":float(r.get("loss_m2t",0)),
        "lossT2T":float(r.get("loss_t2t",0)),
        "maskedTokenAcc":r.get("masked_token_acc"),
        "lr":float(r.get("lr",0)),
        "gradNorm":float(r.get("grad_norm",0)),
        "tokensPerSec":float(r.get("tokens_per_sec",0)),
        "stepTimeS":float(r.get("step_time_s",0)),
        "vramAllocMb":float(r.get("vram_alloc_mb",0)),
        "vramReservedMb":float(r.get("vram_reserved_mb",0)),
        "vramPeakMb":float(r.get("vram_peak_mb",0)),
        "etaStageSec":float(r.get("eta_stage_sec",0)),
        "tSampled":float(r.get("t_sampled",0)),
        "tMean":float(r.get("t_mean",0)),
        "tMin":float(r.get("t_min",0)),
        "tMax":float(r.get("t_max",1)),
        "maskRatioActual":float(r.get("mask_ratio_actual",0)),
        "predPositionsCount":int(r.get("pred_positions_count",0)),
        "wsdPhaseProgress":float(r.get("wsd_phase_progress",0)),
        "bidirectional":bool(r.get("bidirectional",False)),
        "isCausalEffective":bool(r.get("is_causal_effective",True)),
        "attentionMode":str(r.get("attention_mode","")),
        "shiftMode":str(r.get("shift_mode","")),
        "timeParam":str(r.get("time_param","")),
        "lossWeighting":str(r.get("loss_weighting","")),
        "lossByTBucket":dict(r.get("loss_by_t_bucket") or {}),
        "accMaskedByTBucket":dict(r.get("acc_masked_by_t_bucket") or {}),
    }

def _map_fallback_log(entry:dict,idx:int)->dict:
    """fallbacks.jsonl row → NormalizedLogEntry."""
    ev=entry.get("event_type","notice")
    level_map={"error":"error","warning":"warning","notice":"notice","critical":"error","info":"info"}
    level=level_map.get(ev,"info")
    action=entry.get("action","event")
    reason=entry.get("reason","")
    msg=f"{action}: {reason}".rstrip(": ")
    return {
        "id":f"fb-{idx}",
        "tsUtc":entry.get("ts_utc",""),
        "source":"fallback",
        "level":level,
        "module":entry.get("module"),
        "func":entry.get("func"),
        "eventType":ev,
        "action":action,
        "reason":reason or None,
        "message":msg,
        "extra":entry.get("extra") or None,
        "tags":["fallback"],
    }

def _map_run_log_line(line:str,idx:int)->Optional[dict]:
    """cpt_run.log line → NormalizedLogEntry, or None if unparseable."""
    m=_re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)",line.strip())
    if not m:
        return None
    ts_str,msg=m.group(1),m.group(2)
    try:
        ts_utc=_datetime.strptime(ts_str,"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts_utc=ts_str
    upmsg=msg.upper()
    level=("error" if any(k in upmsg for k in ("ERROR","FAIL","OOM","NAN","INF"))
           else "warning" if any(k in upmsg for k in ("WARN","SKIP","FALLBACK"))
           else "info")
    return {"id":f"rl-{idx}","tsUtc":ts_utc,"source":"console","level":level,"message":msg,"tags":["run_log"]}

def _build_wsd_meta(cfg:Any,latest:Optional[dict],wsd_cfg_override:Any=None)->dict:
    """Build WsdMeta dict.

    If `wsd_cfg_override` is provided (e.g. loaded from llada21_dolma_wsd_only.json)
    its wsd/train sections take priority over cfg for schedule figures.
    """
    exp=getattr(cfg,"experiment",None)
    run_id=getattr(exp,"experiment_id","unknown") if exp else "unknown"
    notes=(getattr(exp,"notes","") or "")[:40]
    # Prefer PS1-generated WSD config for schedule fields if available.
    src=wsd_cfg_override if wsd_cfg_override is not None else cfg
    wsd_section=getattr(src,"wsd",None)
    train_section=getattr(src,"train",None)
    stage0_section=getattr(src,"stage0",None)
    # ladder_blocks: prefer stage0, then wsd, then train
    ladder:list
    if stage0_section and getattr(stage0_section,"ladder_blocks",None):
        ladder=list(stage0_section.ladder_blocks)
    elif wsd_section and getattr(wsd_section,"ladder_blocks",None):
        ladder=list(wsd_section.ladder_blocks)
    elif train_section and getattr(train_section,"ladder_blocks",None):
        ladder=list(train_section.ladder_blocks)
    else:
        ladder=[1]
    # warmup/stable/decay steps — prefer stage0 fractional schedule when stage0 is enabled.
    def _steps(section,field,fallback):
        v=getattr(section,field,None) if section else None
        return int(v) if v is not None else fallback
    stage0_enabled=bool(getattr(stage0_section,"enabled",False)) if stage0_section else False
    if stage0_enabled and stage0_section:
        s0_total=int(getattr(stage0_section,"steps_total_stage0",0))
        if s0_total>0:
            warmup=int(s0_total*float(getattr(stage0_section,"warmup_frac",0.1)))
            stable=int(s0_total*float(getattr(stage0_section,"stable_frac",0.7)))
            decay=s0_total-warmup-stable
            total=s0_total
        else:
            warmup=_steps(wsd_section,"warmup_steps",100)
            stable=_steps(wsd_section,"stable_steps",300)
            decay=_steps(wsd_section,"decay_steps",100)
            total=warmup+stable+decay
    else:
        warmup=_steps(wsd_section,"warmup_steps",_steps(getattr(src,"train",None),"warmup_steps",100))
        stable=_steps(wsd_section,"stable_steps",300)
        decay=_steps(wsd_section,"decay_steps",100)
        total=warmup+stable+decay
    try:
        import torch as _torch
        device="cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        device="cpu"
    return {
        "runId":run_id,
        "configDigest":notes if notes else "n/a",
        "optimizer":cfg.train.optimizer,
        "dtype":cfg.train.dtype,
        "device":device,
        "dummyModel":False,
        "phase":latest.get("phase","warmup") if latest else "warmup",
        "blockSize":int(latest.get("block_size",1)) if latest else 1,
        "ladderBlocks":ladder,
        "warmupSteps":warmup,
        "stableSteps":stable,
        "decaySteps":decay,
        "totalSteps":total,
    }

def _build_wsd_insights(raw_rows:List[dict],latest:Optional[dict],run_lines:Optional[List[str]]=None)->List[dict]:
    out:List[dict]=[]
    # Surface errors from the RunManager subprocess first.
    if run_lines:
        joined=" ".join(run_lines[-30:]).upper()
        if "CUDA_UNAVAILABLE" in joined or "CPU_FALLBACK" in joined:
            out.append({"id":"cuda-unavail","title":"CUDA unavailable in API process","metric":"device",
                        "body":"The FE-launched run used the API server's Python (base conda env) which lacks torch+cuda. "
                               "Use start_wsd_full_logs.ps1 to run via the mdm env, or start the server with mdm python.",
                        "tone":"critical"})
        if "STRICT_FALLBACKS VIOLATION" in joined or "RUNTIMEERROR" in joined:
            # Extract the last RuntimeError line for the card body.
            last_err=next((l for l in reversed(run_lines) if "RuntimeError" in l or "strict_fallbacks" in l),"")
            out.append({"id":"strict-fallback","title":"strict_fallbacks violation","metric":"runtime",
                        "body":last_err[:200] or "Run crashed due to strict_fallbacks. Check run log.",
                        "tone":"critical"})
    if not raw_rows:
        out.append({"id":"no-run","title":"No training data yet","metric":"step",
                    "body":"WSD training has not started yet. Run start_wsd_full_logs.ps1.",
                    "tone":"info"})
        return out
    nan_count=sum(1 for r in raw_rows if r.get("nan_inf_detected",False))
    if nan_count>0:
        out.append({"id":"nan-detected","title":f"NaN/Inf: {nan_count} steps","metric":"loss",
                    "body":f"{nan_count} steps had NaN/Inf loss. Inspect grad_clip and LR schedule.",
                    "tone":"critical"})
    if latest:
        if latest.get("phase")=="stable" and not latest.get("bidirectional",False):
            out.append({"id":"causal-stable","title":"Stable: attention still causal","metric":"attentionMode",
                        "body":"Stable phase running with causal attention — bidirectional preflight may have failed.",
                        "tone":"warning"})
        if float(latest.get("vram_peak_mb",0))>7200:
            out.append({"id":"vram-high","title":"VRAM near limit","metric":"vramPeakMb",
                        "body":f"Peak VRAM {latest.get('vram_peak_mb',0):.0f} MB — GTX 1080 limit is 8192 MB.",
                        "tone":"warning"})
        phase=latest.get("phase","warmup")
        progress=float(latest.get("wsd_phase_progress",0))
        bidir=bool(latest.get("bidirectional",False))
        out.append({"id":"phase-progress","title":f"{phase.capitalize()} — {progress*100:.0f}%","metric":"phase",
                    "body":f"Phase '{phase}' at {progress*100:.1f}% · block={latest.get('block_size',1)} · bidirectional={bidir}",
                    "tone":"info"})
    return out

# ---- end /frontend/wsd helpers ------------------------------------

_CHOICE_RE=_re.compile(r"\b([ABCD])\b",_re.IGNORECASE)
_DIGIT_CHOICE_RE=_re.compile(r"\b([1-4])\b")
MMLU_PRO_SYSTEM_PROMPT=(
    "You are an expert scholar. Please answer the following multiple-choice question.\n"
    "First, think step-by-step to reach the correct conclusion.\n"
    "Then, you MUST end your response by explicitly stating the final answer in this exact format: "
    "'The answer is (X)', where X is the correct letter from the provided options."
)
GSM8K_SYSTEM_PROMPT=(
    "You are a highly accurate mathematical assistant. Solve the following math problem step-by-step.\n"
    "After you have completed your reasoning, you MUST provide the final numerical answer on a new line "
    "in the exact format: '#### [Final Number]'.\n"
    "Do not include units or text in the final number, only the digits."
)

_MMLU_FINAL_RE=_re.compile(r"The answer is \(([A-J])\)")
_MMLU_PAREN_LETTER_RE=_re.compile(r"\(([A-J])\)")
_GSM_HASH_RE=_re.compile(r"####\s*(-?[\d\.,]+)")
_GSM_BOXED_RE=_re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")
_GSM_NUM_RE=_re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_SAFE_FILE_RE=_re.compile(r"[^A-Za-z0-9._-]+")
_THINK_TAG_RE=_re.compile(r"<think>([\s\S]*?)</think>",_re.IGNORECASE)
_QWEN_THINK_TAG_RE=_re.compile(r"<\|begin_of_thought\|>([\s\S]*?)<\|end_of_thought\|>",_re.IGNORECASE)

class _InferenceEventBus:
    def __init__(self,maxlen:int=3000)->None:
        self._events:deque=deque(maxlen=max(200,int(maxlen)))
        self._subs:List[_queue.Queue]=[]
        self._next_id:int=1
        self._lock=threading.Lock()

    def _to_int_id(self,value:Any)->int:
        try:
            return max(0,int(str(value or "").strip()))
        except Exception:
            return 0

    def emit(self,payload:Dict[str,Any])->Dict[str,Any]:
        row=dict(payload or {})
        with self._lock:
            row["id"]=str(self._next_id)
            self._next_id+=1
            self._events.append(row)
            dead:List[_queue.Queue]=[]
            for q in list(self._subs):
                try:
                    q.put_nowait(dict(row))
                except _queue.Full:
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                    try:
                        q.put_nowait(dict(row))
                    except Exception:
                        dead.append(q)
                except Exception:
                    dead.append(q)
            if dead:
                self._subs=[q for q in self._subs if q not in dead]
        return row

    def snapshot(self,tail:int=200,after_id:Optional[str]=None)->List[Dict[str,Any]]:
        n=max(1,min(5000,int(tail or 200)))
        after_n=self._to_int_id(after_id)
        with self._lock:
            rows=[dict(x) for x in list(self._events)]
        if after_n>0:
            rows=[x for x in rows if self._to_int_id(x.get("id"))>after_n]
        if len(rows)>n:
            rows=rows[-n:]
        return rows

    def subscribe(self,after_id:Optional[str]=None,tail:int=0)->_queue.Queue:
        q:_queue.Queue=_queue.Queue(maxsize=512)
        after_n=self._to_int_id(after_id)
        with self._lock:
            preload=[dict(x) for x in list(self._events)]
            if after_n>0:
                preload=[x for x in preload if self._to_int_id(x.get("id"))>after_n]
            elif int(tail or 0)>0 and len(preload)>int(tail):
                preload=preload[-int(tail):]
            self._subs.append(q)
        for row in preload:
            try:
                q.put_nowait(row)
            except Exception:
                break
        return q

    def unsubscribe(self,q:_queue.Queue)->None:
        with self._lock:
            self._subs=[x for x in self._subs if x is not q]


_ACTIVE_INFER_BUS:Optional[_InferenceEventBus]=None
_ACTIVE_INFER_BUS_LOCK=threading.Lock()


def _set_active_infer_bus(bus:Optional[_InferenceEventBus])->None:
    global _ACTIVE_INFER_BUS
    with _ACTIVE_INFER_BUS_LOCK:
        _ACTIVE_INFER_BUS=bus


def _get_active_infer_bus()->Optional[_InferenceEventBus]:
    with _ACTIVE_INFER_BUS_LOCK:
        return _ACTIVE_INFER_BUS


def _infer_benchmark_from_event(event:str)->Optional[str]:
    up=str(event or "").upper()
    if "HELLASWAG" in up:
        return "hellaswag"
    if "MMLU" in up:
        return "mmlu-pro"
    if "GSM8K" in up:
        return "gsm8k"
    if "STABILITY" in up:
        return "stability"
    return None


def _server_log(
    event:str,
    msg:str,
    *,
    level:str="info",
    source:str="inference",
    lane:Optional[str]=None,
    scope:Optional[str]=None,
    benchmark:Optional[str]=None,
    meta:Optional[Dict[str,Any]]=None,
)->None:
    ts=_datetime.utcnow().isoformat()+"Z"
    print(f"{ts} {event} {msg}",flush=True)
    bus=_get_active_infer_bus()
    if bus is None:
        return
    bench=benchmark or _infer_benchmark_from_event(event)
    bus.emit(
        {
            "tsUtc":ts,
            "level":str(level or "info"),
            "source":str(source or "inference"),
            "event":str(event or ""),
            "lane":str(lane) if lane else None,
            "scope":str(scope) if scope else None,
            "benchmark":bench,
            "message":str(msg or ""),
            "meta":dict(meta or {}),
        }
    )


def _resolve_scope(scope:str)->Tuple[str,bool,bool]:
    s=str(scope or "DLLM").strip().upper()
    want_ar=s in {"AR","BOTH"}
    want_dllm=s in {"DLLM","BOTH","RCD","OTS"}
    if not (want_ar or want_dllm):
        raise HTTPException(status_code=400,detail="scope must be AR, DLLM, BOTH, RCD, or OTS")
    return s,want_ar,want_dllm


def _decode_params(
    decode_strategy:str,
    temperature:Optional[float]=None,
    top_p:Optional[float]=None,
)->Tuple[Literal["greedy","sampling"],float,float]:
    mode="sampling" if str(decode_strategy or "greedy").strip().lower()=="sampling" else "greedy"
    if mode=="greedy":
        return "greedy",0.0,1.0
    t=0.6 if temperature is None else float(temperature)
    p=0.9 if top_p is None else float(top_p)
    return "sampling",max(0.0,min(2.0,t)),max(0.0,min(1.0,p))


def _truncate_prompt_to_context(prompt:str,tokenizer:Any,context_window:Optional[int])->str:
    base=str(prompt or "")
    if tokenizer is None or context_window is None:
        return base
    try:
        max_ctx=max(32,int(context_window))
    except Exception:
        return base
    if max_ctx<=0:
        return base
    try:
        token_ids=tokenizer.encode(base,add_special_tokens=False)
    except TypeError:
        try:
            token_ids=tokenizer.encode(base)
        except Exception:
            return base
    except Exception:
        return base
    if not isinstance(token_ids,list) or len(token_ids)<=max_ctx:
        return base
    tail=token_ids[-max_ctx:]
    try:
        out=tokenizer.decode(tail,skip_special_tokens=False)
    except TypeError:
        out=tokenizer.decode(tail)
    except Exception:
        return base
    out_text=str(out or "").strip()
    return out_text if out_text else base


def _normalize_space(input_text:str)->str:
    return str(input_text or "").replace("\r"," ").replace("\n"," ").strip()


def _preview_text(input_text:str,max_chars:int)->str:
    base=_normalize_space(input_text)
    n=max(16,int(max_chars or 140))
    if len(base)<=n:
        return base
    return f"{base[: max(1,n-1)]}…"


def _split_thinking_output(text:str)->Tuple[str,str]:
    raw=str(text or "")
    chunks=[str(x or "").strip() for x in _THINK_TAG_RE.findall(raw)]
    if not chunks:
        chunks=[str(x or "").strip() for x in _QWEN_THINK_TAG_RE.findall(raw)]
        answer=_QWEN_THINK_TAG_RE.sub("",raw).strip()
    else:
        answer=_THINK_TAG_RE.sub("",raw).strip()
    thinking="\n\n".join([x for x in chunks if x]).strip()
    return thinking,answer if answer else raw.strip()


def _with_few_shot_chat(user_prompt:str,shots:List[Tuple[str,str]])->str:
    if not shots:
        return str(user_prompt).strip()
    lines:List[str]=[]
    for u,a in shots:
        lines.append(f"User: {str(u).strip()}")
        lines.append(f"Assistant: {str(a).strip()}")
        lines.append("")
    lines.append(f"User: {str(user_prompt).strip()}")
    lines.append("Assistant:")
    return "\n".join(lines).strip()


def _build_hellaswag_user_block(stem:str,endings:List[str])->str:
    labels=["A","B","C","D"]
    lines=[
        "Task: choose the best ending for the context.",
        f"Context: {str(stem).strip()}",
        "Options:",
    ]
    for i,end in enumerate((endings or [])[:4]):
        lines.append(f"{labels[i]}) {str(end).strip()}")
    lines.append("Answer with one letter: A, B, C, or D.")
    return "\n".join(lines)


def _build_hellaswag_prompt(
    stem:str,
    endings:List[str],
    few_shot_items:Optional[List[Dict[str,Any]]]=None,
)->str:
    shots:List[Tuple[str,str]]=[]
    for row in few_shot_items or []:
        if not isinstance(row,dict):
            continue
        q=_build_hellaswag_user_block(str(row.get("stem","")),list(row.get("endings") or []))
        lab=int(row.get("label",0))
        lab=max(0,min(3,lab))
        a=["A","B","C","D"][lab]
        shots.append((q,a))
    return _with_few_shot_chat(_build_hellaswag_user_block(stem,endings),shots)


def _build_mmlu_user_block(question:str,options:List[str],force_cot:bool)->str:
    labels="ABCDEFGHIJ"
    _=force_cot
    opts=[f"{labels[i]}) {str(opt).strip()}" for i,opt in enumerate((options or [])[:10])]
    options_block="\n".join(opts)
    lines=[
        f"Question: {str(question).strip()}",
        "Options:",
        options_block,
        "Answer:",
    ]
    return "\n".join(lines)


def _build_mmlu_system_prompt(force_cot:bool)->str:
    if force_cot:
        return MMLU_PRO_SYSTEM_PROMPT
    return "\n".join([
        "You are an expert scholar. Please answer the following multiple-choice question.",
        "You MUST end your response by explicitly stating the final answer in this exact format: "
        "'The answer is (X)', where X is the correct letter from the provided options.",
    ])


def _build_mmlu_pro_prompt(
    question:str,
    options:List[str],
    force_cot:bool=True,
    few_shot_items:Optional[List[Dict[str,Any]]]=None,
)->str:
    shots:List[Tuple[str,str]]=[]
    for row in few_shot_items or []:
        if not isinstance(row,dict):
            continue
        q=_build_mmlu_user_block(str(row.get("question","")),list(row.get("options") or []),force_cot=force_cot)
        ans=str(row.get("answer_label","")).strip().upper()
        if not _re.match(r"^[A-J]$",ans):
            continue
        shots.append((q,f"The answer is ({ans})"))
    target=_build_mmlu_user_block(question,options,force_cot=force_cot)
    return _with_few_shot_chat(target,shots)


def _format_gsm_answer(number_text:str,target_format:str)->str:
    n=_canonical_number(number_text)
    if n is None:
        n="0"
    if str(target_format or "hash").lower()=="boxed":
        return f"\\boxed{{{n}}}"
    return f"#### {n}"


def _build_gsm_user_block(question:str,target_format:str)->str:
    _=target_format
    return "\n".join([
        f"Problem: {str(question).strip()}",
        "Solution:",
    ])


def _build_gsm8k_prompt(
    question:str,
    target_format:str="hash",
    few_shot_items:Optional[List[Dict[str,Any]]]=None,
)->str:
    shots:List[Tuple[str,str]]=[]
    for row in few_shot_items or []:
        if not isinstance(row,dict):
            continue
        q=_build_gsm_user_block(str(row.get("question","")),target_format=target_format)
        ans_raw=str(row.get("answer_target",""))
        ans_num=_extract_gsm_number(ans_raw,target_format=target_format)
        if ans_num is None:
            continue
        shots.append((q,_format_gsm_answer(ans_num,target_format)))
    target=_build_gsm_user_block(question,target_format=target_format)
    return _with_few_shot_chat(target,shots)


def _parse_choice_idx(text:str)->Optional[int]:
    s=str(text or "").strip().upper()
    m=_CHOICE_RE.search(s)
    if m:
        return {"A":0,"B":1,"C":2,"D":3}.get(m.group(1))
    m=_DIGIT_CHOICE_RE.search(s)
    if m:
        try:
            return int(m.group(1))-1
        except Exception:
            return None
    return None


def _parse_mmlu_answer_label(text:str)->Optional[str]:
    s=str(text or "")
    m=_MMLU_FINAL_RE.search(s)
    if m:
        return m.group(1).upper()
    hits=_MMLU_PAREN_LETTER_RE.findall(s)
    if hits:
        return str(hits[-1]).upper()
    return None


def _canonical_number(text:str)->Optional[str]:
    raw=str(text or "").strip().replace(",","")
    if not raw:
        return None
    try:
        val=_Decimal(raw)
    except _InvalidOperation:
        return None
    if not val.is_finite():
        return None
    out=format(val.normalize(),"f")
    if "." in out:
        out=out.rstrip("0").rstrip(".")
    if out in {"-0","+0",""}:
        out="0"
    return out


def _extract_gsm_number(text:str,target_format:str="hash")->Optional[str]:
    s=str(text or "")
    fmt=str(target_format or "hash").lower()
    m=_GSM_BOXED_RE.search(s) if fmt=="boxed" else _GSM_HASH_RE.search(s)
    if m:
        hit=_canonical_number(m.group(1))
        if hit is not None:
            return hit
    return None


def _to_float(text:Optional[str])->Optional[float]:
    if text is None:
        return None
    raw=str(text).strip().replace(",","")
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _safe_file_part(value:str,fallback:str,max_len:int=48)->str:
    base=_SAFE_FILE_RE.sub("-",str(value or "").strip())
    base=base.strip("._-")
    if not base:
        return fallback
    return base[:max(1,int(max_len))]


def _compact_stats(stats:Any)->Dict[str,Any]:
    if not isinstance(stats,dict):
        return {}
    out:Dict[str,Any]={}
    for k in [
        "tokens_generated",
        "steps",
        "max_steps",
        "finish_reason",
        "truncated",
        "queue_wait_ms",
        "serialized_inference",
        "mode",
        "effort",
        "decode_mode",
        "temperature",
        "top_p",
    ]:
        if k in stats:
            out[k]=stats.get(k)
    return out


def _stable_confidence_points(raw_logs:Any)->List[dict]:
    rows=raw_logs if isinstance(raw_logs,list) else []
    out:List[dict]=[]
    for row in rows:
        if not isinstance(row,dict):
            continue
        try:
            step=int(row.get("step",0))
        except Exception:
            continue
        c=row.get("avg_conf_tokens")
        if c is None:
            c=row.get("avg_conf_masked")
        try:
            conf=float(c) if c is not None else 0.0
        except Exception:
            conf=0.0
        conf=max(0.0,min(1.0,conf))
        try:
            mr=float(row.get("mask_ratio",1.0))
        except Exception:
            mr=1.0
        out.append({
            "step":step,
            "mask_ratio":max(0.0,min(1.0,mr)),
            "mean_confidence":conf,
        })
    return out


def _schedule_mask_ratio(step_idx:int,total_steps:int,mask_schedule:str)->float:
    if total_steps<=1:
        return 0.0
    t=max(0.0,min(1.0,float(step_idx)/float(max(1,total_steps-1))))
    sched=str(mask_schedule or "cosine").strip().lower()
    if sched=="linear":
        return max(0.0,1.0-t)
    return max(0.0,min(1.0,float(_math.cos((t*_math.pi)/2.0)**2)))


def _resample_stability_points(points:List[dict],total_steps:int,mask_schedule:str)->List[dict]:
    tgt=max(1,int(total_steps))
    if not points:
        return [
            {
                "step":i+1,
                "mask_ratio":_schedule_mask_ratio(i,tgt,mask_schedule),
                "mean_confidence":0.0,
            }
            for i in range(tgt)
        ]
    src=list(points)
    if len(src)==tgt:
        out:List[dict]=[]
        for i,row in enumerate(src):
            out.append({
                "step":i+1,
                "mean_confidence":float(row.get("mean_confidence",0.0)),
                "mask_ratio":_schedule_mask_ratio(i,tgt,mask_schedule),
            })
        return out
    out=[]
    denom=max(1,tgt-1)
    src_max=max(0,len(src)-1)
    for i in range(tgt):
        src_i=int(round((float(i)/float(denom))*float(src_max)))
        row=src[max(0,min(src_max,src_i))]
        out.append({
            "step":i+1,
            "mean_confidence":float(row.get("mean_confidence",0.0)),
            "mask_ratio":_schedule_mask_ratio(i,tgt,mask_schedule),
        })
    return out


def _normalize_messages(payload:Optional[List[Dict[str,str]]])->List[Dict[str,str]]:
    out:List[Dict[str,str]]=[]
    for row in payload or []:
        if not isinstance(row,dict):
            continue
        role=str(row.get("role","")).strip().lower()
        content=str(row.get("content","")).strip()
        if role not in {"system","user","assistant"} or not content:
            continue
        out.append({"role":role,"content":content})
    return out


def _build_prompt_from_chat(
    prompt:str,
    messages:Optional[List[Dict[str,str]]],
    system_prompt:Optional[str],
    enable_thinking:Optional[bool],
    tokenizer:Any=None,
)->str:
    msg=_normalize_messages(messages)
    sys_text=str(system_prompt or "").strip()
    if not msg:
        if not sys_text:
            return str(prompt).strip()
        msg=[{"role":"system","content":sys_text},{"role":"user","content":str(prompt).strip()}]
    elif sys_text and not any(x.get("role")=="system" for x in msg):
        msg=[{"role":"system","content":sys_text},*msg]
    if tokenizer is not None and hasattr(tokenizer,"apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            try:
                return str(tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
            except Exception:
                pass
        except Exception:
            pass
    lines:List[str]=[]
    for row in msg:
        lines.append(f"{row['role'].capitalize()}: {row['content']}")
    if enable_thinking is True:
        lines.append("System: /think")
    elif enable_thinking is False:
        lines.append("System: /no_think")
    lines.append("Assistant:")
    return "\n".join(lines).strip()

# ---- run manager -----------------------------------------------
_REPO_ROOT=Path(__file__).resolve().parents[3]
_FULL_WSD_CONFIG=str(_REPO_ROOT/"runs"/"configs"/"llada21_dolma_wsd_only.json")

def _find_train_python()->str:
    """Return the Python executable for training subprocesses.

    Preference order:
    1. mdm conda env (where torch+cuda is installed):
       - $CONDA_PREFIX/../envs/mdm/python.exe (when running from base)
       - $CONDA_PREFIX/python.exe (when already in mdm)
    2. sys.executable as last resort.
    """
    conda_prefix=os.environ.get("CONDA_PREFIX","")
    candidates:list=[]
    if conda_prefix:
        p=Path(conda_prefix)
        # Already inside mdm?
        if p.name.lower()=="mdm":
            candidates.append(p/"python.exe")
        # Base env → look for envs/mdm
        candidates.append(p/"envs"/"mdm"/"python.exe")
        # One level up (e.g. miniconda3/envs/X → miniconda3/envs/mdm)
        candidates.append(p.parent/"mdm"/"python.exe")
    # Absolute fallback search in common miniconda/anaconda locations
    for root in [Path("C:/ProgramData/miniconda3"),Path("C:/ProgramData/anaconda3"),
                 Path.home()/"miniconda3",Path.home()/"anaconda3"]:
        candidates.append(root/"envs"/"mdm"/"python.exe")
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable

class _RunManager:
    """Manages a single background WSD subprocess."""
    def __init__(self,api_config_path:str="")->None:
        self._lock=threading.Lock()
        self._proc:Optional[subprocess.Popen]=None
        self._status="idle"
        self._mode=""
        self._logs:deque=deque(maxlen=500)
        self._exit_code:Optional[int]=None
        self._api_config_path=api_config_path
        self._train_py=_find_train_python()
    def start(self,mode:str)->dict:
        with self._lock:
            if self._status=="running":
                return {"ok":False,"error":"already_running"}
            self._logs.clear()
            self._exit_code=None
            self._mode=mode
            self._status="running"
        py=self._train_py
        if mode=="test":
            config=self._api_config_path or _FULL_WSD_CONFIG
            cmd=[py,"-m","hildanext.cli","convert-wsd",
                 "--config",config,"--steps","10"]
        else:
            cmd=[py,"-m","hildanext.cli","run-wsd",
                 "--config",_FULL_WSD_CONFIG,"--skip-preflight"]
        with self._lock:
            self._logs.append(f"[run_manager] python={py}  config={_FULL_WSD_CONFIG}  mode={mode}")
        def _run()->None:
            try:
                env={**os.environ,"PYTHONUNBUFFERED":"1"}
                proc=subprocess.Popen(
                    cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,bufsize=1,env=env)
                with self._lock:
                    self._proc=proc
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline,""):
                    stripped=line.rstrip()
                    if stripped:
                        with self._lock:
                            self._logs.append(stripped)
                proc.wait()
                with self._lock:
                    self._exit_code=proc.returncode
                    self._status="done" if proc.returncode==0 else "error"
                    self._proc=None
            except Exception as exc:
                with self._lock:
                    self._logs.append(f"[runner_error] {exc}")
                    self._status="error"
                    self._proc=None
        threading.Thread(target=_run,daemon=True).start()
        return {"ok":True,"mode":mode}
    def stop(self)->dict:
        with self._lock:
            proc=self._proc
        if proc is None:
            return {"ok":False,"error":"not_running"}
        try:
            proc.terminate()
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            else:
                proc.kill()
        except Exception:
            pass
        with self._lock:
            # Use "stopped" (not "idle") so FE keeps showing logs after user kills the run.
            self._logs.append("[run_manager] run stopped by user")
            self._status="stopped"
            self._exit_code=-1
            self._proc=None
        return {"ok":True}
    def snapshot(self,tail:int=200)->dict:
        with self._lock:
            return {
                "status":self._status,
                "mode":self._mode,
                "exitCode":self._exit_code,
                "lines":list(self._logs)[-tail:],
            }

class RunStartRequest(BaseModel):
    mode:str  # "test" | "full"
# ---- end run manager -----------------------------------------

def create_app(cfg:AppConfig,config_path:str="")->FastAPI:
    tr=trace_from_cfg(cfg)
    tok=set_active_trace(tr)
    app=FastAPI(title="HildaNext API",version="0.1.0")
    app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
    # Lazy: model weights are loaded only on first /generate or /generate/ar call.
    # /run/*, /frontend/wsd, /health work immediately without loading anything.
    engine=_LazyEngine(cfg,trace=tr)
    jobs:Dict[str,Dict[str,Any]]={}
    run_mgr=_RunManager(api_config_path=config_path)
    inference_lock=threading.Lock()
    serialize_inference=bool(getattr(cfg.runtime,"serialize_inference",True))
    lock_timeout_s=float(getattr(cfg.runtime,"inference_lock_timeout_s",120.0))
    require_cuda_for_inference=bool(getattr(cfg.runtime,"require_cuda_for_inference",True))
    preview_chars=max(32,int(getattr(cfg.runtime,"inference_log_preview_chars",140)))
    sse_keepalive_s=max(2,int(getattr(cfg.runtime,"inference_sse_keepalive_s",10)))
    infer_log_ring_size=max(200,int(getattr(cfg.runtime,"inference_log_ring_size",3000)))
    infer_bus=_InferenceEventBus(maxlen=infer_log_ring_size)
    _set_active_infer_bus(infer_bus)
    app.state.infer_event_bus=infer_bus
    infer_ctx_local=threading.local()

    def _push_infer_context(ctx:Optional[Dict[str,Any]]=None)->None:
        stack=list(getattr(infer_ctx_local,"stack",[]) or [])
        stack.append(dict(ctx or {}))
        infer_ctx_local.stack=stack

    def _pop_infer_context()->None:
        stack=list(getattr(infer_ctx_local,"stack",[]) or [])
        if stack:
            stack.pop()
        infer_ctx_local.stack=stack

    def _current_infer_context()->Dict[str,Any]:
        stack=list(getattr(infer_ctx_local,"stack",[]) or [])
        if not stack:
            return {}
        return dict(stack[-1] or {})

    def _emit_dllm_step_event(row:Dict[str,Any])->None:
        ctx=_current_infer_context()
        scope=str(ctx.get("scope") or "DLLM")
        benchmark=ctx.get("benchmark")
        def _f(v:Any)->float:
            try:
                return float(v)
            except Exception:
                return 0.0
        step=int(row.get("step",0))
        mask_ratio=_f(row.get("mask_ratio",0.0))
        gamma=int(row.get("gamma_count",0))
        delta=int(row.get("delta_count",0))
        msg=(
            f"step={step} mask_ratio={mask_ratio:.4f} gamma={gamma} delta={delta} "
            f"tau_mask={_f(row.get('tau_mask',0.0)):.4f} tau_edit={_f(row.get('tau_edit',0.0)):.4f}"
        )
        meta={
            "step":step,
            "mask_ratio":mask_ratio,
            "gamma_count":gamma,
            "delta_count":delta,
            "avg_conf_masked":row.get("avg_conf_masked"),
            "avg_conf_tokens":row.get("avg_conf_tokens"),
            "tau_mask":row.get("tau_mask"),
            "tau_edit":row.get("tau_edit"),
            "prompt_preview":ctx.get("prompt_preview",""),
        }
        if row.get("stop_guard_reason"):
            meta["stop_guard_reason"]=row.get("stop_guard_reason")
            meta["stop_guard_triggered"]=bool(row.get("stop_guard_triggered"))
        _server_log(
            "DLLM_STEP",
            msg,
            lane="dllm",
            scope=scope,
            benchmark=benchmark if isinstance(benchmark,str) else None,
            meta=meta,
        )

    cfg.runtime.inference_step_callback=_emit_dllm_step_event

    def _ensure_cuda_for_inference()->None:
        if not require_cuda_for_inference:
            return
        if torch.cuda.is_available():
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "CUDA required for inference but not available in this API process. "
                f"python={sys.executable}. Start server with your CUDA env "
                "(e.g. mdm python) or set runtime.require_cuda_for_inference=false."
            ),
        )

    def _is_oom_error(err:Exception)->bool:
        msg=str(err).lower()
        return "out of memory" in msg or "cuda oom" in msg

    def _run_with_inference_lock(fn:Callable[[],Any],*,require_cuda:bool=True)->Tuple[Any,float]:
        if require_cuda:
            _ensure_cuda_for_inference()
        if not serialize_inference:
            return fn(),0.0
        t0=time.perf_counter()
        acquired=inference_lock.acquire(timeout=max(0.0,lock_timeout_s))
        if not acquired:
            raise HTTPException(status_code=503,detail="Inference busy: lock timeout")
        wait_ms=(time.perf_counter()-t0)*1000.0
        try:
            return fn(),wait_ms
        finally:
            inference_lock.release()

    def _run_with_context(ctx:Dict[str,Any],fn:Callable[[],Any])->Any:
        _push_infer_context(ctx)
        try:
            return fn()
        finally:
            _pop_infer_context()

    fewshot_cache:Dict[Tuple[str,int,int,str],List[Dict[str,Any]]]={}
    detailed_log_registry:Dict[str,Path]={}
    detailed_log_lock=threading.Lock()
    detailed_log_root=Path(cfg.paths.logs_dir)/"benchmarks"
    detailed_log_root.mkdir(parents=True,exist_ok=True)

    def _start_detailed_log(
        benchmark:str,
        run_label:Optional[str],
        meta:Optional[Dict[str,Any]]=None,
    )->Dict[str,str]:
        bench_slug=_safe_file_part(benchmark,"benchmark")
        label_slug=_safe_file_part(run_label or "run","run")
        stamp=_datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        token=uuid.uuid4().hex
        file_name=f"{stamp}_{bench_slug}_{label_slug}_{token[:8]}.jsonl"
        path=detailed_log_root/file_name
        start_row={
            "event":"run_start",
            "ts_utc":_datetime.utcnow().isoformat()+"Z",
            "benchmark":benchmark,
            "run_label":str(run_label or "").strip() or None,
            "meta":dict(meta or {}),
        }
        with path.open("a",encoding="utf-8") as f:
            f.write(_json.dumps(start_row,ensure_ascii=False)+"\n")
        with detailed_log_lock:
            detailed_log_registry[token]=path
        return {"token":token,"file_name":file_name,"file_path":str(path)}

    def _append_detailed_log(token:Optional[str],row:Dict[str,Any])->bool:
        tok=str(token or "").strip()
        if not tok:
            return False
        with detailed_log_lock:
            path=detailed_log_registry.get(tok)
        if path is None:
            return False
        try:
            with path.open("a",encoding="utf-8") as f:
                f.write(_json.dumps(dict(row or {}),ensure_ascii=False)+"\n")
            return True
        except Exception:
            return False

    def _finish_detailed_log(
        token:Optional[str],
        status:str,
        summary:Optional[Dict[str,Any]]=None,
    )->Optional[str]:
        tok=str(token or "").strip()
        if not tok:
            return None
        with detailed_log_lock:
            path=detailed_log_registry.get(tok)
        if path is None:
            return None
        row={
            "event":"run_end",
            "ts_utc":_datetime.utcnow().isoformat()+"Z",
            "status":str(status or "completed"),
            "summary":dict(summary or {}),
        }
        _append_detailed_log(tok,row)
        with detailed_log_lock:
            detailed_log_registry.pop(tok,None)
        return str(path)

    def _cached_few_shots(dataset:str,n_shots:int,seed:int,scope_key:str)->List[Dict[str,Any]]:
        n=max(0,int(n_shots))
        if n<=0:
            return []
        key=(dataset,n,int(seed),scope_key)
        cached=fewshot_cache.get(key)
        if cached is not None:
            return cached
        if dataset=="hellaswag":
            payload=load_hellaswag_items(cfg,limit=n,seed=int(seed),split="train")
        elif dataset=="mmlu-pro":
            payload=load_mmlu_pro_items(cfg,limit=n,seed=int(seed),split="train")
        elif dataset=="gsm8k":
            payload=load_gsm8k_items(cfg,limit=n,seed=int(seed),split="train")
        else:
            payload={"items":[]}
        items=[x for x in list(payload.get("items") or []) if isinstance(x,dict)]
        fewshot_cache[key]=items
        return items

    def _engine_tokenizer()->Any:
        bundle=getattr(engine,"bundle",None)
        tok=getattr(bundle,"tokenizer",None) if bundle is not None else None
        if tok is None:
            tok=getattr(getattr(engine,"_engine",None),"tokenizer",None)
        return tok

    def _run_benchmark_lanes(
        *,
        prompt:str,
        system_prompt:Optional[str],
        benchmark:Optional[str],
        scope:str,
        mode:str,
        effort:str,
        max_new_tokens:int,
        seed:Optional[int],
        context_window:Optional[int],
        decode_strategy:str,
        temperature:Optional[float],
        top_p:Optional[float],
    )->Tuple[Dict[str,Any],float]:
        from .ar import generate_ar_from_bundle as _generate_ar_bundle
        from .ar import generate_ar as _generate_ar_slow

        scope_norm,want_ar,want_dllm=_resolve_scope(scope)
        decode_mode,temp,top_p_v=_decode_params(decode_strategy,temperature=temperature,top_p=top_p)

        def _do_eval()->Dict[str,Any]:
            tokenizer=_engine_tokenizer()
            prompt_input=str(prompt or "").strip()
            if system_prompt:
                prompt_input=_build_prompt_from_chat(
                    prompt=prompt_input,
                    messages=[{"role":"user","content":prompt_input}],
                    system_prompt=system_prompt,
                    enable_thinking=None,
                    tokenizer=tokenizer,
                )
            prompt_eval=_truncate_prompt_to_context(prompt_input,tokenizer,context_window)
            prompt_preview=_preview_text(prompt_eval,preview_chars)
            lanes:Dict[str,Any]={}
            if want_dllm:
                try:
                    _server_log(
                        "DLLM_REQ_START",
                        f"mode={mode} effort={effort} max_new_tokens={int(max_new_tokens)} decode={decode_mode} scope={scope_norm}",
                        lane="dllm",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={"prompt_preview":prompt_preview},
                    )
                    if scope_norm=="RCD":
                        from .inference_rcd import rcd_decode as _rcd_decode
                        _rcd_result=_run_with_context(
                            {"scope":scope_norm,"benchmark":benchmark,"prompt_preview":prompt_preview},
                            lambda:_rcd_decode(
                                engine.bundle,prompt_eval,
                                max_new_tokens=max_new_tokens,seed=seed,effort=effort,
                            ),
                        )
                        dllm_text=_rcd_result["text"]
                        dllm_stats=_rcd_result.get("stats",{})
                        dllm_stats["diagnostics"]=_rcd_result.get("diagnostics",{})
                    elif scope_norm=="OTS":
                        from .inference_ots import ots_decode as _ots_decode
                        _ots_result=_run_with_context(
                            {"scope":scope_norm,"benchmark":benchmark,"prompt_preview":prompt_preview},
                            lambda:_ots_decode(
                                engine.bundle,prompt_eval,
                                max_new_tokens=max_new_tokens,seed=seed,effort=effort,
                            ),
                        )
                        dllm_text=_ots_result["text"]
                        dllm_stats=_ots_result.get("stats",{})
                        dllm_stats["diagnostics"]=_ots_result.get("diagnostics",{})
                    elif scope_norm=="INFERENZA2":
                        from .inference2 import inferenza2_decode as _inf2_decode
                        _inf2_result=_run_with_context(
                            {"scope":scope_norm,"benchmark":benchmark,"prompt_preview":prompt_preview},
                            lambda:_inf2_decode(
                                engine.bundle,prompt_eval,
                                max_new_tokens=max_new_tokens,seed=seed,effort=effort,
                            ),
                        )
                        dllm_text=_inf2_result["text"]
                        dllm_stats=_inf2_result.get("stats",{})
                        dllm_stats["diagnostics"]=_inf2_result.get("diagnostics",{})
                    else:
                        dllm_text=_run_with_context(
                            {
                                "scope":scope_norm,
                                "benchmark":benchmark,
                                "prompt_preview":prompt_preview,
                            },
                            lambda:engine.generate(
                                prompt=prompt_eval,
                                mode=mode,
                                tau_mask=None,
                                tau_edit=None,
                                max_new_tokens=max_new_tokens,
                                seed=seed,
                                effort=effort,
                                temperature=temp,
                                top_p=top_p_v,
                                top_k=20 if decode_mode=="sampling" else 0,
                                presence_penalty=0.0,
                                repetition_penalty=1.0,
                            ),
                        )
                        dllm_stats=dict(getattr(engine,"last_stats",{}) or {})
                    _server_log(
                        "DLLM_REQ_DONE",
                        f"finish={str(dllm_stats.get('finish_reason','n/a'))} "
                        f"steps={int(dllm_stats.get('steps',0) or 0)} "
                        f"tokens={int(dllm_stats.get('tokens_generated',0) or 0)}",
                        lane="dllm",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={
                            "prompt_preview":prompt_preview,
                            "answer_preview":_preview_text(str(dllm_text or ""),preview_chars),
                            "stop_guard_reason":str(dllm_stats.get("stop_guard_reason","")),
                        },
                    )
                    lanes["dllm"]={
                        "status":"ok",
                        "text":str(dllm_text).strip(),
                        "stats":dllm_stats,
                    }
                except Exception as e:
                    _server_log(
                        "DLLM_REQ_ERROR",
                        str(e),
                        level="error",
                        lane="dllm",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={"prompt_preview":prompt_preview},
                    )
                    lanes["dllm"]={"status":"error","text":"","error":str(e),"stats":dict(getattr(engine,"last_stats",{}) or {})}
            if want_ar:
                try:
                    _server_log(
                        "AR_REQ_START",
                        f"decode={decode_mode} max_new_tokens={int(max_new_tokens)}",
                        lane="ar",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={"prompt_preview":prompt_preview},
                    )
                    bundle=getattr(engine,"bundle",None)
                    if bundle is None:
                        ar_res=_generate_ar_slow(
                            cfg,
                            prompt_eval,
                            max_new_tokens=max_new_tokens,
                            seed=seed,
                            trace=tr,
                            decode_mode=decode_mode,
                            temperature=temp,
                            top_p=top_p_v,
                            top_k=20 if decode_mode=="sampling" else 0,
                            presence_penalty=0.0,
                            repetition_penalty=1.0,
                        )
                    else:
                        s=cfg.runtime.seed if seed is None else int(seed)
                        ar_res=_generate_ar_bundle(
                            bundle,
                            prompt_eval,
                            max_new_tokens=max_new_tokens,
                            seed=s,
                            decode_mode=decode_mode,
                            temperature=temp,
                            top_p=top_p_v,
                            top_k=20 if decode_mode=="sampling" else 0,
                            presence_penalty=0.0,
                            repetition_penalty=1.0,
                        )
                    ar_stats=dict(ar_res or {})
                    _server_log(
                        "AR_REQ_DONE",
                        f"finish={str(ar_stats.get('finish_reason','n/a'))} tokens={int(ar_stats.get('tokens_generated',0) or 0)}",
                        lane="ar",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={
                            "prompt_preview":prompt_preview,
                            "answer_preview":_preview_text(str(ar_stats.get("text","")),preview_chars),
                        },
                    )
                    lanes["ar"]={"status":"ok","text":str(ar_stats.get("text","")).strip(),"stats":ar_stats}
                except Exception as e:
                    _server_log(
                        "AR_REQ_ERROR",
                        str(e),
                        level="error",
                        lane="ar",
                        scope=scope_norm,
                        benchmark=benchmark,
                        meta={"prompt_preview":prompt_preview},
                    )
                    lanes["ar"]={"status":"error","text":"","error":str(e),"stats":None}
            return {
                "scope":scope_norm,
                "decode_mode":decode_mode,
                "temperature":temp,
                "top_p":top_p_v,
                "prompt_used":prompt_eval,
                "lanes":lanes,
            }

        return _run_with_inference_lock(_do_eval)

    def _selected_lane(scope_norm:str,lanes:Dict[str,Any])->str:
        s=str(scope_norm or "DLLM").upper()
        if s=="AR":
            return "ar"
        if s=="DLLM":
            return "dllm"
        if str((lanes.get("dllm") or {}).get("status","")).lower()=="ok":
            return "dllm"
        if str((lanes.get("ar") or {}).get("status","")).lower()=="ok":
            return "ar"
        return "dllm" if "dllm" in lanes else ("ar" if "ar" in lanes else "")
    # Make run_mgr accessible from frontend/wsd endpoint to include live subprocess output.
    app.state.run_mgr=run_mgr
    @app.on_event("shutdown")
    def _shutdown():
        tr.flush()
        reset_active_trace(tok)
        _set_active_infer_bus(None)
        engine.close()
    @app.get("/health")
    def health():
        # engine._loaded is False until first /generate call — no weights in RAM yet.
        loaded=getattr(engine,"_loaded",True)
        st=engine.last_stats or {}
        bundle=engine.bundle if loaded else None
        dummy=bool(getattr(bundle,"is_dummy",False)) if bundle is not None else bool(st.get("dummy_model",False))
        reason=str(getattr(bundle,"load_reason","") if bundle is not None else st.get("load_reason",""))
        issues=getattr(bundle,"env_issues",{}) if bundle is not None else st.get("env_issues",{})
        device=getattr(bundle,"device",None) if bundle is not None else None
        device_name=str(getattr(device,"type","")) if device is not None else str(st.get("device",""))
        actual_dtype=str(getattr(bundle,"actual_dtype","")) if bundle is not None else str(st.get("actual_dtype",""))
        return {
            "status":"ok",
            "engine":engine.name,
            "model_loaded":loaded,
            "model_dir":cfg.paths.model_dir,
            "dummy_model":dummy,
            "reason":reason,
            "env_issues":issues,
            "device":device_name or None,
            "actual_dtype":actual_dtype or None,
            "cuda_available":bool(torch.cuda.is_available()),
            "require_cuda_for_inference":require_cuda_for_inference,
            "strict_decode_invariants":bool(cfg.inference.strict_decode_invariants),
            "fallbacks":tr.snapshot_fallbacks(limit=64)
        }
    @app.get("/frontend/wsd")
    def frontend_wsd():
        """Return a WsdScenarioData payload built from live run artifacts."""
        logs_dir=Path(cfg.paths.logs_dir)
        # Metrics: read cpt.jsonl (last 2000 steps)
        raw_rows=_read_jsonl_tail(logs_dir/"cpt.jsonl",2000)
        metrics=[_map_metric_row(r) for r in raw_rows]
        latest_raw=raw_rows[-1] if raw_rows else None
        # Logs: fallbacks.jsonl + cpt_run.log (last 200 lines)
        fb_rows=_read_jsonl_tail(logs_dir/"fallbacks.jsonl",500)
        fallback_logs=[_map_fallback_log(e,i) for i,e in enumerate(fb_rows)]
        run_logs:List[dict]=[]
        run_log_path=logs_dir/"cpt_run.log"
        if run_log_path.exists():
            try:
                with run_log_path.open("r",encoding="utf-8",errors="replace") as f:
                    lines=f.readlines()[-200:]
                for i,line in enumerate(lines):
                    entry=_map_run_log_line(line,i)
                    if entry:
                        run_logs.append(entry)
            except Exception:
                pass
        # Include live subprocess lines from RunManager (training, tokenization, file reading).
        live_lines:List[dict]=[]
        snap=run_mgr.snapshot(tail=500)
        if snap.get("lines"):  # show lines for ANY status (running/done/error/stopped/idle w/ prior run)
            for i,line in enumerate(snap.get("lines",[])):
                upmsg=line.upper()
                level=("error" if any(k in upmsg for k in ("ERROR","FAIL","OOM","NAN","INF","RUNNER_ERROR"))
                       else "warning" if any(k in upmsg for k in ("WARN","SKIP","FALLBACK"))
                       else "notice" if any(k in upmsg for k in ("NOTICE","MANIFEST","TOKENIZ","DOLMA","DOC_INDEX","TRAIN_FILE","REAL_OK","PREFLIGHT","PHASE_CHANGE","CHECKPOINT"))
                       else "info")
                # Detect source type from content.
                source:str="console"
                if any(k in upmsg for k in ("TOKENIZ","TOKEN_CACHE","SEQ_LEN","DOC_INDEX","DOLMA","MANIFEST","TRAIN_FILE","LOADING","DATASET","REAL_OK")):
                    source="training"
                action=None
                reason=None
                # Extract action=... reason=... from structured log lines.
                if "action=" in line:
                    _m=_re.search(r"action=(\S+)",line)
                    if _m: action=_m.group(1)
                if "reason=" in line:
                    _m=_re.search(r"reason=(\S+)",line)
                    if _m: reason=_m.group(1)
                live_lines.append({
                    "id":f"rm-{i}",
                    "tsUtc":"",
                    "source":source,
                    "level":level,
                    "module":"run_manager",
                    "action":action,
                    "reason":reason,
                    "message":line,
                    "tags":["live"],
                })
        all_logs=sorted(fallback_logs+run_logs+live_lines,key=lambda x:x.get("tsUtc") or "")
        # Try to load the PS1-generated WSD config for accurate schedule figures.
        wsd_override=None
        try:
            wsd_cfg_path=Path(_FULL_WSD_CONFIG)
            if wsd_cfg_path.exists():
                wsd_override=load_config(str(wsd_cfg_path))
        except Exception:
            pass
        meta=_build_wsd_meta(cfg,latest_raw,wsd_cfg_override=wsd_override)
        return {
            "id":"live_wsd_run",
            "label":f"Live: {meta['runId']}",
            "dataSource":"live",
            "meta":meta,
            "metrics":metrics,
            "logs":all_logs,
            "processes":[],
            "insights":_build_wsd_insights(raw_rows,latest_raw,run_lines=snap.get("lines")),
        }
    @app.post("/inference/unload")
    def inference_unload():
        """Explicitly unload engine weights and release RAM/VRAM."""
        try:
            _server_log("INFER_UNLOAD_START","requested",scope="BOTH")
            def _do_unload()->dict:
                was_loaded=bool(getattr(engine,"_loaded",False))
                engine.close()
                return {"was_loaded":was_loaded}
            payload,wait_ms=_run_with_inference_lock(_do_unload,require_cuda=False)
            _server_log(
                "INFER_UNLOAD_DONE",
                f"queue_wait_ms={wait_ms:.1f} was_loaded={bool(payload.get('was_loaded',False))}",
                scope="BOTH",
            )
            return {
                "status":"ok",
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                **payload,
            }
        except HTTPException:
            raise
        except Exception as e:
            _server_log("INFER_UNLOAD_ERROR",str(e),level="error",scope="BOTH")
            raise HTTPException(status_code=500,detail=str(e))

    @app.get("/inference/logs")
    def inference_logs(tail:int=200,after_id:Optional[str]=None):
        rows=infer_bus.snapshot(tail=int(tail or 200),after_id=after_id)
        last_id=str(rows[-1].get("id","")) if rows else ""
        return {
            "status":"ok",
            "tail":max(1,min(5000,int(tail or 200))),
            "after_id":str(after_id or ""),
            "last_id":last_id,
            "events":rows,
        }

    @app.get("/inference/logs/stream")
    def inference_logs_stream(
        after_id:Optional[str]=None,
        last_event_id:Optional[str]=Header(default=None,alias="Last-Event-ID"),
    ):
        resolved_after=str(after_id or last_event_id or "").strip() or None

        def _stream()->Iterator[str]:
            q=infer_bus.subscribe(after_id=resolved_after,tail=20 if resolved_after is None else 0)
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        row=q.get(timeout=float(sse_keepalive_s))
                    except _queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    payload=_json.dumps(row,ensure_ascii=False)
                    event_id=str(row.get("id",""))
                    yield f"id: {event_id}\nevent: inference\ndata: {payload}\n\n"
            finally:
                infer_bus.unsubscribe(q)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":"no-cache",
                "Connection":"keep-alive",
                "X-Accel-Buffering":"no",
            },
        )

    @app.post("/inference/load")
    def inference_load(req:LoadWeightsRequest):
        from .ar import generate_ar_from_bundle as _generate_ar_bundle
        from .ar import generate_ar as _generate_ar_slow
        _ensure_cuda_for_inference()

        scope=(req.scope or "BOTH").upper()
        _server_log(
            "INFER_LOAD_START",
            f"scope={scope} mode={req.mode} max_new_tokens={req.max_new_tokens} effort={req.effort}",
            scope=scope,
        )
        want_ar=scope in {"AR","BOTH"}
        want_dllm=scope in {"DLLM","BOTH"}
        if not (want_ar or want_dllm):
            raise HTTPException(status_code=400,detail="scope must be AR, DLLM, or BOTH")

        def _do_load()->dict:
            lanes:Dict[str,Any]={}
            bundle=getattr(engine,"bundle",None)
            if want_ar:
                try:
                    if bundle is None:
                        ar_res=_generate_ar_slow(
                            cfg,
                            req.prompt,
                            max_new_tokens=req.max_new_tokens,
                            seed=req.seed,
                            trace=tr,
                            decode_mode="greedy",
                        )
                    else:
                        s=cfg.runtime.seed if req.seed is None else int(req.seed)
                        ar_res=_generate_ar_bundle(
                            bundle,
                            req.prompt,
                            max_new_tokens=req.max_new_tokens,
                            seed=s,
                            decode_mode="greedy",
                        )
                    lanes["ar"]={"status":"loaded","message":"AR warmup completato.","stats":ar_res}
                except Exception as e:
                    lanes["ar"]={"status":"error","message":str(e),"stats":None}
            if want_dllm:
                try:
                    _=engine.generate(
                        prompt=req.prompt,
                        mode=req.mode,
                        tau_mask=None,
                        tau_edit=None,
                        max_new_tokens=req.max_new_tokens,
                        seed=req.seed,
                        effort=req.effort,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        presence_penalty=None,
                        repetition_penalty=None,
                    )
                    st=dict(getattr(engine,"last_stats",{}) or {})
                    lanes["dllm"]={"status":"loaded","message":"dLLM warmup completato.","stats":st}
                except Exception as e:
                    lanes["dllm"]={"status":"error","message":str(e),"stats":None}
            return lanes

        try:
            lanes,wait_ms=_run_with_inference_lock(_do_load)
            lane_states=[v.get("status") for v in lanes.values()]
            if lane_states and all(s=="loaded" for s in lane_states):
                status="ok"
            elif lane_states and any(s=="loaded" for s in lane_states):
                status="partial"
            else:
                status="error"
            lanes_state=",".join(f"{k}:{v.get('status', '?')}" for k,v in lanes.items())
            _server_log(
                "INFER_LOAD_DONE",
                f"status={status} queue_wait_ms={wait_ms:.1f} lanes="
                f"{lanes_state}",
                scope=scope,
            )
            return {
                "status":status,
                "engine":engine.name,
                "model_loaded":bool(getattr(engine,"_loaded",False)),
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                "lanes":lanes,
                "fallbacks":tr.snapshot_fallbacks(limit=64),
            }
        except HTTPException:
            raise
        except Exception as e:
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _server_log("INFER_LOAD_ERROR",f"oom scope={scope} detail={e}",level="error",scope=scope)
                raise HTTPException(status_code=503,detail="CUDA OOM during warmup; reduce max_new_tokens")
            _server_log("INFER_LOAD_ERROR",f"scope={scope} detail={e}",level="error",scope=scope)
            raise HTTPException(status_code=500,detail=str(e))
    @app.post("/stage0/validate/log/start")
    def stage0_validate_log_start(req:Stage0DetailedLogStartRequest):
        meta={
            "scope":str(req.scope).upper(),
            "context_window":req.context_window,
            "decode_strategy":req.decode_strategy,
            "effort":req.effort,
            "max_new_tokens":int(req.max_new_tokens),
        }
        try:
            payload=_start_detailed_log(
                benchmark=req.benchmark,
                run_label=req.run_label,
                meta=meta,
            )
            _server_log(
                "BENCH_LOG_START",
                f"benchmark={req.benchmark} file={payload.get('file_name','')} token={payload.get('token','')[:8]}",
            )
            return {"status":"ok",**payload}
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"cannot create detailed log file: {e}")
    @app.post("/stage0/validate/log/finish")
    def stage0_validate_log_finish(req:Stage0DetailedLogFinishRequest):
        path=_finish_detailed_log(req.token,req.status,summary=req.summary)
        if path is None:
            raise HTTPException(status_code=404,detail="detailed log token not found")
        _server_log("BENCH_LOG_FINISH",f"status={req.status} path={path}")
        return {"status":"ok","file_path":path}
    @app.get("/stage0/validate/hellaswag/items")
    def stage0_hellaswag_items(limit:int=64,seed:int=42,force_download:bool=False,split:str="val"):
        try:
            if int(limit)<1 or int(limit)>2048:
                raise HTTPException(status_code=400,detail="limit must be in [1, 2048]")
            split_norm="train" if str(split or "val").strip().lower()=="train" else "val"
            _server_log(
                "BENCH_HELLASWAG_ITEMS_START",
                f"limit={int(limit)} seed={int(seed)} split={split_norm} force_download={bool(force_download)}",
            )
            payload=load_hellaswag_items(
                cfg,
                limit=int(limit),
                seed=int(seed),
                force_download=bool(force_download),
                split=split_norm,
            )
            _server_log(
                "BENCH_HELLASWAG_ITEMS_DONE",
                f"count={len(payload.get('items') or [])} source={payload.get('source','')}",
            )
            return {
                "status":"ok",
                "limit":int(limit),
                "seed":int(seed),
                **payload,
            }
        except HTTPException:
            raise
        except TimeoutError:
            raise HTTPException(status_code=504,detail="Timeout while downloading HellaSwag dataset")
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"HellaSwag dataset error: {e}")
    @app.post("/stage0/validate/hellaswag-item")
    def stage0_validate_hellaswag_item(req:HellaSwagItemRequest):
        if len(req.endings)!=4:
            raise HTTPException(status_code=400,detail="endings must contain exactly 4 options")
        shot_seed=42 if req.seed is None else int(req.seed)
        few_shots=_cached_few_shots("hellaswag",req.n_shots,shot_seed,"train") if req.n_shots>0 else []
        prompt=_build_hellaswag_prompt(req.stem,req.endings,few_shot_items=few_shots)
        _server_log(
            "BENCH_HELLASWAG_ITEM_START",
            f"scope={req.scope} mode={req.mode} effort={req.effort} decode={req.decode_strategy} "
            f"max_new_tokens={req.max_new_tokens} n_shots={req.n_shots}",
        )
        try:
            result,wait_ms=_run_benchmark_lanes(
                prompt=prompt,
                system_prompt=None,
                benchmark="hellaswag",
                scope=req.scope,
                mode=req.mode,
                effort=req.effort,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
                context_window=req.context_window,
                decode_strategy=req.decode_strategy,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            lanes=dict(result.get("lanes") or {})
            lane_preds:Dict[str,Any]={}
            for lane_name,lane in lanes.items():
                lane_text=str((lane or {}).get("text","")).strip()
                pred_idx=_parse_choice_idx(lane_text) if str((lane or {}).get("status","")).lower()=="ok" else None
                lane_preds[lane_name]={
                    "status":lane.get("status"),
                    "error":lane.get("error"),
                    "predicted_idx":pred_idx,
                    "predicted_label":None if pred_idx is None else ["A","B","C","D"][pred_idx],
                    "raw_text":lane_text,
                    "stats":lane.get("stats"),
                }
            scope_norm=str(result.get("scope","DLLM")).upper()
            selected=_selected_lane(scope_norm,lanes)
            pred_idx=lane_preds.get(selected,{}).get("predicted_idx")
            target=req.label_target if req.label_target is not None else None
            if target is None:
                is_correct=None
            elif scope_norm=="BOTH":
                is_correct=any(
                    p.get("predicted_idx") is not None and int(p.get("predicted_idx"))==int(target)
                    for p in lane_preds.values()
                )
            else:
                is_correct=bool(pred_idx is not None and int(pred_idx)==int(target))
            _server_log(
                "BENCH_HELLASWAG_ITEM_DONE",
                f"scope={scope_norm} selected={selected or 'n/a'} pred={pred_idx} queue_wait_ms={wait_ms:.1f}",
            )
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_result",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"hellaswag",
                        "scope":scope_norm,
                        "selected_lane":selected or None,
                        "is_correct":is_correct,
                        "queue_wait_ms":wait_ms,
                        "input":{
                            "stem":req.stem,
                            "endings":req.endings,
                            "n_shots":int(req.n_shots),
                        },
                        "target":{"label_target":target},
                        "lanes":{
                            ln:{
                                "status":lp.get("status"),
                                "error":lp.get("error"),
                                "predicted_idx":lp.get("predicted_idx"),
                                "predicted_label":lp.get("predicted_label"),
                                "raw_text":lp.get("raw_text"),
                                "stats":_compact_stats(lp.get("stats")),
                            }
                            for ln,lp in lane_preds.items()
                        },
                    },
                )
            return {
                "status":"ok",
                "error":None,
                "scope":scope_norm,
                "selected_lane":selected or None,
                "is_correct":is_correct,
                "predicted_idx":pred_idx,
                "predicted_label":None if pred_idx is None else ["A","B","C","D"][int(pred_idx)],
                "lanes":lane_preds,
                "raw_text":str(lane_preds.get(selected,{}).get("raw_text","")),
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":lane_preds.get(selected,{}).get("stats") or {},
            }
        except HTTPException:
            raise
        except Exception as e:
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _server_log("BENCH_HELLASWAG_ITEM_ERROR",str(e))
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_error",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"hellaswag",
                        "scope":str(req.scope or "DLLM").upper(),
                        "error":str(e),
                        "input":{
                            "stem":req.stem,
                            "endings":req.endings,
                            "label_target":req.label_target,
                            "n_shots":int(req.n_shots),
                        },
                    },
                )
            return {
                "status":"error",
                "error":str(e),
                "scope":str(req.scope or "DLLM").upper(),
                "selected_lane":None,
                "is_correct":False if req.label_target is not None else None,
                "predicted_idx":None,
                "predicted_label":None,
                "lanes":{},
                "raw_text":"",
                "queue_wait_ms":0.0,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":dict(getattr(engine,"last_stats",{}) or {}),
            }
    @app.get("/stage0/validate/mmlu-pro/items")
    def stage0_mmlu_pro_items(limit:int=150,seed:int=42,split:str="test"):
        try:
            if int(limit)<1 or int(limit)>2000:
                raise HTTPException(status_code=400,detail="limit must be in [1, 2000]")
            _server_log("BENCH_MMLU_ITEMS_START",f"limit={int(limit)} seed={int(seed)} split={split}")
            payload=load_mmlu_pro_items(cfg,limit=int(limit),seed=int(seed),split=str(split or "test"))
            _server_log(
                "BENCH_MMLU_ITEMS_DONE",
                f"count={len(payload.get('items') or [])} source={payload.get('source','')}",
            )
            return {
                "status":"ok",
                "limit":int(limit),
                "seed":int(seed),
                **payload,
            }
        except HTTPException:
            raise
        except TimeoutError:
            raise HTTPException(status_code=504,detail="Timeout while downloading MMLU-Pro dataset")
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"MMLU-Pro dataset error: {e}")
    @app.post("/stage0/validate/mmlu-pro-item")
    def stage0_validate_mmlu_pro_item(req:MmluProItemRequest):
        if not (2<=len(req.options)<=10):
            raise HTTPException(status_code=400,detail="options must contain 2..10 entries")
        target=str(req.answer_label or "").strip().upper()
        if not _re.match(r"^[A-J]$",target):
            raise HTTPException(status_code=400,detail="answer_label must be a single letter A..J")
        shot_seed=42 if req.seed is None else int(req.seed)
        few_shots=_cached_few_shots("mmlu-pro",req.n_shots,shot_seed,"train") if req.n_shots>0 else []
        prompt=_build_mmlu_pro_prompt(
            req.question,
            req.options,
            force_cot=bool(req.force_cot),
            few_shot_items=few_shots,
        )
        _server_log(
            "BENCH_MMLU_ITEM_START",
            f"scope={req.scope} mode={req.mode} effort={req.effort} decode={req.decode_strategy} "
            f"max_new_tokens={req.max_new_tokens} n_shots={req.n_shots} force_cot={bool(req.force_cot)}",
        )
        try:
            result,wait_ms=_run_benchmark_lanes(
                prompt=prompt,
                system_prompt=_build_mmlu_system_prompt(bool(req.force_cot)),
                benchmark="mmlu-pro",
                scope=req.scope,
                mode=req.mode,
                effort=req.effort,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
                context_window=req.context_window,
                decode_strategy=req.decode_strategy,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            lanes=dict(result.get("lanes") or {})
            lane_preds:Dict[str,Any]={}
            for lane_name,lane in lanes.items():
                lane_text=str((lane or {}).get("text","")).strip()
                pred=_parse_mmlu_answer_label(lane_text) if str((lane or {}).get("status","")).lower()=="ok" else None
                lane_preds[lane_name]={
                    "status":lane.get("status"),
                    "error":lane.get("error"),
                    "predicted_label":pred,
                    "is_correct":bool(pred==target),
                    "raw_text":lane_text,
                    "stats":lane.get("stats"),
                }
            scope_norm=str(result.get("scope","DLLM")).upper()
            selected=_selected_lane(scope_norm,lanes)
            pred=lane_preds.get(selected,{}).get("predicted_label")
            is_correct=(
                any(bool(x.get("is_correct")) for x in lane_preds.values())
                if scope_norm=="BOTH"
                else bool(lane_preds.get(selected,{}).get("is_correct"))
            )
            _server_log(
                "BENCH_MMLU_ITEM_DONE",
                f"scope={scope_norm} selected={selected or 'n/a'} pred={pred} target={target} queue_wait_ms={wait_ms:.1f}",
            )
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_result",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"mmlu-pro",
                        "scope":scope_norm,
                        "selected_lane":selected or None,
                        "is_correct":is_correct,
                        "queue_wait_ms":wait_ms,
                        "input":{
                            "question":req.question,
                            "options":req.options,
                            "n_shots":int(req.n_shots),
                            "force_cot":bool(req.force_cot),
                        },
                        "target":{"answer_label":target},
                        "lanes":{
                            ln:{
                                "status":lp.get("status"),
                                "error":lp.get("error"),
                                "predicted_label":lp.get("predicted_label"),
                                "is_correct":bool(lp.get("is_correct")),
                                "raw_text":lp.get("raw_text"),
                                "stats":_compact_stats(lp.get("stats")),
                            }
                            for ln,lp in lane_preds.items()
                        },
                    },
                )
            return {
                "status":"ok",
                "error":None,
                "scope":scope_norm,
                "selected_lane":selected or None,
                "predicted_label":pred,
                "target_label":target,
                "is_correct":is_correct,
                "lanes":lane_preds,
                "raw_text":str(lane_preds.get(selected,{}).get("raw_text","")),
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":lane_preds.get(selected,{}).get("stats") or {},
            }
        except HTTPException:
            raise
        except Exception as e:
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _server_log("BENCH_MMLU_ITEM_ERROR",str(e))
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_error",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"mmlu-pro",
                        "scope":str(req.scope or "DLLM").upper(),
                        "error":str(e),
                        "input":{
                            "question":req.question,
                            "options":req.options,
                            "answer_label":target,
                            "n_shots":int(req.n_shots),
                            "force_cot":bool(req.force_cot),
                        },
                    },
                )
            return {
                "status":"error",
                "error":str(e),
                "scope":str(req.scope or "DLLM").upper(),
                "selected_lane":None,
                "predicted_label":None,
                "target_label":target,
                "is_correct":False,
                "lanes":{},
                "raw_text":"",
                "queue_wait_ms":0.0,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":dict(getattr(engine,"last_stats",{}) or {}),
            }
    @app.get("/stage0/validate/gsm8k/items")
    def stage0_gsm8k_items(limit:int=150,seed:int=42,split:str="test"):
        try:
            if int(limit)<1 or int(limit)>2000:
                raise HTTPException(status_code=400,detail="limit must be in [1, 2000]")
            _server_log("BENCH_GSM8K_ITEMS_START",f"limit={int(limit)} seed={int(seed)} split={split}")
            payload=load_gsm8k_items(cfg,limit=int(limit),seed=int(seed),split=str(split or "test"))
            _server_log(
                "BENCH_GSM8K_ITEMS_DONE",
                f"count={len(payload.get('items') or [])} source={payload.get('source','')}",
            )
            return {
                "status":"ok",
                "limit":int(limit),
                "seed":int(seed),
                **payload,
            }
        except HTTPException:
            raise
        except TimeoutError:
            raise HTTPException(status_code=504,detail="Timeout while downloading GSM8K dataset")
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"GSM8K dataset error: {e}")
    @app.post("/stage0/validate/gsm8k-item")
    def stage0_validate_gsm8k_item(req:Gsm8kItemRequest):
        target_fmt="hash"
        shot_seed=42 if req.seed is None else int(req.seed)
        few_shots=_cached_few_shots("gsm8k",req.n_shots,shot_seed,"train") if req.n_shots>0 else []
        prompt=_build_gsm8k_prompt(req.question,target_format=target_fmt,few_shot_items=few_shots)
        _server_log(
            "BENCH_GSM8K_ITEM_START",
            f"scope={req.scope} mode={req.mode} effort={req.effort} decode={req.decode_strategy} "
            f"max_new_tokens={req.max_new_tokens} n_shots={req.n_shots} target_format={target_fmt}",
        )
        try:
            result,wait_ms=_run_benchmark_lanes(
                prompt=prompt,
                system_prompt=GSM8K_SYSTEM_PROMPT,
                benchmark="gsm8k",
                scope=req.scope,
                mode=req.mode,
                effort=req.effort,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
                context_window=req.context_window,
                decode_strategy=req.decode_strategy,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            target_num=_extract_gsm_number(req.answer_target,target_format="hash")
            target_float=_to_float(target_num)
            lanes=dict(result.get("lanes") or {})
            lane_preds:Dict[str,Any]={}
            for lane_name,lane in lanes.items():
                lane_text=str((lane or {}).get("text","")).strip()
                pred_num=_extract_gsm_number(lane_text,target_format="hash") if str((lane or {}).get("status","")).lower()=="ok" else None
                pred_float=_to_float(pred_num)
                lane_preds[lane_name]={
                    "status":lane.get("status"),
                    "error":lane.get("error"),
                    "predicted_number":pred_num,
                    "is_correct":bool(pred_float is not None and target_float is not None and pred_float==target_float),
                    "raw_text":lane_text,
                    "stats":lane.get("stats"),
                }
            scope_norm=str(result.get("scope","DLLM")).upper()
            selected=_selected_lane(scope_norm,lanes)
            pred_num=lane_preds.get(selected,{}).get("predicted_number")
            is_correct=(
                any(bool(x.get("is_correct")) for x in lane_preds.values())
                if scope_norm=="BOTH"
                else bool(lane_preds.get(selected,{}).get("is_correct"))
            )
            _server_log(
                "BENCH_GSM8K_ITEM_DONE",
                f"scope={scope_norm} selected={selected or 'n/a'} pred={pred_num} target={target_num} "
                f"queue_wait_ms={wait_ms:.1f}",
            )
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_result",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"gsm8k",
                        "scope":scope_norm,
                        "selected_lane":selected or None,
                        "is_correct":is_correct,
                        "queue_wait_ms":wait_ms,
                        "target_format":target_fmt,
                        "input":{
                            "question":req.question,
                            "answer_target":req.answer_target,
                            "n_shots":int(req.n_shots),
                        },
                        "target":{"number":target_num},
                        "lanes":{
                            ln:{
                                "status":lp.get("status"),
                                "error":lp.get("error"),
                                "predicted_number":lp.get("predicted_number"),
                                "is_correct":bool(lp.get("is_correct")),
                                "raw_text":lp.get("raw_text"),
                                "stats":_compact_stats(lp.get("stats")),
                            }
                            for ln,lp in lane_preds.items()
                        },
                    },
                )
            return {
                "status":"ok",
                "error":None,
                "scope":scope_norm,
                "selected_lane":selected or None,
                "predicted_number":pred_num,
                "target_number":target_num,
                "is_correct":is_correct,
                "lanes":lane_preds,
                "target_format":target_fmt,
                "raw_text":str(lane_preds.get(selected,{}).get("raw_text","")),
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":lane_preds.get(selected,{}).get("stats") or {},
            }
        except HTTPException:
            raise
        except Exception as e:
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _server_log("BENCH_GSM8K_ITEM_ERROR",str(e))
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"item_error",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"gsm8k",
                        "scope":str(req.scope or "DLLM").upper(),
                        "error":str(e),
                        "target_format":target_fmt,
                        "input":{
                            "question":req.question,
                            "answer_target":req.answer_target,
                            "n_shots":int(req.n_shots),
                        },
                    },
                )
            return {
                "status":"error",
                "error":str(e),
                "scope":str(req.scope or "DLLM").upper(),
                "selected_lane":None,
                "predicted_number":None,
                "target_number":_extract_gsm_number(req.answer_target,target_format="hash"),
                "is_correct":False,
                "lanes":{},
                "target_format":target_fmt,
                "raw_text":"",
                "queue_wait_ms":0.0,
                "serialized_inference":serialize_inference,
                "engine":engine.name,
                "stats":dict(getattr(engine,"last_stats",{}) or {}),
            }
    @app.post("/stage0/validate/stability")
    def stage0_validate_stability(req:Stage0StabilityRequest):
        scope_norm,_,want_dllm=_resolve_scope(req.scope)
        if not want_dllm:
            raise HTTPException(status_code=400,detail="Denoising stability is available only for dLLM")
        decode_mode,temp,top_p_v=_decode_params(req.decode_strategy,temperature=req.temperature,top_p=req.top_p)
        try:
            def _do_eval()->Dict[str,Any]:
                tokenizer=_engine_tokenizer()
                prompt_eval=_truncate_prompt_to_context(req.prompt,tokenizer,req.context_window)
                prompt_preview=_preview_text(prompt_eval,preview_chars)
                prev_steps=int(getattr(cfg.inference,"max_steps",10))
                try:
                    cfg.inference.max_steps=max(1,int(req.total_steps))
                    _server_log(
                        "DLLM_REQ_START",
                        f"stability total_steps={int(req.total_steps)} mode={req.mode} effort={req.effort}",
                        lane="dllm",
                        scope=scope_norm,
                        benchmark="stability",
                        meta={"prompt_preview":prompt_preview},
                    )
                    text=_run_with_context(
                        {
                            "scope":scope_norm,
                            "benchmark":"stability",
                            "prompt_preview":prompt_preview,
                        },
                        lambda:engine.generate(
                            prompt=prompt_eval,
                            mode=req.mode,
                            tau_mask=None,
                            tau_edit=None,
                            max_new_tokens=req.max_new_tokens,
                            seed=req.seed,
                            effort=req.effort,
                            temperature=temp,
                            top_p=top_p_v,
                            top_k=20 if decode_mode=="sampling" else 0,
                            presence_penalty=0.0,
                            repetition_penalty=1.0,
                        ),
                    )
                    st=dict(getattr(engine,"last_stats",{}) or {})
                    _server_log(
                        "DLLM_REQ_DONE",
                        f"finish={str(st.get('finish_reason','n/a'))} steps={int(st.get('steps',0) or 0)}",
                        lane="dllm",
                        scope=scope_norm,
                        benchmark="stability",
                        meta={
                            "prompt_preview":prompt_preview,
                            "answer_preview":_preview_text(str(text or ""),preview_chars),
                            "stop_guard_reason":str(st.get("stop_guard_reason","")),
                        },
                    )
                finally:
                    cfg.inference.max_steps=prev_steps
                return {
                    "text":str(text).strip(),
                    "stats":st,
                    "prompt_used":prompt_eval,
                }
            payload,wait_ms=_run_with_inference_lock(_do_eval)
            st=dict(payload.get("stats") or {})
            points_raw=_stable_confidence_points(st.get("logs",[]))
            points=_resample_stability_points(points_raw,int(req.total_steps),req.mask_schedule)
            st["requested_total_steps"]=int(req.total_steps)
            st["requested_mask_schedule"]=str(req.mask_schedule)
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"run_result",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"stability",
                        "scope":scope_norm,
                        "selected_lane":"dllm",
                        "decode_mode":decode_mode,
                        "queue_wait_ms":wait_ms,
                        "input":{
                            "prompt":req.prompt,
                            "total_steps":int(req.total_steps),
                            "mask_schedule":req.mask_schedule,
                        },
                        "output":{
                            "text":str(payload.get("text","")).strip(),
                            "points":points,
                        },
                        "stats":_compact_stats(st),
                    },
                )
            return {
                "status":"ok",
                "scope":scope_norm,
                "selected_lane":"dllm",
                "decode_mode":decode_mode,
                "text":str(payload.get("text","")).strip(),
                "points":points,
                "queue_wait_ms":wait_ms,
                "serialized_inference":serialize_inference,
                "stats":st,
                "engine":engine.name,
            }
        except HTTPException:
            raise
        except Exception as e:
            _server_log("DLLM_REQ_ERROR",str(e),level="error",lane="dllm",scope=scope_norm,benchmark="stability")
            if req.detailed_log_token:
                _append_detailed_log(
                    req.detailed_log_token,
                    {
                        "event":"run_error",
                        "ts_utc":_datetime.utcnow().isoformat()+"Z",
                        "benchmark":"stability",
                        "scope":scope_norm,
                        "error":str(e),
                        "input":{
                            "prompt":req.prompt,
                            "total_steps":int(req.total_steps),
                            "mask_schedule":req.mask_schedule,
                        },
                    },
                )
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503,detail="CUDA OOM in stability check")
            raise HTTPException(status_code=500,detail=str(e))
    @app.post("/generate",response_model=GenerateResponse)
    def generate(req:GenerateRequest):
        try:
            def _do_generate()->Tuple[str,str]:
                bundle=getattr(engine,"bundle",None)
                tokenizer=getattr(bundle,"tokenizer",None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer=getattr(getattr(engine,"_engine",None),"tokenizer",None)
                prompt_text=_build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                prompt_preview=_preview_text(prompt_text,preview_chars)
                start_msg=(
                    f"mode={req.mode} effort={req.effort} max_new_tokens={int(req.max_new_tokens)} "
                    f"temperature={req.temperature if req.temperature is not None else 'auto'} "
                    f"top_p={req.top_p if req.top_p is not None else 'auto'}"
                )
                _server_log(
                    "INFER_REQ_START",
                    start_msg,
                    scope="DLLM",
                    meta={"prompt_preview":prompt_preview},
                )
                _server_log(
                    "DLLM_REQ_START",
                    start_msg,
                    lane="dllm",
                    scope="DLLM",
                    meta={"prompt_preview":prompt_preview},
                )
                text=_run_with_context(
                    {
                        "scope":"DLLM",
                        "benchmark":None,
                        "prompt_preview":prompt_preview,
                    },
                    lambda:engine.generate(
                        prompt=prompt_text,
                        mode=req.mode,
                        tau_mask=req.tau_mask,
                        tau_edit=req.tau_edit,
                        max_new_tokens=req.max_new_tokens,
                        seed=req.seed,
                        effort=req.effort,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        presence_penalty=req.presence_penalty,
                        repetition_penalty=req.repetition_penalty,
                    ),
                )
                return str(text),prompt_preview
            (text,prompt_preview),wait_ms=_run_with_inference_lock(_do_generate)
            st=dict(getattr(engine,"last_stats",{}) or {})
            st["queue_wait_ms"]=wait_ms
            st["serialized_inference"]=serialize_inference
            if "finish_reason" not in st:
                st["finish_reason"]="converged"
            if "truncated" not in st:
                st["truncated"]=False
            if any(v is not None for v in [req.temperature,req.top_p,req.top_k]):
                st["ignored_sampling_params"]=True
            st["fallbacks"]=tr.snapshot_fallbacks(limit=128)
            thinking,answer=_split_thinking_output(text)
            done_msg=(
                f"finish={str(st.get('finish_reason','n/a'))} "
                f"steps={int(st.get('steps',0) or 0)} "
                f"tokens={int(st.get('tokens_generated',0) or 0)} "
                f"queue_wait_ms={wait_ms:.1f}"
            )
            done_meta={
                "prompt_preview":prompt_preview,
                "answer_preview":_preview_text(answer,preview_chars),
                "thinking_preview":_preview_text(thinking,preview_chars) if thinking else "",
                "stop_guard_reason":str(st.get("stop_guard_reason","")),
            }
            _server_log("INFER_REQ_DONE",done_msg,scope="DLLM",meta=done_meta)
            _server_log("DLLM_REQ_DONE",done_msg,lane="dllm",scope="DLLM",meta=done_meta)
            return GenerateResponse(text=text,stats=st,engine=engine.name)
        except HTTPException:
            raise
        except Exception as e:
            _server_log("INFER_REQ_ERROR",str(e),level="error",scope="DLLM")
            _server_log("DLLM_REQ_ERROR",str(e),level="error",lane="dllm",scope="DLLM")
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503,detail="CUDA OOM: reduce max_new_tokens or effort")
            tr.record_fallback(
                event="fallback",
                module="api",
                func="generate",
                action="api_error",
                reason="generate_exception",
                exception_str=exception_with_stack(e),
                extra_dict={"mode":req.mode}
            )
            raise HTTPException(status_code=500,detail=str(e))
    @app.post("/generate/ar",response_model=GenerateResponse)
    def generate_ar_endpoint(req:ArGenerateRequest):
        """Autoregressive decode reusing the server's already-loaded engine bundle."""
        from .ar import generate_ar_from_bundle as _generate_ar_bundle
        try:
            def _do_generate_ar()->Tuple[dict,str]:
                # Reuse the engine's pre-loaded bundle — avoids reloading the model from disk.
                bundle=getattr(engine,"bundle",None)
                tokenizer=getattr(bundle,"tokenizer",None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer=getattr(getattr(engine,"_engine",None),"tokenizer",None)
                prompt_text=_build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                prompt_preview=_preview_text(prompt_text,preview_chars)
                start_msg=(
                    f"decode={req.decode_mode} max_new_tokens={int(req.max_new_tokens)} "
                    f"temperature={float(req.temperature):.3f} top_p={float(req.top_p):.3f}"
                )
                _server_log("INFER_REQ_START",start_msg,scope="AR",meta={"prompt_preview":prompt_preview})
                _server_log("AR_REQ_START",start_msg,lane="ar",scope="AR",meta={"prompt_preview":prompt_preview})
                if bundle is None:
                    # Fallback: slow path (loads model fresh) for non-TransformersEngine.
                    from .ar import generate_ar as _generate_ar_slow
                    out=_generate_ar_slow(
                        cfg,
                        prompt_text,
                        max_new_tokens=req.max_new_tokens,
                        seed=req.seed,
                        trace=tr,
                        decode_mode=req.decode_mode,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        presence_penalty=req.presence_penalty,
                        repetition_penalty=req.repetition_penalty,
                    )
                    return out,prompt_preview
                s=cfg.runtime.seed if req.seed is None else int(req.seed)
                out=_generate_ar_bundle(
                    bundle,
                    prompt_text,
                    max_new_tokens=req.max_new_tokens,
                    seed=s,
                    decode_mode=req.decode_mode,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    presence_penalty=req.presence_penalty,
                    repetition_penalty=req.repetition_penalty,
                )
                return out,prompt_preview
            (result,prompt_preview),wait_ms=_run_with_inference_lock(_do_generate_ar)
            result["queue_wait_ms"]=wait_ms
            result["serialized_inference"]=serialize_inference
            result["fallbacks"]=tr.snapshot_fallbacks(limit=32)
            thinking,answer=_split_thinking_output(str(result.get("text","")))
            done_msg=(
                f"finish={str(result.get('finish_reason','n/a'))} "
                f"tokens={int(result.get('tokens_generated',0) or 0)} "
                f"queue_wait_ms={wait_ms:.1f}"
            )
            done_meta={
                "prompt_preview":prompt_preview,
                "answer_preview":_preview_text(answer,preview_chars),
                "thinking_preview":_preview_text(thinking,preview_chars) if thinking else "",
            }
            _server_log("INFER_REQ_DONE",done_msg,scope="AR",meta=done_meta)
            _server_log("AR_REQ_DONE",done_msg,lane="ar",scope="AR",meta=done_meta)
            return GenerateResponse(text=result["text"],stats=result,engine=result["engine"])
        except HTTPException:
            raise
        except Exception as e:
            _server_log("INFER_REQ_ERROR",str(e),level="error",scope="AR")
            _server_log("AR_REQ_ERROR",str(e),level="error",lane="ar",scope="AR")
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503,detail="CUDA OOM in AR lane; reduce max_new_tokens")
            tr.record_fallback(
                event="fallback",
                module="api",
                func="generate_ar_endpoint",
                action="api_error",
                reason="generate_ar_exception",
                exception_str=exception_with_stack(e),
                extra_dict={"max_new_tokens":req.max_new_tokens}
            )
            raise HTTPException(status_code=500,detail=str(e))
    @app.post("/jobs/submit",response_model=JobResponse)
    def submit_job(req:GenerateRequest):
        job_id=uuid.uuid4().hex
        jobs[job_id]={"status":"running","created_at":time.time()}
        try:
            def _do_generate()->str:
                bundle=getattr(engine,"bundle",None)
                tokenizer=getattr(bundle,"tokenizer",None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer=getattr(getattr(engine,"_engine",None),"tokenizer",None)
                prompt_text=_build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                return engine.generate(
                    prompt=prompt_text,
                    mode=req.mode,
                    tau_mask=req.tau_mask,
                    tau_edit=req.tau_edit,
                    max_new_tokens=req.max_new_tokens,
                    seed=req.seed,
                    effort=req.effort,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    presence_penalty=req.presence_penalty,
                    repetition_penalty=req.repetition_penalty,
                )
            text,wait_ms=_run_with_inference_lock(_do_generate)
            st=dict(getattr(engine,"last_stats",{}) or {})
            st["queue_wait_ms"]=wait_ms
            st["serialized_inference"]=serialize_inference
            st["fallbacks"]=tr.snapshot_fallbacks(limit=128)
            result=GenerateResponse(text=text,stats=st,engine=engine.name)
            jobs[job_id]={"status":"done","result":result.model_dump()}
            return JobResponse(job_id=job_id,status="done",result=result)
        except HTTPException as e:
            jobs[job_id]={"status":"error","error":str(e.detail)}
            return JobResponse(job_id=job_id,status="error",error=str(e.detail))
        except Exception as e:
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                msg="CUDA OOM: reduce max_new_tokens or effort"
                jobs[job_id]={"status":"error","error":msg}
                return JobResponse(job_id=job_id,status="error",error=msg)
            tr.record_fallback(
                event="fallback",
                module="api",
                func="submit_job",
                action="api_error",
                reason="submit_exception",
                exception_str=exception_with_stack(e),
                extra_dict={"mode":req.mode}
            )
            jobs[job_id]={"status":"error","error":str(e)}
            return JobResponse(job_id=job_id,status="error",error=str(e))
    @app.get("/jobs/{job_id}",response_model=JobResponse)
    def get_job(job_id:str):
        if job_id not in jobs:
            raise HTTPException(status_code=404,detail="job not found")
        j=jobs[job_id]
        if j["status"]=="done":
            return JobResponse(job_id=job_id,status="done",result=GenerateResponse(**j["result"]))
        return JobResponse(job_id=job_id,status=j["status"],error=j.get("error"))
    # ---- run control endpoints ----------------------------------------
    @app.post("/run/start")
    def run_start(body:RunStartRequest):
        return run_mgr.start(body.mode)
    @app.get("/run/status")
    def run_status():
        return run_mgr.snapshot()
    @app.post("/run/stop")
    def run_stop():
        return run_mgr.stop()
    # ---- end run control ----------------------------------------------

    # ---- RCD inference endpoint ----------------------------------------
    @app.post("/inferencercdm")
    def inferencercdm(req:"_InferenceRCDMRequest"):
        from .inference_rcd import InferenceRCDMRequest as _RCDReqSchema  # noqa: F811
        from .inference_rcd import rcd_decode
        from .diffusion import force_noncausal_attention
        from .inference import mode_thresholds, _resolve_effort
        try:
            def _do_rcd():
                bundle = getattr(engine, "bundle", None)
                tokenizer = getattr(bundle, "tokenizer", None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer = getattr(getattr(engine, "_engine", None), "tokenizer", None)
                prompt_text = _build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                if bundle is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                model = bundle.model
                device = bundle.device
                mask_id = bundle.mask_id
                v = bundle.vocab_size
                is_dummy = bundle.is_dummy
                tau_m, tau_e = mode_thresholds(engine._cfg, req.mode, req.tau_mask, req.tau_edit)
                max_new = max(1, int(req.max_new_tokens))
                eff_steps, tau_m, tau_e = _resolve_effort(
                    req.effort, max(1, int(engine._cfg.inference.max_steps)), tau_m, tau_e,
                )
                # Warm-start reference model handling
                warm_start_model = None
                ref_model_used = False
                if req.rcd_warm_start and req.rcd_reference_model:
                    try:
                        from .inference import load_model_bundle
                        # Load separate reference model — expensive, cached externally if needed
                        ref_cfg = engine._cfg
                        warm_start_model = None  # placeholder: would need full load
                        # DEVIATION: full Mref loading not implemented in this MVP;
                        # fallback to same-model warm start.
                    except Exception:
                        pass
                if warm_start_model is None and req.rcd_warm_start and not req.rcd_same_model_warm_start_fallback:
                    if req.rcd_reference_model:
                        raise HTTPException(
                            status_code=400,
                            detail="rcd_reference_model specified but could not be loaded, "
                                   "and rcd_same_model_warm_start_fallback is disabled",
                        )
                text, stats, diag = rcd_decode(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    mask_id=mask_id,
                    vocab_size=v,
                    prompt=prompt_text,
                    max_new_tokens=max_new,
                    tau_mask=tau_m,
                    tau_edit=tau_e,
                    max_steps=eff_steps,
                    t_res=req.rcd_temperature_residual,
                    alpha_mode=req.rcd_alpha_mode,
                    force_mask_only=req.rcd_force_mask_only_injection,
                    warm_start=req.rcd_warm_start,
                    warm_start_model=warm_start_model,
                    store_diagnostics=req.rcd_store_step_diagnostics,
                    seed=req.seed if req.seed is not None else int(engine._cfg.runtime.seed),
                    is_dummy=is_dummy,
                    force_noncausal_ctx=force_noncausal_attention,
                )
                return text, stats
            (text, stats), wait_ms = _run_with_inference_lock(_do_rcd)
            stats["queue_wait_ms"] = wait_ms
            stats["serialized_inference"] = serialize_inference
            stats["fallbacks"] = tr.snapshot_fallbacks(limit=128)
            return {"text": text, "stats": stats, "engine": "rcd"}
        except HTTPException:
            raise
        except Exception as e:
            _server_log("RCD_REQ_ERROR", str(e), level="error", scope="RCD")
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503, detail="CUDA OOM in RCD inference")
            raise HTTPException(status_code=500, detail=str(e))
    # Use the same schema shape but with RCD fields — import deferred above
    from .inference_rcd import InferenceRCDMRequest as _InferenceRCDMRequest  # noqa: E402
    # ---- end RCD endpoint ---------------------------------------------

    # ---- OTS inference endpoint ----------------------------------------
    @app.post("/inferenceots")
    def inferenceots(req:"_InferenceOTSRequest"):
        from .inference_ots import InferenceOTSRequest as _OTSReqSchema  # noqa: F811
        from .inference_ots import ots_decode
        from .diffusion import force_noncausal_attention
        from .inference import mode_thresholds, _resolve_effort
        try:
            def _do_ots():
                bundle = getattr(engine, "bundle", None)
                tokenizer = getattr(bundle, "tokenizer", None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer = getattr(getattr(engine, "_engine", None), "tokenizer", None)
                prompt_text = _build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                if bundle is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                model = bundle.model
                device = bundle.device
                mask_id = bundle.mask_id
                v = bundle.vocab_size
                is_dummy = bundle.is_dummy
                tau_m, tau_e = mode_thresholds(engine._cfg, req.mode, req.tau_mask, req.tau_edit)
                max_new = max(1, int(req.max_new_tokens))
                eff_steps, tau_m, tau_e = _resolve_effort(
                    req.effort, max(1, int(engine._cfg.inference.max_steps)), tau_m, tau_e,
                )
                text, stats, _diag = ots_decode(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    mask_id=mask_id,
                    vocab_size=v,
                    prompt=prompt_text,
                    max_new_tokens=max_new,
                    tau_mask=tau_m,
                    tau_edit=tau_e,
                    max_steps=eff_steps,
                    beam_size=req.ots_beam_size,
                    gumbel_temperature=req.ots_gumbel_temperature,
                    search_interval=req.ots_search_interval,
                    pruning_mode=req.ots_pruning_mode,
                    allow_fallback_score=req.ots_allow_fallback_simple_score,
                    store_trace=req.ots_store_search_trace,
                    seed=req.ots_seed if req.ots_seed is not None else (
                        req.seed if req.seed is not None else int(engine._cfg.runtime.seed)
                    ),
                    is_dummy=is_dummy,
                    force_noncausal_ctx=force_noncausal_attention,
                )
                return text, stats
            (text, stats), wait_ms = _run_with_inference_lock(_do_ots)
            stats["queue_wait_ms"] = wait_ms
            stats["serialized_inference"] = serialize_inference
            stats["fallbacks"] = tr.snapshot_fallbacks(limit=128)
            return {"text": text, "stats": stats, "engine": "ots"}
        except HTTPException:
            raise
        except Exception as e:
            _server_log("OTS_REQ_ERROR", str(e), level="error", scope="OTS")
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503, detail="CUDA OOM in OTS inference")
            raise HTTPException(status_code=500, detail=str(e))
    from .inference_ots import InferenceOTSRequest as _InferenceOTSRequest  # noqa: E402
    # ---- end OTS endpoint ---------------------------------------------

    # ---- Inferenza2 hybrid endpoint ------------------------------------
    @app.post("/inferenza2")
    def inferenza2(req:"_Inference2Request"):
        from .inference2 import Inference2Request as _Inf2ReqSchema  # noqa: F811
        from .inference2 import inferenza2_decode
        from .diffusion import force_noncausal_attention
        from .inference import mode_thresholds, _resolve_effort
        try:
            def _do_inferenza2():
                bundle = getattr(engine, "bundle", None)
                tokenizer = getattr(bundle, "tokenizer", None) if bundle is not None else None
                if tokenizer is None:
                    tokenizer = getattr(getattr(engine, "_engine", None), "tokenizer", None)
                prompt_text = _build_prompt_from_chat(
                    prompt=req.prompt,
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    enable_thinking=req.enable_thinking,
                    tokenizer=tokenizer,
                )
                if bundle is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                model = bundle.model
                device = bundle.device
                mask_id = bundle.mask_id
                v = bundle.vocab_size
                is_dummy = bundle.is_dummy
                tau_m, tau_e = mode_thresholds(engine._cfg, req.mode, req.tau_mask, req.tau_edit)
                max_new = max(1, int(req.max_new_tokens))
                eff_steps, tau_m, tau_e = _resolve_effort(
                    req.effort, max(1, int(engine._cfg.inference.max_steps)), tau_m, tau_e,
                )
                # Resolve ablation mode override
                rcd_on = req.hybrid_enable_rcd
                ots_on = req.hybrid_enable_ots
                if req.hybrid_ablation_mode == "rcd_only":
                    rcd_on, ots_on = True, False
                elif req.hybrid_ablation_mode == "ots_only":
                    rcd_on, ots_on = False, True
                elif req.hybrid_ablation_mode == "full_hybrid":
                    rcd_on, ots_on = True, True
                text, stats, _diag = inferenza2_decode(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    mask_id=mask_id,
                    vocab_size=v,
                    prompt=prompt_text,
                    max_new_tokens=max_new,
                    tau_mask=tau_m,
                    tau_edit=tau_e,
                    max_steps=eff_steps,
                    seed=req.seed if req.seed is not None else int(engine._cfg.runtime.seed),
                    is_dummy=is_dummy,
                    force_noncausal_ctx=force_noncausal_attention,
                    rcd_enabled=rcd_on,
                    t_res=req.rcd_temperature_residual,
                    force_mask_only=req.rcd_force_mask_only_injection,
                    warm_start=req.rcd_warm_start,
                    warm_start_model=None,
                    ots_enabled=ots_on,
                    beam_size=req.ots_beam_size,
                    gumbel_temperature=req.ots_gumbel_temperature,
                    search_interval=req.ots_search_interval,
                    pruning_mode=req.ots_pruning_mode,
                    allow_fallback_score=req.ots_allow_fallback_simple_score,
                    store_diagnostics=req.hybrid_store_diagnostics,
                    store_trace=req.hybrid_store_search_trace,
                )
                return text, stats
            (text, stats), wait_ms = _run_with_inference_lock(_do_inferenza2)
            stats["queue_wait_ms"] = wait_ms
            stats["serialized_inference"] = serialize_inference
            stats["fallbacks"] = tr.snapshot_fallbacks(limit=128)
            return {"text": text, "stats": stats, "engine": "inferenza2"}
        except HTTPException:
            raise
        except Exception as e:
            _server_log("INFERENZA2_REQ_ERROR", str(e), level="error", scope="INFERENZA2")
            if _is_oom_error(e) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise HTTPException(status_code=503, detail="CUDA OOM in inferenza2 inference")
            raise HTTPException(status_code=500, detail=str(e))
    from .inference2 import Inference2Request as _Inference2Request  # noqa: E402
    # ---- end Inferenza2 endpoint --------------------------------------
    return app

def run_server(config_path:str,host:str="127.0.0.1",port:int=8080)->None:
    import uvicorn
    here=os.getcwd()
    print(f"[start_server] cwd={here}",flush=True)
    print(f"[start_server] config={config_path}",flush=True)
    print(f"[start_server] python={sys.executable}",flush=True)
    try:
        print(
            "[start_server] torch="
            f"{torch.__version__} cuda_build={torch.version.cuda} "
            f"cuda_available={bool(torch.cuda.is_available())} "
            f"gpu_count={int(torch.cuda.device_count())}",
            flush=True,
        )
    except Exception as e:
        print(f"[start_server] torch_diag_error={e}",flush=True)
    print("[start_server] model = LAZY (loaded only on first /generate call)",flush=True)
    cfg=load_config(config_path)
    app=create_app(cfg,config_path=config_path)
    print(f"[start_server] starting uvicorn on http://{host}:{int(port)}  (no weights loaded yet)",flush=True)
    uvicorn.run(app,host=host,port=port,log_level="info")

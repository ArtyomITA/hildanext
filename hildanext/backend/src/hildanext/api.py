# FastAPI server for health, generation, and job endpoints.
# Main entrypoints: create_app,run_server.
# Engine is shared and supports dInfer fallback logic.
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any,Dict,Optional,List
import json as _json
import os
import re as _re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime as _datetime
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from .config import AppConfig,load_config
from .inference import build_engine
from .trace import trace_from_cfg,set_active_trace,reset_active_trace,use_trace,exception_with_stack

class GenerateRequest(BaseModel):
    prompt:str
    mode:str="S_MODE"
    tau_mask:Optional[float]=None
    tau_edit:Optional[float]=None
    max_new_tokens:int=Field(default=96,ge=1,le=4096)
    seed:Optional[int]=None
    effort:str=Field(default="medium",description="instant|low|medium|high|adaptive — controls decode steps and tau scaling")

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

def _build_wsd_meta(cfg:Any,latest:Optional[dict])->dict:
    exp=getattr(cfg,"experiment",None)
    run_id=getattr(exp,"experiment_id","unknown") if exp else "unknown"
    notes=(getattr(exp,"notes","") or "")[:40]
    wsd_cfg=getattr(cfg,"wsd",None)
    ladder=list(getattr(wsd_cfg,"ladder_blocks",[1])) if wsd_cfg else [1]
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
    }

def _build_wsd_insights(raw_rows:List[dict],latest:Optional[dict])->List[dict]:
    out:List[dict]=[]
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

# ---- run manager -----------------------------------------------
_REPO_ROOT=Path(__file__).resolve().parents[3]
_FULL_WSD_CONFIG=str(_REPO_ROOT/"runs"/"configs"/"llada21_dolma_wsd_only.json")

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
    def start(self,mode:str)->dict:
        with self._lock:
            if self._status=="running":
                return {"ok":False,"error":"already_running"}
            self._logs.clear()
            self._exit_code=None
            self._mode=mode
            self._status="running"
        if mode=="test":
            config=self._api_config_path or _FULL_WSD_CONFIG
            cmd=[sys.executable,"-m","hildanext.cli","convert-wsd",
                 "--config",config,"--steps","10"]
        else:
            cmd=[sys.executable,"-m","hildanext.cli","run-wsd",
                 "--config",_FULL_WSD_CONFIG,"--skip-preflight"]
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
            self._status="idle"
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
    engine=build_engine(cfg,trace=tr)
    jobs:Dict[str,Dict[str,Any]]={}
    run_mgr=_RunManager(api_config_path=config_path)
    @app.on_event("shutdown")
    def _shutdown():
        tr.flush()
        reset_active_trace(tok)
        engine.close()
    @app.get("/health")
    def health():
        st=getattr(engine,"last_stats",{}) or {}
        bundle=getattr(engine,"bundle",None)
        dummy=bool(getattr(bundle,"is_dummy",False)) if bundle is not None else bool(st.get("dummy_model",False))
        reason=str(getattr(bundle,"load_reason","") if bundle is not None else st.get("load_reason",""))
        issues=getattr(bundle,"env_issues",{}) if bundle is not None else st.get("env_issues",{})
        return {
            "status":"ok",
            "engine":engine.name,
            "model_dir":cfg.paths.model_dir,
            "dummy_model":dummy,
            "reason":reason,
            "env_issues":issues,
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
        all_logs=sorted(fallback_logs+run_logs,key=lambda x:x.get("tsUtc") or "")
        meta=_build_wsd_meta(cfg,latest_raw)
        return {
            "id":"live_wsd_run",
            "label":f"Live: {meta['runId']}",
            "dataSource":"live",
            "meta":meta,
            "metrics":metrics,
            "logs":all_logs,
            "processes":[],
            "insights":_build_wsd_insights(raw_rows,latest_raw),
        }
    @app.post("/generate",response_model=GenerateResponse)
    def generate(req:GenerateRequest):
        try:
            text=engine.generate(
                prompt=req.prompt,
                mode=req.mode,
                tau_mask=req.tau_mask,
                tau_edit=req.tau_edit,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
                effort=req.effort
            )
            st=dict(getattr(engine,"last_stats",{}) or {})
            st["fallbacks"]=tr.snapshot_fallbacks(limit=128)
            return GenerateResponse(text=text,stats=st,engine=engine.name)
        except Exception as e:
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
    @app.post("/jobs/submit",response_model=JobResponse)
    def submit_job(req:GenerateRequest):
        job_id=uuid.uuid4().hex
        jobs[job_id]={"status":"running","created_at":time.time()}
        try:
            text=engine.generate(
                prompt=req.prompt,
                mode=req.mode,
                tau_mask=req.tau_mask,
                tau_edit=req.tau_edit,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
                effort=req.effort
            )
            st=dict(getattr(engine,"last_stats",{}) or {})
            st["fallbacks"]=tr.snapshot_fallbacks(limit=128)
            result=GenerateResponse(text=text,stats=st,engine=engine.name)
            jobs[job_id]={"status":"done","result":result.model_dump()}
            return JobResponse(job_id=job_id,status="done",result=result)
        except Exception as e:
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
    return app

def run_server(config_path:str,host:str="127.0.0.1",port:int=8080)->None:
    import uvicorn
    cfg=load_config(config_path)
    app=create_app(cfg,config_path=config_path)
    uvicorn.run(app,host=host,port=port,log_level="info")

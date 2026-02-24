# FastAPI server for health, generation, and job endpoints.
# Main entrypoints: create_app,run_server.
# Engine is shared and supports dInfer fallback logic.
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any,Dict,Optional
import time
import uuid
from fastapi import FastAPI,HTTPException
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

class GenerateResponse(BaseModel):
    text:str
    stats:Dict[str,Any]
    engine:str

class JobResponse(BaseModel):
    job_id:str
    status:str
    result:Optional[GenerateResponse]=None
    error:Optional[str]=None

def create_app(cfg:AppConfig)->FastAPI:
    tr=trace_from_cfg(cfg)
    tok=set_active_trace(tr)
    app=FastAPI(title="HildaNext API",version="0.1.0")
    engine=build_engine(cfg,trace=tr)
    jobs:Dict[str,Dict[str,Any]]={}
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
    @app.post("/generate",response_model=GenerateResponse)
    def generate(req:GenerateRequest):
        try:
            text=engine.generate(
                prompt=req.prompt,
                mode=req.mode,
                tau_mask=req.tau_mask,
                tau_edit=req.tau_edit,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed
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
                seed=req.seed
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
    return app

def run_server(config_path:str,host:str="127.0.0.1",port:int=8080)->None:
    import uvicorn
    cfg=load_config(config_path)
    app=create_app(cfg)
    uvicorn.run(app,host=host,port=port,log_level="info")

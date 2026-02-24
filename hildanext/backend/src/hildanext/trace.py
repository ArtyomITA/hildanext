# Runtime trace for fallbacks, env issues, and metrics.
# Main entrypoints: trace_from_cfg,RunTrace,ensure_run_id.
# Persists JSONL logs for overnight reproducibility.
from __future__ import annotations
from pathlib import Path
from contextvars import ContextVar
from datetime import datetime,timezone
from typing import Any,Dict,List,Optional
import hashlib
import json
import os
import traceback
import uuid

_ACTIVE_TRACE:ContextVar["RunTrace|None"]=ContextVar("hildanext_active_trace",default=None)


def _now()->str:
    return datetime.now(timezone.utc).isoformat()

def _jsonable(x:Any)->Any:
    if x is None or isinstance(x,(str,int,float,bool)):
        return x
    if isinstance(x,dict):
        return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)):
        return [_jsonable(v) for v in x]
    return str(x)

def ensure_run_id(run_id:Optional[str]=None)->str:
    if run_id and str(run_id).strip():
        return str(run_id).strip()
    env=os.environ.get("HILDANEXT_RUN_ID","")
    if env.strip():
        return env.strip()
    rid=f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    os.environ["HILDANEXT_RUN_ID"]=rid
    return rid

def config_digest_from_cfg(cfg:Any)->str:
    try:
        from .config import to_dict
        payload=to_dict(cfg)
        raw=json.dumps(payload,ensure_ascii=True,sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""

class RunTrace:
    def __init__(self,run_id:str,root_log_dir:str|Path,strict_fallbacks:bool=False,fallback_whitelist:Optional[List[str]]=None,blocking_actions:Optional[List[str]]=None,blocking_reasons:Optional[List[str]]=None,config_digest:str=""):
        self.run_id=ensure_run_id(run_id)
        self.root=Path(root_log_dir)
        self.root.mkdir(parents=True,exist_ok=True)
        self.strict_fallbacks=bool(strict_fallbacks)
        self.fallback_whitelist=list(fallback_whitelist or [])
        self.blocking_actions=set(str(x) for x in (blocking_actions or ["synthetic_dolma","dummy_model_fallback","dataset_empty","download_false_empty"]))
        self.blocking_reasons=set(str(x) for x in (blocking_reasons or ["dolma_unavailable","dataset_empty"]))
        self.config_digest=str(config_digest or "")
        self._fallback_rows:List[Dict[str,Any]]=[]
        self._metric_rows:List[Dict[str,Any]]=[]
        self.fallback_path=self.root/"fallbacks.jsonl"
        self.metric_path=self.root/"metrics.jsonl"
    def _base(self,module:str,func:str,event_type:str,action:str,reason:str,timestamp_utc:Optional[str],exception_str:Optional[str],extra_dict:Optional[Dict[str,Any]])->Dict[str,Any]:
        row={
            "run_id":self.run_id,
            "ts_utc":timestamp_utc or _now(),
            "module":str(module),
            "func":str(func),
            "event_type":str(event_type),
            "action":str(action),
            "reason":str(reason),
            "config_digest":self.config_digest,
            "exception":str(exception_str or ""),
            "extra":_jsonable(extra_dict or {})
        }
        return row
    def _append_jsonl(self,path:Path,row:Dict[str,Any])->None:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=True)+"\n")
    def _emit_console(self,row:Dict[str,Any])->None:
        mode=str(os.environ.get("HILDANEXT_TRACE_CONSOLE","plain")).strip().lower()
        if mode in {"0","off","false","none"}:
            return
        if mode=="json":
            print(json.dumps(row,ensure_ascii=True),flush=True)
            return
        ts=str(row.get("ts_utc",""))
        et=str(row.get("event_type","")).upper()
        mod=str(row.get("module",""))
        fn=str(row.get("func",""))
        if et=="METRIC":
            print(f"{ts} {et} {mod}.{fn} name={row.get('name','')} step={row.get('step',None)} value={row.get('value',None)}",flush=True)
            return
        print(f"{ts} {et} {mod}.{fn} action={row.get('action','')} reason={row.get('reason','')}",flush=True)
    def _is_whitelisted(self,row:Dict[str,Any])->bool:
        reason=str(row.get("reason",""))
        if reason not in self.fallback_whitelist:
            return False
        if reason=="dinfer_missing":
            er=str((row.get("extra") or {}).get("engine_requested","")).lower()
            return er!="dinfer"
        return True
    def is_blocking(self,row:Dict[str,Any],numpy_ok:bool|None=None)->bool:
        et=str(row.get("event_type",""))
        if et not in {"fallback","fallback_event","env_issue"}:
            return False
        action=str(row.get("action",""))
        reason=str(row.get("reason",""))
        if action in self.blocking_actions or reason in self.blocking_reasons:
            return True
        if action=="numpy_dll_unavailable":
            return not bool(numpy_ok)
        return False
    def record_fallback(self,event:str,module:str,func:str,action:str,reason:str,exception_str:Optional[str]=None,extra_dict:Optional[Dict[str,Any]]=None,timestamp_utc:Optional[str]=None)->Dict[str,Any]:
        exc=exception_str or ""
        row=self._base(module=module,func=func,event_type=str(event or "fallback"),action=action,reason=reason,timestamp_utc=timestamp_utc,exception_str=exc,extra_dict=extra_dict)
        self._fallback_rows.append(row)
        self._append_jsonl(self.fallback_path,row)
        self._emit_console(row)
        if self.strict_fallbacks and not self._is_whitelisted(row):
            raise RuntimeError(f"strict_fallbacks violation: {reason} @ {module}.{func}")
        return row
    def record_env_issue(self,name:str,detail:str,module:str="env",func:str="record_env_issue",extra_dict:Optional[Dict[str,Any]]=None,timestamp_utc:Optional[str]=None)->Dict[str,Any]:
        row=self._base(module=module,func=func,event_type="env_issue",action=str(name),reason=str(detail),timestamp_utc=timestamp_utc,exception_str="",extra_dict=extra_dict)
        self._fallback_rows.append(row)
        self._append_jsonl(self.fallback_path,row)
        self._emit_console(row)
        return row
    def record_notice(self,module:str,func:str,action:str,reason:str,extra_dict:Optional[Dict[str,Any]]=None,timestamp_utc:Optional[str]=None)->Dict[str,Any]:
        row=self._base(module=module,func=func,event_type="notice",action=action,reason=reason,timestamp_utc=timestamp_utc,exception_str="",extra_dict=extra_dict)
        self._fallback_rows.append(row)
        self._append_jsonl(self.fallback_path,row)
        self._emit_console(row)
        return row
    def record_metric(self,name:str,value:Any,step:Optional[int]=None,module:str="metrics",func:str="record_metric",extra_dict:Optional[Dict[str,Any]]=None,timestamp_utc:Optional[str]=None)->Dict[str,Any]:
        row={
            "run_id":self.run_id,
            "ts_utc":timestamp_utc or _now(),
            "module":str(module),
            "func":str(func),
            "event_type":"metric",
            "name":str(name),
            "value":_jsonable(value),
            "step":None if step is None else int(step),
            "config_digest":self.config_digest,
            "extra":_jsonable(extra_dict or {})
        }
        self._metric_rows.append(row)
        self._append_jsonl(self.metric_path,row)
        self._emit_console(row)
        return row
    def snapshot_fallbacks(self,limit:int=64)->List[Dict[str,Any]]:
        rows=self._fallback_rows[-max(1,int(limit)):]
        return [_jsonable(x) for x in rows]
    def all_events(self)->List[Dict[str,Any]]:
        return [_jsonable(x) for x in self._fallback_rows]
    def count_fallbacks(self)->int:
        return sum(1 for x in self._fallback_rows if str(x.get("event_type","")) in {"fallback","fallback_event"})
    def count_blocking_fallbacks(self)->int:
        return sum(1 for x in self._fallback_rows if str(x.get("event_type","")) in {"fallback","fallback_event"} and (not self._is_whitelisted(x)))
    def flush(self)->None:
        return


def set_active_trace(trace:Optional[RunTrace]):
    return _ACTIVE_TRACE.set(trace)

def reset_active_trace(token)->None:
    _ACTIVE_TRACE.reset(token)

def active_trace()->Optional[RunTrace]:
    return _ACTIVE_TRACE.get()

def use_trace(cfg:Any=None,trace:Optional[RunTrace]=None)->Optional[RunTrace]:
    if trace is not None:
        return trace
    cur=active_trace()
    if cur is not None:
        return cur
    if cfg is None:
        return None
    tr=trace_from_cfg(cfg)
    set_active_trace(tr)
    return tr

def trace_from_cfg(cfg:Any,run_id:Optional[str]=None)->RunTrace:
    rid=ensure_run_id(run_id or getattr(getattr(cfg,"runtime",None),"run_id",""))
    logs_dir=getattr(getattr(cfg,"paths",None),"logs_dir","") or "runs/logs"
    strict=bool(getattr(getattr(cfg,"runtime",None),"strict_fallbacks",False))
    wl=list(getattr(getattr(cfg,"runtime",None),"fallback_whitelist",[]) or ["flash_attention_unavailable","numpy_dll_unavailable","dinfer_missing"])
    ba=list(getattr(getattr(cfg,"runtime",None),"blocking_fallback_actions",[]) or ["synthetic_dolma","dummy_model_fallback","dataset_empty","download_false_empty"])
    br=list(getattr(getattr(cfg,"runtime",None),"blocking_fallback_reasons",[]) or ["dolma_unavailable","dataset_empty"])
    dig=config_digest_from_cfg(cfg)
    tr=RunTrace(run_id=rid,root_log_dir=logs_dir,strict_fallbacks=strict,fallback_whitelist=wl,blocking_actions=ba,blocking_reasons=br,config_digest=dig)
    return tr

def exception_with_stack(err:Exception)->str:
    tb=traceback.format_exc()
    if tb and tb.strip() and "Traceback" in tb:
        return tb.strip()
    return f"{type(err).__name__}: {err}"

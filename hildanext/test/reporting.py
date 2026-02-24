# Structured test payload logging for readiness report generation.
# Main entrypoints: reset_payload_log,emit_payload,load_payloads.
# Writes JSONL artifacts under runs/reports/logs.
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from typing import Any,Dict,List
import json

ROOT=Path(__file__).resolve().parents[1]
LOG_DIR=ROOT/"runs"/"reports"/"logs"
PAYLOAD_PATH=LOG_DIR/"test_payloads_mdm.jsonl"

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

def reset_payload_log()->None:
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    PAYLOAD_PATH.write_text("",encoding="utf-8")

def emit_payload(test_id:str,description:str,payload:Dict[str,Any])->None:
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    row={"ts":_now(),"test_id":test_id,"description":description,"payload":_jsonable(payload)}
    with PAYLOAD_PATH.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=True)+"\n")

def load_payloads()->List[Dict[str,Any]]:
    if not PAYLOAD_PATH.exists():
        return []
    out=[]
    for line in PAYLOAD_PATH.read_text(encoding="utf-8",errors="ignore").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out

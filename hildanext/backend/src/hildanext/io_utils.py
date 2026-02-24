# File IO helpers for JSON/JSONL and directories.
# Main entrypoints: ensure_dir,read_jsonl,write_jsonl.
# Used by datasets,tokenization,training pipelines.
from __future__ import annotations
from pathlib import Path
from typing import Iterable,Any,Dict,List
from datetime import datetime,timezone
import json

def ensure_dir(path:str|Path)->Path:
    p=Path(path)
    p.mkdir(parents=True,exist_ok=True)
    return p

def read_json(path:str|Path)->Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path:str|Path,data:Any)->None:
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=True,indent=2),encoding="utf-8")

def read_jsonl(path:str|Path,max_rows:int|None=None)->List[Dict[str,Any]]:
    out:List[Dict[str,Any]]=[]
    p=Path(path)
    if not p.exists():
        return out
    with p.open("r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            if not line.strip():
                continue
            out.append(json.loads(line))
            if max_rows is not None and i+1>=max_rows:
                break
    return out

def write_jsonl(path:str|Path,rows:Iterable[Dict[str,Any]])->int:
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with p.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=True)+"\n")
            n+=1
    return n

def append_jsonl(path:str|Path,rows:Iterable[Dict[str,Any]])->int:
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with p.open("a",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=True)+"\n")
            n+=1
    return n

def now_iso()->str:
    return datetime.now(timezone.utc).isoformat()

# Minimal benchmark runners for TinyStories and HumanEval-like dummy.
# Main entrypoint: run_benchmarks.
# Designed for pipeline validation, not leaderboard evaluation.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List
from .io_utils import read_jsonl,write_json
from .config import AppConfig

def _load_tinystories_prompts(cfg:AppConfig,n:int)->List[str]:
    rows=read_jsonl(Path(cfg.paths.processed_dir)/"sft_eval.jsonl",max_rows=n)
    out=[]
    for r in rows:
        p=str(r.get("prompt","")).strip()
        if p:
            out.append(p)
    if not out:
        out=[f"Tell a short story about {i}." for i in range(n)]
    return out[:n]

def _load_humaneval_prompts(cfg:AppConfig,n:int)->List[Dict[str,Any]]:
    rows=read_jsonl(Path(cfg.paths.processed_dir)/"humaneval_dummy.jsonl",max_rows=n)
    if rows:
        return rows
    out=[]
    for i in range(n):
        out.append({"id":f"humaneval-{i}","prompt":f"def add_{i}(x:int)->int:\n    return","tests":[f"assert add_{i}(2)=={i+2}"]})
    return out

def run_benchmarks(cfg:AppConfig,engine,max_items:int=8)->Dict[str,Any]:
    tiny_prompts=_load_tinystories_prompts(cfg,max_items)
    tiny_outputs=[engine.generate(p,mode="Q_MODE",max_new_tokens=48,seed=cfg.runtime.seed+i) for i,p in enumerate(tiny_prompts)]
    tiny_non_empty=sum(1 for x in tiny_outputs if x.strip())
    tiny_avg_len=sum(len(x.split()) for x in tiny_outputs)/max(1,len(tiny_outputs))
    human_rows=_load_humaneval_prompts(cfg,max_items)
    human_outputs=[]
    for i,row in enumerate(human_rows):
        out=engine.generate(str(row.get("prompt","")),mode="Q_MODE",max_new_tokens=64,seed=cfg.runtime.seed+100+i)
        human_outputs.append({"id":row.get("id",f"h-{i}"),"output":out})
    human_non_empty=sum(1 for x in human_outputs if x["output"].strip())
    report={
        "tinystories":{"items":len(tiny_prompts),"non_empty":tiny_non_empty,"avg_words":tiny_avg_len},
        "humaneval_dummy":{"items":len(human_outputs),"non_empty":human_non_empty}
    }
    write_json(Path(cfg.paths.logs_dir)/"benchmarks.summary.json",report)
    return report

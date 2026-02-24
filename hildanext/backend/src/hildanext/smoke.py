# End-to-end smoke test for SAFE pipeline.
# Main entrypoint: run_smoke.
# Validates load, conversion step, SFT step, and inference.
from __future__ import annotations
from typing import Any,Dict
from .config import load_config,clone_with_updates
from .io_utils import ensure_dir,write_json
from .datasets import prepare_data
from .tokenization import tokenize_all
from .training import run_wsd_conversion,run_sft_training
from .inference import build_engine,load_model_bundle
from .benchmarks import run_benchmarks
from .trace import trace_from_cfg,set_active_trace,reset_active_trace

def run_smoke(config_path:str)->Dict[str,Any]:
    cfg=load_config(config_path)
    tr=trace_from_cfg(cfg)
    tk=set_active_trace(tr)
    cfg=clone_with_updates(cfg,{"runtime":{"force_dummy_model":False,"use_dinfer":False,"device":"auto"},"train":{"max_steps":1,"accum_steps":1,"batch_size":1},"data":{"seq_len":min(128,cfg.data.seq_len),"max_samples":min(64,cfg.data.max_samples)}})
    for p in [cfg.paths.raw_dir,cfg.paths.processed_dir,cfg.paths.tokenized_dir,cfg.paths.logs_dir,cfg.paths.checkpoints_dir]:
        ensure_dir(p)
    bundle=load_model_bundle(cfg,for_training=False,trace=tr)
    prep=prepare_data(cfg,download=False,max_samples=cfg.data.max_samples,trace=tr)
    tok=tokenize_all(cfg,max_records=cfg.data.max_samples,trace=tr)
    cpt=run_wsd_conversion(cfg,steps=1,trace=tr,resume=True,ckpt_every=1,eval_every=1)
    sft=run_sft_training(cfg,steps=1,trace=tr,resume=True,ckpt_every=1,eval_every=1)
    engine=build_engine(cfg,trace=tr)
    text=engine.generate("Write one safe test line.",mode="S_MODE",max_new_tokens=24,seed=cfg.runtime.seed)
    bench=run_benchmarks(cfg,engine,max_items=4)
    engine.close()
    tr.flush()
    reset_active_trace(tk)
    ok=bool(text.strip()) and bool(cpt.get("steps",0)>=1) and bool(sft.get("steps",0)>=1)
    report={
        "ok":ok,
        "load_model":{"dummy_model":bundle.is_dummy,"reason":bundle.load_reason,"dtype":bundle.actual_dtype,"env_issues":bundle.env_issues},
        "prepare_data":prep.get("counts",{}),
        "tokenize":{k:v.get("records_out",0) for k,v in tok.items() if isinstance(v,dict)},
        "cpt":cpt,
        "sft":sft,
        "sample_generation":text,
        "benchmarks":bench,
        "fallbacks":tr.snapshot_fallbacks(limit=128)
    }
    write_json(f"{cfg.paths.logs_dir}/smoke.report.json",report)
    return report

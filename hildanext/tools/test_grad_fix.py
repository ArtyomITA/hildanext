# Quick test: run a few optimizer steps to verify grad explosion is fixed.
import sys,os,time
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","backend","src"))
os.environ.setdefault("CONDA_DEFAULT_ENV","mdm")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import torch
from hildanext.config import load_config
from hildanext.training import run_wsd_conversion
from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace

cfg_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","runs","configs","llada21_dolma_wsd_only.json"))
print(f"[TEST] config={cfg_path}",flush=True)
cfg=load_config(cfg_path)
tr=trace_from_cfg(cfg)
tk=set_active_trace(tr)
t0=time.perf_counter()
try:
    rep=run_wsd_conversion(cfg,steps=5,trace=tr,resume=False,ckpt_every=999999,eval_every=999999)
finally:
    reset_active_trace(tk)
t1=time.perf_counter()
print(f"\n[TEST] wall={t1-t0:.1f}s steps_completed={rep.get('steps',0)} loss_last={rep.get('loss_last',0):.4f}",flush=True)
print(f"[TEST] exit_reason={rep.get('exit_reason','?')} tokens/s={rep.get('tokens_per_sec',0):.1f}",flush=True)

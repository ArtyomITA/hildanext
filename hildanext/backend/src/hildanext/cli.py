# CLI entrypoint for SAFE backend operations.
# Required commands include preflight and overnight recipe runner.
# All commands are config-driven and trace-enabled.
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .config import load_config
from .datasets import prepare_data
from .tokenization import tokenize_all
from .training import run_wsd_conversion,run_sft_training,merge_topk_checkpoints
from .inference import build_engine
from .benchmarks import run_benchmarks
from .api import run_server
from .smoke import run_smoke
from .audit import run_audit
from .quant import run_quant_bench
from .recipe import preflight,run_recipe_llada21,dinfer_smoke
from .wsd_stage0 import dolma_manifest,prepare_dolma_only,verify_dolma_only,preflight_wsd,run_wsd,archive_runs,create_stage0_config
from .trace import trace_from_cfg,set_active_trace,reset_active_trace
import os
import sys
import traceback


def _print(data):
    print(json.dumps(data,ensure_ascii=True,indent=2))

def _torch_runtime_info()->dict:
    info={
        "torch_version":str(torch.__version__),
        "torch_cuda_build":str(getattr(torch.version,"cuda",None)),
        "cuda_available":bool(torch.cuda.is_available()),
        "cuda_device_count":int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cudnn_available":bool(torch.backends.cudnn.is_available()),
        "cudnn_version":int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "bf16_supported":bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() and hasattr(torch.cuda,"is_bf16_supported") else False
    }
    if torch.cuda.is_available() and torch.cuda.device_count()>0:
        idx=0
        prop=torch.cuda.get_device_properties(idx)
        info.update({
            "cuda_device_index":idx,
            "cuda_device_name":str(prop.name),
            "cuda_capability":f"{int(prop.major)}.{int(prop.minor)}",
            "cuda_total_mem_gb":round(float(prop.total_memory)/1024**3,3)
        })
    return info

def _cfg_trace(config_path:str):
    cfg=load_config(config_path)
    tr=trace_from_cfg(cfg)
    tk=set_active_trace(tr)
    return cfg,tr,tk

def _finalize_trace(tr,tk):
    tr.flush()
    reset_active_trace(tk)

def cmd_prepare_data(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=prepare_data(cfg,download=args.download,max_samples=args.max_samples,trace=tr)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_tokenize(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=tokenize_all(cfg,max_records=args.max_records,trace=tr)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_convert_wsd(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=run_wsd_conversion(cfg,steps=args.steps,trace=tr,resume=args.resume,ckpt_every=args.ckpt_every,eval_every=args.eval_every)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_sft(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=run_sft_training(cfg,steps=args.steps,trace=tr,resume=args.resume,ckpt_every=args.ckpt_every,eval_every=args.eval_every)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_serve(args):
    run_server(config_path=args.config,host=args.host,port=args.port)

def cmd_smoke(args):
    rep=run_smoke(args.config)
    _print(rep)
    print("OK" if rep.get("ok") else "FAIL")

def cmd_generate(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        eng=build_engine(cfg,trace=tr)
        text=eng.generate(prompt=args.prompt,mode=args.mode,tau_mask=args.tau_mask,tau_edit=args.tau_edit,max_new_tokens=args.max_new_tokens,seed=args.seed)
        rep={"text":text,"stats":eng.last_stats,"engine":eng.name}
        eng.close()
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_benchmark(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        eng=build_engine(cfg,trace=tr)
        rep=run_benchmarks(cfg,eng,max_items=args.max_items)
        eng.close()
        rep["fallbacks"]=tr.snapshot_fallbacks(limit=128)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_merge_topk(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=merge_topk_checkpoints(cfg,checkpoint_dirs=args.checkpoints,output_dir=args.output_dir)
        rep["fallbacks"]=tr.snapshot_fallbacks(limit=64)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_audit(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        root=Path(cfg.paths.root) if cfg.paths.root else Path(args.config).resolve().parents[2]
        out_md=Path(args.out_md) if args.out_md else root/"runs"/"reports"/"formula_audit.md"
        out_json=Path(args.out_json) if args.out_json else root/"runs"/"reports"/"formula_audit.json"
        rep=run_audit(root,out_md,out_json)
        mirror=root/"docs"/"reports"/"formula_audit.md"
        mirror.parent.mkdir(parents=True,exist_ok=True)
        mirror.write_text(out_md.read_text(encoding="utf-8",errors="ignore"),encoding="utf-8")
        rep["paths"]={"md":str(out_md),"json":str(out_json),"docs_md":str(mirror)}
        rep["fallbacks"]=tr.snapshot_fallbacks(limit=128)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_quant_bench(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        modes=[x.strip() for x in args.modes.split(",") if x.strip()]
        out_json=Path(args.out_json) if args.out_json else Path(cfg.paths.root)/"runs"/"reports"/"quant_vram.json"
        rep=run_quant_bench(cfg,modes=modes,prompt=args.prompt,max_new_tokens=args.max_new_tokens,engine_name=args.engine,seed=args.seed,out_json=out_json,train_probe=args.train_probe,trace=tr)
        rep["paths"]={"json":str(out_json)}
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_preflight(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=preflight(cfg,trace=tr)
        _print(rep)
        print("OK" if rep.get("ok") else "FAIL")
    finally:
        _finalize_trace(tr,tk)

def cmd_run_recipe(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=run_recipe_llada21(cfg,trace=tr)
        _print(rep)
        print("OK" if rep.get("ok") else "FAIL")
    finally:
        _finalize_trace(tr,tk)

def cmd_dinfer_smoke(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=dinfer_smoke(cfg,trace=tr)
        _print(rep)
        print("OK" if rep.get("ok") else "SKIP")
    finally:
        _finalize_trace(tr,tk)

def cmd_make_stage0_config(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        out=Path(args.out_config) if args.out_config else Path(cfg.paths.root)/"runs"/"configs"/"llada21_dolma_wsd_only.json"
        dolma=args.dolma_path if args.dolma_path else cfg.data.dolma_path
        new_cfg=create_stage0_config(cfg,out,dolma_path=dolma)
        rep={"ok":True,"out_config":str(out),"dolma_path":new_cfg.data.dolma_path,"run_id":tr.run_id}
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_dolma_manifest(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=dolma_manifest(cfg,trace=tr)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_dolma_prep(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=prepare_dolma_only(cfg,trace=tr)
        _print(rep)
    finally:
        _finalize_trace(tr,tk)

def cmd_dolma_verify(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=verify_dolma_only(cfg,trace=tr)
        _print(rep)
        print("OK" if rep.get("ok") else "FAIL")
    finally:
        _finalize_trace(tr,tk)

def cmd_preflight_wsd(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        rep=preflight_wsd(cfg,trace=tr)
        _print(rep)
        print("OK PREPARED")
    finally:
        _finalize_trace(tr,tk)

def cmd_run_wsd(args):
    cfg,tr,tk=_cfg_trace(args.config)
    try:
        if not args.no_archive:
            ar=archive_runs(cfg,trace=tr)
            _print({"archive":ar})
        if not args.skip_preflight:
            pf=preflight_wsd(cfg,trace=tr)
            _print({"preflight":pf})
        rep=run_wsd(cfg,args.config,trace=tr,skip_dolma_prep=bool(getattr(args,"skip_dolma_prep",False)))
        _print(rep)
        print("OK" if rep.get("ok") else "FAIL")
    finally:
        _finalize_trace(tr,tk)

def cmd_run_stage0_inline(args):
    base_cfg=load_config(args.config)
    out_cfg_path=Path(args.out_config) if args.out_config else Path(args.config)
    dolma=args.dolma_path if args.dolma_path else base_cfg.data.dolma_path
    if args.doc_index_path:
        os.environ["HILDANEXT_DOLMA_DOC_INDEX_PATH"]=args.doc_index_path
    cfg_for_run=create_stage0_config(base_cfg,out_cfg_path,dolma_path=dolma)
    tr=trace_from_cfg(cfg_for_run)
    tk=set_active_trace(tr)
    try:
        print(f"stage=inline event=start run_id={tr.run_id} env={os.environ.get('CONDA_DEFAULT_ENV','')} python={sys.executable}",flush=True)
        print(f"stage=inline config_path={out_cfg_path} dolma_path={cfg_for_run.data.dolma_path} doc_index_path={os.environ.get('HILDANEXT_DOLMA_DOC_INDEX_PATH','')}",flush=True)
        ti=_torch_runtime_info()
        print(f"torch.version={ti['torch_version']} torch.cuda_build={ti['torch_cuda_build']} cuda_available={ti['cuda_available']} cuda_device_count={ti['cuda_device_count']} cudnn_available={ti['cudnn_available']} cudnn_version={ti['cudnn_version']} bf16_supported={ti['bf16_supported']}",flush=True)
        if ti.get("cuda_available"):
            print(f"cuda.device={ti.get('cuda_device_name')} capability={ti.get('cuda_capability')} vram_gb={ti.get('cuda_total_mem_gb')}",flush=True)
        if str(os.environ.get("CONDA_DEFAULT_ENV","")).lower()!="mdm":
            raise RuntimeError("env_not_mdm_activate_conda_mdm_first")
        cp=str(os.environ.get("CONDA_PREFIX","")).strip()
        if cp:
            exe_norm=Path(sys.executable).resolve().as_posix().lower()
            cp_norm=Path(cp).resolve().as_posix().lower()
            if not exe_norm.startswith(cp_norm):
                raise RuntimeError(f"python_not_from_conda_env exe={sys.executable} conda_prefix={cp}")
        if not args.no_archive:
            ar=archive_runs(cfg_for_run,trace=tr)
            print(f"stage=inline event=archive archive_dir={ar.get('archive_dir','')} ops={len(ar.get('ops',[]))}",flush=True)
        if not args.skip_preflight:
            pf=preflight_wsd(cfg_for_run,trace=tr)
            print(f"stage=inline event=preflight ok={bool(pf.get('ok'))} fallbacks_blocking_count={int(pf.get('fallbacks_blocking_count',0))}",flush=True)
        if args.no_run:
            print(f"stage=inline event=no_run_stop ok=True run_id={tr.run_id}",flush=True)
            print("OK",flush=True)
            return
        rep=run_wsd(cfg_for_run,str(out_cfg_path),trace=tr)
        print(f"stage=inline event=done ok={bool(rep.get('ok'))} run_id={tr.run_id} checkpoints={len(rep.get('checkpoints_paths',[]))} fallbacks_blocking_count={int(rep.get('fallbacks_blocking_count',0))}",flush=True)
        print("OK",flush=True)
    except Exception as e:
        print(f"stage=inline event=error error={str(e)}",flush=True)
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        _finalize_trace(tr,tk)

def build_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="hildanext")
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("prepare-data")
    s.add_argument("--config",required=True)
    s.add_argument("--download",action="store_true")
    s.add_argument("--max-samples",type=int,default=None)
    s.set_defaults(func=cmd_prepare_data)
    s=sub.add_parser("tokenize")
    s.add_argument("--config",required=True)
    s.add_argument("--max-records",type=int,default=None)
    s.set_defaults(func=cmd_tokenize)
    s=sub.add_parser("convert-wsd")
    s.add_argument("--config",required=True)
    s.add_argument("--steps",type=int,default=None)
    s.add_argument("--resume",action="store_true")
    s.add_argument("--ckpt-every",type=int,default=None)
    s.add_argument("--eval-every",type=int,default=None)
    s.set_defaults(func=cmd_convert_wsd)
    s=sub.add_parser("sft")
    s.add_argument("--config",required=True)
    s.add_argument("--steps",type=int,default=None)
    s.add_argument("--resume",action="store_true")
    s.add_argument("--ckpt-every",type=int,default=None)
    s.add_argument("--eval-every",type=int,default=None)
    s.set_defaults(func=cmd_sft)
    s=sub.add_parser("serve")
    s.add_argument("--config",required=True)
    s.add_argument("--host",default="127.0.0.1")
    s.add_argument("--port",type=int,default=8080)
    s.set_defaults(func=cmd_serve)
    s=sub.add_parser("smoke-test")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_smoke)
    s=sub.add_parser("generate")
    s.add_argument("--config",required=True)
    s.add_argument("--prompt",required=True)
    s.add_argument("--mode",default="S_MODE")
    s.add_argument("--tau-mask",type=float,default=None)
    s.add_argument("--tau-edit",type=float,default=None)
    s.add_argument("--max-new-tokens",type=int,default=96)
    s.add_argument("--seed",type=int,default=None)
    s.set_defaults(func=cmd_generate)
    s=sub.add_parser("benchmark")
    s.add_argument("--config",required=True)
    s.add_argument("--max-items",type=int,default=8)
    s.set_defaults(func=cmd_benchmark)
    s=sub.add_parser("merge-topk")
    s.add_argument("--config",required=True)
    s.add_argument("--output-dir",required=True)
    s.add_argument("checkpoints",nargs="+")
    s.set_defaults(func=cmd_merge_topk)
    s=sub.add_parser("audit")
    s.add_argument("--config",required=True)
    s.add_argument("--out-md",default=None)
    s.add_argument("--out-json",default=None)
    s.set_defaults(func=cmd_audit)
    s=sub.add_parser("quant-bench")
    s.add_argument("--config",required=True)
    s.add_argument("--prompt",default="Write one short safe line.")
    s.add_argument("--max-new-tokens",type=int,default=24)
    s.add_argument("--modes",default="fp16,nf4,int8")
    s.add_argument("--engine",choices=["transformers","dinfer"],default="transformers")
    s.add_argument("--seed",type=int,default=None)
    s.add_argument("--train-probe",action="store_true")
    s.add_argument("--out-json",default=None)
    s.set_defaults(func=cmd_quant_bench)
    s=sub.add_parser("preflight")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_preflight)
    s=sub.add_parser("run-recipe-llada21")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_run_recipe)
    s=sub.add_parser("dinfer-smoke")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_dinfer_smoke)
    s=sub.add_parser("make-stage0-config")
    s.add_argument("--config",required=True)
    s.add_argument("--out-config",default=None)
    s.add_argument("--dolma-path",default=None)
    s.set_defaults(func=cmd_make_stage0_config)
    s=sub.add_parser("dolma-manifest")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_dolma_manifest)
    s=sub.add_parser("dolma-prep")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_dolma_prep)
    s=sub.add_parser("dolma-verify")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_dolma_verify)
    s=sub.add_parser("preflight-wsd")
    s.add_argument("--config",required=True)
    s.set_defaults(func=cmd_preflight_wsd)
    s=sub.add_parser("run-wsd")
    s.add_argument("--config",required=True)
    s.add_argument("--skip-preflight",action="store_true")
    s.add_argument("--no-archive",action="store_true")
    s.add_argument("--skip-dolma-prep",action="store_true",help="Skip dolma prep/verify; assume tokenized files already exist")
    s.set_defaults(func=cmd_run_wsd)
    s=sub.add_parser("run-stage0-inline")
    s.add_argument("--config",required=True)
    s.add_argument("--out-config",default=None)
    s.add_argument("--dolma-path",default=None)
    s.add_argument("--doc-index-path",default=None)
    s.add_argument("--skip-preflight",action="store_true")
    s.add_argument("--no-archive",action="store_true")
    s.add_argument("--no-run",action="store_true")
    s.set_defaults(func=cmd_run_stage0_inline)
    return p

def main():
    parser=build_parser()
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()

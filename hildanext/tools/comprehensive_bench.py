# Comprehensive WSD benchmark: optimizers, VRAM, speed, grad norms, embed noise fix.
# Tests all available optimizers on real data, measures per-micro-batch throughput,
# peak VRAM, gradient behavior, and validates the embed noise hook fixes grad explosion.
# Usage: python comprehensive_bench.py [--micro-batches N] [--opt-steps N]
import sys,os,time,argparse,json,gc,traceback
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","backend","src"))

_RESULTS=[]

def _gpu_temp():
    try:
        import subprocess
        r=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader"],capture_output=True,text=True,timeout=5)
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1

def _test_optimizer(name,model,cfg,bundle,loader_iter,n_micro,n_opt_steps,use_amp,ct_t_min,ct_t_max,embed_noise=False):
    import torch
    from hildanext.diffusion import compute_m2t_t2t_losses,_install_embed_noise_hook,_remove_embed_noise_hook,set_embed_noise_std
    device=bundle.device
    mask_mode=cfg.llada2.mask_mode
    block_size=cfg.llada2.composite_block_size
    vocab_cap=max(8,bundle.vocab_size)
    model.train()
    # Create optimizer
    lr=float(cfg.train.lr)
    wd=float(cfg.train.weight_decay)
    betas=(0.9,0.95)
    opt=None
    opt_label=name
    try:
        if name=="PagedAdamW8bit":
            import bitsandbytes as bnb
            opt=bnb.optim.PagedAdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas)
        elif name=="AdamW8bit":
            import bitsandbytes as bnb
            opt=bnb.optim.AdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas)
        elif name=="Lion8bit":
            import bitsandbytes as bnb
            if hasattr(bnb.optim,"Lion8bit"):
                opt=bnb.optim.Lion8bit(model.parameters(),lr=lr*0.1,weight_decay=wd,betas=(0.9,0.99))
            else:
                return {"name":name,"status":"unavailable","reason":"bnb.optim.Lion8bit not found"}
        elif name=="Adafactor":
            from transformers.optimization import Adafactor
            opt=Adafactor(model.parameters(),lr=lr,relative_step=False,scale_parameter=False,warmup_init=False,weight_decay=wd)
        elif name=="AdamW_fused":
            from torch.optim import AdamW
            opt=AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=betas,fused=True)
        elif name=="AdamW_vanilla":
            from torch.optim import AdamW
            opt=AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=betas,fused=False)
        elif name=="SGD":
            opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
        else:
            return {"name":name,"status":"unavailable","reason":f"unknown optimizer: {name}"}
    except Exception as e:
        return {"name":name,"status":"unavailable","reason":str(e)[:200]}
    if embed_noise:
        _install_embed_noise_hook(model,bundle.mask_id,noise_std=0.1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    # Warmup 2 micro-batches
    for w in range(2):
        batch=next(loader_iter)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=ct_t_min,t_max=ct_t_max)
        out["loss"].backward()
        opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    # Timed micro-batches
    fwd_times=[]
    bwd_times=[]
    opt_times=[]
    losses=[]
    grad_norms=[]
    t_values=[]
    mask_ratios=[]
    accs=[]
    total_tokens=0
    opt_step_count=0
    grad_acc=max(1,n_opt_steps)
    t_all=time.perf_counter()
    for i in range(n_micro):
        batch=next(loader_iter)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        total_tokens+=int(batch["attention_mask"].sum().item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=ct_t_min,t_max=ct_t_max)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1=time.perf_counter()
        loss=out["loss"]/float(grad_acc)
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2=time.perf_counter()
        fwd_times.append(t1-t0)
        bwd_times.append(t2-t1)
        losses.append(float(out["loss"].detach().item()))
        t_values.append(float(out.get("t_sampled",0.5)))
        mask_ratios.append(float(out.get("mask_ratio_actual",0.0)))
        acc_v=out.get("masked_token_acc")
        if acc_v is not None:
            accs.append(float(acc_v))
        # Optimizer step every grad_acc micro-batches
        if (i+1)%grad_acc==0:
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
            grad_norms.append(gn)
            if gn<100.0:
                t_opt0=time.perf_counter()
                opt.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_opt1=time.perf_counter()
                opt_times.append(t_opt1-t_opt0)
                opt_step_count+=1
            else:
                opt_times.append(0.0)
            opt.zero_grad(set_to_none=True)
            # Decay embed noise linearly
            if embed_noise and opt_step_count>0:
                frac=max(0.0,1.0-float(opt_step_count)/float(max(1,n_micro//grad_acc)))
                set_embed_noise_std(0.1*frac)
    t_all_end=time.perf_counter()
    wall=t_all_end-t_all
    peak_vram=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0
    if embed_noise:
        _remove_embed_noise_hook()
    # Compute VRAM for optimizer states
    vram_after_opt=torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
    import math
    result={
        "name":opt_label,
        "status":"ok",
        "embed_noise":embed_noise,
        "micro_batches":n_micro,
        "opt_steps_completed":opt_step_count,
        "opt_steps_skipped":len(grad_norms)-opt_step_count,
        "wall_time_s":round(wall,2),
        "avg_fwd_s":round(sum(fwd_times)/max(1,len(fwd_times)),4),
        "avg_bwd_s":round(sum(bwd_times)/max(1,len(bwd_times)),4),
        "avg_opt_step_s":round(sum(opt_times)/max(1,len(opt_times)),4) if opt_times else 0,
        "avg_micro_batch_s":round((sum(fwd_times)+sum(bwd_times))/max(1,len(fwd_times)),4),
        "tokens_per_sec":round(total_tokens/max(0.001,wall),1),
        "peak_vram_mb":round(peak_vram,0),
        "vram_current_mb":round(vram_after_opt,0),
        "grad_norms":{"mean":round(sum(grad_norms)/max(1,len(grad_norms)),2) if grad_norms else 0,"min":round(min(grad_norms),2) if grad_norms else 0,"max":round(max(grad_norms),2) if grad_norms else 0,"all":[round(g,2) for g in grad_norms]},
        "losses":{"mean":round(sum(losses)/max(1,len(losses)),4),"min":round(min(losses),4) if losses else 0,"max":round(max(losses),4) if losses else 0,"last":round(losses[-1],4) if losses else 0,"finite_count":sum(1 for l in losses if math.isfinite(l)),"total":len(losses)},
        "t_sampled":{"mean":round(sum(t_values)/max(1,len(t_values)),4) if t_values else 0,"min":round(min(t_values),4) if t_values else 0,"max":round(max(t_values),4) if t_values else 0},
        "mask_ratio":{"mean":round(sum(mask_ratios)/max(1,len(mask_ratios)),4) if mask_ratios else 0},
        "accuracy":{"mean":round(sum(accs)/max(1,len(accs)),4) if accs else 0},
        "gpu_temp_c":_gpu_temp()
    }
    print(f"  [{opt_label}] done: tok/s={result['tokens_per_sec']} peak_vram={result['peak_vram_mb']}MB grad_norm_mean={result['grad_norms']['mean']} loss_mean={result['losses']['mean']} opt_steps={opt_step_count}",flush=True)
    opt.zero_grad(set_to_none=True)
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result

def main():
    ap=argparse.ArgumentParser(description="Comprehensive WSD Benchmark")
    ap.add_argument("--micro-batches",type=int,default=20,help="micro-batches per optimizer test")
    ap.add_argument("--opt-steps",type=int,default=4,help="grad accum steps (micro-batches per opt step)")
    ap.add_argument("--config",type=str,default=None)
    ap.add_argument("--skip-optimizers",type=str,default="",help="comma-separated optimizer names to skip")
    args=ap.parse_args()
    N=args.micro_batches
    grad_acc=args.opt_steps
    skip=set(x.strip() for x in args.skip_optimizers.split(",") if x.strip())
    cfg_path=args.config or os.path.join(os.path.dirname(__file__),"..","runs","configs","llada21_dolma_wsd_only.json")
    cfg_path=os.path.abspath(cfg_path)
    print(f"{'='*70}",flush=True)
    print(f"COMPREHENSIVE WSD BENCHMARK",flush=True)
    print(f"{'='*70}",flush=True)
    print(f"config={cfg_path}",flush=True)
    print(f"micro_batches={N} per optimizer, grad_accum={grad_acc}",flush=True)
    print(f"skip={skip}",flush=True)
    print(f"",flush=True)
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, aborting",flush=True)
        return
    print(f"GPU: {torch.cuda.get_device_name()} VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB",flush=True)
    print(f"PyTorch: {torch.__version__}",flush=True)
    print(f"GPU temp: {_gpu_temp()}C",flush=True)
    # Check available optimizers
    print(f"\n--- Checking available optimizers ---",flush=True)
    avail_opts=["AdamW_fused","AdamW_vanilla"]
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}",flush=True)
        avail_opts.insert(0,"PagedAdamW8bit")
        avail_opts.insert(1,"AdamW8bit")
        if hasattr(bnb.optim,"Lion8bit"):
            avail_opts.append("Lion8bit")
    except Exception as e:
        print(f"bitsandbytes: unavailable ({e})",flush=True)
    try:
        from transformers.optimization import Adafactor
        avail_opts.append("Adafactor")
        print(f"Adafactor: available",flush=True)
    except:
        print(f"Adafactor: unavailable",flush=True)
    avail_opts.append("SGD")
    print(f"Will test: {[o for o in avail_opts if o not in skip]}",flush=True)
    print(f"Skipping: {[o for o in avail_opts if o in skip]}",flush=True)
    # Load config, data, model
    print(f"\n--- Loading config, data, model ---",flush=True)
    from hildanext.config import load_config
    from hildanext.training import MmapShardedDataset,_collate
    from hildanext.inference import load_model_bundle
    from hildanext.tokenization import ensure_mask_token
    from hildanext.utils import force_math_sdpa,seed_everything
    from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace
    from torch.utils.data import DataLoader
    from pathlib import Path
    cfg=load_config(cfg_path)
    tr=trace_from_cfg(cfg)
    tk=set_active_trace(tr)
    seed_everything(42)
    force_math_sdpa()
    ct_t_min=float(cfg.experiment.t_min) if hasattr(cfg,"experiment") else 0.05
    ct_t_max=float(cfg.experiment.t_max) if hasattr(cfg,"experiment") else 0.95
    print(f"t_min={ct_t_min} t_max={ct_t_max}",flush=True)
    dolma_root=str(Path(cfg.data.dolma_path).parent)
    ds=MmapShardedDataset(dolma_root)
    print(f"Dataset: {len(ds)} rows",flush=True)
    # Need enough data for all optimizer tests: (N+2) * len(avail_opts) * 2 (with/without embed noise)
    total_needed=(N+4)*len(avail_opts)*2+10
    loader=DataLoader(dataset=ds,batch_size=1,shuffle=True,num_workers=0,collate_fn=_collate,pin_memory=True)
    print(f"DataLoader: batches_available={len(loader)} needed_approx={total_needed}",flush=True)
    bundle=load_model_bundle(cfg,for_training=True,trace=tr)
    ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
    model=bundle.model
    device=bundle.device
    use_amp=device.type=="cuda"
    print(f"Model: {bundle.model_name_or_path} params={sum(p.numel() for p in model.parameters()):,} dtype={bundle.actual_dtype}",flush=True)
    # Baseline VRAM (model only)
    vram_model=torch.cuda.memory_allocated()/1024**2
    print(f"VRAM model-only: {vram_model:.0f}MB",flush=True)
    print(f"\n{'='*70}",flush=True)
    print(f"PHASE 1: Optimizer comparison WITHOUT embed noise",flush=True)
    print(f"  (this tests raw optimizer performance)",flush=True)
    print(f"{'='*70}",flush=True)
    all_results=[]
    loader_it=iter(loader)
    for opt_name in avail_opts:
        if opt_name in skip:
            continue
        print(f"\n--- Testing: {opt_name} (no embed noise) ---",flush=True)
        seed_everything(42)
        try:
            r=_test_optimizer(opt_name,model,cfg,bundle,loader_it,N,grad_acc,use_amp,ct_t_min,ct_t_max,embed_noise=False)
        except StopIteration:
            print(f"  [{opt_name}] ran out of data, recreating loader",flush=True)
            loader_it=iter(loader)
            try:
                r=_test_optimizer(opt_name,model,cfg,bundle,loader_it,N,grad_acc,use_amp,ct_t_min,ct_t_max,embed_noise=False)
            except Exception as e:
                r={"name":opt_name,"status":"error","reason":str(e)[:300]}
                traceback.print_exc()
        except Exception as e:
            r={"name":opt_name,"status":"error","reason":str(e)[:300]}
            traceback.print_exc()
        all_results.append(r)
    print(f"\n{'='*70}",flush=True)
    print(f"PHASE 2: Grad explosion fix test (WITH embed noise hook)",flush=True)
    print(f"  (tests PagedAdamW8bit only, with embed noise to fix grad explosion)",flush=True)
    print(f"{'='*70}",flush=True)
    opt_for_noise="PagedAdamW8bit"
    if opt_for_noise not in skip:
        print(f"\n--- Testing: {opt_for_noise} WITH embed noise ---",flush=True)
        seed_everything(42)
        try:
            loader_it=iter(loader)
            r=_test_optimizer(opt_for_noise,model,cfg,bundle,loader_it,N,grad_acc,use_amp,ct_t_min,ct_t_max,embed_noise=True)
            r["name"]=f"{opt_for_noise}+embed_noise"
        except Exception as e:
            r={"name":f"{opt_for_noise}+embed_noise","status":"error","reason":str(e)[:300]}
            traceback.print_exc()
        all_results.append(r)
    print(f"\n{'='*70}",flush=True)
    print(f"PHASE 3: t_min/t_max sensitivity (t_min=0.001 vs 0.05 vs 0.1)",flush=True)
    print(f"{'='*70}",flush=True)
    for test_tmin,test_tmax,label in [(0.001,1.0,"tmin001"),(0.05,0.95,"tmin005"),(0.1,0.9,"tmin010"),(0.2,0.8,"tmin020")]:
        if "PagedAdamW8bit" in skip:
            continue
        print(f"\n--- Testing: t_min={test_tmin} t_max={test_tmax} ---",flush=True)
        seed_everything(42)
        try:
            loader_it=iter(loader)
            r=_test_optimizer("PagedAdamW8bit",model,cfg,bundle,loader_it,min(N,12),grad_acc,use_amp,test_tmin,test_tmax,embed_noise=True)
            r["name"]=f"t_sensitivity_{label}"
            r["t_min_tested"]=test_tmin
            r["t_max_tested"]=test_tmax
        except Exception as e:
            r={"name":f"t_sensitivity_{label}","status":"error","reason":str(e)[:300]}
            traceback.print_exc()
        all_results.append(r)
    # Final report
    print(f"\n{'='*70}",flush=True)
    print(f"FINAL REPORT",flush=True)
    print(f"{'='*70}",flush=True)
    print(f"",flush=True)
    print(f"{'Optimizer':<30} {'Status':<8} {'tok/s':>8} {'peak_MB':>8} {'avg_fwd':>8} {'avg_bwd':>8} {'avg_opt':>8} {'grad_nm':>8} {'loss':>10} {'opt_ok':>6} {'opt_skip':>8}",flush=True)
    print(f"{'-'*130}",flush=True)
    for r in all_results:
        if r["status"]!="ok":
            print(f"{r['name']:<30} {r['status']:<8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>6} {r.get('reason','')[:40]}",flush=True)
            continue
        print(f"{r['name']:<30} {r['status']:<8} {r['tokens_per_sec']:>8.1f} {r['peak_vram_mb']:>8.0f} {r['avg_fwd_s']:>8.4f} {r['avg_bwd_s']:>8.4f} {r['avg_opt_step_s']:>8.4f} {r['grad_norms']['mean']:>8.2f} {r['losses']['mean']:>10.4f} {r['opt_steps_completed']:>6} {r['opt_steps_skipped']:>8}",flush=True)
    # GRAD EXPLOSION ANALYSIS
    print(f"\n--- GRAD EXPLOSION ANALYSIS ---",flush=True)
    for r in all_results:
        if r["status"]!="ok":
            continue
        gn=r["grad_norms"]
        exploded=gn["max"]>100.0
        status_icon="EXPLODED" if exploded else "OK"
        print(f"  {r['name']:<30} grad_norm: mean={gn['mean']:.2f} min={gn['min']:.2f} max={gn['max']:.2f} => {status_icon}",flush=True)
        if gn.get("all"):
            print(f"    per-step: {gn['all']}",flush=True)
    # VRAM ANALYSIS
    print(f"\n--- VRAM ANALYSIS ---",flush=True)
    print(f"  Model base VRAM: {vram_model:.0f}MB",flush=True)
    for r in all_results:
        if r["status"]!="ok":
            continue
        overhead=r["peak_vram_mb"]-vram_model
        print(f"  {r['name']:<30} peak={r['peak_vram_mb']:.0f}MB overhead={overhead:.0f}MB current={r['vram_current_mb']:.0f}MB",flush=True)
    # RECOMMENDATIONS
    print(f"\n--- RECOMMENDATIONS ---",flush=True)
    ok_results=[r for r in all_results if r["status"]=="ok" and not r["name"].startswith("t_sensitivity")]
    if ok_results:
        best_speed=max(ok_results,key=lambda r:r["tokens_per_sec"])
        best_vram=min(ok_results,key=lambda r:r["peak_vram_mb"])
        best_stable=min([r for r in ok_results if r["opt_steps_completed"]>0] or ok_results,key=lambda r:r["grad_norms"]["max"])
        print(f"  Fastest:       {best_speed['name']} ({best_speed['tokens_per_sec']:.1f} tok/s)",flush=True)
        print(f"  Lowest VRAM:   {best_vram['name']} ({best_vram['peak_vram_mb']:.0f}MB)",flush=True)
        print(f"  Most stable:   {best_stable['name']} (max grad_norm={best_stable['grad_norms']['max']:.2f})",flush=True)
    # Save full results
    out_path=os.path.join(os.path.dirname(__file__),"comprehensive_bench_results.json")
    with open(out_path,"w") as f:
        json.dump({"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"config":cfg_path,"micro_batches_per_test":N,"grad_accum":grad_acc,"model_vram_mb":round(vram_model,0),"t_min_default":ct_t_min,"t_max_default":ct_t_max,"gpu":torch.cuda.get_device_name(),"pytorch":torch.__version__,"results":all_results},f,indent=2)
    print(f"\nFull results saved to: {out_path}",flush=True)
    reset_active_trace(tk)

if __name__=="__main__":
    main()

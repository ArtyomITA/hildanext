# Comprehensive WSD/LLaDA benchmark suite.
# Tests: all optimizers, VRAM, speed, grad norms, embed noise fix.
# Usage: python bench_comprehensive.py [--micro-batches N] [--skip-opt NAME]
import sys,os,time,argparse,json,gc
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","backend","src"))
os.environ.setdefault("CONDA_DEFAULT_ENV","mdm")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

def _gpu_temp():
    try:
        import subprocess
        r=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
        return int(r.stdout.strip().split("\n")[0])
    except:
        return -1

def _vram_stats():
    import torch
    if not torch.cuda.is_available():
        return {"alloc_mb":0,"reserved_mb":0,"peak_mb":0}
    return {
        "alloc_mb":round(torch.cuda.memory_allocated()/1024**2,1),
        "reserved_mb":round(torch.cuda.memory_reserved()/1024**2,1),
        "peak_mb":round(torch.cuda.max_memory_allocated()/1024**2,1)
    }

def _reset_vram():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def run_optimizer_benchmark(model,loader,bundle,cfg,opt,opt_name,n_micro,device,use_amp,embed_noise_hook=False):
    import torch
    from hildanext.diffusion import compute_m2t_t2t_losses,_install_embed_noise_hook,_remove_embed_noise_hook,set_embed_noise_std
    mask_mode=cfg.llada2.mask_mode
    block_size=cfg.llada2.composite_block_size
    vocab_cap=max(8,bundle.vocab_size)
    grad_acc=4
    exp=cfg.experiment if hasattr(cfg,"experiment") else None
    t_min=float(getattr(exp,"t_min",0.05)) if exp else 0.05
    t_max=float(getattr(exp,"t_max",0.95)) if exp else 0.95
    # Install embed noise hook if requested
    if embed_noise_hook:
        _install_embed_noise_hook(model,bundle.mask_id,noise_std=0.1)
    model.train()
    model.zero_grad(set_to_none=True)
    _reset_vram()
    # Warmup 2 micro-batches (not timed)
    it=iter(loader)
    for w in range(2):
        batch=next(it)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=t_min,t_max=t_max)
        (out["loss"]/grad_acc).backward()
    model.zero_grad(set_to_none=True)
    _reset_vram()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Timed run: n_micro micro-batches with optimizer steps every grad_acc
    fwd_times=[]
    bwd_times=[]
    opt_times=[]
    grad_norms=[]
    losses_raw=[]
    losses_scaled=[]
    m2t_losses=[]
    t2t_losses=[]
    accs=[]
    t_samples=[]
    total_tokens=0
    opt_steps_done=0
    skipped_grad_explosion=0
    t_all_start=time.perf_counter()
    for i in range(n_micro):
        try:
            batch=next(it)
        except StopIteration:
            it=iter(loader)
            batch=next(it)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        total_tokens+=int(batch["attention_mask"].sum().item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=t_min,t_max=t_max)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1=time.perf_counter()
        loss=out["loss"]/grad_acc
        t2=time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3=time.perf_counter()
        fwd_times.append(t1-t0)
        bwd_times.append(t3-t2)
        raw_loss=float(out["loss"].detach().item())
        losses_raw.append(raw_loss)
        losses_scaled.append(float(out.get("loss_m2t_scaled",out["loss_m2t"]).detach().item()) if "loss_m2t_scaled" in out else raw_loss)
        m2t_losses.append(float(out["loss_m2t"].detach().item()))
        t2t_losses.append(float(out["loss_t2t"].detach().item()))
        acc=out.get("masked_token_acc")
        if acc is not None:
            accs.append(float(acc))
        t_samples.append(float(out.get("t_sampled",0)))
        # Optimizer step every grad_acc
        if (i+1)%grad_acc==0:
            import math
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
            grad_norms.append(gn)
            if not math.isfinite(gn) or gn>100.0:
                skipped_grad_explosion+=1
                model.zero_grad(set_to_none=True)
                ot=0.0
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t4=time.perf_counter()
                opt.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t5=time.perf_counter()
                ot=t5-t4
                opt_steps_done+=1
            opt_times.append(ot)
            model.zero_grad(set_to_none=True)
        if (i+1)%5==0:
            vram=_vram_stats()
            print(f"  [{i+1}/{n_micro}] fwd={fwd_times[-1]:.3f}s bwd={bwd_times[-1]:.3f}s loss={raw_loss:.4f} vram_peak={vram['peak_mb']:.0f}MB",flush=True)
    t_all_end=time.perf_counter()
    wall=t_all_end-t_all_start
    if embed_noise_hook:
        _remove_embed_noise_hook()
    vram_final=_vram_stats()
    result={
        "optimizer":opt_name,
        "micro_batches":n_micro,
        "wall_time_s":round(wall,2),
        "avg_fwd_s":round(sum(fwd_times)/len(fwd_times),3) if fwd_times else 0,
        "avg_bwd_s":round(sum(bwd_times)/len(bwd_times),3) if bwd_times else 0,
        "avg_opt_step_s":round(sum(opt_times)/len(opt_times),4) if opt_times else 0,
        "avg_micro_batch_s":round((sum(fwd_times)+sum(bwd_times))/len(fwd_times),3) if fwd_times else 0,
        "tokens_per_sec":round(total_tokens/wall,1) if wall>0 else 0,
        "total_tokens":total_tokens,
        "opt_steps_completed":opt_steps_done,
        "skipped_grad_explosion":skipped_grad_explosion,
        "grad_norms":{
            "mean":round(sum(grad_norms)/len(grad_norms),2) if grad_norms else 0,
            "min":round(min(grad_norms),2) if grad_norms else 0,
            "max":round(max(grad_norms),2) if grad_norms else 0,
            "all":[round(x,2) for x in grad_norms]
        },
        "losses":{
            "mean_raw":round(sum(losses_raw)/len(losses_raw),4) if losses_raw else 0,
            "mean_m2t":round(sum(m2t_losses)/len(m2t_losses),4) if m2t_losses else 0,
            "mean_t2t":round(sum(t2t_losses)/len(t2t_losses),4) if t2t_losses else 0,
            "last_raw":round(losses_raw[-1],4) if losses_raw else 0,
            "last_scaled":round(losses_scaled[-1],4) if losses_scaled else 0,
        },
        "accuracy":{
            "mean":round(sum(accs)/len(accs),4) if accs else None,
            "last":round(accs[-1],4) if accs else None,
        },
        "t_sampled":{
            "mean":round(sum(t_samples)/len(t_samples),4) if t_samples else 0,
            "min":round(min(t_samples),4) if t_samples else 0,
            "max":round(max(t_samples),4) if t_samples else 0,
        },
        "vram":vram_final,
        "embed_noise_hook":embed_noise_hook,
        "gpu_temp_c":_gpu_temp(),
    }
    return result

def _make_all_optimizers(model,lr,wd,betas,device):
    import torch
    opts=[]
    # 1) PagedAdamW8bit — default for HildaNext, pages state to CPU on OOM
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim,"PagedAdamW8bit"):
            opts.append(("PagedAdamW8bit",lambda: bnb.optim.PagedAdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas)))
    except:
        pass
    # 2) AdamW8bit — standard 8bit quantized Adam
    try:
        import bitsandbytes as bnb
        opts.append(("AdamW8bit",lambda: bnb.optim.AdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas)))
    except:
        pass
    # 3) Lion8bit — sign-based optimizer, 1/3 memory of AdamW, needs lower LR
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim,"Lion8bit"):
            opts.append(("Lion8bit",lambda: bnb.optim.Lion8bit(model.parameters(),lr=lr*0.3,weight_decay=wd*3.0,betas=(0.95,0.98))))
    except:
        pass
    # 4) PagedLion8bit — paged variant
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim,"PagedLion8bit"):
            opts.append(("PagedLion8bit",lambda: bnb.optim.PagedLion8bit(model.parameters(),lr=lr*0.3,weight_decay=wd*3.0,betas=(0.95,0.98))))
    except:
        pass
    # 5) AdEMAMix8bit — combines Adam + EMA for smoother updates
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim,"PagedAdEMAMix8bit"):
            opts.append(("PagedAdEMAMix8bit",lambda: bnb.optim.PagedAdEMAMix8bit(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95,0.9999))))
    except:
        pass
    # 6) AdamW (fused) — PyTorch built-in, fp32 states (2x VRAM of 8bit)
    _fused=device.type=="cuda"
    opts.append(("AdamW_fused" if _fused else "AdamW",lambda: torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=betas,fused=_fused)))
    # 7) Adafactor — memory efficient, no stored second moments
    try:
        from transformers.optimization import Adafactor
        opts.append(("Adafactor",lambda: Adafactor(model.parameters(),lr=lr,relative_step=False,scale_parameter=False,warmup_init=False,weight_decay=wd)))
    except:
        pass
    # 8) SGD (baseline reference — no adaptive LR, minimum VRAM)
    opts.append(("SGD",lambda: torch.optim.SGD(model.parameters(),lr=lr*10,weight_decay=wd,momentum=0.9)))
    return opts

def main():
    ap=argparse.ArgumentParser(description="Comprehensive WSD/LLaDA benchmark")
    ap.add_argument("--micro-batches",type=int,default=20,help="Micro-batches per optimizer test")
    ap.add_argument("--skip-opt",type=str,default="",help="Comma-separated optimizer names to skip")
    ap.add_argument("--only-opt",type=str,default="",help="Comma-separated: run only these optimizers")
    ap.add_argument("--config",type=str,default=None)
    ap.add_argument("--no-embed-noise",action="store_true",help="Skip embed noise hook test")
    ap.add_argument("--grad-acc",type=int,default=4)
    args=ap.parse_args()
    cfg_path=args.config or os.path.join(os.path.dirname(__file__),"..","runs","configs","llada21_dolma_wsd_only.json")
    cfg_path=os.path.abspath(cfg_path)
    print(f"{'='*70}",flush=True)
    print(f"  COMPREHENSIVE WSD/LLaDA BENCHMARK SUITE",flush=True)
    print(f"{'='*70}",flush=True)
    print(f"config:       {cfg_path}",flush=True)
    print(f"micro_batches: {args.micro_batches}",flush=True)
    print(f"grad_acc:      {args.grad_acc}",flush=True)
    print(f"embed_noise:   {not args.no_embed_noise}",flush=True)
    print(f"",flush=True)
    import torch
    from hildanext.config import load_config
    from hildanext.training import MmapShardedDataset,_collate
    from hildanext.inference import load_model_bundle
    from hildanext.tokenization import ensure_mask_token
    from hildanext.utils import force_math_sdpa,seed_everything
    from hildanext.trace import trace_from_cfg,set_active_trace,reset_active_trace
    from torch.utils.data import DataLoader
    from pathlib import Path
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB",flush=True)
        print(f"PyTorch: {torch.__version__} CUDA: {torch.version.cuda}",flush=True)
    # Print environment info
    print(f"\n--- Environment ---",flush=True)
    print(f"Python: {sys.version.split()[0]}",flush=True)
    print(f"PyTorch: {torch.__version__}",flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}",flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}",flush=True)
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB",flush=True)
        print(f"GPU temp: {_gpu_temp()}C",flush=True)
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}",flush=True)
    except:
        print(f"bitsandbytes: not available",flush=True)
    try:
        import transformers
        print(f"transformers: {transformers.__version__}",flush=True)
    except:
        pass
    seed_everything(42)
    force_math_sdpa()
    cfg=load_config(cfg_path)
    tr=trace_from_cfg(cfg)
    tk=set_active_trace(tr)
    # Dataset
    dolma_root=str(Path(cfg.data.dolma_path).parent)
    ds=MmapShardedDataset(dolma_root)
    loader=DataLoader(dataset=ds,batch_size=1,shuffle=True,num_workers=0,collate_fn=_collate,pin_memory=True)
    print(f"\n--- Dataset ---",flush=True)
    print(f"Dolma root: {dolma_root}",flush=True)
    print(f"Rows: {len(ds)}",flush=True)
    print(f"Seq len: {cfg.data.seq_len}",flush=True)
    # Config summary
    print(f"\n--- WSD Config ---",flush=True)
    print(f"steps_total:    {cfg.stage0.steps_total_stage0}",flush=True)
    print(f"warmup_frac:    {cfg.stage0.warmup_frac}",flush=True)
    print(f"stable_frac:    {cfg.stage0.stable_frac}",flush=True)
    print(f"decay_frac:     {cfg.stage0.decay_frac}",flush=True)
    print(f"lr:             {cfg.stage0.lr_stage0}",flush=True)
    print(f"grad_accum:     {cfg.stage0.grad_accum_steps}",flush=True)
    print(f"micro_batch:    {cfg.stage0.micro_batch_size}",flush=True)
    print(f"seq_len:        {cfg.stage0.seq_len}",flush=True)
    print(f"mask_mode:      {cfg.stage0.doc_attention_mask_mode}",flush=True)
    print(f"mask_ratio_m2t: {cfg.stage0.mask_ratio_m2t}",flush=True)
    print(f"t2t_edit_ratio: {cfg.stage0.t2t_edit_ratio}",flush=True)
    print(f"m2t_weight:     {cfg.stage0.m2t_weight}",flush=True)
    print(f"t2t_weight:     {cfg.stage0.t2t_weight}",flush=True)
    print(f"ladder_blocks:  {cfg.stage0.ladder_blocks}",flush=True)
    print(f"decay_blocks:   {cfg.stage0.decay_blocks}",flush=True)
    exp=cfg.experiment if hasattr(cfg,"experiment") else None
    print(f"time_param:     {getattr(exp,'time_param','?')}",flush=True)
    print(f"loss_weighting: {getattr(exp,'loss_weighting','?')}",flush=True)
    print(f"t_min:          {getattr(exp,'t_min','?')}",flush=True)
    print(f"t_max:          {getattr(exp,'t_max','?')}",flush=True)
    print(f"attention_mode: {getattr(exp,'attention_mode','?')}",flush=True)
    eff_batch=cfg.stage0.micro_batch_size*cfg.stage0.grad_accum_steps
    tokens_per_step=eff_batch*cfg.stage0.seq_len
    total_tokens=tokens_per_step*cfg.stage0.steps_total_stage0
    print(f"\n--- Effective Training Budget ---",flush=True)
    print(f"Effective batch size: {eff_batch} ({cfg.stage0.micro_batch_size}x{cfg.stage0.grad_accum_steps})",flush=True)
    print(f"Tokens per opt step:  {tokens_per_step:,}",flush=True)
    print(f"Total tokens target: {total_tokens:,} ({total_tokens/1e6:.1f}M)",flush=True)
    # Model
    print(f"\n--- Loading Model ---",flush=True)
    _reset_vram()
    bundle=load_model_bundle(cfg,for_training=True,trace=tr)
    ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
    model=bundle.model
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {bundle.model_name_or_path}",flush=True)
    print(f"Params: {total_params:,} ({total_params/1e6:.1f}M) trainable={trainable_params:,}",flush=True)
    print(f"dtype: {bundle.actual_dtype}, dummy: {bundle.is_dummy}",flush=True)
    print(f"mask_id: {bundle.mask_id}",flush=True)
    vram_model=_vram_stats()
    print(f"VRAM after model load: {vram_model['alloc_mb']:.0f}MB alloc / {vram_model['peak_mb']:.0f}MB peak",flush=True)
    device=bundle.device
    use_amp=device.type=="cuda"
    # Build optimizer list
    lr=float(cfg.stage0.lr_stage0)
    wd=float(cfg.train.weight_decay)
    betas=(0.9,0.95)
    skip=set(x.strip().lower() for x in args.skip_opt.split(",") if x.strip())
    only=set(x.strip().lower() for x in args.only_opt.split(",") if x.strip())
    all_opts=_make_all_optimizers(model,lr,wd,betas,device)
    print(f"\n--- Available Optimizers ---",flush=True)
    for name,_ in all_opts:
        status="SKIP" if name.lower() in skip else ("RUN" if not only or name.lower() in only else "SKIP")
        print(f"  {name}: {status}",flush=True)
    # Run benchmarks
    results=[]
    for opt_name,opt_factory in all_opts:
        if opt_name.lower() in skip:
            continue
        if only and opt_name.lower() not in only:
            continue
        print(f"\n{'='*60}",flush=True)
        print(f"  BENCHMARK: {opt_name}",flush=True)
        print(f"{'='*60}",flush=True)
        # Each optimizer test starts from a clean state
        model.train()
        model.zero_grad(set_to_none=True)
        _reset_vram()
        try:
            opt=opt_factory()
        except Exception as e:
            print(f"  FAILED to create optimizer: {e}",flush=True)
            results.append({"optimizer":opt_name,"error":str(e)})
            continue
        vram_opt=_vram_stats()
        print(f"  VRAM after opt create: alloc={vram_opt['alloc_mb']:.0f}MB",flush=True)
        try:
            r=run_optimizer_benchmark(model,loader,bundle,cfg,opt,opt_name,args.micro_batches,device,use_amp,embed_noise_hook=False)
            results.append(r)
            print(f"\n  --- Result: {opt_name} ---",flush=True)
            print(f"  avg_fwd={r['avg_fwd_s']:.3f}s avg_bwd={r['avg_bwd_s']:.3f}s avg_opt={r['avg_opt_step_s']:.4f}s",flush=True)
            print(f"  avg_micro_batch={r['avg_micro_batch_s']:.3f}s tok/s={r['tokens_per_sec']:.1f}",flush=True)
            print(f"  VRAM peak={r['vram']['peak_mb']:.0f}MB",flush=True)
            print(f"  grad_norm: mean={r['grad_norms']['mean']:.2f} min={r['grad_norms']['min']:.2f} max={r['grad_norms']['max']:.2f}",flush=True)
            print(f"  loss: mean={r['losses']['mean_raw']:.4f} m2t={r['losses']['mean_m2t']:.4f} t2t={r['losses']['mean_t2t']:.4f}",flush=True)
            print(f"  opt_steps_done={r['opt_steps_completed']} skipped_explosion={r['skipped_grad_explosion']}",flush=True)
            print(f"  accuracy: {r['accuracy']['mean']}",flush=True)
        except Exception as e:
            print(f"  BENCHMARK FAILED: {e}",flush=True)
            import traceback
            traceback.print_exc()
            results.append({"optimizer":opt_name,"error":str(e)})
        # Cleanup optimizer state
        del opt
        _reset_vram()
    # Test with embed noise hook (using best available optimizer)
    if not args.no_embed_noise and results:
        best_opt_name=None
        for r in results:
            if "error" not in r and r.get("opt_steps_completed",0)>0:
                best_opt_name=r["optimizer"]
                break
        if best_opt_name is None and results:
            best_opt_name=results[0].get("optimizer","AdamW_fused")
        if best_opt_name:
            print(f"\n{'='*60}",flush=True)
            print(f"  EMBED NOISE HOOK TEST (using {best_opt_name})",flush=True)
            print(f"{'='*60}",flush=True)
            model.train()
            model.zero_grad(set_to_none=True)
            _reset_vram()
            opt_factory=None
            for n,f in all_opts:
                if n==best_opt_name:
                    opt_factory=f
                    break
            if opt_factory:
                try:
                    opt=opt_factory()
                    r=run_optimizer_benchmark(model,loader,bundle,cfg,opt,f"{best_opt_name}+embed_noise",args.micro_batches,device,use_amp,embed_noise_hook=True)
                    results.append(r)
                    print(f"\n  --- Result: {best_opt_name}+embed_noise ---",flush=True)
                    print(f"  avg_micro_batch={r['avg_micro_batch_s']:.3f}s tok/s={r['tokens_per_sec']:.1f}",flush=True)
                    print(f"  VRAM peak={r['vram']['peak_mb']:.0f}MB",flush=True)
                    print(f"  grad_norm: mean={r['grad_norms']['mean']:.2f} min={r['grad_norms']['min']:.2f} max={r['grad_norms']['max']:.2f}",flush=True)
                    print(f"  opt_steps_done={r['opt_steps_completed']} skipped_explosion={r['skipped_grad_explosion']}",flush=True)
                    print(f"  >>> EMBED NOISE REDUCES GRAD EXPLOSION? {r['skipped_grad_explosion']} vs {results[0].get('skipped_grad_explosion','?')} without",flush=True)
                    del opt
                except Exception as e:
                    print(f"  EMBED NOISE TEST FAILED: {e}",flush=True)
                    results.append({"optimizer":f"{best_opt_name}+embed_noise","error":str(e)})
    # Summary
    print(f"\n{'='*70}",flush=True)
    print(f"  SUMMARY TABLE",flush=True)
    print(f"{'='*70}",flush=True)
    print(f"{'Optimizer':<25} {'tok/s':>6} {'VRAM(MB)':>9} {'GradNorm':>9} {'OptSteps':>9} {'Exploded':>9} {'Loss':>8}",flush=True)
    print(f"{'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8}",flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['optimizer']:<25} {'ERROR':>6} {r['error'][:40]}",flush=True)
            continue
        print(f"{r['optimizer']:<25} {r['tokens_per_sec']:>6.1f} {r['vram']['peak_mb']:>9.0f} {r['grad_norms']['mean']:>9.2f} {r['opt_steps_completed']:>9} {r['skipped_grad_explosion']:>9} {r['losses']['mean_raw']:>8.4f}",flush=True)
    # Time estimate for full run
    print(f"\n--- Time Estimates for Full WSD Run ({cfg.stage0.steps_total_stage0} steps) ---",flush=True)
    for r in results:
        if "error" in r:
            continue
        sec_per_step=r["avg_micro_batch_s"]*args.grad_acc+r.get("avg_opt_step_s",0)
        total_sec=sec_per_step*cfg.stage0.steps_total_stage0
        hours=total_sec/3600
        print(f"  {r['optimizer']:<25}: {hours:.1f}h ({sec_per_step:.2f}s/step x {cfg.stage0.steps_total_stage0} steps)",flush=True)
    # Save results
    out_path=os.path.join(os.path.dirname(__file__),"bench_comprehensive.json")
    with open(out_path,"w") as f:
        json.dump({"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"config":cfg_path,"micro_batches":args.micro_batches,"results":results},f,indent=2)
    print(f"\n[BENCH] Results saved to {out_path}",flush=True)
    reset_active_trace(tk)

if __name__=="__main__":
    main()

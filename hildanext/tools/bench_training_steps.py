# Benchmark raw forward+backward throughput: N micro-batches on real data.
# Measures: wall time, sec/micro-batch, peak VRAM.
# Bypasses optimizer step entirely to measure pure compute.
import sys,os,time,argparse,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","backend","src"))
os.environ.setdefault("CONDA_DEFAULT_ENV","mdm")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--micro-batches",type=int,default=20)
    ap.add_argument("--label",type=str,default="baseline")
    ap.add_argument("--config",type=str,default=None)
    args=ap.parse_args()
    N=args.micro_batches
    cfg_path=args.config or os.path.join(os.path.dirname(__file__),"..","runs","configs","llada21_dolma_wsd_only.json")
    cfg_path=os.path.abspath(cfg_path)
    print(f"[BENCH] label={args.label} micro_batches={N} config={cfg_path}",flush=True)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"[BENCH] GPU={torch.cuda.get_device_name()} VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB",flush=True)
    from hildanext.config import load_config
    from hildanext.training import MmapShardedDataset,_collate
    from hildanext.inference import load_model_bundle
    from hildanext.diffusion import compute_m2t_t2t_losses,wsd_block
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
    # Dataset
    dolma_root=str(Path(cfg.data.dolma_path).parent)
    ds=MmapShardedDataset(dolma_root)
    loader=DataLoader(dataset=ds,batch_size=1,shuffle=True,num_workers=0,collate_fn=_collate,pin_memory=True)
    print(f"[BENCH] dataset rows={len(ds)}",flush=True)
    # Model
    bundle=load_model_bundle(cfg,for_training=True,trace=tr)
    ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
    model=bundle.model
    model.train()
    device=bundle.device
    mask_mode=cfg.llada2.mask_mode
    block_size=cfg.llada2.composite_block_size
    vocab_cap=max(8,bundle.vocab_size)
    use_amp=device.type=="cuda"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"[BENCH] model loaded, starting {N} micro-batches (fwd+bwd each)...",flush=True)
    # Warmup: 2 micro-batches (not timed)
    it=iter(loader)
    for w in range(2):
        batch=next(it)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=0.001,t_max=1.0)
        loss=out["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    # Timed run
    fwd_times=[]
    bwd_times=[]
    total_tokens=0
    t_all_start=time.perf_counter()
    for i in range(N):
        batch=next(it)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
        total_tokens+=int(batch["attention_mask"].sum().item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd0=time.perf_counter()
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=use_amp):
            out=compute_m2t_t2t_losses(model=model,input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],doc_ids=batch["doc_ids"],response_mask=batch["response_mask"],mask_id=bundle.mask_id,vocab_size=vocab_cap,cfg=cfg.train,focus_response=False,mask_mode=mask_mode,composite_block_size=block_size,trace=None,cfg_obj=cfg,bidirectional=False,time_param="continuous_time",loss_weighting="inv_t",t_min=0.001,t_max=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd1=time.perf_counter()
        loss=out["loss"]/8.0
        t_bwd0=time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_bwd1=time.perf_counter()
        fwd_times.append(t_fwd1-t_fwd0)
        bwd_times.append(t_bwd1-t_bwd0)
        model.zero_grad(set_to_none=True)
        if (i+1)%5==0:
            print(f"  [{i+1}/{N}] fwd={fwd_times[-1]:.3f}s bwd={bwd_times[-1]:.3f}s total={fwd_times[-1]+bwd_times[-1]:.3f}s",flush=True)
    t_all_end=time.perf_counter()
    wall=t_all_end-t_all_start
    peak_vram=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0
    # Stats
    avg_fwd=sum(fwd_times)/len(fwd_times)
    avg_bwd=sum(bwd_times)/len(bwd_times)
    avg_total=avg_fwd+avg_bwd
    tps=total_tokens/wall if wall>0 else 0
    print(f"\n{'='*60}",flush=True)
    print(f"[BENCH RESULT] label={args.label}",flush=True)
    print(f"  micro_batches={N}",flush=True)
    print(f"  wall_time={wall:.2f}s",flush=True)
    print(f"  avg_fwd={avg_fwd:.3f}s",flush=True)
    print(f"  avg_bwd={avg_bwd:.3f}s",flush=True)
    print(f"  avg_micro_batch={avg_total:.3f}s",flush=True)
    print(f"  tokens_per_sec={tps:.1f}",flush=True)
    print(f"  peak_vram_mb={peak_vram:.0f}",flush=True)
    print(f"{'='*60}",flush=True)
    result={"label":args.label,"micro_batches":N,"wall_time_s":round(wall,2),"avg_fwd_s":round(avg_fwd,3),"avg_bwd_s":round(avg_bwd,3),"avg_micro_batch_s":round(avg_total,3),"tokens_per_sec":round(tps,1),"peak_vram_mb":round(peak_vram,0),"total_tokens":total_tokens}
    out_path=os.path.join(os.path.dirname(__file__),f"bench_{args.label}.json")
    with open(out_path,"w") as f:
        json.dump(result,f,indent=2)
    print(f"[BENCH] saved to {out_path}",flush=True)
    reset_active_trace(tk)

if __name__=="__main__":
    main()

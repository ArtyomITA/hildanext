# Training loops for WSD conversion and SFT (M2T+T2T).
# Main entrypoints: run_wsd_conversion,run_sft_training.
# Includes resume,watchdog,periodic eval and fallback tracing.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Optional,Tuple
import json
import time
import math
import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from .config import AppConfig
from .io_utils import read_jsonl,append_jsonl,write_json,ensure_dir
from .diffusion import compute_m2t_t2t_losses,wsd_block,apply_remask
from .inference import load_model_bundle
from .tokenization import ensure_mask_token
from .formulas import llada21_apply
from .utils import mem_stats,tokens_per_second,seed_everything
from .trace import use_trace,exception_with_stack

class TokenizedDataset(Dataset):
    def __init__(self,path:str,max_rows:int|None=None):
        self.rows=read_jsonl(path,max_rows=max_rows)
    def __len__(self)->int:
        return len(self.rows)
    def __getitem__(self,i:int)->Dict[str,Any]:
        return self.rows[i]

def _collate(batch:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
    ids=torch.tensor([x["input_ids"] for x in batch],dtype=torch.long)
    docs=torch.tensor([x["doc_ids"] for x in batch],dtype=torch.long)
    attn=torch.tensor([x["attention_mask"] for x in batch],dtype=torch.long)
    resp=torch.tensor([x.get("response_mask",[0]*len(x["input_ids"])) for x in batch],dtype=torch.long)
    return {"input_ids":ids,"doc_ids":docs,"attention_mask":attn,"response_mask":resp}

def _save_checkpoint(model,tokenizer,out_dir:Path,tag:str)->str:
    ckpt=out_dir/tag
    ckpt.mkdir(parents=True,exist_ok=True)
    if hasattr(model,"save_pretrained"):
        try:
            model.save_pretrained(str(ckpt))
            if hasattr(tokenizer,"save_pretrained"):
                tokenizer.save_pretrained(str(ckpt))
            return str(ckpt)
        except Exception:
            pass
    torch.save({"state_dict":model.state_dict()},ckpt/"model.pt")
    return str(ckpt)

def _prune_checkpoints(ckpt_dir:Path,keep_last:int)->None:
    keep=max(1,int(keep_last))
    items=[]
    for p in ckpt_dir.glob("step_*"):
        if p.is_dir():
            items.append((_parse_step_from_tag(p.name),p))
    items=[x for x in items if x[0]>=0]
    items.sort(key=lambda x:x[0])
    for _,p in items[:-keep]:
        for c in p.rglob("*"):
            if c.is_file():
                c.unlink(missing_ok=True)
        for c in sorted(p.rglob("*"),reverse=True):
            if c.is_dir():
                c.rmdir()
        if p.exists():
            p.rmdir()

def _parse_step_from_tag(name:str)->int:
    if name.startswith("step_"):
        try:
            return int(name.split("_",1)[1])
        except Exception:
            return -1
    return -1

def _latest_checkpoint(ckpt_dir:Path)->Tuple[int,Optional[Path]]:
    best=-1
    best_path=None
    for p in ckpt_dir.glob("step_*"):
        if not p.is_dir():
            continue
        s=_parse_step_from_tag(p.name)
        if s>best:
            best=s
            best_path=p
    return best,best_path

def _load_checkpoint_model(cfg:AppConfig,model,ckpt_path:Path,device:torch.device,trace=None):
    tr=use_trace(cfg,trace)
    mp=ckpt_path/"model.pt"
    if mp.exists():
        try:
            obj=torch.load(mp,map_location=device)
            sd=obj.get("state_dict",obj)
            model.load_state_dict(sd,strict=False)
            return model
        except Exception as e:
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="training",
                    func="_load_checkpoint_model",
                    action="resume_from_base_model",
                    reason="checkpoint_state_dict_incompatible",
                    exception_str=exception_with_stack(e),
                    extra_dict={"ckpt_path":str(ckpt_path)}
                )
            return model
    try:
        from transformers import AutoModelForCausalLM
        m=AutoModelForCausalLM.from_pretrained(str(ckpt_path),trust_remote_code=cfg.model.trust_remote_code)
        return m.to(device)
    except Exception as e:
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="training",
                func="_load_checkpoint_model",
                action="resume_from_base_model",
                reason="checkpoint_load_failed",
                exception_str=exception_with_stack(e),
                extra_dict={"ckpt_path":str(ckpt_path)}
            )
        return model

def _make_optimizer(model,cfg:AppConfig,device:torch.device,trace=None):
    tr=use_trace(cfg,trace)
    name=(cfg.train.optimizer or "auto").lower()
    lr=float(cfg.train.lr)
    wd=float(cfg.train.weight_decay)
    if name in {"auto","adamw8bit"} and device.type=="cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(),lr=lr,weight_decay=wd),"AdamW8bit"
        except Exception as e:
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="training",
                    func="_make_optimizer",
                    action="optimizer_fallback",
                    reason="adamw8bit_unavailable",
                    exception_str=exception_with_stack(e),
                    extra_dict={"requested":name}
                )
    if name in {"auto","adafactor"}:
        try:
            from transformers.optimization import Adafactor
            return Adafactor(model.parameters(),lr=lr,relative_step=False,scale_parameter=False,warmup_init=False,weight_decay=wd),"Adafactor"
        except Exception as e:
            if tr is not None and name=="adafactor":
                tr.record_fallback(
                    event="fallback",
                    module="training",
                    func="_make_optimizer",
                    action="optimizer_fallback",
                    reason="adafactor_unavailable",
                    exception_str=exception_with_stack(e),
                    extra_dict={"requested":name}
                )
    if name not in {"auto","adamw"} and tr is not None:
        tr.record_fallback(
            event="fallback",
            module="training",
            func="_make_optimizer",
            action="optimizer_fallback",
            reason="optimizer_unknown",
            extra_dict={"requested":name,"used":"AdamW"}
        )
    return AdamW(model.parameters(),lr=lr,weight_decay=wd),"AdamW"

def _decode_safe(tokenizer,ids:torch.Tensor,mask_id:int,dummy:bool=False)->str:
    txt=""
    try:
        txt=tokenizer.decode(ids,skip_special_tokens=True)
    except Exception:
        txt=""
    if not txt.strip():
        raw=[int(x) for x in ids.detach().cpu().tolist() if int(x)!=int(mask_id)]
        txt=" ".join(f"tok{x}" for x in raw[:64]) if raw else ""
    txt=txt.strip() if txt else ""
    if dummy and txt and not txt.startswith("[DUMMY] "):
        txt=f"[DUMMY] {txt}"
    return txt

def _periodic_eval(model,tokenizer,device:torch.device,mask_id:int,cfg:AppConfig,seed:int)->Dict[str,Any]:
    prompts=list(cfg.recipe.eval_prompts or ["Write one sentence about rain.","Q: 5+7? A:","Complete safely: The quick brown fox"])
    rows=[]
    model.eval()
    max_new=min(24,int(cfg.inference.max_new_tokens))
    for i,p in enumerate(prompts[:3]):
        seed_everything(seed+i)
        enc=tokenizer([p],return_tensors="pt")
        inp=enc["input_ids"].to(device)
        plen=int(inp.shape[1])
        seq=inp
        with torch.no_grad():
            for _ in range(max_new):
                logits=model(input_ids=seq).logits[:,-1,:]
                nxt=torch.argmax(logits,dim=-1,keepdim=True)
                seq=torch.cat([seq,nxt],dim=1)
        ar_text=_decode_safe(tokenizer,seq[0,plen:],mask_id)
        seed_everything(seed+i)
        seq2=torch.full((1,plen+max_new),int(mask_id),dtype=torch.long,device=device)
        seq2[:,:plen]=inp
        logs=[]
        with torch.no_grad():
            for st in range(max(1,int(cfg.inference.max_steps))):
                gen_before=seq2[:,plen:]
                pred=torch.zeros_like(gen_before)
                conf=torch.zeros_like(gen_before,dtype=torch.float32)
                work=seq2.clone()
                for t in range(max_new):
                    g=plen+t
                    out=model(input_ids=work[:,:max(1,g)])
                    probs=torch.softmax(out.logits[:,-1,:],dim=-1)
                    c,pid=torch.max(probs,dim=-1)
                    pred[:,t]=pid
                    conf[:,t]=c
                    if int(work[0,g].item())==int(mask_id):
                        work[:,g]=pid
                upd,sets=llada21_apply(gen_before,pred,conf,mask_id,float(cfg.inference.s_mode_tau_mask),float(cfg.inference.s_mode_tau_edit))
                if st+1<int(cfg.inference.max_steps):
                    upd=apply_remask(upd,conf,mask_id,cfg.inference.remask)
                seq2[:,plen:]=upd
                logs.append({"step":st+1,"mask_ratio":float(upd.eq(mask_id).float().mean().item()),"gamma":int(sets.gamma_count),"delta":int(sets.delta_count),"avg_conf_masked":float(conf[gen_before.eq(mask_id)].mean().item()) if gen_before.eq(mask_id).any() else None,"avg_conf_tokens":float(conf[~gen_before.eq(mask_id)].mean().item()) if (~gen_before.eq(mask_id)).any() else None})
                if float(upd.eq(mask_id).float().mean().item())==0.0:
                    break
        dllm_text=_decode_safe(tokenizer,seq2[0,plen:],mask_id)
        rows.append({"prompt":p,"ar":ar_text,"dllm":dllm_text,"decode_logs":logs})
    return {"rows":rows}

def _run(cfg:AppConfig,split_name:str,kind:str,steps:int,focus_response:bool,trace=None,resume:bool=False,ckpt_every:int|None=None,eval_every:int|None=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    tok_path=str(Path(cfg.paths.tokenized_dir)/f"{split_name}.jsonl")
    ds=TokenizedDataset(tok_path)
    if len(ds)==0:
        raise RuntimeError(f"tokenized split missing or empty: {tok_path}")
    data_workers=max(0,int(getattr(cfg.train,"data_num_workers",0) or 0))
    prefetch=max(1,int(getattr(cfg.train,"data_prefetch_factor",2) or 2))
    persistent=bool(getattr(cfg.train,"data_persistent_workers",True))
    pin_memory=bool(getattr(cfg.train,"data_pin_memory",True) and torch.cuda.is_available())
    dl_kwargs={"dataset":ds,"batch_size":max(1,cfg.train.batch_size),"shuffle":True,"num_workers":data_workers,"collate_fn":_collate,"pin_memory":pin_memory}
    if data_workers>0:
        dl_kwargs["prefetch_factor"]=prefetch
        dl_kwargs["persistent_workers"]=persistent
    loader=DataLoader(**dl_kwargs)
    bundle=load_model_bundle(cfg,for_training=True,trace=tr)
    ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
    model=bundle.model
    model.train()
    opt,opt_name=_make_optimizer(model,cfg,bundle.device,trace=tr)
    use_cache=None
    if hasattr(model,"config") and hasattr(model.config,"use_cache"):
        try:
            use_cache=bool(model.config.use_cache)
        except Exception:
            use_cache=None
    grad_acc=max(1,cfg.train.accum_steps)
    max_steps=max(1,int(steps))
    log_path=Path(cfg.paths.logs_dir)/f"{kind}.jsonl"
    eval_path=Path(cfg.paths.logs_dir)/f"{kind}.eval.jsonl"
    ckpt_dir=ensure_dir(cfg.paths.checkpoints_dir)/kind
    ckpt_every=max(1,int(ckpt_every or cfg.train.ckpt_every))
    eval_every=max(1,int(eval_every or cfg.train.eval_every))
    log_every=max(1,int(getattr(cfg.train,"log_every_steps",10) or 10))
    keep_last=max(1,int(getattr(cfg.train,"keep_last_checkpoints",3) or 3))
    cooldown_every=max(0,int(getattr(cfg.train,"cooldown_every_steps",0) or 0))
    cooldown_seconds=max(0,int(getattr(cfg.train,"cooldown_seconds",0) or 0))
    t0=time.time()
    t_last=t0
    sec_roll=None
    token_seen=0
    last_loss=0.0
    opt_steps=0
    resumed_from=""
    if resume:
        last_step,last_ckpt=_latest_checkpoint(Path(ckpt_dir))
        if last_ckpt is not None and last_step>0:
            model=_load_checkpoint_model(cfg,model,last_ckpt,bundle.device,trace=tr)
            opt,opt_name=_make_optimizer(model,cfg,bundle.device,trace=tr)
            opt_steps=last_step
            resumed_from=str(last_ckpt)
            if tr is not None:
                tr.record_notice(module="training",func="_run",action="resume",reason="resumed_from_checkpoint",extra_dict={"kind":kind,"step":last_step,"path":resumed_from})
    oom_happened=False
    nan_happened=False
    epoch=0
    while opt_steps<max_steps:
        epoch+=1
        step_base=(epoch-1)*max(1,len(loader))
        for step,batch in enumerate(loader,start=step_base+1):
            if opt_steps>=max_steps:
                break
            phase=wsd_block(opt_steps,cfg.wsd,seq_len=cfg.data.seq_len) if kind=="cpt" else wsd_block(cfg.wsd.warmup_steps+cfg.wsd.stable_steps,cfg.wsd,seq_len=cfg.data.seq_len)
            batch={k:v.to(bundle.device,non_blocking=pin_memory) for k,v in batch.items()}
            vocab_cap=max(8,bundle.vocab_size)
            batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
            token_seen+=int(batch["attention_mask"].sum().item())
            try:
                run_mask_mode=cfg.llada2.mask_mode if kind=="cpt" else cfg.data.doc_mask_mode
                out=compute_m2t_t2t_losses(
                    model=model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    doc_ids=batch["doc_ids"],
                    response_mask=batch["response_mask"],
                    mask_id=bundle.mask_id,
                    vocab_size=vocab_cap,
                    cfg=cfg.train,
                    focus_response=focus_response,
                    mask_mode=run_mask_mode,
                    composite_block_size=cfg.llada2.composite_block_size,
                    trace=tr,
                    cfg_obj=cfg
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_happened=True
                    emergency=_save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"emergency_oom_step_{opt_steps:05d}")
                    if tr is not None:
                        tr.record_fallback(
                            event="fallback",
                            module="training",
                            func="_run",
                            action="emergency_checkpoint",
                            reason="oom_runtime",
                            exception_str=exception_with_stack(e),
                            extra_dict={"kind":kind,"step":opt_steps,"emergency_checkpoint":emergency}
                        )
                raise
            raw_loss=out["loss"]
            if not bool(torch.isfinite(raw_loss).item()):
                nan_happened=True
                emergency=_save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"emergency_nan_step_{opt_steps:05d}")
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="training",
                        func="_run",
                        action="emergency_checkpoint",
                        reason="loss_nan_inf",
                        extra_dict={"kind":kind,"step":opt_steps,"emergency_checkpoint":emergency}
                    )
                raise RuntimeError("loss_nan_inf")
            loss=raw_loss/float(grad_acc)
            loss.backward()
            if step%grad_acc==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps+=1
                last_loss=float(raw_loss.detach().item())
                elapsed=max(1e-6,time.time()-t0)
                t_now=time.time()
                sec_step=max(1e-6,t_now-t_last)
                t_last=t_now
                sec_roll=sec_step if sec_roll is None else (0.9*sec_roll+0.1*sec_step)
                remain=max(0,max_steps-opt_steps)
                eta=float(remain*(sec_roll or sec_step))
                peak_vram=None
                if bundle.device.type=="cuda":
                    try:
                        peak_vram=float(torch.cuda.max_memory_allocated(bundle.device.index or 0))
                    except Exception:
                        peak_vram=None
                tok_est=int(cfg.train.batch_size)*int(cfg.train.accum_steps)*int(cfg.data.seq_len)*int(opt_steps)
                row={
                    "kind":kind,
                    "epoch":epoch,
                    "step":opt_steps,
                    "phase":phase.phase,
                    "block_size":phase.block_size,
                    "loss":last_loss,
                    "loss_m2t":float(out["loss_m2t"].item()),
                    "loss_t2t":float(out["loss_t2t"].item()),
                    "lr":float(cfg.train.lr),
                    "tokens_per_sec":tokens_per_second(token_seen,elapsed),
                    "mem":mem_stats(bundle.device),
                    "stage":"wsd" if kind=="cpt" else kind,
                    "step_current":int(opt_steps),
                    "steps_total":int(max_steps),
                    "tokens_seen_total":tok_est,
                    "sec_per_step_avg":float(sec_roll or sec_step),
                    "eta_stage_sec":eta,
                    "peak_vram_bytes":peak_vram
                }
                append_jsonl(log_path,[row])
                if opt_steps==1 or opt_steps%log_every==0 or opt_steps==max_steps:
                    ts=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
                    print(f"{ts} stage={row['stage']} step={row['step_current']}/{row['steps_total']} phase={row['phase']} block={row['block_size']} loss={row['loss']:.6f} tok_seen={row['tokens_seen_total']} sec_step={row['sec_per_step_avg']:.3f} eta_sec={row['eta_stage_sec']:.1f} peak_vram={row['peak_vram_bytes']}",flush=True)
                if tr is not None:
                    tr.record_metric(name=f"{kind}.loss_total",value=last_loss,step=opt_steps,module="training",func="_run")
                    tr.record_metric(name=f"{kind}.loss_m2t",value=float(out["loss_m2t"].item()),step=opt_steps,module="training",func="_run")
                    tr.record_metric(name=f"{kind}.loss_t2t",value=float(out["loss_t2t"].item()),step=opt_steps,module="training",func="_run")
                if opt_steps==1 or opt_steps==max_steps or opt_steps%ckpt_every==0:
                    _save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"step_{opt_steps:05d}")
                    _prune_checkpoints(Path(ckpt_dir),keep_last)
                if opt_steps%eval_every==0 or opt_steps==max_steps:
                    ev=_periodic_eval(model,bundle.tokenizer,bundle.device,bundle.mask_id,cfg,seed=cfg.runtime.seed+opt_steps)
                    ev["kind"]=kind
                    ev["step"]=opt_steps
                    append_jsonl(eval_path,[ev])
                if cooldown_every>0 and cooldown_seconds>0 and opt_steps<max_steps and opt_steps%cooldown_every==0:
                    ts=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
                    print(f"{ts} stage={row['stage']} cooldown_start step={opt_steps} sleep_sec={cooldown_seconds}",flush=True)
                    if tr is not None:
                        tr.record_notice(module="training",func="_run",action="cooldown_sleep",reason="thermal_guard",extra_dict={"kind":kind,"step":opt_steps,"sleep_sec":cooldown_seconds})
                    time.sleep(float(cooldown_seconds))
                    ts=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
                    print(f"{ts} stage={row['stage']} cooldown_end step={opt_steps}",flush=True)
                if token_seen>=cfg.train.max_tokens:
                    break
        if token_seen>=cfg.train.max_tokens:
            break
        if len(loader)==0:
            break
    elapsed=max(1e-6,time.time()-t0)
    summary={
        "kind":kind,
        "steps":opt_steps,
        "loss_last":last_loss,
        "dummy_model":bundle.is_dummy,
        "reason":bundle.load_reason,
        "model_name_or_path":bundle.model_name_or_path,
        "dtype":bundle.actual_dtype,
        "requested_dtype":bundle.requested_dtype,
        "grad_ckpt":bool(cfg.train.grad_ckpt),
        "use_cache":use_cache,
        "optimizer_name":opt_name,
        "device":str(bundle.device),
        "env_issues":bundle.env_issues,
        "token_seen":token_seen,
        "tokens_per_sec":tokens_per_second(token_seen,elapsed),
        "log_path":str(log_path),
        "eval_log_path":str(eval_path),
        "checkpoints_dir":str(ckpt_dir),
        "resumed_from":resumed_from,
        "watchdog":{"oom":oom_happened,"nan":nan_happened},
        "fallbacks":tr.snapshot_fallbacks(limit=128) if tr is not None else []
    }
    write_json(Path(cfg.paths.logs_dir)/f"{kind}.summary.json",summary)
    return summary

def run_wsd_conversion(cfg:AppConfig,steps:int|None=None,trace=None,resume:bool=False,ckpt_every:int|None=None,eval_every:int|None=None)->Dict[str,Any]:
    n=steps if steps is not None else cfg.train.max_steps
    return _run(cfg,split_name="train",kind="cpt",steps=n,focus_response=False,trace=trace,resume=resume,ckpt_every=ckpt_every,eval_every=eval_every)

def run_sft_training(cfg:AppConfig,steps:int|None=None,trace=None,resume:bool=False,ckpt_every:int|None=None,eval_every:int|None=None)->Dict[str,Any]:
    n=steps if steps is not None else cfg.train.max_steps
    return _run(cfg,split_name="sft_train",kind="sft",steps=n,focus_response=True,trace=trace,resume=resume,ckpt_every=ckpt_every,eval_every=eval_every)

def merge_topk_checkpoints(cfg:AppConfig,checkpoint_dirs:List[str],output_dir:str)->Dict[str,Any]:
    if len(checkpoint_dirs)<2:
        raise ValueError("need at least 2 checkpoints")
    from transformers import AutoModelForCausalLM
    models=[]
    for p in checkpoint_dirs:
        m=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=cfg.model.trust_remote_code)
        models.append(m.state_dict())
    keys=models[0].keys()
    merged={}
    for k in keys:
        acc=None
        for sd in models:
            t=sd[k].float()
            acc=t if acc is None else acc+t
        merged[k]=(acc/len(models)).to(models[0][k].dtype)
    base=AutoModelForCausalLM.from_pretrained(checkpoint_dirs[0],trust_remote_code=cfg.model.trust_remote_code)
    base.load_state_dict(merged,strict=True)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    base.save_pretrained(str(out))
    return {"merged":len(checkpoint_dirs),"output":str(out)}

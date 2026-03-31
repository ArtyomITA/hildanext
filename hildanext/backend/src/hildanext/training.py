# Training loops for WSD conversion and SFT (M2T+T2T).
# Main entrypoints: run_wsd_conversion,run_sft_training.
# Includes resume,watchdog,periodic eval and fallback tracing.
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,List,Optional,Tuple
import json
import logging
import signal
import threading
import time
import math
import gc
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from .config import AppConfig
from .io_utils import read_jsonl,append_jsonl,write_json,ensure_dir
from .diffusion import compute_m2t_t2t_losses,wsd_block,apply_remask,_install_embed_noise_hook,_remove_embed_noise_hook,set_embed_noise_std,reset_sdpa_probe
from .inference import load_model_bundle
from .tokenization import ensure_mask_token
from .formulas import llada21_apply
from .utils import mem_stats,tokens_per_second,seed_everything
from .trace import use_trace,exception_with_stack

# --- Signal-based checkpoint save (SIGTERM/SIGINT) ---
_signal_save_state:Dict[str,Any]={"requested":False,"model":None,"tokenizer":None,"ckpt_dir":None,"step":0,"optimizer":None}

def _signal_handler(signum,frame):
    """On SIGTERM/SIGINT, save an emergency checkpoint before exiting."""
    _signal_save_state["requested"]=True
    s=_signal_save_state
    if s["model"] is not None and s["ckpt_dir"] is not None:
        try:
            tag=f"signal_{signal.Signals(signum).name}_step_{s['step']:05d}"
            _save_checkpoint(s["model"],s["tokenizer"],Path(s["ckpt_dir"]),tag,optimizer=s["optimizer"])
            print(f"[signal] emergency checkpoint saved: {tag}",flush=True)
        except Exception as e:
            print(f"[signal] emergency checkpoint FAILED: {e}",flush=True)

def _install_signal_handlers():
    try:
        signal.signal(signal.SIGTERM,_signal_handler)
        signal.signal(signal.SIGINT,_signal_handler)
    except (OSError,ValueError):
        pass  # not main thread or platform limitation

# --- Watchdog: detects stalled training loops ---
class _TrainingWatchdog:
    """Background thread that fires if no heartbeat for timeout_sec seconds."""
    def __init__(self,timeout_sec:float=600,callback=None):
        self._timeout=timeout_sec
        self._callback=callback
        self._last_beat=time.time()
        self._stop=threading.Event()
        self._thread=threading.Thread(target=self._run,daemon=True)
        self._thread.start()
    def heartbeat(self):
        self._last_beat=time.time()
    def stop(self):
        self._stop.set()
    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(30)
            if self._stop.is_set():
                break
            elapsed=time.time()-self._last_beat
            if elapsed>self._timeout:
                print(f"[watchdog] STALL DETECTED: no heartbeat for {elapsed:.0f}s (timeout={self._timeout:.0f}s)",flush=True)
                if self._callback:
                    try:
                        self._callback()
                    except Exception:
                        pass
                self._last_beat=time.time()  # reset to avoid spamming

# --- GPU thermal query (pynvml, 65x faster than nvidia-smi subprocess) ---
_nvml_inited=False
def _gpu_temp_celsius()->Optional[int]:
    """Read GPU temperature via pynvml. Falls back to nvidia-smi if pynvml unavailable."""
    global _nvml_inited
    try:
        import pynvml
        if not _nvml_inited:
            pynvml.nvmlInit()
            _nvml_inited=True
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        pass
    # Fallback: nvidia-smi subprocess (slow, ~48ms per call)
    try:
        import subprocess
        r=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
        if r.returncode==0:
            return int(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None

def _compute_lr(step:int,warmup_steps:int,total_steps:int,base_lr:float,min_ratio:float=0.1)->float:
    """Linear warmup then cosine decay to base_lr*min_ratio."""
    if step<warmup_steps:
        return base_lr*float(step)/max(1,warmup_steps)
    progress=float(step-warmup_steps)/max(1,total_steps-warmup_steps)
    return base_lr*(min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*progress)))

def _t_bucket_key(t:float)->str:
    if t<0.1: return "0.0-0.1"
    if t<0.3: return "0.1-0.3"
    if t<0.6: return "0.3-0.6"
    return "0.6-1.0"

_T_BUCKET_NAMES=["0.0-0.1","0.1-0.3","0.3-0.6","0.6-1.0"]

class TokenizedDataset(Dataset):
    def __init__(self,path:str,max_rows:int|None=None):
        self.rows=read_jsonl(path,max_rows=max_rows)
    def __len__(self)->int:
        return len(self.rows)
    def __getitem__(self,i:int)->Dict[str,Any]:
        return self.rows[i]

class MmapShardedDataset(Dataset):
    """Memory-mapped dataset reading directly from Dolma numpy shards.

    Uses ~2 MB RAM (metadata only) instead of ~37.5 GB for the JSONL path.
    Shards are opened lazily with np.load(mmap_mode='r') on first access.
    """
    def __init__(self,shard_root:str,max_rows:int|None=None):
        root=Path(shard_root)
        meta_path=root/"meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {root}")
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        self.seq_len=int(meta["seq_len"])
        self.total_rows=int(meta["rows_total"])
        self.n_shards=int(meta["shards"])
        tok_dir=Path(meta.get("tokens_dir",str(root/"tokens")))
        doc_dir=Path(meta.get("doc_index_dir",str(root/"doc_index")))
        # Build shard file lists and cumulative row counts from meta
        # Standard shards have 1000 rows, last shard has remainder
        self._tok_files:List[Path]=[]
        self._doc_files:List[Path]=[]
        self._cum_rows:List[int]=[0]
        rows_so_far=0
        rows_per_shard=1000  # standard Dolma shard size
        for i in range(self.n_shards):
            tf=tok_dir/f"tokens_{i:05d}.npy"
            df=doc_dir/f"doc_index_{i:05d}.npy"
            if not tf.exists():
                break
            self._tok_files.append(tf)
            self._doc_files.append(df)
            if i==self.n_shards-1:
                # Last shard: remainder rows
                shard_rows=self.total_rows-rows_so_far
            else:
                shard_rows=rows_per_shard
            rows_so_far+=shard_rows
            self._cum_rows.append(rows_so_far)
        self.total_rows=min(self.total_rows,rows_so_far)
        if max_rows is not None and max_rows>0:
            self.total_rows=min(self.total_rows,max_rows)
        # Lazy mmap cache (opened on first __getitem__)
        self._tok_mmaps:Dict[int,np.ndarray]={}
        self._doc_mmaps:Dict[int,np.ndarray]={}

    def __len__(self)->int:
        return self.total_rows

    def _shard_for_row(self,idx:int)->Tuple[int,int]:
        """Binary search for (shard_index, row_within_shard)."""
        lo,hi=0,len(self._cum_rows)-2
        while lo<hi:
            mid=(lo+hi)//2
            if self._cum_rows[mid+1]<=idx:
                lo=mid+1
            else:
                hi=mid
        return lo,idx-self._cum_rows[lo]

    def _get_mmap(self,shard_idx:int)->Tuple[np.ndarray,np.ndarray]:
        if shard_idx not in self._tok_mmaps:
            self._tok_mmaps[shard_idx]=np.load(self._tok_files[shard_idx],mmap_mode="r")
            if shard_idx<len(self._doc_files) and self._doc_files[shard_idx].exists():
                self._doc_mmaps[shard_idx]=np.load(self._doc_files[shard_idx],mmap_mode="r")
            else:
                # No doc_index: synthesize all-zeros (single document)
                self._doc_mmaps[shard_idx]=np.zeros_like(self._tok_mmaps[shard_idx],dtype=np.int32)
        return self._tok_mmaps[shard_idx],self._doc_mmaps[shard_idx]

    def __getitem__(self,idx:int)->Dict[str,torch.Tensor]:
        if idx<0 or idx>=self.total_rows:
            raise IndexError(f"index {idx} out of range [0,{self.total_rows})")
        shard_idx,row_in_shard=self._shard_for_row(idx)
        tok_arr,doc_arr=self._get_mmap(shard_idx)
        # Copy to contiguous numpy then to tensor (torch.from_numpy may fail on Windows DLL issues)
        input_ids=torch.tensor(tok_arr[row_in_shard].astype(np.int64),dtype=torch.long)
        doc_ids=torch.tensor(doc_arr[row_in_shard].astype(np.int64),dtype=torch.long)
        # Attention mask: 1 where doc_id >= 0 (not padding)
        attention_mask=(doc_ids>=0).long()
        # Response mask: zeros (no response masking for pre-training)
        response_mask=torch.zeros(self.seq_len,dtype=torch.long)
        return {"input_ids":input_ids,"doc_ids":doc_ids,"attention_mask":attention_mask,"response_mask":response_mask}

def _collate(batch:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
    # Support both dict-of-lists (TokenizedDataset) and dict-of-tensors (MmapShardedDataset)
    first=batch[0]
    if isinstance(first.get("input_ids"),torch.Tensor):
        return {k:torch.stack([x[k] for x in batch]) for k in ("input_ids","doc_ids","attention_mask","response_mask")}
    ids=torch.tensor([x["input_ids"] for x in batch],dtype=torch.long)
    docs=torch.tensor([x["doc_ids"] for x in batch],dtype=torch.long)
    attn=torch.tensor([x["attention_mask"] for x in batch],dtype=torch.long)
    resp=torch.tensor([x.get("response_mask",[0]*len(x["input_ids"])) for x in batch],dtype=torch.long)
    return {"input_ids":ids,"doc_ids":docs,"attention_mask":attn,"response_mask":resp}

def _save_checkpoint(model,tokenizer,out_dir:Path,tag:str,optimizer=None,watchdog=None)->str:
    ckpt=out_dir/tag
    ckpt.mkdir(parents=True,exist_ok=True)
    saved_via="unknown"
    if watchdog is not None:
        watchdog.heartbeat()
    if hasattr(model,"save_pretrained"):
        try:
            model.save_pretrained(str(ckpt))
            if watchdog is not None:
                watchdog.heartbeat()
            if hasattr(tokenizer,"save_pretrained"):
                tokenizer.save_pretrained(str(ckpt))
            saved_via="save_pretrained"
        except Exception as e:
            logging.warning("save_pretrained failed for tag=%s: %s — falling back to torch.save",tag,e)
            # Clean partial files from failed save_pretrained before fallback
            for partial in ckpt.glob("*.safetensors"):
                partial.unlink(missing_ok=True)
            torch.save({"state_dict":model.state_dict()},ckpt/"model.pt")
            # Still try to save tokenizer in fallback path
            if hasattr(tokenizer,"save_pretrained"):
                try:
                    tokenizer.save_pretrained(str(ckpt))
                except Exception:
                    pass
            saved_via="torch_save_fallback"
    else:
        torch.save({"state_dict":model.state_dict()},ckpt/"model.pt")
        saved_via="torch_save"
    if watchdog is not None:
        watchdog.heartbeat()
    # Save optimizer state for proper resume (AdamW8bit momentum/statistics)
    if optimizer is not None:
        try:
            torch.save(optimizer.state_dict(),ckpt/"optimizer.pt")
        except Exception as e:
            logging.warning("optimizer state save failed for tag=%s: %s",tag,e)
    if watchdog is not None:
        watchdog.heartbeat()
    print(f"[checkpoint] saved {tag} via={saved_via} path={ckpt}",flush=True)
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
            print(f"[resume] model loaded from {mp}",flush=True)
            return model
        except Exception as e:
            logging.warning("checkpoint state_dict load failed at %s: %s — resuming from base model",ckpt_path,e)
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
        m=AutoModelForCausalLM.from_pretrained(str(ckpt_path),trust_remote_code=cfg.model.trust_remote_code,torch_dtype=model.dtype)
        # Copy weights into existing model on GPU instead of creating a second GPU copy
        model.load_state_dict(m.state_dict(),strict=False)
        del m
        print(f"[resume] model loaded via from_pretrained from {ckpt_path}",flush=True)
        return model
    except Exception as e:
        logging.warning("checkpoint from_pretrained failed at %s: %s — resuming from base model",ckpt_path,e)
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

def _load_optimizer_state(optimizer,ckpt_path:Path,device:torch.device)->bool:
    """Try to restore optimizer state from checkpoint. Returns True if successful."""
    op=ckpt_path/"optimizer.pt"
    if not op.exists():
        print(f"[resume] no optimizer.pt in {ckpt_path} — optimizer starts fresh",flush=True)
        return False
    try:
        opt_state=torch.load(op,map_location=device)
        optimizer.load_state_dict(opt_state)
        del opt_state
        print(f"[resume] optimizer state restored from {op}",flush=True)
        return True
    except Exception as e:
        logging.warning("optimizer state restore failed at %s: %s — optimizer starts fresh",ckpt_path,e)
        return False

def _make_optimizer(model,cfg:AppConfig,device:torch.device,trace=None,kind:str="cpt"):
    tr=use_trace(cfg,trace)
    name=(cfg.train.optimizer or "auto").lower()
    lr=float(cfg.train.lr)
    wd=float(cfg.train.weight_decay)
    # CPT benefits from faster variance adaptation (beta2=0.95 vs default 0.999)
    betas=(0.9,0.95) if kind=="cpt" else (0.9,0.999)
    if name in {"bnb_paged_adamw8bit","paged_adamw8bit"} and device.type=="cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas),"PagedAdamW8bit"
        except Exception as e:
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="training",
                    func="_make_optimizer",
                    action="optimizer_fallback",
                    reason="paged_adamw8bit_unavailable",
                    exception_str=exception_with_stack(e),
                    extra_dict={"requested":name}
                )
    if name in {"auto","adamw8bit","bnb_paged_adamw8bit","paged_adamw8bit"} and device.type=="cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(),lr=lr,weight_decay=wd,betas=betas),"AdamW8bit"
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
    _fused=device.type=="cuda"
    return AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=betas,fused=_fused),"AdamW(fused)" if _fused else "AdamW"

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
    with torch.inference_mode():
        max_new=min(24,int(cfg.inference.max_new_tokens))
        for i,p in enumerate(prompts[:3]):
            seed_everything(seed+i)
            enc=tokenizer([p],return_tensors="pt")
            inp=enc["input_ids"].to(device)
            plen=int(inp.shape[1])
            seq=inp
            for _ in range(max_new):
                logits=model(input_ids=seq).logits[:,-1,:]
                nxt=torch.argmax(logits,dim=-1,keepdim=True)
                seq=torch.cat([seq,nxt],dim=1)
            ar_text=_decode_safe(tokenizer,seq[0,plen:],mask_id)
            seed_everything(seed+i)
            seq2=torch.full((1,plen+max_new),int(mask_id),dtype=torch.long,device=device)
            seq2[:,:plen]=inp
            logs=[]
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
    model.train()
    return {"rows":rows}

def _run(cfg:AppConfig,split_name:str,kind:str,steps:int,focus_response:bool,trace=None,resume:bool=False,ckpt_every:int|None=None,eval_every:int|None=None)->Dict[str,Any]:
    tr=use_trace(cfg,trace)
    _t_init_start=time.time()
    print(f"[_run] INIT_START kind={kind} split={split_name} steps={steps}",flush=True)
    # ---- dataset load ----
    tok_path=str(Path(cfg.paths.tokenized_dir)/f"{split_name}.jsonl")
    _t_ds0=time.time()
    # Try memory-mapped Dolma shards first for CPT (avoids loading 5GB JSONL into 37.5GB RAM)
    _dolma_shard_root=None
    if kind=="cpt" and hasattr(cfg.data,"dolma_path") and cfg.data.dolma_path:
        _candidate=Path(cfg.data.dolma_path).parent  # dolma_path points to /raw, shards are at parent
        if (_candidate/"meta.json").exists():
            _dolma_shard_root=str(_candidate)
    if _dolma_shard_root is not None:
        print(f"[_run] DATASET_MMAP_START shard_root={_dolma_shard_root}",flush=True)
        ds=MmapShardedDataset(_dolma_shard_root)
        _t_ds1=time.time()
        print(f"[_run] DATASET_MMAP_DONE rows={len(ds)} shards={ds.n_shards} elapsed={_t_ds1-_t_ds0:.1f}s ram_approx_mb=2",flush=True)
    else:
        _tok_file_size=Path(tok_path).stat().st_size if Path(tok_path).exists() else 0
        print(f"[_run] DATASET_LOAD_START path={tok_path} file_size_mb={_tok_file_size//1024//1024}",flush=True)
        ds=TokenizedDataset(tok_path)
        _t_ds1=time.time()
        print(f"[_run] DATASET_LOAD_DONE rows={len(ds)} elapsed={_t_ds1-_t_ds0:.1f}s ram_approx_mb={len(ds)*4*int(cfg.data.seq_len)//1024//1024}",flush=True)
    if len(ds)==0:
        raise RuntimeError(f"tokenized split missing or empty: {tok_path}")
    data_workers=max(0,int(getattr(cfg.train,"data_num_workers",0) or 0))
    # Windows: DataLoader multiprocessing (spawn) deadlocks silently when num_workers>0.
    # Force workers=0 on Windows regardless of config to avoid eternal hang after EPOCH_START.
    import platform as _platform
    if _platform.system()=="Windows" and data_workers>0:
        logging.warning("[_run] Windows detected: forcing data_num_workers=0 (was %d) to prevent DataLoader spawn deadlock",data_workers)
        print(f"[_run] WINDOWS_WORKERS_OVERRIDE requested={data_workers} forced=0 reason=spawn_deadlock_prevention",flush=True)
        data_workers=0
    prefetch=max(1,int(getattr(cfg.train,"data_prefetch_factor",2) or 2))
    persistent=bool(getattr(cfg.train,"data_persistent_workers",True)) and data_workers>0
    pin_memory=bool(getattr(cfg.train,"data_pin_memory",True) and torch.cuda.is_available())
    print(f"[_run] DATALOADER_CONFIG workers={data_workers} prefetch={prefetch} persistent={persistent} pin_memory={pin_memory}",flush=True)
    dl_kwargs={"dataset":ds,"batch_size":max(1,cfg.train.batch_size),"shuffle":True,"num_workers":data_workers,"collate_fn":_collate,"pin_memory":pin_memory}
    if data_workers>0:
        dl_kwargs["prefetch_factor"]=prefetch
        dl_kwargs["persistent_workers"]=persistent
    _t_dl0=time.time()
    loader=DataLoader(**dl_kwargs)
    _t_dl1=time.time()
    print(f"[_run] DATALOADER_CREATED batches_approx={len(loader)} elapsed={_t_dl1-_t_dl0:.2f}s",flush=True)
    # ---- model load ----
    _vram_before_model=0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _vram_before_model=float(torch.cuda.memory_allocated())/1024/1024
    print(f"[_run] MODEL_LOAD_START vram_before_mb={_vram_before_model:.1f}",flush=True)
    _t_ml0=time.time()
    bundle=load_model_bundle(cfg,for_training=True,trace=tr)
    _t_ml1=time.time()
    _vram_after_model=float(torch.cuda.memory_allocated())/1024/1024 if torch.cuda.is_available() else 0.0
    print(f"[_run] MODEL_LOAD_DONE elapsed={_t_ml1-_t_ml0:.1f}s vram_after_mb={_vram_after_model:.1f} model_vram_mb={_vram_after_model-_vram_before_model:.1f} dtype={bundle.actual_dtype}",flush=True)
    ensure_mask_token(bundle.tokenizer,cfg.model.mask_token,model=bundle.model)
    # ---- P0: hard-cap VRAM to prevent Windows shared memory spill (freeze fix) ----
    if torch.cuda.is_available() and cfg.train.max_vram_pct<1.0:
        try:
            _cap_device=bundle.device.index if bundle.device.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(cfg.train.max_vram_pct,device=_cap_device)
            _total_vram_mb=torch.cuda.get_device_properties(bundle.device).total_memory/1024/1024
            _cap_mb=_total_vram_mb*cfg.train.max_vram_pct
            print(f"[_run] VRAM_CAP set={cfg.train.max_vram_pct:.0%} total={_total_vram_mb:.0f}MB cap={_cap_mb:.0f}MB",flush=True)
        except Exception as _e:
            print(f"[_run] VRAM_CAP_FAILED error={_e}",flush=True)
    model=bundle.model
    model.train()
    # ---- optimizer ----
    print(f"[_run] OPTIMIZER_CREATE_START",flush=True)
    _t_opt0=time.time()
    opt,opt_name=_make_optimizer(model,cfg,bundle.device,trace=tr,kind=kind)
    _t_opt1=time.time()
    _vram_after_opt=float(torch.cuda.memory_allocated())/1024/1024 if torch.cuda.is_available() else 0.0
    print(f"[_run] OPTIMIZER_CREATED name={opt_name} elapsed={_t_opt1-_t_opt0:.2f}s vram_after_mb={_vram_after_opt:.1f} opt_vram_mb={_vram_after_opt-_vram_after_model:.1f}",flush=True)
    # ---- AMP autocast config (model already loads in fp16, so GradScaler is not used) ----
    _use_amp=bundle.device.type=="cuda"
    print(f"[_run] AMP_CONFIG autocast={_use_amp} dtype=float16",flush=True)
    # ---- VRAM cleanup after init ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    _t_init_end=time.time()
    print(f"[_run] INIT_COMPLETE total_init_sec={_t_init_end-_t_init_start:.1f}s",flush=True)
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
    run_log_path=Path(cfg.paths.logs_dir)/f"{kind}_run.log"
    ckpt_dir=ensure_dir(cfg.paths.checkpoints_dir)/kind
    ckpt_every=max(1,int(ckpt_every or cfg.train.ckpt_every))
    eval_every=max(1,int(eval_every or cfg.train.eval_every))
    log_every=max(1,int(getattr(cfg.train,"log_every_steps",10) or 10))
    keep_last=max(1,int(getattr(cfg.train,"keep_last_checkpoints",3) or 3))
    cooldown_every=max(0,int(getattr(cfg.train,"cooldown_every_steps",0) or 0))
    cooldown_seconds=max(0,int(getattr(cfg.train,"cooldown_seconds",0) or 0))
    grad_clip_val=float(getattr(cfg.train,"grad_clip",1.0) or 1.0)
    lr_base=float(cfg.train.lr)
    lr_warmup=max(1,int(cfg.train.warmup_steps))
    lr_min_ratio=float(getattr(cfg.train,"lr_min_ratio",0.1) or 0.1)
    # Experiment flags
    exp=cfg.experiment if hasattr(cfg,"experiment") else None
    attn_mode=getattr(exp,"attention_mode","bidirectional_only_stable") if exp else "bidirectional_only_stable"
    bidir_verified=getattr(exp,"bidirectional_verified",False) if exp else False
    bidir_disabled_reason=getattr(exp,"bidirectional_disabled_reason","") if exp else ""
    time_param=getattr(exp,"time_param","continuous_time") if exp else "continuous_time"
    loss_weighting=getattr(exp,"loss_weighting","inv_t") if exp else "inv_t"
    shift_mode=getattr(exp,"shift_mode","preserve_left_shift") if exp else "preserve_left_shift"
    ct_t_min=float(getattr(exp,"t_min",0.001)) if exp else 0.001
    ct_t_max=float(getattr(exp,"t_max",1.0)) if exp else 1.0
    # Mask / attn-shape derived constants (logged once per phase, zero overhead at step time)
    _llada2_cfg=getattr(cfg,"llada2",None)
    _mask_mode=getattr(_llada2_cfg,"mask_mode","none") if _llada2_cfg else "none"
    _seq=cfg.data.seq_len
    _bs=cfg.train.batch_size
    _attn4d_shape=f"[{_bs},1,{2*_seq},{2*_seq}]" if _mask_mode=="composite_llada20" else f"[{_bs},1,{_seq},{_seq}]"
    # File logger (dual: console + file)
    run_log_path.parent.mkdir(parents=True,exist_ok=True)
    _run_log_fh=open(run_log_path,"a",encoding="utf-8")
    def _log(msg:str):
        ts=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        line=f"[{ts}] {msg}"
        print(line,flush=True)
        _run_log_fh.write(line+"\n")
        _run_log_fh.flush()
    # T-bucket aggregators for continuous-time diagnostics
    t_bucket_loss={k:{"sum":0.0,"count":0} for k in _T_BUCKET_NAMES}
    t_bucket_acc={k:{"sum":0.0,"count":0} for k in _T_BUCKET_NAMES}
    t_agg={"sum":0.0,"min":1.0,"max":0.0,"count":0}
    nan_inf_count=0
    last_phase=""
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
            opt,opt_name=_make_optimizer(model,cfg,bundle.device,trace=tr,kind=kind)
            _load_optimizer_state(opt,last_ckpt,bundle.device)
            opt_steps=last_step
            resumed_from=str(last_ckpt)
            if tr is not None:
                tr.record_notice(module="training",func="_run",action="resume",reason="resumed_from_checkpoint",extra_dict={"kind":kind,"step":last_step,"path":resumed_from})
            # Free temporary tensors from checkpoint loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    oom_happened=False
    nan_happened=False
    epoch=0
    # --- One-time start-of-run log ---
    _log(f"RUN_START kind={kind} max_steps={max_steps} grad_acc={grad_acc} seq_len={cfg.data.seq_len} batch_size={cfg.train.batch_size}")
    _log(f"  optimizer={opt_name} lr={lr_base} lr_warmup={lr_warmup} lr_min_ratio={lr_min_ratio} grad_clip={grad_clip_val} wd={cfg.train.weight_decay}")
    _log(f"  shift_mode={shift_mode} time_param={time_param} loss_weighting={loss_weighting} attention_mode={attn_mode} bidir_verified={bidir_verified}")
    _log(f"  t_min={ct_t_min} t_max={ct_t_max} mask_ratio_discrete={cfg.train.mask_ratio}")
    _log(f"  model={bundle.model_name_or_path} dtype={bundle.actual_dtype} device={bundle.device}")
    _log(f"  labels_offset=+1 first_position_ignored=True shift_mode={shift_mode}")
    _log(f"  attention_backend=mem_efficient+math_sdpa (flash_sdp disabled, cudnn.benchmark=True)")
    if resumed_from:
        _log(f"  resumed_from={resumed_from} opt_steps={opt_steps}")
    print(f"[_run] TRAINING_LOOP_ENTER max_steps={max_steps} opt_steps={opt_steps} resumed_from={resumed_from}",flush=True)
    # --- Install signal handlers for crash resilience ---
    _install_signal_handlers()
    _signal_save_state.update({"model":model,"tokenizer":bundle.tokenizer,"ckpt_dir":str(ckpt_dir),"step":opt_steps,"optimizer":opt})
    # --- Watchdog thread (10 min timeout) ---
    # On stall: log only — do NOT call _save_checkpoint from watchdog thread
    # to avoid concurrent disk writes that can freeze Windows further.
    watchdog=_TrainingWatchdog(timeout_sec=600,callback=None)
    # --- LLaDA2.0 Sec 7.1: embed noise for mask tokens during WSD warmup ---
    _embed_noise_warmup_steps=cfg.wsd.warmup_steps if kind=="cpt" else 0
    if _embed_noise_warmup_steps>0 and opt_steps<_embed_noise_warmup_steps:
        _install_embed_noise_hook(model,bundle.mask_id,noise_std=0.1)
        _log(f"  embed_noise_hook=installed warmup_steps={_embed_noise_warmup_steps} std=0.1")
    # --- NaN/Inf progressive recovery config ---
    _nan_skip_limit=5  # max consecutive NaN batches before hard crash
    _nan_skip_streak=0
    # --- Thermal protection config ---
    _thermal_limit=85  # celsius — throttle if above
    def _write_summary(reason:str="normal")->Dict[str,Any]:
        _elapsed=max(1e-6,time.time()-t0)
        _log(f"RUN_END kind={kind} steps={opt_steps} loss_last={last_loss:.6f} tokens={token_seen} elapsed={int(_elapsed)}s nan_inf_count={nan_inf_count} reason={reason}")
        try:
            _run_log_fh.close()
        except Exception:
            pass
        _summary={
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
            "tokens_per_sec":tokens_per_second(token_seen,_elapsed),
            "log_path":str(log_path),
            "eval_log_path":str(eval_path),
            "checkpoints_dir":str(ckpt_dir),
            "resumed_from":resumed_from,
            "exit_reason":reason,
            "watchdog":{"oom":oom_happened,"nan":nan_happened},
            "fallbacks":tr.snapshot_fallbacks(limit=128) if tr is not None else []
        }
        write_json(Path(cfg.paths.logs_dir)/f"{kind}.summary.json",_summary)
        return _summary
    _global_micro=0  # global micro-batch counter across epochs
    _resumed_opt_steps=opt_steps  # remember starting opt_steps for log gating
    while opt_steps<max_steps:
        epoch+=1
        step_base=(epoch-1)*max(1,len(loader))
        print(f"[_run] EPOCH_START epoch={epoch} loader_len={len(loader)}",flush=True)
        _t_batch_iter_start=time.time()
        for step,batch in enumerate(loader,start=step_base+1):
            _global_micro+=1
            if step==step_base+1:
                _t_first_batch=time.time()
                print(f"[_run] FIRST_BATCH_RECEIVED epoch={epoch} time_to_first_batch={_t_first_batch-_t_batch_iter_start:.2f}s",flush=True)
            if opt_steps>=max_steps:
                break
            phase=wsd_block(opt_steps,cfg.wsd,seq_len=cfg.data.seq_len) if kind=="cpt" else wsd_block(cfg.wsd.warmup_steps+cfg.wsd.stable_steps,cfg.wsd,seq_len=cfg.data.seq_len)
            # S0-A: determine bidirectional attention per WSD phase
            if attn_mode=="bidirectional_always":
                bidirectional=True
            elif attn_mode=="bidirectional_only_stable":
                bidirectional=(phase.phase=="stable")
            else:
                bidirectional=False
            # Log phase transitions with RUNTIME mask_mode (not static config)
            if phase.phase!=last_phase:
                _rt_mask_mode="simple_blockdiag" if (kind=="cpt" and phase.phase=="stable") else _mask_mode
                _rt_seq=_seq if _rt_mask_mode!="composite_llada20" else 2*_seq
                _rt_attn4d=f"[{_bs},1,{_rt_seq},{_rt_seq}]"
                _log(
                    f"PHASE_CHANGE wsd_phase={phase.phase}"
                    f" bidirectional={bidirectional}"
                    f" bidirectional_verified={bidir_verified}"
                    f" mask_mode={_rt_mask_mode}"
                    f" attn4d_shape={_rt_attn4d}"
                    f" block_size={phase.block_size}"
                    f" step={opt_steps}/{max_steps}"
                )
                last_phase=phase.phase
                reset_sdpa_probe()
                # Grad checkpointing: always ON (8GB VRAM cannot fit without it even at seq=1024)
                if kind=="cpt" and hasattr(model,"gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False,"preserve_rng_state":False})
                    except TypeError:
                        model.gradient_checkpointing_enable()
                    _log(f"  grad_ckpt=ON ({phase.phase} phase)")
                # Remove embed noise hook when warmup ends
                if phase.phase!="warmup" and _embed_noise_warmup_steps>0:
                    _remove_embed_noise_hook()
                    _log(f"  embed_noise_hook=removed (warmup phase ended)")
            batch={k:v.to(bundle.device,non_blocking=pin_memory) for k,v in batch.items()}
            vocab_cap=max(8,bundle.vocab_size)
            batch["input_ids"]=torch.remainder(batch["input_ids"],vocab_cap)
            token_seen+=int(batch["attention_mask"].shape[0])*int(batch["attention_mask"].shape[1])
            if _global_micro<=2:
                _vram_pre_fwd=float(torch.cuda.memory_allocated())/1024/1024 if torch.cuda.is_available() else 0.0
                print(f"[_run] PRE_FORWARD opt_step={opt_steps} micro={_global_micro} vram_mb={_vram_pre_fwd:.1f} batch_shape={batch['input_ids'].shape}",flush=True)
            _t_fwd0=time.time()
            _mtf_turns=max(1,int(cfg.train.multi_turn_t2t))
            try:
                if kind=="cpt":
                    run_mask_mode="simple_blockdiag" if phase.phase=="stable" else cfg.llada2.mask_mode
                else:
                    run_mask_mode=cfg.data.doc_mask_mode
                # --- MTF loop (LLaDA 2.1 S3.1): multi-turn forward data augmentation ---
                _mtf_original_ids=batch["input_ids"]
                _mtf_current_ids=_mtf_original_ids
                _mtf_loss_total=0.0
                _mtf_last_out=None
                for _mtf_turn in range(_mtf_turns):
                    _mtf_target=_mtf_original_ids if _mtf_turn>0 else None
                    with torch.amp.autocast("cuda",dtype=torch.float16,enabled=_use_amp):
                        out=compute_m2t_t2t_losses(
                            model=model,
                            input_ids=_mtf_current_ids,
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
                            cfg_obj=cfg,
                            bidirectional=bidirectional,
                            time_param=time_param,
                            loss_weighting=loss_weighting,
                            t_min=ct_t_min,
                            t_max=ct_t_max,
                            target_ids=_mtf_target
                        )
                    _turn_loss=out["loss"]/float(_mtf_turns)
                    _mtf_loss_total+=float(out["loss"].detach().item())
                    # backward each turn immediately to keep activation memory flat
                    (_turn_loss/float(grad_acc)).backward()
                    # Build next turn's input from model predictions at corrupted positions
                    if _mtf_turn<_mtf_turns-1:
                        with torch.no_grad():
                            _next_ids=_mtf_original_ids.clone()
                            _corrupted=out["corrupted_positions"]
                            _next_ids[_corrupted]=out["model_predictions"][_corrupted]
                            _mtf_current_ids=_next_ids
                    _mtf_last_out=out
                    if _mtf_turn<_mtf_turns-1:
                        del out,_turn_loss
                # Use last turn's out for logging, synthesize raw_loss as average
                out=_mtf_last_out
                raw_loss=torch.tensor(_mtf_loss_total/float(_mtf_turns),device=bundle.device)
                del _mtf_last_out,_turn_loss
            except RuntimeError as e:
                _t_fwd1=time.time()
                if _global_micro<=2:
                    print(f"[_run] FORWARD_EXCEPTION opt_step={opt_steps} micro={_global_micro} elapsed={_t_fwd1-_t_fwd0:.3f}s error={str(e)[:100]}",flush=True)
                if "out of memory" in str(e).lower():
                    oom_happened=True
                    emergency=_save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"emergency_oom_step_{opt_steps:05d}",optimizer=opt)
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
                    _write_summary(reason="oom")
                    watchdog.stop()
                raise
            _t_fwd1=time.time()
            if _global_micro<=2:
                _vram_post_fwd=float(torch.cuda.memory_allocated())/1024/1024 if torch.cuda.is_available() else 0.0
                print(f"[_run] FORWARD_DONE opt_step={opt_steps} micro={_global_micro} elapsed={_t_fwd1-_t_fwd0:.3f}s loss={float(raw_loss.item()):.6f} vram_mb={_vram_post_fwd:.1f}",flush=True)
            if not bool(torch.isfinite(raw_loss).item()):
                nan_inf_count+=1
                _nan_skip_streak+=1
                _log(f"NAN_INF_DETECTED step={opt_steps} streak={_nan_skip_streak}/{_nan_skip_limit} total={nan_inf_count}")
                if _nan_skip_streak>=_nan_skip_limit:
                    nan_happened=True
                    emergency=_save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"emergency_nan_step_{opt_steps:05d}",optimizer=opt)
                    if tr is not None:
                        tr.record_fallback(
                            event="fallback",
                            module="training",
                            func="_run",
                            action="emergency_checkpoint",
                            reason="loss_nan_inf_persistent",
                            extra_dict={"kind":kind,"step":opt_steps,"emergency_checkpoint":emergency,"streak":_nan_skip_streak}
                        )
                    _write_summary(reason="nan_inf_persistent")
                    watchdog.stop()
                    raise RuntimeError(f"loss_nan_inf: {_nan_skip_streak} consecutive bad batches")
                # Progressive recovery: skip this batch, zero grads, continue
                del out,raw_loss
                opt.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            # backward already done inside MTF loop — skip redundant loss.backward()
            _nan_skip_streak=0  # reset streak on valid loss
            # Watchdog heartbeat + signal state update
            watchdog.heartbeat()
            _signal_save_state["step"]=opt_steps
            _out_loss_val=float(raw_loss.item())
            _out_t_sampled=float(out.get("t_sampled",cfg.train.mask_ratio))
            _out_mta=out.get("masked_token_acc")
            _out_loss_m2t=float(out["loss_m2t"].detach().item())
            _out_loss_m2t_scaled=float(out.get("loss_m2t_scaled",out["loss_m2t"]).detach().item()) if "loss_m2t_scaled" in out else _out_loss_m2t
            _out_loss_t2t=float(out["loss_t2t"].detach().item())
            _out_mask_ratio=float(out.get("mask_ratio_actual",0))
            _out_pred_count=int(out.get("pred_positions_count",0))
            del out,raw_loss
            _t_bwd1=time.time()
            if _global_micro<=2:
                _vram_post_bwd=float(torch.cuda.memory_allocated())/1024/1024 if torch.cuda.is_available() else 0.0
                print(f"[_run] FWD_BWD_DONE opt_step={opt_steps} micro={_global_micro} elapsed={_t_bwd1-_t_fwd0:.3f}s vram_mb={_vram_post_bwd:.1f}",flush=True)
            if step%grad_acc==0:
                grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip_val))
                clip_applied=bool(grad_norm>grad_clip_val)
                # Gradient explosion detection: if grad_norm is extreme, skip this step
                # With 1/t ELBO weighting, high grad_norm (100-2000) is expected — clip handles it.
                # Only skip for truly catastrophic values (NaN/Inf or > 10000x clip).
                _grad_explosion_threshold=10000*grad_clip_val
                if not math.isfinite(grad_norm) or grad_norm>_grad_explosion_threshold:
                    _log(f"GRAD_EXPLOSION grad_norm={grad_norm:.2f} threshold={_grad_explosion_threshold:.2f} — skipping optimizer step")
                    opt.zero_grad(set_to_none=True)
                    nan_inf_count+=1
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps+=1
                # Linearly decay embed noise during warmup
                if _embed_noise_warmup_steps>0 and opt_steps<_embed_noise_warmup_steps:
                    _noise_frac=1.0-float(opt_steps)/float(_embed_noise_warmup_steps)
                    set_embed_noise_std(0.1*max(0.0,_noise_frac))
                # LR schedule: linear warmup + cosine decay
                current_lr=_compute_lr(opt_steps,lr_warmup,max_steps,lr_base,lr_min_ratio)
                for pg in opt.param_groups:
                    pg["lr"]=current_lr
                last_loss=_out_loss_val
                # T-bucket aggregation
                t_val=_out_t_sampled
                bk=_t_bucket_key(t_val)
                t_bucket_loss[bk]["sum"]+=last_loss
                t_bucket_loss[bk]["count"]+=1
                mta_v=_out_mta
                if mta_v is not None:
                    t_bucket_acc[bk]["sum"]+=mta_v
                    t_bucket_acc[bk]["count"]+=1
                t_agg["sum"]+=t_val
                t_agg["min"]=min(t_agg["min"],t_val)
                t_agg["max"]=max(t_agg["max"],t_val)
                t_agg["count"]+=1
                elapsed=max(1e-6,time.time()-t0)
                t_now=time.time()
                sec_step=max(1e-6,t_now-t_last)
                t_last=t_now
                sec_roll=sec_step if sec_roll is None else (0.9*sec_roll+0.1*sec_step)
                remain=max(0,max_steps-opt_steps)
                eta=float(remain*(sec_roll or sec_step))
                peak_vram=None
                vram_alloc_mb=0.0
                vram_reserved_mb=0.0
                vram_peak_mb=0.0
                if bundle.device.type=="cuda":
                    try:
                        dev_idx=bundle.device.index or 0
                        peak_vram=float(torch.cuda.max_memory_allocated(dev_idx))
                        vram_alloc_mb=float(torch.cuda.memory_allocated(dev_idx))/1024/1024
                        vram_reserved_mb=float(torch.cuda.memory_reserved(dev_idx))/1024/1024
                        vram_peak_mb=peak_vram/1024/1024
                    except Exception:
                        pass
                tok_est=int(cfg.train.batch_size)*int(cfg.train.accum_steps)*int(cfg.data.seq_len)*int(opt_steps)
                tok_per_sec=tokens_per_second(token_seen,elapsed)
                # WSD phase progress
                w_total=cfg.wsd.warmup_steps+cfg.wsd.stable_steps+cfg.wsd.decay_steps
                if phase.phase=="warmup":
                    phase_progress=float(opt_steps)/max(1,cfg.wsd.warmup_steps)
                elif phase.phase=="stable":
                    phase_progress=float(opt_steps-cfg.wsd.warmup_steps)/max(1,cfg.wsd.stable_steps)
                else:
                    phase_progress=float(opt_steps-cfg.wsd.warmup_steps-cfg.wsd.stable_steps)/max(1,cfg.wsd.decay_steps)
                phase_progress=min(1.0,max(0.0,phase_progress))
                # T-bucket averages
                tb_loss={k:(t_bucket_loss[k]["sum"]/max(1,t_bucket_loss[k]["count"])) for k in _T_BUCKET_NAMES}
                tb_acc={k:(t_bucket_acc[k]["sum"]/max(1,t_bucket_acc[k]["count"])) for k in _T_BUCKET_NAMES}
                t_mean=t_agg["sum"]/max(1,t_agg["count"])
                loss_m2t_raw=_out_loss_m2t
                loss_m2t_scaled=_out_loss_m2t_scaled
                row={
                    "kind":kind,
                    "epoch":epoch,
                    "step":opt_steps,
                    "phase":phase.phase,
                    "block_size":phase.block_size,
                    "loss":last_loss,
                    "loss_masked":loss_m2t_raw,
                    "loss_scaled":loss_m2t_scaled,
                    "loss_m2t":loss_m2t_raw,
                    "loss_t2t":_out_loss_t2t,
                    "masked_token_acc":_out_mta,
                    "json_valid_rate":None,
                    "lr":current_lr,
                    "wd":float(cfg.train.weight_decay),
                    "grad_acc":grad_acc,
                    "micro_batch":int(cfg.train.batch_size),
                    "seq_len":int(cfg.data.seq_len),
                    "grad_norm":grad_norm,
                    "clip_applied":clip_applied,
                    "tokens_per_sec":tok_per_sec,
                    "step_time_s":sec_step,
                    "mem":mem_stats(bundle.device),
                    "vram_alloc_mb":round(vram_alloc_mb,1),
                    "vram_reserved_mb":round(vram_reserved_mb,1),
                    "vram_peak_mb":round(vram_peak_mb,1),
                    "nan_inf_detected":not math.isfinite(_out_loss_val),
                    "nan_inf_count":nan_inf_count,
                    "stage":"wsd" if kind=="cpt" else kind,
                    "step_current":int(opt_steps),
                    "steps_total":int(max_steps),
                    "tokens_seen_total":tok_est,
                    "sec_per_step_avg":float(sec_roll or sec_step),
                    "eta_stage_sec":eta,
                    "peak_vram_bytes":peak_vram,
                    # Diffusion-specific (continuous-time)
                    "t_sampled":_out_t_sampled,
                    "t_mean":t_mean,
                    "t_min":t_agg["min"],
                    "t_max":t_agg["max"],
                    "p_mask_expected":t_mean,
                    "mask_ratio_actual":_out_mask_ratio,
                    "loss_by_t_bucket":tb_loss,
                    "acc_masked_by_t_bucket":tb_acc,
                    "pred_positions_count":_out_pred_count,
                    # WSD schedule
                    "wsd_phase":phase.phase,
                    "wsd_block_size":phase.block_size,
                    "wsd_phase_progress":phase_progress,
                    # Bidirectional status
                    "bidirectional":bidirectional,
                    "is_causal_effective":not bidirectional,
                    "attention_mode":attn_mode,
                    "bidirectional_verified":bidir_verified,
                    "bidirectional_disabled_reason":bidir_disabled_reason,
                    # Left-shift
                    "shift_mode":shift_mode,
                    "time_param":time_param,
                    "loss_weighting":loss_weighting,
                }
                append_jsonl(log_path,[row])
                _is_first_step=(opt_steps==_resumed_opt_steps+1)
                if _is_first_step or opt_steps%log_every==0 or opt_steps==max_steps:
                    mta=row['masked_token_acc']
                    mta_s=f"{mta:.4f}" if mta is not None else "n/a"
                    _log(f"stage={row['stage']} step={row['step_current']}/{row['steps_total']} phase={row['phase']} block={row['block_size']} bidir={bidirectional} loss={row['loss']:.6f} loss_m={loss_m2t_raw:.4f} loss_s={loss_m2t_scaled:.4f} mta={mta_s} t={row['t_sampled']:.3f} mask%={row['mask_ratio_actual']:.3f} lr={current_lr:.2e} gn={grad_norm:.2f} clip={clip_applied} tok={row['tokens_seen_total']} sec={row['sec_per_step_avg']:.3f} eta={row['eta_stage_sec']:.0f}s vram={row['vram_peak_mb']:.0f}MB")
                    # Left-shift spot check every 500 steps
                    if opt_steps==1 or opt_steps%500==0:
                        _log(f"  SHIFT_CHECK shift_mode={shift_mode} labels_offset=+1 first_position_ignored=True pred_positions_count={_out_pred_count} masked_pred_positions_count={_out_pred_count}")
                    # T-bucket report every log_every
                    tb_line=" ".join(f"{k}:L={tb_loss[k]:.3f}/A={tb_acc[k]:.3f}" for k in _T_BUCKET_NAMES)
                    _log(f"  T_BUCKETS t_mean={t_mean:.3f} t_min={t_agg['min']:.3f} t_max={t_agg['max']:.3f} {tb_line}")
                if tr is not None and (_is_first_step or opt_steps%log_every==0 or opt_steps==max_steps):
                    tr.record_metric(name=f"{kind}.loss_total",value=last_loss,step=opt_steps,module="training",func="_run")
                    tr.record_metric(name=f"{kind}.loss_m2t",value=_out_loss_m2t,step=opt_steps,module="training",func="_run")
                    tr.record_metric(name=f"{kind}.loss_t2t",value=_out_loss_t2t,step=opt_steps,module="training",func="_run")
                if opt_steps==max_steps or opt_steps%ckpt_every==0:
                    _save_checkpoint(model,bundle.tokenizer,Path(ckpt_dir),f"step_{opt_steps:05d}",optimizer=opt,watchdog=watchdog)
                    _prune_checkpoints(Path(ckpt_dir),keep_last)
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                if opt_steps%eval_every==0 or opt_steps==max_steps:
                    ev=_periodic_eval(model,bundle.tokenizer,bundle.device,bundle.mask_id,cfg,seed=cfg.runtime.seed+opt_steps)
                    # P0.3: compute steps_to_converge per prompt, then average
                    stc_vals=[next((l["step"] for l in r.get("decode_logs",[]) if l.get("mask_ratio",1.0)==0.0),len(r.get("decode_logs",[]))) for r in ev.get("rows",[])]
                    ev["avg_steps_to_converge"]=sum(stc_vals)/len(stc_vals) if stc_vals else None
                    ev["kind"]=kind
                    ev["step"]=opt_steps
                    append_jsonl(eval_path,[ev])
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                # Periodic VRAM defrag: combat allocator fragmentation on tight-VRAM GPUs
                if opt_steps%50==0 and torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                if cooldown_every>0 and cooldown_seconds>0 and opt_steps<max_steps and opt_steps%cooldown_every==0:
                    _log(f"stage={row['stage']} cooldown_start step={opt_steps} sleep_sec={cooldown_seconds}")
                    if tr is not None:
                        tr.record_notice(module="training",func="_run",action="cooldown_sleep",reason="thermal_guard",extra_dict={"kind":kind,"step":opt_steps,"sleep_sec":cooldown_seconds})
                    time.sleep(float(cooldown_seconds))
                    _log(f"stage={row['stage']} cooldown_end step={opt_steps}")
                # Reactive thermal protection: if GPU > _thermal_limit, pause until it cools
                if opt_steps%10==0:
                    _gpu_t=_gpu_temp_celsius()
                    if _gpu_t is not None and _gpu_t>_thermal_limit:
                        _log(f"THERMAL_THROTTLE gpu_temp={_gpu_t}C limit={_thermal_limit}C — pausing until cool")
                        _cool_start=time.time()
                        while True:
                            time.sleep(15)
                            watchdog.heartbeat()
                            _gpu_t2=_gpu_temp_celsius()
                            if _gpu_t2 is None or _gpu_t2<_thermal_limit-5:
                                break
                            if time.time()-_cool_start>300:
                                _log(f"THERMAL_TIMEOUT gpu still {_gpu_t2}C after 5min — continuing anyway")
                                break
                        _log(f"THERMAL_RESUME gpu_temp={_gpu_t2}C cooldown_sec={time.time()-_cool_start:.0f}")
                # Check signal interrupt
                if _signal_save_state.get("requested"):
                    _log(f"SIGNAL_INTERRUPT step={opt_steps} — stopping training")
                    watchdog.stop()
                    return _write_summary(reason="signal_interrupt")
                if token_seen>=cfg.train.max_tokens:
                    break
        if token_seen>=cfg.train.max_tokens:
            break
        if len(loader)==0:
            break
    watchdog.stop()
    _signal_save_state.update({"model":None,"tokenizer":None,"optimizer":None,"ckpt_dir":None})
    summary=_write_summary(reason="normal")
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

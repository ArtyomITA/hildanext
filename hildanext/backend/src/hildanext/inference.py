# Inference engines: dInfer adapter + Transformers fallback.
# Main entrypoints: build_engine,load_model_bundle.
# Implements threshold-edit decode with degeneration guards and telemetry.
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Dict,Optional,Tuple,List
import sys
import time
import torch
from .config import AppConfig
from .tokenization import load_tokenizer,ensure_mask_token
from .diffusion import apply_remask
from .formulas import llada21_apply
from .utils import TinyCausalLM,choose_device,dtype_from_name,force_math_sdpa,seed_everything,tokens_per_second,env_issues
from .trace import use_trace,trace_from_cfg,exception_with_stack

@dataclass
class ModelBundle:
    model:Any
    tokenizer:Any
    device:torch.device
    mask_id:int
    vocab_size:int
    is_dummy:bool
    load_reason:str=""
    env_issues:Dict[str,str]=field(default_factory=dict)
    model_name_or_path:str=""
    requested_dtype:str=""
    actual_dtype:str=""
    fallbacks:List[Dict[str,Any]]=field(default_factory=list)

def _model_dir_ready(model_dir:str)->bool:
    p=Path(model_dir)
    if not p.exists():
        return False
    if not (p/"config.json").exists():
        return False
    return any((p/x).exists() for x in ["model.safetensors","pytorch_model.bin","model-00001-of-00002.safetensors"])

def _infer_param_dtype(model:Any)->str:
    try:
        return str(next(model.parameters()).dtype)
    except Exception:
        return "unknown"

def load_model_bundle(cfg:AppConfig,for_training:bool=False,trace=None)->ModelBundle:
    tr=use_trace(cfg,trace)
    force_math_sdpa()
    if tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="force_math_sdpa",
            reason="flash_attention_unavailable",
            extra_dict={"device_hint":cfg.runtime.device}
        )
    issues=env_issues()
    if tr is not None:
        for n,d in issues.items():
            tr.record_env_issue(name=f"{n}_unavailable",detail=str(d),module="inference",func="load_model_bundle")
    device=choose_device(cfg.runtime.device)
    if cfg.runtime.device=="cuda" and device.type!="cuda" and tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="cpu_fallback",
            reason="cuda_unavailable",
            extra_dict={"device_hint":cfg.runtime.device}
        )
    tok=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code,trace=tr,cfg=cfg)
    model=None
    is_dummy=False
    reason=""
    requested=str(cfg.train.dtype)
    if not cfg.runtime.force_dummy_model:
        if _model_dir_ready(cfg.paths.model_dir):
            try:
                from transformers import AutoModelForCausalLM
                td=dtype_from_name(cfg.train.dtype,device)
                kwargs={"trust_remote_code":cfg.model.trust_remote_code,"dtype":td}
                model=AutoModelForCausalLM.from_pretrained(cfg.paths.model_dir,**kwargs)
                model=model.to(device)
            except Exception as e:
                reason=f"load_failed:{e}"
                if tr is not None:
                    tr.record_fallback(
                        event="fallback",
                        module="inference",
                        func="load_model_bundle",
                        action="dummy_model_fallback",
                        reason="model_load_failed",
                        exception_str=exception_with_stack(e),
                        extra_dict={"model_dir":cfg.paths.model_dir}
                    )
                model=None
        else:
            reason="model_dir_invalid_or_missing"
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="inference",
                    func="load_model_bundle",
                    action="dummy_model_fallback",
                    reason="model_dir_invalid_or_missing",
                    extra_dict={"model_dir":cfg.paths.model_dir}
                )
    else:
        reason="force_dummy_model=true"
        if tr is not None:
            tr.record_fallback(
                event="fallback",
                module="inference",
                func="load_model_bundle",
                action="dummy_model_fallback",
                reason="force_dummy_model",
                extra_dict={"model_dir":cfg.paths.model_dir}
            )
    if model is None:
        is_dummy=True
        vocab=max(int(getattr(tok,"vocab_size",0) or 0),len(tok) if hasattr(tok,"__len__") else 0,32768)
        model=TinyCausalLM(vocab_size=vocab,hidden_size=256).to(device)
    if for_training and cfg.train.grad_ckpt and hasattr(model,"gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if for_training and hasattr(model,"config") and hasattr(model.config,"use_cache"):
        try:
            model.config.use_cache=False
        except Exception:
            pass
    mask_id=ensure_mask_token(tok,cfg.model.mask_token,model=model)
    model_vocab=int(getattr(getattr(model,"lm_head",None),"out_features",0) or 0)
    tok_vocab=int(len(tok) if hasattr(tok,"__len__") else 0)
    if is_dummy and mask_id>=model_vocab:
        new_vocab=max(mask_id+1,tok_vocab,model_vocab)
        model=TinyCausalLM(vocab_size=new_vocab,hidden_size=256).to(device)
        model_vocab=int(getattr(model.lm_head,"out_features",new_vocab))
    vocab_size=max(tok_vocab,model_vocab)
    model.train(mode=for_training)
    actual_dtype=_infer_param_dtype(model)
    if requested.lower() in {"bfloat16","bf16"} and "float16" in actual_dtype and tr is not None:
        tr.record_fallback(
            event="fallback",
            module="inference",
            func="load_model_bundle",
            action="dtype_fallback",
            reason="bf16_to_fp16",
            extra_dict={"requested_dtype":requested,"actual_dtype":actual_dtype}
        )
    return ModelBundle(
        model=model,
        tokenizer=tok,
        device=device,
        mask_id=mask_id,
        vocab_size=vocab_size,
        is_dummy=is_dummy,
        load_reason=reason,
        env_issues=issues,
        model_name_or_path=cfg.paths.model_dir,
        requested_dtype=requested,
        actual_dtype=actual_dtype,
        fallbacks=tr.snapshot_fallbacks(limit=64) if tr is not None else []
    )

def mode_thresholds(cfg:AppConfig,mode:str,tau_mask:Optional[float],tau_edit:Optional[float])->Tuple[float,float]:
    m=(mode or "S_MODE").upper()
    if tau_mask is not None and tau_edit is not None:
        return float(tau_mask),float(tau_edit)
    if m=="Q_MODE":
        return float(cfg.inference.q_mode_tau_mask),float(cfg.inference.q_mode_tau_edit)
    return float(cfg.inference.s_mode_tau_mask),float(cfg.inference.s_mode_tau_edit)

def _predict_autoregressive_candidates(model:Any,seq:torch.Tensor,prompt_len:int,max_new:int,mask_id:int)->Tuple[torch.Tensor,torch.Tensor]:
    work=seq.clone()
    pred=torch.zeros((1,max_new),dtype=torch.long,device=seq.device)
    conf=torch.zeros((1,max_new),dtype=torch.float32,device=seq.device)
    for i in range(max_new):
        g=prompt_len+i
        prefix=work[:,:g]
        if prefix.shape[1]==0:
            prefix=work[:,:1]
        out=model(input_ids=prefix)
        logits=out.logits[:,-1,:]
        probs=torch.softmax(logits,dim=-1)
        c,p=torch.max(probs,dim=-1)
        pred[:,i]=p
        conf[:,i]=c
        if int(work[0,g].item())==int(mask_id):
            work[:,g]=p
    return pred,conf

def _decode_text(tok:Any,out_ids:torch.Tensor,mask_id:int,is_dummy:bool)->str:
    text=""
    if hasattr(tok,"decode"):
        try:
            text=tok.decode(out_ids,skip_special_tokens=True)
        except Exception:
            text=""
    if text and text.strip():
        t=text.strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    raw=[int(x) for x in out_ids.detach().cpu().tolist() if int(x)!=int(mask_id)]
    if raw:
        t=" ".join(f"tok{x}" for x in raw[:64]).strip()
        return f"[DUMMY] {t}" if is_dummy and not t.startswith("[DUMMY] ") else t
    return "[DUMMY] dummy-output" if is_dummy else ""

class BaseEngine:
    name="base"
    def __init__(self,cfg:AppConfig,trace=None):
        self.cfg=cfg
        self.trace=use_trace(cfg,trace)
        self.last_stats:Dict[str,Any]={}
    def generate(self,prompt:str,mode:str="S_MODE",tau_mask:float|None=None,tau_edit:float|None=None,max_new_tokens:int|None=None,seed:int|None=None)->str:
        raise NotImplementedError
    def close(self)->None:
        return

class TransformersEngine(BaseEngine):
    name="transformers"
    def __init__(self,cfg:AppConfig,fallback_reason:str="",trace=None):
        super().__init__(cfg,trace=trace)
        self.bundle=load_model_bundle(cfg,for_training=False,trace=self.trace)
        self.fallback_reason=fallback_reason or self.bundle.load_reason
    def _decode(self,prompt:str,mode:str,tau_mask:float|None,tau_edit:float|None,max_new_tokens:int|None,seed:int|None)->str:
        if seed is None:
            seed=self.cfg.runtime.seed
        seed_everything(seed)
        tau_m,tau_e=mode_thresholds(self.cfg,mode,tau_mask,tau_edit)
        max_new=max(1,int(max_new_tokens or self.cfg.inference.max_new_tokens))
        tok=self.bundle.tokenizer
        model=self.bundle.model
        device=self.bundle.device
        mask_id=self.bundle.mask_id
        enc=tok([prompt],return_tensors="pt")
        input_ids=enc["input_ids"].to(device)
        prompt_len=int(input_ids.shape[1])
        seq=torch.full((1,prompt_len+max_new),mask_id,dtype=torch.long,device=device)
        seq[:,:prompt_len]=input_ids
        steps=max(1,int(self.cfg.inference.max_steps))
        logs=[]
        t0=time.time()
        model.eval()
        zero_gamma_streak=0
        patience=max(1,int(self.cfg.inference.degenerate_patience))
        strict=bool(self.cfg.inference.strict_decode_invariants)
        allow_fallback=bool(self.cfg.inference.allow_tau_fallback_on_degenerate)
        with torch.no_grad():
            for step in range(steps):
                gen_before=seq[:,prompt_len:]
                pred,conf=_predict_autoregressive_candidates(model,seq,prompt_len,max_new,mask_id)
                masked_before=gen_before.eq(mask_id)
                avg_masked=float(conf[masked_before].mean().item()) if masked_before.any() else None
                avg_tokens=float(conf[~masked_before].mean().item()) if (~masked_before).any() else None
                updated,sets=llada21_apply(gen_before,pred,conf,mask_id,tau_m,tau_e)
                if step+1<steps:
                    updated=apply_remask(updated,conf,mask_id,self.cfg.inference.remask)
                seq[:,prompt_len:]=updated
                remain=float(updated.eq(mask_id).float().mean().item())
                row={
                    "step":step+1,
                    "mask_ratio":remain,
                    "gamma_count":int(sets.gamma_count),
                    "delta_count":int(sets.delta_count),
                    "avg_conf_masked":avg_masked,
                    "avg_conf_tokens":avg_tokens,
                    "tau_mask":float(tau_m),
                    "tau_edit":float(tau_e)
                }
                logs.append(row)
                if sets.gamma_count==0:
                    zero_gamma_streak+=1
                else:
                    zero_gamma_streak=0
                if (not self.bundle.is_dummy) and zero_gamma_streak>=patience:
                    if allow_fallback:
                        tau_m=max(float(self.cfg.inference.min_tau_mask),float(tau_m)*float(self.cfg.inference.degenerate_tau_scale))
                        zero_gamma_streak=0
                        logs[-1]["tau_fallback_applied"]=True
                        logs[-1]["tau_mask_after_fallback"]=float(tau_m)
                        if self.trace is not None:
                            self.trace.record_fallback(
                                event="fallback",
                                module="inference",
                                func="TransformersEngine._decode",
                                action="tau_mask_relax",
                                reason="degenerate_tau_relax",
                                extra_dict={"step":step+1,"tau_mask":float(tau_m)}
                            )
                    elif strict:
                        elapsed=max(1e-6,time.time()-t0)
                        if self.trace is not None:
                            self.trace.record_fallback(
                                event="fallback",
                                module="inference",
                                func="TransformersEngine._decode",
                                action="decode_abort",
                                reason="degenerate_decoding",
                                extra_dict={"step":step+1,"patience":patience}
                            )
                        self.last_stats={
                            "engine":self.name,
                            "mode":mode,
                            "tau_mask":tau_m,
                            "tau_edit":tau_e,
                            "steps":len(logs),
                            "logs":logs,
                            "tokens_per_sec":tokens_per_second(int(max_new),elapsed),
                            "fallback_reason":self.fallback_reason,
                            "dummy_model":self.bundle.is_dummy,
                            "load_reason":self.bundle.load_reason,
                            "env_issues":self.bundle.env_issues,
                            "fallbacks":self.trace.snapshot_fallbacks(limit=64) if self.trace is not None else [],
                            "error":"degenerate decoding"
                        }
                        raise RuntimeError("degenerate decoding: gamma_count stayed 0")
                if remain==0.0 and sets.delta_count==0:
                    break
        elapsed=max(1e-6,time.time()-t0)
        out_ids=seq[0,prompt_len:]
        text=_decode_text(tok,out_ids,mask_id,self.bundle.is_dummy)
        self.last_stats={
            "engine":self.name,
            "mode":mode,
            "tau_mask":tau_m,
            "tau_edit":tau_e,
            "steps":len(logs),
            "logs":logs,
            "tokens_per_sec":tokens_per_second(int(max_new),elapsed),
            "fallback_reason":self.fallback_reason,
            "dummy_model":self.bundle.is_dummy,
            "load_reason":self.bundle.load_reason,
            "model_name_or_path":self.bundle.model_name_or_path,
            "requested_dtype":self.bundle.requested_dtype,
            "actual_dtype":self.bundle.actual_dtype,
            "env_issues":self.bundle.env_issues,
            "fallbacks":self.trace.snapshot_fallbacks(limit=64) if self.trace is not None else []
        }
        return text.strip()
    def generate(self,prompt:str,mode:str="S_MODE",tau_mask:float|None=None,tau_edit:float|None=None,max_new_tokens:int|None=None,seed:int|None=None)->str:
        return self._decode(prompt,mode,tau_mask,tau_edit,max_new_tokens,seed)

class DInferEngine(BaseEngine):
    name="dinfer"
    def __init__(self,cfg:AppConfig,trace=None):
        super().__init__(cfg,trace=trace)
        vendor_py=Path(cfg.paths.vendor_dinfer)/"python"
        if not vendor_py.exists():
            raise RuntimeError("dInfer python package path missing")
        sys.path.insert(0,str(vendor_py))
        from dinfer import SamplingParams,DiffusionLLMServing
        self.SamplingParams=SamplingParams
        self.DiffusionLLMServing=DiffusionLLMServing
        self.tokenizer=load_tokenizer(cfg.paths.model_dir,cfg.model.trust_remote_code,trace=self.trace,cfg=cfg)
        self.server=None
        self._init_server()
    def _init_server(self)->None:
        sample=self.SamplingParams(
            threshold=self.cfg.inference.q_mode_tau_mask,
            cache="prefix",
            temperature=0.0,
            early_stop=True,
            cont_weight=0.0,
            prefix_look=0,
            after_look=0,
            warmup_steps=0,
            enable_torch_compile=False,
            parallel_decoding="threshold",
            use_credit=False,
            use_bd=True,
            max_length=max(1024,self.cfg.data.seq_len),
            batch_size=1,
            mini_batch_size=1
        )
        self.server=self.DiffusionLLMServing(
            self.cfg.paths.model_dir,
            model_type=self.cfg.runtime.dinfer_model_type,
            sample_params=sample,
            backend=self.cfg.runtime.dinfer_backend,
            num_gpus=1,
            dp_size=1,
            tpep_size=1
        )
    def close(self)->None:
        try:
            if self.server is not None and hasattr(self.server,"stop_serving"):
                self.server.stop_serving()
        except Exception:
            pass
    def generate(self,prompt:str,mode:str="S_MODE",tau_mask:float|None=None,tau_edit:float|None=None,max_new_tokens:int|None=None,seed:int|None=None)->str:
        tau_m,_=mode_thresholds(self.cfg,mode,tau_mask,tau_edit)
        if hasattr(self.server,"sample_params"):
            try:
                self.server.sample_params.threshold=float(tau_m)
            except Exception:
                pass
        max_new=int(max_new_tokens or self.cfg.inference.max_new_tokens)
        x=self.tokenizer([prompt],return_tensors="pt")["input_ids"]
        t0=time.time()
        out=self.server.generate(x,gen_length=max_new,block_length=self.cfg.inference.block_size)
        elapsed=max(1e-6,time.time()-t0)
        prompt_len=int(x.shape[1])
        text=self.tokenizer.decode(out[0,prompt_len:],skip_special_tokens=True)
        if not text.strip():
            text="[DUMMY] dummy-output"
        self.last_stats={"engine":self.name,"mode":mode,"tau_mask":tau_m,"tau_edit":tau_edit,"steps":-1,"tokens_per_sec":tokens_per_second(int(max_new),elapsed),"fallbacks":self.trace.snapshot_fallbacks(limit=64) if self.trace is not None else []}
        return text.strip()

def build_engine(cfg:AppConfig,trace=None)->BaseEngine:
    tr=use_trace(cfg,trace)
    err=""
    if cfg.runtime.use_dinfer:
        try:
            return DInferEngine(cfg,trace=tr)
        except Exception as e:
            err=str(e)
            if tr is not None:
                tr.record_fallback(
                    event="fallback",
                    module="inference",
                    func="build_engine",
                    action="engine_fallback",
                    reason="dinfer_missing",
                    exception_str=exception_with_stack(e),
                    extra_dict={"engine_requested":"dinfer","backend":cfg.runtime.dinfer_backend}
                )
    return TransformersEngine(cfg,fallback_reason=err,trace=tr)

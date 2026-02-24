# Config loader for HildaNext SAFE backend.
# Main entrypoints: load_config,save_config,default_config.
# Format is JSON and fully config-driven.
from __future__ import annotations
from dataclasses import dataclass,asdict,field
from pathlib import Path
from typing import Any,Dict,List
import copy
import json
import os

@dataclass
class PathsConfig:
    root:str=""
    model_dir:str=""
    exports_dir:str=""
    raw_dir:str=""
    processed_dir:str=""
    tokenized_dir:str=""
    logs_dir:str=""
    checkpoints_dir:str=""
    vendor_dinfer:str=""
    vendor_llada:str=""

@dataclass
class DataConfig:
    dolma_path:str=""
    tinystories_path:str=""
    seq_len:int=256
    eval_ratio:float=0.05
    max_samples:int=4096
    doc_mask_mode:str="simple_blockdiag"
    eval_pct_stage0:float=0.01

@dataclass
class LLaDA2Config:
    mask_mode:str="composite_llada20"
    composite_block_size:int=32

@dataclass
class ModelConfig:
    mask_token:str="<|mask|>"
    trust_remote_code:bool=True

@dataclass
class WSDConfig:
    warmup_steps:int=100
    stable_steps:int=300
    decay_steps:int=100
    start_block_size:int=1
    max_block_size:int=512
    end_block_size:int=32
    ladder_blocks:List[int]=field(default_factory=list)
    decay_blocks:List[int]=field(default_factory=list)
    enforce_divisibility:bool=True

@dataclass
class TrainConfig:
    dtype:str="fp16"
    batch_size:int=1
    accum_steps:int=32
    grad_ckpt:bool=True
    max_vram_pct:float=0.85
    lr:float=1e-4
    warmup_steps:int=100
    max_steps:int=200
    max_tokens:int=2_000_000
    m2t_weight:float=0.6
    t2t_weight:float=0.4
    mask_ratio:float=0.15
    t2t_noise_ratio:float=0.15
    multi_turn_t2t:int=2
    optimizer:str="auto"
    weight_decay:float=0.01
    ckpt_every:int=50
    eval_every:int=100
    log_every_steps:int=10
    keep_last_checkpoints:int=3
    data_num_workers:int=4
    data_prefetch_factor:int=2
    data_persistent_workers:bool=True
    data_pin_memory:bool=True
    cooldown_every_steps:int=0
    cooldown_seconds:int=0

@dataclass
class Stage0Config:
    enabled:bool=True
    steps_total_stage0:int=10000
    lr_stage0:float=3e-5
    micro_batch_size:int=1
    grad_accum_steps:int=32
    seq_len:int=256
    log_every_steps:int=10
    eval_every_steps:int=0
    save_every_steps:int=200
    keep_last_checkpoints:int=3
    objective_mode:str="llada21_mixture"
    t2t_enabled:bool=True
    mask_ratio_m2t:float=0.15
    t2t_edit_ratio:float=0.10
    m2t_weight:float=1.0
    t2t_weight:float=1.0
    warmup_frac:float=0.2
    stable_frac:float=0.6
    decay_frac:float=0.2
    ladder_blocks:List[int]=field(default_factory=lambda:[1,4,32,64,256])
    decay_blocks:List[int]=field(default_factory=lambda:[256,128,64,32])
    doc_packing:bool=True
    doc_attention_mask_mode:str="composite_llada20"

@dataclass
class RecipeConfig:
    stage0_steps:int=400
    stage1_steps:int=400
    ckpt_every:int=50
    eval_every:int=100
    min_sft_records:int=64
    sft_wrap_from_tiny:bool=True
    eval_prompts:List[str]=field(default_factory=lambda:["Write one sentence about rain.","Q: 5+7? A:","Complete safely: The quick brown fox"])

@dataclass
class RemaskConfig:
    target_ratio:float=0.15
    min_ratio:float=0.05
    block_size:int=64
    block_stride:int=32
    percentile_safety:float=0.95

@dataclass
class InferenceConfig:
    s_mode_tau_mask:float=0.08
    s_mode_tau_edit:float=0.08
    q_mode_tau_mask:float=0.18
    q_mode_tau_edit:float=0.16
    max_steps:int=10
    max_new_tokens:int=96
    block_size:int=32
    strict_decode_invariants:bool=True
    allow_tau_fallback_on_degenerate:bool=False
    degenerate_patience:int=2
    degenerate_tau_scale:float=0.85
    min_tau_mask:float=0.05
    remask:RemaskConfig=field(default_factory=RemaskConfig)

@dataclass
class RuntimeConfig:
    seed:int=42
    device:str="auto"
    use_dinfer:bool=False
    dinfer_model_type:str="llada2-mini"
    dinfer_backend:str="vllm"
    force_dummy_model:bool=False
    run_id:str=""
    strict_fallbacks:bool=False
    fallback_whitelist:List[str]=field(default_factory=lambda:["flash_attention_unavailable","numpy_dll_unavailable","dinfer_missing"])
    blocking_fallback_actions:List[str]=field(default_factory=lambda:["synthetic_dolma","dummy_model_fallback","dataset_empty","download_false_empty"])
    blocking_fallback_reasons:List[str]=field(default_factory=lambda:["dolma_unavailable","dataset_empty"])
    min_disk_free_gb:float=20.0
    gibberish_ratio_warn:float=0.4
    vram_margin_gb:float=1.2
    oom_policy:str="downscale_seq_then_accum"

@dataclass
class AppConfig:
    paths:PathsConfig=field(default_factory=PathsConfig)
    data:DataConfig=field(default_factory=DataConfig)
    llada2:LLaDA2Config=field(default_factory=LLaDA2Config)
    model:ModelConfig=field(default_factory=ModelConfig)
    wsd:WSDConfig=field(default_factory=WSDConfig)
    train:TrainConfig=field(default_factory=TrainConfig)
    stage0:Stage0Config=field(default_factory=Stage0Config)
    recipe:RecipeConfig=field(default_factory=RecipeConfig)
    inference:InferenceConfig=field(default_factory=InferenceConfig)
    runtime:RuntimeConfig=field(default_factory=RuntimeConfig)

def _merge_dataclass(dc:Any,payload:Dict[str,Any])->Any:
    for k,v in payload.items():
        if not hasattr(dc,k):
            continue
        cur=getattr(dc,k)
        if hasattr(cur,"__dataclass_fields__") and isinstance(v,dict):
            _merge_dataclass(cur,v)
        else:
            setattr(dc,k,v)
    return dc

def _expand(s:str)->str:
    return os.path.expandvars(os.path.expanduser(s))

def resolve_paths(cfg:AppConfig)->AppConfig:
    for k,v in vars(cfg.paths).items():
        if isinstance(v,str) and v:
            setattr(cfg.paths,k,_expand(v))
    return cfg

def to_dict(cfg:AppConfig)->Dict[str,Any]:
    return asdict(cfg)

def from_dict(payload:Dict[str,Any])->AppConfig:
    cfg=AppConfig()
    _merge_dataclass(cfg,payload or {})
    return resolve_paths(cfg)

def load_config(path:str|Path)->AppConfig:
    p=Path(path)
    data=json.loads(p.read_text(encoding="utf-8"))
    return from_dict(data)

def save_config(cfg:AppConfig,path:str|Path)->None:
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(to_dict(cfg),indent=2,ensure_ascii=True),encoding="utf-8")

def default_config(root:str|Path)->AppConfig:
    r=Path(root)
    cfg=AppConfig()
    cfg.paths.root=str(r)
    cfg.paths.model_dir=str(r/"models"/"qwen3-0.6b")
    cfg.paths.exports_dir=str(r/"models"/"exports")
    cfg.paths.raw_dir=str(r/"data"/"raw")
    cfg.paths.processed_dir=str(r/"data"/"processed")
    cfg.paths.tokenized_dir=str(r/"data"/"tokenized")
    cfg.paths.logs_dir=str(r/"runs"/"logs")
    cfg.paths.checkpoints_dir=str(r/"runs"/"checkpoints")
    cfg.paths.vendor_dinfer=str(r/"vendor"/"dinfer")
    cfg.paths.vendor_llada=str(r/"vendor"/"llada")
    cfg.data.dolma_path=os.environ.get("HILDANEXT_DOLMA_PATH",str(Path(r).parent/"dolma_v1_6_sample_1767050862"))
    cfg.data.tinystories_path=os.environ.get("HILDANEXT_TINYSTORIES_PATH",str(r/"data"/"raw"/"tinystories_seed.jsonl"))
    return resolve_paths(cfg)

def clone_with_updates(cfg:AppConfig,updates:Dict[str,Any])->AppConfig:
    base=to_dict(cfg)
    def merge(a:Dict[str,Any],b:Dict[str,Any])->Dict[str,Any]:
        out=copy.deepcopy(a)
        for k,v in b.items():
            if k in out and isinstance(out[k],dict) and isinstance(v,dict):
                out[k]=merge(out[k],v)
            else:
                out[k]=v
        return out
    return from_dict(merge(base,updates))

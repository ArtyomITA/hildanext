"""Shared loader for overhead benchmarks.
Loads config, model, dataset once; provides helpers for optimizer creation and data iteration.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def fix_stdout_encoding():
    """Force UTF-8 stdout on Windows (cp1252 can't handle unicode chars)."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import torch
from torch.utils.data import DataLoader

from hildanext.config import AppConfig, load_config, clone_with_updates
from hildanext.training import MmapShardedDataset, _collate
from hildanext.inference import load_model_bundle
from hildanext.tokenization import ensure_mask_token
from hildanext.utils import force_math_sdpa, seed_everything
from hildanext.diffusion import (
    compute_m2t_t2t_losses,
    _install_embed_noise_hook,
    _remove_embed_noise_hook,
    set_embed_noise_std,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs" / "configs" / "llada21_dolma_wsd_only.json"
)


def _gpu_temp() -> int:
    """GPU temperature via nvidia-smi (fallback -1)."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def _gpu_temp_pynvml() -> int:
    """GPU temperature via pynvml (fallback -1)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return int(t)
    except Exception:
        return -1


def gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu": "N/A", "vram_gb": 0}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "temp_c": _gpu_temp(),
    }


def load_cfg(cfg_path: Optional[str] = None,
             overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    p = Path(cfg_path or DEFAULT_CFG_PATH)
    cfg = load_config(str(p))
    if overrides:
        cfg = clone_with_updates(cfg, overrides)
    return cfg


def load_bundle(cfg: AppConfig, for_training: bool = True):
    """Load model bundle with mask token ensured."""
    seed_everything(42)
    force_math_sdpa()
    bundle = load_model_bundle(cfg, for_training=for_training, trace=None)
    ensure_mask_token(bundle.tokenizer, cfg.model.mask_token, model=bundle.model)
    return bundle


def make_loader(cfg: AppConfig, shuffle: bool = True) -> DataLoader:
    dolma_root = str(Path(cfg.data.dolma_path).parent)
    ds = MmapShardedDataset(dolma_root)
    return DataLoader(
        dataset=ds, batch_size=max(1, cfg.train.batch_size),
        shuffle=shuffle, num_workers=0,
        collate_fn=_collate, pin_memory=torch.cuda.is_available(),
    )


def make_optimizer(name: str, model, lr: float, wd: float,
                   betas: Tuple[float, float] = (0.9, 0.95)):
    """Create optimizer by name. Returns (optimizer, label)."""
    if name == "PagedAdamW8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr,
                                         weight_decay=wd, betas=betas), name
    if name == "AdamW8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr,
                                    weight_decay=wd, betas=betas), name
    if name == "Adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(model.parameters(), lr=lr, relative_step=False,
                         scale_parameter=False, warmup_init=False,
                         weight_decay=wd), name
    if name == "AdamW_fused":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=betas, fused=True), name
    if name == "AdamW_vanilla":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=betas, fused=False), name
    raise ValueError(f"Unknown optimizer: {name}")


def reset_vram():
    """Empty cache and reset peak stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()


def vram_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "alloc_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def append_jsonl(path: Path, rows: list):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def forward_backward(model, batch, bundle, cfg, use_amp: bool,
                     ct_t_min: float, ct_t_max: float,
                     mask_mode: str = "simple_blockdiag",
                     bidirectional: bool = False,
                     grad_acc: int = 1):
    """Single forward+backward micro-batch. Returns (loss_val, out_dict, fwd_s, bwd_s)."""
    vocab_cap = max(8, bundle.vocab_size)
    batch["input_ids"] = torch.remainder(batch["input_ids"], vocab_cap)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        out = compute_m2t_t2t_losses(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            doc_ids=batch["doc_ids"],
            response_mask=batch["response_mask"],
            mask_id=bundle.mask_id,
            vocab_size=vocab_cap,
            cfg=cfg.train,
            focus_response=False,
            mask_mode=mask_mode,
            composite_block_size=cfg.llada2.composite_block_size,
            trace=None,
            cfg_obj=cfg,
            bidirectional=bidirectional,
            time_param="continuous_time",
            loss_weighting="inv_t",
            t_min=ct_t_min,
            t_max=ct_t_max,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    loss = out["loss"] / float(grad_acc)
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    loss_val = float(out["loss"].detach().item())
    return loss_val, out, t1 - t0, t2 - t1

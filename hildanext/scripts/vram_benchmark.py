#!/usr/bin/env python3
"""
VRAM Stability Benchmark Harness for HildaNext WSD Training
============================================================

Tests all combinations of (precision × optimizer × seq_len × grad_accum)
on the REAL Qwen3-0.6B-Base model backbone with SYNTHETIC input data.

Outputs:
  - Console: live progress + per-config PASS/FAIL
  - reports/vram_benchmark_results.json: full structured results
  - reports/vram_benchmark_report.txt: human-readable ranked table
  - reports/vram_benchmark_chart.png: matplotlib bar chart of VRAM usage

Usage:
  python -m hildanext.scripts.vram_benchmark
  python scripts/vram_benchmark.py
  python scripts/vram_benchmark.py --config-filter "fp16"
  python scripts/vram_benchmark.py --quick        # small matrix for fast test

Does NOT run actual WSD training. Does NOT touch real data.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vram_benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASK_TOKEN_ID = 151_643  # Qwen3 [MASK] or pad token
DEFAULT_VOCAB_SIZE = 151_669  # Qwen3-0.6B vocab
BENCHMARK_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """Single benchmark configuration to test."""
    name: str
    precision: str          # "fp16", "bf16", "fp32"
    optimizer: str          # "adamw", "adamw_8bit", "sgd", "adafactor"
    seq_len: int            # 128, 256, 512, 1024
    grad_accum: int         # 1, 4, 8, 16, 32
    micro_batch: int = 1
    gradient_checkpointing: bool = False
    pin_memory: bool = True
    num_workers: int = 0    # 0 = main process only (safe on Windows)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    def short_label(self) -> str:
        gc_tag = "+gc" if self.gradient_checkpointing else ""
        return f"{self.precision}_{self.optimizer}_s{self.seq_len}_ga{self.grad_accum}{gc_tag}"


@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    config: BenchConfig
    status: str             # "PASS", "OOM", "NAN", "CUDA_ERROR", "ERROR"
    error_msg: str = ""
    # VRAM (MB)
    vram_after_model_mb: float = 0.0
    vram_after_optimizer_mb: float = 0.0
    vram_peak_forward_mb: float = 0.0
    vram_peak_backward_mb: float = 0.0
    vram_peak_step_mb: float = 0.0
    vram_reserved_peak_mb: float = 0.0
    vram_fragmentation_pct: float = 0.0
    # Timing (seconds)
    model_load_sec: float = 0.0
    optimizer_create_sec: float = 0.0
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    optimizer_step_sec: float = 0.0
    total_step_sec: float = 0.0
    # Stability
    loss_value: float = 0.0
    loss_is_finite: bool = False
    grad_norm: float = 0.0
    grad_norm_is_finite: bool = False
    steps_completed: int = 0
    steps_attempted: int = 0
    # System
    gpu_name: str = ""
    gpu_total_mb: float = 0.0
    cuda_version: str = ""
    torch_version: str = ""

    def score(self) -> float:
        """Composite score: lower is better. Penalizes failures heavily."""
        if self.status != "PASS":
            return 99999.0
        # Weighted: 60% peak VRAM, 20% step time, 20% stability
        vram_score = self.vram_peak_step_mb / max(1, self.gpu_total_mb) * 100
        time_score = self.total_step_sec * 10  # normalize
        stability_penalty = 0.0 if (self.loss_is_finite and self.grad_norm_is_finite) else 50.0
        return vram_score * 0.6 + time_score * 0.2 + stability_penalty * 0.2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _vram_mb() -> float:
    """Current VRAM allocated in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated()) / 1024 / 1024


def _vram_reserved_mb() -> float:
    """Current VRAM reserved by allocator in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_reserved()) / 1024 / 1024


def _vram_peak_mb() -> float:
    """Peak VRAM allocated in MB since last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1024 / 1024


def _reset_peak():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _full_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _gpu_info() -> Tuple[str, float, str]:
    """Return (gpu_name, total_mb, cuda_version)."""
    if not torch.cuda.is_available():
        return ("N/A", 0.0, "N/A")
    name = torch.cuda.get_device_name(0)
    total = float(torch.cuda.get_device_properties(0).total_mem) / 1024 / 1024
    cuda_ver = torch.version.cuda or "N/A"
    return (name, total, cuda_ver)


def _make_synthetic_batch(
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Create a synthetic training batch mimicking TokenizedDataset output."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    doc_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "doc_ids": doc_ids,
        "response_mask": response_mask,
    }


def _get_torch_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping.get(precision, torch.float16)


def _force_math_sdpa():
    """Force math SDPA backend (required for Pascal/GTX 1080)."""
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        log.info("Forced math SDPA backend (Flash/MemEfficient disabled)")
    except Exception as e:
        log.warning(f"Could not force math SDPA: {e}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_model_path() -> str:
    """Locate the Qwen3-0.6B model. Tries local path first, then HF hub name."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "Qwen3-0.6B",
        Path("E:/DIFFUSION/HildaNext/Qwen3-0.6B"),
        Path("Qwen3-0.6B"),
    ]
    for p in candidates:
        if p.exists() and (p / "config.json").exists():
            log.info(f"Found local model at {p}")
            return str(p)
    # Fallback to HF hub
    return "Qwen/Qwen3-0.6B-Base"


def _load_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Load the real Qwen3-0.6B model for benchmarking."""
    from transformers import AutoModelForCausalLM, AutoConfig

    log.info(f"Loading model from {model_path} dtype={dtype} device={device}")
    t0 = time.time()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,  # manual placement
    )
    model = model.to(device)
    model.train()

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            log.info("Gradient checkpointing enabled")
        except Exception as e:
            log.warning(f"Could not enable gradient checkpointing: {e}")

    elapsed = time.time() - t0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"Model loaded: {params_m:.1f}M params ({trainable_m:.1f}M trainable) in {elapsed:.1f}s")
    return model


def _make_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Create optimizer by name."""
    params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Creating optimizer: {optimizer_name} lr={lr} wd={weight_decay} params={len(params)}")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            log.warning("bitsandbytes not available, falling back to standard AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(
                params, lr=lr, weight_decay=weight_decay,
                relative_step=False, warmup_init=False,
            )
        except ImportError:
            log.warning("Adafactor not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_single_benchmark(
    cfg: BenchConfig,
    model_path: str,
    device: torch.device,
    num_steps: int = 3,
    warmup_steps: int = 1,
) -> BenchResult:
    """
    Run a single benchmark configuration.

    1. Load model with specified precision
    2. Create optimizer
    3. Run `num_steps` forward+backward+optimizer.step with synthetic data
    4. Measure VRAM, timing, and stability at each phase
    5. Return structured result
    """
    gpu_name, gpu_total_mb, cuda_ver = _gpu_info()
    result = BenchResult(
        config=cfg,
        status="ERROR",
        gpu_name=gpu_name,
        gpu_total_mb=gpu_total_mb,
        cuda_version=cuda_ver,
        torch_version=torch.__version__,
        steps_attempted=num_steps,
    )

    dtype = _get_torch_dtype(cfg.precision)
    model = None
    opt = None

    try:
        # ---- Cleanup before start ----
        _full_cleanup()
        _reset_peak()

        # ---- Step 1: Load model ----
        t0 = time.time()
        model = _load_model(model_path, dtype, device, cfg.gradient_checkpointing)
        result.model_load_sec = time.time() - t0
        result.vram_after_model_mb = _vram_mb()
        log.info(f"  VRAM after model: {result.vram_after_model_mb:.1f} MB")

        # ---- Step 2: Create optimizer ----
        _reset_peak()
        t0 = time.time()
        opt = _make_optimizer(model, cfg.optimizer)
        result.optimizer_create_sec = time.time() - t0
        result.vram_after_optimizer_mb = _vram_mb()
        log.info(f"  VRAM after optimizer: {result.vram_after_optimizer_mb:.1f} MB (delta: {result.vram_after_optimizer_mb - result.vram_after_model_mb:.1f} MB)")

        # ---- Step 3: Training steps ----
        vocab_size = DEFAULT_VOCAB_SIZE
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            vocab_size = model.config.vocab_size

        # Use AMP scaler for fp16
        use_amp = cfg.precision in ("fp16", "bf16")
        amp_dtype = dtype if use_amp else torch.float32
        scaler = torch.amp.GradScaler('cuda', enabled=(cfg.precision == "fp16"))

        total_forward = 0.0
        total_backward = 0.0
        total_opt_step = 0.0
        last_loss = 0.0
        last_grad_norm = 0.0
        peak_forward = 0.0
        peak_backward = 0.0
        peak_step = 0.0

        for step_i in range(1, num_steps + 1):
            _reset_peak()
            batch = _make_synthetic_batch(cfg.seq_len, cfg.micro_batch, vocab_size, device, dtype)

            # ---- Forward pass ----
            torch.cuda.synchronize()
            t_fwd_start = time.time()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],  # self-supervised
                )
                raw_loss = outputs.loss

            torch.cuda.synchronize()
            t_fwd_end = time.time()
            fwd_time = t_fwd_end - t_fwd_start

            _peak_after_fwd = _vram_peak_mb()
            peak_forward = max(peak_forward, _peak_after_fwd)

            # ---- Check loss ----
            loss_val = float(raw_loss.detach().item())
            if not math.isfinite(loss_val):
                result.status = "NAN"
                result.error_msg = f"Loss is {loss_val} at step {step_i}"
                result.loss_value = loss_val
                result.loss_is_finite = False
                log.error(f"  NaN/Inf loss at step {step_i}: {loss_val}")
                break

            # ---- Backward pass ----
            loss_scaled = raw_loss / float(cfg.grad_accum)
            torch.cuda.synchronize()
            t_bwd_start = time.time()

            scaler.scale(loss_scaled).backward()

            torch.cuda.synchronize()
            t_bwd_end = time.time()
            bwd_time = t_bwd_end - t_bwd_start

            _peak_after_bwd = _vram_peak_mb()
            peak_backward = max(peak_backward, _peak_after_bwd)

            # ---- Optimizer step (every grad_accum steps, or on every step for simplicity) ----
            torch.cuda.synchronize()
            t_opt_start = time.time()

            if step_i % cfg.grad_accum == 0 or step_i == num_steps:
                scaler.unscale_(opt)
                grad_norm_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                grad_norm_val = 0.0

            torch.cuda.synchronize()
            t_opt_end = time.time()
            opt_time = t_opt_end - t_opt_start

            _peak_after_opt = _vram_peak_mb()
            peak_step = max(peak_step, _peak_after_opt)

            # ---- Record ----
            total_forward += fwd_time
            total_backward += bwd_time
            total_opt_step += opt_time
            last_loss = loss_val
            last_grad_norm = grad_norm_val
            result.steps_completed = step_i

            is_warmup = step_i <= warmup_steps
            tag = "WARMUP" if is_warmup else "BENCH"
            log.info(
                f"  [{tag}] step={step_i}/{num_steps} "
                f"loss={loss_val:.4f} grad_norm={grad_norm_val:.4f} "
                f"fwd={fwd_time:.3f}s bwd={bwd_time:.3f}s opt={opt_time:.3f}s "
                f"vram_peak={_peak_after_opt:.0f}MB"
            )

            # Free batch to reduce memory pressure between steps
            del batch, outputs, raw_loss, loss_scaled
            _full_cleanup()

        # ---- Record final metrics ----
        if result.status != "NAN":
            result.status = "PASS"
        result.loss_value = last_loss
        result.loss_is_finite = math.isfinite(last_loss)
        result.grad_norm = last_grad_norm
        result.grad_norm_is_finite = math.isfinite(last_grad_norm)
        result.vram_peak_forward_mb = peak_forward
        result.vram_peak_backward_mb = peak_backward
        result.vram_peak_step_mb = peak_step
        result.vram_reserved_peak_mb = _vram_reserved_mb()
        result.forward_sec = total_forward / max(1, num_steps)
        result.backward_sec = total_backward / max(1, num_steps)
        result.optimizer_step_sec = total_opt_step / max(1, num_steps)
        result.total_step_sec = result.forward_sec + result.backward_sec + result.optimizer_step_sec

        # Fragmentation
        alloc = _vram_mb()
        reserved = _vram_reserved_mb()
        if reserved > 0:
            result.vram_fragmentation_pct = max(0, (reserved - alloc) / reserved * 100)

    except RuntimeError as e:
        err_str = str(e).lower()
        if "out of memory" in err_str:
            result.status = "OOM"
            result.error_msg = str(e)[:200]
            log.error(f"  OOM: {result.error_msg}")
        elif "cuda" in err_str:
            result.status = "CUDA_ERROR"
            result.error_msg = str(e)[:200]
            log.error(f"  CUDA error: {result.error_msg}")
        else:
            result.status = "ERROR"
            result.error_msg = traceback.format_exc()[-500:]
            log.error(f"  Error: {result.error_msg}")
    except Exception as e:
        result.status = "ERROR"
        result.error_msg = traceback.format_exc()[-500:]
        log.error(f"  Unexpected error: {result.error_msg}")
    finally:
        # Aggressive cleanup
        if model is not None:
            del model
        if opt is not None:
            del opt
        _full_cleanup()

    return result


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

def build_full_matrix() -> List[BenchConfig]:
    """Build the full configuration matrix to test."""
    configs: List[BenchConfig] = []

    precisions = ["fp16", "fp32"]
    # bf16 not supported on GTX 1080 (Pascal), but include for completeness
    # It will fail gracefully and be marked as such

    optimizers = ["adamw", "adamw_8bit", "sgd"]
    seq_lens = [128, 256, 512, 1024]
    grad_accums = [1, 4, 8, 16]

    # Also test gradient checkpointing for larger configs
    gc_options = [False, True]

    for prec in precisions:
        for opt in optimizers:
            for sl in seq_lens:
                for ga in grad_accums:
                    for gc_on in gc_options:
                        # Skip redundant combos: gc only helps with larger seq_lens
                        if gc_on and sl <= 128:
                            continue
                        # Skip fp32 + large seq_len (guaranteed OOM on 8GB)
                        if prec == "fp32" and sl >= 1024 and not gc_on:
                            continue

                        name = f"{prec}_{opt}_s{sl}_ga{ga}"
                        if gc_on:
                            name += "_gc"

                        configs.append(BenchConfig(
                            name=name,
                            precision=prec,
                            optimizer=opt,
                            seq_len=sl,
                            grad_accum=ga,
                            gradient_checkpointing=gc_on,
                        ))

    return configs


def build_quick_matrix() -> List[BenchConfig]:
    """Build a quick matrix for fast testing (fewer combos)."""
    configs: List[BenchConfig] = []

    precisions = ["fp16"]
    optimizers = ["adamw", "adamw_8bit"]
    seq_lens = [256, 512, 1024]
    grad_accums = [1, 16]

    for prec in precisions:
        for opt in optimizers:
            for sl in seq_lens:
                for ga in grad_accums:
                    name = f"{prec}_{opt}_s{sl}_ga{ga}"
                    configs.append(BenchConfig(
                        name=name,
                        precision=prec,
                        optimizer=opt,
                        seq_len=sl,
                        grad_accum=ga,
                    ))

    # Add one gradient checkpointing test
    configs.append(BenchConfig(
        name="fp16_adamw_s1024_ga16_gc",
        precision="fp16",
        optimizer="adamw",
        seq_len=1024,
        grad_accum=16,
        gradient_checkpointing=True,
    ))

    return configs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report_table(results: List[BenchResult]) -> str:
    """Generate a human-readable ranked report table."""
    lines: List[str] = []
    lines.append("=" * 120)
    lines.append("VRAM STABILITY BENCHMARK REPORT")
    lines.append(f"Version: {BENCHMARK_VERSION}")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if results:
        r0 = results[0]
        lines.append(f"GPU: {r0.gpu_name} ({r0.gpu_total_mb:.0f} MB)")
        lines.append(f"CUDA: {r0.cuda_version}  PyTorch: {r0.torch_version}")
    lines.append("=" * 120)
    lines.append("")

    # Sort by score (lower is better)
    ranked = sorted(results, key=lambda r: r.score())

    # Header
    hdr = (
        f"{'Rank':>4}  {'Status':>6}  {'Config':<40}  "
        f"{'VRAM Peak':>10}  {'VRAM Model':>10}  {'VRAM Opt':>10}  "
        f"{'Fwd(s)':>8}  {'Bwd(s)':>8}  {'Step(s)':>8}  "
        f"{'Loss':>10}  {'GradNorm':>10}  {'Score':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    best_pass = None
    for rank, r in enumerate(ranked, 1):
        if r.status == "PASS" and best_pass is None:
            best_pass = r
            marker = " ★ RECOMMENDED"
        else:
            marker = ""

        row = (
            f"{rank:>4}  {r.status:>6}  {r.config.short_label():<40}  "
            f"{r.vram_peak_step_mb:>9.0f}M  {r.vram_after_model_mb:>9.0f}M  {r.vram_after_optimizer_mb:>9.0f}M  "
            f"{r.forward_sec:>8.3f}  {r.backward_sec:>8.3f}  {r.total_step_sec:>8.3f}  "
            f"{r.loss_value:>10.4f}  {r.grad_norm:>10.4f}  {r.score():>8.1f}"
            f"{marker}"
        )
        lines.append(row)

    lines.append("")
    lines.append("=" * 120)

    # Summary statistics
    passed = [r for r in results if r.status == "PASS"]
    oom = [r for r in results if r.status == "OOM"]
    nan_r = [r for r in results if r.status == "NAN"]
    errs = [r for r in results if r.status in ("ERROR", "CUDA_ERROR")]

    lines.append(f"SUMMARY: {len(passed)} PASS / {len(oom)} OOM / {len(nan_r)} NAN / {len(errs)} ERROR  (total: {len(results)})")
    lines.append("")

    if best_pass:
        lines.append("★ RECOMMENDED CONFIGURATION:")
        lines.append(f"  Precision:      {best_pass.config.precision}")
        lines.append(f"  Optimizer:      {best_pass.config.optimizer}")
        lines.append(f"  Seq Length:     {best_pass.config.seq_len}")
        lines.append(f"  Grad Accum:     {best_pass.config.grad_accum}")
        lines.append(f"  Grad Ckpt:      {best_pass.config.gradient_checkpointing}")
        lines.append(f"  VRAM Peak:      {best_pass.vram_peak_step_mb:.0f} MB / {best_pass.gpu_total_mb:.0f} MB ({best_pass.vram_peak_step_mb/max(1,best_pass.gpu_total_mb)*100:.1f}%)")
        lines.append(f"  Step Time:      {best_pass.total_step_sec:.3f}s")
        lines.append(f"  Fragmentation:  {best_pass.vram_fragmentation_pct:.1f}%")
        lines.append(f"  Score:          {best_pass.score():.1f}")
        lines.append("")

    # OOM analysis
    if oom:
        lines.append("OOM CONFIGURATIONS (exceeded 8GB VRAM):")
        for r in oom:
            lines.append(f"  {r.config.short_label()}: peaked at ~{r.vram_peak_step_mb:.0f}MB")
        lines.append("")

    # Stability warnings
    fragmented = [r for r in passed if r.vram_fragmentation_pct > 20]
    if fragmented:
        lines.append("⚠ HIGH FRAGMENTATION (>20%) CONFIGS:")
        for r in fragmented:
            lines.append(f"  {r.config.short_label()}: {r.vram_fragmentation_pct:.1f}%")
        lines.append("")

    # VRAM headroom analysis
    if passed:
        lines.append("VRAM HEADROOM ANALYSIS:")
        for r in sorted(passed, key=lambda x: x.vram_peak_step_mb):
            headroom = r.gpu_total_mb - r.vram_peak_step_mb
            pct = r.vram_peak_step_mb / max(1, r.gpu_total_mb) * 100
            bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
            lines.append(f"  {r.config.short_label():<45} [{bar}] {pct:5.1f}%  headroom={headroom:.0f}MB")
        lines.append("")

    lines.append("=" * 120)
    return "\n".join(lines)


def generate_chart(results: List[BenchResult], output_path: Path):
    """Generate a matplotlib bar chart of VRAM usage by config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping chart generation")
        return

    passed = [r for r in results if r.status == "PASS"]
    if not passed:
        log.warning("No PASS results to chart")
        return

    # Sort by VRAM peak
    passed.sort(key=lambda r: r.vram_peak_step_mb)

    labels = [r.config.short_label() for r in passed]
    vram_model = [r.vram_after_model_mb for r in passed]
    vram_opt_delta = [r.vram_after_optimizer_mb - r.vram_after_model_mb for r in passed]
    vram_act_delta = [r.vram_peak_step_mb - r.vram_after_optimizer_mb for r in passed]

    fig, ax = plt.subplots(figsize=(max(12, len(passed) * 0.8), 8))

    x = range(len(passed))
    bar_width = 0.6

    ax.bar(x, vram_model, bar_width, label="Model Weights", color="#2196F3")
    ax.bar(x, vram_opt_delta, bar_width, bottom=vram_model, label="Optimizer States", color="#FF9800")
    ax.bar(x, vram_act_delta, bar_width,
           bottom=[m + o for m, o in zip(vram_model, vram_opt_delta)],
           label="Activations + Gradients", color="#F44336")

    # GPU total line
    if passed:
        gpu_total = passed[0].gpu_total_mb
        ax.axhline(y=gpu_total, color='red', linestyle='--', linewidth=2, label=f"GPU Total ({gpu_total:.0f} MB)")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("VRAM (MB)", fontsize=12)
    ax.set_title("VRAM Usage by Configuration – HildaNext WSD Benchmark", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    log.info(f"Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    matrix: List[BenchConfig],
    model_path: str,
    output_dir: Path,
    num_steps: int = 3,
    warmup_steps: int = 1,
    config_filter: Optional[str] = None,
) -> List[BenchResult]:
    """
    Run the full benchmark matrix.

    Args:
        matrix: List of configs to test
        model_path: Path to Qwen3-0.6B model
        output_dir: Where to save reports
        num_steps: Training steps per config
        warmup_steps: Warmup steps (excluded from timing avg)
        config_filter: Optional substring filter on config names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.error("No CUDA GPU available. Benchmark requires GPU.")
        sys.exit(1)

    # Force math SDPA for Pascal
    _force_math_sdpa()

    # Filter configs
    if config_filter:
        matrix = [c for c in matrix if config_filter.lower() in c.name.lower()]
        log.info(f"Filtered to {len(matrix)} configs matching '{config_filter}'")

    gpu_name, gpu_total_mb, cuda_ver = _gpu_info()
    log.info(f"GPU: {gpu_name} ({gpu_total_mb:.0f} MB)")
    log.info(f"CUDA: {cuda_ver}  PyTorch: {torch.__version__}")
    log.info(f"Model: {model_path}")
    log.info(f"Configs to test: {len(matrix)}")
    log.info(f"Steps per config: {num_steps} (warmup: {warmup_steps})")
    log.info("=" * 80)

    results: List[BenchResult] = []

    for i, cfg in enumerate(matrix, 1):
        log.info(f"\n{'='*60}")
        log.info(f"[{i}/{len(matrix)}] Testing: {cfg.short_label()}")
        log.info(f"  precision={cfg.precision} optimizer={cfg.optimizer} seq_len={cfg.seq_len} grad_accum={cfg.grad_accum} gc={cfg.gradient_checkpointing}")
        log.info(f"{'='*60}")

        # Aggressive cleanup between configs
        _full_cleanup()
        time.sleep(1)  # Let GPU cool down briefly

        result = run_single_benchmark(
            cfg=cfg,
            model_path=model_path,
            device=device,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
        )
        results.append(result)

        log.info(f"  Result: {result.status} | VRAM peak={result.vram_peak_step_mb:.0f}MB | score={result.score():.1f}")

    # ---- Generate outputs ----
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / "vram_benchmark_results.json"
    json_data = {
        "version": BENCHMARK_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": gpu_name,
        "gpu_total_mb": gpu_total_mb,
        "cuda_version": cuda_ver,
        "torch_version": torch.__version__,
        "model_path": model_path,
        "num_configs": len(matrix),
        "num_steps": num_steps,
        "results": [
            {
                "config": asdict(r.config),
                "status": r.status,
                "error_msg": r.error_msg,
                "vram_after_model_mb": r.vram_after_model_mb,
                "vram_after_optimizer_mb": r.vram_after_optimizer_mb,
                "vram_peak_forward_mb": r.vram_peak_forward_mb,
                "vram_peak_backward_mb": r.vram_peak_backward_mb,
                "vram_peak_step_mb": r.vram_peak_step_mb,
                "vram_reserved_peak_mb": r.vram_reserved_peak_mb,
                "vram_fragmentation_pct": r.vram_fragmentation_pct,
                "model_load_sec": r.model_load_sec,
                "optimizer_create_sec": r.optimizer_create_sec,
                "forward_sec": r.forward_sec,
                "backward_sec": r.backward_sec,
                "optimizer_step_sec": r.optimizer_step_sec,
                "total_step_sec": r.total_step_sec,
                "loss_value": r.loss_value,
                "loss_is_finite": r.loss_is_finite,
                "grad_norm": r.grad_norm,
                "grad_norm_is_finite": r.grad_norm_is_finite,
                "steps_completed": r.steps_completed,
                "steps_attempted": r.steps_attempted,
                "score": r.score(),
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    log.info(f"JSON results saved to {json_path}")

    # Text report
    report_path = output_dir / "vram_benchmark_report.txt"
    report_text = generate_report_table(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"Text report saved to {report_path}")
    print("\n" + report_text)

    # Chart
    chart_path = output_dir / "vram_benchmark_chart.png"
    generate_chart(results, chart_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VRAM Stability Benchmark for HildaNext WSD Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick matrix (fewer combos) instead of full matrix",
    )
    parser.add_argument(
        "--config-filter", type=str, default=None,
        help="Only test configs whose name contains this substring",
    )
    parser.add_argument(
        "--steps", type=int, default=3,
        help="Number of training steps per config (default: 3)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup steps excluded from timing average (default: 1)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to Qwen3-0.6B model (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for reports (default: runs/reports/)",
    )

    args = parser.parse_args()

    # Model path
    model_path = args.model_path or _find_model_path()

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try to find hildanext root
        candidates = [
            Path(__file__).resolve().parent.parent.parent.parent / "runs" / "reports",
            Path("E:/DIFFUSION/HildaNext/hildanext/runs/reports"),
            Path("runs/reports"),
        ]
        output_dir = next((p.parent.parent / "runs" / "reports" for p in candidates if p.parent.exists()), Path("reports"))

    # Build matrix
    if args.quick:
        matrix = build_quick_matrix()
        log.info(f"Quick matrix: {len(matrix)} configs")
    else:
        matrix = build_full_matrix()
        log.info(f"Full matrix: {len(matrix)} configs")

    # Run
    results = run_benchmark(
        matrix=matrix,
        model_path=model_path,
        output_dir=output_dir,
        num_steps=args.steps,
        warmup_steps=args.warmup,
        config_filter=args.config_filter,
    )

    # Exit code: 0 if at least one PASS, 1 otherwise
    passed = any(r.status == "PASS" for r in results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

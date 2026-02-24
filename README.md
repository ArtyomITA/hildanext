# HILDA

> **Give any autoregressive transformer a discrete-diffusion brain, teach it to reason, and serve it on hardware from 2017.**

HILDA is a research architecture and end-to-end backend for **converting AR transformers into editable discrete diffusion language models (dLLMs)**, then pushing them through supervised fine-tuning, RL-based reasoning alignment, and accelerated serving — all on a single consumer GPU.

The thesis: diffusion LMs can match or beat AR models of the same size when built with the right training mechanics, and the full pipeline should fit inside 8 GB VRAM.

---

## The Architecture

HILDA is built around four interlocking design pillars.

### 1 — Editable Diffusion Core (LLaDA 2.1 style)

Classical masked diffusion only knows one move: `[MASK] → token`.  
HILDA's core adds a second move: `token → token` (T2T editing).

This means the model can **revisit and correct already-placed tokens** during decoding, not just fill in blanks. The two operations are interleaved via configurable thresholds:

```
Γt  unmasking  — MASK → token  when p > τ_mask   (commit)
Δt  editing    — token → token when p > τ_edit   (correct)
```

Two runtime presets expose the speed/quality knob explicitly:

| Preset | Behaviour |
|---|---|
| `S_MODE` (quality) | low τ_mask, more correction passes — best output |
| `Q_MODE` (speed) | high τ_mask, conservative drafting — fastest throughput |

Training objective is a mixture of M2T and T2T losses with doc-level attention masking built from `doc_ids`, preventing cross-document attention leakage.

### 2 — WSD Conversion Schedule

HILDA converts an existing AR model rather than training from scratch. The **Warmup-Stable-Decay** schedule bridges the two paradigms:

```
Warmup  → block size grows 1 → N  (AR treated as BDLM, block size = 1)
Stable  → full-sequence MDLM regime (stabilise ELBO and diffusion dynamics)
Decay   → shrink block, consolidate editable representation
```

The CPT objective combines M2T loss on masked positions with T2T loss on noised observed positions. This is the cheapest path from a pretrained AR to a capable dLLM — no training from scratch, no architecture surgery.

### 3 — Structural Supervision: C2DLM *(FULL profile)*

On top of the diffusion core, the FULL architecture adds **concept-level causal supervision** via C2DLM:

- A concept-level causal graph constrains which attention heads can attend to which positions.
- A supervised attention mask enforces causal consistency during both M2T and T2T passes.
- V-aware re-attention re-weights value vectors based on concept role.

In SAFE mode this layer is optional. In FULL mode it is applied preferentially during T2T correction passes, where causal consistency matters most (reasoning chains, multi-turn coherence).

### 4 — RL Reasoning Alignment (Stage 2)

Post-SFT alignment uses **ESPO** as the principled default: an ELBO sequence-level RL objective with ratio-stabilised KL that avoids the token-level factorisation problem inherent to standard PPO on dLLMs.

Planned upgrades (FULL profile):

| Method | What it adds |
|---|---|
| **AGRPO** | Monte-Carlo faithful policy gradient, explicitly designed for dLLM step structure |
| **wd1 / wd1++** | Ratio-free weighted log-likelihood — lower compute, same stability |
| **STP** | Spatio-temporal pruning of redundant denoising steps — reduces gradient variance |
| **LENS** | Instruction-token interference filtering — improves RLVR rollout quality |
| **R³L** | Reflect-then-retry credit assignment for agentic tasks |

---

## System Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Pretrained AR  (Qwen3-0.6B / 1.7B-Base, Apache-2.0)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Stage 0 — CPT / WSD   │  Dolma v1.6 + TinyStories
              │   M2T + T2T objective   │  doc-level attention mask
              │   Warmup→Stable→Decay   │  packed shards w/ doc_ids
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Stage 1 — SFT         │  response-focused masking
              │   mixture M2T + T2T     │  multi-turn (2× T2T/step)
              │   [+ C2DLM FULL]        │  C2DLM on correction passes
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Stage 2 — RL          │  ESPO (ELBO seq-level)
              │   verifiable rewards    │  GSM8K / HumanEval
              │   [+ AGRPO / wd1 FULL]  │  STP, LENS, R³L
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Stage 3 — Serving     │  dInfer → Transformers
              │   threshold-edit loop   │  S_MODE / Q_MODE
              │   REST API              │  /health /generate /jobs/*
              │   [+ RCD / D2F FULL]    │  Order-Token Search, KVzap
              └─────────────────────────┘
```

**SAFE profile** — everything above the `[FULL]` lines. Stable, reproducible, Pascal-compatible.  
**FULL profile** — adds C2DLM supervision, non-Markovian RCD residuals, advanced RL estimators, and hierarchical KV caching. Treated as ablation layers, activated one at a time.

---

## Acceleration Roadmap (Stage 3, FULL)

| Technique | What it does |
|---|---|
| **RCD** — Residual Context Diffusion | Recycles hidden states discarded by remasking as residual context for the next step — reduces wasted compute |
| **D2F / Fast-dLLM v2** | Hierarchical KV caching (block + sub-block level) enabling inter-block parallelism; ~1B token fine-tune cost |
| **CARD** | Confidence-adaptive token generation: more tokens per step at high confidence, sequential fallback otherwise |
| **Order-Token Search** | Decoding-time search over generation order and token trajectories — quality gains with no training change |
| **KVzap / KVpress** | Adaptive KV cache pruning, 2–4× compression with minimal quality loss |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Deep learning | PyTorch ≥ 2.4 + CUDA 12.1 |
| Model loading | HuggingFace Transformers ≥ 4.51 |
| Inference engine | [dInfer](https://github.com/inclusionAI/dInfer) (guarded) + Transformers fallback |
| API server | FastAPI + Uvicorn |
| Quantisation | bitsandbytes — QLoRA/NF4 (cc ≥ 6.0), 8-bit optimiser |
| Data | HuggingFace Datasets + Dolma v1.6 sample |
| Config | JSON, fully config-driven |
| Packaging | `pyproject.toml` / setuptools |
| CLI | `hildanext` (argparse) |

---

## Repository Layout

```
hildanext/
├── backend/src/hildanext/
│   ├── cli.py              # CLI entrypoint
│   ├── api.py              # FastAPI /health /generate /jobs/*
│   ├── wsd_stage0.py       # WSD conversion schedule
│   ├── diffusion.py        # M2T + T2T forward / corruption
│   ├── masks.py            # doc-level attention mask builder
│   ├── training.py         # SFT trainer loop
│   ├── inference.py        # threshold-edit decode (S/Q mode)
│   ├── ar.py               # AR baseline wrapper
│   ├── datasets.py         # Dolma + TinyStories loader
│   ├── tokenization.py     # packing with doc_ids
│   └── recipe.py           # end-to-end run orchestrator
├── test/                   # 25+ unit + smoke tests
├── docs/                   # architecture references
├── runs/configs/           # JSON run configs
├── DESIGN.md               # SAFE design decisions log
└── VERSIONS.md             # pinned commits and deps
```

---

## Quests / Roadmap

### Stage 0 — CPT / WSD Conversion ✅
- [x] WSD schedule (warmup / stable / decay)
- [x] M2T + T2T training objective
- [x] Doc-level attention masking from `doc_ids`
- [x] Dolma v1.6 + TinyStories data pipeline
- [x] Tokenisation and packing with doc boundary tracking

### Stage 1 — SFT ✅
- [x] Response-focused M2T+T2T mixture loss
- [x] Multi-turn forward (two T2T noising passes per step)
- [x] SFT smoke test

### Stage 2 — Inference & Serving ✅
- [x] Threshold-edit decode loop (Γt + Δt)
- [x] `S_MODE` / `Q_MODE` presets
- [x] dInfer adapter + Transformers fallback
- [x] FastAPI REST server
- [x] CLI `hildanext generate`

### Stage 3 — RL Reasoning 🔬

This is the most research-heavy stage. Each component needs its own ablation run before stacking.

**3a — Reward infrastructure**
- [ ] Verifiable reward runner: GSM8K exact-match scorer, HumanEval pass@k executor
- [ ] Rollout sampler: generate N completions per prompt under current dLLM policy (S_MODE / Q_MODE)
- [ ] Rollout budget cap per step (critical on 8 GB — no uncapped MC rollouts)
- [ ] Reward normalisation and advantage estimation utilities
- [ ] Baseline: measure raw SFT model accuracy on GSM8K and HumanEval before any RL

**3b — ESPO (default RL objective)**
- [ ] ELBO-based sequence-level reward proxy implementation
- [ ] Ratio-stabilised KL penalty (clipped, no token-level factorisation)
- [ ] Training loop: rollout → reward → ELBO gradient → parameter update
- [ ] Evaluation: ESPO vs SFT baseline — GSM8K accuracy δ, ELBO convergence curve
- [ ] Ablation: KL weight sweep (β ∈ {0.01, 0.05, 0.1}) — report accuracy vs reward collapse

**3c — wd1 / wd1++ (ratio-free alternative)**
- [ ] Implement weighted log-likelihood objective without ratio estimation
- [ ] wd1++ step-wise variant (per-diffusion-step weighting)
- [ ] Comparison run: wd1 vs ESPO on same GSM8K subset, same compute budget
- [ ] Metric: gradient variance per step, accuracy, training stability (loss curve)

**3d — AGRPO (Monte-Carlo faithful policy gradient)**
- [ ] MC rollout estimator for step-aware dLLM policy gradient
- [ ] Integration with existing threshold-edit decode loop as action space
- [ ] Comparison run: AGRPO vs ESPO vs wd1 — three-way table on GSM8K + HumanEval
- [ ] Ablation: number of MC samples (K ∈ {4, 8, 16}) vs accuracy vs VRAM

**3e — Efficiency upgrades for 8 GB VRAM**
- [ ] STP (spatio-temporal pruning): skip redundant denoising steps during rollout
  - Baseline: rollout step count and wall time without STP
  - After: same accuracy target, measure step reduction %
- [ ] LENS: filter instruction-interfering tokens from rollout prompts
  - Baseline: rollout success rate on GSM8K prompts without filtering
  - After: success rate δ and variance δ across prompt phrasings
- [ ] R³L reflect-then-retry (bounded to max 2 retries per step to avoid forward-pass explosion)
  - Comparison: R³L vs AGRPO on multi-step reasoning tasks (MATH subset)

**3f — Full RL evaluation suite**
- [ ] GSM8K (math word problems, exact match)
- [ ] MATH-500 subset (harder symbolic reasoning, directional signal)
- [ ] HumanEval (code, pass@1 and pass@10)
- [ ] TinyStories perplexity regression check (make sure RL doesn't break fluency)
- [ ] Final comparison table: SFT → ESPO → best-RL across all benchmarks

### Stage 4 — FULL Acceleration 🔬

All components here must be benchmarked against the Stage 2 threshold-edit decode baseline before being considered merged.

**4a — Residual Context Diffusion (RCD)**
- [ ] Implement residual carry-over state between denoise steps (hidden representations from remasked positions)
- [ ] Two-stage training: Stage-0 base → RCD fine-tune (keep same corruption process, same ELBO estimator)
- [ ] ELBO audit: verify estimator alignment before and after RCD activation
- [ ] Baseline comparison: with/without RCD — perplexity, mask ratio per step, total denoise steps to convergence
- [ ] Ablation: residual injection weight α ∈ {0.1, 0.3, 0.5}
- [ ] Cost check: measure added VRAM and wall-time per step vs quality gain

**4b — D2F / Fast-dLLM v2 hierarchical KV caching**
- [ ] Block-level KV cache implementation (cache per diffusion block, invalidate on edit)
- [ ] Sub-block cache layer (fine-grained reuse within a block across denoising steps)
- [ ] Baseline: tokens/sec and VRAM peak without caching (current threshold-edit loop)
- [ ] After: tokens/sec δ, VRAM δ, quality delta on HumanEval and TinyStories
- [ ] Integration test: verify cache invalidation correctness under T2T edits (no stale KV)
- [ ] Target: ~1B token adaptation budget — track token count in training logs

**4c — CARD confidence-adaptive generation**
- [ ] Variable tokens-per-step decoding: generate more tokens when max confidence > threshold, fall back to sequential when uncertain
- [ ] Confidence estimator: calibrated softmax head or running mean of top-1 probability
- [ ] Baseline: fixed tokens-per-step at τ_mask=0.5 (current Q_MODE)
- [ ] After: average steps to complete a 256-token response, quality on GSM8K
- [ ] Ablation: confidence gate threshold sweep vs speed/quality Pareto frontier

**4d — Order-Token Search (decoding plugin)**
- [ ] Implement search over generation order trajectories in addition to token choices
- [ ] Beam budget parameter (B ∈ {1, 2, 4}) — B=1 must equal current greedy baseline exactly
- [ ] Baseline: greedy threshold-edit (current S_MODE) on HumanEval pass@1 and GSM8K
- [ ] After: pass@1 δ and pass@10 δ at B=2 and B=4
- [ ] Cost/quality table: inference FLOPs and wall-time vs accuracy for each B
- [ ] Note: this is a zero-training-change plugin — comparison must hold training constant

**4e — KVzap / KVpress cache compression**
- [ ] KVzap adaptive pruning: drop least-attended KV entries with explicit target compression ratio
- [ ] Target ratios: 2× and 4× — measure quality degradation at each
- [ ] Baseline: full KV cache, no pruning — tokens/sec, VRAM, quality
- [ ] After: tokens/sec δ, VRAM δ, perplexity δ at 2× and 4× compression
- [ ] Integration: combine with D2F block cache and verify no double-pruning bugs

**4f — C2DLM structural supervision (FULL training)**
- [ ] Build concept-level causal graph from training data (entity/clause extraction)
- [ ] Supervised attention mask injection on T2T correction passes
- [ ] V-aware re-attention weighting implementation
- [ ] Baseline: SFT without C2DLM on GSM8K chain-of-thought quality (step correctness rate)
- [ ] After: step correctness rate δ and TinyStories coherence score δ
- [ ] Ablation: C2DLM applied only on T2T passes vs applied always (expected: T2T-only is more stable)

### Hardware / Tooling
- [x] Pascal sm_61 compatible — no FlashAttention, no vLLM required
- [x] CPU-only demo mode
- [x] QLoRA / bitsandbytes optional path
- [ ] ONNX export
- [ ] Automated benchmark runner: one command to reproduce all comparison tables

---

## Quick Start

```bash
git clone https://github.com/ArtyomITA/hildanext.git
cd hildanext
python -m venv .venv && .venv\Scripts\activate
pip install -e hildanext/backend

# smoke test (CPU, no model weights needed)
python hildanext/test/run_tests.py

# start API server
hildanext serve --config hildanext/runs/configs/default.json

# generate
hildanext generate --prompt "Once upon a time" --mode S_MODE
```

---

## Hardware Target

Built and tested on a **GTX 1080 (Pascal, sm_61, 8 GB VRAM)** with CUDA 12.1 / PyTorch 2.4.  
CPU fallback is fully functional. No custom CUDA extensions — forward-compatible through Pascal's remaining driver lifetime.

> CUDA 13.0 drops offline compilation for Maxwell/Pascal/Volta. This backend avoids all custom CUDA kernels intentionally.

---

## Vendored Dependencies *(excluded from this repo)*

| Path | Repo | Pinned commit |
|---|---|---|
| `hildanext/vendor/llada` | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | `570f290` |
| `hildanext/vendor/dinfer` | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | `1ffeb96` |
| `LLaDA/` | ML-GSAI/LLaDA | upstream |
| `Qwen3-0.6B/` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | — |

---

## License

Original code under `hildanext/` — MIT.  
Vendored repos retain their upstream licenses.

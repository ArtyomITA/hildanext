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

Each component is ablated independently before stacking. Benchmarks run once at the end of each sub-phase against the SFT baseline, not after every individual item.

**3a — Reward infrastructure**
- [ ] Verifiable reward runner: GSM8K exact-match scorer, HumanEval pass@k executor
- [ ] Rollout sampler under current dLLM policy (S_MODE / Q_MODE), with hard budget cap per step (8 GB constraint)
- [ ] Reward normalisation and advantage estimation utilities
- [ ] Record SFT-only baseline numbers (GSM8K, HumanEval) before touching any RL objective

**3b — RL objective comparison (ESPO / wd1 / AGRPO)**  
Three objectives, one controlled comparison run on the same GSM8K subset and compute budget:
- [ ] **ESPO**: ELBO sequence-level proxy, ratio-stabilised KL (β sweep: 0.01 / 0.05 / 0.1)
- [ ] **wd1 / wd1++**: ratio-free weighted log-likelihood, step-wise variant
- [ ] **AGRPO**: MC rollout estimator step-aware policy gradient (K samples: 4 / 8 / 16)
- [ ] Output: three-way comparison table — accuracy δ, gradient variance, VRAM peak, training stability

**3c — Efficiency plugins for 8 GB rollouts**
- [ ] **STP**: spatio-temporal pruning of redundant denoising steps — target: same accuracy, fewer steps
- [ ] **LENS**: filter instruction-interfering tokens before rollout — target: higher success rate, lower variance across prompt phrasings
- [ ] **R³L**: reflect-then-retry credit assignment, max 2 retries per step to cap forward-pass cost
- [ ] Applied on top of the best objective from 3b; one combined comparison vs 3b-winner baseline

**3d — Full evaluation**
- [ ] GSM8K (exact match), MATH-500 subset (directional), HumanEval (pass@1 + pass@10)
- [ ] TinyStories perplexity regression (RL must not break fluency)
- [ ] Final table: SFT → best-RL-objective → best-RL+efficiency-plugins

### Stage 4 — FULL Acceleration 🔬

All components are benchmarked against the Stage 2 threshold-edit decode baseline (no caching, no search, no residuals). Metrics shared across all items: tokens/sec, VRAM peak, perplexity δ, HumanEval pass@1 δ.

**4a — Residual Context Diffusion (RCD)**
- [ ] Residual carry-over state between denoise steps (hidden reps from remasked positions)
- [ ] Two-stage fine-tune on top of Stage-0 base; ELBO audit before/after to verify estimator alignment
- [ ] Ablation: injection weight α ∈ {0.1, 0.3, 0.5} — pick best before moving on

**4b — D2F / Fast-dLLM v2 hierarchical KV caching**
- [ ] Block-level KV cache (invalidate on T2T edit) + sub-block reuse within a block
- [ ] Integration test: no stale KV under edit — must pass before measuring speed
- [ ] Target: ~1B token adaptation budget tracked in training logs

**4c — CARD confidence-adaptive generation**
- [ ] Variable tokens-per-step: more tokens when top-1 confidence > gate, sequential fallback otherwise
- [ ] Confidence gate sweep → speed/quality Pareto curve; compare vs fixed Q_MODE τ_mask=0.5

**4d — Order-Token Search (decoding plugin, zero training change)**
- [ ] Search over generation order trajectories; beam B ∈ {1, 2, 4} — B=1 must reproduce greedy baseline exactly
- [ ] Cost/quality table: FLOPs and wall-time per B value

**4e — KVzap / KVpress cache compression**
- [ ] Adaptive KV pruning at 2× and 4× target ratios
- [ ] Integration check: combine with D2F block cache, verify no double-pruning

**4f — C2DLM structural supervision (FULL training)**
- [ ] Concept-level causal graph from training data; supervised attention mask on T2T passes only
- [ ] V-aware re-attention weighting
- [ ] Ablation: T2T-only vs always-on (hypothesis: T2T-only is more stable on small models)

**4g — Full acceleration evaluation**
- [ ] Comparison table: Stage-2 baseline → 4b → 4b+4e → 4b+4c+4e (speed stack)
- [ ] Quality table: Stage-1 SFT → 4a → 4a+4f (quality stack) on GSM8K and TinyStories
- [ ] Best decoding: greedy vs Order-Token Search B=2, quality-only track (no speed penalty expected)

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

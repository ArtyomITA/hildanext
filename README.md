# HILDA

> **Pushing discrete diffusion language models past what autoregressive transformers can do — one ablation at a time.**

HILDA is a research architecture for advancing the state of the art in discrete diffusion language models. The goal is not to fine-tune existing models: it is to explore and validate a full stack of architectural decisions — from how an AR model is converted into a dLLM, to how it reasons, to how structural attention supervision and inference acceleration interact — in order to build architectures that are **potentially competitive with or superior to AR baselines of the same size**.

Everything is designed to be modular, ablatable, and reproducible on a single consumer GPU.

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

| Preset               | Behaviour                                                 |
| -------------------- | --------------------------------------------------------- |
| `S_MODE` (quality) | low τ_mask, more correction passes — best output        |
| `Q_MODE` (speed)   | high τ_mask, conservative drafting — fastest throughput |

Training objective is a mixture of M2T and T2T losses with doc-level attention masking built from `doc_ids`, preventing cross-document attention leakage.

### 2 — WSD Conversion Schedule

HILDA converts an existing AR model rather than training from scratch via **Warmup-Stable-Decay**:

```
Warmup  → block size grows 1 → N  (AR treated as BDLM, block size = 1)
Stable  → full-sequence MDLM regime (stabilise ELBO and diffusion dynamics)
Decay   → shrink block, consolidate editable representation
```

CPT objective: M2T loss on masked positions + T2T loss on noised observed positions. No training from scratch, no architecture surgery.

### 3 — Reasoning Alignment via RL

After SFT, HILDA targets **verifiable reasoning tasks** (math, code) through RL objectives specifically designed for the dLLM setting, where token-level PPO is ill-posed due to the absence of a natural likelihood factorisation.

The default objective is **ESPO**: an ELBO sequence-level proxy with ratio-stabilised KL. Rather than committing to one estimator, HILDA treats the RL objective itself as an ablation axis — comparing ESPO, wd1, and AGRPO under the same compute budget to find what actually moves reasoning quality on small models.

| Method | Role |
|---|---|
| **ESPO** | Principled ELBO-level baseline; ratio-stabilised KL |
| **wd1 / wd1++** | Ratio-free alternative; lower variance, step-wise variant |
| **AGRPO** | MC-faithful policy gradient designed for dLLM step structure |
| **STP** | Spatio-temporal pruning — fewer denoising steps per rollout |
| **LENS** | Filters instruction-interfering tokens before rollout |
| **R³L** | Reflect-then-retry credit assignment for multi-step reasoning |

### 4 — Inference, Attention & Structural Supervision

The fourth pillar covers everything that happens **after the model is trained**: how attention is structured during generation, how computation is reused across denoising steps, and how decoding can be made faster or smarter without retraining.

**Structural attention (C2DLM)** — applied preferentially on T2T correction passes:
- Concept-level causal graph constrains which heads can attend to which positions
- Supervised attention mask enforces causal consistency during reasoning chains
- V-aware re-attention re-weights value vectors by concept role

**Inference acceleration stack:**
- **RCD** — recycles hidden states discarded by remasking as residual context for the next step
- **D2F / Fast-dLLM v2** — hierarchical KV caching (block + sub-block), ~1B token adaptation cost
- **CARD** — confidence-adaptive token generation: more tokens per step when top-1 confidence is high
- **Order-Token Search** — decoding-time search over generation order and token trajectories; zero training change
- **KVzap / KVpress** — adaptive KV cache pruning, 2–4× compression

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

**SAFE profile** — everything above the `[FULL]` lines. Stable, reproducible, no custom CUDA extensions.  
**FULL profile** — adds advanced RL estimators (AGRPO, wd1, STP, LENS, R³L), structural attention supervision (C2DLM), and inference acceleration (RCD, D2F, CARD, Order-Token Search, KVzap). Each layer is an independent ablation.

---

## Tech Stack

| Layer            | Tool                                                                           |
| ---------------- | ------------------------------------------------------------------------------ |
| Language         | Python 3.11                                                                    |
| Deep learning    | PyTorch ≥ 2.4 + CUDA 12.1                                                     |
| Model loading    | HuggingFace Transformers ≥ 4.51                                               |
| Inference engine | [dInfer](https://github.com/inclusionAI/dInfer) (guarded) + Transformers fallback |
| API server       | FastAPI + Uvicorn                                                              |
| Quantisation     | bitsandbytes — QLoRA/NF4 (cc ≥ 6.0), 8-bit optimiser                         |
| Data             | HuggingFace Datasets + Dolma v1.6 sample                                       |
| Config           | JSON, fully config-driven                                                      |
| Packaging        | `pyproject.toml` / setuptools                                                |
| CLI              | `hildanext` (argparse)                                                       |

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

- [X] WSD schedule (warmup / stable / decay)
- [X] M2T + T2T training objective
- [X] Doc-level attention masking from `doc_ids`
- [X] Dolma v1.6 + TinyStories data pipeline
- [X] Tokenisation and packing with doc boundary tracking
- [X] Special token registration for `[MASK]` without embedding remap
- [X] ELBO logging per training step

### Stage 1 — SFT ✅

- [X] Response-focused M2T+T2T mixture loss
- [X] Multi-turn forward (two T2T noising passes per step)
- [X] Multi-turn conversation format with turn boundary masking
- [X] Train/eval split with held-out SFT shard
- [X] SFT smoke test (dummy batch, loss finite check)

### Stage 2 — Inference & Serving ✅

- [X] Threshold-edit decode loop (Γt + Δt)
- [X] `S_MODE` / `Q_MODE` presets
- [X] dInfer adapter + Transformers fallback
- [X] FastAPI REST server with `/health`, `/generate`, `/jobs/*`
- [X] CLI `hildanext generate`
- [X] Per-step decode tracing (mask ratio, edit count, throughput estimate)
- [X] Inference smoke test against dummy model (no weights required)

### Stage 3 — RL Reasoning 🔬

Each component is ablated independently before stacking. Benchmarks run once at the end of each sub-phase against the SFT baseline, not after every individual item.

**3a — Reward infrastructure**

- [ ] Verifiable reward runner: GSM8K exact-match scorer, HumanEval pass@k executor
- [ ] Rollout sampler under current dLLM policy (S_MODE / Q_MODE), with hard budget cap per step (8 GB constraint)
- [ ] Reward normalisation and advantage estimation utilities
- [ ] Record SFT-only baseline numbers (GSM8K, HumanEval) before touching any RL objective

**3b — RL objective comparison (ESPO / wd1 / AGRPO)**Three objectives, one controlled comparison run on the same GSM8K subset and compute budget:

- [ ] **ESPO**: ELBO sequence-level proxy, ratio-stabilised KL (β sweep: 0.01 / 0.05 / 0.1)
- [ ] **wd1 / wd1++**: ratio-free weighted log-likelihood, step-wise variant
- [ ] **AGRPO**: MC rollout estimator step-aware policy gradient (K samples: 4 / 8 / 16)
- [ ] Output: three-way comparison table — accuracy δ, gradient variance, VRAM peak, training stability

**3c — Efficiency plugins for small-model rollouts**

- [ ] **STP**: spatio-temporal pruning of redundant denoising steps — same accuracy, fewer steps
- [ ] **LENS**: filter instruction-interfering tokens before rollout — higher success rate, lower variance across prompt phrasings
- [ ] **R³L**: reflect-then-retry credit assignment, max 2 retries per step to keep forward-pass count tractable on small models
- [ ] Applied on top of the best objective from 3b; one combined comparison vs 3b-winner baseline

**3d — Full evaluation**

- [ ] GSM8K (exact match), MATH-500 subset (directional), HumanEval (pass@1 + pass@10)
- [ ] TinyStories perplexity regression (RL must not break fluency)
- [ ] Final table: SFT → best-RL-objective → best-RL+efficiency-plugins

### Stage 4 — FULL Acceleration 🔬

All components benchmarked against the Stage 2 threshold-edit decode baseline. Shared metrics: tokens/sec, VRAM peak, perplexity δ, HumanEval pass@1 δ.

**4a — Quality stack: RCD + C2DLM**

- [ ] **RCD**: residual carry-over of remasked hidden states between denoise steps; ELBO audit before/after; ablation on injection weight α ∈ {0.1, 0.3, 0.5}
- [ ] **C2DLM**: concept-level causal graph, supervised attention mask applied on T2T passes only; V-aware re-attention weighting; ablation T2T-only vs always-on
- [ ] Evaluation: quality stack table — SFT → +RCD → +RCD+C2DLM on GSM8K and TinyStories

**4b — Speed stack: KV caching + compression + adaptive decoding**

- [ ] **D2F / Fast-dLLM v2**: block-level KV cache (invalidate on T2T edit) + sub-block reuse; integration test for stale-KV correctness under edits
- [ ] **KVzap**: adaptive KV pruning at 2× and 4× ratios; verify no double-pruning with D2F cache
- [ ] **CARD**: variable tokens-per-step when top-1 confidence > gate; Pareto curve vs fixed Q_MODE τ_mask=0.5
- [ ] Evaluation: speed stack table — baseline → +D2F → +D2F+KVzap → +D2F+KVzap+CARD

**4c — Decoding search: Order-Token Search**

- [ ] Search over generation order trajectories; beam B ∈ {1, 2, 4} — B=1 must reproduce greedy baseline exactly
- [ ] Zero training change; compare vs S_MODE greedy on HumanEval pass@1 and GSM8K; cost/quality table per B

### Hardware / Tooling

- [X] Pascal sm_61 compatible — no FlashAttention, no vLLM required
- [X] CPU-only demo mode
- [X] QLoRA / bitsandbytes optional path
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

Built and tested on a **GTX 1080 (Pascal, sm_61, 8 GB VRAM)** with CUDA 12.1 / PyTorch 2.4.CPU fallback is fully functional. No custom CUDA extensions — forward-compatible through Pascal's remaining driver lifetime.

> CUDA 13.0 drops offline compilation for Maxwell/Pascal/Volta. This backend avoids all custom CUDA kernels intentionally.

---

## Vendored Dependencies *(excluded from this repo)*

| Path                        | Repo                                                     | Pinned commit |
| --------------------------- | -------------------------------------------------------- | ------------- |
| `hildanext/vendor/llada`  | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)           | `570f290`   |
| `hildanext/vendor/dinfer` | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | `1ffeb96`   |
| `LLaDA/`                  | ML-GSAI/LLaDA                                            | upstream      |
| `Qwen3-0.6B/`             | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)   | —            |

---

## License

Original code under `hildanext/` — MIT.
Vendored repos retain their upstream licenses.

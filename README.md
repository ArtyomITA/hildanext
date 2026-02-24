# HildaNext

**Converting autoregressive LLMs to discrete diffusion language models — end-to-end research backend.**

HildaNext is a self-contained Python backend that takes a vanilla AR transformer (starting from **Qwen3-0.6B**) and converts it into a **discrete diffusion language model (dLLM)** following the LLaDA 2.1 recipe: WSD training schedule, block diffusion, document-level attention masking, M2T+T2T training objective, and threshold-based decoding with speed/quality presets.

Everything runs on **consumer GPU (GTX 1080, sm_61, 8 GB VRAM)** — no FlashAttention, no vLLM mandatory; pure PyTorch + Transformers.

---

## Architecture at a glance

```
AR model (Qwen3-0.6B)
        │
        ▼  Stage 0 — CPT / WSD Conversion
┌───────────────────────────────────────────────┐
│  Warmup   → block size 1 → N  (AR → BDLM)    │
│  Stable   → full-sequence MDLM                │
│  Decay    → compact block, stabilise ELBO     │
│  Objective: M2T on masked + T2T on noised     │
│  Doc-level attention mask from doc_ids        │
└───────────────────────────────────────────────┘
        │
        ▼  Stage 1 — SFT
┌───────────────────────────────────────────────┐
│  Mixture M2T + T2T, response-focused masking  │
│  Multi-turn forward (two T2T passes/step)     │
└───────────────────────────────────────────────┘
        │
        ▼  Stage 2 — RL (ESPO / AGRPO)  [planned]
        │
        ▼  Stage 3 — Serving
┌───────────────────────────────────────────────┐
│  dInfer engine  →  fallback: Transformers     │
│  Threshold-edit decode loop:                  │
│    Γt  unmasking  (MASK→token)  τ_mask        │
│    Δt  editing    (token→token) τ_edit        │
│  S_MODE (quality) / Q_MODE (speed) presets    │
│  REST API: /health  /generate  /jobs/*        │
└───────────────────────────────────────────────┘
```

Design is **HILDA SAFE**: no speculative decoding, no non-Markovian forward process, no invasive tokenizer changes. Stable and reproducible on small compute.

---

## Key research foundations

| Component | Paper / source |
|---|---|
| Backbone | [LLaDA 2.1](https://arxiv.org/abs/2502.xxxxx) — M2T + T2T editing, threshold decode |
| Block diffusion | [LLaDA 2.0 / BD3-LMs](https://arxiv.org/abs/2502.xxxxx) — block-level masking |
| WSD schedule | Warmup-Stable-Decay from LLaDA 2.0 § 3 |
| AR → dLLM conversion | DiffuGPT / DiffuLLaMA conversion approach |
| RL objective | ESPO (ELBO sequence-level, ratio-stabilised KL) |
| Inference engine | [dInfer](https://github.com/inclusionAI/dInfer) (guarded; Transformers fallback) |
| Dataset | [Dolma v1.6](https://huggingface.co/datasets/allenai/dolma) sample + TinyStories |

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Deep learning | PyTorch ≥ 2.4 + CUDA 12.1 |
| Model loading | Transformers ≥ 4.51 (HuggingFace) |
| API server | FastAPI + Uvicorn |
| Config | JSON (config-driven, no hard-coded paths) |
| Packaging | `pyproject.toml` / setuptools |
| Quantisation | bitsandbytes (optional, QLoRA-ready) |
| Data | HuggingFace Datasets (optional dep) |
| CLI | `hildanext` entrypoint (argparse) |

---

## Repository layout

```
hildanext/
├── backend/
│   ├── pyproject.toml           # package manifest
│   └── src/hildanext/
│       ├── cli.py               # CLI entrypoint
│       ├── api/                 # FastAPI routes (/health /generate /jobs/*)
│       ├── conversion/          # WSD schedule, CPT forward pass
│       ├── sft/                 # SFT trainer, M2T+T2T objective
│       ├── inference/           # threshold-edit decode loop, S/Q presets
│       ├── data/                # Dolma loader, tokenizer, packing, doc_ids
│       └── engine/              # dInfer adapter + Transformers fallback
├── data/
│   ├── processed/               # train.jsonl, eval.jsonl, sft_*.jsonl
│   └── tokenized/               # packed, doc_id-indexed shards
├── models/
│   ├── qwen3-0.6b/              # local HF weights (symlinked)
│   └── exports/                 # converted / quantised checkpoints
├── runs/
│   ├── configs/                 # JSON run configs
│   ├── logs/                    # training logs
│   └── reports/                 # eval outputs
├── vendor/
│   ├── llada/                   # ML-GSAI/LLaDA @ pinned commit
│   └── dinfer/                  # inclusionAI/dInfer @ pinned commit
├── docs/                        # architecture references
├── test/                        # smoke + unit tests
├── DESIGN.md                    # SAFE design decisions
└── VERSIONS.md                  # pinned deps and commit hashes
```

> `LLaDA/`, `Qwen3-0.6B/`, and `vendor/*` are third-party repos and are excluded from this repo via `.gitignore`.

---

## Quests / Roadmap

### Stage 0 — CPT conversion ✅
- [x] WSD schedule (warmup / stable / decay)
- [x] M2T + T2T training objective
- [x] Doc-level attention masking from `doc_ids`
- [x] Dolma v1.6 + TinyStories data pipeline
- [x] Tokenization and packing with doc boundary tracking

### Stage 1 — SFT ✅
- [x] Response-focused M2T+T2T mixture loss
- [x] Multi-turn forward (two T2T noising passes per step)
- [x] SFT smoke test (dummy batch, 1 step)

### Stage 2 — Inference & serving ✅
- [x] Threshold-edit decode loop (Γt + Δt alternation)
- [x] `S_MODE` (quality) and `Q_MODE` (speed) presets
- [x] dInfer adapter with guarded import + Transformers fallback
- [x] FastAPI server (`/health`, `/generate`, `/jobs/*`)
- [x] CLI `hildanext generate`

### Stage 3 — RL reasoning 🔬
- [ ] ESPO baseline implementation (ELBO sequence-level reward)
- [ ] AGRPO adapter for step-aware dLLM policy gradient
- [ ] Reward model integration (math / coding benchmarks)

### Stage 4 — Acceleration 🔬
- [ ] Residual Context Diffusion (RCD) remasking carry-over
- [ ] D2F / Fast-dLLM v2 hierarchical KV caching
- [ ] Order-Token Search decoding plugin
- [ ] KVzap compression (KVpress ecosystem)

### Hardware & tooling
- [x] Pascal/sm_61 compatible (GTX 1080, 8 GB) — no FlashAttention required
- [x] CPU-only demo mode (no GPU needed for smoke tests)
- [x] QLoRA / bitsandbytes optional quantisation path
- [ ] ONNX export for edge inference

---

## Quick start

```bash
# 1. clone
git clone https://github.com/<you>/hildanext.git
cd hildanext

# 2. create env (Python 3.11)
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e hildanext/backend

# 3. smoke test (no GPU required)
python hildanext/test/run_tests.py

# 4. run API server
hildanext serve --config hildanext/runs/configs/default.json

# 5. generate
hildanext generate --prompt "Once upon a time" --mode S_MODE
```

---

## Hardware target

Developed and tested on:

- **GPU:** NVIDIA GTX 1080 (Pascal, sm_61, 8 GB VRAM)
- **CUDA:** 12.1 / PyTorch 2.4
- **CPU fallback:** fully functional for demo and testing

> Pascal sm_61 support will be dropped by CUDA 13.0. This backend avoids custom CUDA extensions to maximise forward compatibility.

---

## Vendored dependencies (read-only, not pushed here)

| Folder | Repo | Commit |
|---|---|---|
| `hildanext/vendor/llada` | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | `570f290` |
| `hildanext/vendor/dinfer` | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | `1ffeb96` |
| `LLaDA/` *(excluded)* | ML-GSAI/LLaDA | upstream main |
| `Qwen3-0.6B/` *(excluded)* | [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B) | — |

---

## License

This codebase (everything under `hildanext/`) is original work and released under MIT.  
Vendored repos retain their respective upstream licenses.

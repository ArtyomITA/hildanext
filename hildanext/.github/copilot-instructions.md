# HildaNext — Copilot Instructions

> Diffusion Language Model (dLLM) research platform: AR→dLLM conversion via WSD schedule, SFT, and **inference serving** on Qwen3-0.6B backbone.
> Target hardware: GTX 1080 / Pascal sm_61, 8 GB VRAM.

---

## Quick-start commands

```powershell
# Avvia backend + frontend contemporaneamente (PowerShell)
powershell -ExecutionPolicy Bypass -File .\dev.ps1

# Solo backend (conda env mdm, porta 8080)
conda activate mdm
python start_server.py

# Solo frontend (porta 5173, proxy /api → :8080)
cd frontend && npm run dev

# Test backend
cd backend && pytest -q

# Test frontend
cd frontend && npm test

# Build frontend
cd frontend && npm run build
```

---

## Architecture overview

```
hildanext/
├── start_server.py          # Launcher → uvicorn, lazy model loading
├── backend/
│   └── src/hildanext/
│       ├── api.py            # FastAPI app factory (create_app), tutti gli endpoint
│       ├── inference.py      # Engine hierarchy: dInfer → Transformers → TinyCausalLM
│       ├── diffusion.py      # WSD schedule, M2T/T2T losses, force_noncausal_attention
│       ├── formulas.py       # LLaDA/LLaDA2.0/2.1 formula checks (llada21_apply)
│       ├── ar.py             # AR baseline generazione side-by-side
│       ├── config.py         # Dataclass config (AppConfig), tutto config-driven JSON
│       ├── training.py       # WSD training loop
│       ├── masks.py          # Composite/block-diagonal attention masks
│       ├── tokenization.py   # Tokenizer loading + mask_token injection
│       ├── trace.py          # Telemetry/fallback recording
│       ├── quant.py          # Quantization helpers
│       └── cli.py            # CLI entrypoint (hildanext command)
│   └── tests/
├── frontend/                 # React 19 + Vite + TypeScript
│   └── src/
│       ├── domain/           # types.ts, backendAdapter.ts, formatters.ts
│       ├── features/
│       │   ├── chat/         # Chat orchestrator, prompt composer, catalog
│       │   ├── compare/      # InferenceSplitPane, PromptLab (AR vs dLLM)
│       │   ├── diffusion-viz/# DiffusionStepTimeline, TokenMaskCanvas
│       │   ├── run/          # useArGenerate, useDllmGenerate hooks
│       │   └── stage0/       # Stage0 benchmark UI
│       ├── store/            # Zustand (dataStore, chatStore, uiStore)
│       └── routes/
│           ├── inference/    # InferencePage — compare AR vs diffusion
│           ├── chat/         # ChatPage — dual-lane chat
│           ├── wsd/          # WSD training dashboard
│           └── benchmark/    # Stage0 benchmark runner
├── vendor/
│   ├── dinfer/               # dInfer (inclusionAI) — diffusion serving runtime
│   └── llada/                # LLaDA reference repo (ML-GSAI)
├── models/qwen3-0.6b/        # Pesi modello locale (junction → Qwen3-0.6B)
└── docs/                     # Design docs, architecture, inventory
```

---

## Inference pipeline — regole per l'agente

### Backend (PRIORITÀ ALTA)

1. **Engine hierarchy** (`inference.py`):
   - `build_engine()` tenta in ordine: **dInfer** → **TransformersEngine** → **TinyCausalLM** (dummy).
   - `_LazyEngine` wrappa il tutto: i pesi si caricano solo alla prima `/generate`.
   - Mai rompere la catena di fallback; ogni fallback è registrato via `trace`.

2. **Decode loop dLLM** (threshold-edit, stile LLaDA2.1):
   ```
   Γt (unmasking):  MASK → token quando conf > τ_mask
   Δt (editing):    token → token quando conf > τ_edit
   ```
   - **effort knob**: `instant|low|medium|high|adaptive` — scala `max_steps` e `tau`.
   - **Degeneration guards**: `zero_gamma_streak`, plateau detection, cycle detection, EOS guard.
   - **Remask** (`apply_remask`): solo tra step intermedi, mai sull'ultimo.
   - **Stop conditions**: EOS cut, plateau patience, cycle hash, `mask_ratio==0`.

3. **AR baseline** (`ar.py`):
   - `generate_ar_from_bundle()` — greedy o sampling, KV-cache, penalties.
   - Endpoint separato: `POST /generate/ar`.

4. **Attenzione bidirezionale** (`diffusion.py → force_noncausal_attention`):
   - Context manager che disabilita `is_causal` + monkey-patches `create_causal_mask` in 3 moduli (masking_utils, qwen3, qwen2).
   - Su transformers 4.57 la maschera 4D composita da sola neutralizza la causalità; il CM è safety net.
   - Il patch usa identity check (`result is user_mask`) per preservare maschere composite 4D → se transformers copia il tensor internamente, il check si rompe silenziosamente.
   - **Non toccare** a meno che non cambi la versione di transformers.
   - Dettagli: [docs/BIDIRECTIONAL_ATTENTION_HANDOFF.md](../docs/BIDIRECTIONAL_ATTENTION_HANDOFF.md)

5. **Config** (`config.py`):
   - Tutto è in dataclass (`AppConfig`), serializzato JSON in `runs/configs/`.
   - Mai hard-code valori: aggiungi il campo in config, leggi da `cfg.xxx`.
   - `ExperimentConfig` controlla ablation: `mask_strategy`, `attention_mode`, `time_param`, ecc.

6. **Serializzazione inferenza**:
   - `serialize_inference=True` di default — un solo `/generate` alla volta (lock con timeout).
   - `require_cuda_for_inference=True` — ritorna 503 se CUDA non disponibile.
   - Il lock è necessario anche perché `force_noncausal_attention` patcha attributi globali (non thread-safe).

7. **Decode policies** (5 policy nel decode loop, default `shadow_llada21_cap`):
   - `current_base` (top-k puro), `threshold_cap`, `shadow_llada21`, `shadow_llada21_cap`, `shadow_llada21_delayed_edit`.
   - Controllate da `cfg.runtime.dllm_base_policy`.

8. **Shadow decode** (off di default, `dllm_shadow_enabled=False`):
   - Esegue un secondo `diagnostic_dllm_decode` con policy/config separati per telemetria.
   - Raddoppia il costo inferenza; risultato in `last_stats["shadow"]`.

### API Endpoints (FastAPI, porta 8080)

| Metodo | Path | Scopo |
|--------|------|-------|
| `GET` | `/health` | Health check (non carica pesi) |
| `POST` | `/generate` | Generazione dLLM (threshold-edit) |
| `POST` | `/generate/ar` | Generazione AR baseline |
| `POST` | `/inference/load` | Carica pesi esplicitamente |
| `POST` | `/inference/unload` | Scarica pesi, libera VRAM |
| `GET` | `/inference/logs` | Log ring buffer inferenza |
| `GET` | `/inference/logs/stream` | SSE stream log inferenza real-time |
| `POST` | `/jobs/submit` | Job asincrono generazione |
| `GET` | `/jobs/{job_id}` | Stato job |
| `GET` | `/frontend/wsd` | Dati WSD per dashboard frontend |
| `POST` | `/run/start` | Avvia training run |
| `GET` | `/run/status` | Stato training run |
| `POST` | `/run/stop` | Ferma training run |
| `POST` | `/stage0/validate/*` | Benchmark endpoints (HellaSwag, MMLU-Pro, GSM8K, stability) |

### Frontend (React 19 + TypeScript)

1. **Proxy**: Vite proxy `/api/*` → `http://127.0.0.1:8080` (rewrite rimuove `/api`).
2. **Stato**: Zustand stores (`dataStore`, `chatStore`, `uiStore`) — no Redux.
3. **Chat duale**: `ChatPage` supporta AR + dLLM side-by-side via `orchestrator.ts`.
   - `runChatTurn()` chiama `/generate` (dLLM) e `/generate/ar` in parallelo.
   - `LaneResult` normalizza risposte AR e dLLM con metriche comuni.
4. **Inference viz**: `DiffusionStepTimeline` + `TokenMaskCanvas` mostrano step-by-step del decode.
5. **PromptLab**: permette regolazione interattiva di `temperature`, `topP`, `effort`, `tauMask/tauEdit`.
6. **Backend adapter** (`backendAdapter.ts`): WSD wired, inference **non ancora wired** (fallback a mock).
7. **Tipi** (`domain/types.ts`): `InferenceRun`, `TokenFrame`, `InferenceTraceStep`, `ChatRunConfig`, `LaneResult`.
8. **Styling**: CSS Modules (`.module.css`), no Tailwind. Grafica con **uPlot** (no seaborn, no d3).
9. **Test**: Vitest + Testing Library (unit), Playwright (e2e).

---

## Convenzioni codice

### Python (backend)
- Python `>=3.10` (target package `>=3.11`); conda env `mdm` (torch 2.4 + CUDA 12.1).
- Type hints ovunque, docstring breve in prima riga del modulo.
- `pathlib.Path` per tutti i path, nessun hard-code assoluto.
- Logging via `trace.py` (record_fallback, record_env_issue) — no `print()` in produzione.
- PEP8, ma il codebase usa **compact formatting** (spazi ridotti) — mantieni lo stile esistente.
- Test in `backend/tests/` con pytest; smoke test con `force_dummy_model=True` per CI senza GPU.

### TypeScript (frontend)
- React 19 con hooks funzionali, no class components.
- Zustand per state management, zod per validazione.
- Import path relativi dentro `src/`.
- `vitest` per unit test, `@playwright/test` per e2e.

### Backbone: Qwen3-0.6B-Base
- **bos_token**: None — non esiste nel tokenizer, non iniettare.
- **eos_token**: `<|im_end|>` (id 151645) — separatore documento nel packing.
- **pad_token**: `<|im_start|>` (id 151643) — solo filler, mai in loss.
- **mask_token**: `<|mask|>` — aggiunto come special token.
- **shift_mode**: `preserve_left_shift` — decisione fissa, non cambiare.

---

## Pitfalls & guardrails

- **dInfer** hard-importa vLLM/sglang — il backend ha fallback protetto; non assumere che dInfer funzioni.
- **flash_attention** non disponibile su Pascal — `force_math_sdpa()` forza il kernel math.
- **VRAM** 8 GB — `max_vram_pct=0.85`, gradient checkpointing on, `low_cpu_mem_usage=True`.
- **`create_causal_mask` monkey-patch**: cambia tra versioni transformers — verificare se si aggiorna transformers.
- **Non usare `bos_and_shift`** su Qwen3-Base (inietta rumore strutturale nel pretraining diffusion).
- **`inference_lock`**: un solo request inferenza alla volta — non parallelizzare chiamate `/generate`.
- **Left-shift off-by-one**: `gen_logits = logits[:, prompt_len-1 : prompt_len-1+max_new]` — dimenticare il `-1` produce predizioni sbagliate senza errore.
- **`ExperimentConfig.attention_mode` ≠ controllo effettivo**: il decode loop legge solo `cfg.runtime.dllm_base_use_bidirectional`. Settare `attention_mode="bidirectional_always"` senza anche `dllm_base_use_bidirectional=True` non ha effetto.
- **KV-cache sempre disabilitato** (`use_cache=False`): necessario per forward full-sequence dLLM. Non riabilitare.
- **Campi config vestigiali**: `RemaskConfig.percentile_safety/block_size/block_stride`, `InferenceConfig.strict_decode_invariants` — definiti ma non usati nel codice.

### Cross-file dependency graph (inferenza)

```
inference.py
  ├── force_noncausal_attention ← diffusion.py  (_forward_full_sequence)
  ├── apply_remask             ← diffusion.py  (_apply_decode_policy_step)
  ├── llada21_sets             ← formulas.py   (threshold-edit policies)
  └── NON importa masks.py (usato solo in training)

diffusion.py
  ├── batch_doc_attention_mask ← masks.py      (_forward training)
  └── llada2_wsd_block         ← formulas.py   (wsd_block)
```
- **Dolma token ID space** può differire dal tokenizer Qwen — il backend normalizza out-of-range.

---

## Docs esistenti (link, non duplicare)

- [DESIGN.md](../DESIGN.md) — decisioni architetturali SAFE, WSD schedule, effort knob
- [VERSIONS.md](../VERSIONS.md) — dependency pins, vendored repos, model paths
- [docs/BIDIRECTIONAL_ATTENTION_HANDOFF.md](../docs/BIDIRECTIONAL_ATTENTION_HANDOFF.md) — dettagli noncausal patch
- [docs/full_architecture_reference.md](../docs/full_architecture_reference.md) — architettura completa
- [docs/INTERFACCIA_GRAFICA_HILDANEXT.md](../docs/INTERFACCIA_GRAFICA_HILDANEXT.md) — specifiche UI
- [docs/PYTHON_INVENTORY.md](../docs/PYTHON_INVENTORY.md) — inventario moduli Python
- [docs/WSD_INVESTIGATION_REPORT.md](../docs/WSD_INVESTIGATION_REPORT.md) — report investigazione WSD

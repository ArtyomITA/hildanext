# HildaNext SAFE Design
Date: 2026-02-22 (created) — last updated: 2026-02-28

## Scope
- Backend SAFE AR→dLLM conversion, SFT, inference, serving.
- Componenti speculative/non-Markoviani (FULL) esclusi.
- Target hardware: GTX 1080 / Pascal sm_61, 8 GB VRAM.

---

## SAFE Decisions
- Language: Python 3.11+.
- CLI: argparse.
- Config format: JSON. Tutto config-driven via `AppConfig`.
- API server: FastAPI — `/health`, `/generate`, `/jobs/*`.
- Inference engines:
  1. dInfer adapter (quando disponibile, richiede runtime LLaDA-family)
  2. Transformers threshold-edit loop (default funzionante)
  3. TinyCausalLM dummy per smoke/demo senza pesi

---

## Backbone: Qwen3-0.6B-Base
| Proprietà | Valore |
|---|---|
| Parametri | 0.6B (0.44B non-embedding) |
| vocab_size | 151 669 |
| bos_token | **None** — non esiste nel tokenizer |
| eos_token | `<|im_end|>` id=151645 — separatore documento nel packing |
| pad_token | `<|im_start|>` id=151643 — solo filler, mai in loss (attention_mask=0) |
| mask_token | `<|mask|>` aggiunto come special token |
| Licenza | Apache-2.0 |

**Decisione fissa:** `shift_mode=preserve_left_shift`, nessun BOS iniettato nel pretraining diffusion.  
`<|im_start|>` è un delimitatore chat-template Instruct, non un BOS neutro — iniettarlo in raw Dolma introduce rumore strutturale.

---

## Data Pipeline

```
dolma_v1_6_sample (103 .json.gz raw, ~16 GB)
  └─ _split_processed()          → data/processed/train.jsonl (36 GB), eval.jsonl (363 MB)
       └─ tokenize_split()       → data/tokenized/train.jsonl, eval.jsonl
            └─ _build_artifacts()→ dolma_v1_6_sample/tokens/*.npy, doc_index/*.npy
```

### tokenize_split — modifiche 2026-02-28
- **Streaming chunk-based** (CHUNK_ROWS=5000) — nessun OOM su 14 GB RAM.
- **Batch encoding** via `_encode_records_batch()`: usa `tokenizer(list_of_texts)` (backend Rust) → 5-8× più veloce della codifica riga per riga (~6-8k rows/s vs ~1.5k rows/s).
- **Random-length exposure** (TRUNC_PROB=0.01): 1% delle sequenze piene viene troncato a lunghezza random ∈ [seq_len//4, len-1] prima del padding — tecnica pplx-embed.
- **Checkpoint/resume**: ogni 50k righe scrive `train.jsonl.ckpt`; al prossimo avvio rileva il checkpoint e riprende. Eliminato a fine run pulita.

### Resume da run interrotta
```
python scripts/make_resume_ckpt.py
# conta righe, tronca al chunk boundary, scrive .ckpt
# poi rilancia dolma-prep normalmente
```

### prepare_dolma_only — checkpoint resume
- Checkpoint 1: `processed/train.jsonl` + `eval.jsonl` entrambi >1 MB → skip `_split_processed`.
- Checkpoint 2: `tokenized/train.jsonl` + `eval.jsonl` entrambi >1 MB → skip `tokenize_split`.

---

## WSD Conversion Schedule
- Warmup: block size 1 → full sequence.
- Stable: full-sequence MDLM regime.
- Decay: block size → compact block per inferenza efficiente.
- CPT objective: M2T loss su posizioni mascherate + T2T loss su posizioni rumorose.

---

## SFT Objective
- Mixture M2T+T2T con response-focused masking.
- Multi-turn forward minimal: 2 T2T noising pass per step.

---

## Inference Logic (LLaDA2.1 style)
```
Γt  unmasking  MASK→token  quando conf > τ_mask
Δt  editing    token→token quando conf > τ_edit
```
- `S_MODE`: τ_mask basso, più correction passes — qualità massima.
- `Q_MODE`: τ_mask alto, drafting conservativo — throughput massimo.

### P0.2 — Effort knob (2026-02-28)
| effort | max_steps | tau × |
|---|---|---|
| instant | 1 | 2.0 |
| low | 3 | 1.5 |
| medium | config | 1.0 |
| high | 20 | 0.7 |
| adaptive | 128 | stop@mask_ratio=0 |

Parametro `effort` esposto in `GenerateRequest` via API.

---

## Metriche sempre loggiate (P0.3 — 2026-02-28)
| Metrica | Dove |
|---|---|
| `masked_token_acc` | ogni training step (da `compute_m2t_t2t_losses`) |
| `steps_to_converge` | ogni generate + eval periodica |
| `vram_peak_bytes` | ogni generate + training step |
| `tokens_per_sec` | ogni training step + generate |
| `json_valid_rate` | stub None, si popola Stage1 |

Training print: `... loss=X.XXXXXX mta=0.8123 tok_seen=N sec_step=X.XXX eta_sec=N peak_vram=N`

---

## Ablation control (P0.1 — 2026-02-28)
`ExperimentConfig` in `config.py` — campo `experiment` dell'AppConfig:
- `mask_strategy`: `special_mask_token` | `repurpose_rare_token`
- `attention_mode`: `bidirectional_always` | `bidirectional_only_stable`
- `time_param`: `discrete` | `continuous_time`
- `loss_weighting`: `none` | `inv_t`
- `shift_mode`: `preserve_left_shift` | `bos_and_shift`
- `effort`: `instant` | `low` | `medium` | `high` | `adaptive`
- `experiment_id`, `notes`: annotazione run

Template ablation: `runs/configs/experiment_template.yaml`

---

## TODO / Uncertain
- dInfer richiede LLaDA-family runtime; serving Qwen dLLM via dInfer non garantito.
- Dolma shards pretokenizzati: ID space può differire da Qwen tokenizer. SAFE normalizza out-of-range.
- `json_valid_rate` stub — si popola in Stage1 quando il modello genera JSON strutturato.
- `bos_and_shift` implementato in `ExperimentConfig` ma **non usare** su Qwen3-Base (vedi nota backbone).

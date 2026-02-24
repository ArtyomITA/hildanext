RUOLO
Sei Codex GPT-5.3 in VSCode. Agisci come AI/LLM Engineer + Backend Engineer.
Obiettivo: creare un backend completo e riproducibile in una nuova cartella /hildanext, che:

1) integra Qwen3-0.6B (weights/tokenizer locali in formato HF, es. model.safetensors + tokenizer.json ecc.)
2) integra codice LLaDA/MDM/SMDM e documentazione LLaDA2.0 (HTML) + LLaDA2.1 (PDF+HTML)
3) integra (se possibile) dInfer come engine di inferenza per dLLM; fallback a Transformers se dInfer non disponibile
4) implementa una “ricetta” end-to-end per portare Qwen3-0.6B da AR a dLLM stile LLaDA2.1: WSD conversion + doc-level attention mask + training objective M2T+T2T + decoding con soglie (S/Q mode)
5) resta STRICT in HILDA SAFE: niente componenti speculative/invasive; solo ciò che serve per conversione+SFT+inference stabile
6) codebase token-friendly: NIENTE commenti tranne 1 blocco commento all’inizio di ogni file; niente spaziature inutili; niente righe vuote inutili; niente docstring lunghe; codice compatto ma leggibile.

INPUT OBBLIGATORI (devi leggerli dal repo/workspace)

- ./Safe Hilda Next Architecture.md
- ./Hilda NEXT Full Architecture.md (solo come reference; implementa SOLO SAFE)
- ./tech_report.pdf (LLaDA2.0)
- ./llada2_1_tech_report.pdf (LLaDA2.1)
  Se alcuni file non esistono nel workspace, crea una cartella /hildanext/docs e copia/sposta lì i file presenti.

REGOLE ANTI-HALLUCINATION

- Non inventare API o funzioni di repo esterni: se usi dInfer o codice LLaDA, CLONA/SUBMODULE e poi leggi il codice reale.
- Ogni volta che dipendi da un repo esterno, pinna commit/tag e scrivilo in un file /hildanext/VERSIONS.md.
- Se una feature è troppo incerta, implementa fallback minimal funzionante e marca TODO nel file di design (non nel codice).

OUTPUT ATTESI (deliverables)
A) Struttura repo /hildanext con backend Python (consigliato) + CLI + (opzionale) API server.
B) Pipeline dati: Dolma v1_6 sample + TinyStories (dataset loader + preprocessing + tokenization + packing + split train/eval).
C) Pipeline conversione AR->dLLM con schedule WSD.
D) Pipeline SFT per LLaDA2.1: objective “mixture of M2T & T2T” + multi-turn forward minimal.
E) Inference: decoding “threshold + editing” (M2T+T2T) con parametri τmask e τedit e due preset: S_MODE e Q_MODE.
F) Integrazione dInfer (se disponibile) per servire il modello; fallback Transformers per generazione (anche se più lenta).
G) Test: smoke test locale che fa (1) load model (2) 1 step finto di conversione (3) 1 batch di SFT dummy (4) 1 inferenza.

VINCOLI PRATICI (GTX 1080 / Pascal, 8GB)

- Evita dipendenze che richiedono compute capability >= 7.0 (es. alcuni runtime tipo vLLM). Preferisci Transformers/torch standard.
- Supporta quantizzazione/QLoRA come opzione, ma non renderla obbligatoria (backend deve funzionare anche CPU-only in modalità demo).

DECISIONE TECNICA (backend)

- Linguaggio: Python 3.11+.
- Packaging: pyproject.toml.
- CLI: typer o argparse (scegli una, semplice).
- Server API: FastAPI opzionale, ma consigliato (endpoint /health, /generate, /jobs/*).
- Config: YAML o JSON (scegli una). Tutto config-driven.

STRUTTURA CARTELLE (da creare)
hildanext/
  backend/
    pyproject.toml
    README.md
    src/hildanext/...
    tests/...
  models/
    qwen3-0.6b/            (cartella HF locale, con safetensors+tokenizer)
    exports/               (checkpoint dLLM, merged, quantized)
  vendor/
    llada/                 (repo LLaDA/MDM/SMDM pin)
    dinfer/                (repo inclusionAI/dInfer pin)
  data/
    raw/                   (dolma v1_6 sample, tinystories)
    processed/             (jsonl, shards)
    tokenized/             (bin/arrow/parquet o HF datasets cache)
  runs/
    configs/
    logs/
    checkpoints/
  docs/
    llada2_0.html (se disponibile)
    llada2_1.pdf + llada2_1.html (se disponibili)
  VERSIONS.md
  DESIGN.md

CONCETTI CHE DEVI RISPETTARE (implementazione)

1) WSD conversion (LLaDA2.0)

- Warmup: aumenta progressivamente block size (tratta AR come BDLM con block size 1) fino a full-sequence (MDLM).
- Stable: training in regime MDLM per stabilizzare dinamiche diffusion.
- Decay: riduci block size per tornare a BDLM efficiente in inferenza.
- Implementa doc-level attention mask per packing: quando concateni più sample nello stesso sequence, impedisci attenzione cross-document.
- Opzionale ma utile: top-k checkpoint merge (implementa tool semplice che media pesi dei migliori k checkpoint).

2) LLaDA2.1 editing + soglie

- Decoding deve supportare due insiemi: Unmasking Set Γt (MASK->token) e Editing Set Δt (token->token).
- Due soglie configurabili: τmask e τedit.
- Modalità:
  - S_MODE: τmask più basso (draft veloce), più passate T2T per correggere.
  - Q_MODE: τmask più alto (draft più conservativo), meno correzioni, qualità più stabile.
- Implementa un decode loop che alterna M2T e T2T fino a stop (o max steps) e logga: steps, %mask rimaste, #edit applicati.

3) Training objective (CPT/SFT)

- CPT: su Dolma v1_6 sample, obiettivo misto:
  - M2T: predici token sulle posizioni mascherate
  - T2T: introduci rumore su token osservati e addestra a ripristinare token corretti (editing)
- SFT: su TinyStories o mini-instruction set (anche dummy), mantieni mixture M2T+T2T ma focalizza la mascheratura sulla “risposta” (instruction-following).
- Multi-turn forward minimal: genera stati intermedi/varianti (anche solo 2 turni) per aumentare casi di editing.

4) Inference engine (dInfer)

- Se dInfer è disponibile e supporta block diffusion/dLLM, crea adapter “Engine” con stessa interfaccia del fallback Transformers.
- Interfaccia unica:
  generate(prompt:str, mode:str, tau_mask:float, tau_edit:float, max_new_tokens:int, seed:int)->str
- Se dInfer fallisce, fallback a Transformers + implementazione decode (anche se lenta).

5) Benchmark / logging

- Implementa runner semplice per valutare almeno: TinyStories (per coerenza) e un micro HumanEval-like dummy (solo per pipeline).
- Log obbligatori: loss M2T, loss T2T, steps di denoise, throughput stimato (tokens/sec se misurabile), memoria.

PIANO DI LAVORO (devi seguirlo)
STEP 0: Scansione e inventario

- Leggi struttura workspace e crea /hildanext.
- Copia/posiziona docs in /hildanext/docs.
- Aggiungi /hildanext/VERSIONS.md e /hildanext/DESIGN.md (brevi ma chiari).
  STEP 1: Scaffold backend
- Crea backend Python installabile (pyproject) con moduli:
  config, io, datasets, tokenization, masks, diffusion, training, inference, api, cli, utils.
- Implementa CLI con comandi:
  hildanext prepare-data
  hildanext tokenize
  hildanext convert-wsd
  hildanext sft
  hildanext serve
  hildanext smoke-test
  STEP 2: Dati (Dolma v1_6 + TinyStories)
- Implementa downloader opzionale (se l’utente non ha già i file) e loader che accetta path locali.
- Implementa packing + doc-boundary mask.
  STEP 3: Conversione WSD (minimal funzionante)
- Implementa training loop CPT con schedule WSD parametrico.
- Supporta training su subset piccolo per debug (es. 1-10M token).
  STEP 4: LLaDA2.1 editing decode + objective
- Implementa decode loop M2T+T2T threshold-based.
- Implementa T2T noise injection + loss.
  STEP 5: Inference engine adapter
- Implementa adapter dInfer + fallback Transformers.
- Espone API e CLI generate.
  STEP 6: Smoke test + README
- Smoke test deve girare end-to-end su CPU in modalità demo (anche con batch tiny).
- README con comandi.

STILE CODICE (TOKEN-FRIENDLY, OBBLIGATORIO)

- Ogni file: 1 blocco commento iniziale (max 3-6 righe) con scopo e main entrypoints.
- Nessun altro commento salvo casi indispensabili (massimo 1 riga).
- Evita righe vuote superflue; no “pretty formatting” inutile.
- Nomi chiari, funzioni piccole.
- Error handling minimale ma robusto.

CRITERI DI “DONE”

- `pip install -e hildanext/backend` funziona.
- `hildanext smoke-test` stampa OK.
- `hildanext serve` espone /health e /generate e risponde (anche con modello dummy se weights non presenti).
- Tutto configurabile via file in /hildanext/runs/configs.

ORA ESEGUI:

- Crea /hildanext e implementa tutto lo scaffold + pipeline minimal.
- Non chiedermi conferme: fai assunzioni ragionevoli, documentale in DESIGN.md.
- Se ti manca un dato (es. path dataset), usa placeholder e env var, ma non bloccare la build.


 ti ho convertito i file pdf in html per comodità tua ### Remask Config (inference)

```python

@dataclass

class RemaskConfig:

    target_ratio: float = 0.15      # Target % tokens to remask

    min_ratio: float = 0.05         # Minimum remask ratio

    block_size: int = 64            # Block size for remasking

    block_stride: int = 32          # Stride between blocks

    percentile_safety: float = 0.95 # Confidence cap percentile

```

### Train Config (Pascal-safe)

```python

@dataclass

class TrainConfig:

    dtype: str = "bfloat16"         # Emulated on Pascal

    batch_size: int = 1             # Keep small for GTX 1080

    accum_steps: int = 8            # Effective batch = 8

    grad_ckpt: bool = True          # Required for VRAM

    max_vram_pct: float = 0.85      # Stay under 85%

    lr: float = 1e-4

    warmup_steps: int = 100

```## 11. Troubleshooting


### FlashAttention Error on Pascal


```python

# inference/flash_attn.py forces math SDPA:

torch.backends.cuda.enable_flash_sdp(False)

torch.backends.cuda.enable_mem_efficient_sdp(False)

torch.backends.cuda.enable_math_sdp(True)

```

### VRAM OOM

- Reduce `batch_size` to 1
- Enable `grad_ckpt: True`
- Reduce `seq_len` to 512
- Use 4-bit quantization (bitsandbytes)

### bfloat16 Issues

Pascal GPUs emulate bfloat16 → slower but works. For pure fp16:

```python

model = model.to(dtype=torch.float16)

```## Pascal (GTX 1080) Notes


- Uses bfloat16 (emulated)

- FlashAttention disabled → math SDPA

- Custom rotary_emb.py fallback
```

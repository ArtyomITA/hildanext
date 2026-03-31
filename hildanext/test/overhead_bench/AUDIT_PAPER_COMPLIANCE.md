# Audit di conformita HildaNext vs Paper — Classificazione rigorosa

**Scope**: ogni componente del training+inference pipeline classificato come:
- **MATCH** — implementazione conforme al paper
- **ADATTAMENTO** — plausibile ma non letterale (deviazione giustificata)
- **DEVIAZIONE** — divergenza dalla ricetta del paper
- **NON VERIFICABILE** — i paper non forniscono dettagli sufficienti

**Fonti normative**: LLaDA 2.0 (arXiv 2512.15745), LLaDA 2.1 tech report, MDLM (Sahoo et al.)

---

## 1. WSD Block-Size Schedule (fonte: LLaDA 2.0 S4.1)

### 1a. Struttura tre fasi: warmup → stable → decay

| Aspetto | Paper LLaDA 2.0 | HildaNext (`formulas.py`, `training.py`) | Classificazione |
|---------|------------------|------------------------------------------|-----------------|
| Tre fasi WSD | warmup → stable → decay | warmup → stable → decay (`llada2_wsd_block` return `phase`) | **MATCH** |
| Warmup: block size progressivo | L_B: 1→4→32→64→4096 | `ladder_blocks: [1,4,32,64,128,256,512,1024]` | **ADATTAMENTO** — 8 rungs vs 5, target 1024 vs 4096 (seq_len-limited) |
| Stable: block fissato al max | L_B = 4096 (= seq_len) | `block_size = seq_len = 1024` | **MATCH** — stesso principio: 1 block = intera sequenza = MDLM |
| Decay: block discendente | 4096→2048→...→32 | `decay_blocks: [1024,512,256,128,64,32]` | **ADATTAMENTO** — target 32 identico, valori scalati a seq_len=1024 |
| Divisibilita seq_len % block_size | "seq length must be divisible" | `enforce_divisibility=True`, `_align_divisor()` | **MATCH** |
| Interpolazione dentro la ladder | Non specificato nel paper | `_ladder_step`: linear index interpolation | **NON VERIFICABILE** — paper non dice come distribuire i steps tra i livelli della ladder |

### 1b. Durate delle fasi

| Aspetto | Paper LLaDA 2.0 | HildaNext (runtime, da `_apply_stage0_to_cfg`) | Classificazione |
|---------|------------------|-------------------------------------------------|-----------------|
| Rapporto warmup/stable/decay | Non specificato (paper usa "moderate-scale" per warmup) | 10%/70%/20% di max_steps → 400/2800/800 steps | **NON VERIFICABILE** — paper non fornisce percentuali |
| WSD come "conversion" da AR | "Convert AR → BDLM via progressive block warmup" | Qwen3-0.6B (AR) → training con masked diffusion | **MATCH** — stesso paradigma di conversione AR→dLLM |

---

## 2. Document-Level Attention Mask (fonte: LLaDA 2.0 S4.2, Eq.3)

### 2a. Composite mask (warmup + decay: BDLM phases)

| Aspetto | Paper Eq.3 | HildaNext (`_composite_llada20_mask`) | Classificazione |
|---------|-----------|---------------------------------------|-----------------|
| Sequenza composita [x_t, x_0] di lunghezza 2L | Si | `ids2 = cat([input_ids, x0], dim=1)`, `base_len = L` | **MATCH** |
| Quadrante x_t→x_t: block-diagonal | `1[b(i)=b(j)]` | `blk_i == blk_j` per posizioni in x_t | **MATCH** |
| Quadrante x_t→x_0: offset block-causal | `1[b(i) > b(j-L)]` (strictly earlier) | `blk_i > blk_j` per posizioni x_t→x_0 | **MATCH** |
| Quadrante x_0→x_0: block-causal | `1[b(i-L) >= b(j-L)]` | `blk_i >= blk_j` per posizioni x_0 | **MATCH** |
| Quadrante x_0→x_t: bloccato | `0` | Non incluso in `base = cond_xt_xt \| cond_xt_x0 \| cond_x0_x0` | **MATCH** |
| Document gating (cross-doc = 0) | "belong to same document" (Eq.4) | `same_doc & valid` AND-gated su tutti i quadranti | **MATCH** |

### 2b. MDLM mask (stable phase)

| Aspetto | Paper Eq.4 | HildaNext (`batch_doc_attention_mask` mode=`simple_blockdiag`) | Classificazione |
|---------|-----------|--------------------------------------------------------------|-----------------|
| Full bidirectional within document | `1[same document]` | `doc_attention_mask(causal=False)` → block-diagonal per doc | **MATCH** |
| No x_0 copy in stable phase | "clean part no longer needed" | Quando `mask_mode=simple_blockdiag`, nessun concat, input e solo x_t | **MATCH** |

### 2c. Switching automatico warmup↔stable↔decay

| Aspetto | Paper | HildaNext (`training.py` L730-L740) | Classificazione |
|---------|-------|-------------------------------------|-----------------|
| BDLM mask per warmup/decay | Composite 2L mask | `run_mask_mode = cfg.llada2.mask_mode` ("composite_llada20") | **MATCH** |
| MDLM mask per stable | Simple doc-level bidirectional | `run_mask_mode = "simple_blockdiag"` when `phase=="stable"` | **MATCH** |
| Fase automatica | Non descritto (il paper e per cluster multi-GPU) | Phase change detection ogni step via `wsd_block()` | **ADATTAMENTO** — stesso effetto, implementazione single-GPU |

---

## 3. Training Objective / Loss (fonte: LLaDA 2.0 Eq.1-2, MDLM)

### 3a. Masked CE loss su posizioni [MASK]

| Aspetto | Paper (MDLM/LLaDA 2.0 Eq.2) | HildaNext (`compute_m2t_t2t_losses`, `_causal_loss`) | Classificazione |
|---------|-------------------------------|------------------------------------------------------|-----------------|
| Loss solo su posizioni mascherate | `1[x_t^i = MASK] * log p(x_0^i \| x_t)` | `labels = -100` ovunque, poi `labels[m2t_pos] = input_ids[m2t_pos]` | **MATCH** (via `ignore_index=-100`) |
| Cross-entropy come loss base | Masked CE | `F.cross_entropy(... ignore_index=-100)` | **MATCH** |
| **Left-shift**: `logits[:,:-1]` vs `labels[:,1:]` | Paper: `log p(x_0^i \| x_t)` — UNO-a-UNO, posizione i predice posizione i | HildaNext: `_causal_loss` fa `logits[:,:-1]` vs `labels[:,1:]` | **DEVIAZIONE** |

**Dettaglio sulla deviazione del left-shift:**
- Nei modelli **masked diffusion** puri (MDLM, LLaDA), la loss e position-wise: logit alla posizione `i` predice il token alla posizione `i`.
- Il left-shift `logits[:,:-1]` vs `labels[:,1:]` e una convenzione da modelli **autoregressivi** (GPT-style), dove il logit alla posizione `i` predice la posizione `i+1`.
- HildaNext usa `preserve_left_shift` come scelta esplicita, probabilmente perche Qwen3 e addestrato come modello AR e la sua `lm_head` e calibrata per predire il token successivo.
- Il config ha `shift_mode: "preserve_left_shift"` e commenti: `"labels_offset=+1 first_position_ignored=True"`.
- **Questa e una deviazione architetturale consapevole**: se il modello AR predice `p(x_{i+1} | x_{<=i})`, usare left-shift durante la conversione diffusion mantiene la calibrazione della lm_head, ma sposta le predizioni di 1 posizione rispetto alla formulazione MDLM pura.

### 3b. Continuous-time t sampling

| Aspetto | Paper (MDLM ELBO) | HildaNext | Classificazione |
|---------|-------------------|-----------|-----------------|
| t campionato da U[t_min, t_max] | MDLM: `t ~ U[0,1]`, alpha_t = 1-t | `t ~ U(0.001, 1.0)` via `torch.empty(1).uniform_(t_min, t_max)` | **ADATTAMENTO** — stessa distribuzione, con epsilon floor per stabilita numerica |
| Masking i.i.d. con probabilita t | Ogni token mascherato indipendentemente con prob = 1 - alpha_t | `rand < t → mask` per ogni posizione candidata | **MATCH** — equivalente con alpha_t = 1-t |
| Un singolo t per tutto il batch | MDLM: un t per sequenza. Block Diffusion: un t per blocco | Un singolo t per l'intero batch (tutti gli item) | **ADATTAMENTO** — semplificazione per batch_size=1; differisce dalla formulazione block diffusion che campiona t per-block |

### 3c. Loss weighting 1/t (ELBO)

| Aspetto | Paper (MDLM Eq.2) | HildaNext (`diffusion.py` L396-398) | Classificazione |
|---------|-------------------|--------------------------------------|-----------------|
| Peso ELBO: `alpha'_t / (1-alpha_t)` | Con absorbing state, se alpha_t=1-t → peso = `1/(1-alpha_t)` = `1/t` | `loss_m2t = loss_m2t_raw / max(t, 0.001)` | **ADATTAMENTO** |

**Dettaglio**: Il peso ELBO completo e `alpha'_t / (1-alpha_t)`. Con la schedulazione lineare `alpha_t = 1 - t`:
- `alpha'_t = -1` (costante)
- `peso = |-1| / t = 1/t`

HildaNext implementa `1/t` direttamente. Questo e **matematicamente equivalente** alla derivazione ELBO per absorbing-state diffusion con schedule lineare, a meno della costante `|-1|` che non cambia la direzione del gradiente. Il `max(t, 0.001)` e un clamp numerico ragionevole.

**Nota**: il peso `1/t` si applica **solo alla M2T loss**, non alla T2T loss. Questo non e specificato nei paper (che non descrivono la loss T2T nel contesto ELBO weighting).

---

## 4. M2T + T2T Mixture (fonte: LLaDA 2.1 S3)

### 4a. M2T (Mask-to-Token) stream

| Aspetto | Paper LLaDA 2.1 | HildaNext | Classificazione |
|---------|-----------------|-----------|-----------------|
| Posizioni [MASK] → predict original token | "predict correct token at each masked position" | `m2t_pos → labels = input_ids[m2t_pos]` | **MATCH** (a meno del left-shift, vedi 3a) |
| M2T come drafting stream | "establishing foundational drafting capability" | M2T e la componente primaria della loss | **MATCH** |

### 4b. T2T (Token-to-Token) editing stream

| Aspetto | Paper LLaDA 2.1 | HildaNext (`t2t_corrupt_tokens`, `compute_m2t_t2t_losses`) | Classificazione |
|---------|-----------------|-------------------------------------------------------------|-----------------|
| Token gia visibili → corrotti con rumore | "randomly replaced with incorrect tokens" | `randint → replace; force != original` | **MATCH** |
| Modello deve identificare e correggere | "recover original tokens from random noise perturbations" | `t2t_labels[t2t_pos] = input_ids[t2t_pos]` | **MATCH** (a meno del left-shift) |
| T2T edit ratio | Paper: valore non estratto (in LaTeX base64) | Config: `t2t_noise_ratio = 0.15` | **NON VERIFICABILE** — valore specifico non leggibile dal report 2.1 |
| Posizioni M2T e T2T disgiunte | Implicito nel paper (sono due stream separati) | `remaining = cand & ~m2t_pos; t2t_pos = remaining & rand < ratio` | **MATCH** — set esplicitamente disgiunti |
| Single forward pass per entrambe | Non specificato (dipende dall'implementazione) | Mixed input → single `_forward()` → split loss | **ADATTAMENTO** — efficiente ma non descritto nel paper |
| Loss combinate con pesi | Paper: lambda per M2T e T2T (non estratto) | `loss = m2t_weight * loss_m2t + t2t_weight * loss_t2t` (entrambi = 1.0) | **NON VERIFICABILE** — pesi esatti non disponibili dal paper 2.1 |

### 4c. Multi-Turn Forward (MTF) data augmentation

| Aspetto | Paper LLaDA 2.1 | HildaNext | Classificazione |
|---------|-----------------|-----------|-----------------|
| MTF: multi-turn sequences che simulano il processo iterativo | "Multi-turn Forward data augmentation" | `multi_turn_t2t: 1` nel config (disabilitato de facto) | **DEVIAZIONE** — presente nel config ma non attivo (valore=1 = single turn) |

---

## 5. Inference: Threshold Decoding (fonte: LLaDA 2.1 S2, Eq.1-3)

### 5a. Gamma set (unmask) e Delta set (edit)

| Aspetto | Paper Eq.1-2 | HildaNext (`llada21_sets`) | Classificazione |
|---------|-------------|---------------------------|-----------------|
| Gamma = {i : x_i = MASK AND p > tau_mask} | Eq.1 | `gamma = masked & confidence >= tau_mask` | **MATCH** |
| Delta = {i : x_i != v_i AND p > tau_edit} | Eq.2 | `delta = ne(mask_id) & ne(tokens) & conf >= tau_edit` | **MATCH** |
| Transition: x_i^{t-1} = v_i if i in Gamma+Delta | Eq.3 | `out[gamma] = pred[gamma]; out[delta] = pred[delta]` | **MATCH** |
| v_i = argmax_v p(v \| x^t) | "top-candidate predicted token" | `pred_ids = logits.argmax(-1)` | **MATCH** |

### 5b. S-mode vs Q-mode

| Aspetto | Paper | HildaNext (`mode_thresholds`) | Classificazione |
|---------|-------|-------------------------------|-----------------|
| S-mode: tau basse (aggressivo) | "aggressively lowered" | `s_mode_tau_mask=0.08, s_mode_tau_edit=0.08` | **MATCH** concettualmente; valori esatti **NON VERIFICABILI** |
| Q-mode: tau alte (conservativo) | "strict threshold adherence" | `q_mode_tau_mask=0.18, q_mode_tau_edit=0.16` | **MATCH** concettualmente; valori esatti **NON VERIFICABILI** |

### 5c. Remasking

| Aspetto | Paper / Block Diffusion | HildaNext (`apply_remask`) | Classificazione |
|---------|------------------------|---------------------------|-----------------|
| Re-mask posizioni a bassa confidenza | Standard in block diffusion iterativo | `topk lowest-confidence → re-mask` con target_ratio=0.15 | **ADATTAMENTO** — principio corretto, parametri non dal paper 2.1 specificamente |
| Remask solo tra steps, mai all'ultimo | Standard | `if step + 1 < steps: apply_remask()` | **MATCH** |

### 5d. Block-based generation

| Aspetto | Paper LLaDA 2.0/2.1 | HildaNext | Classificazione |
|---------|---------------------|-----------|-----------------|
| Block-parallel decoding con KV cache | "block-wise causal masked attention with KV cache" | `TransformersEngine._decode()`: flat sequence, no block splitting | **DEVIAZIONE** — solo `DInferEngine` (non usato) supporta block_size |
| Multiple Block Editing (MBE) | LLaDA 2.1: "revisit and revise previously generated blocks" | Non implementato | **DEVIAZIONE** — funzionalita 2.1-only non presente |

---

## 6. Embed Noise (fonte: LLaDA 2.0 S7.1)

| Aspetto | Paper | HildaNext (`_install_embed_noise_hook`) | Classificazione |
|---------|-------|----------------------------------------|-----------------|
| Gaussian noise su embedding dei [MASK] | "add independent Gaussian noise to the output of the embedding layer for each masked token" | Hook su `embed_tokens`, rumore `N(0, std^2)` solo su posizioni `== mask_id` | **MATCH** |
| Solo durante iterazioni iniziali | "during initial iterations" | `_embed_noise_warmup_steps = wsd.warmup_steps`, rimosso a fine warmup | **MATCH** |
| Sigma value | **Non specificato** nel paper | `noise_std = 0.1` | **NON VERIFICABILE** |
| Decadimento lineare del noise | Non descritto | `set_embed_noise_std(0.1 * (1 - step/warmup))` | **ADATTAMENTO** — aggiunta plausibile non descritta nel paper |

---

## 7. Attention Mode: Bidirectional vs Causal (fonte: LLaDA 2.0 S4.1-4.2)

| Aspetto | Paper | HildaNext | Classificazione |
|---------|-------|-----------|-----------------|
| Stable phase: bidirectional within document | "MDLM: full bidirectional" (Eq.4) | `bidirectional_only_stable` → `bidirectional = (phase == "stable")` | **MATCH** |
| Warmup/decay: block-causal composite | BDLM mask (Eq.3) | `composite_llada20` mask attivata per fasi non-stable | **MATCH** |
| Monkey-patching di `is_causal` e `create_causal_mask` | Non applicabile (paper usa custom infra) | `force_noncausal_attention` context manager | **ADATTAMENTO** — necessario per Qwen3/HuggingFace, non descritto nel paper |
| Grad checkpoint adattivo: ON warmup/decay, OFF stable | Non descritto (paper usa HPC) | `training.py L720-730`: disable in stable, enable in warmup/decay | **ADATTAMENTO** — ottimizzazione HW-specifica, non nel paper |

---

## 8. Optimizer e Hyperparameters

| Aspetto | Paper LLaDA 2.0 | HildaNext | Classificazione |
|---------|-----------------|-----------|-----------------|
| Optimizer | **Non specificato** | PagedAdamW8bit (runtime-selected) | **NON VERIFICABILE** |
| Learning rate | **Non specificato** | 5e-5 | **NON VERIFICABILE** |
| Betas | **Non specificato** | (0.9, 0.95) via bitsandbytes default | **NON VERIFICABILE** |
| Weight decay | **Non specificato** | 0.01 | **NON VERIFICABILE** |
| LR schedule | **Non specificato** | Linear warmup + cosine decay | **NON VERIFICABILE** |
| Batch size | **Non specificato** | micro=1, accum=8 → effective=8 | **NON VERIFICABILE** |
| Gradient clipping | **Non specificato** | 1.0 (max norm) | **NON VERIFICABILE** |
| dtype training | **Non specificato** | fp16 (autocast) | **NON VERIFICABILE** |
| GradScaler | **Non specificato** | Non usato (modello gia in fp16) | **NON VERIFICABILE** |

---

## 9. Top-k Checkpoint Merge (fonte: LLaDA 2.0 S4.3)

| Aspetto | Paper | HildaNext (`merge_topk_checkpoints`) | Classificazione |
|---------|-------|--------------------------------------|-----------------|
| Average aritmetica dei pesi | "arithmetic average of parameters" | `acc / len(models)` | **MATCH** |
| Selezione top-k per validation perplexity | "best-performing by validation PPL" | Selezione manuale (lista di dirs) | **ADATTAMENTO** — merge implementato, selezione automatica no |
| Post-training, offline | Si | Funzione standalone | **MATCH** |

---

## 10. Riepilogo classificazioni

### MATCH (conformi al paper) — 22 items
- Tre fasi WSD (warmup/stable/decay)
- Stable → block=seq_len (MDLM equivalenza)
- Divisibilita block_size
- Composite mask 4 quadranti (Eq.3): tutti e 4 corretti
- Document gating su composite mask
- MDLM mask bidirectional per stable (Eq.4)
- Switch automatico BDLM↔MDLM per fase
- Masked CE solo su posizioni [MASK]
- Cross-entropy come loss base
- Masking i.i.d. con probabilita t
- T2T corruption: random replacement con force!=original
- T2T set disgiunto da M2T
- M2T come drafting stream
- Gamma set (Eq.1)
- Delta set (Eq.2)
- Transition operator (Eq.3)
- argmax candidate selection
- S-mode = tau basse, Q-mode = tau alte
- Remask solo tra steps
- Embed noise gaussiano su [MASK] embedding
- Embed noise solo durante warmup
- Top-k merge come average aritmetica

### ADATTAMENTO (plausibile ma non letterale) — 10 items
- Ladder warmup: 8 rungs (1→1024) vs paper 5 rungs (1→4096) — scalamento a seq_len
- Ladder decay: scalata a 1024 vs 4096
- Continuous-time epsilon floor (t_min=0.001)
- Un t per-batch vs per-block (batch_size=1 mitiga)
- 1/t weighting vs peso ELBO completo (matematicamente equivalente con schedule lineare)
- Single forward per M2T+T2T mixed
- Remasking parametri non da paper specifico
- force_noncausal_attention monkey-patching per HuggingFace
- Grad checkpoint adattivo (ottimizzazione HW non nel paper)
- Embed noise con decadimento lineare (non descritto)

### DEVIAZIONE (divergenza dalla ricetta) — 3 items
1. **Left-shift nella loss** (`logits[:,:-1]` vs `labels[:,1:]`): i modelli masked diffusion usano loss position-wise (posizione i → token i), non left-shift AR (posizione i → token i+1). Questa scelta e consapevole ("preserve_left_shift") e motivata dalla calibrazione AR della lm_head di Qwen3, ma diverge dalla formulazione MDLM/LLaDA pura.

2. **Multi-Turn Forward disabilitato** (`multi_turn_t2t: 1`): LLaDA 2.1 descrive MTF come fondamentale per la capacita di editing. Con valore=1 non c'e simulazione multi-turn.

3. **Block-based generation assente** nel TransformersEngine: il decode opera su sequenza flat senza block-parallel processing. MBE (Multiple Block Editing) non implementato.

### NON VERIFICABILE — 12 items
- Distribuzione steps nella ladder warmup
- Rapporto percentuale warmup/stable/decay
- T2T edit ratio esatto (0.15 usato)
- Pesi M2T/T2T (entrambi 1.0)
- Valori esatti tau in S-mode/Q-mode
- Sigma del noise embedding (0.1 usato)
- Optimizer, LR, betas, weight decay, batch size, LR schedule, gradient clipping, dtype, GradScaler — tutti non specificati nei paper

---

## 11. Le 3 deviazioni in dettaglio

### DEVIAZIONE 1: Left-shift nella loss (CRITICA)

**Paper (MDLM/LLaDA 2.0 Eq.2):**
$$\mathcal{L} = -\sum_i \mathbb{1}[x_t^i = \text{MASK}] \cdot \log p_\theta(x_0^i \mid \bm{x}_t)$$
La loss e calcolata **posizione per posizione**: logit alla posizione `i` predice il token alla posizione `i`.

**HildaNext:**
```python
logits[:,:-1,:]  # logit posizione i predice...
labels[:,1:]     # ...il token alla posizione i+1
```
Con il left-shift, se la posizione `i` e mascherata e ha label, il logit che la predice e quello alla posizione `i-1`. Questo e lo schema AR standard.

**Impatto**: dipende da come il modello gestisce il contesto bidirectional. In un modello AR puro (causal), il logit alla posizione `i` "vede" solo `x_{<=i}` e predice `x_{i+1}`. In un modello bidirectional (stable phase), il logit alla posizione `i` "vede" tutto il contesto — il left-shift diventa una convenzione della lm_head, non una necessita causale.

**Perche e stato fatto**: Qwen3 e un modello AR. La sua `lm_head` e addestrata per `logits[i] → token[i+1]`. Mantenere il left-shift durante la conversione diffusion evita di dover ri-calibrare la lm_head. Questo e un trade-off pragmatico: perdi 1 posizione di contesto per label, ma mantieni la compatibilita con i pesi pre-addestrati.

**Severita**: Media-bassa. Con seq_len=1024 e ~15% mask ratio, perdere la prima posizione e trascurabile. La lm_head produce comunque predizioni corrette per le posizioni successive dato il contesto bidirectional. Tuttavia, formalmente, non e conforme alla formulazione MDLM.

### DEVIAZIONE 2: MTF disabilitato

**Paper LLaDA 2.1**: "Multi-Turn Forward data augmentation [...] expose the model to a wider variety of editing scenarios, enhance the model's editing capabilities."

**HildaNext**: `multi_turn_t2t: 1` nel config = single turn = nessun MTF.

**Impatto**: il modello non viene esposto a scenari di editing multi-step durante il training. La capacita di T2T editing sara meno robusta rispetto a un modello addestrato con MTF. Tuttavia, per Stage 0 CPT con 4000 steps, MTF potrebbe essere prematuro.

**Severita**: Bassa per Stage 0, Alta per SFT/post-training futuri.

### DEVIAZIONE 3: No block-based inference

**Paper LLaDA 2.0/2.1**: block-parallel decoding con KV cache e fondamentale per l'efficienza a inference time. LLaDA 2.1 aggiunge MBE.

**HildaNext TransformersEngine**: il decode opera sulla sequenza intera per ogni step (no block splitting). Solo `DInferEngine` (non utilizzato) supporta `block_size`.

**Impatto**: inference molto piu lenta del paper. Nessun KV cache reuse.

**Severita**: Media per qualita (non impatta il training), Alta per throughput inference.

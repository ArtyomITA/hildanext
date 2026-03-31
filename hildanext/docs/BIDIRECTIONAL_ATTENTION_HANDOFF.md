# Bidirectional Attention — Implementation Handoff

**Data**: 2 Marzo 2026  
**Autore**: Copilot (agent)  
**Target**: ML Engineer  
**Stack**: Qwen3-0.6B · transformers 4.57.3 · PyTorch 2.4.0+cu121 · GTX 1080 8GB (SM_61, no FlashAttention)

---

## 1. Problema

Il modello `Qwen3ForCausalLM` ha **due pathway indipendenti** che impongono causal masking anche quando gli passiamo una 4D attention mask bidirectional:

| # | Pathway | Dove | Effetto |
|---|---------|------|---------|
| 1 | `Qwen3Attention.is_causal = True` | Ogni `self_attn` layer (`model.model.layers[i].self_attn`) | L'SDPA backend usa `is_causal = ... and attention_mask is None and getattr(module, "is_causal", True)`. Se `is_causal=True` e non c'è mask 4D → triangolare |
| 2 | `create_causal_mask()` | Chiamata in `Qwen3Model.forward()` → produce un mapping `{"full_attention": mask}` | Genera una maschera 4D lower-triangular che viene passata a ogni layer decoder |

Il pathway 2 ha un early-exit: se l'utente fornisce una mask 4D, `_preprocess_mask_arguments` la restituisce as-is senza aggiungere causality. Nella nostra architettura composite la mask 4D è **sempre presente** → il pathway 2 è già neutralizzato automaticamente.

Il pathway 1 è rilevante solo quando **non** passiamo una mask 4D. Nel nostro caso (composite_llada20 mask sempre presente), SDPA vede `attention_mask is not None` → `is_causal=False` automaticamente.

**Conclusione**: con la nostra 4D composite mask, il modello è già bidirectional. Ma il context manager `force_noncausal_attention` serve come **safety net** per edge case (fallback path senza mask 4D, future versioni di transformers).

---

## 2. Soluzione implementata

### 2.1 Context Manager: `force_noncausal_attention`

**File**: `backend/src/hildanext/diffusion.py` (righe 20–109)

```python
@contextlib.contextmanager
def force_noncausal_attention(model: torch.nn.Module):
```

**Cosa fa** (2 azioni):

1. **Flip `is_causal=False`** su tutti i sub-moduli con attributo `is_causal`:
   ```python
   for name, mod in model.named_modules():
       if hasattr(mod, "is_causal"):
           mod.is_causal = False
   ```

2. **Monkey-patch `create_causal_mask`** in 3 moduli (`masking_utils`, `qwen3`, `qwen2`):
   - Se il risultato è la **mask 4D dell'utente** (early-exit) → la passa attraverso inalterata (preserva il composite blocking + doc gating)
   - Se il risultato è una **maschera causale fresca** → la sostituisce con `torch.zeros_like(result)` (fully permissive)
   - Ripristina tutto nel `finally`

**Primo bug trovato e corretto**: la versione iniziale faceva `return None`. Questo significava che `causal_mask_mapping["full_attention"] = None` → ogni layer riceveva `attention_mask=None` → **nessun masking** → doc gating leak. Corretto con la strategia identity-or-zeros.

### 2.2 Integrazione in `_forward()`

**File**: `backend/src/hildanext/diffusion.py` (righe ~169–182)

Sia il path `composite_llada20` che il path standard wrappano la chiamata a `model(...)` nel context manager quando `bidirectional=True`:

```python
if bidirectional:
    with force_noncausal_attention(model):
        out = model(input_ids=ids2, attention_mask=attn4d)
else:
    out = model(input_ids=ids2, attention_mask=attn4d)
```

### 2.3 Determinazione phase-aware

**File**: `backend/src/hildanext/training.py` (righe ~462–468)

```python
if attn_mode == "bidirectional_always":
    bidirectional = True
elif attn_mode == "bidirectional_only_stable":
    bidirectional = (phase.phase == "stable")
else:
    bidirectional = False
```

---

## 3. Composite mask: `_composite_llada20_mask`

**File**: `backend/src/hildanext/masks.py`

La mask composite ha struttura `[xt | x0]` dove `base_len = len(xt) = len(x0)`, `total = 2 * base_len`.

### Regole di visibilità (block-level)

| i in | j in | Condizione | Tipo |
|------|------|------------|------|
| xt | xt | `block(i) == block(j)` | Same-block bidirectional |
| xt | x0 | `block(i) > block(j)` | Past-block x0 → xt |
| x0 | x0 | `block(i) >= block(j)` | Past-or-same block x0 |
| x0 | xt | Mai | x0 non vede xt |

Dove `block(i) = i // block_size` (posizione locale nel rispettivo half).

### Doc gating

Tutte le regole di visibilità sono AND con:
$$
\text{same\_doc}(i, j) = (\text{doc\_id}[i] = \text{doc\_id}[j]) \wedge (\text{doc\_id}[i] \geq 0) \wedge (\text{doc\_id}[j] \geq 0)
$$

### Conversione in additive mask

```python
def _attn_for_model(mask, model):
    # [B, S, S] bool → [B, 1, S, S] float
    out = torch.zeros(m.shape, dtype=model_dtype)
    out = out.masked_fill(~m, torch.finfo(dtype).min)  # -inf per posizioni bloccate
    return out
```

Questo produce una **4D additive mask** che transformers 4.57 riconosce via `_preprocess_mask_arguments` (early-exit su `len(attention_mask.shape) == 4`).

---

## 4. Formula SDPA nel backend Math

Su Pascal SM_61 (no FlashAttention), usiamo `force_math_sdp(True)`:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{additive}}\right) V
$$

dove $M_{\text{additive}}[i,j] = 0$ (visibile) o $-\infty$ (bloccato), generato dalla composite mask.

L'argomento `is_causal` in `F.scaled_dot_product_attention` è determinato da:
```python
# In transformers/integrations/sdpa_attention.py:
is_causal = (
    query.shape[2] > 1
    and attention_mask is None        # ← False perché passiamo 4D mask
    and getattr(module, "is_causal", True)
)
```
Quindi con la nostra 4D mask: `is_causal = False` automaticamente.

---

## 5. Loss functions

### M2T (Mask-to-Token) con continuous time

$$
t \sim \mathcal{U}(t_{\min}, t_{\max}), \quad t_{\min}=0.001, \; t_{\max}=1.0
$$

Ogni token è mascherato i.i.d. con probabilità $t$. Loss con ELBO weighting:

$$
\mathcal{L}_{M2T} = \frac{1}{t} \cdot \text{CE}(\text{logits}[:-1], \text{labels}[1:])
$$

### T2T (Token-to-Token)

Token corrotti con sostituzione random, stessa CE loss (senza weighting).

### Loss totale

$$
\mathcal{L} = w_{M2T} \cdot \mathcal{L}_{M2T} + w_{T2T} \cdot \mathcal{L}_{T2T}
$$

### Left-shift (S0-D)

Il left-shift standard di causal LM è preservato: `logits[:, :-1]` predice `labels[:, 1:]`. La posizione 0 non è mai predetta.

---

## 6. Test implementati

### 6.1 Test: Same-Block Bidirectional

**File**: `tools/tests_wsd/test_bidirectional_composite_runtime.py`

**Idea**: Se l'attenzione è davvero bidirezionale dentro un blocco, perturbando il token $j > i$ nello **stesso blocco xt**, i logits alla posizione $i$ **devono cambiare**.

**Procedura**:
1. Genera `input_ids` random, seq_len=128, block_size=32
2. Costruisci la composite mask (`_composite_llada20_mask`)
3. Posiziona i=5, j=31 (stesso blocco 0 del half xt)
4. **Run 1**: forward con input originale → `logits_base[i]`
5. **Run 2**: perturba `input_ids[j]` (+1 mod vocab) → `logits_mod[i]`
6. $\delta = \text{mean}(|\text{logits\_mod}[i] - \text{logits\_base}[i]|)$
7. **PASS** se $\delta > 10^{-5}$

**Risultato**: $\delta = 1.243$ → **PASS**

Nota: `delta_with_override == delta_without_override == 1.243`, confermando che la 4D mask da sola disabilita già la causalità nel backend SDPA.

### 6.2 Test: Doc Gating Leakage

**Idea**: Se il doc gating funziona, perturbando un token del **doc B**, i logits di un token del **doc A** non devono cambiare.

**Procedura**:
1. `doc_ids[:, :64] = 0` (doc A), `doc_ids[:, 64:] = 1` (doc B)
2. Probe at i=5 (doc A), perturb at j=69 (doc B)
3. Il mask composite deve avere `mask[i, j] = False` (cross-doc bloccato)
4. $\delta = \text{mean}(|\text{logits\_mod}[i] - \text{logits\_base}[i]|)$
5. **PASS** se $\delta < 10^{-6}$

**Risultato**: $\delta = 0.0$ esatto → **PASS**

### Risultato completo dal test report

```json
{
  "all_pass": true,
  "bidirectional_pass": true,
  "doc_gating_pass": true,
  "bidirectional_test": {
    "delta_with_override": 1.24316406,
    "delta_without_override": 1.24316406,
    "i": 5, "j": 31, "block_size": 32
  },
  "doc_gating_test": {
    "delta": 0.0,
    "i": 5, "j": 69, "doc_i": 0, "doc_j": 1
  },
  "elapsed_sec": 28.1,
  "model_dir": "Qwen3-0.6B",
  "backend": "math_sdpa"
}
```

---

## 7. Preflight failsafe

**File**: `backend/src/hildanext/wsd_stage0.py` (`preflight_wsd()`)

Flow inserito **dopo** il test AR e **prima** del probe di training:

```
AR_TEST → CACHE_CLEAR → BIDIR_TEST → [failsafe] → PROBE → OK
```

### Logica failsafe

```
if attention_mode ∈ {bidirectional_always, bidirectional_only_stable}:
    run bidir_test()
    if PASS:
        bidirectional_verified = True
    else:
        attention_mode = "causal_always"       # override in-place
        bidirectional_disabled_reason = <reason>
        save effective config to runs/reports/<run_id>_config_effective.json
        record_fallback(action="bidirectional_disabled")
```

Possibili `disabled_reason`:
- `force_noncausal_not_effective` — Test 1 fallito
- `doc_gating_leakage` — Test 2 fallito
- `bidir_test_exception: <error>` — Eccezione durante il test

---

## 8. Logging

### ExperimentConfig (nuovi campi)

**File**: `backend/src/hildanext/config.py`

```python
bidirectional_verified: bool = False
bidirectional_disabled_reason: str = ""
```

### JSONL log rows (ogni step)

Campi aggiunti in ogni riga del log `cpt.jsonl`:

```json
{
  "bidirectional": true,
  "is_causal_effective": false,
  "attention_mode": "bidirectional_only_stable",
  "bidirectional_verified": true,
  "bidirectional_disabled_reason": ""
}
```

### RUN_START log

```
shift_mode=preserve_left_shift time_param=continuous_time loss_weighting=inv_t
attention_mode=bidirectional_only_stable bidir_verified=True
```

### PHASE_CHANGE log

```
PHASE_CHANGE wsd_phase=stable block_size=32 bidirectional=True
is_causal_effective=False bidirectional_verified=True step=100/5000
```

---

## 9. File modificati

| File | Tipo | Descrizione |
|------|------|-------------|
| `backend/src/hildanext/diffusion.py` | MODIFIED | Context manager `force_noncausal_attention`, wrapping in `_forward()` |
| `backend/src/hildanext/wsd_stage0.py` | MODIFIED | Bidirectional runtime test + failsafe in `preflight_wsd()` |
| `backend/src/hildanext/training.py` | MODIFIED | Campi `bidirectional_verified`, `bidirectional_disabled_reason` in log/PHASE_CHANGE |
| `backend/src/hildanext/config.py` | MODIFIED | 2 nuovi campi in `ExperimentConfig` |
| `tools/tests_wsd/test_bidirectional_composite_runtime.py` | NEW | Runtime test (same-block + doc gating) |
| `tools/tests_wsd/__init__.py` | NEW | Package init |
| `tools/__init__.py` | NEW | Package init |
| `reports/bidir_composite_test.json` | NEW | Test report JSON |

---

## 10. Scoperte chiave

1. **La 4D mask è sufficiente** su transformers 4.57 + SDPA: quando passi una mask 4D, `_preprocess_mask_arguments` fa early-exit, e SDPA imposta `is_causal=False` perché `attention_mask is not None`. Il context manager è una safety net, non una necessità stretta.

2. **Monkey-patch `create_causal_mask` → None è sbagliato**: restituire `None` fa sì che ogni layer riceva `attention_mask=None`, eliminando anche la composite mask → doc leakage. La fix corretta è passare attraverso la mask dell'utente (identity-or-zeros strategy).

3. **Il vecchio test `attn_mask_test.json` era difettoso**: usava una mask 2D (tutti 1), che non bypassa `create_causal_mask` in transformers 4.57. Per testare la bidirezionalità serve una mask **4D** + verifica empirica con perturbazione.

---

## 11. Come runnare il test manualmente

```powershell
cd hildanext
conda activate mdm
python -u tools/tests_wsd/test_bidirectional_composite_runtime.py `
    --model-dir E:/DIFFUSION/HildaNext/Qwen3-0.6B `
    --block-size 32 `
    --seq-len 128 `
    --out reports/bidir_composite_test.json
```

Output atteso:
```
[bidir_test] PASS=True delta_with=1.24316406 delta_without=1.24316406
[bidir_test] PASS=True delta=0.0000000000
[bidir_test] === PASS ===
```

---

## 12. Cosa NON è stato toccato

- `masks.py` — invariato, la composite mask funzionava già correttamente
- `start_wsd_full_logs.ps1` — nessuna modifica necessaria (preflight → run-wsd come processi separati, il failsafe agisce nel preflight)
- Nessun framework web introdotto
- Nessun combinatore somme implementato

# Selezione del modello e HILDA SAFE vNext per RL e reasoning su GTX 1080

## Obiettivo sperimentale e vincoli reali di compute

Il tuo obiettivo è chiarissimo: dimostrare che una pipeline “diffusion-first” (stile LLaDA) **può portare modelli piccoli a livelli molto alti**, e che scalando lo stesso impianto si può ottenere qualità comparabile con un costo inferiore. Questo impone due requisiti: confronto “apple-to-apple” e scelte architetturali “SAFE ma SOTA”, cioè **minimo indispensabile ma già “la scelta che faresti nel FULL”** (non sostituti temporanei).  

Il vincolo hardware (GTX 1080, cc 6.1 / Pascal) è gestibile, ma con alcune conseguenze pratiche importanti:

- **bitsandbytes** su Pascal è una carta forte: supporta GPU con Compute Capability **≥ 6.0** per 8-bit optimizers/quantization e NF4/FP4 (QLoRA), mentre **LLM.int8() richiede 7.5+** (quindi non è il tuo caso). citeturn16search0  
- Il “serving moderno” tipo vLLM è spesso **fuori portata** su Pascal: vLLM richiede GPU con compute capability **≥ 7.0** (e diversi casi reali su 6.1 falliscono con “no kernel image”). citeturn21search5turn21search1  
- Il canale più robusto su Pascal per inferenza locale resta spesso **Transformers classico** o **llama.cpp/GGUF** (specie se vuoi testare più modelli e molte quantizzazioni). Qwen rilascia direttamente GGUF ufficiali per Qwen3 0.6B e documenta l’uso con llama.cpp. citeturn21search3  
- Sul lato toolchain: NVIDIA ha annunciato che in CUDA 13.0 è stato rimosso il supporto di offline compilation e librerie per architetture pre-7.5; quindi, nel tempo, diventa più importante ridurre dipendenze da estensioni CUDA “esotiche” e tenere un setup compatibile (CUDA 12.9 / driver branch indicato da NVIDIA per compatibilità legacy). citeturn16search6turn16search18  

Per il tuo caso, quindi, il “sweet spot” è: **modello 0.6B–2B**, training principalmente via **QLoRA + gradient checkpointing**, e report sperimentali che misurino miglioramenti su reasoning/coding con costi controllati.

## Modelli open consigliati per RL/reasoning e confronto architetturale

Qui la tua intuizione è corretta: **MDLM-130M** è ottimo come “laboratorio di meccaniche”, ma quando vuoi misurare (a) RL, (b) reasoning verificabile, (c) differenze sottili tra decoding/trajectory e training objective, un backbone troppo piccolo rischia di saturare o essere instabile. TinyStories mostra che anche modelli minuscoli possono imparare coerenza e una forma di “reasoning” narrativo, però compiti come GSM8K/MATH richiedono più capacità per produrre segnale misurabile e non puro rumore. citeturn15search0turn15search1turn15search2  

### Criteri di scelta del backbone per HILDA SAFE

Per la tua “SAFE = subset del FULL”, il backbone deve soddisfare:

- **Licenza davvero permissiva** (idealmente Apache-2.0) per poter fare training/derivati senza attriti.  
- **Disponibilità di variante Base** (pretraining) per fare conversione/continual pretraining in modo pulito.  
- **Architettura standard e ben supportata** dall’ecosistema PyTorch/Transformers.  
- **Tokenizer gestibile**: in diffusion ti serve un token/funzione equivalente a [MASK], e devi poter controllare bene special tokens e masking. (Se un token speciale non è registrato, può venire spezzato/gestito male dai runtime). citeturn11search18turn11search4  
- **Taglia sufficiente** per vedere differenze su reasoning/coding con RL leggero (idealmente ≥ ~1B, oppure un 0.6B molto “forte” e moderno).

### Shortlist ad alta probabilità di successo sulla tua GPU

#### Qwen3 (consigliato come asse principale)

La famiglia Qwen3 è particolarmente adatta al tuo obiettivo perché combina: open weights Apache-2.0, taglie piccole, supporto lungo contesto, e soprattutto un’impostazione “reasoning-aware” (thinking vs non-thinking) già integrata. citeturn6view0turn8view1turn22search7turn22search16  

- **Qwen3-0.6B / Qwen3-0.6B-Base**  
  - Parametri: 0.6B (0.44B non-embedding), contesto 32,768, Apache-2.0. citeturn6view0turn6view1  
  - Pro: perfetto come “dev backbone” (conversione dLLM + RL leggero) e per iterare veloce.  
  - Contro: per benchmark seri di reasoning (GSM8K/MATH/HumanEval) rischi ancora una ceiling bassa, ma è già anni luce più interessante di 130M.  

- **Qwen3-1.7B-Base** (questa è la mia raccomandazione principale per HILDA SAFE “seria”)  
  - Parametri: 1.7B (1.4B non-embedding), 32,768 contesto, Apache-2.0, disponibile come Base (pretraining). citeturn8view0turn22search7  
  - Pro: è abbastanza grande da rendere **significative** differenze tra (baseline AR) vs (conversione dLLM + RCD + RL).  
  - Nota operativa: Qwen3 in Transformers richiede versioni recenti; i model card avvertono che con Transformers sotto una certa soglia puoi avere errori (KeyError del tipo “qwen3”). citeturn6view1turn8view1  

Se hai budget per un “main run” e vuoi davvero misurare RL/reasoning in modo credibile, **Qwen3-1.7B-Base** è il punto di equilibrio.

#### SmolLM2-1.7B (alternativa forte, molto “scientifica”)

SmolLM2 è interessante perché è pensato “data-centrico” come small model, con report di valutazione già nel model card, ed è Apache-2.0. citeturn10view0turn0search10  
Nel model card si vedono numeri direttamente utili per te (HellaSwag/ARC/MMLU-pro/GSM8K ecc.) e quindi puoi usarlo come baseline comparabile senza dover “indovinare” se il modello è troppo debole. citeturn10view0  

Trade-off: SmolLM2 è soprattutto **English-first** e Llama-like (ottimo per benchmark standard), mentre Qwen3 ti dà più copertura multilingual e un ecosistema “reasoning mode” già delineato. citeturn10view0turn6view0  

#### OLMo-1B (alternativa “Dolma-native”, full open science)

Se vuoi massimizzare coerenza tra “data” e “filosofia di training”, OLMo è nato su Dolma e si posiziona come progetto molto aperto: model card indica licenza Apache 2.0 per codice e modello. citeturn19search2turn0search4  
È però più datato come famiglia rispetto a Qwen3, quindi lo vedo soprattutto come baseline “reproducible scientific stack”.

#### Qwen2.5-0.5B (se vuoi massimizzare facilità VRAM/KV per inference)

Qwen2.5-0.5B è Apache-2.0 ed è noto per essere molto comodo su hardware piccolo; ha GQA con soli **2 KV heads**, che riduce pressione su KV-cache in inference rispetto a varianti con più KV heads. citeturn23view0  
Però, se lo scopo è ragionamento+RL e vuoi “moderno”, **Qwen3 0.6/1.7** tende ad essere un miglior asse.

### Raccomandazione finale “pragmatica”

- **Dev / debug / iterazione veloce:** Qwen3-0.6B-Base citeturn6view1  
- **Run principale per RL + reasoning misurabile:** Qwen3-1.7B-Base citeturn8view0  
- **Baseline alternativa con eval trasparente:** SmolLM2-1.7B citeturn10view0  

Sono tutte scelte Apache-2.0 (nessun attrito di licenza) e hanno ecosistema forte su entity["company","Hugging Face","platform, us"].

## Strategia dati con Dolma v1.6 sample e TinyStories

Correzione terminologica: “Delma 1.6” con ogni probabilità è **Dolma v1.6** (dataset open di AI2). La repo dataset espone esplicitamente file e URL con path `dolma-v1_6/…`, e il dataset card lo collega al paper Dolma. citeturn18view0turn0search0turn0search20  

### Cosa significa “20GB tokenized” in termini di progetto

Senza sapere il formato (uint16, int32, ecc.) è impossibile tradurre esattamente in numero token; però 20GB di token pre-serializzati spesso corrispondono a **miliardi** di token. Questo è importante perché:

- LLaDA2.0 e Fast-dLLM v2 sono esempi di metodi che dichiarano conversioni/adattamenti “leggeri” nell’ordine di ~1B token (che è già enorme su una 1080, ma concettualmente molto meno di un pretraining da zero). citeturn17search0turn13search1turn12search1  
- RCD dichiara conversione efficiente a RCD con ~1B token e guadagni 5–10 punti accuracy con overhead minimo, ma quel budget è ancora “big” per Pascal: va pianificato come obiettivo di lungo periodo o come “scala-down test” con subset coerenti. citeturn12search1  

### Come usare Dolma + TinyStories senza rovinare l’esperimento

Il tuo desiderio di usare “tutto Dolma + tutto TinyStories” è comprensibile, ma per test architetturali l’aspetto più critico è **ridurre varianza e confondenti**.

Propongo una strutturazione che ti dà sia “uso totale” (se vuoi davvero) sia un protocollo che produce risultati interpretabili.

**Uso di TinyStories**  
TinyStories è un ottimo “microscopio” per emergenza di coerenza e abilità narrativa anche in modelli piccoli; nasce proprio per questo. citeturn15search0  
Però è una distribuzione molto specifica (storie semplici e linguaggio controllato). Quindi:

- Se TinyStories entra nel training mix, deve entrare identicamente sia nella baseline AR sia nella tua HILDA-dLLM, altrimenti stai misurando “data shift” e non architettura.
- Se lo scopo è misurare ragionamento matematico/coding, TinyStories può restare come *evaluation set separato* (stabilità e coerenza), mentre il training principale per capacità generali rimane Dolma.

**Uso di Dolma v1.6**  
Dolma è un corpus enorme pensato per language modeling; il paper lo descrive come corpus multi-sorgente su scala trilioni di token e il dataset card di HF lo presenta come risorsa massiva per training. citeturn0search20turn0search0  

Per un confronto pulito, ti consiglierei:

- Un “fixed token budget” per run (es. X milioni di token) per confrontare architetture a parità di compute.
- Una “long run” fuori benchmark (se vuoi davvero consumare l’intero 20GB tokenized) come *scaling test* secondario, non come test principale di architettura.

In pratica: prima fai **esperimenti confrontabili**, poi fai **scaling**.

## HILDA SAFE vNext con Qwen3 + LLaDA2.1 + RCD + wd1++

Questa è una proposta SAFE che rispetta esattamente il tuo criterio: **minimo indispensabile, ma già le scelte che useresti nel FULL**. Il FULL aggiungerà solo cose “invasive” e/o più speculative (tipo non-Markovian dynamics), ma la SAFE non usa placeholder.

### Backbone e conversione: da AR (Qwen3) a dLLM “editable”

**Backbone suggerito:** Qwen3-1.7B-Base come modello di partenza. citeturn8view0turn22search16  

**Obiettivo di conversione (core): paradigma LLaDA2.1**  
LLaDA2.1 introduce Token-to-Token editing (T2T) integrato nello schema Mask-to-Token (M2T), con threshold decoding configurabile e due modalità operative (Speedy vs Quality). citeturn12search0turn17search16  

Per HILDA SAFE vNext questo implica:

- Il tuo dLLM non deve solo “imputare [MASK] → token”, ma deve imparare anche **correzioni token→token** come operazione nativa (editing), perché è il meccanismo che ti permette di spingere parallelismo senza far crollare qualità. citeturn12search0  

**WSD / staged conversion**  
LLaDA2.0 formalizza un WSD (Warmup–Stable–Decay) a livello block diffusion per convertire AR → dLLM: warmup con block-size crescente, stable su full-sequence diffusion, decay tornando a block-size compatto. citeturn17search0turn17search7  
In SAFE vNext, la filosofia è identica: anche se tu lo farai in scala piccola, il processo resta lo stesso.

**Nota tokenizer / [MASK]**  
Qwen non ti garantisce “extra tokens” inattivi utilizzabili come [MASK] (ci sono issue storiche proprio su questo), quindi l’approccio più robusto è: **aggiungere un token speciale reale** e gestirlo come atomico. citeturn11search6turn11search4turn11search18  
Questo è importante perché dLLM vive e muore sulla stabilità del masking.

### RCD come upgrade core-compatibile

**Resid­ual Context Diffusion (RCD)** prende il problema centrale dei block dLLM: con remasking scarti computazione. RCD ricicla le rappresentazioni scartate come residui contestuali reiniettati nello step successivo; il paper dichiara miglioramenti 5–10 punti accuracy e riduzione dei denoising steps fino a 4–5× a parità di accuracy, e una conversione “economica” nell’ordine di ~1B token. citeturn12search1  

In SAFE vNext, RCD è perfetto perché:

- non richiede introdurre non‑Markovian forward process (che sarebbe FULL-tier),
- ma ti dà “memoria/riuso” e spesso riduce steps (quindi riduce compute, che è la tua risorsa più scarsa).

### Stage 1: SFT per reasoning senza “dataset giganti”

Come baseline principled, d1 propone un post-training per dLLM: masked SFT + RL (diffu-GRPO) per scalare reasoning. citeturn14search2  
Per SAFE vNext, però, la chiave è: **fai SFT in modo coerente col core LLaDA2.1** (M2T + esempi di editing T2T), e usa dataset con reward verificabile per RL.

In pratica, Stage 1 dovrebbe includere:

- **Masked SFT** stile d1 (mascheri principalmente la risposta, distilli ragionamento e auto-correzione). citeturn14search2  
- “Editable SFT”: includi minibatch dove la risposta è parzialmente “sporca” e supervisioni correzioni T2T (coerente col core LLaDA2.1). citeturn12search0turn17search16  

Se vuoi aggiungere struttura senza esplodere la complessità, **C2DLM** resta un’opzione molto allineata (teacher causal graph → guida l’attenzione verso relazioni causali tra concetti), con claim di miglioramenti su task di reasoning e anche speedup in specifici setting. citeturn14search0turn14search4  
Io però la vedrei come “SAFE+”: attivala se ti regge il costo di estrazione del causal graph.

### Stage 2: RL “minimo indispensabile” che puoi davvero eseguire su 1080

Tu hai chiesto esplicitamente **wd++** (wd1++). È una scelta centrata per il tuo vincolo compute.

- wd1 propone policy optimization **ratio-free** riformulando l’obiettivo RL come weighted log-likelihood, evitando overhead di ratio/likelihood multipli;  
- wd1++ estende a un’ottimizzazione **denoising-stepwise** e dichiara SOTA math performance con **solo 20 RL steps** (MATH500 e GSM8K) su LLaDA-8B, con meno costo dei baseline. citeturn12search2  

Per HILDA SAFE vNext io farei:

- **Default RL = wd1++** (non alternativo), perché è “SOTA nel campo dell’innovazione” e riduce compute, quindi è una scelta che useresti anche nel FULL. citeturn12search2  

Upgrade coerente e recentissimo: **STP (Spatio-Temporal Pruning)** è uscito il 9 febbraio 2026 e attacca precisamente i tuoi pain point: compressione della ridondanza generativa e salto di refinement tardivo; il paper collega STP a riduzione della varianza dell’estimatore e stabilità delle policy update. citeturn12search3turn12search6  
Se vuoi “minimo indispensabile” ma davvero utile su compute scarso, **STP è un candidato fortissimo come wrapper** attorno al tuo RL (anche perché nasce proprio per efficienza+stabilità RL su dLLM). citeturn12search3  

Nota: ESPO è una baseline principled molto forte (sequence-level action e ELBO proxy) e in un mondo “ideale” io la terrei come baseline scientifica sempre presente. citeturn14search1  
Ma se devi scegliere **un solo** RL method come “default mainline” per SAFE, wd1++ + (eventualmente) STP è più compatibile con la tua compute.

### Stage 3: ottimizzazioni “tier 3/4” per inferenza che hanno senso su Pascal

Qui separo “metodi dLLM-specifici” da “sistema”.

**Decoding/search (alta leva, spesso training-free)**  
- **Order-Token Search** (gennaio 2026) propone ricerca congiunta nello spazio “ordine di generazione + token”; mostra miglioramenti su GSM8K/MATH500/Countdown/HumanEval e stabilisce la search come componente chiave per decoding in DLM. citeturn13search2  
Per te è oro perché puoi applicarlo anche a modelli piccoli per vedere differenze reali senza rifare training.

**Ibridi per speed (dLLM → faster-than-AR)**  
- **D2F (Discrete Diffusion Forcing)** ibridizza AR e dLLM per usare KV-cache e parallelismo inter-blocco; mira esplicitamente a superare AR in speed e ha codice pubblico. citeturn13search0turn13search12turn13search16  
- **Fast-dLLM v2** propone ricetta di adattamento da AR a dLLM con ~1B tokens e caching gerarchico; dichiara speedup fino a 2.5× con qualità preservata. citeturn13search1turn13search9  

Su GTX 1080, però, devi stare attento: molte implementazioni “veloci” presuppongono stack moderni (FlashAttention, vLLM). Quindi in SAFE vNext io li terrei come **moduli opzionali di Stage 3**, con fallback su Transformers/llama.cpp.

**KV/cache pruning (utile soprattutto per AR baseline e ibridi)**  
- **KVzap** (gennaio 2026) nasce per essere adottabile nei motori reali: pruning input-adaptive sia in prefilling sia in decoding, motivato dal fatto che altri metodi non erano stati adottati per trade-off speed/accuracy. citeturn13search3turn13search7  
Questo è più rilevante se continui ad avere componenti AR/ibridi nel tuo stack di serving.

### Stage 4: cosa mettere e cosa tenere separato

Nella tua filosofia originaria, Tier-4 era giustamente “branch separato”. È ancora sensato: Tier-4 dovrebbe includere cose come backend alternativi (es. discrete flow matching) che cambiano paradigma e rendono difficile mantenere coerenza degli estimator RL/ELBO.

Nella SAFE vNext che vuoi tu, io metterei Stage 4 come:

- **Long-run branch** per backend alternativi e/o tokenizer-differentiable (tipo Zonkey), **non** nella mainline di prove architetturali. (Questo è coerente con l’idea che SAFE dev’essere subset del FULL: il FULL può includere branch, ma non devi mischiarli nella stessa serie di esperimenti). citeturn12search1turn12search0  

## Protocollo di benchmark per dimostrare che HILDA “alza il livello” nei small model

Per dimostrare la tua tesi (“piccolissimi a livelli altissimi”) ti serve un protocollo che misuri:

- qualità generativa generale (LM),
- reasoning verificabile (math),
- coding funzionale,
- e costo (steps / throughput).

### Benchmark consigliati

- **TinyStories** come test di coerenza/grammatica/stabilità (anche se non lo usi nel train). citeturn15search0  
- **GSM8K** per reasoning matematico verificabile. citeturn15search1  
- **MATH / MATH500** per stress test più difficile. citeturn15search2turn12search2  
- **HumanEval** per coding funzionale. citeturn15search3turn13search2  

### Misure “architetturali” che devi loggare sempre

Per confrontare baseline AR vs HILDA-dLLM:

- accuracy su GSM8K/MATH/HumanEval (o pass@k)  
- numero effettivo di denoising steps (e quanto RCD li riduce) citeturn12search1  
- costo di RL in “rollouts per gain” (wd1++ enfatizza risultati forti con pochi steps, che è perfetto per te) citeturn12search2  
- throughput di decoding (se attivi search o hybrid forcing) citeturn13search2turn13search12  

## Raccomandazione concreta per partire senza perdere mesi

Se vuoi massimizzare probabilità di arrivare a un risultato “publishable” con la tua compute:

- Usa **Qwen3-0.6B-Base** per validare tutta la pipeline dLLM (mask token, conversione WSD in scala ridotta, compatibilità training loop). citeturn6view1turn17search0  
- Poi fai il “main experiment” su **Qwen3-1.7B-Base** con HILDA SAFE vNext:  
  **LLaDA2.1-style editing + RCD + wd1++ (default RL) + Order-Token Search (decoding)**. citeturn12search0turn12search1turn12search2turn13search2  
- Se devi scegliere un singolo acceleratore “serio” per RL su compute ridotto: considera **STP** come l’aggiunta più allineata al tuo vincolo (efficienza+stabilità RL su dLLM). citeturn12search3  

Sul lato operativo, per training su GTX 1080: QLoRA e 8-bit optimizers sono supportati (Compute Capability ≥6.0), quindi puoi spingere fino a 1–2B in modo realistico. citeturn16search0  
Per serving, evita vLLM su Pascal (compute capability ≥7.0) e preferisci Transformers o llama.cpp/GGUF (Qwen3 rilascia GGUF e documentazione). citeturn21search5turn21search3
# Audit e aggiornamento dell’architettura HILDA per dLLM su GPU Pascal a bassa VRAM

## Contesto, obiettivi e baseline tecnica

Questa analisi parte da tre input: la tua architettura HILDA “token‑friendly” (due varianti SAFE e FULL), una ricerca Perplexity e una ricerca Gemini che hai allegato. fileciteturn0file1 fileciteturn0file0 fileciteturn0file2  
La data di riferimento è **domenica 22 febbraio 2026** (Europe/Rome), quindi ho privilegiato fonti tra **gennaio 2026 e febbraio 2026**, includendo lavori pubblicati **negli ultimissimi giorni**.

La baseline HILDA che hai descritto è concettualmente coerente e già allineata a una “spina dorsale” dLLM moderna:

- Core: **LLaDA2.0 / MDLM‑BlockDiffusion** con **block diffusion**, **document-level attention masking** e training schedule **WSD (Warmup‑Stable‑Decay)**. fileciteturn0file1 citeturn0search0turn0search4turn0search8  
- Post‑training: pipeline **d1** (SFT poi RL for reasoning), con **ESPO** come default RL (ELBO sequence‑level, ratio stabilizzato, robust KL). fileciteturn0file1 citeturn16search3turn0search1  
- Supervisione strutturale: **C2DLM** (concept‑level causal graph, maschera supervisionata su attenzione, V‑aware re‑attention). fileciteturn0file1 citeturn0search2turn0search6  
- Stage 3 (serving): acceleratori/ibridi tipo **D2F** e **Fast‑dLLM v2**. fileciteturn0file1 citeturn5search1turn5search2  
- FULL aggiunge componenti “invasivi” sulla dinamica/obiettivo (CaDDi non‑Markoviano, PAPL, scheduling dinamico). fileciteturn0file1 citeturn5search3turn17search0turn17search4  

Il vincolo hardware (GTX 1080, sm_61, 8GB) è oggi più delicato che in passato, non tanto per la VRAM in sé, quanto per la compatibilità dello stack CUDA/PyTorch: per Pascal spesso serve **pinning** di build specifiche (es. build PyTorch con CUDA 12.6) perché build più nuove possono non includere più sm_61. citeturn10view2turn10view1turn10view0  
In parallelo, il futuro tooling CUDA tende a tagliare via il supporto “offline compilation/library” per Maxwell/Pascal/Volta (rimosso nel major CUDA 13.0), aumentando il valore di una pipeline che minimizzi custom CUDA extensions. citeturn12view0

## Miglioramenti nel core

**Quest 1 richiesto:** aggiornamenti del backbone e sostituzione/upgrade dei componenti core (incluso un “proto‑adattamento” per ciascun pezzo, per quanto possibile).

### Factcheck Perplexity e Gemini sul core

**Perplexity (ricerca):** il tuo dump Perplexity elenca correttamente diversi paper chiave e la loro direzione è sostanzialmente corretta:  
- la conversione **AR → diffusion** è un filone consolidato (es. DiffuGPT/DiffuLLaMA), coerente con il tuo Stage‑0 “conversion‑first”. fileciteturn0file0 citeturn4search0  
- “Zonkey” esiste e introduce tokenizzazione differenziabile e meccanismi probabilistici per gestire lunghezze/segmentazioni in modo soft; è reale ma è **invasivo** (tocca tokenizer e architettura) ed è più da “Architecture 3” che da patch sul tuo SAFE. fileciteturn0file0 citeturn1search0turn1search4  
- “CARD” esiste e formalizza un ibrido “causal diffusion” con **dynamic parallel decoding** e KV caching; è coerente con la tua intuizione D2F/semi‑AR, ma con un design più end‑to‑end. fileciteturn0file0 citeturn1search1turn1search9  
- “Diffusion in Diffusion” esiste: è letteralmente un “draft‑then‑refine” per recuperare coerenza globale nei block‑based dLLM. fileciteturn0file0 citeturn2search2turn2search6  
- “Residual Context Diffusion Language Models” esiste e Perplexity nella tua nota lo marcava come “da leggere perché poco chiaro”; oggi è chiaro dall’abstract: propone di riusare computazione “scartata” dal remasking sotto forma di residual contextual injection, con un pipeline di training in due stadi per evitare bottleneck di memoria e con claim di conversione efficiente (~1B token) e miglioramenti consistenti. fileciteturn0file0 citeturn3search0turn3search4turn3search12  

**Limite principale Perplexity sul core (alla data tua):** la finestra dichiarata si ferma a circa **6 febbraio 2026**, quindi manca almeno un upgrade core enorme uscito subito dopo: **LLaDA2.1** (sottomissione arXiv v1 il 9 febbraio 2026). fileciteturn0file0 citeturn13view1  

**Gemini (ricerca):** nel file Gemini compare come “unico update core” **LLaDA2.1**; questo punto è **corretto** e oggi è verificabile con arXiv + model card. fileciteturn0file2 citeturn13view1turn13view2  
Tuttavia Gemini la descrive con una metafora “drafting stream + error correcting”; nella versione ufficiale, il concetto è: integrare **Token‑to‑Token editing (T2T)** dentro lo schema **Mask‑to‑Token (M2T)** con **threshold decoding configurabile** e due modalità (Speedy vs Quality). È “draft‑then‑refine” nel senso funzionale, ma la meccanica precisa è M2T + T2T editing. citeturn13view1turn13view0  

### Ricerca aggiornata e proposta di upgrade core per HILDA

Qui propongo upgrade core **ordinati per “ROI su HILDA”** e includo un proto‑adattamento per ciascun componente, mantenendo la tua distinzione SAFE vs FULL. La logica è: **massimizzare la qualità dell’upgrade senza distruggere la compatibilità con ESPO/ELBO e con la tua infrastruttura “token‑friendly”.** fileciteturn0file1 citeturn0search1turn0search0  

#### Sostituzione del backbone: LLaDA2.0 → LLaDA2.1 (upgrade “diretto”)

**Perché è un upgrade core (non solo inference):** LLaDA2.1 non cambia “solo il decoding”, ma estende il paradigma di diffusione discreta oltre la rigidità dell’absorbing‑state [MASK] puro introducendo un’evoluzione “editable” in cui il modello può **correggere token già decisi** (T2T). Questo impatta la semantica di come allinei training/inference, e apre una knob “speed‑quality” esplicita. citeturn13view1turn13view0  

**Proto‑adattamento HILDA (SAFE):**
- Stage 0: mantieni la tua idea WSD e packing doc‑level, ma fai sì che il tuo “forward corruption” e il supervision target includano **casi di editing** (token non‑[MASK] che vengono perturbati e richiedono correzione), come descritto nel paper. citeturn13view0  
- Stage 1 (SFT): sposta l’obiettivo da “solo masked positions” a “masked + editable”, soprattutto sui token di risposta (coerente con LLaDA che in SFT maschera tipicamente la risposta). citeturn23search14turn13view0  
- C2DLM: applica la supervisione causale **preferenzialmente nelle passate T2T** (correzione), perché lì stai “riparando” inconsistenze locali e chain‑of‑reasoning, mentre in Speedy Mode potresti accettare più rumore. La razionalità è allineata al fatto che LLaDA2.1 usa T2T per rettificare errori introdotti da threshold più aggressivi. citeturn13view1turn0search6  

**Proto‑adattamento HILDA (FULL):**
- se già attivi CaDDi/PAPL, devi rivedere “chi possiede la traiettoria”: LLaDA2.1 già introduce un concetto di traiettoria di stato “editable”; sovrapporre un non‑Markovian memory e un path reweighting può creare doppi conteggi/instabilità se non delimitati con ablation clean. citeturn5search3turn17search4turn13view0  

#### Upgrade “core‑adjacent” ma ad alto impatto: Residual Context Diffusion (RCD)

Il problema che RCD attacca è molto concreto: nei block dLLM moderni, le policy di remasking scartano token “non abbastanza confidenti”, buttando via computazione. RCD propone di trasformare queste rappresentazioni scartate in **residui contestuali** reiniettati nello step successivo, migliorando accuracy e/o riducendo denoising steps, e con un training pipeline pensato per aggirare bottleneck di memoria. citeturn3search0turn3search4turn3search12  

**Proto‑adattamento HILDA (SAFE):**
- non devi cambiare la tua forward diffusion: puoi inserire RCD come **modulo di stato** nell’iterazione di denoise, limitandoti a “carry‑over residuals” tra step.
- compatibilità con ESPO: ESPO lavora con ELBO come proxy della likelihood sequence‑level; introdurre residuals che riciclano hidden states potrebbe cambiare lievemente la distribuzione del denoiser ma non necessariamente la struttura dell’ELBO, se resti dentro la stessa definizione di processo di corruzione. Il punto pratico è: serve un “ELBO audit” (stesso processo, stesso estimator) per evitare mismatch tra policy e likelihood proxy. citeturn0search1turn3search0  

**Proto‑adattamento HILDA (FULL):**
- RCD può diventare la tua alternativa “meno rischiosa” a CaDDi: ottieni memoria/riuso di informazione senza introdurre un vero non‑Markovian forward process. CaDDi punta a integrare traiettoria esplicita (“temporal trajectory”) nel framework non‑Markoviano. citeturn5search3turn3search0  

#### Upgrade alternativo per esperimenti su piccola compute: MDLM come core “laboratorio”

Per i tuoi test su GTX 1080, l’opzione più pragmatica è usare un core diffusion già pronto e piccolo, dove l’intera ingegneria è pensata per funzionare anche senza FlashAttention e spesso in fp32. MDLM (Masked Diffusion Language Models) è una baseline forte e “semplice” con SUBS parameterization e un obiettivo Rao‑Blackwellized a bassa varianza, e offre modelli pubblici “no_flashattn‑fp32‑owt” (~130M non‑embedding). citeturn16search1turn16search2turn16search5  

**Proto‑adattamento HILDA:**
- usa MDLM come “mini‑backbone” per validare: mascheramento, schedule, logging ELBO, pipeline RL‑compatibile (anche solo a scopo di sanity check), e soprattutto la tua infrastruttura di dataset/tokenizer/masking.
- poi, una volta stabile, “trasli” lo stesso scaffolding su un backbone LLaDA‑style (Gemma → conversione block diffusion), riducendo i gradi di libertà che cambiano simultaneamente.  
Questa scelta è direttamente giustificata anche dall’evidenza recente che i DLM possono essere sorprendentemente **data‑efficient** in regimi a dati unici limitati. citeturn4search3turn16search1  

#### Un altro aggiornamento “fresh” (ultimi giorni): scaling oltre l’MDLM classico

Nel giro di pochi giorni è uscito un lavoro di scaling laws su diffusion discreti oltre il masked diffusion dominante, mostrando – tra le altre cose – che un obiettivo a bassa varianza può spostare “compute‑optimal checkpoints” verso modelli più piccoli, riducendo costo d’inferenza e rendendo più sensato sperimentare con small models. citeturn16search0turn16search4turn16search8  
Per HILDA questo è rilevante perché giustifica una strategia “small‑first” senza rinunciare al rigore sperimentale.

## Miglioramenti fuori dal core

**Quest 2 richiesto:** potenziamenti “extra‑core” (reasoning, RL, inferenza/serving, decoding, KV, ecc.).

### Factcheck Perplexity e Gemini fuori dal core

**Perplexity (ricerca):** qui è sostanzialmente centrata. I principali elementi citati esistono e sono estremamente pertinenti a Stage‑2/Stage‑3:
- **KVzap** esiste e propone pruning adattivo del KV cache con target esplicito “engine‑adoption” e claim di compressione 2–4× con perdita minima, e si appoggia all’ecosistema KVpress. fileciteturn0file0 citeturn1search3turn20search1turn20search4  
- **Beyond Speedup: Utilizing KV Cache for Sampling and Reasoning** esiste e formalizza riuso del KV cache come rappresentazione per compiti post‑hoc (Chain‑of‑Embedding; fast/slow thinking switching). fileciteturn0file0 citeturn2search3turn2search7  
- **Order‑Token Search** esiste ed è un upgrade “decoding‑only” molto naturale per dLLM, perché esplora esplicitamente lo spazio delle traiettorie di ordine + token, migliorando risultati su reasoning/coding in modo consistente. fileciteturn0file0 citeturn1search2turn1search6  
- **R³L** e **LENS** esistono, con meccanismi utili per migliorare esplorazione/credit assignment e “pulizia dell’istruzione” in RLVR. fileciteturn0file0 citeturn2search0turn2search1  

**Gemini (ricerca):** sul “fuori core” Gemini riprende molte delle stesse idee, ma:
- tende a proporre un solo approccio dominante (es. D2F come “best”), mentre la letteratura recente indica che si sta consolidando un “menu” di opzioni (ibridi AR‑diffusion: D2F, Fast‑dLLM v2, CARD; e decoding search: Order‑Token Search; e editing: LLaDA2.1). citeturn5search1turn5search2turn1search1turn1search2turn13view1  

### Ricerca aggiornata e raccomandazioni fuori dal core

#### Stage 3: inferenza e serving su hardware piccolo

Qui conviene separare due dimensioni: (a) accelerazioni “dLLM‑native” e (b) ottimizzazioni di sistema/caching generiche.

1) **Ibridi AR‑diffusion e caching gerarchico**  
- **D2F**: propone un paradigma ibrido che abilita KV cache e parallelismo inter‑blocco tramite distillazione asimmetrica, con un’implementazione esplicita di trade‑off efficacia/efficienza. citeturn5search1turn5search9  
- **Fast‑dLLM v2**: formalizza block diffusion + maschere complementari + caching gerarchico (block‑level e sub‑block), e dichiara che l’adattamento da AR a dLLM può richiedere ~1B token di fine‑tuning (dato importante per te). citeturn5search2turn5search14  
- **CARD**: aggiunge un’idea operativa molto utile: generare più token per step quando la confidenza è alta e degradare verso sequenzialità quando serve, usando KV caching in modo “confidence‑adaptive”. citeturn1search9turn1search1  

2) **Decoding search per dLLM**  
Se hai un dLLM stabile ma “piatto” in reasoning, la leva più “pulita” (non cambia training) è potenziare la search:
- **Order‑Token Search** fornisce un framework per esplorare diverse traiettorie di generazione (ordine + token). È perfetto come plugin Stage‑3 nell’idea StreamLab che già hai menzionato. citeturn1search2turn1search6  
- Il punto chiave è: puoi testarlo anche su modelli piccoli come “proof of value” perché la differenza che misura è spesso **relativa** (migliora rispetto al tuo decoding baseline) più che assoluta.

3) **KV cache: compressione e offload**
- **KVzap** e l’ecosistema **KVpress** sono importanti perché avvicinano metodi di pruning a una forma riusabile in pipeline reali. citeturn1search3turn20search1turn20search4  
- Sul tuo vincolo VRAM, le feature già disponibili nelle “cache strategies” di entity["company","Hugging Face","ai platform company"] Transformers (cache offloaded su CPU, cache statica, ecc.) sono pragmatiche e immediatamente sfruttabili per inference AR (e spesso anche per componenti ibridi). citeturn20search0turn20search3turn20search10  

#### Stage 2: reasoning/RL oltre ESPO (senza perdere rigore ELBO)

Tu hai scelto ESPO come default e, ad oggi, è una scelta “principled”: nasce proprio dalla constatazione che token‑level RL classico è mal posto sui dLLM perché manca una factorization naturale della likelihood; ESPO sposta la view a livello sequenza usando l’ELBO come proxy. citeturn0search1turn0search5  

Gli upgrade più interessanti, nel tuo contesto (compute limitata + bisogno di stabilità), sono quelli che riducono varianza/costo dei gradienti e/o aumentano efficienza di rollout.

- **wd1**: riformula l’obiettivo RL come weighted log‑likelihood “ratio‑free” e introduce anche una variante step‑wise (wd1++). Questo è particolarmente interessante per te perché il costo di stimare ratio/likelihood multipli in dLLM è un killer su GPU piccola. citeturn17search2turn17search6  
- **AGRPO**: è esplicitamente progettato per dLLM, usa Monte Carlo per ottenere un policy gradient “faithful” e si posiziona come alternativa principled a diffu‑GRPO su compiti math/reasoning. Se vuoi esplorare RL “step‑aware” senza inventarti nuovi estimator, AGRPO è una delle opzioni più pulite. citeturn19search0turn19search4  
- **STP (Spatio‑Temporal Pruning)**: questo è un contributo freschissimo (9–10 febbraio 2026). Riduce ridondanza (spatial) e salta refinement tardivo (temporal) dichiarando anche una riduzione teorica della varianza della stima della log‑likelihood/ELBO e miglioramenti di efficienza. Per un setup come il tuo, è interessante perché “fa risparmiare compute” in modo allineato con la natura multi‑step del denoise. citeturn18view0turn18view1  
- **LENS**: è più “procedurale” che “algoritmica”: identifica token d’istruzione interferenti che abbattono la sampling success rate e propone un trasferimento dei rollout “purificati” per robustezza su prompt rumorosi. È utile per te come *data pipeline upgrade* per RLVR quando i rollout sono pochi e costosi. citeturn2search1turn2search5  
- **R³L**: se in futuro vuoi una modalità “agentica” (retry da failure points, pivotal credit assignment), R³L è un riferimento forte; però su GPU 1080 devi stare attento: il meccanismo di reflect‑then‑retry può esplodere il numero di forward pass se non lo limiti duramente o non lo combini con caching. citeturn2search0turn2search3  

## Unione dei concetti core e fuori‑core

**Quest 3 richiesto:** fusione dei concetti (1) e (2) in una proposta integrata, sempre divisa in factcheck e ricerca aggiornata.

### Factcheck Perplexity e Gemini sulla “sintesi” complessiva

**Perplexity** propone correttamente una “compatibility matrix” e una short‑list di priorità; come metodo di engineering è valido perché molte tecniche sono **non composabili** senza disciplina sperimentale (es. tokenizer differenziabile tipo Zonkey + ibridi D2F + alla fine RL). fileciteturn0file0 citeturn1search0turn5search1  

**Gemini** propone una sintesi concreta (SmolLM2 piccolo → conversione dLLM → C2DLM LoRA → ESPO) e, come direzione, è ragionevole: SmolLM2 esiste e documenta obiettivi/data mix “data‑centric” per small LMs. fileciteturn0file2 citeturn7search1turn6search2  
Il limite è che Gemini tende a sottostimare il valore di testare separatamente *decoding/search* (Order‑Token Search) e *RL efficiency tricks* (STP, LENS), che su hardware piccolo possono dare più segnale di un “upgrade core gigantesco”. citeturn1search2turn18view1turn2search1  

### Ricerca aggiornata: HILDA “vNext” come design modulare e testabile su GTX 1080

Propongo una **HILDA vNext** concettuale in due profili (SAFE e FULL), ma con un principio ingegneristico nuovo: **“core editabile + residual context + serving/decoding plug‑in + RL a varianza controllata”**, con una pipeline di ablation più “economica” per VRAM.

#### SAFE vNext: massima compatibilità, upgrade mirati

Backbone:
- passa a **LLaDA2.1‑style editing** come target di design (anche se non alleni *quel* 16B/100B, replichi la meccanica su small). Il punto non è avere il modello SOTA, ma avere un core che permette di spostare il trade‑off speed/quality via T2T. citeturn13view1turn13view0  
- integra **RCD** come “recycling residuals” nel loop di denoise per non buttare via computazione del remasking. citeturn3search0turn3search4  

Stage 1:
- mantieni SFT in stile d1 (masked SFT) ma aggiungi esempi che espongano la correzione T2T e, se usi C2DLM, applicala con priorità sulle passate di correzione. citeturn16search3turn0search6turn13view0  

Stage 2:
- baseline ESPO (coerente e già “principled”). citeturn0search1  
- upgrade opzionale leggero: se vuoi “risparmiare compute e stabilizzare”, sperimenta **wd1** (ratio‑free) oppure **STP** (pruning spatio‑temporale), ma uno alla volta. citeturn17search2turn18view1  
- come “data pipeline trick” per RLVR: integra LENS per aumentare success rate dei rollout quando i budget sono ridotti. citeturn2search1turn2search5  

Stage 3:
- non cambiare training semantics: qui i plug‑in migliori sono **Order‑Token Search** (quality) e **KV/cache strategies** (memory), più eventualmente un ibrido **D2F** quando vuoi speed. citeturn1search2turn5search1turn20search0  

#### FULL vNext: dove ha senso essere “invasivi” senza perdersi

Il FULL originale introduce CaDDi + PAPL + schedule dinamico. fileciteturn0file1 citeturn5search3turn17search4  
Il rischio in FULL oggi è accumulare troppi meccanismi che alterano traiettorie/ELBO e rendere impossibile attribuire miglioramenti.

Per rendere FULL “scientificamente testabile” su GPU piccola, suggerisco una gerarchia di invasività:

1) **Invasivo con alto ROI e bassa interazione**: RCD (perché non ridefinisce un forward non‑Markoviano; “ricicla” rappresentazioni). citeturn3search0  
2) **Invasivo ma con story ELBO più delicata**: CaDDi (non‑Markovian diffusion) e PAPL (path reweighting). Qui devi stabilire un protocollo di audit su ESPO/ELBO: se cambi la dinamica, devi riallineare l’estimator della likelihood proxy. citeturn5search3turn17search4turn0search1  
3) **Invasivo sulla tokenizzazione**: Zonkey (tokenizer differenziabile) lo terrei fuori dal FULL “mainline” e lo tratterei come terza linea sperimentale separata, perché cambia completamente dati, tokenizer, e probabilmente anche i tuoi test/benchmark devono essere reinterpretati. citeturn1search8turn1search4  

## Modelli open consigliati per test con GTX 1080 (sm_61, 8GB)

Qui separo i consigli per **(a) test di training/architettura** e **(b) inferenza/benchmarking**. L’obiettivo è darti modelli che:
- siano davvero utilizzabili con Pascal,
- abbiano tokenizer gestibile per masking/editing,
- non richiedano pretraining giganteschi per dare segnale sperimentale.

### Vincoli software/hardware che impattano la scelta del modello

- La compatibilità Pascal con build recenti di PyTorch è variabile: su GTX 1080 spesso la soluzione è usare build che includono esplicitamente sm_61 (tipicamente build con CUDA 12.6 secondo i thread ufficiali). citeturn10view2turn10view1  
- CUDA Toolkit 13.x ha rimosso supporto offline compilation/librerie per Maxwell/Pascal/Volta; questo aumenta il rischio che nuove estensioni CUDA non siano più buildabili/targettabili per sm_61. citeturn12view0  
- Le librerie di serving moderne tipo vLLM richiedono spesso compute capability ≥ 7.0, quindi non sono la strada primaria su Pascal. citeturn15search3turn15search6  
- bitsandbytes supporta GPU NVIDIA da Compute Capability 6.0+ per 8‑bit optimizers e 4‑bit quantization, mentre LLM.int8 “full” richiede 7.5+. Quindi su GTX 1080 hai accesso realistico a QLoRA/4‑bit e 8‑bit optimizer, ma non necessariamente a tutte le modalità int8. citeturn8view0  

### Modelli “AR piccoli” per conversione / Stage‑0 e sanity check

- **SmolLM2‑135M / 360M**: ottimi per testare pipeline, tokenizer, training loop, logging e piccoli benchmark; la famiglia è documentata e nasce come “small LMs” con attenzione alla qualità dei dati (anche se il modello forte è 1.7B). citeturn6search2turn7search1turn7search9  
- **Qwen2.5‑0.5B**: modello 0.5B con contesto ampio e architettura standard Transformer; è un buon “mid‑small” per vedere segnali su benchmark semplici senza andare su 2B+. citeturn7search0turn7search4  
- **Gemma 3 270M**: è esplicitamente pensata per deployment/finetuning “hyper‑efficient”, con varianti open weights e contesto fino a 32k per la taglia 270M; per te è interessante come base per esperimenti “tool/agent” e come target di conversione. citeturn6search4turn23search16turn6search1  

Nota pratica “mask token”: Gemma ha token inutilizzati riservati nel tokenizer (utile per assegnare un [MASK] senza riscalare embedding, o comunque ridurre il caos). citeturn23search0  

### Modelli “diffusion nativi” piccoli per testare dLLM senza conversione complessa

- **MDLM (~130M) no_flashattn‑fp32‑owt**: è probabilmente la miglior base “da laboratorio” per te: è diffusion masked, piccolo, e personalmente lo considero il candidato più realistico per convalidare rapidamente remasking/schedule/denoise loop su Pascal. citeturn16search2turn16search1  

### Inference su 8GB: cosa è realistico

- 1B–2B (quantizzati) è un range realista per inference locale; sopra, il costo KV cache esplode. Qui le strategie di cache (offload/quantized cache) diventano centrali. citeturn20search0turn20search3  
- Per runtime “semplice”, llama.cpp rimane spesso una via robusta su hardware vecchio, ma su Pascal può richiedere build con flag arch corretti e non offre le stesse API di sperimentazione training. citeturn15search2turn15search1  

In generale, su GTX 1080 la roadmap più stabile è: training/finetuning con PyTorch + Transformers (pinned), inference con Transformers (o llama.cpp per testing rapido), evitando stack che richiedono compute capability ≥7.0.

## Dataset e benchmark “small‑first” per misurare miglioramenti HILDA

Tu chiedi esplicitamente dataset e benchmark che:
- siano utilizzabili anche con modelli small,
- ti permettano di misurare miglioramenti/peggioramenti dell’architettura,
- non richiedano RL gigantico o pretraining enorme per vedere segnale.

Propongo una griglia di valutazione coerente con la tua pipeline (Stage0‑1‑2‑3).

### Stage 0 / Stage 1: qualità linguistica, controllabilità, stabilità del denoise

- **TinyStories**: è *perfetta* per vedere rapidamente se l’architettura mantiene grammatica e coerenza con modelli 100M‑scale; nasce proprio per far emergere competenze linguistiche/coerenza su modelli piccoli. citeturn21search0turn21search4  
- Aggiunta utile (se vuoi un filo più “reale”): piccoli slice di web‑text (es. OWT) sono ok, ma per te il punto è misurare differenze relative tra varianti (LLaDA2.1‑style editing vs baseline; RCD vs no‑RCD), non raggiungere SOTA.

Metriche consigliate (per test scientifici su small):  
- loss/perplexity *intra‑famiglia* (stesso core), sapendo che across diffusion families la perplexity può essere un indicatore fuorviante; questa cautela è sottolineata anche da lavori di scaling recenti su diffusion LM. citeturn16search8turn16search0  

### Stage 2: RLVR con reward verificabile e budget ridotto

Qui serve verifiable reward “chiuso” per evitare reward models pesanti.

- **GSM8K**: benchmark standard di math word problems con risposte verificabili; ottimo per RLVR/ESPO/wd1/AGRPO anche su subset ridotti, perché l’accuracy è un segnale chiaro e comparabile. citeturn21search1turn21search5  
- **MATH** (anche solo subset tipo MATH500): più duro; utile quando vuoi stressare davvero la qualità del reasoning, ma per modelli molto piccoli potresti usarlo solo come “direzione” (tendenza). citeturn21search2turn21search10  

Collegamento diretto al tuo stack RL dLLM:  
- d1 formalizza SFT+RL per reasoning nei masked dLLM e introduce diffu‑GRPO. citeturn16search3turn16search7  
- ESPO formalizza la view sequence‑level ELBO‑based per RL e mostra guadagni marcati su task come Countdown/Math/Coding. citeturn0search1  
- AGRPO è esplicitamente “principled policy gradients per dLLM”. citeturn19search0  
- STP è un upgrade freschissimo che mira a ridurre ridondanza e varianza, cioè esattamente ciò che rende RL su 1080 difficile. citeturn18view1  

### Stage 3: benchmark per serving/decoding e qualità sotto accelerazione

Per misurare “inferenza intelligente” (non solo qualità statica), serve una coppia: benchmark + misure di throughput/latency.

- **HumanEval**: anche se è un benchmark nato in un paper su Codex, è rimasto uno standard per functional correctness nel code generation. Per diffusion coders, puoi usare sia HumanEval sia benchmark più recenti (se hai risorse), ma HumanEval è già sufficiente per differenze relative. citeturn21search15  
- **ARC (Challenge/Easy)**, **HellaSwag**, **MMLU**, **BBH**: li userei soprattutto quando sali almeno a 0.5B–1.7B e vuoi una vista “broad”; per piccoli (≤360M) possono essere troppo duri, ma su subset danno comunque tendenze. citeturn22search3turn22search0turn22search1turn22search2  

Per il tuo caso, l’asse più importante in Stage‑3 è:  
- baseline decoding  
- vs Order‑Token Search (quality) citeturn1search2  
- vs D2F / Fast‑dLLM v2 / CARD (speed) citeturn5search1turn5search2turn1search9  
misurando qualità *e* costo (passi di denoise, step di editing, cache footprint).

### Nota finale “small models e dataset”: cosa è realistico aspettarsi

La tua critica (“modelli troppo piccoli non fanno benchmark seri”) è spesso vera per benchmark generalisti; però la letteratura recente sui diffusion LMs indica due cose che rendono sensati gli esperimenti small‑first:

- i DLM possono superare AR di pari taglia quando la quantità di dati unici è limitata, semplicemente “overtraining” in modo controllato. citeturn4search3  
- alcune conversioni/upgrade (Fast‑dLLM v2, RCD) dichiarano adattamenti efficaci con budget dell’ordine di ~1B token, che è più vicino a regimi sperimentali reali rispetto ai pretraining da centinaia di miliardi/trilioni di token. citeturn5search2turn3search0  

Se l’obiettivo è **testare HILDA**, la metrica corretta non è “punteggio assoluto su MMLU con 135M”, ma: *a parità di compute e dati, la variante A batte la variante B?* — e in questo senso TinyStories+GSM8K(+subset) è una combinazione sorprendentemente informativa. citeturn21search0turn21search1turn4search3
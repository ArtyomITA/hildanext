# HildaNext - Riferimento completo dell'interfaccia grafica

## Scopo del documento
Questo file descrive tutta la UI frontend presente nel repository `hildanext/frontend`, divisa per pagine, con due prospettive per ogni sezione:
- **Cosa serve**: funzione pratica del pannello o del blocco UI.
- **Come e' costruito**: componenti React, store, hook, endpoint backend e librerie usate.

Il documento include:
- le **pagine attive** realmente esposte dal router;
- la **pagina legacy** ancora presente nel codice ma oggi non piu' instradata;
- i **componenti trasversali** condivisi da piu' pagine.

---

## 1. Stack tecnico della GUI

### 1.1 Bootstrap frontend
File principali:
- `frontend/src/main.tsx`
- `frontend/src/app/App.tsx`
- `frontend/src/app/router.tsx`
- `frontend/src/app/providers.tsx`

Tecnologia usata:
- **Vite** come dev server e build tool.
- **React 19** per la UI.
- **React Router 7** per routing e redirect.
- **Zustand** per gli store client-side.
- **CSS Modules** per gli stili per-componente.
- **uPlot** per i grafici ad alte prestazioni.
- **@tanstack/react-virtual** per log e transcript lunghi.
- **Web Worker** per filtrare i log senza bloccare il thread UI.

Come parte:
1. `main.tsx` monta `App`.
2. `App.tsx` avvolge il router in `AppProviders`.
3. `AppProviders` oggi e' volutamente vuoto: e' un placeholder per futuri provider globali.
4. `router.tsx` definisce le pagine vere dell'app.

### 1.2 Rotte reali del router
File: `frontend/src/app/router.tsx`

| Rotta | Stato | Componente reale |
|---|---|---|
| `/` | redirect | `/chat` |
| `/chat` | attiva | `ChatPage` |
| `/inference` | redirect | `/chat` |
| `/benchmark` | attiva | `BenchmarkPage` |
| `/legacy/wsd` | attiva | `WsdPage` |
| `/wsd` | redirect | `/legacy/wsd` |

Nota importante:
- **`InferencePage.tsx` esiste ancora nel codice**, ma non e' piu' raggiungibile dal router.
- **`Stage0Validation`** compare sia nella pagina `/benchmark` sia dentro `/legacy/wsd`.

### 1.3 Shell globale
File:
- `frontend/src/shell/AppShell.tsx`
- `frontend/src/shell/TopNav.tsx`

#### AppShell
**Cosa serve**
- E' il contenitore esterno comune a tutte le pagine.
- Applica backdrop, area `main` e animazione `fade-in`.

**Come e' costruito**
- Usa `useLocation()` di React Router per leggere il path corrente.
- Renderizza:
  - sfondo decorativo;
  - `TopNav`;
  - `<Outlet />` per la pagina attiva.

#### TopNav
**Cosa serve**
- Permette di navigare fra aree funzionali:
  - Inferenza
  - Benchmark
  - WSD Legacy

**Come e' costruito**
- Usa `NavLink` e `clsx`.
- Evidenzia la tab attiva con logica custom basata su `pathname.startsWith(...)`.
- Le voci correnti sono:
  - `Inferenza` -> `/chat`
  - `Benchmark` -> `/benchmark`
  - `WSD Legacy` -> `/legacy/wsd`

### 1.4 Stile globale
File:
- `frontend/src/styles/tokens.css`
- `frontend/src/styles/globals.css`
- `frontend/src/styles/motion.css`

**Cosa serve**
- Definisce identita' visiva comune dell'app.

**Come e' costruito**
- `tokens.css` definisce variabili CSS:
  - palette scura tecnica;
  - tipografia (`Space Grotesk`, `IBM Plex Sans`, `IBM Plex Mono`);
  - spaziature e raggi.
- `globals.css` applica:
  - sfondo a gradienti;
  - griglia di fondo;
  - font globali;
  - focus visibile.
- `motion.css` definisce classi di animazione:
  - `fade-in`
  - `slide-up`

---

## 2. Componenti UI condivisi

### 2.1 `Panel`
File: `frontend/src/components/layout/Panel.tsx`

**Cosa serve**
- E' il mattone base di quasi tutta la UI.
- Standardizza header, kicker, titolo, azioni e contenuto.

**Come e' costruito**
- Wrapper React molto semplice.
- Se presenti, mostra:
  - `kicker`
  - `title`
  - `actions`
- Viene riutilizzato in WSD, benchmark, inferenza legacy, glossary, insights, process rail.

### 2.2 `DataSourceBar`
File: `frontend/src/components/layout/DataSourceBar.tsx`

**Cosa serve**
- Dice chiaramente **che tipo di dati** l'utente sta guardando:
  - live
  - mockup/offline
  - missing

**Come e' costruito**
- Mostra una pill visiva:
  - `● LIVE`
  - `◎ OFFLINE MOCKUP`
  - `✕ DATA MISSING`
- Accetta una lista `items` label/value.
- Se la sorgente e' `missing`, mostra anche un `hint`.

### 2.3 `MetricHeroCard`
File: `frontend/src/components/cards/MetricHeroCard.tsx`

**Cosa serve**
- Mostra KPI sintetici in alto pagina.

**Come e' costruito**
- Card minimale con:
  - `label`
  - `value`
  - `meta`
  - `accent` (`cyan`, `lime`, `orange`, `red`)

### 2.4 `SeverityBadge`
File: `frontend/src/components/badges/SeverityBadge.tsx`

**Cosa serve**
- Evidenzia severita' o tono di un evento.

**Come e' costruito**
- Badge con classi CSS per:
  - `info`
  - `notice`
  - `warning`
  - `error`

### 2.5 Tabelle e transcript virtualizzati
File:
- `frontend/src/components/tables/VirtualLogTable.tsx`
- `frontend/src/components/terminals/TerminalTranscript.tsx`

**Cosa serve**
- Visualizzare grandi volumi di log senza rallentare l'interfaccia.

**Come e' costruito**
- Entrambi usano `@tanstack/react-virtual`.
- `VirtualLogTable`:
  - rende righe cliccabili;
  - mostra timestamp, titolo principale e badge severita';
  - serve per log strutturati.
- `TerminalTranscript`:
  - imita un transcript console;
  - auto-scrolla in fondo quando arrivano nuove righe;
  - serve per output lunghi stile CMD/PowerShell.

### 2.6 Grafici
File:
- `frontend/src/components/charts/TimeseriesChart.tsx`
- `frontend/src/components/cards/PhaseTimeline.tsx`
- `frontend/src/components/cards/TBucketHeatStrip.tsx`

**Cosa serve**
- Dare lettura immediata di andamento, fasi e bucket diagnostici.

**Come e' costruito**
- `TimeseriesChart` usa **uPlot**:
  - costruisce `Float64Array`;
  - usa `ResizeObserver`;
  - non usa SVG React, quindi regge meglio dataset piu' grandi.
- `PhaseTimeline`:
  - segmenta `warmup`, `stable`, `decay`;
  - usa metriche reali se disponibili, altrimenti i piani di config.
- `TBucketHeatStrip`:
  - mostra `lossByTBucket` e `accMaskedByTBucket` su bucket di tempo/rumore.

### 2.7 Filtri log in background
File:
- `frontend/src/features/logs/useLogFeed.ts`
- `frontend/src/workers/logWorker.ts`

**Cosa serve**
- Filtrare i log per query, severity e source senza bloccare il rendering.

**Come e' costruito**
- `useLogFeed` crea un `Worker`.
- Il worker riceve due messaggi:
  - `load`
  - `filter`
- Il filtraggio usa:
  - query testuale;
  - set di livelli;
  - set di sorgenti.
- C'e' anche un limite di sicurezza:
  - `MAX_VISIBLE = 10000`

---

## 3. Pagina Chat - `/chat`
File principale: `frontend/src/routes/chat/ChatPage.tsx`

### 3.1 Scopo della pagina
Questa e' la pagina principale attuale dell'app. Serve a fare **inferenza chat-first** con una o due lane:
- **AR**
- **dLLM**
- **BOTH**

La pagina combina:
- gestione thread chat locali;
- configurazione inferenza;
- warmup/caricamento pesi;
- visualizzazione risposte lane-per-lane;
- log realtime inferenza.

### 3.2 Architettura tecnica della pagina
Elementi chiave:
- Stato locale React per prompt, modal raw, stato load pesi e log realtime.
- `useChatStore` (Zustand) per:
  - thread;
  - config;
  - preset;
  - persistenza `localStorage`.
- `composeChatInput(...)` per preparare cronologia + system prompt entro il budget token.
- `runChatTurn(...)` per chiamare backend inferenza.
- `EventSource` per stream log live con fallback a polling HTTP.

### 3.3 Colonna sinistra - Sidebar thread
#### Header `Thread`
**Cosa serve**
- Fa da mini file manager delle conversazioni.
- Permette di aprire subito una nuova chat.

**Come e' costruito**
- Bottone `+ nuova chat`.
- Chiama `createThread()` dello store.

#### Campo `Filtra thread...`
**Cosa serve**
- Filtra la lista delle chat per titolo o testo prompt.

**Come e' costruito**
- Valore nello store `filter`.
- `filteredThreads` usa `useMemo` e controlla:
  - `thread.title`
  - `turn.prompt`

#### Lista thread
**Cosa serve**
- Mostra le conversazioni esistenti con anteprima e data di aggiornamento.

**Come e' costruito**
- Ogni riga e' il componente interno `ThreadRow`.
- Ogni riga mostra:
  - titolo;
  - orario `updatedAt`;
  - preview ultimo prompt;
  - numero turni;
  - azioni `rinomina` / `elimina`.
- `rename` usa `window.prompt`.
- `delete` usa `window.confirm`.

### 3.4 Area centrale - Timeline conversazione
#### Header `Chat-First Inference Studio`
**Cosa serve**
- Dichiara lo stato operativo globale della pagina.

**Come e' costruito**
- Mostra tre badge testuali:
  - `engineMode`
  - `weights ready/missing`
  - `running/ready`

#### Lista turni chat
**Cosa serve**
- Visualizza la cronologia dell'interazione utente-assistente.

**Come e' costruito**
- Ogni turno e' una `turnCard`.
- La card contiene:
  - bubble utente;
  - blocco assistant;
  - una o due lane card.
- Se il turno e' in corso:
  - mostra `Esecuzione lane in corso...`

#### Header assistant per singolo turno
**Cosa serve**
- Riassume con che impostazioni e' stato prodotto quel turno.

**Come e' costruito**
- Deriva `turnConfig` facendo merge fra config salvata nel turno e `DEFAULT_CHAT_CONFIG`.
- Mostra:
  - engine mode;
  - decode mode;
  - effort;
  - thinking on/off/auto;
  - system prompt on/off.

### 3.5 Lane card AR / dLLM
Componente interno: `LaneCard`

**Cosa serve**
- E' il cuore della pagina: visualizza il risultato di una lane.

**Come e' costruito**
- Supporta tre stati:
  - `success`
  - `offline`
  - `error`
- In caso di successo mostra:
  - lane (`AR` o `dLLM`);
  - modello selezionato;
  - engine backend;
  - testo risposta;
  - badge di warning/info;
  - metriche tecniche.

Dettagli tecnici importanti:
- Se presente `rawText`, prova a separare il **thinking** dall'**answer**.
- I pattern riconosciuti sono:
  - tag `<think>...</think>`
  - tag Qwen `<|begin_of_thought|> ... <|end_of_thought|>`
- Metriche mostrate:
  - `finishReason`
  - `tokensPerSec`
  - `stepsToConverge`
  - `vramPeakBytes`
  - `dtype`
  - `device`
- Flag mostrati:
  - output troncato;
  - CPU fallback;
  - sampling params ignorati.

### 3.6 Modal `Raw risposta`
**Cosa serve**
- Permette di ispezionare il payload completo di una lane riuscita.

**Come e' costruito**
- Si apre cliccando una lane card di successo.
- E' chiudibile:
  - con click overlay;
  - con bottone `Chiudi`;
  - con tasto `Escape`.
- Mostra:
  - `rawText`
  - `rawStats` JSON

### 3.7 Composer in basso
**Cosa serve**
- Inserimento del nuovo prompt da inviare.

**Come e' costruito**
- `textarea` + bottone `Invia`.
- Shortcut:
  - `Ctrl+Invio` o `Cmd+Invio`.
- Il composer viene bloccato se:
  - inferenza in corso;
  - load pesi in corso;
  - lane richieste non ancora caricate.

Dettaglio tecnico:
- `handleSend()`:
  1. pulisce il prompt;
  2. usa `composeChatInput(...)`;
  3. garantisce l'esistenza del thread (`ensureThread()`);
  4. crea un `ChatTurn` con stato `running`;
  5. chiama `runChatTurn(...)`;
  6. aggiorna il turno con i `LaneResult`.

### 3.8 Pannello destro - `Load Weights`
**Cosa serve**
- Carica in anticipo i modelli/lane necessari prima dell'uso in chat.

**Come e' costruito**
- Mostra lo stato lane `AR` e `dLLM`:
  - `idle`
  - `loading`
  - `loaded`
  - `offline`
  - `error`
- `handleLoadWeights()` usa questa strategia:
  1. prova `POST /api/inference/unload`
  2. prova `POST /api/inference/load`
  3. se fallisce, esegue fallback con `runChatTurn("__hildanext_load_weights__")`

Dettaglio importante:
- Se il backend risponde con `device=cpu` o senza VRAM, la UI marca la lane come **CPU fallback attivo**.

### 3.9 Pannello destro - `Inference Realtime Log`
**Cosa serve**
- Fa da telemetria viva dell'inferenza backend.

**Come e' costruito**
- Prova prima uno stream **SSE**:
  - `GET /api/inference/logs/stream`
- Se SSE cade o non esiste, passa a polling:
  - `GET /api/inference/logs?tail=200&after_id=...`
- Mantiene massimo `500` righe in memoria.
- Ogni evento mostra:
  - orario;
  - lane o `SYS`;
  - nome evento;
  - messaggio;
  - dettagli `meta` in `<details>`.

### 3.10 Pannello destro - `Config base`
**Cosa serve**
- Espone i parametri principali dell'inferenza.

**Come e' costruito**
- Usa `updateConfig(...)` dello store.
- Campi mostrati:
  - `Engine mode`
  - `Modello AR`
  - `Modello dLLM`
  - `Max new tokens`
  - `Seed`

Modelli disponibili nel catalogo:
- `ar_qwen3_0_6b`
- `dllm_hilda_default`

### 3.11 Pannello destro - `Mostra avanzate`
**Cosa serve**
- Espone i parametri tecnici piu' fini della generazione.

**Come e' costruito**
- Stato persistente `advancedOpen` nello store.
- Campi presenti:
  - `Decode mode`
  - `Thinking mode`
  - `AR decode mode`
  - `System prompt`
  - `Context window tokens`
  - `Effort`
  - `Tau mask`
  - `Tau edit`
  - `Temperature`
  - `Top-p`
  - `Top-k`
  - `Presence penalty`
  - `Repetition penalty`

Comportamento tecnico:
- Alcuni cambi aggiornano automaticamente altri campi:
  - `Decode mode`
  - `Thinking mode`
- Se `engineMode === "AR"` e `arDecodeMode === "greedy"`, alcuni sampling params vengono disabilitati.

### 3.12 Pannello destro - `Shadow Layer Terminologia`
**Cosa serve**
- Spiega rapidamente concetti diffusion/decoding senza uscire dalla pagina.

**Come e' costruito**
- `<details>` con mini-card statiche.
- Termini inclusi:
  - `S_MODE`
  - `Q_MODE`
  - `tau_mask`
  - `tau_edit`
  - `effort`

### 3.13 Pannello destro - `Preset locali`
**Cosa serve**
- Salva e riapplica configurazioni ricorrenti.

**Come e' costruito**
- Persistiti in `localStorage` tramite `useChatStore`.
- Azioni:
  - `Salva`
  - `Applica`
  - `X` per eliminazione

### 3.14 Persistenza e store della pagina chat
File:
- `frontend/src/store/chatStore.ts`
- `frontend/src/features/chat/storage.ts`
- `frontend/src/features/chat/catalog.ts`

**Cosa serve**
- Mantenere la chat anche dopo refresh browser.

**Come e' costruito**
- Chiavi storage:
  - `hildanext.chat_studio.v2`
  - compatibilita' legacy con `v1`
- Persistiti:
  - thread
  - selected thread
  - preset
  - ultima config
  - apertura pannello avanzato

### 3.15 Composizione del prompt e chiamate backend
File:
- `frontend/src/features/chat/promptComposer.ts`
- `frontend/src/features/chat/orchestrator.ts`

**Cosa serve**
- Convertire la cronologia UI in payload backend.

**Come e' costruito**
- `composeChatInput(...)`:
  - stima token in modo euristico;
  - rispetta `contextWindowTokens`;
  - include `systemPrompt`;
  - seleziona solo la storia che entra nel budget;
  - in modalita' `BOTH`, se esistono due risposte precedenti, le concatena come:
    - `[AR]`
    - `[dLLM]`
- `runChatTurn(...)`:
  - `POST /api/generate/ar` per lane AR
  - `POST /api/generate` per lane dLLM
  - in modalita' BOTH usa `Promise.allSettled`

---

## 4. Pagina Benchmark - `/benchmark`
File pagina: `frontend/src/routes/benchmark/BenchmarkPage.tsx`

Componente reale usato:
- `frontend/src/features/stage0/Stage0Validation.tsx`

### 4.1 Scopo della pagina
Questa pagina e' dedicata alla **validazione Stage 0**. Serve a misurare rapidamente se il sistema regge benchmark base e stability check prima di passare a fasi successive.

La pagina contiene un solo grande pannello, ma internamente e' molto ricco.

### 4.2 Struttura tecnica generale
La pagina usa soprattutto:
- `useState` intensivo per ogni benchmark;
- `useMemo` per score e grafico stability;
- `AbortController` per fermare run in corso;
- log interni per benchmark;
- chiamate backend dedicate sotto `/api/stage0/validate/*`.

Vincoli operativi:
- un benchmark per volta (`busy`);
- pause/resume cooperative;
- stop esplicito con abort request.

### 4.3 Pannello `Global Evaluation Settings`
**Cosa serve**
- Definisce i parametri comuni a tutte le run benchmark.

**Come e' costruito**
- Campi:
  - `Model Scope`: `AR`, `DLLM`, `BOTH`
  - `Context Window`
  - `Generation Effort`
  - `Decoding Strategy`
- Toggle aggiuntivi:
  - `Log dettagliati (Q/A + thinking)`
  - `Salva log dettagliato su file (backend)`
- Campo:
  - `Run Label (file)`
- Bottone:
  - `Mostra/Nascondi Scoreboard`

Dettagli tecnici:
- `Generation Effort` mappa internamente a `max_new_tokens`:
  - `low -> 256`
  - `medium -> 1024`
  - `high -> 2048`
- `Decoding Strategy` imposta:
  - `greedy -> temperature 0.0, top_p 1.0`
  - `sampling -> temperature 0.6, top_p 0.9`

### 4.4 Pannello opzionale `Scoreboard`
**Cosa serve**
- Storico locale delle run benchmark eseguite in quella sessione pagina.

**Come e' costruito**
- Mostra una tabella con:
  - benchmark
  - modello/scope
  - effort
  - decode
  - punteggio
  - data
  - subset/dataset
  - tokens
  - tempo
- Supporta filtro per benchmark.
- Supporta `Reset`.

### 4.5 Card `HellaSwag (Zero-Shot)`
**Cosa serve**
- Verifica retention/commonsense su subset HellaSwag.

**Come e' costruito**
- Pulsanti:
  - `Run HellaSwag`
  - `Pause/Resume`
  - `Stop`
- Settings:
  - `N-Shots`: `0`, `3`, `5`
- UI risultati:
  - progress bar
  - gauge vs baseline
  - meta dataset
  - log realtime benchmark

Dettaglio tecnico:
- Baseline visuale: `45%`
- Subset limit: `8`
- Endpoint usati:
  - `GET /api/stage0/validate/hellaswag/items`
  - `POST /api/stage0/validate/hellaswag-item`

### 4.6 Card `MMLU-Pro (Chain-of-Thought)`
**Cosa serve**
- Valuta reasoning e knowledge multi-opzione.

**Come e' costruito**
- Pulsanti e log come HellaSwag.
- Settings:
  - `N-Shots`: `0`, `5`
  - `Force CoT (Thinking)` checkbox
- UI risultati:
  - progress
  - gauge vs baseline
  - meta dataset/split
  - log realtime

Dettaglio tecnico:
- Baseline visuale: `24.7%`
- Subset limit: `150`
- Endpoint usati:
  - `GET /api/stage0/validate/mmlu-pro/items`
  - `POST /api/stage0/validate/mmlu-pro-item`
- Le run item usano `mode: "S_MODE"`.

### 4.7 Card `GSM8K (Math Reasoning)`
**Cosa serve**
- Valuta ragionamento matematico con exact match del numero finale.

**Come e' costruito**
- Settings:
  - `N-Shots`: `0`, `4`, `8`
- UI risultati:
  - progress
  - gauge
  - meta dataset/split
  - log realtime

Dettaglio tecnico:
- Baseline visuale: `59.6%`
- Subset limit: `100`
- Endpoint usati:
  - `GET /api/stage0/validate/gsm8k/items`
  - `POST /api/stage0/validate/gsm8k-item`

### 4.8 Card `Denoising Stability`
**Cosa serve**
- Controlla se il denoising diffusion evolve nella direzione attesa.

**Come e' costruito**
- Pulsante `Run Stability Check`
- Settings:
  - `Total Denoising Steps` slider `10-100`
  - `Mask Schedule`: `linear` o `cosine`
- Campo prompt personalizzabile.
- Output:
  - grafico `TimeseriesChart` su `mean_confidence`
  - testo finale generato
  - log realtime

Dettagli tecnici:
- E' **disabilitato se `Model Scope = AR`**.
- Usa `mode: "S_MODE"`.
- Endpoint:
  - `POST /api/stage0/validate/stability`

### 4.9 Logging benchmark dettagliato su file
**Cosa serve**
- Salvare un log backend piu' ricco per auditing o debugging.

**Come e' costruito**
- Apertura file:
  - `POST /api/stage0/validate/log/start`
- Chiusura file:
  - `POST /api/stage0/validate/log/finish`
- La UI memorizza `lastDetailedLogPath`.

### 4.10 Log realtime benchmark
**Cosa serve**
- Spiegare passo-passo cosa sta succedendo durante una run.

**Come e' costruito**
- Ogni benchmark ha la propria lista `BenchLogEntry`.
- I log hanno livelli:
  - `info`
  - `ok`
  - `warn`
  - `error`
- Se `detailedBenchLogs` e' attivo, i log includono anche:
  - domanda;
  - opzioni;
  - target;
  - thinking+answer troncato.

---

## 5. Pagina WSD Legacy - `/legacy/wsd`
File principale: `frontend/src/routes/wsd/WsdPage.tsx`

### 5.1 Scopo della pagina
Questa e' la pagina di osservabilita' WSD/training piu' completa del progetto. Serve a leggere:
- stato run;
- fasi warmup/stable/decay;
- metriche;
- transcript console;
- log strutturati;
- processi;
- insight e glossario.

### 5.2 Sorgente dati della pagina
Store e adapter:
- `frontend/src/store/dataStore.ts`
- `frontend/src/domain/adapters.ts`
- `frontend/src/domain/backendAdapter.ts`

**Cosa serve**
- Ottenere il pacchetto WSD normalizzato da backend.

**Come e' costruito**
- `WsdPage` chiama `setWsdScenario("live_wsd_run")`.
- L'adapter attivo e' sempre `backendAdapter`.
- Endpoint principale:
  - `GET /api/frontend/wsd`
- Se il backend WSD non risponde, lo store puo' ripiegare su mockup offline in altri flussi.

### 5.3 `DataSourceBar`
**Cosa serve**
- Fa capire subito se i dati WSD sono live o no.

**Come e' costruito**
- Mostra:
  - `run`
  - `steps`
  - `phase`
  - `optimizer`
  - `logs`
- Se manca la sorgente, suggerisce il comando backend da lanciare.

### 5.4 Hero metrics
**Cosa serve**
- Dare una fotografia immediata del run.

**Come e' costruito**
- Quattro `MetricHeroCard`:
  1. `Current phase`
  2. `Masked token acc`
  3. `Throughput`
  4. `Peak VRAM`
- Ogni card legge `latest = wsd.metrics.at(-1)`.

### 5.5 `RunControlPanel`
File:
- `frontend/src/features/run/RunControlPanel.tsx`
- `frontend/src/features/run/useRunStatus.ts`

**Cosa serve**
- Avviare o fermare un run WSD direttamente dal browser.

**Come e' costruito**
- Bottoni:
  - `wsd-log-test`
  - `full wsd`
  - `stop`
- Stato visuale:
  - `idle`
  - `running`
  - `done`
  - `error`
  - `stopped`
- Sotto c'e' un log tail auto-scrollante.

Dettagli tecnici:
- `useRunStatus` polla:
  - `GET /api/run/status`
- Avvio run:
  - `POST /api/run/start`
- Stop run:
  - `POST /api/run/stop`
- Il polling ha backoff esponenziale da `2s` a `30s`.
- Quando il run termina, richiama `setWsdScenario("live_wsd_run")` per ricaricare le metriche.

### 5.6 `Stage0Validation` embedded
**Cosa serve**
- Permette di lanciare benchmark direttamente dalla stessa schermata WSD.

**Come e' costruito**
- E' lo stesso identico componente della pagina `/benchmark`.
- Quindi la stessa UI benchmark e' riusata in due posti:
  - pagina dedicata benchmark;
  - pagina WSD legacy.

### 5.7 Pannello `Warmup -> stable -> decay`
**Cosa serve**
- Far leggere rapidamente il calendario di fase del training/run.

**Come e' costruito**
- Usa `PhaseTimeline`.
- Se le metriche reali esistono, le larghezze dei segmenti derivano dalle righe loggate.
- Se non esistono ancora, usa i valori pianificati in `wsd.meta`.

### 5.8 Pannello `Loss, throughput and VRAM in one viewport`
**Cosa serve**
- Correlare in un colpo solo loss, memoria e accuratezza.

**Come e' costruito**
- Usa `TimeseriesChart` con tre serie:
  - `loss`
  - `vram`
  - `mta` (masked token acc scalata x1000)

### 5.9 Pannello `t-bucket analysis`
**Cosa serve**
- Leggere la diagnostica a bucket di tempo/rumore del diffusion training.

**Come e' costruito**
- Usa `TBucketHeatStrip`.
- Mostra per ogni bucket:
  - range bucket;
  - loss;
  - accuracy masked.

### 5.10 Pannello `CMD transcript`
**Cosa serve**
- Leggere il flusso console come se fosse un terminale vero.

**Come e' costruito**
- Usa `TerminalTranscript`.
- Filtra solo log di source:
  - `console`
  - `training`
- Mostra `pinnedTag` statico:
  - `RUN_START / PHASE_CHANGE / OOM`

### 5.11 Pannello `Structured logs`
**Cosa serve**
- Esplorare i log normalizzati, non solo la console grezza.

**Come e' costruito**
- Barra `StickyFilterBar`:
  - query testuale;
  - livelli `notice`, `warning`, `error`;
  - sorgenti `console`, `metric`, `fallback`, `training`, `eval`
- `useSearchParams` salva i filtri nella URL.
- `useLogFeed` filtra nel worker.
- `VirtualLogTable` mostra le righe visibili.

Nota tecnica:
- La selezione log viene salvata anche in `useUiStore` (`selectedLogId`).

### 5.12 Colonna destra - `StatusRail`
File: `frontend/src/shell/StatusRail.tsx`

**Cosa serve**
- Riassume in poco spazio la postura generale del run WSD.

**Come e' costruito**
- Mostra:
  - `Phase`
  - `VRAM ceiling`
  - `Masked acc`
  - `Fallback heat`
- Ha anche una seconda sezione con obiettivi della pagina:
  - lettura transcript lunghi;
  - anticipazione phase change;
  - tracking saturazione VRAM.

### 5.13 Colonna destra - `ProcessRail`
**Cosa serve**
- Mostrare l'impatto dei processi sul sistema.

**Come e' costruito**
- Riassunto del sample piu' recente:
  - GPU VRAM
  - GPU util
  - System RAM
- Tabella ultimi 8 snapshot processo:
  - nome processo
  - VRAM
  - status

### 5.14 Colonna destra - `InsightCallout`
**Cosa serve**
- Dire all'utente dove guardare prima.

**Come e' costruito**
- Renderizza `InsightCard[]`.
- Ogni insight mostra:
  - titolo;
  - metrica;
  - tono con `SeverityBadge`;
  - spiegazione sintetica.

### 5.15 Colonna destra - `GlossaryInspector`
**Cosa serve**
- Evitare che la UI usi termini vaghi o mal interpretati.

**Come e' costruito**
- Tabs interne sui termini del glossario.
- Mostra per ogni termine:
  - frase inglese breve;
  - hint in italiano;
  - spiegazione operativa.

Termini presi da:
- `frontend/src/domain/glossary.ts`

### 5.16 Stato UI della pagina WSD
Store usati:
- `useDataStore`
- `useUiStore`

**Cosa serve**
- Separare stato dati e stato puramente visuale.

**Come e' costruito**
- `useDataStore` gestisce:
  - `wsd`
  - `inference`
  - availability backend
- `useUiStore` gestisce:
  - densita'
  - followTail
  - pause
  - `selectedLogId`
  - `selectedStep`

Nota:
- Alcune API dello store, come `startPolling()`, sono pronte ma non sono il flusso principale attivo della pagina WSD attuale.

---

## 6. Pagina Inference Legacy - componente presente ma non instradato
File principale: `frontend/src/routes/inference/InferencePage.tsx`

### 6.1 Stato attuale
Questa pagina e' ancora codificata e completa, ma **il router oggi non la espone**:
- `/inference` -> redirect a `/chat`

Quindi va letta come **UI legacy o prototipo ancora mantenuto nel codice**.

### 6.2 Scopo della pagina
- Confrontare direttamente lane **AR** e **diffusion**.
- Generare scenari mock interattivi.
- Visualizzare step diffusion e token state replay.

### 6.3 `DataSourceBar` e hero cards
**Cosa serve**
- Mostrare se i dati arrivano da scenario mock, scenario interattivo o run AR reale.

**Come e' costruito**
- Se c'e' `interactiveInference`, la pagina marca la sorgente come `live`.
- Hero card mostrate:
  - `Prompt`
  - `Throughput`
  - `Converge`
  - `Fallback posture`

### 6.4 `PromptLab`
File: `frontend/src/features/compare/PromptLab.tsx`

**Cosa serve**
- Preparare un prompt test e i parametri di una run comparativa.

**Come e' costruito**
- Campi:
  - prompt
  - temperature
  - top-p
  - max new tokens
  - seed
  - mode
  - effort
  - tau mask
  - tau edit
  - profile/scenario flavor
- Azioni:
  - `Generate mock run`
  - `▶ Run AR on Qwen`
  - `Reset to scenario`

Dettaglio tecnico:
- La run reale usa `useArGenerate()` e chiama:
  - `POST /api/generate/ar`
- La run mock usa `generateInteractiveInferenceScenario(...)`.

### 6.5 `InferenceSplitPane`
File: `frontend/src/features/compare/InferenceSplitPane.tsx`

**Cosa serve**
- Confrontare visivamente output AR e diffusion.

**Come e' costruito**
- Due card affiancate:
  - AR lane
  - Diffusion lane
- Ogni card mostra output e poche metriche chiave.
- Sotto c'e' una ribbon concettuale sull'ordine di certezza.

### 6.6 `DiffusionStepTimeline`
File: `frontend/src/features/diffusion-viz/DiffusionStepTimeline.tsx`

**Cosa serve**
- Far navigare gli step del decode diffusion uno per uno.

**Come e' costruito**
- Ogni step e' un bottone-card.
- Mostra:
  - step
  - eventuale `tau relax`
  - mask ratio
  - gamma
  - delta
  - confidence

### 6.7 `TokenMaskCanvas`
File: `frontend/src/features/diffusion-viz/TokenMaskCanvas.tsx`

**Cosa serve**
- Visualizzare in modo grafico lo stato dei token a uno step selezionato.

**Come e' costruito**
- Usa direttamente un elemento `<canvas>`.
- Disegna celle colorate per stato token:
  - `prompt`
  - `masked`
  - `new`
  - `edited`
  - `stable`
- La card e' un replay visuale del `TokenFrame`.

### 6.8 Pannello `Virtualized log stream`
**Cosa serve**
- Esplorare fallback ed eventi ambientali della run.

**Come e' costruito**
- Stesso pattern WSD:
  - `StickyFilterBar`
  - `useLogFeed`
  - `VirtualLogTable`
- Usa `useSearchParams` per tenere `step`, `q`, `level`, `source`, `log`.

### 6.9 Side rail legacy
La colonna destra usa:
- `StatusRail`
- `InsightCallout`
- `GlossaryInspector`

Quindi riutilizza gli stessi building block di WSD.

### 6.10 Sorgente dati legacy
Dettaglio importante:
- `backendAdapter.loadInferenceScenario(...)` oggi lancia errore `inference_not_wired`.
- Quindi questa pagina, se fosse riattivata, si appoggerebbe soprattutto a:
  - mock generator
  - run AR reale puntuale

---

## 7. Mappa dei principali store e tipi dati UI

### 7.1 `useChatStore`
File: `frontend/src/store/chatStore.ts`

Serve a gestire:
- thread chat;
- filtro thread;
- config run;
- preset;
- stato `running`;
- persistenza `localStorage`.

### 7.2 `useDataStore`
File: `frontend/src/store/dataStore.ts`

Serve a gestire:
- scenario WSD;
- scenario inference;
- disponibilita' backend;
- caricamento da adapter.

### 7.3 `useUiStore`
File: `frontend/src/store/uiStore.ts`

Serve a gestire stato puramente visuale condiviso:
- `density`
- `followTail`
- `paused`
- `selectedLogId`
- `selectedStep`

### 7.4 Tipi dati base della UI
File: `frontend/src/domain/types.ts`

Tipi importanti:
- `NormalizedLogEntry`
- `WsdMetricRow`
- `InferenceTraceStep`
- `TokenFrame`
- `InferenceRun`
- `WsdScenarioData`
- `InferenceScenarioData`
- `ChatRunConfig`
- `ChatThread`
- `ChatTurn`
- `LaneResult`

Questi tipi sono la vera interfaccia tecnica tra:
- backend;
- store;
- componenti React.

---

## 8. Endpoint backend usati dalla GUI

### 8.1 Chat
- `GET /api/health`
- `POST /api/inference/unload`
- `POST /api/inference/load`
- `GET /api/inference/logs/stream`
- `GET /api/inference/logs`
- `POST /api/generate/ar`
- `POST /api/generate`

### 8.2 Benchmark / Stage 0
- `POST /api/stage0/validate/log/start`
- `POST /api/stage0/validate/log/finish`
- `GET /api/stage0/validate/hellaswag/items`
- `POST /api/stage0/validate/hellaswag-item`
- `GET /api/stage0/validate/mmlu-pro/items`
- `POST /api/stage0/validate/mmlu-pro-item`
- `GET /api/stage0/validate/gsm8k/items`
- `POST /api/stage0/validate/gsm8k-item`
- `POST /api/stage0/validate/stability`

### 8.3 WSD legacy
- `GET /api/frontend/wsd`
- `GET /api/run/status`
- `POST /api/run/start`
- `POST /api/run/stop`

---

## 9. Sintesi finale
La GUI di HildaNext oggi e' organizzata in questo modo:
- **`/chat`**: pagina primaria, orientata a chat inference reale con gestione thread, warmup lane, configurazione e log live.
- **`/benchmark`**: pagina dedicata ai sanity check Stage 0.
- **`/legacy/wsd`**: pagina tecnica di osservabilita' WSD, molto ricca, ancora utile per diagnosi training/run.
- **`/inference`**: pagina legacy non instradata, focalizzata sul confronto AR vs diffusion.

Dal punto di vista del coding, l'impianto e' coerente:
- React + router per il layout pagine;
- Zustand per stato persistente e stato UI;
- worker e virtualizzazione per log pesanti;
- uPlot e canvas per viste tecniche ad alte prestazioni;
- backend FastAPI come sorgente dei payload runtime.

In pratica, la UI non e' solo estetica: ogni pannello e' legato a un flusso tecnico preciso, spesso con endpoint, store e tipi dati dedicati.

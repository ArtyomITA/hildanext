# HildaNext Frontend Handoff

Questo file serve all'agente che colleghera' il backend al frontend.

Scope attuale:
- frontend standalone
- nessun accesso a filesystem locale
- nessun backend wiring reale
- tutti i dati arrivano oggi da mock scenario
- la pagina Inference supporta anche generazione mock locale da prompt libero

Mockup statici:
- i file mock statici vivono in `frontend/public/mockup/`
- se la cartella `public/mockup` o i singoli file vengono rimossi, il frontend non crasha
- al posto dei dati mostra stato `Mockup missing`

## Stack
- Vite
- React 19
- TypeScript
- React Router
- Zustand
- `@tanstack/react-virtual`
- `uPlot`

## Entry points
- `src/main.tsx`
  - bootstrap React
  - carica stili globali
- `src/app/App.tsx`
  - root app
- `src/app/providers.tsx`
  - esegue `useDataStore().hydrate()`
- `src/app/router.tsx`
  - redirect `/` -> `/chat`
  - route `/chat` (ChatPage)
  - redirect `/inference` -> `/chat`
  - route `/inferenceplus` (InferencePlusPage)
  - route `/benchmark` (BenchmarkPage)
  - route `/legacy/wsd` (WsdPage)
  - redirect `/wsd` -> `/legacy/wsd`

## Shell
- `src/shell/AppShell.tsx`
  - layout globale
  - monta top nav, outlet, status rail
- `src/shell/TopNav.tsx`
  - navigazione primaria Chat / Inference+ / Benchmark / Legacy WSD
- `src/shell/StatusRail.tsx`
  - side summary dipendente dalla route

## Store

### `src/store/dataStore.ts`
Store dati applicativi.

Responsabilita':
- `hydrate()`
  - carica scenario iniziale WSD e Inference
- `setWsdScenario(id)`
  - cambia dataset WSD
- `setInferenceScenario(id)`
  - cambia dataset Inference

Quando si fara' il wiring backend, questo e' il punto principale da adattare.

Nota:
- la pagina Inference puo' usare anche un override locale generato al volo dal prompt lab
- quell'override oggi non passa dallo store globale, resta nello stato locale della pagina

### `src/store/uiStore.ts`
Store UI cross-page.

Responsabilita':
- densita' vista
- follow tail
- pausa autoscroll
- log selezionato
- step diffusion selezionato

Non contiene dati backend, solo stato di interazione.

## Adapter layer

### `src/domain/adapters.ts`
Attuale adapter mock:
- `frontendAdapter.loadWsdScenario(id)`
- `frontendAdapter.loadInferenceScenario(id)`

Questo file va sostituito o esteso con un adapter reale senza cambiare il resto della UI.

Comportamento attuale:
- l'adapter fa `fetch()` da:
  - `/mockup/wsd/{id}.json`
  - `/mockup/inference/{id}.json`
- se il file non esiste:
  - il loader lancia errore
  - `dataStore` costruisce uno scenario `missing`
  - la pagina mostra che i mockup non ci sono

Suggerimento pratico:
- lasciare invariata l'interfaccia `FrontendDataAdapter`
- creare un adapter reale che converte payload backend -> tipi frontend
- usare mock adapter come fallback/dev mode

## Tipi da rispettare

### `src/domain/types.ts`
Tipi principali attesi dalla UI:
- `NormalizedLogEntry`
- `ProcessSnapshot`
- `WsdMetricRow`
- `InferenceTraceStep`
- `TokenFrame`
- `InferenceRun`
- `WsdScenarioData`
- `InferenceScenarioData`
- `FrontendDataAdapter`

Questi tipi sono il contratto piu' importante per il backend wiring.

## Pages

### `src/routes/wsd/WsdPage.tsx`
Pagina monitoraggio conversion/training.

Legge:
- `useDataStore().wsd`
- `useLogFeed(...)`
- `useUiStore()` per log selection
- query params URL

Componenti usati:
- `MetricHeroCard`
- `PhaseTimeline`
- `TimeseriesChart`
- `TBucketHeatStrip`
- `TerminalTranscript`
- `VirtualLogTable`
- `ProcessRail`
- `InsightCallout`
- `GlossaryInspector`

Query params usati:
- `scenario`
- `q`
- `level`
- `source`
- `log`

Payload minimi richiesti:
- `wsd.meta`
- `wsd.metrics`
- `wsd.logs`
- `wsd.processes`
- `wsd.insights`

### `src/routes/inference/InferencePage.tsx`
Pagina confronto AR vs diffusion.

Legge:
- `useDataStore().inference`
- stato locale `interactiveInference`
- stato locale controlli prompt lab
- `useUiStore()` per step e log selezionati
- `useLogFeed(...)`
- query params URL

Componenti usati:
- `MetricHeroCard`
- `PromptLab`
- `InferenceSplitPane`
- `DiffusionStepTimeline`
- `TokenMaskCanvas`
- `VirtualLogTable`
- `InsightCallout`
- `GlossaryInspector`

Query params usati:
- `scenario`
- `q`
- `level`
- `source`
- `step`
- `log`

Payload minimi richiesti:
- `inference.ar`
- `inference.diffusion`
- `inference.logs`
- `inference.insights`

Comportamento attuale:
- se l'utente preme `Generate mock run`, la pagina genera una nuova run inference locale
- questa run locale ha:
  - prompt personalizzato
  - parametri inference personalizzati
  - output AR mock
  - output diffusion mock
  - log stream mock coerente con i parametri
  - trace steps mock
  - token frames mock
- il dataset scenario caricato da store resta come baseline
- tutte le sezioni principali della pagina espongono un badge:
  - `Mockup data`
  - oppure `Mockup missing`

## Log pipeline

### `src/features/logs/useLogFeed.ts`
Hook che:
- crea un Web Worker
- manda `load` con tutti i log
- manda `filter` con query/levels/sources
- riceve snapshot filtrata

Usa:
- `useDeferredValue`
- `startTransition`

Scopo:
- non bloccare la UI su filtri e ricerca

### `src/workers/logWorker.ts`
Worker per log virtualizzati.

Responsabilita':
- tiene `allLogs` in memoria worker
- filtra per:
  - `query`
  - `levels`
  - `sources`
- cap renderizzabile:
  - `MAX_VISIBLE = 10000`
- restituisce:
  - `rows`
  - `summary`

Il backend non deve conoscere il worker, deve solo fornire log convertibili in `NormalizedLogEntry[]`.

## Components e funzione pratica

### Layout
- `src/components/layout/Panel.tsx`
  - wrapper visuale standard pannelli

### Metriche / WSD
- `src/components/cards/MetricHeroCard.tsx`
  - KPI hero
- `src/components/cards/PhaseTimeline.tsx`
  - timeline warmup/stable/decay
- `src/components/cards/TBucketHeatStrip.tsx`
  - visual bucket `t`
- `src/components/charts/TimeseriesChart.tsx`
  - serie temporali loss/VRAM/MTA

### Logs
- `src/features/logs/StickyFilterBar.tsx`
  - search + toggle filtri
- `src/components/tables/VirtualLogTable.tsx`
  - lista log virtualizzata
- `src/components/terminals/TerminalTranscript.tsx`
  - transcript stile terminale

### Processi
- `src/features/processes/ProcessRail.tsx`
  - snapshot CPU/RAM/GPU/process state

### Inference
- `src/features/compare/PromptLab.tsx`
  - form prompt-first per test inference locale
  - controlli:
    - `prompt`
    - `temperature`
    - `topP`
    - `maxNewTokens`
    - `seed`
    - `mode`
    - `effort`
    - `tauMask`
    - `tauEdit`
    - `scenarioFlavor`
  - azioni:
    - `Generate mock run`
    - `Reset to scenario`
  - se i file mockup statici mancano, la generazione viene disabilitata
- `src/features/compare/InferenceSplitPane.tsx`
  - confronto testo e metriche AR vs diffusion
- `src/features/diffusion-viz/DiffusionStepTimeline.tsx`
  - step cards diffusion
- `src/features/diffusion-viz/TokenMaskCanvas.tsx`
  - canvas token state replay

### Supporto semantico
- `src/features/insights/InsightCallout.tsx`
  - note sintetiche guidate dai dati
- `src/features/glossary/GlossaryInspector.tsx`
  - glossario tecnico
- `src/components/badges/SeverityBadge.tsx`
  - badge severity

## Mock system attuale
- `public/mockup/wsd/*.json`
  - scenari WSD statici
- `public/mockup/inference/*.json`
  - scenari inference statici
- `public/mockup/shared/scenarios.json`
  - manifest scenari statici
- `src/mocks/generators.ts`
  - generatori usati per:
    - esportare i JSON mockup
    - generare localmente la run interattiva del prompt lab
- `src/mocks/fixtures.ts`
  - manifest scenario usato dalla UI

Questi file non vanno usati dal backend in produzione.
Servono come riferimento per capire la shape dati che la UI si aspetta.

Nota importante:
- i dati mock statici non sono piu' hardcoded nella UI
- vengono letti da `public/mockup`
- questo permette di eliminare la cartella mockup e verificare subito il comportamento `missing`

## Formato dati raccomandato dal backend

### 1. WSD
Il backend dovrebbe poter costruire un oggetto equivalente a:

```ts
interface WsdScenarioData {
  id: string;
  label: string;
  meta: WsdMeta;
  metrics: WsdMetricRow[];
  logs: NormalizedLogEntry[];
  processes: ProcessSnapshot[];
  insights: InsightCard[];
}
```

### 2. Inference
Il backend dovrebbe poter costruire un oggetto equivalente a:

```ts
interface InferenceScenarioData {
  id: string;
  label: string;
  ar: InferenceRun;
  diffusion: InferenceRun;
  logs: NormalizedLogEntry[];
  insights: InsightCard[];
}
```

Se si vuole sostituire anche il prompt lab mock con backend reale, serve in piu' un DTO richiesta equivalente a:

```ts
interface GenerateInferenceRequest {
  prompt: string;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  seed: number;
  mode: "S_MODE" | "Q_MODE";
  effort: "instant" | "low" | "medium" | "high" | "adaptive";
  tauMask: number;
  tauEdit: number;
}
```

## Mapping consigliato dal backend attuale al frontend

### WSD
Dalla repo backend esistono gia' dati adatti a popolare:
- `wsd.meta`
  - `run_id`
  - `config_digest`
  - optimizer/dtype/device dal summary run
- `wsd.metrics`
  - da `cpt.jsonl` / `sft.jsonl`
- `wsd.logs`
  - da `fallbacks.jsonl`
  - da `metrics.jsonl`
  - da console transcript o run logs
- `wsd.processes`
  - per ora il backend dovra' aggiungere una fonte vera se vuole CPU/RAM/GPU real-time
- `wsd.insights`
  - possono essere generati server-side o lasciati frontend-side

Campi backend particolarmente utili che mappano bene:
- `phase`
- `block_size`
- `loss`
- `loss_m2t`
- `loss_t2t`
- `masked_token_acc`
- `tokens_per_sec`
- `step_time_s`
- `vram_alloc_mb`
- `vram_reserved_mb`
- `vram_peak_mb`
- `eta_stage_sec`
- `t_sampled`
- `t_mean`
- `mask_ratio_actual`
- `bidirectional`
- `attention_mode`
- `shift_mode`
- `loss_weighting`

### Inference
Dati backend gia' vicini:
- `GenerateResponse.stats`
- `stats.logs`
- `steps_to_converge`
- `tokens_per_sec`
- `vram_peak_bytes`
- `dummy_model`
- `env_issues`
- `fallbacks`
- `tau_mask`
- `tau_edit`
- `mode`
- `effort`

Per la nuova UX prompt-first servono anche:
- `prompt`
- `temperature`
- `top_p`
- `max_new_tokens`
- `seed`
- opzionalmente un profilo/debug flag per simulare:
  - `clean`
  - `degenerate`
  - `dummy`

Per la pagina inference serve in piu':
- lane AR
  - se non esiste ancora endpoint AR, si puo' usare un adapter stub iniziale
- `tokenFrames`
  - oggi sono mock
  - domani possono essere derivati:
    - dal trace per step
    - da un payload dedicato backend che espone token state per ogni pass

## Dove collegare il backend

Ordine consigliato:

1. `src/domain/adapters.ts`
   - introdurre `BackendFrontendAdapter`
   - mantenere `MockFrontendAdapter`

2. `src/store/dataStore.ts`
   - scegliere adapter reale o mock
   - sostituire `hydrate()`, `setWsdScenario()`, `setInferenceScenario()`

3. eventuale nuovo file:
   - `src/domain/backendMappers.ts`
   - converte DTO backend -> tipi frontend

4. solo se necessario:
   - `src/domain/api.ts`
   - fetch client

5. per il prompt lab:
   - collegare `InferencePage.tsx`
   - sostituire `generateInteractiveInferenceScenario(...)` con una chiamata backend reale
   - mantenere il mock locale come fallback dev

## Strategia consigliata per il wiring

### Modalita' minima
- backend espone due endpoint:
  - `GET /frontend/wsd`
  - `GET /frontend/inference`
- frontend adapter fa fetch e mappa i dati

Per il prompt lab:
- backend espone anche un endpoint generate, ad esempio:
  - `POST /frontend/inference/generate`
- response gia' mappabile in `InferenceScenarioData` oppure in un DTO poi convertito

### Modalita' migliore
- backend espone:
  - `GET /frontend/wsd?run_id=...`
  - `GET /frontend/inference?job_id=...`
  - `GET /frontend/logs?...`
- `POST /frontend/inference/generate`
- frontend continua a usare `useLogFeed` per il filtro locale finale

### Modalita' realtime futura
- bootstrap con snapshot iniziale
- stream incrementale con SSE o websocket
- adapter aggiorna store senza cambiare componenti

## Limiti attuali da sapere
- `TokenMaskCanvas` oggi usa frame mock, non dati reali
- `ProcessRail` oggi usa process snapshot mock
- `InsightCallout` oggi usa insight mock/generati
- `GlossaryInspector` resta frontend-side anche dopo il wiring
- non esiste ancora un adapter backend reale
- il prompt lab oggi genera tutto in locale, non chiama nessun modello vero
- il prompt lab interattivo dipende comunque dalla presenza del dataset mockup statico come baseline UX

## Non rompere queste assunzioni
- la UI si aspetta array gia' normalizzati, non raw backend eterogeneo
- i log devono arrivare gia' convertiti in `NormalizedLogEntry`
- per i grafici conviene mandare numeri gia' puliti, non stringhe
- i campi `null` sono ammessi dove i tipi lo prevedono
- `step` e `tsUtc` devono essere stabili

## Checklist per l'agente backend
- implementare adapter reale senza cambiare le pagine
- mappare `cpt/sft/fallbacks/metrics` nei tipi frontend
- decidere fonte reale per `ProcessSnapshot`
- decidere payload reale per `TokenFrame`
- decidere il contratto reale del `POST` per inference da prompt
- collegare `PromptLab` a un endpoint vero mantenendo i controlli attuali
- mantenere `MockFrontendAdapter` per dev/offline mode
- non toccare i componenti UI se basta lavorare su adapter + mapper

## Build e test frontend
- `npm install`
- `npm run build`
- `npm test`
- `npm run test:e2e`

Nota pratica:
- in questa macchina i test completi possono essere lenti perche' stanno girando training e preflight in parallelo
- la build production del frontend e' gia' passata

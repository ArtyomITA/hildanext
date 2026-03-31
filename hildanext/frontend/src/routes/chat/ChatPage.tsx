import { KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ArDecodeMode,
  ChatEngineMode,
  ChatLane,
  ChatThread,
  ChatTurn,
  LaneResult,
  ThinkingMode,
} from "../../domain/types";
import { DEFAULT_CHAT_CONFIG, MODEL_CATALOG } from "../../features/chat/catalog";
import { runChatTurn } from "../../features/chat/orchestrator";
import { composeChatInput } from "../../features/chat/promptComposer";
import { useChatStore } from "../../store/chatStore";
import { formatBytes, formatCompact } from "../../domain/formatters";
import styles from "./ChatPage.module.css";

function laneLabel(lane: ChatLane) {
  return lane === "ar" ? "AR" : "dLLM";
}

function laneModelLabel(modelId: string) {
  return MODEL_CATALOG.find((item) => item.id === modelId)?.label ?? modelId;
}

type LaneLoadStatus = "idle" | "loading" | "loaded" | "error" | "offline";

interface LaneLoadInfo {
  status: LaneLoadStatus;
  message: string;
  updatedAt: string | null;
  cpuFallback: boolean;
}

type InferenceRealtimeLevel = "info" | "warning" | "error" | "notice";

interface InferenceRealtimeEvent {
  id: string;
  tsUtc: string;
  level: InferenceRealtimeLevel;
  source: string;
  event: string;
  lane: string | null;
  scope: string | null;
  benchmark: string | null;
  message: string;
  meta: Record<string, unknown>;
}

const MAX_INFERENCE_LOG_ROWS = 500;

function createInitialLaneLoadState(): Record<ChatLane, LaneLoadInfo> {
  return {
    ar: { status: "idle", message: "AR non caricato.", updatedAt: null, cpuFallback: false },
    dllm: { status: "idle", message: "dLLM non caricato.", updatedAt: null, cpuFallback: false },
  };
}

function requiredLanes(mode: ChatEngineMode): ChatLane[] {
  if (mode === "AR") return ["ar"];
  if (mode === "DLLM") return ["dllm"];
  return ["ar", "dllm"];
}

function modeScopeLabel(mode: ChatEngineMode) {
  if (mode === "AR") return "AR";
  if (mode === "DLLM") return "dLLM";
  return "AR + dLLM";
}

function laneLoadClass(status: LaneLoadStatus) {
  if (status === "loaded") return styles.loadLoaded;
  if (status === "loading") return styles.loadLoading;
  if (status === "offline") return styles.loadOffline;
  if (status === "error") return styles.loadError;
  return styles.loadIdle;
}

function normalizeRealtimeEvent(payload: unknown): InferenceRealtimeEvent | null {
  if (!payload || typeof payload !== "object") return null;
  const row = payload as Record<string, unknown>;
  const id = String(row.id ?? "").trim();
  const tsUtc = String(row.tsUtc ?? new Date().toISOString());
  const levelRaw = String(row.level ?? "info").toLowerCase();
  const level: InferenceRealtimeLevel =
    levelRaw === "error" || levelRaw === "warning" || levelRaw === "notice" ? levelRaw : "info";
  return {
    id: id || `${Date.now()}`,
    tsUtc,
    level,
    source: String(row.source ?? "inference"),
    event: String(row.event ?? "EVENT"),
    lane: row.lane == null ? null : String(row.lane),
    scope: row.scope == null ? null : String(row.scope),
    benchmark: row.benchmark == null ? null : String(row.benchmark),
    message: String(row.message ?? ""),
    meta: row.meta && typeof row.meta === "object" ? (row.meta as Record<string, unknown>) : {},
  };
}

function thinkingLabel(mode: ThinkingMode, decodeMode: "S_MODE" | "Q_MODE") {
  if (mode === "on") return "think:on";
  if (mode === "off") return "think:off";
  return decodeMode === "S_MODE" ? "think:auto(on)" : "think:auto(off)";
}

function isThinkingEnabled(mode: ThinkingMode, decodeMode: "S_MODE" | "Q_MODE"): boolean {
  if (mode === "on") return true;
  if (mode === "off") return false;
  return decodeMode === "S_MODE";
}

function splitThinkingOutput(raw: string): { thinking: string; answer: string } {
  const text = String(raw || "");
  const thinkTag = /<think>([\s\S]*?)<\/think>/gi;
  const qwenTag = /<\|begin_of_thought\|>([\s\S]*?)<\|end_of_thought\|>/gi;
  let thinking = "";
  let answer = text;

  const extract = (pattern: RegExp) => {
    const chunks: string[] = [];
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      chunks.push(String(match[1] || "").trim());
    }
    if (chunks.length > 0) {
      thinking = chunks.join("\n\n").trim();
      answer = text.replace(pattern, "").trim();
      return true;
    }
    return false;
  };

  if (!extract(thinkTag)) {
    extract(qwenTag);
  }
  if (!answer) {
    answer = text.trim();
  }
  return { thinking, answer };
}

function shouldIgnoreRawOpen(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest("[data-raw-ignore='true']"));
}

function turnStatusTitle(turn: ChatTurn) {
  if (turn.status === "running") return "Generazione in corso…";
  const offline = turn.lanes.filter((lane) => lane.status === "offline").length;
  const errors = turn.lanes.filter((lane) => lane.status === "error").length;
  if (offline > 0) return "Parziale: backend offline";
  if (errors > 0) return "Parziale: errore lane";
  return "Completato";
}

function ThreadRow({
  thread,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  thread: ChatThread;
  active: boolean;
  onSelect: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const preview = thread.turns.at(-1)?.prompt ?? "Nessun messaggio";
  return (
    <button
      type="button"
      className={`${styles.threadRow} ${active ? styles.threadRowActive : ""}`}
      onClick={onSelect}
    >
      <div className={styles.threadHead}>
        <strong>{thread.title || "Nuova chat"}</strong>
        <span>{new Date(thread.updatedAt).toLocaleTimeString()}</span>
      </div>
      <p>{preview}</p>
      <div className={styles.threadActions}>
        <span>{thread.turns.length} turni</span>
        <div>
          <a
            href="#rename"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRename();
            }}
          >
            rinomina
          </a>
          <a
            href="#delete"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onDelete();
            }}
          >
            elimina
          </a>
        </div>
      </div>
    </button>
  );
}

function LaneCard({
  lane,
  thinkingEnabled,
  onOpenRaw,
}: {
  lane: LaneResult;
  thinkingEnabled: boolean;
  onOpenRaw: (lane: LaneResult) => void;
}) {
  const laneClass =
    lane.status === "success"
      ? styles.laneSuccess
      : lane.status === "offline"
      ? styles.laneOffline
      : styles.laneError;
  const rawText = String(lane.rawText || lane.text || "");
  const split = splitThinkingOutput(rawText);
  return (
    <article
      className={`${styles.laneCard} ${laneClass} ${lane.status === "success" ? styles.laneClickable : ""}`}
      onClick={(event) => {
        if (shouldIgnoreRawOpen(event.target)) return;
        if (lane.status === "success") onOpenRaw(lane);
      }}
      role={lane.status === "success" ? "button" : undefined}
      tabIndex={lane.status === "success" ? 0 : undefined}
      onKeyDown={(event) => {
        if (lane.status !== "success") return;
        if (shouldIgnoreRawOpen(event.target)) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpenRaw(lane);
        }
      }}
      aria-label={lane.status === "success" ? "Apri vista raw risposta" : undefined}
    >
      <header>
        <div>
          <strong>{laneLabel(lane.lane)}</strong>
          <span>{laneModelLabel(lane.modelId)}</span>
        </div>
        <small>{lane.engine}</small>
      </header>
      {lane.status === "success" ? (
        <>
          <div className={styles.rawHint}>Click per aprire raw</div>
          <div className={styles.laneFlags}>
            {lane.truncated ? <span className={styles.flagWarn}>Troncata a max_new_tokens</span> : null}
            {lane.cpuFallback ? <span className={styles.flagWarn}>CPU fallback attivo</span> : null}
            {lane.ignoredSamplingParams ? (
              <span className={styles.flagInfo}>sampling params ignorati</span>
            ) : null}
          </div>
          <p>{split.answer || "Risposta vuota"}</p>
          {thinkingEnabled ? (
            split.thinking ? (
              <details
                className={styles.thinkDetails}
                data-raw-ignore="true"
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
              >
                <summary data-raw-ignore="true">Thinking interno</summary>
                <pre>{split.thinking}</pre>
              </details>
            ) : (
              <p className={styles.thinkNotice}>Thinking abilitato ma non presente nel payload.</p>
            )
          ) : (
            <p className={styles.thinkDisabled}>Thinking disattivato per questo turno.</p>
          )}
          <dl>
            <div>
              <dt>Finish</dt>
              <dd>{lane.finishReason ?? "n/a"}</dd>
            </div>
            <div>
              <dt>Throughput</dt>
              <dd>{lane.tokensPerSec ? `${formatCompact(lane.tokensPerSec)} tok/s` : "n/a"}</dd>
            </div>
            <div>
              <dt>Converge</dt>
              <dd>{lane.stepsToConverge ?? "n/a"}</dd>
            </div>
            <div>
              <dt>VRAM</dt>
              <dd>{lane.vramPeakBytes ? formatBytes(lane.vramPeakBytes) : "n/a"}</dd>
            </div>
            <div>
              <dt>Dtype</dt>
              <dd>{lane.dtype ?? "n/a"}</dd>
            </div>
            <div>
              <dt>Device</dt>
              <dd>{lane.device ?? "n/a"}</dd>
            </div>
          </dl>
        </>
      ) : (
        <p className={styles.laneMessage}>{lane.message}</p>
      )}
    </article>
  );
}

const TERMINOLOGY = [
  {
    term: "S_MODE",
    body: "Preset di decoding quality-first: piu pass di refinement (M2T/T2T), maggiore probabilita di convergenza semantica, latenza superiore.",
  },
  {
    term: "Q_MODE",
    body: "Preset speed-first: drafting conservativo e meno pass. Riduce latenza e consumo, ma puo rinunciare a parte del refinement finale.",
  },
  {
    term: "tau_mask",
    body: "Soglia di commit MASK->token (Gamma_t). Piu basso = si smaschera prima; piu alto = serve confidenza maggiore prima del commit.",
  },
  {
    term: "tau_edit",
    body: "Soglia di edit token->token (Delta_t). Regola aggressivita della correzione: alto = editing piu selettivo, basso = editing piu frequente.",
  },
  {
    term: "effort",
    body: "Budget operativo del decode loop: influenza numero step, aggressivita del fallback e trade-off qualita/tempo.",
  },
];

export function ChatPage() {
  const [prompt, setPrompt] = useState("");
  const [presetName, setPresetName] = useState("");
  const [loadingWeights, setLoadingWeights] = useState(false);
  const [loadFeedback, setLoadFeedback] = useState("");
  const [laneLoad, setLaneLoad] = useState<Record<ChatLane, LaneLoadInfo>>(createInitialLaneLoadState);
  const [rawLane, setRawLane] = useState<LaneResult | null>(null);
  const [inferEvents, setInferEvents] = useState<InferenceRealtimeEvent[]>([]);
  const [inferLogTransport, setInferLogTransport] = useState<"idle" | "sse" | "polling" | "error">("idle");
  const [inferLogError, setInferLogError] = useState("");
  const inferLogViewportRef = useRef<HTMLDivElement | null>(null);
  const inferStickBottomRef = useRef(true);
  const lastInferEventIdRef = useRef("");
  const {
    hydrated,
    filter,
    running,
    advancedOpen,
    config,
    threads,
    selectedThreadId,
    presets,
    hydrate,
    setFilter,
    setRunning,
    setAdvancedOpen,
    updateConfig,
    resetConfig,
    ensureThread,
    selectThread,
    createThread,
    renameThread,
    deleteThread,
    addTurn,
    updateTurn,
    savePreset,
    applyPreset,
    deletePreset,
  } = useChatStore();

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    let cancelled = false;
    async function probeHealth() {
      try {
        const res = await fetch("/api/health");
        if (!res.ok) return;
        const payload = (await res.json()) as { model_loaded?: boolean; device?: string };
        if (!payload.model_loaded || cancelled) return;
        const ts = new Date().toISOString();
        const cpuFallback = String(payload.device ?? "").toLowerCase() === "cpu";
        setLaneLoad({
          ar: {
            status: "loaded",
            message: cpuFallback ? "AR pronto (health probe) - CPU fallback attivo." : "AR pronto (health probe).",
            updatedAt: ts,
            cpuFallback,
          },
          dllm: {
            status: "loaded",
            message: cpuFallback ? "dLLM pronto (health probe) - CPU fallback attivo." : "dLLM pronto (health probe).",
            updatedAt: ts,
            cpuFallback,
          },
        });
      } catch {
        // backend non raggiungibile: keep idle state
      }
    }
    void probeHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!rawLane) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setRawLane(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [rawLane]);

  useEffect(() => {
    const node = inferLogViewportRef.current;
    if (!node || !inferStickBottomRef.current) return;
    node.scrollTop = node.scrollHeight;
  }, [inferEvents.length]);

  useEffect(() => {
    let closed = false;
    let pollTimer: number | null = null;
    let stream: EventSource | null = null;
    let polling = false;

    function schedulePoll(ms: number) {
      if (closed) return;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(() => {
        void pollLogs();
      }, ms) as number;
    }

    async function pollLogs() {
      if (closed) return;
      try {
        const after = lastInferEventIdRef.current;
        const query = new URLSearchParams();
        query.set("tail", "200");
        if (after) query.set("after_id", after);
        const res = await fetch(`/api/inference/logs?${query.toString()}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as { events?: unknown[] };
        const rows = (payload.events ?? [])
          .map((row) => normalizeRealtimeEvent(row))
          .filter(Boolean) as InferenceRealtimeEvent[];
        appendRealtimeEvents(rows);
        setInferLogTransport("polling");
        setInferLogError("");
      } catch (error) {
        setInferLogTransport("error");
        setInferLogError(`Polling logs failed: ${String(error)}`);
      } finally {
        schedulePoll(1000);
      }
    }

    function startPolling() {
      if (polling || closed) return;
      polling = true;
      setInferLogTransport("polling");
      schedulePoll(1000);
    }

    if (typeof EventSource !== "undefined") {
      const after = lastInferEventIdRef.current;
      const url = after
        ? `/api/inference/logs/stream?after_id=${encodeURIComponent(after)}`
        : "/api/inference/logs/stream";
      try {
        stream = new EventSource(url);
        setInferLogTransport("sse");
        setInferLogError("");
        stream.onopen = () => {
          if (closed) return;
          setInferLogTransport("sse");
          setInferLogError("");
        };
        stream.onmessage = (event: MessageEvent<string>) => {
          if (closed) return;
          let parsed: unknown = {};
          try {
            parsed = JSON.parse(String(event.data || "{}"));
          } catch {
            return;
          }
          const row = normalizeRealtimeEvent(parsed);
          if (!row) return;
          appendRealtimeEvents([row]);
        };
        stream.onerror = () => {
          if (closed) return;
          setInferLogError("SSE disconnected, switching to polling.");
          try {
            stream?.close();
          } catch {
            // noop
          }
          stream = null;
          startPolling();
        };
      } catch (error) {
        setInferLogError(`SSE unavailable: ${String(error)}`);
        startPolling();
      }
    } else {
      setInferLogError("EventSource unavailable, using polling.");
      startPolling();
    }

    return () => {
      closed = true;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      try {
        stream?.close();
      } catch {
        // noop
      }
    };
  }, []);

  const selectedThread = useMemo(
    () => threads.find((thread) => thread.id === selectedThreadId) ?? null,
    [selectedThreadId, threads],
  );

  const lanesNeeded = useMemo(() => requiredLanes(config.engineMode), [config.engineMode]);
  const composerUnlocked = useMemo(
    () => lanesNeeded.every((lane) => laneLoad[lane].status === "loaded"),
    [laneLoad, lanesNeeded],
  );

  const lockReason = useMemo(() => {
    if (loadingWeights) return "Composer bloccato: caricamento pesi in corso.";
    if (composerUnlocked) return "";
    const missing = lanesNeeded.map((lane) => laneLabel(lane)).join(" + ");
    return `Composer bloccato: carica prima i pesi per ${missing}.`;
  }, [composerUnlocked, lanesNeeded, loadingWeights]);

  function appendRealtimeEvents(items: InferenceRealtimeEvent[]) {
    if (items.length === 0) return;
    setInferEvents((prev) => {
      const next = [...prev];
      const seen = new Set(prev.map((row) => row.id));
      for (const row of items) {
        if (seen.has(row.id)) continue;
        next.push(row);
        seen.add(row.id);
      }
      if (next.length > MAX_INFERENCE_LOG_ROWS) {
        return next.slice(-MAX_INFERENCE_LOG_ROWS);
      }
      return next;
    });
    const last = items[items.length - 1];
    if (last?.id) {
      lastInferEventIdRef.current = last.id;
    }
  }

  const filteredThreads = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return threads;
    return threads.filter((thread) => {
      const inTitle = thread.title.toLowerCase().includes(q);
      const inTurns = thread.turns.some((turn) => turn.prompt.toLowerCase().includes(q));
      return inTitle || inTurns;
    });
  }, [filter, threads]);

  async function handleLoadWeights() {
    if (running || loadingWeights) return;
    const startedMs = Date.now();
    setLoadFeedback(`Richiesta inviata al backend (${modeScopeLabel(config.engineMode)}).`);
    setLaneLoad((prev) => {
      const next = { ...prev };
      for (const lane of lanesNeeded) {
        next[lane] = {
          status: "loading",
          message: "Caricamento pesi avviato...",
          updatedAt: new Date().toISOString(),
          cpuFallback: false,
        };
      }
      return next;
    });
    setLoadingWeights(true);
    try {
      // Best effort: clear previous engine weights/cache to avoid repeated-load buildup.
      try {
        await fetch("/api/inference/unload", { method: "POST" });
      } catch {
        // ignore: unload endpoint may be unavailable on older backend
      }
      let loadedViaDedicatedApi = false;
      try {
        const res = await fetch("/api/inference/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scope: config.engineMode,
            prompt: "__hildanext_load_weights__",
            mode: config.decodeMode,
            max_new_tokens: 1,
            seed: config.seed,
            effort: "instant",
          }),
        });
        if (res.ok) {
          const payload = (await res.json()) as {
            lanes?: Partial<
              Record<ChatLane, { status?: string; message?: string; stats?: Record<string, unknown> | null }>
            >;
          };
          if (payload.lanes) {
            loadedViaDedicatedApi = true;
            setLaneLoad((prev) => {
              const next = { ...prev };
              for (const lane of lanesNeeded) {
                const lanePayload = payload.lanes?.[lane];
                const status = String(lanePayload?.status ?? "error");
                const stats = lanePayload?.stats ?? {};
                const device = typeof stats?.device === "string" ? stats.device : "";
                const vram = typeof stats?.vram_peak_bytes === "number" ? stats.vram_peak_bytes : null;
                const cpuFallback = device === "cpu" || vram === null;
                next[lane] = {
                  status: status === "loaded" ? "loaded" : status === "offline" ? "offline" : "error",
                  message:
                    (lanePayload?.message || "Warmup lane non disponibile.") +
                    (status === "loaded" && cpuFallback ? " CPU fallback attivo." : ""),
                  updatedAt: new Date().toISOString(),
                  cpuFallback,
                };
              }
              return next;
            });
            const elapsedS = ((Date.now() - startedMs) / 1000).toFixed(1);
            setLoadFeedback(`Caricamento completato in ${elapsedS}s.`);
          }
        }
      } catch {
        // fallback to legacy warmup call
      }
      if (!loadedViaDedicatedApi) {
        const warmupCfg = { ...config, maxNewTokens: 1 };
        const warmupResults = await runChatTurn("__hildanext_load_weights__", warmupCfg);
        const byLane = new Map(warmupResults.map((lane) => [lane.lane, lane] as const));
        setLaneLoad((prev) => {
          const next = { ...prev };
          for (const lane of lanesNeeded) {
            const result = byLane.get(lane);
            if (!result) {
              next[lane] = {
                status: "error",
                message: "Warmup incompleto: lane non restituita.",
                updatedAt: new Date().toISOString(),
                cpuFallback: false,
              };
              continue;
            }
            if (result.status === "success") {
              next[lane] = {
                status: "loaded",
                message:
                  `Pesi pronti via ${result.engine}.` +
                  (result.cpuFallback ? " CPU fallback attivo." : ""),
                updatedAt: new Date().toISOString(),
                cpuFallback: Boolean(result.cpuFallback),
              };
              continue;
            }
            next[lane] = {
              status: result.status,
              message: result.message || "Errore durante warmup lane.",
              updatedAt: new Date().toISOString(),
              cpuFallback: false,
            };
          }
          return next;
        });
        const elapsedS = ((Date.now() - startedMs) / 1000).toFixed(1);
        setLoadFeedback(`Caricamento completato in ${elapsedS}s (fallback warmup).`);
      }
    } catch (error) {
      const msg = String(error);
      setLaneLoad((prev) => {
        const next = { ...prev };
        for (const lane of lanesNeeded) {
          next[lane] = {
            status: "error",
            message: `Warmup fallito: ${msg}`,
            updatedAt: new Date().toISOString(),
            cpuFallback: false,
          };
        }
        return next;
      });
      setLoadFeedback(`Caricamento fallito: ${msg}`);
    } finally {
      setLoadingWeights(false);
    }
  }

  async function handleSend() {
    const text = prompt.trim();
    if (!text || running || loadingWeights || !composerUnlocked) return;
    const composed = composeChatInput(selectedThread?.turns ?? [], text, config);
    const threadId = ensureThread();
    const turnId = crypto.randomUUID();
    const createdAt = new Date().toISOString();
    addTurn(threadId, {
      id: turnId,
      createdAt,
      prompt: text,
      config: { ...config },
      status: "running",
      lanes: [],
    });
    setPrompt("");
    setRunning(true);
    try {
      const lanes = await runChatTurn(
        {
          prompt: composed.prompt,
          messages: composed.messages,
          enableThinking: composed.enableThinking,
        },
        config,
      );
      updateTurn(threadId, turnId, { status: "completed", lanes });
    } catch (error) {
      const fallback: LaneResult = {
        lane: config.engineMode === "DLLM" ? "dllm" : "ar",
        modelId: config.engineMode === "DLLM" ? config.dllmModelId : config.arModelId,
        status: "error",
        text: "",
        message: String(error),
        engine: "unknown",
        tokensPerSec: null,
        stepsToConverge: null,
        vramPeakBytes: null,
        dummyModel: false,
      };
      updateTurn(threadId, turnId, { status: "completed", lanes: [fallback] });
    } finally {
      setRunning(false);
    }
  }

  function onComposerKey(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void handleSend();
    }
  }

  return (
    <div className={styles.page}>
      <aside className={styles.sidebar}>
        <header>
          <h2>Thread</h2>
          <button type="button" onClick={createThread}>
            + nuova chat
          </button>
        </header>
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filtra thread..."
          aria-label="Filtra thread"
        />
        <div className={styles.threadList}>
          {filteredThreads.map((thread) => (
            <ThreadRow
              key={thread.id}
              thread={thread}
              active={thread.id === selectedThreadId}
              onSelect={() => selectThread(thread.id)}
              onRename={() => {
                const next = window.prompt("Nuovo nome thread", thread.title);
                if (next && next.trim()) renameThread(thread.id, next.trim());
              }}
              onDelete={() => {
                if (window.confirm("Eliminare questo thread?")) deleteThread(thread.id);
              }}
            />
          ))}
          {filteredThreads.length === 0 ? (
            <p className={styles.emptyThreads}>Nessun thread trovato.</p>
          ) : null}
        </div>
      </aside>

      <section className={styles.timeline}>
        <header className={styles.timelineHead}>
          <h1>Chat-First Inference Studio</h1>
          <div>
            <span>{config.engineMode}</span>
            <span>{composerUnlocked ? "weights ready" : "weights missing"}</span>
            <span>{running ? "running" : "ready"}</span>
          </div>
        </header>

        <div className={styles.turns}>
          {!hydrated ? <p className={styles.emptyTimeline}>Caricamento stato locale...</p> : null}
          {hydrated && !selectedThread ? (
            <p className={styles.emptyTimeline}>
              Crea una chat o invia un prompt per iniziare.
            </p>
          ) : null}
          {selectedThread?.turns.map((turn) => {
            const turnConfig = { ...DEFAULT_CHAT_CONFIG, ...(turn.config ?? {}) };
            return (
            <article key={turn.id} className={styles.turnCard}>
              <div className={styles.userBubble}>
                <small>Tu</small>
                <p>{turn.prompt}</p>
              </div>
              <div className={styles.assistantBlock}>
                <div className={styles.assistantHead}>
                  <strong>Assistant · {turnStatusTitle(turn)}</strong>
                  <span>
                    {turnConfig.engineMode} · {turnConfig.decodeMode} · effort {turnConfig.effort} ·{" "}
                    {thinkingLabel(turnConfig.thinkingMode, turnConfig.decodeMode)} · system{" "}
                    {String(turnConfig.systemPrompt ?? "").trim() ? "on" : "off"}
                  </span>
                </div>
                <div className={`${styles.laneGrid} ${turn.lanes.length > 1 ? styles.laneGridTwo : ""}`}>
                  {turn.status === "running" ? (
                    <p className={styles.turnRunning}>Esecuzione lane in corso…</p>
                  ) : (
                    turn.lanes.map((lane) => (
                      <LaneCard
                        key={`${turn.id}-${lane.lane}`}
                        lane={lane}
                        thinkingEnabled={isThinkingEnabled(turnConfig.thinkingMode, turnConfig.decodeMode)}
                        onOpenRaw={setRawLane}
                      />
                    ))
                  )}
                </div>
              </div>
            </article>
            );
          })}
        </div>

        <footer className={styles.composer}>
          <div className={styles.composerField}>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={onComposerKey}
              placeholder="Scrivi prompt... Ctrl+Invio per inviare"
              rows={3}
              disabled={running || loadingWeights || !composerUnlocked}
            />
            {lockReason ? <p className={styles.composerLock}>{lockReason}</p> : null}
          </div>
          <button
            type="button"
            disabled={!prompt.trim() || running || loadingWeights || !composerUnlocked}
            onClick={() => void handleSend()}
          >
            {running ? "Invio..." : "Invia"}
          </button>
        </footer>
      </section>

      <aside className={styles.config}>
        <section className={styles.block}>
          <h3>Load Weights</h3>
          <p className={styles.loadLead}>
            Carica i pesi lane prima di scrivere in chat. Scope attuale: <strong>{modeScopeLabel(config.engineMode)}</strong>.
          </p>
          <div className={styles.loadStateGrid}>
            {(["ar", "dllm"] as ChatLane[]).map((lane) => (
              <div key={lane} className={styles.loadStateCard}>
                <div className={styles.loadStateHead}>
                  <strong>{laneLabel(lane)}</strong>
                  <span className={`${styles.loadBadge} ${laneLoadClass(laneLoad[lane].status)}`}>
                    {laneLoad[lane].status}
                  </span>
                </div>
                <p>{laneLoad[lane].message}</p>
              </div>
            ))}
          </div>
          {lanesNeeded.some((lane) => laneLoad[lane].cpuFallback) ? (
            <p className={styles.cpuWarn}>CPU fallback attivo: throughput ridotto, VRAM GPU non usata.</p>
          ) : null}
          <button
            type="button"
            className={styles.loadBtn}
            disabled={running || loadingWeights}
            onClick={() => void handleLoadWeights()}
          >
            {loadingWeights ? "Caricamento pesi..." : "Carica pesi modello"}
          </button>
          {loadFeedback ? <p className={styles.loadFeedback}>{loadFeedback}</p> : null}
        </section>

        <section className={styles.block}>
          <h3>Inference Realtime Log</h3>
          <p className={styles.logMetaLine}>
            transport: <strong>{inferLogTransport}</strong>
            {inferLogError ? ` · ${inferLogError}` : ""}
          </p>
          <div
            className={styles.inferLogViewport}
            ref={inferLogViewportRef}
            onScroll={(event) => {
              const node = event.currentTarget;
              const nearBottom = node.scrollHeight - node.scrollTop - node.clientHeight <= 24;
              inferStickBottomRef.current = nearBottom;
            }}
          >
            {inferEvents.length === 0 ? (
              <p className={styles.inferLogEmpty}>Nessun evento inference ancora ricevuto.</p>
            ) : (
              inferEvents.map((row) => (
                <article key={row.id} className={styles.inferLogRow} data-level={row.level}>
                  <p className={styles.inferLogHead}>
                    <span>{new Date(row.tsUtc).toLocaleTimeString()}</span>
                    <span>{row.lane ? row.lane.toUpperCase() : "SYS"}</span>
                    <span>{row.event}</span>
                  </p>
                  <p className={styles.inferLogMessage}>{row.message || "(empty)"}</p>
                  {Object.keys(row.meta).length > 0 ? (
                    <details className={styles.inferLogDetails}>
                      <summary>details</summary>
                      <pre>{JSON.stringify(row.meta, null, 2)}</pre>
                    </details>
                  ) : null}
                </article>
              ))
            )}
          </div>
        </section>

        <section className={styles.block}>
          <h3>Config base</h3>
          <label>
            <span>Engine mode</span>
            <select
              value={config.engineMode}
              onChange={(event) => updateConfig("engineMode", event.target.value as typeof config.engineMode)}
            >
              <option value="AR">AR</option>
              <option value="DLLM">dLLM</option>
              <option value="BOTH">Both</option>
            </select>
          </label>
          <label>
            <span>Modello AR</span>
            <select value={config.arModelId} onChange={(event) => updateConfig("arModelId", event.target.value)}>
              {MODEL_CATALOG.filter((model) => model.lane === "ar").map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Modello dLLM</span>
            <select
              value={config.dllmModelId}
              onChange={(event) => updateConfig("dllmModelId", event.target.value)}
            >
              {MODEL_CATALOG.filter((model) => model.lane === "dllm").map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Max new tokens</span>
            <input
              type="number"
              min={8}
              max={1024}
              value={config.maxNewTokens}
              onChange={(event) => updateConfig("maxNewTokens", Number(event.target.value) || 8)}
            />
          </label>
          <label>
            <span>Seed</span>
            <input
              type="number"
              value={config.seed}
              onChange={(event) => updateConfig("seed", Number(event.target.value) || 0)}
            />
          </label>
        </section>

        <section className={styles.block}>
          <button
            type="button"
            className={styles.advancedToggle}
            onClick={() => setAdvancedOpen(!advancedOpen)}
          >
            {advancedOpen ? "Nascondi avanzate" : "Mostra avanzate"}
          </button>
          {advancedOpen ? (
            <div className={styles.advancedBody}>
              <label>
                <span>Decode mode</span>
                <select
                  value={config.decodeMode}
                  onChange={(event) => {
                    const nextMode = event.target.value as typeof config.decodeMode;
                    updateConfig("decodeMode", nextMode);
                    if (config.thinkingMode === "auto") {
                      if (nextMode === "S_MODE") {
                        updateConfig("temperature", 0.6);
                        updateConfig("topP", 0.95);
                        updateConfig("topK", 20);
                      } else {
                        updateConfig("temperature", 0.7);
                        updateConfig("topP", 0.8);
                        updateConfig("topK", 20);
                      }
                    }
                  }}
                >
                  <option value="S_MODE">S_MODE</option>
                  <option value="Q_MODE">Q_MODE</option>
                </select>
                <small className={styles.fieldHint}>S_MODE: quality-first; Q_MODE: speed-first.</small>
              </label>
              <label>
                <span>Thinking mode</span>
                <select
                  value={config.thinkingMode}
                  onChange={(event) => {
                    const next = event.target.value as ThinkingMode;
                    updateConfig("thinkingMode", next);
                    if (next === "on") {
                      updateConfig("temperature", 0.6);
                      updateConfig("topP", 0.95);
                      updateConfig("topK", 20);
                    } else if (next === "off") {
                      updateConfig("temperature", 0.7);
                      updateConfig("topP", 0.8);
                      updateConfig("topK", 20);
                    }
                  }}
                >
                  <option value="auto">auto</option>
                  <option value="on">on</option>
                  <option value="off">off</option>
                </select>
              </label>
              <label>
                <span>AR decode mode</span>
                <select
                  value={config.arDecodeMode}
                  onChange={(event) => updateConfig("arDecodeMode", event.target.value as ArDecodeMode)}
                >
                  <option value="sampling">sampling</option>
                  <option value="greedy">greedy</option>
                </select>
              </label>
              <label>
                <span>System prompt</span>
                <textarea
                  value={config.systemPrompt}
                  rows={4}
                  onChange={(event) => updateConfig("systemPrompt", event.target.value)}
                />
              </label>
              <label>
                <span>Context window tokens</span>
                <input
                  type="number"
                  min={512}
                  max={32768}
                  step={128}
                  value={config.contextWindowTokens}
                  onChange={(event) => updateConfig("contextWindowTokens", Number(event.target.value) || 4096)}
                />
              </label>
              <label>
                <span>Effort</span>
                <select
                  value={config.effort}
                  onChange={(event) => updateConfig("effort", event.target.value as typeof config.effort)}
                >
                  <option value="instant">instant</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                  <option value="adaptive">adaptive</option>
                </select>
              </label>
              <label>
                <span>Tau mask</span>
                <input
                  type="number"
                  min={0.01}
                  max={1}
                  step={0.01}
                  value={config.tauMask}
                  onChange={(event) => updateConfig("tauMask", Number(event.target.value))}
                />
                <small className={styles.fieldHint}>Soglia commit MASK-&gt;token (Gamma_t).</small>
              </label>
              <label>
                <span>Tau edit</span>
                <input
                  type="number"
                  min={0.01}
                  max={1}
                  step={0.01}
                  value={config.tauEdit}
                  onChange={(event) => updateConfig("tauEdit", Number(event.target.value))}
                />
                <small className={styles.fieldHint}>Soglia edit token-&gt;token (Delta_t).</small>
              </label>
              <label>
                <span>Temperature</span>
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.01}
                  value={config.temperature}
                  onChange={(event) => updateConfig("temperature", Number(event.target.value))}
                  disabled={config.engineMode === "AR" && config.arDecodeMode === "greedy"}
                />
              </label>
              <label>
                <span>Top-p</span>
                <input
                  type="number"
                  min={0.1}
                  max={1}
                  step={0.01}
                  value={config.topP}
                  onChange={(event) => updateConfig("topP", Number(event.target.value))}
                  disabled={config.engineMode === "AR" && config.arDecodeMode === "greedy"}
                />
              </label>
              <label>
                <span>Top-k</span>
                <input
                  type="number"
                  min={0}
                  max={2000}
                  step={1}
                  value={config.topK}
                  onChange={(event) => updateConfig("topK", Number(event.target.value) || 0)}
                  disabled={config.engineMode === "AR" && config.arDecodeMode === "greedy"}
                />
              </label>
              <label>
                <span>Presence penalty</span>
                <input
                  type="number"
                  min={0}
                  max={5}
                  step={0.1}
                  value={config.presencePenalty}
                  onChange={(event) => updateConfig("presencePenalty", Number(event.target.value) || 0)}
                  disabled={config.engineMode === "AR" && config.arDecodeMode === "greedy"}
                />
              </label>
              <label>
                <span>Repetition penalty</span>
                <input
                  type="number"
                  min={0}
                  max={5}
                  step={0.1}
                  value={config.repetitionPenalty}
                  onChange={(event) => updateConfig("repetitionPenalty", Number(event.target.value) || 1)}
                  disabled={config.engineMode === "AR" && config.arDecodeMode === "greedy"}
                />
              </label>
              <button type="button" className={styles.resetBtn} onClick={resetConfig}>
                Reset config
              </button>
            </div>
          ) : null}
        </section>

        <section className={styles.block}>
          <details className={styles.shadowLayer}>
            <summary>Shadow Layer Terminologia</summary>
            <p className={styles.shadowLead}>
              Terminologia tecnica invariata, con spiegazione rapida in contesto operativo.
            </p>
            <div className={styles.shadowGrid}>
              {TERMINOLOGY.map((item) => (
                <article key={item.term} className={styles.shadowCard}>
                  <strong>{item.term}</strong>
                  <p>{item.body}</p>
                </article>
              ))}
            </div>
          </details>
        </section>

        <section className={styles.block}>
          <h3>Preset locali</h3>
          <div className={styles.presetCreate}>
            <input
              placeholder="Nome preset"
              value={presetName}
              onChange={(event) => setPresetName(event.target.value)}
            />
            <button
              type="button"
              onClick={() => {
                if (!presetName.trim()) return;
                savePreset(presetName);
                setPresetName("");
              }}
            >
              Salva
            </button>
          </div>
          <div className={styles.presetList}>
            {presets.map((preset) => (
              <div key={preset.id} className={styles.presetRow}>
                <strong>{preset.name}</strong>
                <div>
                  <button type="button" onClick={() => applyPreset(preset.id)}>
                    Applica
                  </button>
                  <button type="button" onClick={() => deletePreset(preset.id)}>
                    X
                  </button>
                </div>
              </div>
            ))}
            {presets.length === 0 ? <p className={styles.emptyPreset}>Nessun preset salvato.</p> : null}
          </div>
        </section>
      </aside>
      {rawLane ? (
        <div className={styles.rawOverlay} onClick={() => setRawLane(null)}>
          <section
            className={styles.rawModal}
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label={`Raw risposta ${laneLabel(rawLane.lane)}`}
          >
            <header className={styles.rawHead}>
              <div>
                <strong>Raw risposta · {laneLabel(rawLane.lane)}</strong>
                <span>{laneModelLabel(rawLane.modelId)}</span>
              </div>
              <button type="button" onClick={() => setRawLane(null)}>
                Chiudi
              </button>
            </header>
            <div className={styles.rawBody}>
              <article>
                <h4>Raw text</h4>
                <pre>{rawLane.rawText || rawLane.text || ""}</pre>
              </article>
              <article>
                <h4>Raw stats</h4>
                <pre>{JSON.stringify(rawLane.rawStats ?? {}, null, 2)}</pre>
              </article>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

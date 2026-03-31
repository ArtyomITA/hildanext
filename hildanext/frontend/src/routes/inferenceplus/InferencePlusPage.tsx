import { KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import styles from "./InferencePlusPage.module.css";

type InferenceMode = "RCD" | "OTS";

type DllmLoadStatus = "idle" | "loading" | "loaded" | "error" | "offline";

interface DllmLoadState {
  status: DllmLoadStatus;
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

const MAX_LOG_ROWS = 500;

interface TurnResult {
  id: string;
  prompt: string;
  mode: InferenceMode;
  status: "running" | "success" | "error";
  text: string;
  engine: string;
  error: string;
  stats: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  elapsedMs: number | null;
}

interface RcdConfig {
  rcd_temperature_residual: number;
  rcd_warm_start: boolean;
}

interface OtsConfig {
  ots_beam_size: number;
  ots_gumbel_temperature: number;
}

function defaultRcd(): RcdConfig {
  return { rcd_temperature_residual: 1.0, rcd_warm_start: true };
}

function defaultOts(): OtsConfig {
  return { ots_beam_size: 3, ots_gumbel_temperature: 0.6 };
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? v.toLocaleString("it-IT", { maximumFractionDigits: 2 }) : "—";
  if (typeof v === "boolean") return v ? "Sì" : "No";
  return String(v);
}

function loadStatusClass(status: DllmLoadStatus): string {
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

export function InferencePlusPage() {
  const [mode, setMode] = useState<InferenceMode>("RCD");
  const [turns, setTurns] = useState<TurnResult[]>([]);
  const [running, setRunning] = useState(false);
  const [draft, setDraft] = useState("");
  const [maxTokens, setMaxTokens] = useState(512);
  const [effort, setEffort] = useState("medium");
  const [seed, setSeed] = useState<number | null>(null);
  const [rcd, setRcd] = useState(defaultRcd);
  const [ots, setOts] = useState(defaultOts);
  const turnsRef = useRef<HTMLDivElement>(null);

  // ── Weight loading state ──
  const [dllmLoad, setDllmLoad] = useState<DllmLoadState>({
    status: "idle",
    message: "dLLM non caricato.",
    updatedAt: null,
    cpuFallback: false,
  });
  const [loadingWeights, setLoadingWeights] = useState(false);
  const [loadFeedback, setLoadFeedback] = useState("");

  // ── Inference log state ──
  const [inferEvents, setInferEvents] = useState<InferenceRealtimeEvent[]>([]);
  const [inferLogTransport, setInferLogTransport] = useState<"sse" | "polling" | "error">("polling");
  const [inferLogError, setInferLogError] = useState("");
  const inferLogViewportRef = useRef<HTMLDivElement>(null);
  const inferStickBottomRef = useRef(true);
  const lastInferEventIdRef = useRef<string>("");

  const weightsReady = dllmLoad.status === "loaded";

  function appendRealtimeEvents(items: InferenceRealtimeEvent[]) {
    if (items.length === 0) return;
    setInferEvents((prev) => {
      const next = [...prev];
      const seen = new Set(prev.map((r) => r.id));
      for (const r of items) {
        if (seen.has(r.id)) continue;
        next.push(r);
        seen.add(r.id);
      }
      return next.length > MAX_LOG_ROWS ? next.slice(-MAX_LOG_ROWS) : next;
    });
    const last = items[items.length - 1];
    if (last?.id) lastInferEventIdRef.current = last.id;
  }

  // ── Health probe on mount ──
  useEffect(() => {
    let cancelled = false;
    async function probeHealth() {
      try {
        const res = await fetch("/api/health");
        if (!res.ok || cancelled) return;
        const payload = (await res.json()) as { model_loaded?: boolean; device?: string };
        if (!payload.model_loaded || cancelled) return;
        const cpuFallback = String(payload.device ?? "").toLowerCase() === "cpu";
        setDllmLoad({
          status: "loaded",
          message: cpuFallback ? "dLLM pronto (health probe) — CPU fallback." : "dLLM pronto (health probe).",
          updatedAt: new Date().toISOString(),
          cpuFallback,
        });
      } catch {
        // backend non raggiungibile
      }
    }
    void probeHealth();
    return () => { cancelled = true; };
  }, []);

  // ── Inference log SSE / polling ──
  useEffect(() => {
    let closed = false;
    let pollTimer: number | null = null;
    let stream: EventSource | null = null;
    let polling = false;

    function schedulePoll(ms: number) {
      if (closed) return;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(() => { void pollLogs(); }, ms) as number;
    }

    async function pollLogs() {
      if (closed) return;
      try {
        const after = lastInferEventIdRef.current;
        const query = new URLSearchParams();
        query.set("tail", "200");
        if (after) query.set("after_id", after);
        const res = await fetch(`/api/inference/logs?${query.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as { events?: unknown[] };
        const rows = (payload.events ?? [])
          .map((r) => normalizeRealtimeEvent(r))
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
          try { parsed = JSON.parse(String(event.data || "{}")); } catch { return; }
          const row = normalizeRealtimeEvent(parsed);
          if (!row) return;
          appendRealtimeEvents([row]);
        };
        stream.onerror = () => {
          if (closed) return;
          setInferLogError("SSE disconnected, switching to polling.");
          try { stream?.close(); } catch { /* noop */ }
          stream = null;
          startPolling();
        };
      } catch {
        startPolling();
      }
    } else {
      startPolling();
    }

    return () => {
      closed = true;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      try { stream?.close(); } catch { /* noop */ }
    };
  }, []);

  // ── Auto-scroll log viewport ──
  useEffect(() => {
    const node = inferLogViewportRef.current;
    if (!node || !inferStickBottomRef.current) return;
    node.scrollTop = node.scrollHeight;
  }, [inferEvents.length]);

  // ── Load weights (dLLM only) ──
  async function handleLoadWeights() {
    if (running || loadingWeights) return;
    const startedMs = Date.now();
    setLoadFeedback("Richiesta inviata al backend (dLLM).");
    setDllmLoad({
      status: "loading",
      message: "Caricamento pesi avviato...",
      updatedAt: new Date().toISOString(),
      cpuFallback: false,
    });
    setLoadingWeights(true);
    try {
      try { await fetch("/api/inference/unload", { method: "POST" }); } catch { /* ignore */ }
      const res = await fetch("/api/inference/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scope: "DLLM",
          prompt: "__hildanext_load_weights__",
          mode: "S_MODE",
          max_new_tokens: 1,
          seed: null,
          effort: "instant",
        }),
      });
      if (res.ok) {
        const payload = (await res.json()) as {
          lanes?: Partial<Record<string, { status?: string; message?: string; stats?: Record<string, unknown> | null }>>;
        };
        const lanePayload = payload.lanes?.dllm;
        const status = String(lanePayload?.status ?? "error");
        const stats = lanePayload?.stats ?? {};
        const device = typeof stats?.device === "string" ? stats.device : "";
        const vram = typeof stats?.vram_peak_bytes === "number" ? stats.vram_peak_bytes : null;
        const cpuFallback = device === "cpu" || vram === null;
        setDllmLoad({
          status: status === "loaded" ? "loaded" : status === "offline" ? "offline" : "error",
          message:
            (lanePayload?.message || "Warmup completato.") +
            (status === "loaded" && cpuFallback ? " CPU fallback attivo." : ""),
          updatedAt: new Date().toISOString(),
          cpuFallback,
        });
        const elapsedS = ((Date.now() - startedMs) / 1000).toFixed(1);
        setLoadFeedback(`Caricamento completato in ${elapsedS}s.`);
      } else {
        const body = await res.text().catch(() => "");
        throw new Error(body || `HTTP ${res.status}`);
      }
    } catch (error) {
      setDllmLoad({
        status: "error",
        message: `Caricamento fallito: ${String(error)}`,
        updatedAt: new Date().toISOString(),
        cpuFallback: false,
      });
      setLoadFeedback(`Caricamento fallito: ${String(error)}`);
    } finally {
      setLoadingWeights(false);
    }
  }

  async function send() {
    const prompt = draft.trim();
    if (!prompt || running) return;
    setDraft("");
    const id = crypto.randomUUID();
    const turn: TurnResult = {
      id,
      prompt,
      mode,
      status: "running",
      text: "",
      engine: "",
      error: "",
      stats: {},
      diagnostics: {},
      elapsedMs: null,
    };
    setTurns((prev) => [...prev, turn]);
    setRunning(true);

    const t0 = performance.now();
    try {
      const endpoint = mode === "RCD" ? "/api/inferencercdm" : "/api/inferenceots";
      const extra =
        mode === "RCD"
          ? {
              rcd_temperature_residual: rcd.rcd_temperature_residual,
              rcd_warm_start: rcd.rcd_warm_start,
            }
          : {
              ots_beam_size: ots.ots_beam_size,
              ots_gumbel_temperature: ots.ots_gumbel_temperature,
            };
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          seed,
          effort,
          ...extra,
        }),
      });
      const elapsed = performance.now() - t0;
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(body || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? {
                ...t,
                status: "success" as const,
                text: String(payload.text ?? ""),
                engine: String(payload.engine ?? mode),
                stats: payload.stats ?? {},
                diagnostics: payload.diagnostics ?? {},
                elapsedMs: elapsed,
              }
            : t,
        ),
      );
    } catch (err) {
      const elapsed = performance.now() - t0;
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? { ...t, status: "error" as const, error: String(err), elapsedMs: elapsed }
            : t,
        ),
      );
    } finally {
      setRunning(false);
      setTimeout(() => turnsRef.current?.scrollTo({ top: turnsRef.current.scrollHeight, behavior: "smooth" }), 60);
    }
  }

  function onKeyDown(e: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className={styles.page}>
      {/* ── Chat timeline ── */}
      <section className={styles.timeline}>
        <div className={styles.timelineHead}>
          <h1>Inferenza+</h1>
          <div>
            <span>{mode}</span>
            <span>{effort}</span>
          </div>
        </div>

        <div className={styles.turns} ref={turnsRef}>
          {turns.length === 0 && (
            <p className={styles.emptyTimeline}>
              Scrivi un prompt per generare con {mode === "RCD" ? "Residual Context Diffusion" : "Order-Token Search"}.
            </p>
          )}
          {turns.map((t) => (
            <div key={t.id} className={styles.turnCard}>
              <div className={styles.userBubble}>
                <small>{t.mode}</small>
                <p>{t.prompt}</p>
              </div>
              {t.status === "running" && <p className={styles.turnRunning}>Generazione in corso…</p>}
              {t.status === "error" && (
                <div className={`${styles.laneCard} ${styles.laneError}`}>
                  <header>
                    <strong>{t.mode}</strong>
                    <small>errore</small>
                  </header>
                  <p>{t.error}</p>
                </div>
              )}
              {t.status === "success" && (
                <div className={styles.assistantBlock}>
                  <div className={styles.assistantHead}>
                    <strong>{t.mode} — {t.engine}</strong>
                    <span>{t.elapsedMs != null ? `${(t.elapsedMs / 1000).toFixed(2)}s` : ""}</span>
                  </div>
                  <div className={`${styles.laneCard} ${styles.laneSuccess}`}>
                    <header>
                      <strong>{t.mode}</strong>
                      <small>{t.engine}</small>
                    </header>
                    <p style={{ whiteSpace: "pre-wrap" }}>{t.text || "(vuoto)"}</p>
                    <dl>
                      <dt>Engine</dt><dd>{t.engine}</dd>
                      <dt>Tempo</dt><dd>{t.elapsedMs != null ? `${(t.elapsedMs / 1000).toFixed(2)}s` : "—"}</dd>
                      <dt>Tok/s</dt><dd>{fmt(t.stats.tokens_per_sec)}</dd>
                      <dt>Steps</dt><dd>{fmt(t.stats.steps_to_converge ?? t.stats.total_denoising_steps)}</dd>
                    </dl>
                    {/* Diagnostics badges */}
                    {Object.keys(t.diagnostics).length > 0 && (
                      <div className={styles.diagnostics}>
                        {t.mode === "RCD" && (
                          <>
                            <span className={styles.diagBadge}>warm_start: {fmt(t.diagnostics.warm_start_used)}</span>
                            <span className={styles.diagBadge}>T_res: {fmt(t.diagnostics.T_res_used)}</span>
                            <span className={styles.diagBadge}>avg_α: {fmt(t.diagnostics.avg_alpha)}</span>
                          </>
                        )}
                        {t.mode === "OTS" && (
                          <>
                            <span className={styles.diagBadge}>beams: {fmt(t.diagnostics.total_beams_explored)}</span>
                            <span className={styles.diagBadge}>checkpoints: {fmt(t.diagnostics.total_search_checkpoints)}</span>
                            <span className={styles.diagBadge}>pruning: {fmt(t.diagnostics.pruning_mode_used)}</span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className={styles.composer}>
          <div className={styles.composerField}>
            <textarea
              placeholder={`Scrivi il prompt per ${mode}…`}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={running || loadingWeights || !weightsReady}
              rows={3}
            />
            {!weightsReady && !loadingWeights && (
              <p className={styles.composerLock}>Carica prima i pesi dLLM per abilitare l'inferenza.</p>
            )}
            {loadingWeights && (
              <p className={styles.composerLock}>Caricamento pesi in corso…</p>
            )}
          </div>
          <button onClick={send} disabled={running || !draft.trim() || loadingWeights || !weightsReady}>
            {running ? "…" : "Genera"}
          </button>
        </div>
      </section>

      {/* ── Config panel ── */}
      <section className={styles.config}>
        {/* ── Load Weights ── */}
        <div className={styles.block}>
          <h3>Load Weights</h3>
          <p className={styles.loadLead}>Carica i pesi dLLM prima di usare RCD / OTS.</p>
          <div className={styles.loadStateCard}>
            <div className={styles.loadStateHead}>
              <strong>dLLM</strong>
              <span className={`${styles.loadBadge} ${loadStatusClass(dllmLoad.status)}`}>
                {dllmLoad.status.toUpperCase()}
              </span>
            </div>
            <p>{dllmLoad.message}</p>
          </div>
          {dllmLoad.cpuFallback && (
            <p className={styles.cpuWarn}>CPU fallback attivo: throughput ridotto.</p>
          )}
          <button
            className={styles.loadBtn}
            disabled={running || loadingWeights}
            onClick={() => void handleLoadWeights()}
          >
            {loadingWeights ? "Caricamento pesi…" : "Carica pesi modello"}
          </button>
          {loadFeedback ? <p className={styles.loadFeedback}>{loadFeedback}</p> : null}
        </div>

        {/* ── Inference Realtime Log ── */}
        <div className={styles.block}>
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
        </div>

        <div className={styles.block}>
          <h3>Modalità Inferenza</h3>
          <div className={styles.modeSelector}>
            <button
              className={`${styles.modeBtn} ${mode === "RCD" ? styles.modeBtnActive : ""}`}
              onClick={() => setMode("RCD")}
            >
              RCD
            </button>
            <button
              className={`${styles.modeBtn} ${mode === "OTS" ? styles.modeBtnActive : ""}`}
              onClick={() => setMode("OTS")}
            >
              OTS
            </button>
          </div>
          <p style={{ margin: 0, color: "var(--text-dim)", fontSize: "0.78rem" }}>
            {mode === "RCD"
              ? "Residual Context Diffusion: ricicla i token scartati come prior contestuale."
              : "Order-Token Search: ricerca congiunta nell'ordine + spazio token."}
          </p>
        </div>

        <div className={styles.block}>
          <h3>Parametri Comuni</h3>
          <div className={styles.fieldRow}>
            <div className={styles.field}>
              <label>Max Tokens</label>
              <input
                type="number"
                min={32}
                max={4096}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </div>
            <div className={styles.field}>
              <label>Effort</label>
              <select value={effort} onChange={(e) => setEffort(e.target.value)}>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
          <div className={styles.field}>
            <label>Seed (vuoto = random)</label>
            <input
              type="number"
              value={seed ?? ""}
              onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : null)}
              placeholder="random"
            />
          </div>
        </div>

        {mode === "RCD" && (
          <div className={styles.block}>
            <h3>RCD Config</h3>
            <div className={styles.field}>
              <label>T_res (temperatura residui)</label>
              <input
                type="number"
                min={0.1}
                max={5.0}
                step={0.1}
                value={rcd.rcd_temperature_residual}
                onChange={(e) => setRcd((prev) => ({ ...prev, rcd_temperature_residual: Number(e.target.value) }))}
              />
            </div>
            <div className={styles.field}>
              <label>
                <input
                  type="checkbox"
                  checked={rcd.rcd_warm_start}
                  onChange={(e) => setRcd((prev) => ({ ...prev, rcd_warm_start: e.target.checked }))}
                />{" "}
                Warm Start
              </label>
            </div>
          </div>
        )}

        {mode === "OTS" && (
          <div className={styles.block}>
            <h3>OTS Config</h3>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Beam Size</label>
                <input
                  type="number"
                  min={1}
                  max={16}
                  value={ots.ots_beam_size}
                  onChange={(e) => setOts((prev) => ({ ...prev, ots_beam_size: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Gumbel τ</label>
                <input
                  type="number"
                  min={0}
                  max={2.0}
                  step={0.1}
                  value={ots.ots_gumbel_temperature}
                  onChange={(e) => setOts((prev) => ({ ...prev, ots_gumbel_temperature: Number(e.target.value) }))}
                />
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

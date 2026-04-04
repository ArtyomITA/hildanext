import { KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import { MODEL_CATALOG, DLLM_MODEL_PATHS } from "../../features/chat/catalog";
import styles from "./InferencePlusPage.module.css";

type InferenceMode = "RCD" | "OTS" | "S2D2" | "EntRGi";

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
  rcd_latent_logits_temperature: string;
  rcd_warm_start: boolean;
  rcd_reference_model: string;
  rcd_same_model_warm_start_fallback: boolean;
  rcd_single_token_mode: boolean;
  rcd_force_mask_only_injection: boolean;
}

interface OtsConfig {
  ots_beam_size: number;
  ots_gumbel_temperature: number;
  ots_search_interval: number;
  ots_block_size: number;
  ots_pruning_mode: "diffusion_likelihood" | "fallback_confidence";
}

interface S2d2Config {
  s2d2_block_size: number;
  s2d2_denoising_steps: number;
  s2d2_routing_policy: "min_span" | "score_threshold" | "hysteresis" | "always" | "never";
  s2d2_min_verify_span: number;
  s2d2_score_threshold: number;
  s2d2_confidence_threshold: number;
  s2d2_acceptance_estimator: "entropy" | "margin";
  s2d2_entropy_beta: number;
}

function defaultRcd(): RcdConfig {
  return {
    rcd_temperature_residual: 1.0,
    rcd_latent_logits_temperature: "",
    rcd_warm_start: true,
    rcd_reference_model: "",
    rcd_same_model_warm_start_fallback: true,
    rcd_single_token_mode: true,
    rcd_force_mask_only_injection: true,
  };
}

function defaultOts(): OtsConfig {
  return {
    ots_beam_size: 3,
    ots_gumbel_temperature: 0.6,
    ots_search_interval: 0,
    ots_block_size: 32,
    ots_pruning_mode: "diffusion_likelihood",
  };
}

function defaultS2d2(): S2d2Config {
  return {
    s2d2_block_size: 32,
    s2d2_denoising_steps: 0,
    s2d2_routing_policy: "min_span",
    s2d2_min_verify_span: 2,
    s2d2_score_threshold: 0.0,
    s2d2_confidence_threshold: 0.3,
    s2d2_acceptance_estimator: "entropy",
    s2d2_entropy_beta: 1.0,
  };
}

interface EntRGiConfig {
  entrgi_guidance_scale: number;
  entrgi_guidance_steps: number;
  entrgi_temperature: number;
  entrgi_confidence_threshold: number;
  entrgi_disable_guidance: boolean;
}

function defaultEntRGi(): EntRGiConfig {
  return {
    entrgi_guidance_scale: 0.5,
    entrgi_guidance_steps: 3,
    entrgi_temperature: 0.7,
    entrgi_confidence_threshold: 0.3,
    entrgi_disable_guidance: false,
  };
}

function parseOptionalNumber(raw: string): number | undefined {
  const value = raw.trim();
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function describeRcdModelUsage(config: RcdConfig): {
  title: string;
  body: string;
  tone: "calm" | "accent";
} {
  if (!config.rcd_warm_start) {
    return {
      title: "1 modello",
      body: "Il target model genera direttamente senza warm-start RCD. Nessun reference model viene caricato.",
      tone: "calm",
    };
  }
  if (config.rcd_reference_model.trim()) {
    return {
      title: "2 modelli",
      body: "Il target model genera i token. Il reference model separato viene caricato solo per inizializzare il warm-start RCD.",
      tone: "accent",
    };
  }
  return {
    title: "1 modello",
    body: "Il target model fa sia warm-start sia generazione. E' la modalita piu leggera, ma meno fedele alla recipe ufficiale con Mref separato.",
    tone: "calm",
  };
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

function toNumberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
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
  const [s2d2, setS2d2] = useState(defaultS2d2);
  const [entrgi, setEntRGi] = useState(defaultEntRGi);
  const [dllmModelId, setDllmModelId] = useState("dllm_wsd_step_04000");
  const dllmModels = MODEL_CATALOG.filter((m) => m.lane === "dllm");
  const turnsRef = useRef<HTMLDivElement>(null);
  const abortCtrlRef = useRef<AbortController | null>(null);

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
  const rcdUsage = describeRcdModelUsage(rcd);

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

  function appendFrontendDiagnosticEvent(mode: InferenceMode, diagnostics: Record<string, unknown>, stats: Record<string, unknown>) {
    if (mode !== "S2D2" || Object.keys(diagnostics).length === 0) return;
    const routeScore = toNumberOrNull(diagnostics.last_route_score);
    appendRealtimeEvents([
      {
        id: `frontend-s2d2-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        tsUtc: new Date().toISOString(),
        level: "notice",
        source: "frontend",
        event: "S2D2_DIAGNOSTICS",
        lane: "s2d2",
        scope: "inference",
        benchmark: null,
        message:
          `verify=${fmt(diagnostics.verifier_invocations)} skip=${fmt(diagnostics.verifier_skips)} ` +
          `block=${fmt(diagnostics.last_block_index)} span=${fmt(diagnostics.last_span_start)}-${fmt(diagnostics.last_span_end)} ` +
          `route=${routeScore == null ? "—" : routeScore.toFixed(3)} verifier=${fmt(diagnostics.verifier_mode_used)} ` +
          `kv=${fmt(diagnostics.kv_cache_mode)} cache_hits=${fmt(diagnostics.verifier_cache_hits)}`,
        meta: {
          mode,
          verifier_invocations: diagnostics.verifier_invocations,
          verifier_skips: diagnostics.verifier_skips,
          avg_accepted_prefix_length: diagnostics.avg_accepted_prefix_length,
          fallback_to_diffusion_count: diagnostics.fallback_to_diffusion_count,
          routing_policy_used: diagnostics.routing_policy_used,
          verifier_mode_used: diagnostics.verifier_mode_used,
          kv_cache_mode: diagnostics.kv_cache_mode,
          verifier_cache_hits: diagnostics.verifier_cache_hits,
          last_block_index: diagnostics.last_block_index,
          last_span_start: diagnostics.last_span_start,
          last_span_end: diagnostics.last_span_end,
          last_route_score: diagnostics.last_route_score,
          stats_verifier_mode: stats.verifier_mode,
          stats_kv_cache_mode: stats.kv_cache_mode,
        },
      },
    ]);
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
        const _handleSseMsg = (event: MessageEvent<string>) => {
          if (closed) return;
          let parsed: unknown = {};
          try { parsed = JSON.parse(String(event.data || "{}")); } catch { return; }
          const row = normalizeRealtimeEvent(parsed);
          if (!row) return;
          appendRealtimeEvents([row]);
        };
        stream.onmessage = _handleSseMsg;
        // Backend emits named event "inference", not default "message"
        stream.addEventListener("inference", _handleSseMsg as EventListener);
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
          model_override: DLLM_MODEL_PATHS[dllmModelId] ?? undefined,
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

  function handleCancel() {
    abortCtrlRef.current?.abort();
  }

  async function send() {
    const prompt = draft.trim();
    if (!prompt || running) return;
    setDraft("");
    abortCtrlRef.current?.abort();
    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;
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
      const endpoint =
        mode === "RCD" ? "/api/inferencercdm"
        : mode === "OTS" ? "/api/inferenceots"
        : mode === "S2D2" ? "/api/inferences2d2"
        : "/api/inferenceentrgi";
      const extra =
        mode === "RCD"
          ? {
              rcd_temperature_residual: rcd.rcd_temperature_residual,
              rcd_latent_logits_temperature: parseOptionalNumber(rcd.rcd_latent_logits_temperature),
              rcd_warm_start: rcd.rcd_warm_start,
              rcd_reference_model: rcd.rcd_reference_model.trim() || undefined,
              rcd_same_model_warm_start_fallback: rcd.rcd_same_model_warm_start_fallback,
              rcd_single_token_mode: rcd.rcd_single_token_mode,
              rcd_force_mask_only_injection: rcd.rcd_force_mask_only_injection,
            }
          : mode === "OTS"
          ? {
              ots_beam_size: ots.ots_beam_size,
              ots_block_size: ots.ots_block_size,
              ots_gumbel_temperature: ots.ots_gumbel_temperature,
              ots_search_interval: ots.ots_search_interval,
              ots_pruning_mode: ots.ots_pruning_mode,
            }
          : mode === "S2D2"
          ? {
              s2d2_block_size: s2d2.s2d2_block_size,
              s2d2_denoising_steps: s2d2.s2d2_denoising_steps,
              s2d2_routing_policy: s2d2.s2d2_routing_policy,
              s2d2_min_verify_span: s2d2.s2d2_min_verify_span,
              s2d2_score_threshold: s2d2.s2d2_score_threshold,
              s2d2_confidence_threshold: s2d2.s2d2_confidence_threshold,
              s2d2_acceptance_estimator: s2d2.s2d2_acceptance_estimator,
              s2d2_entropy_beta: s2d2.s2d2_entropy_beta,
            }
          : {
              entrgi_guidance_scale: entrgi.entrgi_guidance_scale,
              entrgi_guidance_steps: entrgi.entrgi_guidance_steps,
              entrgi_temperature: entrgi.entrgi_temperature,
              entrgi_confidence_threshold: entrgi.entrgi_confidence_threshold,
              entrgi_disable_guidance: entrgi.entrgi_disable_guidance,
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
        signal: ctrl.signal,
      });
      const elapsed = performance.now() - t0;
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(body || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      const stats = payload.stats ?? {};
      const diagnostics = payload.diagnostics ?? {};
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? {
                ...t,
                status: "success" as const,
                text: String(payload.text ?? ""),
                engine: String(payload.engine ?? mode),
                stats,
                diagnostics,
                elapsedMs: elapsed,
              }
            : t,
        ),
      );
      appendFrontendDiagnosticEvent(mode, diagnostics, stats);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setTurns((prev) => prev.filter((t) => t.id !== id));
        setRunning(false);
        return;
      }
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
                    <strong>{t.mode}</strong>
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
                      <dt>Steps</dt><dd>{fmt(t.stats.steps ?? t.stats.total_denoising_steps)}</dd>
                    </dl>
                    {/* Diagnostics badges */}
                    {Object.keys(t.diagnostics).length > 0 && (
                      <div className={styles.diagnostics}>
                        {t.mode === "RCD" && (
                          <>
                            <span className={styles.diagBadge}>warm_start: {fmt(t.diagnostics.warm_start_used)}</span>
                            <span className={styles.diagBadge}>ref_model: {fmt(t.diagnostics.reference_model_used)}</span>
                            <span className={styles.diagBadge}>T_res: {fmt(t.diagnostics.t_res)}</span>
                            <span className={styles.diagBadge}>steps: {fmt(t.diagnostics.total_denoising_steps)}</span>
                          </>
                        )}
                        {t.mode === "OTS" && (
                          <>
                            <span className={styles.diagBadge}>beams: {fmt(t.diagnostics.total_beams_explored)}</span>
                            <span className={styles.diagBadge}>checkpoints: {fmt(t.diagnostics.total_search_checkpoints)}</span>
                            <span className={styles.diagBadge}>block_size: {fmt(t.stats.block_size)}</span>
                            <span className={styles.diagBadge}>pruning: {fmt(t.diagnostics.pruning_mode_used)}</span>
                            <span className={styles.diagBadge}>best_score: {fmt(t.diagnostics.chosen_beam_score)}</span>
                          </>
                        )}
                        {t.mode === "S2D2" && (
                          <>
                            <span className={styles.diagBadge}>verify: {fmt(t.diagnostics.verifier_invocations)}</span>
                            <span className={styles.diagBadge}>skip: {fmt(t.diagnostics.verifier_skips)}</span>
                            <span className={styles.diagBadge}>avg_accept: {fmt(t.diagnostics.avg_accepted_prefix_length)}</span>
                            <span className={styles.diagBadge}>fallback: {fmt(t.diagnostics.fallback_to_diffusion_count)}</span>
                            <span className={styles.diagBadge}>routing: {fmt(t.diagnostics.routing_policy_used)}</span>
                            <span className={styles.diagBadge}>block_size: {fmt(t.diagnostics.block_size)}</span>
                            <span className={styles.diagBadge}>verifier: {fmt(t.diagnostics.verifier_mode_used)}</span>
                            <span className={styles.diagBadge}>kv_cache: {fmt(t.diagnostics.kv_cache_mode)}</span>
                            <span className={styles.diagBadge}>cache_hits: {fmt(t.diagnostics.verifier_cache_hits)}</span>
                            <span className={styles.diagBadge}>last_block: {fmt(t.diagnostics.last_block_index)}</span>
                            <span className={styles.diagBadge}>last_span: {fmt(t.diagnostics.last_span_start)}-{fmt(t.diagnostics.last_span_end)}</span>
                            <span className={styles.diagBadge}>route_score: {fmt(t.diagnostics.last_route_score)}</span>
                          </>
                        )}
                        {t.mode === "EntRGi" && (
                          <>
                            <span className={styles.diagBadge}>reward: {fmt(t.diagnostics.reward_model_loaded)}</span>
                            <span className={styles.diagBadge}>tok_align: {fmt(t.diagnostics.reward_tokenizer_aligned)}</span>
                            <span className={styles.diagBadge}>align_mode: {fmt(t.diagnostics.tokenizer_alignment_mode)}</span>
                            <span className={styles.diagBadge}>η: {fmt(t.diagnostics.guidance_scale)}</span>
                            <span className={styles.diagBadge}>M: {fmt(t.diagnostics.guidance_steps)}</span>
                            <span className={styles.diagBadge}>calls: {fmt(t.diagnostics.number_of_guidance_calls)}</span>
                            <span className={styles.diagBadge}>select: {fmt(t.diagnostics.selection_policy_used)}</span>
                            <span className={styles.diagBadge}>avg_H: {fmt(t.diagnostics.avg_masked_entropy)}</span>
                            <span className={styles.diagBadge}>avg_sel_H: {fmt(t.diagnostics.avg_selected_entropy)}</span>
                            <span className={styles.diagBadge}>avg_w: {fmt(t.diagnostics.avg_entropy_weight)}</span>
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
          <button onClick={running ? handleCancel : send} disabled={loadingWeights || !weightsReady || (!running && !draft.trim())}>
            {running ? "Annulla" : "Genera"}
          </button>
        </div>
      </section>

      {/* ── Config panel ── */}
      <section className={styles.config}>
        {/* ── Load Weights ── */}
        <div className={styles.block}>
          <h3>Load Weights</h3>
          <p className={styles.loadLead}>Carica i pesi dLLM prima di usare RCD / OTS.</p>
          <label style={{ display: "block", marginBottom: 8 }}>
            <span style={{ fontSize: "0.78rem", color: "var(--text-dim)" }}>Modello dLLM</span>
            <select
              value={dllmModelId}
              onChange={(e) => setDllmModelId(e.target.value)}
              disabled={running || loadingWeights}
              style={{ display: "block", width: "100%", marginTop: 2 }}
            >
              {dllmModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <span style={{ fontSize: "0.7rem", color: "var(--text-dim)", opacity: 0.7 }}>
              {dllmModels.find((m) => m.id === dllmModelId)?.description ?? ""}
            </span>
          </label>
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
            <button
              className={`${styles.modeBtn} ${mode === "S2D2" ? styles.modeBtnActive : ""}`}
              onClick={() => setMode("S2D2")}
            >
              S2D2
            </button>
            <button
              className={`${styles.modeBtn} ${mode === "EntRGi" ? styles.modeBtnActive : ""}`}
              onClick={() => setMode("EntRGi")}
            >
              EntRGi
            </button>
          </div>
          <p style={{ margin: 0, color: "var(--text-dim)", fontSize: "0.78rem" }}>
            {mode === "RCD"
              ? "Residual Context Diffusion: ricicla i token scartati come prior contestuale."
              : mode === "OTS"
              ? "Order-Token Search: ricerca congiunta nell'ordine + spazio token."
              : mode === "S2D2"
              ? "S2D2: self-speculative decoding training-free. Stesso modello = drafter + verifier AR."
              : "EntRGi: reward guidance entropy-aware. Gradienti dal reward model guidano i logit mascherati."}
          </p>
          {mode === "RCD" ? (
            <div className={`${styles.explainCard} ${rcdUsage.tone === "accent" ? styles.explainAccent : ""}`}>
              <div className={styles.explainHead}>
                <strong>{rcdUsage.title}</strong>
                <span>RCD runtime</span>
              </div>
              <p>{rcdUsage.body}</p>
            </div>
          ) : mode === "OTS" ? (
            <div className={styles.explainCard}>
              <div className={styles.explainHead}>
                <strong>Ricerca a beam</strong>
                <span>OTS runtime</span>
              </div>
              <p>
                OTS usa un solo modello, ma moltiplica le traiettorie attive con i beam e i checkpoint.
                I parametri sotto controllano quanto ramifica e come pota i candidati.
              </p>
            </div>
          ) : mode === "S2D2" ? (
            <div className={styles.explainCard}>
              <div className={styles.explainHead}>
                <strong>Self-Speculation</strong>
                <span>S2D2 runtime</span>
              </div>
              <p>
                S2D2 usa lo stesso modello come drafter (diffusion) e verifier (AR block-size-1).
                Nessun retraining: il routing decide quando verificare e il rejection sampling corregge i draft.
              </p>
            </div>
          ) : (
            <div className={styles.explainCard}>
              <div className={styles.explainHead}>
                <strong>Reward Guidance</strong>
                <span>EntRGi runtime</span>
              </div>
              <p>
                EntRGi usa un reward model frozen (Skywork-Reward-V2-Qwen3-0.6B) per guidare i logit
                alle posizioni mascherate. L'interpolazione entropy-aware bilancia soft/hard embedding.
              </p>
            </div>
          )}
        </div>
        <div className={styles.block}>
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
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>T_res</label>
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
                <label>Latent Logits Temp</label>
                <input
                  type="number"
                  min={0.1}
                  max={5.0}
                  step={0.01}
                  value={rcd.rcd_latent_logits_temperature}
                  onChange={(e) => setRcd((prev) => ({ ...prev, rcd_latent_logits_temperature: e.target.value }))}
                  placeholder="vuoto = usa T_res"
                />
              </div>
            </div>
            <div className={styles.fieldCheck}>
              <label>
                <input
                  type="checkbox"
                  checked={rcd.rcd_warm_start}
                  onChange={(e) => setRcd((prev) => ({ ...prev, rcd_warm_start: e.target.checked }))}
                />
                <span>Warm Start RCD</span>
              </label>
              <small>Se disattivo, usa solo il target model e salta l'inizializzazione residua.</small>
            </div>
            {rcd.rcd_warm_start && (
              <>
                <div className={styles.field}>
                  <label>Reference Model Path</label>
                  <input
                    type="text"
                    value={rcd.rcd_reference_model}
                    onChange={(e) => setRcd((prev) => ({ ...prev, rcd_reference_model: e.target.value }))}
                    placeholder="opzionale: path locale a Mref"
                  />
                </div>
                <div className={styles.fieldCheck}>
                  <label>
                    <input
                      type="checkbox"
                      checked={rcd.rcd_same_model_warm_start_fallback}
                      onChange={(e) => setRcd((prev) => ({ ...prev, rcd_same_model_warm_start_fallback: e.target.checked }))}
                    />
                    <span>Fallback allo stesso target model se Mref manca</span>
                  </label>
                </div>
              </>
            )}
            <div className={styles.fieldCheck}>
              <label>
                <input
                  type="checkbox"
                  checked={rcd.rcd_single_token_mode}
                  onChange={(e) => setRcd((prev) => ({ ...prev, rcd_single_token_mode: e.target.checked }))}
                />
                <span>Single-token-per-step</span>
              </label>
              <small>Consigliato per recipe LLaDA/RCD ufficiali e per setup Qwen + WSD LLaDA.</small>
            </div>
            <div className={styles.fieldCheck}>
              <label>
                <input
                  type="checkbox"
                  checked={rcd.rcd_force_mask_only_injection}
                  onChange={(e) => setRcd((prev) => ({ ...prev, rcd_force_mask_only_injection: e.target.checked }))}
                />
                <span>Residual injection solo su [MASK]</span>
              </label>
            </div>
            <p className={styles.inlineNote}>
              `Latent Logits Temp` corrisponde al naming usato nella repo ufficiale RCD-LLaDA.
              Se lo lasci vuoto, il backend riusa `T_res`.
            </p>
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
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Block Size</label>
                <input
                  type="number"
                  min={1}
                  max={512}
                  step={8}
                  value={ots.ots_block_size}
                  onChange={(e) => setOts((prev) => ({ ...prev, ots_block_size: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Search Interval</label>
                <input
                  type="number"
                  min={0}
                  max={256}
                  value={ots.ots_search_interval}
                  onChange={(e) => setOts((prev) => ({ ...prev, ots_search_interval: Number(e.target.value) }))}
                />
              </div>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Pruning Mode</label>
                <select
                  value={ots.ots_pruning_mode}
                  onChange={(e) =>
                    setOts((prev) => ({
                      ...prev,
                      ots_pruning_mode: e.target.value as OtsConfig["ots_pruning_mode"],
                    }))
                  }
                >
                  <option value="diffusion_likelihood">diffusion_likelihood</option>
                  <option value="fallback_confidence">fallback_confidence</option>
                </select>
              </div>
            </div>
            <p className={styles.inlineNote}>
              `Block Size = 32` è il default del paper (Alg.1). `Search Interval = 0` = auto.
              `diffusion_likelihood` è fedele a Eq.(2), `fallback_confidence` è diagnostica.
            </p>
          </div>
        )}

        {mode === "S2D2" && (
          <div className={styles.block}>
            <h3>S2D2 Config</h3>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Block Size</label>
                <input
                  type="number"
                  min={1}
                  max={512}
                  step={8}
                  value={s2d2.s2d2_block_size}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_block_size: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Denoising Steps (0=auto)</label>
                <input
                  type="number"
                  min={0}
                  max={256}
                  value={s2d2.s2d2_denoising_steps}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_denoising_steps: Number(e.target.value) }))}
                />
              </div>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Routing Policy</label>
                <select
                  value={s2d2.s2d2_routing_policy}
                  onChange={(e) =>
                    setS2d2((prev) => ({
                      ...prev,
                      s2d2_routing_policy: e.target.value as S2d2Config["s2d2_routing_policy"],
                    }))
                  }
                >
                  <option value="min_span">min_span</option>
                  <option value="score_threshold">score_threshold</option>
                  <option value="hysteresis">hysteresis</option>
                  <option value="always">always</option>
                  <option value="never">never</option>
                </select>
              </div>
              <div className={styles.field}>
                <label>Min Verify Span (τ_span)</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={s2d2.s2d2_min_verify_span}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_min_verify_span: Number(e.target.value) }))}
                />
              </div>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Confidence Threshold (τ)</label>
                <input
                  type="number"
                  min={0}
                  max={1.0}
                  step={0.05}
                  value={s2d2.s2d2_confidence_threshold}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_confidence_threshold: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Acceptance Estimator</label>
                <select
                  value={s2d2.s2d2_acceptance_estimator}
                  onChange={(e) =>
                    setS2d2((prev) => ({
                      ...prev,
                      s2d2_acceptance_estimator: e.target.value as S2d2Config["s2d2_acceptance_estimator"],
                    }))
                  }
                >
                  <option value="entropy">entropy</option>
                  <option value="margin">margin</option>
                </select>
              </div>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Entropy β</label>
                <input
                  type="number"
                  min={0.01}
                  max={5.0}
                  step={0.25}
                  value={s2d2.s2d2_entropy_beta}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_entropy_beta: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Score Threshold (τ_score)</label>
                <input
                  type="number"
                  min={-10}
                  max={10}
                  step={0.5}
                  value={s2d2.s2d2_score_threshold}
                  onChange={(e) => setS2d2((prev) => ({ ...prev, s2d2_score_threshold: Number(e.target.value) }))}
                />
              </div>
            </div>
            <p className={styles.inlineNote}>
              S2D2 (arXiv:2603.25702): training-free self-speculative decoding.
              `min_span` = verifica se lo span mascherato ≥ τ_span.
              `always` = verifica ogni step. `never` = solo diffusion.
            </p>
          </div>
        )}

        {mode === "EntRGi" && (
          <div className={styles.block}>
            <h3>EntRGi Config</h3>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Guidance Scale (η)</label>
                <input
                  type="number"
                  min={0}
                  max={5.0}
                  step={0.1}
                  value={entrgi.entrgi_guidance_scale}
                  onChange={(e) => setEntRGi((prev) => ({ ...prev, entrgi_guidance_scale: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Guidance Steps (M)</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={entrgi.entrgi_guidance_steps}
                  onChange={(e) => setEntRGi((prev) => ({ ...prev, entrgi_guidance_steps: Number(e.target.value) }))}
                />
              </div>
            </div>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Temperature (τ)</label>
                <input
                  type="number"
                  min={0.01}
                  max={5.0}
                  step={0.1}
                  value={entrgi.entrgi_temperature}
                  onChange={(e) => setEntRGi((prev) => ({ ...prev, entrgi_temperature: Number(e.target.value) }))}
                />
              </div>
              <div className={styles.field}>
                <label>Confidence Threshold</label>
                <input
                  type="number"
                  min={0}
                  max={1.0}
                  step={0.05}
                  value={entrgi.entrgi_confidence_threshold}
                  onChange={(e) => setEntRGi((prev) => ({ ...prev, entrgi_confidence_threshold: Number(e.target.value) }))}
                />
              </div>
            </div>
            <div className={styles.fieldCheck}>
              <label>
                <input
                  type="checkbox"
                  checked={entrgi.entrgi_disable_guidance}
                  onChange={(e) => setEntRGi((prev) => ({ ...prev, entrgi_disable_guidance: e.target.checked }))}
                />
                <span>Disabilita guidance (ablazione)</span>
              </label>
              <small>Se attivo, bypassa il reward model e usa solo denoising standard.</small>
            </div>
            <p className={styles.inlineNote}>
              EntRGi (arXiv:2602.05000): entropy-aware reward guidance. η=0.5, M=3, τ=0.7 sono i default del paper.
              Reward model: Skywork-Reward-V2-Qwen3-0.6B (caricato automaticamente).
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

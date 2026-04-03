import { Dispatch, SetStateAction, useMemo, useRef, useState } from "react";
import { Panel } from "../../components/layout/Panel";
import { TimeseriesChart } from "../../components/charts/TimeseriesChart";
import styles from "./Stage0Validation.module.css";

const AR_BASELINE_PCT = 45;
const MMLU_PRO_BASELINE_PCT = 24.7;
const GSM8K_BASELINE_PCT = 59.6;
const MAX_BENCH_LOGS = 120;

const HELLASWAG_SUBSET_LIMIT = 8;
const MMLU_PRO_SUBSET_LIMIT = 150;
const GSM8K_SUBSET_LIMIT = 100;

type BenchScope = "AR" | "DLLM" | "BOTH" | "RCD" | "OTS" | "S2D2" | "ENTRGI";
type GlobalEffort = "low" | "medium" | "high";
type DecodeStrategy = "greedy" | "sampling";
type StabilityMaskSchedule = "linear" | "cosine";

const EFFORT_MAX_NEW_TOKENS: Record<GlobalEffort, number> = {
  low: 256,
  medium: 1024,
  high: 2048,
};

interface HellaSwagItem {
  id: string;
  stem: string;
  endings: [string, string, string, string];
  label: number;
}

interface MmluProItem {
  id: string;
  question: string;
  options: string[];
  answer_label: string;
}

interface Gsm8kItem {
  id: string;
  question: string;
  answer_target: string;
}

interface ItemsResponse<TItem> {
  items?: TItem[];
  dataset_path?: string;
  dataset_id?: string;
  source?: string;
  split?: string;
  config?: string;
  total_items?: number;
}

interface StabilityPoint {
  step: number;
  mean_confidence: number;
  mask_ratio: number;
}

type BenchLogLevel = "info" | "ok" | "warn" | "error";

interface BenchLogEntry {
  id: string;
  ts: string;
  level: BenchLogLevel;
  message: string;
  details?: string;
}

type BenchmarkKind = "hellaswag" | "mmlu-pro" | "gsm8k" | "stability";

interface ScoreboardEntry {
  id: string;
  benchmark: BenchmarkKind;
  model: string;
  effort: GlobalEffort;
  decodeStrategy: DecodeStrategy;
  startedAtIso: string;
  elapsedMs: number;
  accuracy: number | null;
  correct: number;
  total: number;
  dataLabel: string;
  tokens: number | null;
}

interface DetailedLogHandle {
  token: string;
  filePath: string;
  fileName: string;
}

function pct(v: number | null) {
  if (v === null || Number.isNaN(v)) return "n/a";
  return `${v.toFixed(1)}%`;
}

function clampPct(v: number) {
  return Math.max(0, Math.min(100, v));
}

function scopeLabel(scope: BenchScope) {
  if (scope === "AR") return "AR";
  if (scope === "DLLM") return "dLLM";
  if (scope === "RCD") return "RCD";
  if (scope === "OTS") return "OTS";
  return "AR+dLLM";
}

function normalizeSpace(input: string) {
  return String(input || "").replace(/\s+/g, " ").trim();
}

function trimForLog(input: string, max = 220) {
  const base = normalizeSpace(input);
  if (base.length <= max) return base;
  return `${base.slice(0, max - 1)}…`;
}

function parseTokensFromStats(stats: unknown): number {
  if (!stats || typeof stats !== "object") return 0;
  const direct = (stats as Record<string, unknown>).tokens_generated;
  if (typeof direct === "number" && Number.isFinite(direct) && direct > 0) return direct;
  const nested = (stats as Record<string, unknown>).stats;
  if (nested && typeof nested === "object") {
    const n = (nested as Record<string, unknown>).tokens_generated;
    if (typeof n === "number" && Number.isFinite(n) && n > 0) return n;
  }
  return 0;
}

function makeLog(level: BenchLogLevel, message: string, details?: string): BenchLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    ts: new Date().toLocaleTimeString(),
    level,
    message,
    details: details && details.trim() ? details.trim() : undefined,
  };
}

function pushLog(
  setter: Dispatch<SetStateAction<BenchLogEntry[]>>,
  level: BenchLogLevel,
  message: string,
  details?: string,
) {
  setter((prev) => [...prev.slice(-(MAX_BENCH_LOGS - 1)), makeLog(level, message, details)]);
}

function BenchmarkGauge({
  score,
  baseline,
  legend,
  modelLabel,
}: {
  score: number | null;
  baseline: number;
  legend: string[];
  modelLabel: string;
}) {
  const barWidth = clampPct(score ?? 0);
  return (
    <div className={styles.gauge}>
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${barWidth}%` }} />
        <div className={styles.baseline} style={{ left: `${baseline}%` }} />
      </div>
      <div className={styles.legend}>
        <span>
          {modelLabel}: {pct(score)}
        </span>
        <span>Target: {baseline.toFixed(1)}%</span>
        {legend.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>
    </div>
  );
}

function Progress({ value }: { value: number }) {
  return (
    <div className={styles.progressWrap} aria-live="polite">
      <progress max={100} value={value} />
      <span>{value.toFixed(0)}%</span>
    </div>
  );
}

function BenchmarkLogs({ logs }: { logs: BenchLogEntry[] }) {
  return (
    <div className={styles.logBox}>
      <div className={styles.logHead}>Log realtime (max {MAX_BENCH_LOGS})</div>
      <div className={styles.logBody}>
        {logs.length === 0 ? (
          <p className={styles.logEmpty}>Nessun log.</p>
        ) : (
          logs.map((row) => (
            <div key={row.id} className={styles.logItem}>
              <p className={styles.logRow} data-level={row.level}>
                <span>{row.ts}</span>
                <span>{row.message}</span>
              </p>
              {row.details ? (
                <details className={styles.logDetails}>
                  <summary>Dettagli</summary>
                  <pre>{row.details}</pre>
                </details>
              ) : null}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function Stage0Validation() {
  const [busy, setBusy] = useState<"none" | "hellaswag" | "stability" | "mmlu" | "gsm8k">("none");

  const [modelScope, setModelScope] = useState<BenchScope>("DLLM");
  const [contextWindow, setContextWindow] = useState<1024 | 2048 | 4096 | 8192>(4096);
  const [generationEffort, setGenerationEffort] = useState<GlobalEffort>("medium");
  const [decodeStrategy, setDecodeStrategy] = useState<DecodeStrategy>("greedy");
  const [detailedBenchLogs, setDetailedBenchLogs] = useState(false);
  const [persistDetailedLogToFile, setPersistDetailedLogToFile] = useState(false);
  const [logRunLabel, setLogRunLabel] = useState("");
  const [lastDetailedLogPath, setLastDetailedLogPath] = useState("");
  const [showScoreboard, setShowScoreboard] = useState(false);
  const [scoreboard, setScoreboard] = useState<ScoreboardEntry[]>([]);
  const [scoreboardFilter, setScoreboardFilter] = useState<"all" | BenchmarkKind>("all");
  const [runPaused, setRunPaused] = useState(false);
  const [runStopRequested, setRunStopRequested] = useState(false);
  const pauseRef = useRef(false);
  const stopRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const [hellaNShots, setHellaNShots] = useState<0 | 3 | 5>(0);
  const [mmluNShots, setMmluNShots] = useState<0 | 5>(0);
  const [mmluForceCot, setMmluForceCot] = useState(true);
  const [gsmNShots, setGsmNShots] = useState<0 | 4 | 8>(0);
  const [stabilityTotalSteps, setStabilityTotalSteps] = useState(50);
  const [stabilityMaskSchedule, setStabilityMaskSchedule] = useState<StabilityMaskSchedule>("cosine");

  const [hellaProgress, setHellaProgress] = useState(0);
  const [hellaCorrect, setHellaCorrect] = useState(0);
  const [hellaTotal, setHellaTotal] = useState(0);
  const [hellaAccuracy, setHellaAccuracy] = useState<number | null>(null);
  const [hellaError, setHellaError] = useState("");
  const [hellaDatasetPath, setHellaDatasetPath] = useState("");
  const [hellaDatasetSource, setHellaDatasetSource] = useState("");
  const [hellaDatasetTotal, setHellaDatasetTotal] = useState<number | null>(null);
  const [hellaLogs, setHellaLogs] = useState<BenchLogEntry[]>([]);

  const [mmluProgress, setMmluProgress] = useState(0);
  const [mmluCorrect, setMmluCorrect] = useState(0);
  const [mmluTotal, setMmluTotal] = useState(0);
  const [mmluAccuracy, setMmluAccuracy] = useState<number | null>(null);
  const [mmluError, setMmluError] = useState("");
  const [mmluDataset, setMmluDataset] = useState("");
  const [mmluDatasetSource, setMmluDatasetSource] = useState("");
  const [mmluDatasetSplit, setMmluDatasetSplit] = useState("");
  const [mmluDatasetTotal, setMmluDatasetTotal] = useState<number | null>(null);
  const [mmluLogs, setMmluLogs] = useState<BenchLogEntry[]>([]);

  const [gsmProgress, setGsmProgress] = useState(0);
  const [gsmCorrect, setGsmCorrect] = useState(0);
  const [gsmTotal, setGsmTotal] = useState(0);
  const [gsmAccuracy, setGsmAccuracy] = useState<number | null>(null);
  const [gsmError, setGsmError] = useState("");
  const [gsmDataset, setGsmDataset] = useState("");
  const [gsmDatasetSource, setGsmDatasetSource] = useState("");
  const [gsmDatasetSplit, setGsmDatasetSplit] = useState("");
  const [gsmDatasetTotal, setGsmDatasetTotal] = useState<number | null>(null);
  const [gsmLogs, setGsmLogs] = useState<BenchLogEntry[]>([]);

  const [stabilityPrompt, setStabilityPrompt] = useState("The capital of France is");
  const [stabilityPoints, setStabilityPoints] = useState<StabilityPoint[]>([]);
  const [stabilityOutput, setStabilityOutput] = useState("");
  const [stabilityError, setStabilityError] = useState("");
  const [stabilityLogs, setStabilityLogs] = useState<BenchLogEntry[]>([]);

  const anyRunning = busy !== "none";
  const hellaRunning = busy === "hellaswag";
  const mmluRunning = busy === "mmlu";
  const gsmRunning = busy === "gsm8k";
  const stabilityRunning = busy === "stability";
  const stabilityDisabled = modelScope === "AR";

  const maxNewTokens = EFFORT_MAX_NEW_TOKENS[generationEffort];
  const decodeParams = useMemo(() => {
    if (decodeStrategy === "sampling") {
      return { temperature: 0.6, top_p: 0.9 };
    }
    return { temperature: 0.0, top_p: 1.0 };
  }, [decodeStrategy]);

  const filteredScoreboard = useMemo(() => {
    if (scoreboardFilter === "all") return scoreboard;
    return scoreboard.filter((row) => row.benchmark === scoreboardFilter);
  }, [scoreboard, scoreboardFilter]);

  function resetRunControlState() {
    pauseRef.current = false;
    stopRef.current = false;
    setRunPaused(false);
    setRunStopRequested(false);
  }

  async function waitIfPausedOrStop() {
    while (pauseRef.current && !stopRef.current) {
      await new Promise((resolve) => setTimeout(resolve, 220));
    }
    if (stopRef.current) {
      throw new Error("__BENCH_STOPPED__");
    }
  }

  function togglePauseRun() {
    if (busy === "none") return;
    pauseRef.current = !pauseRef.current;
    setRunPaused(pauseRef.current);
  }

  function stopCurrentRun() {
    if (busy === "none") return;
    stopRef.current = true;
    setRunStopRequested(true);
    abortRef.current?.abort();
  }

  function isStoppedError(error: unknown) {
    const text = String(error ?? "");
    return text.includes("__BENCH_STOPPED__") || text.includes("AbortError");
  }

  function appendScore(entry: Omit<ScoreboardEntry, "id">) {
    setScoreboard((prev) => [
      {
        id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        ...entry,
      },
      ...prev,
    ]);
  }

  async function startDetailedLogFile(
    benchmark: BenchmarkKind,
    signal: AbortSignal,
  ): Promise<DetailedLogHandle | null> {
    if (!persistDetailedLogToFile) return null;
    const res = await fetch("/api/stage0/validate/log/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        benchmark,
        scope: modelScope,
        context_window: contextWindow,
        decode_strategy: decodeStrategy,
        effort: generationEffort,
        max_new_tokens: maxNewTokens,
        run_label: normalizeSpace(logRunLabel) || null,
      }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(body || `HTTP ${res.status}`);
    }
    const payload = (await res.json()) as {
      status?: string;
      token?: string;
      file_path?: string;
      file_name?: string;
    };
    const token = String(payload.token ?? "").trim();
    const filePath = String(payload.file_path ?? "").trim();
    const fileName = String(payload.file_name ?? "").trim();
    if (!token || !filePath) {
      throw new Error("Backend detailed log start non ha restituito token/path validi.");
    }
    setLastDetailedLogPath(filePath);
    return { token, filePath, fileName };
  }

  async function finishDetailedLogFile(
    handle: DetailedLogHandle | null,
    status: "completed" | "stopped" | "error",
    summary: Record<string, unknown>,
  ) {
    if (!handle) return;
    try {
      await fetch("/api/stage0/validate/log/finish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: handle.token,
          status,
          summary,
        }),
      });
    } catch {
      // Best-effort close. Run already ended; avoid surfacing noisy UI errors here.
    }
  }

  async function runHellaSwag() {
    if (anyRunning) return;
    resetRunControlState();
    abortRef.current = new AbortController();
    const runStartedIso = new Date().toISOString();
    const runStartMs = Date.now();
    setBusy("hellaswag");
    setHellaLogs([
      makeLog(
        "info",
        `Run HellaSwag avviata. scope=${modelScope} ctx=${contextWindow} effort=${generationEffort} decode=${decodeStrategy} n-shots=${hellaNShots}`,
      ),
    ]);
    setHellaProgress(0);
    setHellaCorrect(0);
    setHellaTotal(0);
    setHellaAccuracy(null);
    setHellaError("");
    setHellaDatasetPath("");
    setHellaDatasetSource("");
    setHellaDatasetTotal(null);
    let correct = 0;
    let errors = 0;
    let tokensTotal = 0;
    let runTotal = 0;
    let logHandle: DetailedLogHandle | null = null;
    let runStatus: "completed" | "stopped" | "error" = "completed";
    try {
      await waitIfPausedOrStop();
      logHandle = await startDetailedLogFile("hellaswag", abortRef.current.signal);
      if (logHandle) {
        pushLog(setHellaLogs, "info", `Log dettagliato su file: ${logHandle.filePath}`);
      }
      const itemsRes = await fetch(
        `/api/stage0/validate/hellaswag/items?limit=${HELLASWAG_SUBSET_LIMIT}&seed=42`,
        { signal: abortRef.current.signal },
      );
      if (!itemsRes.ok) {
        const body = await itemsRes.text().catch(() => "");
        throw new Error(body || `HTTP ${itemsRes.status}`);
      }
      const itemsPayload = (await itemsRes.json()) as ItemsResponse<HellaSwagItem>;
      const items = Array.isArray(itemsPayload.items) ? itemsPayload.items : [];
      if (items.length === 0) throw new Error("HellaSwag subset vuoto o non disponibile dal backend.");
      runTotal = items.length;
      setHellaTotal(items.length);
      setHellaDatasetPath(String(itemsPayload.dataset_path ?? ""));
      setHellaDatasetSource(String(itemsPayload.source ?? ""));
      setHellaDatasetTotal(
        typeof itemsPayload.total_items === "number" ? itemsPayload.total_items : null,
      );
      pushLog(setHellaLogs, "info", `Subset acquisito: ${items.length} item.`);
      for (let i = 0; i < items.length; i += 1) {
        await waitIfPausedOrStop();
        const item = items[i];
        const res = await fetch("/api/stage0/validate/hellaswag-item", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortRef.current.signal,
          body: JSON.stringify({
            stem: item.stem,
            endings: item.endings,
            label_target: item.label,
            scope: modelScope,
            context_window: contextWindow,
            decode_strategy: decodeStrategy,
            temperature: decodeParams.temperature,
            top_p: decodeParams.top_p,
            n_shots: hellaNShots,
            mode: "Q_MODE",
            max_new_tokens: maxNewTokens,
            effort: generationEffort,
            seed: 42 + i,
            detailed_log_token: logHandle?.token ?? null,
          }),
        });
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          errors += 1;
          pushLog(setHellaLogs, "error", `[${i + 1}/${items.length}] HTTP ${res.status} ${body || "item failed"}`);
          setHellaProgress(((i + 1) / items.length) * 100);
          continue;
        }
        const payload = (await res.json()) as {
          status?: string;
          error?: string | null;
          predicted_idx?: number | null;
          is_correct?: boolean;
          selected_lane?: string | null;
          raw_text?: string;
          stats?: unknown;
        };
        if (String(payload.status ?? "ok") === "error") {
          errors += 1;
          pushLog(setHellaLogs, "warn", `[${i + 1}/${items.length}] errore decode: ${payload.error || "unknown"}`);
          setHellaProgress(((i + 1) / items.length) * 100);
          continue;
        }
        const isCorrect =
          typeof payload.is_correct === "boolean"
            ? payload.is_correct
            : typeof payload.predicted_idx === "number" && payload.predicted_idx === item.label;
        if (isCorrect) correct += 1;
        tokensTotal += parseTokensFromStats(payload.stats);
        const details = detailedBenchLogs
          ? [
              `Q: ${trimForLog(item.stem, 420)}`,
              `Opzioni:`,
              `A) ${trimForLog(item.endings[0], 260)}`,
              `B) ${trimForLog(item.endings[1], 260)}`,
              `C) ${trimForLog(item.endings[2], 260)}`,
              `D) ${trimForLog(item.endings[3], 260)}`,
              `Target: ${item.label}`,
              `Thinking+Answer: ${trimForLog(String(payload.raw_text ?? "").trim() || "n/a", 1800)}`,
            ].join("\n")
          : undefined;
        pushLog(
          setHellaLogs,
          isCorrect ? "ok" : "warn",
          `[${i + 1}/${items.length}] lane=${payload.selected_lane ?? "n/a"} pred=${payload.predicted_idx ?? "n/a"} target=${item.label}`,
          details,
        );
        setHellaCorrect(correct);
        setHellaProgress(((i + 1) / items.length) * 100);
      }
      setHellaAccuracy((correct / items.length) * 100);
      pushLog(setHellaLogs, "info", `Completato: correct=${correct}/${items.length}, errori=${errors}.`);
    } catch (error) {
      if (isStoppedError(error)) {
        runStatus = "stopped";
        pushLog(setHellaLogs, "warn", "Run interrotta dall'utente.");
      } else {
        runStatus = "error";
        const msg = String(error);
        setHellaError(msg);
        pushLog(setHellaLogs, "error", msg);
      }
    } finally {
      const elapsedMs = Date.now() - runStartMs;
      if (runTotal > 0) {
        appendScore({
          benchmark: "hellaswag",
          model: scopeLabel(modelScope),
          effort: generationEffort,
          decodeStrategy,
          startedAtIso: runStartedIso,
          elapsedMs,
          accuracy: (correct / runTotal) * 100,
          correct,
          total: runTotal,
          dataLabel: `HellaSwag val · n=${runTotal}`,
          tokens: tokensTotal > 0 ? tokensTotal : null,
        });
      }
      await finishDetailedLogFile(logHandle, runStatus, {
        benchmark: "hellaswag",
        correct,
        total: runTotal,
        errors,
        elapsed_ms: elapsedMs,
      });
      abortRef.current = null;
      resetRunControlState();
      setBusy("none");
    }
  }

  async function runMmluPro() {
    if (anyRunning) return;
    resetRunControlState();
    abortRef.current = new AbortController();
    const runStartedIso = new Date().toISOString();
    const runStartMs = Date.now();
    setBusy("mmlu");
    setMmluLogs([makeLog("info", `Run MMLU-Pro avviata. scope=${modelScope} n-shots=${mmluNShots} force_cot=${mmluForceCot}`)]);
    setMmluProgress(0);
    setMmluCorrect(0);
    setMmluTotal(0);
    setMmluAccuracy(null);
    setMmluError("");
    setMmluDataset("");
    setMmluDatasetSource("");
    setMmluDatasetSplit("");
    setMmluDatasetTotal(null);
    let correct = 0;
    let errors = 0;
    let tokensTotal = 0;
    let runTotal = 0;
    let logHandle: DetailedLogHandle | null = null;
    let runStatus: "completed" | "stopped" | "error" = "completed";
    try {
      await waitIfPausedOrStop();
      logHandle = await startDetailedLogFile("mmlu-pro", abortRef.current.signal);
      if (logHandle) {
        pushLog(setMmluLogs, "info", `Log dettagliato su file: ${logHandle.filePath}`);
      }
      const itemsRes = await fetch(
        `/api/stage0/validate/mmlu-pro/items?limit=${MMLU_PRO_SUBSET_LIMIT}&seed=42&split=test`,
        { signal: abortRef.current.signal },
      );
      if (!itemsRes.ok) throw new Error((await itemsRes.text().catch(() => "")) || `HTTP ${itemsRes.status}`);
      const itemsPayload = (await itemsRes.json()) as ItemsResponse<MmluProItem>;
      const items = Array.isArray(itemsPayload.items) ? itemsPayload.items : [];
      if (items.length === 0) throw new Error("MMLU-Pro subset vuoto o non disponibile dal backend.");
      runTotal = items.length;
      setMmluTotal(items.length);
      setMmluDataset(String(itemsPayload.dataset_id ?? ""));
      setMmluDatasetSource(String(itemsPayload.source ?? ""));
      setMmluDatasetSplit(String(itemsPayload.split ?? ""));
      setMmluDatasetTotal(typeof itemsPayload.total_items === "number" ? itemsPayload.total_items : null);
      pushLog(setMmluLogs, "info", `Subset acquisito: ${items.length} item.`);
      for (let i = 0; i < items.length; i += 1) {
        await waitIfPausedOrStop();
        const item = items[i];
        const res = await fetch("/api/stage0/validate/mmlu-pro-item", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortRef.current.signal,
          body: JSON.stringify({
            question: item.question,
            options: item.options,
            answer_label: item.answer_label,
            scope: modelScope,
            context_window: contextWindow,
            decode_strategy: decodeStrategy,
            temperature: decodeParams.temperature,
            top_p: decodeParams.top_p,
            n_shots: mmluNShots,
            force_cot: mmluForceCot,
            mode: "S_MODE",
            max_new_tokens: maxNewTokens,
            effort: generationEffort,
            seed: 4242 + i,
            detailed_log_token: logHandle?.token ?? null,
          }),
        });
        if (!res.ok) {
          errors += 1;
          const body = await res.text().catch(() => "");
          pushLog(setMmluLogs, "error", `[${i + 1}/${items.length}] HTTP ${res.status} ${body || "item failed"}`);
          setMmluProgress(((i + 1) / items.length) * 100);
          continue;
        }
        const payload = (await res.json()) as {
          status?: string;
          error?: string | null;
          is_correct?: boolean;
          predicted_label?: string | null;
          target_label?: string | null;
          selected_lane?: string | null;
          raw_text?: string;
          stats?: unknown;
        };
        if (String(payload.status ?? "ok") === "error") {
          errors += 1;
          pushLog(setMmluLogs, "warn", `[${i + 1}/${items.length}] errore decode: ${payload.error || "unknown"}`);
          setMmluProgress(((i + 1) / items.length) * 100);
          continue;
        }
        if (payload.is_correct) correct += 1;
        tokensTotal += parseTokensFromStats(payload.stats);
        const details = detailedBenchLogs
          ? [
              `Q: ${trimForLog(item.question, 420)}`,
              `Opzioni: ${trimForLog(item.options.map((opt, idx) => `${String.fromCharCode(65 + idx)}) ${opt}`).join(" | "), 700)}`,
              `Target: ${item.answer_label}`,
              `Thinking+Answer: ${trimForLog(String(payload.raw_text ?? "").trim() || "n/a", 1800)}`,
            ].join("\n")
          : undefined;
        pushLog(
          setMmluLogs,
          payload.is_correct ? "ok" : "warn",
          `[${i + 1}/${items.length}] lane=${payload.selected_lane ?? "n/a"} pred=${payload.predicted_label ?? "n/a"} target=${payload.target_label ?? "n/a"}`,
          details,
        );
        setMmluCorrect(correct);
        setMmluProgress(((i + 1) / items.length) * 100);
      }
      setMmluAccuracy((correct / items.length) * 100);
      pushLog(setMmluLogs, "info", `Completato: correct=${correct}/${items.length}, errori=${errors}.`);
    } catch (error) {
      if (isStoppedError(error)) {
        runStatus = "stopped";
        pushLog(setMmluLogs, "warn", "Run interrotta dall'utente.");
      } else {
        runStatus = "error";
        const msg = String(error);
        setMmluError(msg);
        pushLog(setMmluLogs, "error", msg);
      }
    } finally {
      const elapsedMs = Date.now() - runStartMs;
      if (runTotal > 0) {
        appendScore({
          benchmark: "mmlu-pro",
          model: scopeLabel(modelScope),
          effort: generationEffort,
          decodeStrategy,
          startedAtIso: runStartedIso,
          elapsedMs,
          accuracy: (correct / runTotal) * 100,
          correct,
          total: runTotal,
          dataLabel: `MMLU-Pro test · n=${runTotal}`,
          tokens: tokensTotal > 0 ? tokensTotal : null,
        });
      }
      await finishDetailedLogFile(logHandle, runStatus, {
        benchmark: "mmlu-pro",
        correct,
        total: runTotal,
        errors,
        elapsed_ms: elapsedMs,
      });
      abortRef.current = null;
      resetRunControlState();
      setBusy("none");
    }
  }

  async function runGsm8k() {
    if (anyRunning) return;
    resetRunControlState();
    abortRef.current = new AbortController();
    const runStartedIso = new Date().toISOString();
    const runStartMs = Date.now();
    setBusy("gsm8k");
    setGsmLogs([makeLog("info", `Run GSM8K avviata. scope=${modelScope} n-shots=${gsmNShots}`)]);
    setGsmProgress(0);
    setGsmCorrect(0);
    setGsmTotal(0);
    setGsmAccuracy(null);
    setGsmError("");
    setGsmDataset("");
    setGsmDatasetSource("");
    setGsmDatasetSplit("");
    setGsmDatasetTotal(null);
    let correct = 0;
    let errors = 0;
    let tokensTotal = 0;
    let runTotal = 0;
    let logHandle: DetailedLogHandle | null = null;
    let runStatus: "completed" | "stopped" | "error" = "completed";
    try {
      await waitIfPausedOrStop();
      logHandle = await startDetailedLogFile("gsm8k", abortRef.current.signal);
      if (logHandle) {
        pushLog(setGsmLogs, "info", `Log dettagliato su file: ${logHandle.filePath}`);
      }
      const itemsRes = await fetch(
        `/api/stage0/validate/gsm8k/items?limit=${GSM8K_SUBSET_LIMIT}&seed=42&split=test`,
        { signal: abortRef.current.signal },
      );
      if (!itemsRes.ok) throw new Error((await itemsRes.text().catch(() => "")) || `HTTP ${itemsRes.status}`);
      const itemsPayload = (await itemsRes.json()) as ItemsResponse<Gsm8kItem>;
      const items = Array.isArray(itemsPayload.items) ? itemsPayload.items : [];
      if (items.length === 0) throw new Error("GSM8K subset vuoto o non disponibile dal backend.");
      runTotal = items.length;
      setGsmTotal(items.length);
      setGsmDataset(String(itemsPayload.dataset_id ?? ""));
      setGsmDatasetSource(String(itemsPayload.source ?? ""));
      setGsmDatasetSplit(String(itemsPayload.split ?? ""));
      setGsmDatasetTotal(typeof itemsPayload.total_items === "number" ? itemsPayload.total_items : null);
      pushLog(setGsmLogs, "info", `Subset acquisito: ${items.length} item.`);
      for (let i = 0; i < items.length; i += 1) {
        await waitIfPausedOrStop();
        const item = items[i];
        const res = await fetch("/api/stage0/validate/gsm8k-item", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortRef.current.signal,
          body: JSON.stringify({
            question: item.question,
            answer_target: item.answer_target,
            scope: modelScope,
            context_window: contextWindow,
            decode_strategy: decodeStrategy,
            temperature: decodeParams.temperature,
            top_p: decodeParams.top_p,
            n_shots: gsmNShots,
            mode: "S_MODE",
            max_new_tokens: maxNewTokens,
            effort: generationEffort,
            seed: 5252 + i,
            detailed_log_token: logHandle?.token ?? null,
          }),
        });
        if (!res.ok) {
          errors += 1;
          const body = await res.text().catch(() => "");
          pushLog(setGsmLogs, "error", `[${i + 1}/${items.length}] HTTP ${res.status} ${body || "item failed"}`);
          setGsmProgress(((i + 1) / items.length) * 100);
          continue;
        }
        const payload = (await res.json()) as {
          status?: string;
          error?: string | null;
          is_correct?: boolean;
          predicted_number?: string | null;
          target_number?: string | null;
          selected_lane?: string | null;
          raw_text?: string;
          stats?: unknown;
        };
        if (String(payload.status ?? "ok") === "error") {
          errors += 1;
          pushLog(setGsmLogs, "warn", `[${i + 1}/${items.length}] errore decode: ${payload.error || "unknown"}`);
          setGsmProgress(((i + 1) / items.length) * 100);
          continue;
        }
        if (payload.is_correct) correct += 1;
        tokensTotal += parseTokensFromStats(payload.stats);
        const details = detailedBenchLogs
          ? [
              `Problem: ${trimForLog(item.question, 420)}`,
              `Target: ${trimForLog(item.answer_target, 220)}`,
              `Thinking+Answer: ${trimForLog(String(payload.raw_text ?? "").trim() || "n/a", 1800)}`,
            ].join("\n")
          : undefined;
        pushLog(
          setGsmLogs,
          payload.is_correct ? "ok" : "warn",
          `[${i + 1}/${items.length}] lane=${payload.selected_lane ?? "n/a"} pred=${payload.predicted_number ?? "n/a"} target=${payload.target_number ?? "n/a"}`,
          details,
        );
        setGsmCorrect(correct);
        setGsmProgress(((i + 1) / items.length) * 100);
      }
      setGsmAccuracy((correct / items.length) * 100);
      pushLog(setGsmLogs, "info", `Completato: correct=${correct}/${items.length}, errori=${errors}.`);
    } catch (error) {
      if (isStoppedError(error)) {
        runStatus = "stopped";
        pushLog(setGsmLogs, "warn", "Run interrotta dall'utente.");
      } else {
        runStatus = "error";
        const msg = String(error);
        setGsmError(msg);
        pushLog(setGsmLogs, "error", msg);
      }
    } finally {
      const elapsedMs = Date.now() - runStartMs;
      if (runTotal > 0) {
        appendScore({
          benchmark: "gsm8k",
          model: scopeLabel(modelScope),
          effort: generationEffort,
          decodeStrategy,
          startedAtIso: runStartedIso,
          elapsedMs,
          accuracy: (correct / runTotal) * 100,
          correct,
          total: runTotal,
          dataLabel: `GSM8K test · n=${runTotal}`,
          tokens: tokensTotal > 0 ? tokensTotal : null,
        });
      }
      await finishDetailedLogFile(logHandle, runStatus, {
        benchmark: "gsm8k",
        correct,
        total: runTotal,
        errors,
        elapsed_ms: elapsedMs,
      });
      abortRef.current = null;
      resetRunControlState();
      setBusy("none");
    }
  }

  async function runStability() {
    if (anyRunning || stabilityDisabled) return;
    resetRunControlState();
    abortRef.current = new AbortController();
    const runStartedIso = new Date().toISOString();
    const runStartMs = Date.now();
    let pointsCount = 0;
    let tokensTotal = 0;
    let logHandle: DetailedLogHandle | null = null;
    let runStatus: "completed" | "stopped" | "error" = "completed";
    setBusy("stability");
    setStabilityLogs([
      makeLog("info", `Stability check avviato. scope=${modelScope} steps=${stabilityTotalSteps} schedule=${stabilityMaskSchedule}`),
    ]);
    setStabilityError("");
    setStabilityOutput("");
    setStabilityPoints([]);
    try {
      await waitIfPausedOrStop();
      logHandle = await startDetailedLogFile("stability", abortRef.current.signal);
      if (logHandle) {
        pushLog(setStabilityLogs, "info", `Log dettagliato su file: ${logHandle.filePath}`);
      }
      const res = await fetch("/api/stage0/validate/stability", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          prompt: stabilityPrompt.trim() || "The capital of France is",
          scope: modelScope,
          context_window: contextWindow,
          decode_strategy: decodeStrategy,
          temperature: decodeParams.temperature,
          top_p: decodeParams.top_p,
          total_steps: stabilityTotalSteps,
          mask_schedule: stabilityMaskSchedule,
          mode: "S_MODE",
          effort: generationEffort,
          max_new_tokens: maxNewTokens,
          seed: 42,
          detailed_log_token: logHandle?.token ?? null,
        }),
      });
      if (!res.ok) throw new Error((await res.text().catch(() => "")) || `HTTP ${res.status}`);
      const payload = (await res.json()) as { points?: StabilityPoint[]; text?: string; stats?: unknown };
      const points = Array.isArray(payload.points) ? payload.points : [];
      pointsCount = points.length;
      tokensTotal = parseTokensFromStats(payload.stats);
      setStabilityPoints(points);
      setStabilityOutput(String(payload.text ?? "").trim());
      pushLog(
        setStabilityLogs,
        "ok",
        `Completato: ${points.length} punti di confidenza acquisiti.`,
        detailedBenchLogs
          ? [
              `Prompt: ${trimForLog(stabilityPrompt, 420)}`,
              `Output: ${trimForLog(String(payload.text ?? "").trim() || "n/a", 1800)}`,
            ].join("\n")
          : undefined,
      );
    } catch (error) {
      if (isStoppedError(error)) {
        runStatus = "stopped";
        pushLog(setStabilityLogs, "warn", "Run interrotta dall'utente.");
      } else {
        runStatus = "error";
        const msg = String(error);
        setStabilityError(msg);
        pushLog(setStabilityLogs, "error", msg);
      }
    } finally {
      const elapsedMs = Date.now() - runStartMs;
      appendScore({
        benchmark: "stability",
        model: scopeLabel(modelScope),
        effort: generationEffort,
        decodeStrategy,
        startedAtIso: runStartedIso,
        elapsedMs,
        accuracy: null,
        correct: pointsCount,
        total: pointsCount,
        dataLabel: `Stability steps=${stabilityTotalSteps}`,
        tokens: tokensTotal > 0 ? tokensTotal : null,
      });
      await finishDetailedLogFile(logHandle, runStatus, {
        benchmark: "stability",
        points: pointsCount,
        elapsed_ms: elapsedMs,
      });
      abortRef.current = null;
      resetRunControlState();
      setBusy("none");
    }
  }

  const stabilityChart = useMemo(() => {
    const points = stabilityPoints
      .filter((p) => Number.isFinite(p.step) && Number.isFinite(p.mean_confidence))
      .sort((a, b) => a.step - b.step);
    return {
      x: points.map((p) => p.step),
      y: points.map((p) => p.mean_confidence),
    };
  }, [stabilityPoints]);

  return (
    <Panel
      kicker="Stage 0 Validation"
      title="Sanity Check: HellaSwag + MMLU-Pro + GSM8K + Denoising Stability"
    >
      <section className={styles.globalPanel}>
        <header className={styles.globalHead}>
          <h3>Global Evaluation Settings</h3>
          <p>
            Shared config per benchmark run. Effort {generationEffort.toUpperCase()} ={" "}
            {maxNewTokens} max_new_tokens.
          </p>
        </header>
        <div className={styles.globalGrid}>
          <label className={styles.inlineField}>
            <span>Model Scope</span>
            <select
              aria-label="Model scope"
              value={modelScope}
              onChange={(event) => setModelScope(event.target.value as BenchScope)}
              disabled={anyRunning}
            >
              <option value="AR">AR</option>
              <option value="DLLM">dLLM</option>
              <option value="BOTH">Both</option>
              <option value="RCD">dLLM (RCD)</option>
              <option value="OTS">dLLM (OTS)</option>
              <option value="S2D2">dLLM (S2D2)</option>
              <option value="ENTRGI">dLLM (EntRGi)</option>
            </select>
          </label>
          <label className={styles.inlineField}>
            <span>Context Window</span>
            <select
              aria-label="Context window"
              value={String(contextWindow)}
              onChange={(event) =>
                setContextWindow(Number(event.target.value) as 1024 | 2048 | 4096 | 8192)
              }
              disabled={anyRunning}
            >
              <option value="1024">1024</option>
              <option value="2048">2048</option>
              <option value="4096">4096</option>
              <option value="8192">8192</option>
            </select>
          </label>
          <label className={styles.inlineField}>
            <span>Generation Effort</span>
            <select
              aria-label="Generation effort"
              value={generationEffort}
              onChange={(event) => setGenerationEffort(event.target.value as GlobalEffort)}
              disabled={anyRunning}
            >
              <option value="low">Low (256)</option>
              <option value="medium">Medium (1024)</option>
              <option value="high">High (2048)</option>
            </select>
          </label>
          <label className={styles.inlineField}>
            <span>Decoding Strategy</span>
            <select
              aria-label="Decoding strategy"
              value={decodeStrategy}
              onChange={(event) => setDecodeStrategy(event.target.value as DecodeStrategy)}
              disabled={anyRunning}
            >
              <option value="greedy">Greedy (temp 0.0)</option>
              <option value="sampling">Sampling (temp 0.6, top_p 0.9)</option>
            </select>
          </label>
        </div>
        <div className={styles.globalActions}>
          <label className={styles.checkboxField}>
            <input
              aria-label="Detailed benchmark logs"
              type="checkbox"
              checked={detailedBenchLogs}
              onChange={(event) => setDetailedBenchLogs(event.target.checked)}
              disabled={anyRunning}
            />
            <span>Log dettagliati (Q/A + thinking)</span>
          </label>
          <label className={styles.checkboxField}>
            <input
              aria-label="Persist detailed logs to file"
              type="checkbox"
              checked={persistDetailedLogToFile}
              onChange={(event) => setPersistDetailedLogToFile(event.target.checked)}
              disabled={anyRunning}
            />
            <span>Salva log dettagliato su file (backend)</span>
          </label>
          <label className={styles.inlineField}>
            <span>Run Label (file)</span>
            <input
              aria-label="Detailed log run label"
              value={logRunLabel}
              onChange={(event) => setLogRunLabel(event.target.value)}
              disabled={anyRunning || !persistDetailedLogToFile}
              placeholder="es. qwen06b_testA"
            />
          </label>
          <button
            type="button"
            className={styles.secondaryBtn}
            onClick={() => setShowScoreboard((prev) => !prev)}
          >
            {showScoreboard ? "Nascondi Scoreboard" : "Mostra Scoreboard"}
          </button>
        </div>
        {lastDetailedLogPath ? (
          <p className={styles.meta}>Ultimo file log dettagliato: {lastDetailedLogPath}</p>
        ) : null}
      </section>

      {showScoreboard ? (
        <section className={styles.scoreboardPanel}>
          <header className={styles.scoreboardHead}>
            <h3>Scoreboard</h3>
            <button
              type="button"
              className={styles.secondaryBtn}
              onClick={() => setScoreboard([])}
              disabled={scoreboard.length === 0 || anyRunning}
            >
              Reset
            </button>
          </header>
          <section className={styles.scoreboardFilterPanel}>
            <label className={styles.inlineField}>
              <span>Benchmark Filter</span>
              <select
                aria-label="Scoreboard benchmark filter"
                value={scoreboardFilter}
                onChange={(event) => setScoreboardFilter(event.target.value as "all" | BenchmarkKind)}
              >
                <option value="all">Tutti</option>
                <option value="hellaswag">HellaSwag</option>
                <option value="mmlu-pro">MMLU-Pro</option>
                <option value="gsm8k">GSM8K</option>
                <option value="stability">Stability</option>
              </select>
            </label>
          </section>
          {scoreboard.length === 0 ? (
            <p className={styles.empty}>Nessuna run registrata.</p>
          ) : filteredScoreboard.length === 0 ? (
            <p className={styles.empty}>Nessuna run per il benchmark selezionato.</p>
          ) : (
            <div className={styles.scoreboardTableWrap}>
              <table className={styles.scoreboardTable}>
                <thead>
                  <tr>
                    <th>Benchmark</th>
                    <th>Modello</th>
                    <th>Effort</th>
                    <th>Decode</th>
                    <th>Punteggio</th>
                    <th>Data</th>
                    <th>Data/Subset</th>
                    <th>Tokens</th>
                    <th>Tempo</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredScoreboard.map((row) => (
                    <tr key={row.id}>
                      <td>{row.benchmark}</td>
                      <td>{row.model}</td>
                      <td>{row.effort}</td>
                      <td>{row.decodeStrategy}</td>
                      <td>
                        {row.accuracy === null ? "n/a" : `${row.accuracy.toFixed(1)}%`} ({row.correct}/{row.total})
                      </td>
                      <td>{new Date(row.startedAtIso).toLocaleString()}</td>
                      <td>{row.dataLabel}</td>
                      <td>{typeof row.tokens === "number" ? row.tokens.toLocaleString() : "n/a"}</td>
                      <td>{(row.elapsedMs / 1000).toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : null}

      <div className={styles.grid}>
        <section className={styles.card}>
          <header className={styles.cardHead}>
            <h3>HellaSwag (Zero-Shot)</h3>
            <div className={styles.cardActions}>
              <button type="button" disabled={anyRunning} onClick={() => void runHellaSwag()}>
                {hellaRunning ? "Running..." : "Run HellaSwag"}
              </button>
              {hellaRunning ? (
                <>
                  <button type="button" className={styles.secondaryBtn} onClick={togglePauseRun}>
                    {runPaused ? "Resume" : "Pause"}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={stopCurrentRun}
                    disabled={runStopRequested}
                  >
                    Stop
                  </button>
                </>
              ) : null}
            </div>
          </header>
          <details className={styles.settingsBox}>
            <summary>⚙️ Settings</summary>
            <div className={styles.settingsGrid}>
              <label className={styles.inlineField}>
                <span>N-Shots</span>
                <select
                  aria-label="HellaSwag n-shots"
                  value={String(hellaNShots)}
                  onChange={(event) => setHellaNShots(Number(event.target.value) as 0 | 3 | 5)}
                  disabled={anyRunning}
                >
                  <option value="0">0-shot</option>
                  <option value="3">3-shot</option>
                  <option value="5">5-shot</option>
                </select>
              </label>
            </div>
          </details>
          <p className={styles.copy}>
            Knowledge retention check su subset reale HellaSwag (val). Baseline AR fissata a{" "}
            {AR_BASELINE_PCT}%.
          </p>
          {hellaDatasetPath ? (
            <p className={styles.meta}>
              Dataset: {hellaDatasetPath} · source: {hellaDatasetSource || "n/a"} · totale:{" "}
              {hellaDatasetTotal ?? "n/a"}
            </p>
          ) : null}
          <Progress value={hellaProgress} />
          <BenchmarkGauge
            score={hellaAccuracy}
            baseline={AR_BASELINE_PCT}
            modelLabel={scopeLabel(modelScope)}
            legend={[
              `Correct: ${hellaCorrect}/${hellaTotal || HELLASWAG_SUBSET_LIMIT}`,
              `Scope: ${scopeLabel(modelScope)}`,
            ]}
          />
          {hellaError ? <p className={styles.error}>{hellaError}</p> : null}
          <BenchmarkLogs logs={hellaLogs} />
        </section>

        <section className={styles.card}>
          <header className={styles.cardHead}>
            <h3>MMLU-Pro (Chain-of-Thought)</h3>
            <div className={styles.cardActions}>
              <button type="button" disabled={anyRunning} onClick={() => void runMmluPro()}>
                {mmluRunning ? "Running..." : "Run MMLU-Pro"}
              </button>
              {mmluRunning ? (
                <>
                  <button type="button" className={styles.secondaryBtn} onClick={togglePauseRun}>
                    {runPaused ? "Resume" : "Pause"}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={stopCurrentRun}
                    disabled={runStopRequested}
                  >
                    Stop
                  </button>
                </>
              ) : null}
            </div>
          </header>
          <details className={styles.settingsBox}>
            <summary>⚙️ Settings</summary>
            <div className={styles.settingsGrid}>
              <label className={styles.inlineField}>
                <span>N-Shots</span>
                <select
                  aria-label="MMLU n-shots"
                  value={String(mmluNShots)}
                  onChange={(event) => setMmluNShots(Number(event.target.value) as 0 | 5)}
                  disabled={anyRunning}
                >
                  <option value="0">0-shot</option>
                  <option value="5">5-shot</option>
                </select>
              </label>
              <label className={styles.checkboxField}>
                <input
                  aria-label="Force CoT"
                  type="checkbox"
                  checked={mmluForceCot}
                  onChange={(event) => setMmluForceCot(event.target.checked)}
                  disabled={anyRunning}
                />
                <span>Force CoT (Thinking)</span>
              </label>
            </div>
          </details>
          <p className={styles.copy}>
            Reasoning + knowledge su opzioni multi-classe (A..J), con prompt CoT e valutazione exact
            match della label finale.
          </p>
          {mmluDataset ? (
            <p className={styles.meta}>
              Dataset: {mmluDataset} · split: {mmluDatasetSplit || "test"} · source:{" "}
              {mmluDatasetSource || "n/a"} · totale: {mmluDatasetTotal ?? "n/a"}
            </p>
          ) : null}
          <Progress value={mmluProgress} />
          <BenchmarkGauge
            score={mmluAccuracy}
            baseline={MMLU_PRO_BASELINE_PCT}
            modelLabel={scopeLabel(modelScope)}
            legend={[
              `Correct: ${mmluCorrect}/${mmluTotal || MMLU_PRO_SUBSET_LIMIT}`,
              `Scope: ${scopeLabel(modelScope)}`,
            ]}
          />
          {mmluError ? <p className={styles.error}>{mmluError}</p> : null}
          <BenchmarkLogs logs={mmluLogs} />
        </section>

        <section className={styles.card}>
          <header className={styles.cardHead}>
            <h3>GSM8K (Math Reasoning)</h3>
            <div className={styles.cardActions}>
              <button type="button" disabled={anyRunning} onClick={() => void runGsm8k()}>
                {gsmRunning ? "Running..." : "Run GSM8K"}
              </button>
              {gsmRunning ? (
                <>
                  <button type="button" className={styles.secondaryBtn} onClick={togglePauseRun}>
                    {runPaused ? "Resume" : "Pause"}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={stopCurrentRun}
                    disabled={runStopRequested}
                  >
                    Stop
                  </button>
                </>
              ) : null}
            </div>
          </header>
          <details className={styles.settingsBox}>
            <summary>⚙️ Settings</summary>
            <div className={styles.settingsGrid}>
              <label className={styles.inlineField}>
                <span>N-Shots</span>
                <select
                  aria-label="GSM8K n-shots"
                  value={String(gsmNShots)}
                  onChange={(event) => setGsmNShots(Number(event.target.value) as 0 | 4 | 8)}
                  disabled={anyRunning}
                >
                  <option value="0">0-shot</option>
                  <option value="4">4-shot</option>
                  <option value="8">8-shot</option>
                </select>
              </label>
            </div>
          </details>
          <p className={styles.copy}>
            Exact match sul numero finale estratto dal formato `#### [Final Number]`.
          </p>
          {gsmDataset ? (
            <p className={styles.meta}>
              Dataset: {gsmDataset} · split: {gsmDatasetSplit || "test"} · source:{" "}
              {gsmDatasetSource || "n/a"} · totale: {gsmDatasetTotal ?? "n/a"}
            </p>
          ) : null}
          <Progress value={gsmProgress} />
          <BenchmarkGauge
            score={gsmAccuracy}
            baseline={GSM8K_BASELINE_PCT}
            modelLabel={scopeLabel(modelScope)}
            legend={[
              `Correct: ${gsmCorrect}/${gsmTotal || GSM8K_SUBSET_LIMIT}`,
              `Scope: ${scopeLabel(modelScope)}`,
            ]}
          />
          {gsmError ? <p className={styles.error}>{gsmError}</p> : null}
          <BenchmarkLogs logs={gsmLogs} />
        </section>

        <section className={`${styles.card} ${stabilityDisabled ? styles.disabledCard : ""}`}>
          <header className={styles.cardHead}>
            <h3>Denoising Stability</h3>
            <div className={styles.cardActions}>
              <button
                type="button"
                disabled={anyRunning || stabilityDisabled}
                onClick={() => void runStability()}
              >
                {stabilityRunning ? "Running..." : "Run Stability Check"}
              </button>
              {stabilityRunning ? (
                <>
                  <button type="button" className={styles.secondaryBtn} onClick={togglePauseRun}>
                    {runPaused ? "Resume" : "Pause"}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryBtn}
                    onClick={stopCurrentRun}
                    disabled={runStopRequested}
                  >
                    Stop
                  </button>
                </>
              ) : null}
            </div>
          </header>
          <details className={styles.settingsBox} open={!stabilityDisabled}>
            <summary>⚙️ Settings</summary>
            <div className={styles.settingsGrid}>
              <label className={styles.inlineField}>
                <span>Total Denoising Steps</span>
                <input
                  aria-label="Total denoising steps"
                  type="range"
                  min={10}
                  max={100}
                  value={stabilityTotalSteps}
                  onChange={(event) => setStabilityTotalSteps(Number(event.target.value))}
                  disabled={anyRunning || stabilityDisabled}
                />
                <strong>{stabilityTotalSteps}</strong>
              </label>
              <label className={styles.inlineField}>
                <span>Mask Schedule</span>
                <select
                  aria-label="Mask schedule"
                  value={stabilityMaskSchedule}
                  onChange={(event) =>
                    setStabilityMaskSchedule(event.target.value as StabilityMaskSchedule)
                  }
                  disabled={anyRunning || stabilityDisabled}
                >
                  <option value="linear">Linear</option>
                  <option value="cosine">Cosine</option>
                </select>
              </label>
            </div>
          </details>
          <p className={styles.copy}>
            Curva attesa: confidenza media token unmasked crescente durante i passi di denoising.
          </p>
          {stabilityDisabled ? (
            <p className={styles.meta}>Disabilitato: Stability richiede lane dLLM (scope AR non supportato).</p>
          ) : null}
          <label className={styles.promptField}>
            <span>Prompt</span>
            <input
              value={stabilityPrompt}
              onChange={(event) => setStabilityPrompt(event.target.value)}
              disabled={anyRunning || stabilityDisabled}
            />
          </label>
          <div className={styles.chartWrap}>
            {stabilityChart.x.length > 0 ? (
              <TimeseriesChart
                x={stabilityChart.x}
                yLabel="Mean token confidence (0.0 - 1.0)"
                series={[{ label: "mean_confidence", stroke: "#47d6ff", values: stabilityChart.y }]}
              />
            ) : (
              <p className={styles.empty}>
                {stabilityDisabled
                  ? "Imposta Model Scope su dLLM o Both per eseguire lo Stability Check."
                  : 'Nessuna run eseguita. Premi "Run Stability Check".'}
              </p>
            )}
          </div>
          {stabilityOutput ? (
            <div className={styles.output}>
              <strong>Output</strong>
              <p>{stabilityOutput}</p>
            </div>
          ) : null}
          {stabilityError ? <p className={styles.error}>{stabilityError}</p> : null}
          <BenchmarkLogs logs={stabilityLogs} />
        </section>
      </div>
    </Panel>
  );
}

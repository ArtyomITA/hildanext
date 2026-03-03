import { useEffect, useRef } from "react";
import { useDataStore } from "../../store/dataStore";
import { Panel } from "../../components/layout/Panel";
import { useRunStatus } from "./useRunStatus";
import styles from "./RunControlPanel.module.css";

const STATUS_LABEL: Record<string, string> = {
  idle: "idle",
  running: "running…",
  done: "done ✓",
  error: "error ✗",
  stopped: "stopped ■",
};

export function RunControlPanel() {
  const refresh = useDataStore((s) => s.setWsdScenario);
  // Re-load WSD data when a run finishes so metrics update immediately
  const { runStatus, startRun, stopRun, starting, stopping } = useRunStatus(() => {
    void refresh("live_wsd_run");
  });

  const logBoxRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    const el = logBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [runStatus.lines]);

  const isRunning = runStatus.status === "running";
  const isIdle = runStatus.status === "idle";
  const canStart = !isRunning && !starting;

  return (
    <Panel
      kicker="Run controls"
      title="Launch a WSD run from the browser"
      actions={
        <span className={`${styles.statusBadge} ${styles[runStatus.status]}`}>
          {isRunning && <span className={styles.pulse} />}
          {STATUS_LABEL[runStatus.status] ?? runStatus.status}
          {runStatus.mode ? ` · ${runStatus.mode}` : ""}
          {runStatus.exitCode !== null ? ` · exit ${runStatus.exitCode}` : ""}
        </span>
      }
    >
      <div className={styles.toolbar}>
        <button
          className={`${styles.btn} ${styles.btnTest}`}
          disabled={!canStart}
          onClick={() => void startRun("test")}
          title="Run 10 WSD training steps — wsd-log-test"
        >
          {starting && runStatus.mode !== "full" ? "…" : "▶"} wsd-log-test
          <span className={styles.btnSub}>10 steps</span>
        </button>

        <button
          className={`${styles.btn} ${styles.btnFull}`}
          disabled={!canStart}
          onClick={() => void startRun("full")}
          title="Full WSD run — reads llada21_dolma_wsd_only.json (from PS1 config)"
        >
          {starting && runStatus.mode !== "test" ? "…" : "▶"} full wsd
          <span className={styles.btnSub}>llada21_dolma_wsd_only.json</span>
        </button>

        <button
          className={`${styles.btn} ${styles.btnStop}`}
          disabled={!isRunning || stopping}
          onClick={() => void stopRun()}
          title="Terminate the running subprocess"
        >
          {stopping ? "…" : "■"} stop
        </button>
      </div>

      {/* Live log tail */}
      <div
        ref={logBoxRef}
        className={`${styles.logBox} ${runStatus.lines.length === 0 ? styles.logBoxEmpty : ""}`}
      >
        {runStatus.lines.length === 0 ? (
          <span className={styles.emptyHint}>
            {isIdle
              ? "No run started yet — click a button above."
              : "Waiting for output…"}
          </span>
        ) : (
          runStatus.lines.map((line, i) => (
            <div
              key={i}
              className={`${styles.logLine} ${
                line.includes("error") || line.includes("ERROR") || line.includes("runner_error")
                  ? styles.logError
                  : line.includes("WARN") || line.includes("warn")
                    ? styles.logWarn
                    : ""
              }`}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </Panel>
  );
}

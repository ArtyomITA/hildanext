/**
 * useRunStatus — polls /api/run/status and exposes startRun / stopRun.
 *
 * Backoff: when the backend is unreachable the poll interval backs off
 * exponentially (2s → 4s → 8s … max 30s) so a stopped/restarting server
 * doesn't flood Vite with ECONNREFUSED errors.  Interval resets to 2s as
 * soon as the backend responds again.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface RunStatus {
  status: "idle" | "running" | "done" | "error" | "stopped";
  mode: string;
  exitCode: number | null;
  lines: string[];
}

const INITIAL: RunStatus = { status: "idle", mode: "", exitCode: null, lines: [] };
const POLL_MS_MIN  = 2_000;
const POLL_MS_MAX  = 30_000;
const BACKOFF_MULT = 2;

export function useRunStatus(onFinished?: () => void) {
  const [runStatus, setRunStatus] = useState<RunStatus>(INITIAL);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const prevStatus   = useRef<string>("idle");
  const pollInterval = useRef<number>(POLL_MS_MIN);
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleNext = useCallback((ms: number) => {
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => { void poll(); }, ms);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const poll = useCallback(async (): Promise<string | null> => {
    try {
      const res = await fetch("/api/run/status");
      if (res.ok) {
        const data = (await res.json()) as RunStatus;
        setRunStatus(data);
        const prev = prevStatus.current;
        if (prev === "running" && (data.status === "done" || data.status === "error" || data.status === "stopped")) {
          onFinished?.();
        }
        prevStatus.current = data.status;
        pollInterval.current = POLL_MS_MIN;
        scheduleNext(POLL_MS_MIN);
        return data.status;
      }
      // 503 = backend down, back off silently.
      pollInterval.current = Math.min(pollInterval.current * BACKOFF_MULT, POLL_MS_MAX);
    } catch {
      pollInterval.current = Math.min(pollInterval.current * BACKOFF_MULT, POLL_MS_MAX);
    }
    scheduleNext(pollInterval.current);
    return null;
  }, [onFinished, scheduleNext]);

  useEffect(() => {
    void poll();
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [poll]);

  // When a run is in progress keep the fast cadence; after it stops it backs
  // off naturally on the next failed fetch — no special handling needed.

  const startRun = useCallback(
    async (mode: "test" | "full") => {
      setStarting(true);
      // Reset backoff when user explicitly starts a run.
      pollInterval.current = POLL_MS_MIN;
      try {
        const res = await fetch("/api/run/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        });
        if (!res.ok) {
          const err = await res.text();
          setRunStatus((prev) => ({
            ...prev,
            status: "error",
            lines: [...prev.lines, `[start_error] ${err}`],
          }));
          return;
        }
        // Immediately reflect as running
        prevStatus.current = "idle";
        await poll();
      } catch (e) {
        setRunStatus((prev) => ({
          ...prev,
          status: "error",
          lines: [...prev.lines, `[start_error] ${String(e)}`],
        }));
      } finally {
        setStarting(false);
      }
    },
    [poll],
  );

  const stopRun = useCallback(async () => {
    setStopping(true);
    try {
      await fetch("/api/run/stop", { method: "POST" });
      await poll();
    } catch {
      // ignore
    } finally {
      setStopping(false);
    }
  }, [poll]);

  return { runStatus, startRun, stopRun, starting, stopping };
}

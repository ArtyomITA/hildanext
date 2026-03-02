/**
 * useRunStatus — polls /api/run/status every 2s and exposes startRun / stopRun.
 * Works independently of the main WSD data store.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface RunStatus {
  status: "idle" | "running" | "done" | "error";
  mode: string;
  exitCode: number | null;
  lines: string[];
}

const INITIAL: RunStatus = { status: "idle", mode: "", exitCode: null, lines: [] };
const POLL_MS = 2000;

export function useRunStatus(onFinished?: () => void) {
  const [runStatus, setRunStatus] = useState<RunStatus>(INITIAL);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const prevStatus = useRef<string>("idle");

  const fetchStatus = useCallback(async (): Promise<string | null> => {
    try {
      const res = await fetch("/api/run/status");
      if (res.ok) {
        const data = (await res.json()) as RunStatus;
        setRunStatus(data);
        // Fire onFinished callback when a run transitions from running → done/error
        const prev = prevStatus.current;
        if (prev === "running" && (data.status === "done" || data.status === "error")) {
          onFinished?.();
        }
        prevStatus.current = data.status;
        return data.status;
      }
    } catch {
      // backend offline — leave last known state
    }
    return null;
  }, [onFinished]);

  // Always poll; interval is cheap when idle (just confirms "idle")
  useEffect(() => {
    void fetchStatus();
    const id = window.setInterval(() => {
      void fetchStatus();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [fetchStatus]);

  const startRun = useCallback(
    async (mode: "test" | "full") => {
      setStarting(true);
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
        await fetchStatus();
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
    [fetchStatus],
  );

  const stopRun = useCallback(async () => {
    setStopping(true);
    try {
      await fetch("/api/run/stop", { method: "POST" });
      await fetchStatus();
    } catch {
      // ignore
    } finally {
      setStopping(false);
    }
  }, [fetchStatus]);

  return { runStatus, startRun, stopRun, starting, stopping };
}

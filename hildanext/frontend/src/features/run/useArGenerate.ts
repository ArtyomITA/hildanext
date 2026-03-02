/**
 * useArGenerate
 * Calls POST /api/generate/ar (→ FastAPI /generate/ar → ar.generate_ar()).
 * Returns { running, error, runAr } where runAr resolves to the raw API
 * response or null on network/HTTP error.
 */
import { useState } from "react";

export interface ArRunStats {
  engine: string;
  dummy_model: boolean;
  load_reason?: string;
  actual_dtype?: string;
  tokens_generated?: number;
  tokens_per_sec?: number;
  fallbacks?: unknown[];
}

export interface ArRunResult {
  text: string;
  engine: string;
  stats: ArRunStats;
}

export function useArGenerate() {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runAr(
    prompt: string,
    maxNewTokens: number,
    seed: number,
  ): Promise<ArRunResult | null> {
    setRunning(true);
    setError(null);
    try {
      const res = await fetch("/api/generate/ar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens, seed }),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`HTTP ${res.status}: ${detail}`);
      }
      return (await res.json()) as ArRunResult;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setRunning(false);
    }
  }

  return { running, error, runAr };
}

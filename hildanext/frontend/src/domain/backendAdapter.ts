/**
 * BackendFrontendAdapter
 * Calls the FastAPI server (proxied via Vite at /api/*).
 * WSD endpoint: GET /api/frontend/wsd  → WsdScenarioData
 * Inference not yet wired — throws, caller falls back to MockFrontendAdapter.
 */
import type { FrontendDataAdapter, InferenceScenarioData, WsdScenarioData } from "./types";

class BackendFrontendAdapter implements FrontendDataAdapter {
  async loadWsdScenario(_id: string): Promise<WsdScenarioData> {
    const res = await fetch("/api/frontend/wsd");
    if (!res.ok) {
      throw new Error(`backend_error: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as WsdScenarioData;
  }

  // Inference not wired yet — caller will catch and use missing-scenario stub.
  async loadInferenceScenario(_id: string): Promise<InferenceScenarioData> {
    throw new Error("inference_not_wired: backend inference endpoint not available yet");
  }
}

export const backendAdapter: FrontendDataAdapter = new BackendFrontendAdapter();

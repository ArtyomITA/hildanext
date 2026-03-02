import { create } from "zustand";
import { frontendAdapter } from "../domain/adapters";
import { InferenceScenarioData, WsdScenarioData } from "../domain/types";
import { generateWsdScenario, generateInferenceScenario } from "../mocks/generators";

function offlineWsd(): WsdScenarioData {
  const s = generateWsdScenario("offline_mockup", "Offline mockup", "healthy");
  return { ...s, dataSource: "mockup" };
}

function offlineInference(): InferenceScenarioData {
  const s = generateInferenceScenario("offline_mockup", "Offline mockup", "clean");
  return { ...s, dataSource: "mockup" };
}

interface DataState {
  ready: boolean;
  wsd: WsdScenarioData;
  inference: InferenceScenarioData;
  hydrate: () => Promise<void>;
  setWsdScenario: (id: string) => Promise<void>;
  setInferenceScenario: (id: string) => Promise<void>;
  /** Start polling the active WSD scenario every `intervalMs` ms. */
  startPolling: (intervalMs: number) => void;
  /** Stop any active polling timer. */
  stopPolling: () => void;
}

// Module-level poll timer so it survives re-renders and can be cleared.
let _pollTimer: number | null = null;
function stopPolling() {
  if (_pollTimer !== null) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

export const useDataStore = create<DataState>((set, get) => ({
  ready: false,
  wsd: {
    id: "loading",
    label: "Loading",
    dataSource: "mockup",
    meta: {
      runId: "loading",
      configDigest: "loading",
      optimizer: "loading",
      dtype: "loading",
      device: "loading",
      dummyModel: false,
      phase: "warmup",
      blockSize: 1,
      ladderBlocks: [1],
    },
    metrics: [],
    logs: [],
    processes: [],
    insights: [],
  },
  inference: {
    id: "loading",
    label: "Loading",
    dataSource: "mockup",
    ar: {
      engine: "ar",
      mode: "Q_MODE",
      effort: "low",
      prompt: "",
      outputText: "",
      tauMask: 1,
      tauEdit: 0,
      steps: 1,
      stepsToConverge: 1,
      tokensPerSec: null,
      vramPeakBytes: null,
      dummyModel: false,
      envIssues: {},
      fallbacks: [],
      logs: [],
      tokenFrames: [],
    },
    diffusion: {
      engine: "transformers",
      mode: "S_MODE",
      effort: "medium",
      prompt: "",
      outputText: "",
      tauMask: 0.08,
      tauEdit: 0.08,
      steps: 0,
      stepsToConverge: null,
      tokensPerSec: null,
      vramPeakBytes: null,
      dummyModel: false,
      envIssues: {},
      fallbacks: [],
      logs: [],
      tokenFrames: [],
    },
    logs: [],
    insights: [],
  },
  async hydrate() {
    const [wsd, inference] = await Promise.all([
      frontendAdapter
        .loadWsdScenario("live_wsd_run")
        .catch(() => offlineWsd()),
      frontendAdapter
        .loadInferenceScenario("ar_vs_diffusion_compare")
        .catch(() => offlineInference()),
    ]);
    set({ ready: true, wsd, inference });
  },
  async setWsdScenario(id) {
    const wsd = await frontendAdapter
      .loadWsdScenario(id)
      .catch(() => offlineWsd());
    set({ wsd });
  },
  async setInferenceScenario(id) {
    const inference = await frontendAdapter
      .loadInferenceScenario(id)
      .catch(() => offlineInference());
    set({ inference });
  },
  startPolling(intervalMs: number) {
    stopPolling();
    _pollTimer = window.setInterval(() => {
      void get().setWsdScenario("live_wsd_run");
    }, intervalMs) as number;
  },
  stopPolling() {
    stopPolling();
  },
}));

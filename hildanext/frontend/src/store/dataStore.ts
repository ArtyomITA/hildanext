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

const POLL_MIN_MS = 5_000;
const POLL_MAX_MS = 30_000;
const POLL_BACKOFF = 2;

interface DataState {
  ready: boolean;
  wsd: WsdScenarioData;
  inference: InferenceScenarioData;
  backendAlive: boolean;
  hydrate: () => Promise<void>;
  setWsdScenario: (id: string) => Promise<void>;
  setInferenceScenario: (id: string) => Promise<void>;
  startPolling: (intervalMs: number) => void;
  stopPolling: () => void;
}

let _pollTimer: number | null = null;
let _pollInterval = POLL_MIN_MS;

function stopPollingTimer() {
  if (_pollTimer !== null) {
    clearTimeout(_pollTimer);
    _pollTimer = null;
  }
}

export const useDataStore = create<DataState>((set, get) => ({
  ready: false,
  backendAlive: false,
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
        .then((w) => { set({ backendAlive: true }); return w; })
        .catch(() => { set({ backendAlive: false }); return offlineWsd(); }),
      frontendAdapter
        .loadInferenceScenario("ar_vs_diffusion_compare")
        .catch(() => offlineInference()),
    ]);
    set({ ready: true, wsd, inference });
  },
  async setWsdScenario(id) {
    try {
      const wsd = await frontendAdapter.loadWsdScenario(id);
      _pollInterval = POLL_MIN_MS;
      set({ wsd, backendAlive: true });
    } catch {
      _pollInterval = Math.min(_pollInterval * POLL_BACKOFF, POLL_MAX_MS);
      set({ backendAlive: false });
    }
  },
  async setInferenceScenario(id) {
    const inference = await frontendAdapter
      .loadInferenceScenario(id)
      .catch(() => offlineInference());
    set({ inference });
  },
  startPolling(_intervalMs: number) {
    stopPollingTimer();
    const tick = () => {
      void get().setWsdScenario("live_wsd_run");
      _pollTimer = window.setTimeout(tick, _pollInterval) as number;
    };
    _pollTimer = window.setTimeout(tick, _pollInterval) as number;
  },
  stopPolling() {
    stopPollingTimer();
  },
}));

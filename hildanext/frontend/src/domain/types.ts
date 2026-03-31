export type LogKind = "console" | "metric" | "fallback" | "training" | "eval" | "process";

export interface NormalizedLogEntry {
  id: string;
  tsUtc: string;
  source: LogKind;
  level: "info" | "notice" | "warning" | "error";
  module?: string;
  func?: string;
  eventType?: string;
  action?: string;
  reason?: string;
  message: string;
  extra?: Record<string, unknown>;
  tags: string[];
}

export interface ProcessSnapshot {
  tsUtc: string;
  processName: string;
  pid: number;
  cpuPct: number;
  ramMb: number;
  gpuVramMb?: number;
  gpuUtilPct?: number;
  status: "running" | "sleeping" | "crashed" | "restarting";
}

export interface WsdMetricRow {
  kind: "cpt" | "sft";
  step: number;
  phase: "warmup" | "stable" | "decay";
  blockSize: number;
  loss: number;
  lossM2T: number;
  lossT2T: number;
  maskedTokenAcc: number | null;
  lr: number;
  gradNorm: number;
  tokensPerSec: number;
  stepTimeS: number;
  vramAllocMb: number;
  vramReservedMb: number;
  vramPeakMb: number;
  etaStageSec: number;
  tSampled: number;
  tMean: number;
  tMin: number;
  tMax: number;
  maskRatioActual: number;
  predPositionsCount: number;
  wsdPhaseProgress: number;
  bidirectional: boolean;
  isCausalEffective: boolean;
  attentionMode: string;
  shiftMode: string;
  timeParam: string;
  lossWeighting: string;
  lossByTBucket: Record<string, number>;
  accMaskedByTBucket: Record<string, number>;
}

export interface InferenceTraceStep {
  step: number;
  maskRatio: number;
  gammaCount: number;
  deltaCount: number;
  avgConfMasked: number | null;
  avgConfTokens: number | null;
  tauMask: number;
  tauEdit: number;
  tauFallbackApplied?: boolean;
  tauMaskAfterFallback?: number;
}

export interface TokenFrame {
  step: number;
  tokens: Array<{
    index: number;
    text: string;
    state: "prompt" | "masked" | "new" | "edited" | "stable";
    confidence?: number;
    lane: "ar" | "diffusion";
  }>;
}

export interface InferenceRun {
  engine: "transformers" | "dinfer" | "ar";
  mode: "S_MODE" | "Q_MODE";
  effort: "instant" | "low" | "medium" | "high" | "adaptive";
  prompt: string;
  outputText: string;
  tauMask: number;
  tauEdit: number;
  steps: number;
  stepsToConverge: number | null;
  tokensPerSec: number | null;
  vramPeakBytes: number | null;
  dummyModel: boolean;
  loadReason?: string;
  envIssues: Record<string, string>;
  fallbacks: NormalizedLogEntry[];
  logs: InferenceTraceStep[];
  tokenFrames: TokenFrame[];
}

export interface GlossaryTerm {
  key: string;
  shortLabel: string;
  english: string;
  italianHint: string;
  explanation: string;
  relatedKeys: string[];
}

export interface WsdMeta {
  runId: string;
  configDigest: string;
  optimizer: string;
  dtype: string;
  device: string;
  dummyModel: boolean;
  phase: "warmup" | "stable" | "decay";
  blockSize: number;
  ladderBlocks: number[];
  /** Schedule step counts from the actual WSD config (llada21_dolma_wsd_only.json). */
  warmupSteps?: number;
  stableSteps?: number;
  decaySteps?: number;
  totalSteps?: number;
}

export interface InsightCard {
  id: string;
  title: string;
  metric: string;
  body: string;
  tone: "info" | "warning" | "critical";
}

export interface WsdScenarioData {
  id: string;
  label: string;
  dataSource: "live" | "mockup" | "missing";
  meta: WsdMeta;
  metrics: WsdMetricRow[];
  logs: NormalizedLogEntry[];
  processes: ProcessSnapshot[];
  insights: InsightCard[];
}

export interface InferenceScenarioData {
  id: string;
  label: string;
  dataSource: "live" | "mockup" | "missing";
  ar: InferenceRun;
  diffusion: InferenceRun;
  logs: NormalizedLogEntry[];
  insights: InsightCard[];
}

export interface FrontendDataAdapter {
  loadWsdScenario(id: string): Promise<WsdScenarioData>;
  loadInferenceScenario(id: string): Promise<InferenceScenarioData>;
}

export interface LogSummary {
  total: number;
  visible: number;
  byLevel: Record<string, number>;
  byAction: Record<string, number>;
  byReason: Record<string, number>;
}

export type ChatEngineMode = "AR" | "DLLM" | "BOTH";
export type ChatLane = "ar" | "dllm";
export type ThinkingMode = "auto" | "on" | "off";
export type ArDecodeMode = "greedy" | "sampling";

export interface ModelCatalogEntry {
  id: string;
  lane: ChatLane;
  label: string;
  description: string;
}

export interface ChatRunConfig {
  engineMode: ChatEngineMode;
  arModelId: string;
  dllmModelId: string;
  decodeMode: "S_MODE" | "Q_MODE";
  effort: "instant" | "low" | "medium" | "high" | "adaptive";
  maxNewTokens: number;
  seed: number;
  temperature: number;
  topP: number;
  topK: number;
  presencePenalty: number;
  repetitionPenalty: number;
  tauMask: number;
  tauEdit: number;
  systemPrompt: string;
  thinkingMode: ThinkingMode;
  contextWindowTokens: number;
  arDecodeMode: ArDecodeMode;
}

export interface SavedPreset {
  id: string;
  name: string;
  config: ChatRunConfig;
}

export interface LaneResult {
  lane: ChatLane;
  modelId: string;
  status: "success" | "error" | "offline";
  text: string;
  message: string;
  engine: string;
  tokensPerSec: number | null;
  stepsToConverge: number | null;
  vramPeakBytes: number | null;
  dtype?: string;
  dummyModel: boolean;
  finishReason?: string;
  truncated?: boolean;
  device?: string;
  ignoredSamplingParams?: boolean;
  cpuFallback?: boolean;
  rawText?: string;
  rawStats?: Record<string, unknown>;
}

export interface ChatTurn {
  id: string;
  createdAt: string;
  prompt: string;
  config: ChatRunConfig;
  status: "running" | "completed";
  lanes: LaneResult[];
}

export interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  turns: ChatTurn[];
}

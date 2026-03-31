import { ChatRunConfig, ChatThread, LaneResult } from "../../domain/types";
import { ChatMessage, laneCpuFallback } from "./promptComposer";

interface GenerateApiResult {
  text: string;
  engine: string;
  stats?: Record<string, unknown>;
}

export interface ChatTurnRequest {
  prompt: string;
  messages?: ChatMessage[];
  enableThinking?: boolean | null;
}

function normalizeInput(input: string | ChatTurnRequest): ChatTurnRequest {
  if (typeof input === "string") return { prompt: input };
  return input;
}

function nowIso() {
  return new Date().toISOString();
}

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readMessage(status: number, statusText: string, body: string): string {
  if (body) return body.slice(0, 400);
  return `HTTP ${status}: ${statusText}`;
}

function isOfflineError(status: number | null, error: unknown): boolean {
  if (status === 503) return true;
  const msg = String(error ?? "").toLowerCase();
  return msg.includes("failed to fetch") || msg.includes("networkerror");
}

function mapLaneSuccess(
  lane: "ar" | "dllm",
  modelId: string,
  payload: GenerateApiResult,
): LaneResult {
  const stats = payload.stats ?? {};
  const out: LaneResult = {
    lane,
    modelId,
    status: "success",
    text: String(payload.text ?? "").trim(),
    message: "",
    engine: String(payload.engine ?? lane),
    tokensPerSec: toNumber(stats.tokens_per_sec),
    stepsToConverge: toNumber(stats.steps_to_converge),
    vramPeakBytes: toNumber(stats.vram_peak_bytes),
    dtype: typeof stats.actual_dtype === "string" ? stats.actual_dtype : undefined,
    dummyModel: Boolean(stats.dummy_model),
    finishReason: typeof stats.finish_reason === "string" ? stats.finish_reason : undefined,
    truncated: Boolean(stats.truncated),
    device: typeof stats.device === "string" ? stats.device : undefined,
    ignoredSamplingParams: Boolean(stats.ignored_sampling_params),
    rawText: typeof payload.text === "string" ? payload.text : String(payload.text ?? ""),
    rawStats: stats,
  };
  out.cpuFallback = laneCpuFallback(out);
  return out;
}

function mapLaneFailure(lane: "ar" | "dllm", modelId: string, message: string, offline: boolean): LaneResult {
  return {
    lane,
    modelId,
    status: offline ? "offline" : "error",
    text: "",
    message: offline ? "Backend offline o non raggiungibile per questa lane." : message,
    engine: lane,
    tokensPerSec: null,
    stepsToConverge: null,
    vramPeakBytes: null,
    dummyModel: false,
    truncated: false,
    cpuFallback: false,
    rawText: "",
    rawStats: {},
  };
}

async function callAr(input: ChatTurnRequest, cfg: ChatRunConfig): Promise<LaneResult> {
  let status: number | null = null;
  try {
    const res = await fetch("/api/generate/ar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: input.prompt,
        messages: input.messages,
        system_prompt: cfg.systemPrompt,
        enable_thinking: input.enableThinking,
        max_new_tokens: cfg.maxNewTokens,
        seed: cfg.seed,
        decode_mode: cfg.arDecodeMode,
        temperature: cfg.temperature,
        top_p: cfg.topP,
        top_k: cfg.topK,
        presence_penalty: cfg.presencePenalty,
        repetition_penalty: cfg.repetitionPenalty,
      }),
    });
    status = res.status;
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(readMessage(res.status, res.statusText, body));
    }
    const payload = (await res.json()) as GenerateApiResult;
    const lane = mapLaneSuccess("ar", cfg.arModelId, payload);
    if (lane.stepsToConverge === null) lane.stepsToConverge = 1;
    return lane;
  } catch (error) {
    return mapLaneFailure(
      "ar",
      cfg.arModelId,
      String(error),
      isOfflineError(status, error),
    );
  }
}

async function callDllm(input: ChatTurnRequest, cfg: ChatRunConfig): Promise<LaneResult> {
  let status: number | null = null;
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: input.prompt,
        messages: input.messages,
        system_prompt: cfg.systemPrompt,
        enable_thinking: input.enableThinking,
        mode: cfg.decodeMode,
        tau_mask: cfg.tauMask,
        tau_edit: cfg.tauEdit,
        max_new_tokens: cfg.maxNewTokens,
        seed: cfg.seed,
        effort: cfg.effort,
        temperature: cfg.temperature,
        top_p: cfg.topP,
        top_k: cfg.topK,
        presence_penalty: cfg.presencePenalty,
        repetition_penalty: cfg.repetitionPenalty,
      }),
    });
    status = res.status;
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(readMessage(res.status, res.statusText, body));
    }
    const payload = (await res.json()) as GenerateApiResult;
    return mapLaneSuccess("dllm", cfg.dllmModelId, payload);
  } catch (error) {
    return mapLaneFailure(
      "dllm",
      cfg.dllmModelId,
      String(error),
      isOfflineError(status, error),
    );
  }
}

export async function runChatTurn(input: string | ChatTurnRequest, cfg: ChatRunConfig): Promise<LaneResult[]> {
  const req = normalizeInput(input);
  if (cfg.engineMode === "AR") {
    return [await callAr(req, cfg)];
  }
  if (cfg.engineMode === "DLLM") {
    return [await callDllm(req, cfg)];
  }
  const settled = await Promise.allSettled([callAr(req, cfg), callDllm(req, cfg)]);
  return settled.map((item, idx) => {
    if (item.status === "fulfilled") return item.value;
    const lane = idx === 0 ? "ar" : "dllm";
    const modelId = lane === "ar" ? cfg.arModelId : cfg.dllmModelId;
    return mapLaneFailure(lane, modelId, String(item.reason), isOfflineError(null, item.reason));
  });
}

export function createInitialThread(): ChatThread {
  const id = crypto.randomUUID();
  const ts = nowIso();
  return {
    id,
    title: "Nuova chat",
    createdAt: ts,
    updatedAt: ts,
    turns: [],
  };
}

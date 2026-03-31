import { ChatRunConfig, ChatTurn, LaneResult, ThinkingMode } from "../../domain/types";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ComposedChatInput {
  prompt: string;
  messages: ChatMessage[];
  enableThinking: boolean | null;
  includedTurns: number;
  approxTokens: number;
}

function estimateTokens(text: string): number {
  const t = text.trim();
  if (!t) return 0;
  const words = t.split(/\s+/).length;
  return Math.max(1, Math.ceil(words * 1.5));
}

function laneTextForTurn(turn: ChatTurn, cfg: ChatRunConfig): string {
  const success = turn.lanes.filter((lane) => lane.status === "success" && lane.text.trim());
  if (success.length === 0) return "";
  if (cfg.engineMode === "AR") return success.find((lane) => lane.lane === "ar")?.text ?? success[0].text;
  if (cfg.engineMode === "DLLM") return success.find((lane) => lane.lane === "dllm")?.text ?? success[0].text;
  const ar = success.find((lane) => lane.lane === "ar")?.text ?? "";
  const dllm = success.find((lane) => lane.lane === "dllm")?.text ?? "";
  if (ar && dllm) return `[AR]\n${ar}\n\n[dLLM]\n${dllm}`;
  return dllm || ar;
}

function mapThinkingMode(mode: ThinkingMode, decodeMode: ChatRunConfig["decodeMode"]): boolean | null {
  if (mode === "on") return true;
  if (mode === "off") return false;
  return decodeMode === "S_MODE";
}

export function composeChatInput(
  priorTurns: ChatTurn[],
  userPrompt: string,
  cfg: ChatRunConfig,
): ComposedChatInput {
  const prompt = userPrompt.trim();
  const systemPrompt = cfg.systemPrompt.trim();
  const budget = Math.max(512, Number(cfg.contextWindowTokens) || 4096);
  const systemTokens = estimateTokens(systemPrompt);
  const userTokens = estimateTokens(prompt);
  const maxHistory = Math.max(0, budget - systemTokens - userTokens);

  const history: Array<{ user: string; assistant: string; tokens: number }> = [];
  for (const turn of priorTurns) {
    const user = turn.prompt.trim();
    const assistant = laneTextForTurn(turn, cfg).trim();
    if (!user || !assistant) continue;
    history.push({
      user,
      assistant,
      tokens: estimateTokens(user) + estimateTokens(assistant),
    });
  }

  const picked: Array<{ user: string; assistant: string }> = [];
  let used = 0;
  for (let i = history.length - 1; i >= 0; i -= 1) {
    const item = history[i];
    if (used + item.tokens > maxHistory) continue;
    picked.push({ user: item.user, assistant: item.assistant });
    used += item.tokens;
  }
  picked.reverse();

  const messages: ChatMessage[] = [];
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }
  for (const item of picked) {
    messages.push({ role: "user", content: item.user });
    messages.push({ role: "assistant", content: item.assistant });
  }
  messages.push({ role: "user", content: prompt });

  return {
    prompt,
    messages,
    enableThinking: mapThinkingMode(cfg.thinkingMode, cfg.decodeMode),
    includedTurns: picked.length,
    approxTokens: systemTokens + userTokens + used,
  };
}

export function laneCpuFallback(lane: LaneResult): boolean {
  if (lane.device === "cpu") return true;
  return lane.vramPeakBytes === null;
}

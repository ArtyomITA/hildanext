import { describe, expect, it } from "vitest";
import { DEFAULT_CHAT_CONFIG } from "../features/chat/catalog";
import { composeChatInput } from "../features/chat/promptComposer";
import { ChatTurn } from "../domain/types";

function makeTurn(prompt: string, text: string): ChatTurn {
  return {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    prompt,
    status: "completed",
    config: { ...DEFAULT_CHAT_CONFIG },
    lanes: [
      {
        lane: "ar",
        modelId: "ar_qwen3_0_6b",
        status: "success",
        text,
        message: "",
        engine: "ar-greedy",
        tokensPerSec: 10,
        stepsToConverge: 1,
        vramPeakBytes: 100,
        dummyModel: false,
      },
    ],
  };
}

describe("promptComposer", () => {
  it("creates system+history+user order", () => {
    const turns = [makeTurn("u1", "a1"), makeTurn("u2", "a2")];
    const composed = composeChatInput(turns, "final user", {
      ...DEFAULT_CHAT_CONFIG,
      contextWindowTokens: 8192,
      systemPrompt: "SYS",
    });
    expect(composed.messages[0]).toEqual({ role: "system", content: "SYS" });
    expect(composed.messages.at(-1)).toEqual({ role: "user", content: "final user" });
    expect(composed.includedTurns).toBe(2);
  });

  it("trims history when budget is small", () => {
    const longUser = Array.from({ length: 260 }, () => "user").join(" ");
    const longAssistant = Array.from({ length: 260 }, () => "assistant").join(" ");
    const turns = [
      makeTurn(longUser, longAssistant),
      makeTurn(longUser, longAssistant),
      makeTurn(longUser, longAssistant),
    ];
    const composed = composeChatInput(turns, "new question", {
      ...DEFAULT_CHAT_CONFIG,
      contextWindowTokens: 520,
    });
    expect(composed.includedTurns).toBeLessThan(3);
  });

  it("maps thinking auto from decode mode", () => {
    const sMode = composeChatInput([], "prompt", {
      ...DEFAULT_CHAT_CONFIG,
      thinkingMode: "auto",
      decodeMode: "S_MODE",
    });
    const qMode = composeChatInput([], "prompt", {
      ...DEFAULT_CHAT_CONFIG,
      thinkingMode: "auto",
      decodeMode: "Q_MODE",
    });
    expect(sMode.enableThinking).toBe(true);
    expect(qMode.enableThinking).toBe(false);
  });
});

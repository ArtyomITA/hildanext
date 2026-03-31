import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CHAT_CONFIG } from "../features/chat/catalog";
import { runChatTurn } from "../features/chat/orchestrator";

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("chat orchestrator", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps both lanes when one succeeds and one fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/generate/ar")) {
          return jsonResponse({
            text: "AR ok",
            engine: "ar-greedy",
            stats: { tokens_per_sec: 40, dummy_model: false },
          });
        }
        return jsonResponse({ detail: "dLLM failed" }, 500);
      }),
    );

    const lanes = await runChatTurn("prompt", {
      ...DEFAULT_CHAT_CONFIG,
      engineMode: "BOTH",
    });

    expect(lanes).toHaveLength(2);
    expect(lanes[0]).toMatchObject({ lane: "ar", status: "success", text: "AR ok" });
    expect(lanes[1]).toMatchObject({ lane: "dllm", status: "error" });
  });

  it("marks a lane as offline without breaking the other lane", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/generate/ar")) {
          throw new TypeError("Failed to fetch");
        }
        return jsonResponse({
          text: "dLLM ok",
          engine: "transformers",
          stats: { tokens_per_sec: 30, steps_to_converge: 4, dummy_model: false },
        });
      }),
    );

    const lanes = await runChatTurn("prompt", {
      ...DEFAULT_CHAT_CONFIG,
      engineMode: "BOTH",
    });

    expect(lanes).toHaveLength(2);
    expect(lanes[0]).toMatchObject({ lane: "ar", status: "offline" });
    expect(lanes[1]).toMatchObject({ lane: "dllm", status: "success", text: "dLLM ok" });
  });
});

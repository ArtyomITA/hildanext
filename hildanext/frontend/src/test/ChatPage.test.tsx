import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatPage } from "../routes/chat/ChatPage";
import { clearChatStorageForTests } from "../store/chatStore";

interface MockEventSource {
  url: string;
  onopen: (() => void) | null;
  onmessage: ((event: MessageEvent<string>) => void) | null;
  onerror: ((event: Event) => void) | null;
  close: ReturnType<typeof vi.fn>;
}

let eventSources: MockEventSource[] = [];

function mockJson(data: unknown) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ChatPage", () => {
  beforeEach(() => {
    clearChatStorageForTests();
    localStorage.clear();
    vi.restoreAllMocks();
    eventSources = [];
    class FakeEventSource {
      url: string;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent<string>) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      close = vi.fn();
      constructor(url: string) {
        this.url = String(url);
        eventSources.push(this as unknown as MockEventSource);
      }
    }
    vi.stubGlobal("EventSource", FakeEventSource);
  });

  it("sends requests in AR/dLLM/Both modes", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return mockJson({ status: "ok", model_loaded: false });
      }
      if (url.includes("/api/generate/ar")) {
        return mockJson({
          text: "AR output",
          engine: "ar-greedy",
          stats: { tokens_per_sec: 42, tokens_generated: 12, dummy_model: false },
        });
      }
      return mockJson({
        text: "dLLM output",
        engine: "transformers",
        stats: { tokens_per_sec: 28, steps_to_converge: 4, dummy_model: false },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ChatPage />);

    await user.click(screen.getByRole("button", { name: /Carica pesi modello/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Scrivi prompt/i)).not.toBeDisabled();
    });
    fetchMock.mockClear();

    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "primo prompt");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[0]?.[0]).toContain("/api/generate/ar");
    expect(fetchMock.mock.calls[1]?.[0]).toContain("/api/generate");

    await user.selectOptions(screen.getByLabelText("Engine mode"), "AR");
    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "secondo prompt");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(fetchMock.mock.calls[2]?.[0]).toContain("/api/generate/ar");

    await user.selectOptions(screen.getByLabelText("Engine mode"), "DLLM");
    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "terzo prompt");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
    expect(fetchMock.mock.calls[3]?.[0]).toContain("/api/generate");
  });

  it("handles advanced drawer and presets", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/health")) return mockJson({ status: "ok", model_loaded: true });
        return mockJson({});
      }),
    );
    render(<ChatPage />);

    await user.click(screen.getByRole("button", { name: /Mostra avanzate/i }));
    const tauMask = screen.getByRole("spinbutton", { name: /Tau mask/i }) as HTMLInputElement;
    await user.clear(tauMask);
    await user.type(tauMask, "0.22");
    expect(tauMask.value).toBe("0.22");

    await user.type(screen.getByPlaceholderText("Nome preset"), "Preset test");
    await user.click(screen.getByRole("button", { name: "Salva" }));
    expect(screen.getByText("Preset test")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Reset config/i }));
    expect((screen.getByRole("spinbutton", { name: /Tau mask/i }) as HTMLInputElement).value).toBe("0.08");

    await user.click(screen.getByRole("button", { name: "Applica" }));
    expect((screen.getByRole("spinbutton", { name: /Tau mask/i }) as HTMLInputElement).value).toBe("0.22");
  });

  it("renders lane-specific offline state without crashing the turn", async () => {
    const user = userEvent.setup();
    let arCalls = 0;
    let dllmCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/health")) {
          return mockJson({ status: "ok", model_loaded: false });
        }
        if (url.includes("/api/generate/ar")) {
          arCalls += 1;
          if (arCalls === 1) {
            return mockJson({
              text: "warmup-ar",
              engine: "ar-greedy",
              stats: { tokens_per_sec: 20, tokens_generated: 1, dummy_model: false },
            });
          }
          return new Response("offline", { status: 503 });
        }
        dllmCalls += 1;
        return mockJson({
          text: dllmCalls === 1 ? "warmup-dllm" : "dLLM online",
          engine: "transformers",
          stats: { tokens_per_sec: 19, steps_to_converge: 6, dummy_model: false },
        });
      }),
    );

    render(<ChatPage />);
    await user.click(screen.getByRole("button", { name: /Carica pesi modello/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Scrivi prompt/i)).not.toBeDisabled();
    });
    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "prompt offline");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => {
      expect(screen.getByText(/Parziale: backend offline/i)).toBeInTheDocument();
    });
    expect(screen.getByText("dLLM online")).toBeInTheDocument();
    expect(screen.getByText(/Backend offline o non raggiungibile/i)).toBeInTheDocument();
  });

  it("opens raw viewer when clicking a successful lane response", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return mockJson({ status: "ok", model_loaded: false });
      }
      if (url.includes("/api/generate/ar")) {
        return mockJson({
          text: "AR raw output",
          engine: "ar-sampling",
          stats: { tokens_per_sec: 11, finish_reason: "length", truncated: true, dummy_model: false },
        });
      }
      return mockJson({
        text: "dLLM raw output",
        engine: "transformers",
        stats: { tokens_per_sec: 9, steps_to_converge: 3, dummy_model: false },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ChatPage />);
    await user.click(screen.getByRole("button", { name: /Carica pesi modello/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Scrivi prompt/i)).not.toBeDisabled();
    });
    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "prompt raw");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => expect(screen.getAllByLabelText(/Apri vista raw risposta/i).length).toBeGreaterThan(0));
    await user.click(screen.getAllByLabelText(/Apri vista raw risposta/i)[0]);
    const dialog = screen.getByRole("dialog", { name: /Raw risposta/i });
    expect(within(dialog).getByText(/Raw text/i)).toBeInTheDocument();
    expect(within(dialog).getByText(/AR raw output|dLLM raw output/i)).toBeInTheDocument();
  });

  it("does not open raw viewer when expanding thinking dropdown", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return mockJson({ status: "ok", model_loaded: false });
      }
      if (url.includes("/api/generate/ar")) {
        return mockJson({
          text: "<think>ragionamento ar</think>Risposta AR finale",
          engine: "ar-sampling",
          stats: { tokens_per_sec: 13, dummy_model: false },
        });
      }
      return mockJson({
        text: "dLLM output",
        engine: "transformers",
        stats: { tokens_per_sec: 9, dummy_model: false },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ChatPage />);
    await user.click(screen.getByRole("button", { name: /Carica pesi modello/i }));
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Scrivi prompt/i)).not.toBeDisabled();
    });
    await user.type(screen.getByPlaceholderText(/Scrivi prompt/i), "prompt think");
    await user.click(screen.getByRole("button", { name: "Invia" }));

    await waitFor(() => expect(screen.getByText(/Thinking interno/i)).toBeInTheDocument());
    await user.click(screen.getByText(/Thinking interno/i));
    expect(screen.getByText(/ragionamento ar/i)).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /Raw risposta/i })).not.toBeInTheDocument();
  });

  it("renders realtime inference panel and ingests SSE log events", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/health")) return mockJson({ status: "ok", model_loaded: false });
        if (url.includes("/api/generate/ar")) {
          return mockJson({ text: "AR out", engine: "ar-greedy", stats: { dummy_model: false } });
        }
        return mockJson({ text: "dLLM out", engine: "transformers", stats: { dummy_model: false } });
      }),
    );

    render(<ChatPage />);
    expect(screen.getByText(/Inference Realtime Log/i)).toBeInTheDocument();
    expect(eventSources.length).toBeGreaterThan(0);

    await act(async () => {
      eventSources[0]?.onmessage?.(
        {
          data: JSON.stringify({
            id: "11",
            tsUtc: new Date().toISOString(),
            level: "info",
            source: "inference",
            event: "DLLM_REQ_DONE",
            lane: "dllm",
            scope: "DLLM",
            benchmark: null,
            message: "finish=converged tokens=12",
            meta: { answer_preview: "ciao" },
          }),
        } as MessageEvent<string>,
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/DLLM_REQ_DONE/i)).toBeInTheDocument();
      expect(screen.getByText(/finish=converged tokens=12/i)).toBeInTheDocument();
    });

    await user.click(screen.getByText("details"));
    expect(screen.getByText(/answer_preview/i)).toBeInTheDocument();
  });

  it("falls back to polling when SSE stream errors", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/health")) return mockJson({ status: "ok", model_loaded: false });
      if (url.includes("/api/inference/logs")) {
        return mockJson({
          status: "ok",
          events: [
            {
              id: "21",
              tsUtc: new Date().toISOString(),
              level: "info",
              source: "inference",
              event: "INFER_REQ_DONE",
              lane: null,
              scope: "DLLM",
              benchmark: null,
              message: "polling row",
              meta: {},
            },
          ],
        });
      }
      return mockJson({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ChatPage />);
    expect(eventSources.length).toBeGreaterThan(0);
    await act(async () => {
      eventSources[0]?.onerror?.(new Event("error"));
    });

    await waitFor(
      () => {
        expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/api/inference/logs"));
        expect(screen.getByText(/polling row/i)).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });
});

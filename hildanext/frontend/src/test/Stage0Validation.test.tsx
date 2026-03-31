import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReactNode } from "react";

vi.mock("../components/charts/TimeseriesChart", () => ({
  TimeseriesChart: ({ yLabel }: { yLabel: string; children?: ReactNode }) => (
    <div data-testid="timeseries-mock">{yLabel}</div>
  ),
}));

import { Stage0Validation } from "../features/stage0/Stage0Validation";

function mockJson(data: unknown) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Stage0Validation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockImplementation(() => ({
        matches: false,
        media: "",
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    );
  });

  it("does not auto-start benchmarks on mount", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<Stage0Validation />);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("runs denoising stability only on explicit click", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/stability")) {
        return mockJson({
          status: "ok",
          text: "Paris.",
          points: [
            { step: 1, mean_confidence: 0.21, mask_ratio: 0.82 },
            { step: 2, mean_confidence: 0.48, mask_ratio: 0.44 },
            { step: 3, mean_confidence: 0.73, mask_ratio: 0.0 },
          ],
        });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);
    await user.click(screen.getByRole("button", { name: /Run Stability Check/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/api/stage0/validate/stability");
    expect(screen.getByText("Paris.")).toBeInTheDocument();
  });

  it("runs HellaSwag benchmark with progress updates", async () => {
    const user = userEvent.setup();
    const items = [
      {
        id: "hs-1",
        stem: "A chef slices vegetables on a cutting board.",
        endings: ["They cook a soup.", "They fly to space.", "They freeze the pan.", "They close a browser tab."],
        label: 0,
      },
      {
        id: "hs-2",
        stem: "A player dribbles past a defender.",
        endings: ["They take a shot.", "They water plants.", "They knit a scarf.", "They shut down the arena."],
        label: 0,
      },
      {
        id: "hs-3",
        stem: "A coder runs tests before merge.",
        endings: ["They inspect failures.", "They eat the keyboard.", "They remove electricity.", "They plant trees in RAM."],
        label: 0,
      },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/hellaswag/items")) {
        return mockJson({
          status: "ok",
          items,
          dataset_path: "data/benchmarks/hellaswag/hellaswag_val.jsonl",
          source: "cache",
          total_items: 10042,
        });
      }
      if (url.includes("/api/stage0/validate/hellaswag-item")) {
        return mockJson({ predicted_idx: 0, predicted_label: "A", raw_text: "A" });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);
    await user.click(screen.getByRole("button", { name: /Run HellaSwag/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1 + items.length));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("runs MMLU-Pro benchmark only on explicit click", async () => {
    const user = userEvent.setup();
    const items = [
      {
        id: "m1",
        question: "Q1",
        options: ["o1", "o2", "o3", "o4"],
        answer_label: "A",
      },
      {
        id: "m2",
        question: "Q2",
        options: ["o1", "o2", "o3", "o4"],
        answer_label: "B",
      },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/mmlu-pro/items")) {
        return mockJson({
          status: "ok",
          items,
          dataset_id: "TIGER-Lab/MMLU-Pro",
          source: "huggingface",
          split: "test",
          total_items: 12032,
        });
      }
      if (url.includes("/api/stage0/validate/mmlu-pro-item")) {
        return mockJson({ is_correct: true, predicted_label: "A", target_label: "A" });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);
    await user.click(screen.getByRole("button", { name: /Run MMLU-Pro/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1 + items.length));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("runs GSM8K benchmark only on explicit click", async () => {
    const user = userEvent.setup();
    const items = [
      { id: "g1", question: "1+1?", answer_target: "#### 2" },
      { id: "g2", question: "2+2?", answer_target: "#### 4" },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/gsm8k/items")) {
        return mockJson({
          status: "ok",
          items,
          dataset_id: "openai/gsm8k",
          source: "huggingface",
          split: "test",
          total_items: 1319,
        });
      }
      if (url.includes("/api/stage0/validate/gsm8k-item")) {
        return mockJson({ is_correct: true, predicted_number: "2", target_number: "2" });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);
    await user.click(screen.getByRole("button", { name: /Run GSM8K/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1 + items.length));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("forwards global and benchmark-specific settings in benchmark payload", async () => {
    const user = userEvent.setup();
    const items = [
      {
        id: "hs-1",
        stem: "A chef slices vegetables on a cutting board.",
        endings: ["They cook a soup.", "They fly to space.", "They freeze the pan.", "They close a browser tab."],
        label: 0,
      },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/hellaswag/items")) {
        return mockJson({ status: "ok", items, dataset_path: "x", source: "cache", total_items: 1 });
      }
      if (url.includes("/api/stage0/validate/hellaswag-item")) {
        return mockJson({ status: "ok", predicted_idx: 0, selected_lane: "dllm" });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);

    await user.selectOptions(screen.getByLabelText("Model scope"), "BOTH");
    await user.selectOptions(screen.getByLabelText("Context window"), "8192");
    await user.selectOptions(screen.getByLabelText("Generation effort"), "high");
    await user.selectOptions(screen.getByLabelText("Decoding strategy"), "sampling");
    await user.click(screen.getAllByText("⚙️ Settings")[0]);
    await user.selectOptions(screen.getByLabelText("HellaSwag n-shots"), "5");
    await user.click(screen.getByRole("button", { name: /Run HellaSwag/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const req =
      ((fetchMock.mock.calls as unknown as Array<[RequestInfo | URL, RequestInit | undefined]>)[1]?.[1] ??
        {}) as RequestInit;
    const body = JSON.parse(String(req.body ?? "{}"));
    expect(body.scope).toBe("BOTH");
    expect(body.context_window).toBe(8192);
    expect(body.decode_strategy).toBe("sampling");
    expect(body.n_shots).toBe(5);
    expect(body.max_new_tokens).toBe(2048);
  });

  it("can persist detailed benchmark logs to backend file without FE detailed view", async () => {
    const user = userEvent.setup();
    const items = [
      {
        id: "hs-1",
        stem: "A chef slices vegetables on a cutting board.",
        endings: ["They cook a soup.", "They fly to space.", "They freeze the pan.", "They close a browser tab."],
        label: 0,
      },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/stage0/validate/log/start")) {
        return mockJson({
          status: "ok",
          token: "tok_12345678",
          file_path: "runs/logs/benchmarks/hs.jsonl",
          file_name: "hs.jsonl",
        });
      }
      if (url.includes("/api/stage0/validate/hellaswag/items")) {
        return mockJson({ status: "ok", items, dataset_path: "x", source: "cache", total_items: 1 });
      }
      if (url.includes("/api/stage0/validate/hellaswag-item")) {
        return mockJson({ status: "ok", predicted_idx: 0, selected_lane: "dllm" });
      }
      if (url.includes("/api/stage0/validate/log/finish")) {
        return mockJson({ status: "ok", file_path: "runs/logs/benchmarks/hs.jsonl" });
      }
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Stage0Validation />);
    await user.click(screen.getByLabelText("Persist detailed logs to file"));
    await user.type(screen.getByLabelText("Detailed log run label"), "my-run");
    await user.click(screen.getByRole("button", { name: /Run HellaSwag/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
    const calls = fetchMock.mock.calls as unknown as Array<[RequestInfo | URL, RequestInit | undefined]>;
    const startReq = calls.find(([input]) => String(input).includes("/api/stage0/validate/log/start"));
    const itemReq = calls.find(([input]) => String(input).includes("/api/stage0/validate/hellaswag-item"));
    expect(startReq).toBeTruthy();
    expect(itemReq).toBeTruthy();
    const startBody = JSON.parse(String(startReq?.[1]?.body ?? "{}"));
    const itemBody = JSON.parse(String(itemReq?.[1]?.body ?? "{}"));
    expect(startBody.benchmark).toBe("hellaswag");
    expect(startBody.run_label).toBe("my-run");
    expect(itemBody.detailed_log_token).toBe("tok_12345678");
  });

  it("disables denoising stability when scope is AR", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn());
    render(<Stage0Validation />);

    await user.selectOptions(screen.getByLabelText("Model scope"), "AR");
    expect(screen.getByRole("button", { name: /Run Stability Check/i })).toBeDisabled();
    expect(screen.getByText(/Stability richiede lane dLLM/i)).toBeInTheDocument();
  });
});

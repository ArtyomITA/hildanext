import {
  InferenceRun,
  InferenceScenarioData,
  InferenceTraceStep,
  InsightCard,
  NormalizedLogEntry,
  ProcessSnapshot,
  TokenFrame,
  WsdMeta,
  WsdMetricRow,
  WsdScenarioData,
} from "../domain/types";

export interface MockInferenceControls {
  prompt: string;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  seed: number;
  mode: "S_MODE" | "Q_MODE";
  effort: "instant" | "low" | "medium" | "high" | "adaptive";
  tauMask: number;
  tauEdit: number;
  scenarioFlavor: "clean" | "degenerate" | "dummy";
}

function isoFrom(base: Date, offsetSeconds: number) {
  return new Date(base.getTime() + offsetSeconds * 1000).toISOString();
}

function createLog(
  id: string,
  tsUtc: string,
  entry: Partial<NormalizedLogEntry> & Pick<NormalizedLogEntry, "source" | "message">,
): NormalizedLogEntry {
  return {
    id,
    tsUtc,
    level: "info",
    tags: [],
    ...entry,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function deriveTopic(prompt: string) {
  const cleaned = prompt.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "diffusion inference behavior";
  }
  return cleaned.length > 72 ? `${cleaned.slice(0, 72).trim()}...` : cleaned;
}

function buildArOutput(prompt: string, maxNewTokens: number) {
  const topic = deriveTopic(prompt);
  const base =
    `AR answers ${topic} in a single left-to-right pass, so it commits early and revises only indirectly through later context.`;
  return base.split(" ").slice(0, Math.max(12, Math.floor(maxNewTokens * 0.55))).join(" ");
}

function buildDiffusionOutput(prompt: string, controls: MockInferenceControls) {
  const topic = deriveTopic(prompt);
  const style =
    controls.temperature > 0.85
      ? "with wider exploration and more revision pressure"
      : controls.temperature < 0.35
        ? "with tighter confidence thresholds and lower variance"
        : "with balanced drafting and correction passes";

  const base =
    `WSD diffusion answers ${topic} ${style}, drafting uncertain spans in parallel, then using gamma commits and delta edits until the mask ratio collapses under the selected thresholds.`;
  return base
    .split(" ")
    .slice(0, Math.max(16, Math.floor(controls.maxNewTokens * 0.72)))
    .join(" ");
}

export function generateWsdMetrics(variant: "healthy" | "phase" | "pressure" | "resume") {
  const metrics: WsdMetricRow[] = [];

  for (let step = 1; step <= 180; step += 1) {
    let phase: WsdMetricRow["phase"] = "warmup";
    let phaseProgress = step / 40;
    let blockSize = 1;

    if (step > 40 && step <= 140) {
      phase = "stable";
      phaseProgress = (step - 40) / 100;
      blockSize = 512;
    } else if (step > 140) {
      phase = "decay";
      phaseProgress = (step - 140) / 40;
      blockSize = Math.max(32, 512 - (step - 140) * 12);
    } else if (step > 20) {
      blockSize = 64;
    } else if (step > 8) {
      blockSize = 32;
    } else if (step > 3) {
      blockSize = 4;
    }

    const pressureBoost = variant === "pressure" ? 2100 : 0;
    const lossBump = variant === "phase" && step > 138 && step < 150 ? 0.17 : 0;
    const resumeDrop = variant === "resume" && step === 112 ? 0.35 : 0;
    const loss = Math.max(
      0.58,
      3.2 - step * 0.012 + Math.sin(step / 11) * 0.08 + lossBump + resumeDrop,
    );
    const maskedAcc = Math.min(0.95, 0.39 + step * 0.0028 + (phase === "stable" ? 0.06 : 0));
    const tokensPerSec =
      710 + step * 8 + (phase === "stable" ? 160 : 0) - (variant === "pressure" ? step * 0.9 : 0);
    const vramAllocMb = 3320 + step * 7 + pressureBoost + (phase === "stable" ? 520 : 0);
    const vramPeakMb = vramAllocMb + 420 + Math.cos(step / 4) * 70;
    const tMean = phase === "stable" ? 0.47 : phase === "warmup" ? 0.26 : 0.62;

    metrics.push({
      kind: "cpt",
      step,
      phase,
      blockSize,
      loss,
      lossM2T: loss * 0.61,
      lossT2T: loss * 0.39,
      maskedTokenAcc: maskedAcc,
      lr: step < 100 ? 1e-4 : 7e-5,
      gradNorm: 0.82 + Math.sin(step / 9) * 0.24 + (variant === "pressure" ? 0.2 : 0),
      tokensPerSec,
      stepTimeS: Math.max(0.6, 1.8 - step * 0.003 + (variant === "pressure" ? 0.35 : 0)),
      vramAllocMb,
      vramReservedMb: vramAllocMb + 640,
      vramPeakMb,
      etaStageSec: Math.max(120, (180 - step) * 14),
      tSampled: Math.max(0.02, Math.min(0.97, tMean + Math.sin(step / 6) * 0.11)),
      tMean,
      tMin: 0.001,
      tMax: 1,
      maskRatioActual: Math.max(0.03, 0.34 - step * 0.0012 + (phase === "decay" ? 0.05 : 0)),
      predPositionsCount: 900 + step * 4,
      wsdPhaseProgress: Math.min(1, phaseProgress),
      bidirectional: phase === "stable",
      isCausalEffective: phase !== "stable",
      attentionMode: "bidirectional_only_stable",
      shiftMode: "preserve_left_shift",
      timeParam: "continuous_time",
      lossWeighting: "inv_t",
      lossByTBucket: {
        "0.0-0.1": loss * 0.79,
        "0.1-0.3": loss * 0.88,
        "0.3-0.6": loss,
        "0.6-1.0": loss * 1.08,
      },
      accMaskedByTBucket: {
        "0.0-0.1": maskedAcc + 0.06,
        "0.1-0.3": maskedAcc + 0.03,
        "0.3-0.6": maskedAcc,
        "0.6-1.0": maskedAcc - 0.05,
      },
    });
  }

  return metrics;
}

export function generateProcesses(
  metrics: WsdMetricRow[],
  variant: "healthy" | "phase" | "pressure" | "resume",
) {
  return metrics
    .filter((row) => row.step % 6 === 0)
    .map<ProcessSnapshot>((row, index) => ({
      tsUtc: isoFrom(new Date("2026-03-02T00:55:41.000Z"), index * 18),
      processName: "python.exe",
      pid: 20844,
      cpuPct: 43 + Math.sin(index / 3) * 14,
      ramMb: 7560 + index * 9,
      gpuVramMb: row.vramPeakMb,
      gpuUtilPct: 71 + Math.cos(index / 4) * 11 - (variant === "pressure" ? 9 : 0),
      status: variant === "resume" && index > 13 && index < 15 ? "restarting" : "running",
    }));
}

export function generateWsdLogs(
  metrics: WsdMetricRow[],
  variant: "healthy" | "phase" | "pressure" | "resume",
) {
  const base = new Date("2026-03-02T00:55:41.000Z");
  const logs: NormalizedLogEntry[] = [];

  logs.push(
    createLog("console-0", isoFrom(base, 0), {
      source: "console",
      message:
        "[2026-03-02 00:55:41] START run-wsd attempt=1 config=runs/configs/llada21_dolma_wsd_only.json",
      tags: ["RUN_START"],
    }),
  );

  for (const row of metrics) {
    const stamp = isoFrom(base, row.step * 13);

    logs.push(
      createLog(`train-${row.step}`, stamp, {
        source: "training",
        level: row.vramPeakMb > 7000 ? "warning" : "info",
        module: "training",
        func: "_run",
        eventType: "metric",
        action: "step",
        reason: row.phase,
        message: `stage=wsd step=${row.step}/180 phase=${row.phase} block=${row.blockSize} loss=${row.loss.toFixed(4)} mta=${row.maskedTokenAcc?.toFixed(3)} tok=${Math.round(row.tokensPerSec)} vram=${Math.round(row.vramPeakMb)}MB`,
        tags: [row.phase, row.bidirectional ? "BIDIRECTIONAL" : "CAUSAL"],
        extra: {
          step: row.step,
          phase: row.phase,
          blockSize: row.blockSize,
          vramPeakMb: row.vramPeakMb,
        },
      }),
    );

    if (row.step % 12 === 0) {
      logs.push(
        createLog(`metric-${row.step}`, stamp, {
          source: "metric",
          level: "notice",
          module: "training",
          func: "_run",
          eventType: "metric",
          action: "masked_token_acc",
          reason: "rolling",
          message: `metric cpt.loss_total=${row.loss.toFixed(4)} cpt.mta=${row.maskedTokenAcc?.toFixed(4)}`,
          tags: ["metrics"],
          extra: { loss: row.loss, maskedTokenAcc: row.maskedTokenAcc },
        }),
      );
    }

    if (row.step === 41 || row.step === 141) {
      logs.push(
        createLog(`phase-${row.step}`, stamp, {
          source: "console",
          level: "notice",
          module: "training",
          func: "_run",
          eventType: "notice",
          action: "PHASE_CHANGE",
          reason: row.phase,
          message: `PHASE_CHANGE wsd_phase=${row.phase} block_size=${row.blockSize} bidirectional=${row.bidirectional}`,
          tags: ["PHASE_CHANGE", row.phase],
        }),
      );
    }
  }

  logs.push(
    createLog("fallback-flash", isoFrom(base, 39), {
      source: "fallback",
      level: "notice",
      module: "inference",
      func: "load_model_bundle",
      eventType: "fallback",
      action: "force_math_sdpa",
      reason: "flash_attention_unavailable",
      message: "fallback force_math_sdpa because flash_attention_unavailable",
      tags: ["fallback", "cuda"],
    }),
  );

  if (variant === "pressure") {
    logs.push(
      createLog("pressure-oom", isoFrom(base, 1540), {
        source: "fallback",
        level: "error",
        module: "training",
        func: "_run",
        eventType: "fallback",
        action: "emergency_checkpoint",
        reason: "oom_runtime",
        message: "emergency checkpoint saved after OOM runtime; seq_len downscale suggested",
        tags: ["OOM", "checkpoint"],
      }),
    );
  }

  if (variant === "resume") {
    logs.push(
      createLog("resume", isoFrom(base, 1490), {
        source: "console",
        level: "notice",
        module: "training",
        func: "_run",
        eventType: "notice",
        action: "resume",
        reason: "resumed_from_checkpoint",
        message: "resume from step_00110 after overnight restart",
        tags: ["resume", "checkpoint"],
      }),
    );
  }

  return logs.sort((a, b) => a.tsUtc.localeCompare(b.tsUtc));
}

export function generateInsights(
  variant: "healthy" | "phase" | "pressure" | "resume" | "clean" | "degenerate" | "dummy",
): InsightCard[] {
  return [
    {
      id: "masked-acc",
      title: "Masked accuracy is leading loss improvement",
      metric: "masked_token_acc",
      body: "Masked token accuracy climbs before the loss fully settles. This is a good sign that diffusion supervision is tightening before convergence becomes visually obvious.",
      tone: "info",
    },
    {
      id: "bidir",
      title: "Stable-only bidirectional attention is active",
      metric: "bidirectional_only_stable",
      body: "Warmup stays causal, stable opens the context, decay closes again. This mirrors the training logic already present in the backend tests and config defaults.",
      tone: "info",
    },
    {
      id: "pressure",
      title: variant === "pressure" || variant === "degenerate" ? "Risk envelope is tightening" : "Throughput remains clean",
      metric: "vram_peak_mb",
      body:
        variant === "pressure" || variant === "degenerate"
          ? "The interesting state is where speed, confidence and memory stop agreeing. This UI keeps those signals adjacent on purpose."
          : "Peak allocation and token throughput are moving together, so the run still looks healthy.",
      tone: variant === "pressure" || variant === "degenerate" ? "critical" : "info",
    },
  ];
}

export function generateWsdScenario(
  id: WsdScenarioData["id"],
  label: string,
  variant: "healthy" | "phase" | "pressure" | "resume",
): WsdScenarioData {
  const metrics = generateWsdMetrics(variant);
  const meta: WsdMeta = {
    runId: "run-20260302T005541Z-1131409a",
    configDigest: "be4ed061cdafa358",
    optimizer: variant === "pressure" ? "AdamW" : "AdamW8bit",
    dtype: "torch.float16",
    device: "cuda:0",
    dummyModel: false,
    phase: metrics.at(-1)?.phase ?? "decay",
    blockSize: metrics.at(-1)?.blockSize ?? 32,
    ladderBlocks: [1, 4, 32, 64, 512],
  };

  return {
    id,
    label,
    dataSource: "mockup",
    meta,
    metrics,
    logs: generateWsdLogs(metrics, variant),
    processes: generateProcesses(metrics, variant),
    insights: generateInsights(variant),
  };
}

function tokenizeWords(text: string, lane: "ar" | "diffusion") {
  return text.split(" ").map((word, index) => {
    const state: TokenFrame["tokens"][number]["state"] = lane === "ar" ? "stable" : "masked";
    return {
      index,
      text: word,
      state,
      confidence: lane === "ar" ? 1 : 0.12,
      lane,
    };
  });
}

export function generateInferenceScenario(
  id: InferenceScenarioData["id"],
  label: string,
  mode: "clean" | "degenerate" | "dummy",
): InferenceScenarioData {
  const prompt =
    "Summarize why WSD diffusion can revise tokens while AR decoding only commits left-to-right.";
  const arOutput =
    "AR decoding commits one token at a time, so corrections need later context or extra reranking.";
  const diffusionOutput =
    "WSD diffusion drafts in parallel, then revises uncertain positions through gamma commits and delta edits until the mask ratio collapses.";
  const base = new Date("2026-03-02T01:12:00.000Z");

  const diffusionLogs: InferenceTraceStep[] = [
    { step: 1, maskRatio: 0.78, gammaCount: 11, deltaCount: 0, avgConfMasked: 0.62, avgConfTokens: null, tauMask: 0.08, tauEdit: 0.08 },
    { step: 2, maskRatio: 0.51, gammaCount: 10, deltaCount: 2, avgConfMasked: 0.69, avgConfTokens: 0.74, tauMask: 0.08, tauEdit: 0.08 },
    { step: 3, maskRatio: 0.28, gammaCount: 7, deltaCount: 4, avgConfMasked: 0.73, avgConfTokens: 0.81, tauMask: 0.08, tauEdit: 0.08 },
    { step: 4, maskRatio: mode === "degenerate" ? 0.24 : 0.08, gammaCount: mode === "degenerate" ? 0 : 4, deltaCount: 3, avgConfMasked: 0.75, avgConfTokens: 0.86, tauMask: 0.08, tauEdit: 0.08, tauFallbackApplied: mode === "degenerate" || undefined, tauMaskAfterFallback: mode === "degenerate" ? 0.068 : undefined },
    { step: 5, maskRatio: mode === "degenerate" ? 0.12 : 0, gammaCount: mode === "degenerate" ? 3 : 1, deltaCount: 1, avgConfMasked: 0.81, avgConfTokens: 0.89, tauMask: mode === "degenerate" ? 0.068 : 0.08, tauEdit: 0.08 },
  ];

  const baseTokens = tokenizeWords(diffusionOutput, "diffusion");
  const tokenFrames: TokenFrame[] = diffusionLogs.map((log) => ({
    step: log.step,
    tokens: baseTokens.map((token) => {
      if (token.index < 5) {
        return { ...token, state: "prompt", confidence: 1 };
      }
      if (log.step === 1) {
        return { ...token, state: token.index % 2 === 0 ? "masked" : "new", confidence: 0.61 };
      }
      if (log.step === 2) {
        return { ...token, state: token.index % 4 === 0 ? "edited" : "stable", confidence: 0.74 };
      }
      if (log.step === 3) {
        return { ...token, state: token.index % 5 === 0 ? "edited" : "stable", confidence: 0.81 };
      }
      if (log.step === 4 && mode === "degenerate" && token.index % 7 === 0) {
        return { ...token, state: "masked", confidence: 0.42 };
      }
      return { ...token, state: "stable", confidence: 0.91 };
    }),
  }));

  const ar: InferenceRun = {
    engine: "ar",
    mode: "Q_MODE",
    effort: "low",
    prompt,
    outputText: arOutput,
    tauMask: 1,
    tauEdit: 0,
    steps: 1,
    stepsToConverge: 1,
    tokensPerSec: 1420,
    vramPeakBytes: 3.4 * 1024 * 1024 * 1024,
    dummyModel: false,
    envIssues: {},
    fallbacks: [],
    logs: [
      { step: 1, maskRatio: 0, gammaCount: arOutput.split(" ").length, deltaCount: 0, avgConfMasked: null, avgConfTokens: 0.93, tauMask: 1, tauEdit: 0 },
    ],
    tokenFrames: [
      {
        step: 1,
        tokens: tokenizeWords(arOutput, "ar").map((token) => ({
          ...token,
          state: "stable",
          confidence: 0.95,
        })),
      },
    ],
  };

  const fallbackLogs: NormalizedLogEntry[] = [
    createLog("infer-fallback-1", isoFrom(base, 1), {
      source: "fallback",
      level: "notice",
      module: "inference",
      func: "load_model_bundle",
      eventType: "fallback",
      action: "force_math_sdpa",
      reason: "flash_attention_unavailable",
      message: "flash attention unavailable, using math sdpa",
      tags: ["fallback", "cuda"],
    }),
    createLog("infer-env-1", isoFrom(base, 2), {
      source: "fallback",
      level: mode === "dummy" ? "warning" : "notice",
      module: "inference",
      func: "load_model_bundle",
      eventType: "env_issue",
      action: mode === "dummy" ? "dummy_model_fallback" : "numpy_dll_unavailable",
      reason: mode === "dummy" ? "force_dummy_model" : "Numpy is not available",
      message:
        mode === "dummy"
          ? "dummy model fallback forced for smoke-quality serving"
          : "numpy dll unavailable in current environment",
      tags: ["env", mode === "dummy" ? "dummy" : "numpy"],
    }),
  ];

  const diffusion: InferenceRun = {
    engine: "transformers",
    mode: "S_MODE",
    effort: mode === "degenerate" ? "adaptive" : "medium",
    prompt,
    outputText: diffusionOutput,
    tauMask: 0.08,
    tauEdit: 0.08,
    steps: diffusionLogs.length,
    stepsToConverge: mode === "degenerate" ? 5 : 4,
    tokensPerSec: mode === "degenerate" ? 620 : 890,
    vramPeakBytes: (mode === "degenerate" ? 4.8 : 4.1) * 1024 * 1024 * 1024,
    dummyModel: mode === "dummy",
    loadReason: mode === "dummy" ? "force_dummy_model=true" : "",
    envIssues:
      mode === "dummy" ? { dummy_model: "Forced mock-only mode" } : { numpy: "DLL unavailable" },
    fallbacks: fallbackLogs,
    logs: diffusionLogs,
    tokenFrames,
  };

  return {
    id,
    label,
    dataSource: "mockup",
    ar,
    diffusion,
    logs: [
      ...fallbackLogs,
      ...diffusionLogs.map((log) =>
        createLog(`infer-step-${log.step}`, isoFrom(base, 4 + log.step * 2), {
          source: "metric",
          level: log.gammaCount === 0 ? "warning" : "info",
          module: "inference",
          func: "TransformersEngine._decode",
          eventType: "metric",
          action: "decode_step",
          reason: `step_${log.step}`,
          message: `step=${log.step} mask_ratio=${log.maskRatio.toFixed(2)} gamma=${log.gammaCount} delta=${log.deltaCount} tau=${log.tauMask.toFixed(3)}`,
          tags: ["decode", log.gammaCount === 0 ? "plateau" : "active"],
        }),
      ),
    ],
    insights: generateInsights(mode),
  };
}

export function generateInteractiveInferenceScenario(
  id: InferenceScenarioData["id"],
  label: string,
  controls: MockInferenceControls,
): InferenceScenarioData {
  const base = new Date("2026-03-02T01:42:00.000Z");
  const prompt = controls.prompt.trim() || "Explain WSD diffusion with editable decoding.";
  const arOutput = buildArOutput(prompt, controls.maxNewTokens);
  const diffusionOutput = buildDiffusionOutput(prompt, controls);
  const scenarioFlavor = controls.scenarioFlavor;

  const baseSteps =
    controls.effort === "instant"
      ? 1
      : controls.effort === "low"
        ? 3
        : controls.effort === "medium"
          ? 5
          : controls.effort === "high"
            ? 7
            : 9;
  const modeAdjustment = controls.mode === "Q_MODE" ? -1 : 1;
  const temperatureAdjustment = controls.temperature > 0.9 ? 2 : controls.temperature < 0.35 ? -1 : 0;
  const stepCount = clamp(baseSteps + modeAdjustment + temperatureAdjustment, 1, 10);
  const initialMask = clamp(0.88 - controls.topP * 0.28 + controls.temperature * 0.12, 0.32, 0.94);
  const tauMask = clamp(controls.tauMask, 0.01, 1);
  const tauEdit = clamp(controls.tauEdit, 0.01, 1);
  const degrade = scenarioFlavor === "degenerate";
  const dummy = scenarioFlavor === "dummy";
  const throughputBase = controls.mode === "Q_MODE" ? 1080 : 840;
  const throughput =
    throughputBase -
    controls.temperature * 130 -
    stepCount * 22 +
    (controls.effort === "instant" ? 120 : 0) -
    (degrade ? 140 : 0);

  const diffusionLogs: InferenceTraceStep[] = Array.from({ length: stepCount }, (_, index) => {
    const step = index + 1;
    const progress = step / stepCount;
    const maskRatio =
      step === stepCount
        ? degrade
          ? 0.08
          : 0
        : clamp(initialMask - progress * (degrade ? 0.74 : 0.96), 0.04, 0.96);
    const gammaCount =
      degrade && step === Math.max(2, stepCount - 1)
        ? 0
        : Math.max(1, Math.round((controls.maxNewTokens * (1 - progress * 0.6)) / 9));
    const deltaCount = Math.max(
      0,
      Math.round((controls.temperature * 5 + (controls.mode === "S_MODE" ? 2 : 0)) * progress),
    );
    const avgConfMasked = clamp(0.51 + progress * 0.27 - controls.temperature * 0.08, 0.34, 0.94);
    const avgConfTokens = clamp(0.62 + progress * 0.24 - controls.temperature * 0.05, 0.4, 0.97);

    return {
      step,
      maskRatio,
      gammaCount,
      deltaCount,
      avgConfMasked,
      avgConfTokens,
      tauMask,
      tauEdit,
      tauFallbackApplied: degrade && step === Math.max(2, stepCount - 1) ? true : undefined,
      tauMaskAfterFallback:
        degrade && step === Math.max(2, stepCount - 1) ? clamp(tauMask * 0.85, 0.01, 1) : undefined,
    };
  });

  const arTokens = tokenizeWords(arOutput, "ar").map((token) => ({
    ...token,
    state: "stable" as const,
    confidence: clamp(0.91 - controls.temperature * 0.18, 0.5, 0.97),
  }));

  const diffusionBaseTokens = tokenizeWords(diffusionOutput, "diffusion");
  const tokenFrames: TokenFrame[] = diffusionLogs.map((log) => ({
    step: log.step,
    tokens: diffusionBaseTokens.map((token) => {
      if (token.index < 4) {
        return { ...token, state: "prompt", confidence: 1 };
      }
      if (log.step === 1) {
        return {
          ...token,
          state: token.index % 2 === 0 ? "masked" : "new",
          confidence: clamp(log.avgConfMasked ?? 0.5, 0.2, 0.95),
        };
      }
      if (log.tauFallbackApplied && token.index % 6 === 0) {
        return { ...token, state: "masked", confidence: 0.42 };
      }
      if (token.index % 5 === log.step % 5) {
        return { ...token, state: "edited", confidence: clamp(log.avgConfTokens ?? 0.6, 0.3, 0.98) };
      }
      return { ...token, state: "stable", confidence: clamp(log.avgConfTokens ?? 0.7, 0.3, 0.98) };
    }),
  }));

  const fallbackLogs: NormalizedLogEntry[] = [
    createLog("interactive-fallback-flash", isoFrom(base, 1), {
      source: "fallback",
      level: "notice",
      module: "inference",
      func: "load_model_bundle",
      eventType: "fallback",
      action: "force_math_sdpa",
      reason: "flash_attention_unavailable",
      message: "flash attention unavailable, using math sdpa",
      tags: ["fallback", "cuda"],
    }),
    createLog("interactive-params", isoFrom(base, 2), {
      source: "metric",
      level: "notice",
      module: "api",
      func: "generate",
      eventType: "metric",
      action: "request_profile",
      reason: "mock_generation",
      message: `temperature=${controls.temperature.toFixed(2)} top_p=${controls.topP.toFixed(2)} max_new_tokens=${controls.maxNewTokens} seed=${controls.seed}`,
      tags: ["params", controls.mode, controls.effort],
      extra: {
        temperature: controls.temperature,
        topP: controls.topP,
        maxNewTokens: controls.maxNewTokens,
        seed: controls.seed,
      },
    }),
  ];

  if (dummy) {
    fallbackLogs.push(
      createLog("interactive-dummy", isoFrom(base, 3), {
        source: "fallback",
        level: "warning",
        module: "inference",
        func: "load_model_bundle",
        eventType: "fallback",
        action: "dummy_model_fallback",
        reason: "force_dummy_model",
        message: "dummy model fallback forced for mock-only run",
        tags: ["dummy", "fallback"],
      }),
    );
  }

  if (degrade) {
    fallbackLogs.push(
      createLog("interactive-degenerate", isoFrom(base, 4), {
        source: "fallback",
        level: "warning",
        module: "inference",
        func: "TransformersEngine._decode",
        eventType: "fallback",
        action: "tau_mask_relax",
        reason: "degenerate_tau_relax",
        message: `tau mask relaxed from ${tauMask.toFixed(3)} to ${clamp(tauMask * 0.85, 0.01, 1).toFixed(3)}`,
        tags: ["degenerate", "tau_relax"],
      }),
    );
  }

  const ar: InferenceRun = {
    engine: "ar",
    mode: "Q_MODE",
    effort: controls.effort === "adaptive" ? "low" : controls.effort,
    prompt,
    outputText: arOutput,
    tauMask: 1,
    tauEdit: 0,
    steps: 1,
    stepsToConverge: 1,
    tokensPerSec: Math.round(throughput + 420),
    vramPeakBytes: (3.2 + controls.maxNewTokens / 220) * 1024 * 1024 * 1024,
    dummyModel: false,
    envIssues: {},
    fallbacks: [],
    logs: [
      {
        step: 1,
        maskRatio: 0,
        gammaCount: arTokens.length,
        deltaCount: 0,
        avgConfMasked: null,
        avgConfTokens: clamp(0.92 - controls.temperature * 0.16, 0.45, 0.98),
        tauMask: 1,
        tauEdit: 0,
      },
    ],
    tokenFrames: [{ step: 1, tokens: arTokens }],
  };

  const diffusion: InferenceRun = {
    engine: "transformers",
    mode: controls.mode,
    effort: controls.effort,
    prompt,
    outputText: diffusionOutput,
    tauMask,
    tauEdit,
    steps: stepCount,
    stepsToConverge: degrade ? stepCount : Math.max(1, stepCount - 1),
    tokensPerSec: Math.round(Math.max(220, throughput)),
    vramPeakBytes:
      (4.0 +
        controls.maxNewTokens / 180 +
        controls.temperature * 0.3 +
        (controls.effort === "adaptive" ? 0.4 : 0) +
        (degrade ? 0.5 : 0)) *
      1024 *
      1024 *
      1024,
    dummyModel: dummy,
    loadReason: dummy ? "force_dummy_model=true" : "",
    envIssues: dummy ? { dummy_model: "Forced mock-only mode" } : { numpy: "DLL unavailable" },
    fallbacks: fallbackLogs,
    logs: diffusionLogs,
    tokenFrames,
  };

  const liveLogs = diffusionLogs.map((log) =>
    createLog(`interactive-step-${log.step}`, isoFrom(base, 5 + log.step * 2), {
      source: "metric",
      level: log.gammaCount === 0 ? "warning" : "info",
      module: "inference",
      func: "TransformersEngine._decode",
      eventType: "metric",
      action: "decode_step",
      reason: `step_${log.step}`,
      message: `step=${log.step} mask_ratio=${log.maskRatio.toFixed(2)} gamma=${log.gammaCount} delta=${log.deltaCount} tau=${log.tauMask.toFixed(3)} temp=${controls.temperature.toFixed(2)}`,
      tags: ["decode", controls.mode, controls.effort],
      extra: {
        temperature: controls.temperature,
        topP: controls.topP,
        seed: controls.seed,
      },
    }),
  );

  return {
    id,
    label,
    dataSource: "mockup",
    ar,
    diffusion,
    logs: [...fallbackLogs, ...liveLogs],
    insights: generateInsights(scenarioFlavor),
  };
}

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useDataStore } from "../../store/dataStore";
import { useUiStore } from "../../store/uiStore";
import { MetricHeroCard } from "../../components/cards/MetricHeroCard";
import { DataSourceBar } from "../../components/layout/DataSourceBar";
import { Panel } from "../../components/layout/Panel";
import { VirtualLogTable } from "../../components/tables/VirtualLogTable";
import { StickyFilterBar } from "../../features/logs/StickyFilterBar";
import { useLogFeed } from "../../features/logs/useLogFeed";
import { InferenceSplitPane } from "../../features/compare/InferenceSplitPane";
import { PromptLab, PromptLabControls } from "../../features/compare/PromptLab";
import { DiffusionStepTimeline } from "../../features/diffusion-viz/DiffusionStepTimeline";
import { TokenMaskCanvas } from "../../features/diffusion-viz/TokenMaskCanvas";
import { GlossaryInspector } from "../../features/glossary/GlossaryInspector";
import { InsightCallout } from "../../features/insights/InsightCallout";
import { InferenceRun, InferenceScenarioData } from "../../domain/types";
import { generateInteractiveInferenceScenario } from "../../mocks/generators";
import { useArGenerate } from "../../features/run/useArGenerate";
import { StatusRail } from "../../shell/StatusRail";
import styles from "./InferencePage.module.css";

function toggle(items: string[], value: string) {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

export function InferencePage() {
  const [params, setParams] = useSearchParams();
  const inference = useDataStore((state) => state.inference);
  const setScenario = useDataStore((state) => state.setInferenceScenario);
  const selectedStep = useUiStore((state) => state.selectedStep);
  const selectStep = useUiStore((state) => state.selectStep);
  const selectLog = useUiStore((state) => state.selectLog);
  const selectedLogId = useUiStore((state) => state.selectedLogId);
  const [interactiveInference, setInteractiveInference] = useState<InferenceScenarioData | null>(null);
  const { running: arRunning, runAr } = useArGenerate();
  const [controls, setControls] = useState<PromptLabControls>({
    prompt:
      "Explain how editable diffusion decoding differs from autoregressive decoding for WSD inference.",
    temperature: 0.72,
    topP: 0.9,
    maxNewTokens: 96,
    seed: 42,
    mode: "S_MODE",
    effort: "medium",
    tauMask: 0.08,
    tauEdit: 0.08,
    scenarioFlavor: "clean",
  });

  const scenario = params.get("scenario") ?? "ar_vs_diffusion_compare";
  const query = params.get("q") ?? "";
  const levels = params.getAll("level");
  const sources = params.getAll("source");
  const stepParam = Number(params.get("step") ?? selectedStep);
  const activeInference = interactiveInference ?? inference;

  const dsItems = interactiveInference?.id === "real_ar_run"
    ? [
        { label: "source", value: interactiveInference.ar.dummyModel ? "Qwen 0.6B (dummy)" : "Qwen 0.6B" },
        { label: "engine", value: interactiveInference.ar.engine },
        { label: "mode", value: interactiveInference.ar.mode },
        { label: "effort", value: interactiveInference.ar.effort },
      ]
    : interactiveInference
    ? [
        { label: "source", value: "interactive" },
        { label: "engine", value: activeInference.diffusion.engine },
        { label: "mode", value: activeInference.diffusion.mode },
        { label: "effort", value: activeInference.diffusion.effort },
      ]
    : [
        { label: "source", value: activeInference.dataSource === "live" ? "backend API" : "offline mockup" },
        { label: "engine", value: activeInference.diffusion.engine },
        { label: "mode", value: activeInference.diffusion.mode },
        { label: "effort", value: activeInference.diffusion.effort },
      ];
  const inferenceDataSource = interactiveInference ? "live" : activeInference.dataSource;

  useEffect(() => {
    void setScenario(scenario);
    setInteractiveInference(null);
  }, [scenario, setScenario]);

  useEffect(() => {
    if (Number.isFinite(stepParam) && stepParam > 0) {
      selectStep(stepParam);
    }
  }, [selectStep, stepParam]);

  const snapshot = useLogFeed({ logs: activeInference.logs, query, levels, sources });

  function updateControl<K extends keyof PromptLabControls>(key: K, value: PromptLabControls[K]) {
    setControls((current) => ({ ...current, [key]: value }));
  }

  function setParamsClone(mutator: (clone: URLSearchParams) => void) {
    const clone = new URLSearchParams(params);
    mutator(clone);
    setParams(clone);
  }

  function handleGenerate() {
    const generated = generateInteractiveInferenceScenario(
      "interactive_prompt_run",
      "Interactive prompt run",
      controls,
    );
    setInteractiveInference(generated);
    selectStep(1);
    selectLog(null);
    setParamsClone((clone) => {
      clone.set("step", "1");
      clone.delete("log");
    });
  }

  async function handleGenerateReal() {
    const result = await runAr(controls.prompt, controls.maxNewTokens, controls.seed);
    if (!result) return; // error already captured in hook.error
    const arRun: InferenceRun = {
      engine: "ar",
      mode: "Q_MODE",
      effort: controls.effort,
      prompt: controls.prompt,
      outputText: result.text,
      tauMask: 1.0,
      tauEdit: 0.0,
      steps: 1,
      stepsToConverge: 1,
      tokensPerSec: result.stats?.tokens_per_sec ?? null,
      vramPeakBytes: null,
      dummyModel: result.stats?.dummy_model ?? false,
      loadReason: result.stats?.load_reason ?? "",
      envIssues: {},
      fallbacks: [],
      logs: [],
      tokenFrames: [
        {
          step: 1,
          tokens: result.text
            .split(" ")
            .slice(0, 64)
            .map((word, i) => ({
              index: i,
              text: word,
              state: "stable" as const,
              confidence: 0.95,
              lane: "ar" as const,
            })),
        },
      ],
    };
    const realScenario: InferenceScenarioData = {
      id: "real_ar_run",
      label: `AR Qwen — ${new Date().toLocaleTimeString()}`,
      dataSource: "live",
      ar: arRun,
      diffusion: activeInference.diffusion,
      logs: [
        {
          id: "ar-run-ok",
          tsUtc: new Date().toISOString(),
          source: "console",
          level: "info",
          message: `AR inference OK — ${result.stats?.tokens_generated ?? "?"} tokens @ ${Math.round(result.stats?.tokens_per_sec ?? 0)} tok/s · model=${result.stats?.dummy_model ? "dummy" : "Qwen-0.6B"} · dtype=${result.stats?.actual_dtype ?? "?"}`,
          tags: ["ar", "real"],
        },
      ],
      insights: [
        {
          id: "ar-real-run",
          title: `AR Qwen 0.6B — ${result.stats?.dummy_model ? "DUMMY model" : "live model"}`,
          metric: "tokensPerSec",
          body: `Generated ${result.stats?.tokens_generated ?? "?"} tokens at ${Math.round(result.stats?.tokens_per_sec ?? 0)} tok/s. Dtype: ${result.stats?.actual_dtype ?? "?"}. Engine: ${result.engine}.`,
          tone: result.stats?.dummy_model ? "warning" : "info",
        },
      ],
    };
    setInteractiveInference(realScenario);
    selectStep(1);
    selectLog(null);
    setParamsClone((clone) => {
      clone.set("step", "1");
      clone.delete("log");
    });
  }

  function handleReset() {
    setInteractiveInference(null);
    selectStep(1);
    selectLog(null);
    setParamsClone((clone) => {
      clone.set("step", "1");
      clone.delete("log");
    });
  }

  return (
    <div className={styles.page}>
      <DataSourceBar
        dataSource={inferenceDataSource}
        items={dsItems}
        hint="Inference backend not wired — use PromptLab to generate locally."
      />

      <section className={styles.hero}>
        <MetricHeroCard
          label="Prompt"
          value={interactiveInference ? "Live prompt" : "AR vs WSD"}
          accent="cyan"
          meta={activeInference.diffusion.prompt}
        />
        <MetricHeroCard
          label="Throughput"
          value={`${Math.round(activeInference.diffusion.tokensPerSec ?? 0)} tok/s`}
          accent="lime"
          meta={`AR ${Math.round(activeInference.ar.tokensPerSec ?? 0)} tok/s`}
        />
        <MetricHeroCard
          label="Converge"
          value={`${activeInference.diffusion.stepsToConverge ?? "n/a"} steps`}
          accent="orange"
          meta={`tau ${activeInference.diffusion.tauMask.toFixed(2)} / ${activeInference.diffusion.tauEdit.toFixed(2)}`}
        />
        <MetricHeroCard
          label="Fallback posture"
          value={activeInference.diffusion.dummyModel ? "Dummy" : "Live-style"}
          accent="red"
          meta={Object.keys(activeInference.diffusion.envIssues).join(", ") || "No critical env issues"}
        />
      </section>

      <div className={styles.layout}>
        <div className={styles.primary}>
          <PromptLab
            controls={controls}
            onChange={updateControl}
            onGenerate={handleGenerate}
            onReset={handleReset}
            onGenerateReal={handleGenerateReal}
            realRunning={arRunning}
            disabled={false}
          />

          <InferenceSplitPane
            ar={activeInference.ar}
            diffusion={activeInference.diffusion}
          />

          <Panel kicker="Diffusion mechanics" title="Step timeline">
            <DiffusionStepTimeline
              rows={activeInference.diffusion.logs}
              selectedStep={selectedStep}
              onSelect={(step) => {
                selectStep(step);
                setParamsClone((clone) => {
                  clone.set("step", String(step));
                });
              }}
            />
          </Panel>

          <Panel kicker="Token mask theater" title="Canvas-based token state replay">
            <TokenMaskCanvas frames={activeInference.diffusion.tokenFrames} selectedStep={selectedStep} />
          </Panel>

          <Panel kicker="Fallbacks + env" title="Virtualized log stream">
            <StickyFilterBar
              query={query}
              onQueryChange={(value) => {
                setParamsClone((clone) => {
                  clone.set("q", value);
                });
              }}
              levels={levels}
              onToggleLevel={(value) => {
                const next = toggle(levels, value);
                setParamsClone((clone) => {
                  clone.delete("level");
                  next.forEach((item) => clone.append("level", item));
                });
              }}
              sources={sources}
              onToggleSource={(value) => {
                const next = toggle(sources, value);
                setParamsClone((clone) => {
                  clone.delete("source");
                  next.forEach((item) => clone.append("source", item));
                });
              }}
            />
            <div className={styles.logMeta}>
              <span>{snapshot.summary.visible} visible</span>
              <span>{snapshot.summary.total} total</span>
            </div>
            <VirtualLogTable
              rows={snapshot.rows}
              selectedId={selectedLogId}
              onSelect={(id) => {
                selectLog(id);
                setParamsClone((clone) => {
                  clone.set("log", id);
                });
              }}
            />
          </Panel>
        </div>

        <div className={styles.side}>
          <StatusRail pathname="/inference" />
          <InsightCallout items={activeInference.insights} />
          <GlossaryInspector />
        </div>
      </div>
    </div>
  );
}

import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useDataStore } from "../../store/dataStore";
import { useUiStore } from "../../store/uiStore";
import { formatDuration, formatNumber, formatPct } from "../../domain/formatters";
import { MetricHeroCard } from "../../components/cards/MetricHeroCard";
import { PhaseTimeline } from "../../components/cards/PhaseTimeline";
import { TBucketHeatStrip } from "../../components/cards/TBucketHeatStrip";
import { TimeseriesChart } from "../../components/charts/TimeseriesChart";
import { DataSourceBar } from "../../components/layout/DataSourceBar";
import { Panel } from "../../components/layout/Panel";
import { RunControlPanel } from "../../features/run/RunControlPanel";
import { VirtualLogTable } from "../../components/tables/VirtualLogTable";
import { TerminalTranscript } from "../../components/terminals/TerminalTranscript";
import { ProcessRail } from "../../features/processes/ProcessRail";
import { GlossaryInspector } from "../../features/glossary/GlossaryInspector";
import { InsightCallout } from "../../features/insights/InsightCallout";
import { StickyFilterBar } from "../../features/logs/StickyFilterBar";
import { useLogFeed } from "../../features/logs/useLogFeed";
import { StatusRail } from "../../shell/StatusRail";
import styles from "./WsdPage.module.css";

function toggle(items: string[], value: string) {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

export function WsdPage() {
  const [params, setParams] = useSearchParams();
  const wsd = useDataStore((state) => state.wsd);
  const setScenario = useDataStore((state) => state.setWsdScenario);
  const selectedLogId = useUiStore((state) => state.selectedLogId);
  const selectLog = useUiStore((state) => state.selectLog);

  const scenario = "live_wsd_run";
  const query = params.get("q") ?? "";
  const levels = params.getAll("level");
  const sources = params.getAll("source");

  useEffect(() => {
    void setScenario(scenario);
  }, [scenario, setScenario]);

  const snapshot = useLogFeed({ logs: wsd.logs, query, levels, sources });
  const latest = wsd.metrics.at(-1);

  const dsItems = [
    { label: "run", value: wsd.meta.runId },
    { label: "steps", value: latest ? String(latest.step) : "—" },
    { label: "phase", value: latest?.phase ?? wsd.meta.phase },
    { label: "optimizer", value: wsd.meta.optimizer },
    { label: "logs", value: String(wsd.logs.length) },
  ];

  return (
    <div className={styles.page}>
      <DataSourceBar
        dataSource={wsd.dataSource}
        items={dsItems}
        hint="Start the API: python -m hildanext.api serve --config runs/configs/llada21_dolma_wsd_only.json"
      />

      <section className={styles.hero}>
        <MetricHeroCard
          label="Current phase"
          value={latest?.phase ?? "warmup"}
          accent="cyan"
          meta={`Block ${latest?.blockSize ?? 1}`}
        />
        <MetricHeroCard
          label="Masked token acc"
          value={latest?.maskedTokenAcc ? formatPct(latest.maskedTokenAcc) : "n/a"}
          accent="lime"
          meta={latest?.bidirectional ? "Stable bidirectional" : "Causal effective"}
        />
        <MetricHeroCard
          label="Throughput"
          value={latest ? `${formatNumber(latest.tokensPerSec, 0)} tok/s` : "n/a"}
          accent="orange"
          meta={latest ? `${latest.stepTimeS.toFixed(2)} sec/step` : "Waiting"}
        />
        <MetricHeroCard
          label="Peak VRAM"
          value={latest ? `${Math.round(latest.vramPeakMb)} MB` : "n/a"}
          accent="red"
          meta={latest ? `ETA ${formatDuration(latest.etaStageSec)}` : "No run"}
        />
      </section>

      <div className={styles.layout}>
        <div className={styles.primary}>
          <RunControlPanel />

          <Panel kicker="Schedule" title="Warmup -> stable -> decay">
            <PhaseTimeline metrics={wsd.metrics} ladderBlocks={wsd.meta.ladderBlocks} />
          </Panel>

          <Panel kicker="Metrics" title="Loss, throughput and VRAM in one viewport">
            <TimeseriesChart
              x={wsd.metrics.map((row) => row.step)}
              yLabel="Loss / MB"
              series={[
                { label: "loss", stroke: "#3fe0ff", values: wsd.metrics.map((row) => row.loss) },
                { label: "vram", stroke: "#ff8559", values: wsd.metrics.map((row) => row.vramPeakMb) },
                { label: "mta", stroke: "#c9ff44", values: wsd.metrics.map((row) => (row.maskedTokenAcc ?? 0) * 1000) },
              ]}
            />
          </Panel>

          {latest ? (
            <Panel
              kicker="t-bucket analysis"
              title="Continuous-time diagnostics"
              actions={
                <span className={styles.helper}>low-noise buckets should tighten first</span>
              }
            >
              <TBucketHeatStrip row={latest} />
            </Panel>
          ) : null}

          <Panel kicker="CMD transcript" title="Console-first reading flow">
            <TerminalTranscript
              rows={snapshot.rows.filter((row) => row.source === "console" || row.source === "training")}
              pinnedTag="RUN_START / PHASE_CHANGE / OOM"
            />
          </Panel>

          <Panel kicker="Structured logs" title="Virtualized log window">
            <StickyFilterBar
              query={query}
              onQueryChange={(value) => {
                params.set("q", value);
                setParams(params);
              }}
              levels={levels}
              onToggleLevel={(value) => {
                const next = toggle(levels, value);
                params.delete("level");
                next.forEach((item) => params.append("level", item));
                setParams(params);
              }}
              sources={sources}
              onToggleSource={(value) => {
                const next = toggle(sources, value);
                params.delete("source");
                next.forEach((item) => params.append("source", item));
                setParams(params);
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
                params.set("log", id);
                setParams(params);
              }}
            />
          </Panel>
        </div>

        <div className={styles.side}>
          <StatusRail pathname="/wsd" />
          <ProcessRail rows={wsd.processes} />
          <InsightCallout items={wsd.insights} />
          <GlossaryInspector />
        </div>
      </div>
    </div>
  );
}

import { WsdMeta, WsdMetricRow } from "../../domain/types";
import styles from "./PhaseTimeline.module.css";

const PHASES = ["warmup", "stable", "decay"] as const;

export function PhaseTimeline({
  metrics,
  ladderBlocks,
  meta,
}: {
  metrics: WsdMetricRow[];
  ladderBlocks: number[];
  meta?: Pick<WsdMeta, "warmupSteps" | "stableSteps" | "decaySteps" | "totalSteps">;
}) {
  // If real metrics exist, compute widths from logged step counts.
  // Otherwise use the planned step proportions from the config.
  const totalSteps = meta?.totalSteps ?? 0;
  const planned: Record<string, number> = {
    warmup: meta?.warmupSteps ?? 0,
    stable: meta?.stableSteps ?? 0,
    decay:  meta?.decaySteps  ?? 0,
  };

  const groups = PHASES.map((p) => ({
    phase: p,
    rows: metrics.filter((r) => r.phase === p),
  }));

  const useMetrics = metrics.length > 0;
  const metricTotal = Math.max(metrics.length, 1);

  return (
    <div className={styles.timeline}>
      <div className={styles.track}>
        {groups.map(({ phase, rows }) => {
          const pct = useMetrics
            ? (rows.length / metricTotal) * 100
            : totalSteps > 0
            ? (planned[phase] / totalSteps) * 100
            : 100 / 3;
          const label = useMetrics
            ? (rows.at(-1)?.blockSize ?? "-")
            : planned[phase] > 0
            ? `${planned[phase]} steps`
            : "-";
          return (
            <div
              key={phase}
              className={`${styles.segment} ${styles[phase]}`}
              style={{ width: `${pct}%` }}
              title={`${phase}: ${planned[phase] ?? "?"} steps planned`}
            >
              <span>{phase}</span>
              <strong>{label}</strong>
            </div>
          );
        })}
      </div>
      <div className={styles.markers}>
        {ladderBlocks.map((block) => (
          <span key={block}>{block}</span>
        ))}
      </div>
    </div>
  );
}

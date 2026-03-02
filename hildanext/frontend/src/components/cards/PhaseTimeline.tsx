import { WsdMetricRow } from "../../domain/types";
import styles from "./PhaseTimeline.module.css";

export function PhaseTimeline({
  metrics,
  ladderBlocks,
}: {
  metrics: WsdMetricRow[];
  ladderBlocks: number[];
}) {
  const groups = [
    metrics.filter((row) => row.phase === "warmup"),
    metrics.filter((row) => row.phase === "stable"),
    metrics.filter((row) => row.phase === "decay"),
  ];

  return (
    <div className={styles.timeline}>
      <div className={styles.track}>
        {groups.map((group) => {
          const phase = group[0]?.phase ?? "warmup";
          const width = `${(group.length / Math.max(metrics.length, 1)) * 100}%`;
          return (
            <div key={phase} className={`${styles.segment} ${styles[phase]}`} style={{ width }}>
              <span>{phase}</span>
              <strong>{group.at(-1)?.blockSize ?? "-"}</strong>
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

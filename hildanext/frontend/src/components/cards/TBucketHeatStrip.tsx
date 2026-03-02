import { WsdMetricRow } from "../../domain/types";
import { formatPct } from "../../domain/formatters";
import styles from "./TBucketHeatStrip.module.css";

const BUCKETS = ["0.0-0.1", "0.1-0.3", "0.3-0.6", "0.6-1.0"] as const;

export function TBucketHeatStrip({ row }: { row: WsdMetricRow }) {
  return (
    <div className={styles.grid}>
      {BUCKETS.map((bucket) => (
        <article key={bucket} className={styles.cell}>
          <span>{bucket}</span>
          <strong>{row.lossByTBucket[bucket].toFixed(3)}</strong>
          <small>{formatPct(row.accMaskedByTBucket[bucket])}</small>
        </article>
      ))}
    </div>
  );
}

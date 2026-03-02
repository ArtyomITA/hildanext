import { ProcessSnapshot } from "../../domain/types";
import { formatCompact } from "../../domain/formatters";
import { Panel } from "../../components/layout/Panel";
import styles from "./ProcessRail.module.css";

export function ProcessRail({
  rows,
  sourceLabel,
}: {
  rows: ProcessSnapshot[];
  sourceLabel?: string;
}) {
  const latest = rows.at(-1);

  return (
    <Panel
      kicker="Resource rail"
      title="Process + VRAM posture"
      actions={sourceLabel ? <span className={styles.badge}>{sourceLabel}</span> : undefined}
    >
      <div className={styles.summary}>
        <article>
          <span>GPU VRAM</span>
          <strong>{latest ? `${Math.round(latest.gpuVramMb ?? 0)} MB` : "n/a"}</strong>
        </article>
        <article>
          <span>GPU util</span>
          <strong>{latest ? `${Math.round(latest.gpuUtilPct ?? 0)}%` : "n/a"}</strong>
        </article>
        <article>
          <span>System RAM</span>
          <strong>{latest ? `${formatCompact(latest.ramMb)} MB` : "n/a"}</strong>
        </article>
      </div>
      <div className={styles.table}>
        {rows.slice(-8).map((row) => (
          <div key={row.tsUtc} className={styles.row}>
            <span>{row.processName}</span>
            <strong>{Math.round(row.gpuVramMb ?? 0)} MB</strong>
            <small>{row.status}</small>
          </div>
        ))}
      </div>
    </Panel>
  );
}

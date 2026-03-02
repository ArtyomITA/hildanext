import { InferenceRun } from "../../domain/types";
import { formatBytes, formatCompact } from "../../domain/formatters";
import { Panel } from "../../components/layout/Panel";
import styles from "./InferenceSplitPane.module.css";

export function InferenceSplitPane({
  ar,
  diffusion,
  sourceLabel,
}: {
  ar: InferenceRun;
  diffusion: InferenceRun;
  sourceLabel?: string;
}) {
  return (
    <Panel
      kicker="Compare"
      title="AR lane vs diffusion lane"
      actions={sourceLabel ? <span className={styles.badge}>{sourceLabel}</span> : undefined}
    >
      <div className={styles.grid}>
        <article className={`${styles.card} ${styles.ar}`}>
          <header>
            <span>AR lane</span>
            <strong>{ar.mode}</strong>
          </header>
          <p>{ar.outputText}</p>
          <dl>
            <div>
              <dt>Throughput</dt>
              <dd>{formatCompact(ar.tokensPerSec ?? 0)} tok/s</dd>
            </div>
            <div>
              <dt>Peak VRAM</dt>
              <dd>{formatBytes(ar.vramPeakBytes ?? 0)}</dd>
            </div>
          </dl>
        </article>
        <article className={`${styles.card} ${styles.diffusion}`}>
          <header>
            <span>Diffusion lane</span>
            <strong>{diffusion.mode}</strong>
          </header>
          <p>{diffusion.outputText}</p>
          <dl>
            <div>
              <dt>Converge</dt>
              <dd>{diffusion.stepsToConverge ?? "n/a"} steps</dd>
            </div>
            <div>
              <dt>Peak VRAM</dt>
              <dd>{formatBytes(diffusion.vramPeakBytes ?? 0)}</dd>
            </div>
          </dl>
        </article>
      </div>
      <div className={styles.ribbon}>
        <span>Order of certainty</span>
        <div className={styles.scale}>
          <i>left-to-right</i>
          <b>parallel drafting + revision</b>
        </div>
      </div>
    </Panel>
  );
}

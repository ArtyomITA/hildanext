import styles from "./StatusRail.module.css";
import { useDataStore } from "../store/dataStore";
import { formatBytes, formatCompact, formatPct } from "../domain/formatters";

export function StatusRail({ pathname }: { pathname: string }) {
  const wsd = useDataStore((state) => state.wsd);
  const inference = useDataStore((state) => state.inference);

  const isInference = pathname.includes("/inference");

  if (isInference) {
    return (
      <aside className={`${styles.rail} slide-up`}>
        <section className={styles.block}>
          <p className={styles.label}>Active inference</p>
          <h2>{inference.diffusion.engine.toUpperCase()} / {inference.diffusion.mode}</h2>
          <dl className={styles.grid}>
            <div>
              <dt>Throughput</dt>
              <dd>{formatCompact(inference.diffusion.tokensPerSec ?? 0)} tok/s</dd>
            </div>
            <div>
              <dt>Converge</dt>
              <dd>{inference.diffusion.stepsToConverge ?? "n/a"} steps</dd>
            </div>
            <div>
              <dt>Peak VRAM</dt>
              <dd>{formatBytes(inference.diffusion.vramPeakBytes ?? 0)}</dd>
            </div>
            <div>
              <dt>Fallbacks</dt>
              <dd>{inference.logs.filter((row) => row.level !== "info").length}</dd>
            </div>
          </dl>
        </section>
      </aside>
    );
  }

  const latest = wsd.metrics.at(-1);

  return (
    <aside className={`${styles.rail} slide-up`}>
      <section className={styles.block}>
        <p className={styles.label}>Run posture</p>
        <h2>{wsd.meta.runId}</h2>
        <dl className={styles.grid}>
          <div>
            <dt>Phase</dt>
            <dd>{latest?.phase ?? "warmup"}</dd>
          </div>
          <div>
            <dt>VRAM ceiling</dt>
            <dd>{latest ? `${Math.round(latest.vramPeakMb)} MB` : "n/a"}</dd>
          </div>
          <div>
            <dt>Masked acc</dt>
            <dd>{latest?.maskedTokenAcc ? formatPct(latest.maskedTokenAcc) : "n/a"}</dd>
          </div>
          <div>
            <dt>Fallback heat</dt>
            <dd>{wsd.logs.filter((row) => row.level !== "info").length} notices</dd>
          </div>
        </dl>
      </section>
      <section className={styles.block}>
        <p className={styles.label}>What this page optimizes</p>
        <ul className={styles.list}>
          <li>Read long-running CMD transcripts without heap spikes</li>
          <li>Spot WSD phase changes before they hit loss cliffs</li>
          <li>Track VRAM saturation alongside token throughput</li>
        </ul>
      </section>
    </aside>
  );
}

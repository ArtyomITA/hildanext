import { InferenceTraceStep } from "../../domain/types";
import styles from "./DiffusionStepTimeline.module.css";

export function DiffusionStepTimeline({
  rows,
  selectedStep,
  onSelect,
}: {
  rows: InferenceTraceStep[];
  selectedStep: number;
  onSelect: (step: number) => void;
}) {
  return (
    <div className={styles.timeline}>
      {rows.map((row) => (
        <button
          key={row.step}
          className={`${styles.card} ${selectedStep === row.step ? styles.active : ""}`}
          type="button"
          onClick={() => onSelect(row.step)}
        >
          <header>
            <strong>Step {row.step}</strong>
            {row.tauFallbackApplied ? <span>tau relax</span> : null}
          </header>
          <dl>
            <div>
              <dt>Mask</dt>
              <dd>{(row.maskRatio * 100).toFixed(0)}%</dd>
            </div>
            <div>
              <dt>Gamma</dt>
              <dd>{row.gammaCount}</dd>
            </div>
            <div>
              <dt>Delta</dt>
              <dd>{row.deltaCount}</dd>
            </div>
            <div>
              <dt>Conf</dt>
              <dd>{row.avgConfTokens?.toFixed(2) ?? "n/a"}</dd>
            </div>
          </dl>
        </button>
      ))}
    </div>
  );
}

/**
 * DataSourceBar — single source of truth for "what data am I seeing?"
 *
 * Shows a sticky pill + metadata row at the top of every page.
 * One component for both WSD and Inference pages.
 */
import styles from "./DataSourceBar.module.css";

export type DataSource = "live" | "mockup" | "missing";

export interface DataSourceBarItem {
  label: string;
  value: string;
}

interface Props {
  dataSource: DataSource;
  items: DataSourceBarItem[];
  /** Shown as a dim hint when source=missing. Tells the user how to fix it. */
  hint?: string;
}

const PILL_LABEL: Record<DataSource, string> = {
  live:    "● LIVE",
  mockup:  "◎ OFFLINE MOCKUP",
  missing: "✕ DATA MISSING",
};

export function DataSourceBar({ dataSource, items, hint }: Props) {
  return (
    <div className={styles.bar} data-source={dataSource}>
      <span className={styles.pill} data-testid="ds-status-pill">{PILL_LABEL[dataSource]}</span>

      {items.map((item) => (
        <span key={item.label} className={styles.item}>
          <span className={styles.itemLabel}>{item.label}</span>
          <span className={styles.itemValue}>{item.value}</span>
        </span>
      ))}

      {dataSource === "missing" && hint && (
        <span className={styles.hint}>{hint}</span>
      )}
    </div>
  );
}

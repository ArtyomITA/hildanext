import { InsightCard } from "../../domain/types";
import { Panel } from "../../components/layout/Panel";
import { SeverityBadge } from "../../components/badges/SeverityBadge";
import styles from "./InsightCallout.module.css";

export function InsightCallout({
  items,
  sourceLabel,
}: {
  items: InsightCard[];
  sourceLabel?: string;
}) {
  return (
    <Panel
      kicker="Insights"
      title="What to read first"
      actions={sourceLabel ? <span className={styles.badge}>{sourceLabel}</span> : undefined}
    >
      <div className={styles.list}>
        {items.map((item) => (
          <article key={item.id} className={styles.item}>
            <div className={styles.head}>
              <strong>{item.title}</strong>
              <SeverityBadge
                level={item.tone === "critical" ? "error" : item.tone === "warning" ? "warning" : "notice"}
              >
                {item.metric}
              </SeverityBadge>
            </div>
            <p>{item.body}</p>
          </article>
        ))}
      </div>
    </Panel>
  );
}

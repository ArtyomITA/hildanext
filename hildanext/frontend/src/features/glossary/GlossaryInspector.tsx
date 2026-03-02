import { useState } from "react";
import { glossaryTerms } from "../../domain/glossary";
import { Panel } from "../../components/layout/Panel";
import styles from "./GlossaryInspector.module.css";

export function GlossaryInspector({ sourceLabel }: { sourceLabel?: string }) {
  const [active, setActive] = useState(glossaryTerms[0]?.key ?? "");
  const term = glossaryTerms.find((item) => item.key === active) ?? glossaryTerms[0];

  return (
    <Panel
      kicker="Glossary"
      title="Correct terms, not vague dashboard copy"
      actions={sourceLabel ? <span className={styles.badge}>{sourceLabel}</span> : undefined}
    >
      <div className={styles.tabs}>
        {glossaryTerms.map((item) => (
          <button
            key={item.key}
            className={active === item.key ? styles.active : styles.tab}
            type="button"
            onClick={() => setActive(item.key)}
          >
            {item.shortLabel}
          </button>
        ))}
      </div>
      <div className={styles.body}>
        <p className={styles.en}>{term.english}</p>
        <p className={styles.it}>{term.italianHint}</p>
        <p>{term.explanation}</p>
      </div>
    </Panel>
  );
}

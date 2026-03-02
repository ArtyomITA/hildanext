import { PropsWithChildren, ReactNode } from "react";
import clsx from "clsx";
import styles from "./Panel.module.css";

interface PanelProps extends PropsWithChildren {
  title?: string;
  kicker?: string;
  actions?: ReactNode;
  className?: string;
}

export function Panel({ title, kicker, actions, className, children }: PanelProps) {
  return (
    <section className={clsx(styles.panel, className)}>
      {(title || kicker || actions) && (
        <header className={styles.header}>
          <div>
            {kicker ? <p className={styles.kicker}>{kicker}</p> : null}
            {title ? <h2 className={styles.title}>{title}</h2> : null}
          </div>
          {actions ? <div className={styles.actions}>{actions}</div> : null}
        </header>
      )}
      {children}
    </section>
  );
}
